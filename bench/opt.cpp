// "Picker": CMA-ES search of balancer parameters on a push / commanded-step test set.
//   emil_bench opt <name> [threads=6] [lambda=16] [generations=1000] [--sigma 0.2] [--list file] [--weights file]
//                  [--budget s] [--control N] [--scale 0.5]
// Candidate score: stagger scenes of the target table (stood + phase distance; the main part), constraints = hold,
// commanded steps and stand not worse than the start set. Control set: stagger 20 x 6 moments + sweep 96 + holds +
// steps + stand; best.params = most stagger-stood, ties by phase distance.
// Curriculum: the stagger impulses are scaled by `scale` (start 0.5, --scale s0); after a control round in which the
// best candidate stood >= 60 % of the control stagger scenes the scale grows by 0.1 (max 1.0). Ranking key = (scale,
// stood, phase).
// Every --control N (default 5) generations the 3 best candidates of that generation and the distribution mean are
// scored on the fixed control set (full sweep 96 + 8 hold scenes + 16 commanded steps + stand, the same scenes always);
// the 3 best sets over the whole run are kept (top.txt, top1..3.params, best.params = top 1, control.csv).
//   emil_bench opt test            determinism check of the evaluation (same candidate twice on one reused world)
//   emil_bench opt eval [-p ...]   score of the current parameters (one candidate, terms printed)
// Output in $EMIL_TMP/opt/<name>/: log.csv (per generation: best, mean, sigma, terms of the best, full sweep),
// best.params (name=value, feed with `emil_bench --params file ...`), state.txt (resume), params.txt, weights.txt
// (copies of the inputs).
// Search space file: lines `name [min max [start]]` (defaults: kParamDefs min / max / value); weights file: lines
// `term=weight`.
// Candidate = point in [0,1]^n (normalised by min / max), out-of-range points are reflected and charged a small
// penalty.
#include <mujoco/mujoco.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tmp_dir.h"
#include "cma.h"
#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p or --params
bool OverrideValue(int idx, float* v);

namespace {

// ---------------------------------------------------------------- inputs
struct ParamSpec { int idx; double lo, hi, start; };

bool LoadList(const char* path, std::vector<ParamSpec>* out) {
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
  char line[512];
  while (fgets(line, sizeof line, f)) {
    char nm[128];
    double a, b, c;
    const char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '#' || *p == '\n' || *p == 0) continue;
    const int k = sscanf(p, "%127s %lf %lf %lf", nm, &a, &b, &c);
    if (k < 1) continue;
    int idx = -1;
    for (int j = 0; j < P_COUNT; ++j) if (!std::strcmp(kParamDefs[j].name, nm)) idx = j;
    if (idx < 0) { fprintf(stderr, "unknown parameter in list: %s\n", nm); fclose(f); return false; }
    ParamSpec s{idx, kParamDefs[idx].min, kParamDefs[idx].max, kParamDefs[idx].value};
    if (k >= 3) { s.lo = a; s.hi = b; }
    if (k >= 4) s.start = c;
    out->push_back(s);
  }
  fclose(f);
  return !out->empty();
}

const char* kDefaultList =
    "# name [min max [start]]; defaults from kParamDefs. Step and balance parameters, no switches, no reactions.\n"
    "stiff_scale\nknee_bend\nankle_dorsi\ncom_fwd\ncom_kp\ncom_kd\ntrunk_kp\ntoe_out\nstance_w\ncp_margin\nclear\nreach\nmin_width\n"
    "torso_kp\ntorso_kd\nswing_t\nstep_extra\nshift_skip\ntoe_lift\nvf_kp\nvf_kd\nvf_max\nthigh_max\nzmp_frac\nzmp_diamond\n"
    "toe_land\nbrake\nbrake_lat\nflex_max\nlean\n";

struct Weights {
  std::map<std::string, double> w = {
      {"stag_fall", 10.0},   // 1 - fraction stood on the stagger scenes (same dv for every body): the main term
      {"stag_phase", 4.0},   // mean distance of the stagger scenes from the stagger target table (0..3 band widths; fell = 2..3)
      {"hold", 3.0},         // constraint: penalty x (hold-without-step fraction worse than the start set), ladder step 1
      {"steps", 3.0},        // constraint: penalty x (commanded steps not stood, worse than the start set), ladder step 4
      {"stand", 2.0},        // constraint: penalty x (stand outside the rest band, worse than the start set)
      {"bound", 5.0},        // distance of the raw candidate outside [0,1]
  };
};

void LoadWeights(const char* path, Weights* W) {
  FILE* f = fopen(path, "r");
  if (!f) return;
  char line[256];
  while (fgets(line, sizeof line, f)) {
    char* eq = std::strchr(line, '=');
    if (line[0] == '#' || !eq) continue;
    *eq = 0;
    std::string k = line;
    while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
    W->w[k] = std::atof(eq + 1);
  }
  fclose(f);
}

// ---------------------------------------------------------------- scenes
struct PushScene { double ang_deg, J, t_push; BodyParams body; bool hold; };
struct Gen { std::vector<PushScene> push; std::vector<stag::Item> stag; uint64_t seed = 0; };  // push = 8 hold scenes

const double kAng[8] = {0, 45, 90, 135, 180, 225, 270, 315};  // heading +x, +90 = left
const double kHoldJ[8] = {15, 30, 25, 30, 20, 30, 25, 30};    // step 1 of the ladder: fwd 15, back 20, side 25, diagonal 30 N s

// Random test set of one generation (same for every candidate): 8 hold forces at nominal bodies, 4 random sweep forces per
// direction (5..60 N s), every second of them with a varied body (height 1.65-1.95, mass 60-100, build -0.5..0.5).
Gen MakeGen(uint64_t seed, double scale) {
  std::mt19937_64 rng(seed * 7919 + 17);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Gen g;
  g.seed = seed;
  for (int d = 0; d < 8; ++d) g.push.push_back({kAng[d], kHoldJ[d], 1.5 + 1.5 * u(rng), BodyParams(), true});
  for (int sc = 0; sc < 20; ++sc) {  // stagger scenes: random push time 1.5..4 s, every second body varied
    BodyParams b;
    if (sc % 2) {
      b.height = (float)(1.65 + 0.30 * u(rng));
      b.mass = (float)(60.0 + 40.0 * u(rng));
      b.build = (float)(-0.5 + u(rng));
      b.female = u(rng) < 0.3;
    }
    g.stag.push_back({sc, 1.5 + 2.5 * u(rng), b, scale});
  }
  return g;
}

// ---------------------------------------------------------------- stagger target table (80 kg)
struct RefStep { double lift, swing, len, tz, knee; };  // knee < 0: not scored
struct Ref { bool valid; RefStep s1, s2; double trunk_lo, trunk_hi, drop; };
// type 0 back, 1 forward, 2 side; size 0 small (35..50 N s), 1 medium (50..80)
const Ref kRef[3][2] = {
    {{true, {0.39, 0.20, 0.27, 0.01, -1}, {0.81, 0.23, 0.36, 0.02, 56}, -3, 5, 0.04},
     {true, {0.13, 0.13, 0.30, 0.15, -1}, {0.43, 0.30, 0.75, 0.00, 36}, -10, 5, 0.07}},
    {{true, {0.15, 0.17, 0.35, -0.03, -1}, {0.42, 0.15, 0.21, 0.07, 39}, 0, 9, 0.03},
     {true, {0.12, 0.18, 0.49, 0.07, -1}, {0.33, 0.23, 0.45, 0.09, 37}, 0, 12, 0.05}},
    {{false, {}, {}, 0, 0, 0},
     {true, {0.08, 0.27, 0.36, 0.23, -1}, {0.38, 0.24, 0.53, 0.01, 30}, -16, 11, 0.03}}};

// ---------------------------------------------------------------- per-bot tracker
struct Swing { int foot; double t_lift, t_td, len, tz, knee; };

struct Track {
  PushScene sc;
  double J_eff = 0;
  bool pushed = false, fell = false;
  double fall_t = -1, t_end = 0;
  double ux = 0, uy = 0;  // push direction
  int steps0 = 0, steps = 0;
  std::vector<double> qx, qy;  // quiet stand: CoM positions per frame (t >= 1 s, before the push)
  double width_sum = 0, fwd_sum = 0;
  int nq = 0;
  double ank0[2][2] = {{0, 0}, {0, 0}}, ank1[2][2] = {{0, 0}, {0, 0}};
  double pz0 = 0, yaw0 = 0, yaw_max = 0, drop = 0, lean_lo = 0, lean_hi = 0, flex_max = 0;
  long loaded_frames = 0, edge_frames = 0, torque_frames = 0, frames = 0;
  double lift_t[2] = {-1, -1}, lift_p[2][2] = {{0, 0}, {0, 0}};
  std::vector<Swing> sw;
  bool stood = false;
};

struct Terms {
  double fall = 0, hold = 0, phase = 0, stand = 0, steps = 0, yaw = 0, flex = 0, slide = 0, edge = 0, torque = 0, bound = 0;
  double stood_n = 0, stood_of = 0, sweep_stood = -1, cost = 0;
  double stag_fall = 1, stag_phase = 3, stag_stood = 0, stag_n = 0, key = 0, scale = 1;  // stagger part (key: control ranking)
  double width = 0, sway = 0, pfwd = 0;  // diagnostics: mean stand width [m], sway rms [mm], pelvis forward [m]
};

double Clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
double Band(double x, double r, double rel, double minband) {  // violation beyond +-max(rel r, minband), in band widths
  const double band = std::fmax(rel * std::fabs(r), minband);
  return Clamp(std::fabs(x - r) / band - 1.0, 0.0, 3.0);
}

constexpr double kSpacing = 3.0;  // bots along x [m]
constexpr double kTail = 6.0;     // seconds simulated after the push

void Observe(World& w, int i, Track& tk, double t) {
  const mjData* d = w.data();
  Bot& b = w.bot(i);
  const double* pp = d->xpos + 3 * b.body0;
  const double* R = d->xmat + 9 * b.body0;
  const double hn = std::hypot(R[0], R[3]) + 1e-9, hx = R[0] / hn, hy = R[3] / hn;
  double ank[2][3];
  for (int f = 0; f < 2; ++f)
    for (int k = 0; k < 3; ++k) ank[f][k] = d->xanchor[3 * (b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)) + k];
  const double hs = b.body.height / 1.78;
  if (!tk.pushed) {
    if (t >= 1.0 && t >= tk.sc.t_push - 1.0) {
      const double* com = d->subtree_com + 3 * b.body0;
      tk.qx.push_back(com[0]);
      tk.qy.push_back(com[1]);
      tk.width_sum += std::fabs(-(ank[0][0] - ank[1][0]) * hy + (ank[0][1] - ank[1][1]) * hx) / hs;
      tk.fwd_sum += ((pp[0] - 0.5 * (ank[0][0] + ank[1][0])) * hx + (pp[1] - 0.5 * (ank[0][1] + ank[1][1])) * hy) / hs;
      if (tk.nq == 0) for (int f = 0; f < 2; ++f) { tk.ank0[f][0] = ank[f][0]; tk.ank0[f][1] = ank[f][1]; }
      for (int f = 0; f < 2; ++f) { tk.ank1[f][0] = ank[f][0]; tk.ank1[f][1] = ank[f][1]; }
      ++tk.nq;
    }
    return;
  }
  if (tk.fell) return;
  ++tk.frames;
  if (pp[2] < 0.5) { tk.fell = true; tk.fall_t = t; return; }
  const double W = b.dims.total_mass * 9.81;
  double dy = std::atan2(R[3], R[0]) - tk.yaw0;
  while (dy > 3.14159265) dy -= 6.2831853;
  while (dy < -3.14159265) dy += 6.2831853;
  tk.yaw_max = std::fmax(tk.yaw_max, std::fabs(dy) * 57.29578);
  tk.drop = std::fmax(tk.drop, tk.pz0 - pp[2]);
  const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
  const double lean = std::asin(Clamp(Rc[2] * tk.ux + Rc[5] * tk.uy, -1.0, 1.0)) * 57.29578;  // + along the push
  tk.lean_lo = std::fmin(tk.lean_lo, lean);
  tk.lean_hi = std::fmax(tk.lean_hi, lean);
  tk.flex_max = std::fmax(tk.flex_max, std::fmax(d->qpos[b.qadr0 + J_HIP_FLEX_L], d->qpos[b.qadr0 + J_HIP_FLEX_R]) * 57.29578);
  for (int f = 0; f < 2; ++f)
    if (b.foot_fn[f] > 0.3 * W) {
      ++tk.loaded_frames;
      const double yz = d->geom_xmat[9 * b.geom_foot[f] + 7];
      if (std::fabs(std::asin(Clamp(yz, -1.0, 1.0))) * 57.29578 > 8.0) ++tk.edge_frames;
    }
  bool lim = false;
  for (int j = J_HIP_FLEX_L; j < J_COUNT && !lim; ++j)
    if (std::fabs(d->ctrl[b.act0 + j]) >= 0.98 * b.js[j].tau_max * b.in.leg_weak[j >= J_HIP_FLEX_R ? 1 : 0]) lim = true;
  if (lim) ++tk.torque_frames;
  for (int f = 0; f < 2; ++f) {  // swings: foot 2 cm above its rest ankle height, back within 1.2 cm with the contact flag
    const double up = ank[f][2] - b.dims.ankle_h;
    if (tk.lift_t[f] < 0 && up > 0.02) {
      tk.lift_t[f] = t;
      tk.lift_p[f][0] = ank[f][0];
      tk.lift_p[f][1] = ank[f][1];
    } else if (tk.lift_t[f] >= 0 && up < 0.012 && b.foot_on[f]) {
      Swing s;
      s.foot = f;
      s.t_lift = tk.lift_t[f];
      s.t_td = t;
      s.len = std::hypot(ank[f][0] - tk.lift_p[f][0], ank[f][1] - tk.lift_p[f][1]);
      const Percept& P = b.pc[b.cur];
      const double om = std::sqrt(9.81 / std::fmax(P.com[2], 0.3));
      s.tz = (P.com[0] + P.com_vel[0] / om - ank[f][0]) * tk.ux + (P.com[1] + P.com_vel[1] / om - ank[f][1]) * tk.uy;
      s.knee = d->qpos[b.qadr0 + (f ? J_KNEE_R : J_KNEE_L)] * 57.29578;
      if (tk.sw.size() < 4) tk.sw.push_back(s);
      tk.lift_t[f] = -1;
    }
  }
}

struct CmdScene { double dx, dy; int foot, cross; };

// One multi-bot world run. Scenes: pushes (cmd == nullptr) or commanded steps at t_cmd.
void RunWorld(World& w, const std::vector<PushScene>& sc, const std::vector<CmdScene>* cmd, double t_cmd, std::vector<Track>* tr) {
  const int n = w.NumBots();
  tr->assign(n, Track());
  w.Reset();
  double t_end = 0;
  for (int i = 0; i < n; ++i) {
    Track& tk = (*tr)[i];
    tk.sc = cmd ? PushScene{0, 0, t_cmd, w.bot(i).body, false} : sc[i];
    const double a = tk.sc.ang_deg * 0.017453293;
    tk.ux = std::cos(a);
    tk.uy = std::sin(a);
    tk.J_eff = tk.sc.J * w.bot(i).body.mass / 80.0;  // same velocity change for every body
    tk.t_end = tk.sc.t_push + kTail;
    t_end = std::fmax(t_end, tk.t_end);
  }
  const mjModel* m = w.model();
  mjData* d = w.data();
  const int body00 = w.bot(0).body0;
  const int frames = (int)std::lround(t_end * 60.0) + 1;
  for (int f = 0; f < frames; ++f) {
    w.StepFrame();
    const double t = d->time;
    for (int i = 0; i < n; ++i) {
      Track& tk = (*tr)[i];
      if (t > tk.t_end) continue;
      Bot& b = w.bot(i);
      if (!tk.pushed && t >= tk.sc.t_push) {
        tk.pushed = true;
        tk.pz0 = d->xpos[3 * b.body0 + 2];
        tk.yaw0 = std::atan2(d->xmat[9 * b.body0 + 3], d->xmat[9 * b.body0]);
        tk.steps0 = b.steps;
        if (cmd) {
          const CmdScene& c = (*cmd)[i];
          double dyc = c.dy;
          if (c.cross) {  // 0.1 m past the other ankle across the pelvis heading (as the steps bench)
            const double* R = d->xmat + 9 * b.body0;
            const double hn = std::hypot(R[0], R[3]) + 1e-9, lx = -R[3] / hn, ly = R[0] / hn;
            const double* a0 = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L);
            const double* a1 = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_R);
            dyc = -(std::fabs((a0[0] - a1[0]) * lx + (a0[1] - a1[1]) * ly) + c.dy);
          }
          w.CommandStep(i, c.foot, c.dx, dyc, c.cross != 0);
        } else {
          const double imp[3] = {tk.ux * tk.J_eff, tk.uy * tk.J_eff, 0};
          w.PushAt(i, imp, d->xipos + 3 * (b.body0 + B_CHEST), B_CHEST);  // standard chest push point
        }
        continue;
      }
      Observe(w, i, tk, t);
    }
    for (int c = 0; c < d->ncon; ++c) {  // pelvis .. head on the floor = fell
      const mjContact& cn = d->contact[c];
      const int g = cn.geom1 == 0 ? cn.geom2 : (cn.geom2 == 0 ? cn.geom1 : -1);
      if (g < 0) continue;
      const int rel0 = m->geom_bodyid[g] - body00;
      if (rel0 < 0) continue;
      const int i = rel0 / B_COUNT, rel = rel0 % B_COUNT;
      if (i < n && rel >= B_PELVIS && rel <= B_HEAD) {
        Track& tk = (*tr)[i];
        if (tk.pushed && !tk.fell && t <= tk.t_end) { tk.fell = true; tk.fall_t = t; }
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    Track& tk = (*tr)[i];
    Bot& b = w.bot(i);
    tk.steps = b.steps - tk.steps0;
    tk.stood = tk.pushed && !tk.fell && b.mode != 2 && d->xpos[3 * b.body0 + 2] > 0.6;
  }
}

// ---------------------------------------------------------------- scoring
const int kRefSteps[3][2] = {{2, 4}, {2, 3}, {0, 4}};

// Mean band violation of the phase table over the metrics of one pushed bot; -1 = no table row for this push.
double PhasePenalty(const Track& tk) {
  const double J = tk.sc.J;
  if (tk.sc.hold || J < 35) return -1;
  const double c = std::fabs(tk.ux), s = std::fabs(tk.uy);
  const int type = c >= s ? (tk.ux < 0 ? 0 : 1) : 2;
  if (J > 45 && J < 55) return -1;  // between small (39 N s) and medium (58-78 N s) of the table: no table row
  const int size = J <= 45 ? 0 : 1;
  const Ref& r = kRef[type][size];
  if (!r.valid) return -1;
  if (!tk.stood) {  // fell or gave up: 2..3 by how close its first step was to the table (a stood bot is usually below 2)
    double m1 = 3.0;
    if (!tk.sw.empty()) {
      const Swing* f = &tk.sw[0];
      for (const Swing& q : tk.sw)
        if (q.t_lift < f->t_lift) f = &q;
      m1 = 0.25 * (Band(f->t_lift - tk.sc.t_push, r.s1.lift, 0.3, 0.05) + Band(f->t_td - f->t_lift, r.s1.swing, 0.3, 0.05) +
                   Band(f->len, r.s1.len, 0.3, 0.05) + Band(f->tz, r.s1.tz, 0.0, 0.1));
    }
    return 2.0 + m1 * (1.0 / 3.0);
  }
  std::vector<Swing> sw = tk.sw;
  std::sort(sw.begin(), sw.end(), [](const Swing& a, const Swing& b) { return a.t_lift < b.t_lift; });
  double sum = 0;
  int n = 0;
  auto add = [&](double v) { sum += v; ++n; };
  for (int k = 0; k < 2; ++k) {
    const RefStep& rs = k == 0 ? r.s1 : r.s2;
    if ((int)sw.size() <= k) {
      for (int q = 0; q < (k == 0 ? 4 : (rs.knee > 0 ? 5 : 4)); ++q) add(2.0);
      continue;
    }
    const Swing& s2 = sw[k];
    add(Band(s2.t_lift - tk.sc.t_push, rs.lift, 0.3, 0.05));
    add(Band(s2.t_td - s2.t_lift, rs.swing, 0.3, 0.05));
    add(Band(s2.len, rs.len, 0.3, 0.05));
    add(Band(s2.tz, rs.tz, 0.0, 0.1));
    if (k == 1 && rs.knee > 0) add(Band(s2.knee, rs.knee, 0.3, 5.0));
  }
  const double lo = r.trunk_lo - (0.3 * std::fabs(r.trunk_lo) + 3), hi = r.trunk_hi + (0.3 * std::fabs(r.trunk_hi) + 3);
  add(Clamp(std::fmax(0.0, std::fmax(lo - tk.lean_lo, tk.lean_hi - hi)) / 3.0, 0.0, 3.0));
  add(Band(tk.drop, r.drop, 0.3, 0.03));
  add(Band(tk.steps, kRefSteps[type][size], 0.3, 1.0));
  return sum / n;
}

Terms Score(const std::vector<Track>& P, const std::vector<Track>& S, const Weights& W) {
  Terms t;
  int np = 0, ns = 0, nh = 0, hold_ok = 0, nph = 0, nst = 0;
  double ph = 0, st = 0, yaw = 0, flex = 0, slide = 0, edge = 0, tq = 0, wd = 0, sw = 0, pf = 0;
  for (const Track& tk : P) {
    if (tk.sc.hold) { ++nh; hold_ok += tk.stood && tk.steps == 0; continue; }
    ++np;
    ns += tk.stood;
    const double p = PhasePenalty(tk);
    if (p >= 0) { ph += p; ++nph; }
    yaw += Clamp((tk.yaw_max - 30.0) / 30.0, 0.0, 3.0);
    flex += Clamp((tk.flex_max - 60.0) / 30.0, 0.0, 3.0);
    if (tk.loaded_frames > 0) edge += (double)tk.edge_frames / tk.loaded_frames;
    if (tk.frames > 0) tq += (double)tk.torque_frames / tk.frames;
  }
  for (const Track& tk : P) {  // quiet stand of every pushed bot, before its push
    if (tk.nq < 10) continue;
    ++nst;
    const double width = tk.width_sum / tk.nq, fwd = tk.fwd_sum / tk.nq;
    // Sway = rms of the CoM about its straight-line trend (the settling drift after the spawn is not sway).
    auto resid2 = [](const std::vector<double>& q) {
      const int m = (int)q.size();
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (int k = 0; k < m; ++k) { sx += k; sy += q[k]; sxx += (double)k * k; sxy += k * q[k]; }
      const double den = m * sxx - sx * sx, a = den > 0 ? (m * sxy - sx * sy) / den : 0.0, b0 = (sy - a * sx) / m;
      double v = 0;
      for (int k = 0; k < m; ++k) v += (q[k] - a * k - b0) * (q[k] - a * k - b0);
      return v;
    };
    const double v = resid2(tk.qx) + resid2(tk.qy);
    const double rms = std::sqrt(v / tk.qx.size()) * 1e3;
    st += (Clamp(std::fmax(0.0, std::fmax(0.21 - width, width - 0.26)) / 0.03, 0.0, 3.0) +
           Clamp(std::fmax(0.0, std::fmax(2.0 - rms, rms - 6.0)) / 2.0, 0.0, 3.0) +
           Clamp(std::fmax(0.0, std::fmax(0.05 - fwd, fwd - 0.08)) / 0.02, 0.0, 3.0)) / 3.0;
    wd += width; sw += rms; pf += fwd;
    const double dr = 0.5 * (std::hypot(tk.ank1[0][0] - tk.ank0[0][0], tk.ank1[0][1] - tk.ank0[0][1]) +
                             std::hypot(tk.ank1[1][0] - tk.ank0[1][0], tk.ank1[1][1] - tk.ank0[1][1])) * 1e3;
    slide += Clamp((dr - 3.0) / 3.0, 0.0, 3.0);
  }
  int nsteps = 0, nsteps_ok = 0;
  for (const Track& tk : S) { ++nsteps; nsteps_ok += tk.stood; }
  t.stood_n = ns; t.stood_of = np;
  t.fall = np ? 1.0 - (double)ns / np : 1.0;
  t.hold = nh ? 1.0 - (double)hold_ok / nh : 1.0;
  t.phase = nph ? ph / nph : 0.0;
  t.stand = nst ? st / nst : 3.0;
  t.steps = nsteps ? 1.0 - (double)nsteps_ok / nsteps : 1.0;
  t.yaw = np ? yaw / np : 0; t.flex = np ? flex / np : 0; t.edge = np ? edge / np : 0; t.torque = np ? tq / np : 0;
  t.slide = nst ? slide / nst : 0;
  t.width = nst ? wd / nst : 0; t.sway = nst ? sw / nst : 0; t.pfwd = nst ? pf / nst : 0;
  auto g = [&](const char* k) { auto it = W.w.find(k); return it == W.w.end() ? 0.0 : it->second; };
  t.cost = g("fall") * t.fall + g("hold") * t.hold + g("phase") * t.phase + g("stand") * t.stand + g("steps") * t.steps +
           g("yaw") * t.yaw + g("flex") * t.flex + g("slide") * t.slide + g("edge") * t.edge + g("torque") * t.torque;
  return t;
}

// ---------------------------------------------------------------- workers
std::mutex g_build_mu;

struct Worker {
  std::unique_ptr<World> P, S, C;
  int gen_built = -1;
  bool s_built = false, c_built = false;
  std::vector<CmdScene> cmds;
  std::vector<PushScene> sweep;  // the full sweep, chunk of 16 per worker index set by the caller
};

bool BuildWorld(World& w, const std::vector<BodyParams>& bodies, uint64_t seed) {
  std::lock_guard<std::mutex> lk(stag::BuildMutex());
  char err[512];
  if (!w.Build(bodies, kSpacing, seed, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return false; }
  return true;
}

void SetParams(World& w, const std::vector<ParamSpec>& list, const std::vector<double>& xnorm) {
  for (int i = 0; i < P_COUNT; ++i) w.param[i] = kParamDefs[i].value;
  ApplyOverrides(w);
  for (size_t k = 0; k < list.size(); ++k) w.param[list[k].idx] = (float)(list[k].lo + Clamp(xnorm[k], 0.0, 1.0) * (list[k].hi - list[k].lo));
}

const struct { double dx, dy; int cross; } kStepKinds[8] = {{0.15, 0, 0}, {0.25, 0, 0}, {0.35, 0, 0}, {-0.15, 0, 0},
                                                          {-0.25, 0, 0}, {0, 0.15, 0}, {0, 0.25, 0}, {0.15, 0.10, 1}};

std::vector<CmdScene> MakeCmds();

void AddStag(Terms* t, const std::vector<stag::Res>& rs, double scale) {
  int stood = 0;
  double ph = 0;
  for (const stag::Res& r : rs) {
    stood += r.stood;
    ph += stag::PhasePenalty(r);
  }
  t->stag_n = (double)rs.size();
  t->stag_stood = stood;
  t->stag_fall = rs.empty() ? 1.0 : 1.0 - (double)stood / rs.size();
  t->stag_phase = rs.empty() ? 3.0 : ph / rs.size();
  t->scale = scale;
  // larger scale first, then stood count, phase breaks ties (phase <= 3): a set that survives a harder push ranks higher
  t->key = (2.0 - scale) * 1000.0 + (t->stag_n - t->stag_stood) + 0.1 * t->stag_phase;
}

// Cost: stagger terms (main) + penalties for hold / commanded steps / stand worse than the baseline (start) set
// + bound.
double Combine(const Terms& t, const Terms& ref, const Weights& W) {
  auto g = [&](const char* k) { auto it = W.w.find(k); return it == W.w.end() ? 0.0 : it->second; };
  return g("stag_fall") * t.stag_fall + g("stag_phase") * t.stag_phase + g("hold") * std::fmax(0.0, t.hold - ref.hold) +
         g("steps") * std::fmax(0.0, t.steps - ref.steps) + g("stand") * std::fmax(0.0, t.stand - ref.stand) + g("bound") * t.bound;
}

// Raw terms of one candidate on the generation set: 8 hold scenes, 20 stagger scenes, 16 commanded steps (cost set by Combine).
Terms Evaluate(Worker& wk, const Gen& g, const std::vector<ParamSpec>& list, const std::vector<double>& xnorm, const Weights& W) {
  {
    std::vector<BodyParams> bodies;
    for (const PushScene& s : g.push) bodies.push_back(s.body);
    wk.P.reset(new World());
    BuildWorld(*wk.P, bodies, g.seed + 1);
  }
  {
    wk.cmds = MakeCmds();
    wk.S.reset(new World());
    BuildWorld(*wk.S, std::vector<BodyParams>(16), 5);
  }
  std::vector<Track> tp, ts;
  SetParams(*wk.P, list, xnorm);
  RunWorld(*wk.P, g.push, nullptr, 0, &tp);
  SetParams(*wk.S, list, xnorm);
  RunWorld(*wk.S, {}, &wk.cmds, 2.0, &ts);
  Terms t = Score(tp, ts, W);
  std::vector<stag::Res> rs;
  stag::RunWorld(g.stag, [&](World& w) { SetParams(w, list, xnorm); }, &rs);
  AddStag(&t, rs, g.stag.empty() ? 1.0 : g.stag[0].scale);
  t.cost = Combine(t, t, W);
  return t;
}

// Parallel for: workers pull indices; each worker thread owns Worker `wi`.
template <class F>
void ParallelFor(int n, int threads, std::vector<Worker*>& wk, F fn) {
  std::atomic<int> next(0);
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t)
    th.emplace_back([&, t]() {
      for (;;) {
        const int i = next.fetch_add(1);
        if (i >= n) break;
        fn(*wk[t], i);
      }
    });
  for (auto& x : th) x.join();
}

std::vector<CmdScene> MakeCmds() {
  std::vector<CmdScene> c;
  for (int i = 0; i < 16; ++i) c.push_back({kStepKinds[i % 8].dx, kStepKinds[i % 8].dy, i / 8, kStepKinds[i % 8].cross});
  return c;
}

// Control set, fixed scenes (never changes): chunks 0-5 = full sweep (8 directions x 12 forces 5..60 N s, nominal body, push at
// 2.0 s, 16 bots each), chunk 6 = 8 hold scenes (ladder step 1), chunk 7 = 16 commanded steps. Returns the tracks of the chunk.
constexpr int kCtlChunks = 8;
void ControlChunk(Worker& wk, int chunk, const std::vector<ParamSpec>& list, const std::vector<double>& xnorm, std::vector<Track>* out) {
  std::vector<PushScene> sc;
  if (chunk < 6) {
    for (int i = 0; i < 16; ++i) {
      const int k = chunk * 16 + i;
      sc.push_back({kAng[k / 12], 5.0 * (k % 12 + 1), 2.0, BodyParams(), false});
    }
    wk.C.reset(new World());
    BuildWorld(*wk.C, std::vector<BodyParams>(16), 9);
    SetParams(*wk.C, list, xnorm);
    RunWorld(*wk.C, sc, nullptr, 0, out);
  } else if (chunk == 6) {
    for (int d = 0; d < 8; ++d) sc.push_back({kAng[d], kHoldJ[d], 2.0, BodyParams(), true});
    wk.C.reset(new World());
    BuildWorld(*wk.C, std::vector<BodyParams>(8), 11);
    SetParams(*wk.C, list, xnorm);
    RunWorld(*wk.C, sc, nullptr, 0, out);
  } else {
    wk.cmds = MakeCmds();
    wk.S.reset(new World());
    BuildWorld(*wk.S, std::vector<BodyParams>(16), 5);
    SetParams(*wk.S, list, xnorm);
    RunWorld(*wk.S, {}, &wk.cmds, 2.0, out);
  }
}

// Stood count of one sweep chunk (the per-generation monitor of the best candidate; sweep = chunks 0-5).
int SweepChunk(Worker& wk, int chunk, const std::vector<ParamSpec>& list, const std::vector<double>& xnorm) {
  std::vector<Track> tr;
  ControlChunk(wk, chunk, list, xnorm, &tr);
  int n = 0;
  for (const Track& t : tr) n += t.stood;
  return n;
}

// Control scores of several candidates (jobs = candidates x chunks, parallel); same Score() as the generation set.
constexpr int kStagChunks = 24;  // control stagger set: 6 moments x 4 worlds of 5 scenes
const double kCtlMoments[6] = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0};

// Control scores of several candidates (jobs = candidates x chunks, parallel): sweep 96 + 8 holds + 16 commanded steps
// + stand (Score) and 20 stagger scenes x 6 moments at the nominal body. ref = constraint baseline (nullptr: the
// candidate itself).
std::vector<Terms> ControlScores(std::vector<Worker*>& wk, int threads, const std::vector<ParamSpec>& list,
                                 const std::vector<std::vector<double>>& xs, const Weights& W, const Terms* ref, double scale) {
  const int K = (int)xs.size(), per = kCtlChunks + kStagChunks;
  std::vector<std::vector<Track>> tr(K * kCtlChunks);
  std::vector<std::vector<stag::Res>> sr(K * kStagChunks);
  ParallelFor(K * per, threads, wk, [&](Worker& w, int j) {
    const int k = j / per, c = j % per;
    if (c < kCtlChunks) {
      ControlChunk(w, c, list, xs[k], &tr[k * kCtlChunks + c]);
    } else {
      const int q = c - kCtlChunks;
      std::vector<stag::Item> items;
      for (int i = 0; i < 5; ++i) items.push_back({(q % 4) * 5 + i, kCtlMoments[q / 4], BodyParams(), scale});
      stag::RunWorld(items, [&](World& ww) { SetParams(ww, list, xs[k]); }, &sr[k * kStagChunks + q]);
    }
  });
  std::vector<Terms> out;
  for (int k = 0; k < K; ++k) {
    std::vector<Track> P, S = tr[k * kCtlChunks + 7];
    int sw = 0;
    for (int c = 0; c < 7; ++c)
      for (const Track& t : tr[k * kCtlChunks + c]) {
        P.push_back(t);
        if (c < 6) sw += t.stood;
      }
    Terms t = Score(P, S, W);
    t.sweep_stood = sw;
    std::vector<stag::Res> all;
    for (int q = 0; q < kStagChunks; ++q) all.insert(all.end(), sr[k * kStagChunks + q].begin(), sr[k * kStagChunks + q].end());
    AddStag(&t, all, scale);
    t.cost = Combine(t, ref ? *ref : t, W);
    out.push_back(t);
  }
  return out;
}

void Mkdir(const std::string& p) { mkdir(p.c_str(), 0755); }

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void WriteParams(const std::string& path, const std::vector<ParamSpec>& list, const std::vector<double>& xnorm, const char* head) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) return;
  fprintf(f, "# %s\n", head);
  for (size_t k = 0; k < list.size(); ++k)
    fprintf(f, "%s=%.6g\n", kParamDefs[list[k].idx].name, list[k].lo + Clamp(xnorm[k], 0.0, 1.0) * (list[k].hi - list[k].lo));
  fclose(f);
}

void PrintTerms(const Terms& t) {
  printf("cost %.3f | stagger stood %.0f/%.0f phase %.3f | hold %.3f steps %.3f stand %.3f | width %.3f sway %.1f mm pelvis-fwd %.3f | sweep %.0f/96\n",
         t.cost, t.stag_stood, t.stag_n, t.stag_phase, t.hold, t.steps, t.stand, t.width, t.sway, t.pfwd, t.sweep_stood);
}

}  // namespace

int RunOptBench(int argc, char** argv) {
  std::string name = argc >= 3 ? argv[2] : "run";
  int threads = 6, lambda = 16, gens = 1000;
  double sigma0 = 0.2, budget = 1e18, cur_scale = 0.5;
  int ctl_every = 5;
  std::string list_file, weights_file;
  int pos = 0;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--sigma") && i + 1 < argc) sigma0 = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--list") && i + 1 < argc) list_file = argv[++i];
    else if (!std::strcmp(argv[i], "--weights") && i + 1 < argc) weights_file = argv[++i];
    else if (!std::strcmp(argv[i], "--budget") && i + 1 < argc) budget = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--control") && i + 1 < argc) ctl_every = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) cur_scale = std::atof(argv[++i]);
    else if (pos == 0) { threads = std::atoi(argv[i]); ++pos; }
    else if (pos == 1) { lambda = std::atoi(argv[i]); ++pos; }
    else if (pos == 2) { gens = std::atoi(argv[i]); ++pos; }
  }
  Mkdir(TmpDir().c_str());
  Mkdir(TmpPath("opt").c_str());
  const std::string dir = TmpPath("opt/").c_str() + (name == "test" || name == "eval" ? std::string("_") + name : name);
  Mkdir(dir);
  if (list_file.empty()) {
    list_file = dir + "/params.txt";
    FILE* f = fopen(list_file.c_str(), "r");
    if (f) fclose(f);
    else { f = fopen(list_file.c_str(), "w"); if (f) { fputs(kDefaultList, f); fclose(f); } }
  }
  std::vector<ParamSpec> list;
  if (!LoadList(list_file.c_str(), &list)) return 2;
  for (ParamSpec& sp : list) {  // -p / --params give the start point (opt eval, or a new search from a found set)
    float v;
    if (OverrideValue(sp.idx, &v)) sp.start = v;
  }
  Weights W;
  if (weights_file.empty()) {
    weights_file = dir + "/weights.txt";
    FILE* f = fopen(weights_file.c_str(), "r");
    if (f) fclose(f);
    else {
      f = fopen(weights_file.c_str(), "w");
      if (f) { fputs("# term=weight (cost = sum weight x term; terms in log.csv)\n", f); for (auto& kv : W.w) fprintf(f, "%s=%g\n", kv.first.c_str(), kv.second); fclose(f); }
    }
  }
  LoadWeights(weights_file.c_str(), &W);
  const int n = (int)list.size();
  std::vector<Worker*> wk;
  for (int t = 0; t < threads; ++t) wk.push_back(new Worker());
  std::vector<double> x0(n);
  for (int k = 0; k < n; ++k) x0[k] = Clamp((list[k].start - list[k].lo) / (list[k].hi - list[k].lo), 0.0, 1.0);

  if (name == "phases") {  // measured phases of 16 pushes (4 directions x 4 forces) next to the table, penalty per scene
    std::vector<PushScene> sc;
    const double dirs[4] = {180, 0, 90, 270}, js[4] = {40, 45, 55, 60};
    for (int d = 0; d < 4; ++d)
      for (int k = 0; k < 4; ++k) sc.push_back({dirs[d], js[k], 2.0, BodyParams(), false});
    World* w = new World();
    BuildWorld(*w, std::vector<BodyParams>(16), 9);
    SetParams(*w, list, x0);
    std::vector<Track> tr;
    RunWorld(*w, sc, nullptr, 0, &tr);
    printf("ang  J  stood steps | per step: lift swing len tz knee (table in brackets) | trunk lo..hi (table) drop (table) | penalty\n");
    for (Track& tk : tr) {
      const double c = std::fabs(tk.ux), sn = std::fabs(tk.uy);
      const int type = c >= sn ? (tk.ux < 0 ? 0 : 1) : 2, size = tk.sc.J <= 45 ? 0 : 1;
      const Ref& r = kRef[type][size];
      printf("%3.0f %3.0f  %d  %d |", tk.sc.ang_deg, tk.sc.J, (int)tk.stood, tk.steps);
      std::sort(tk.sw.begin(), tk.sw.end(), [](const Swing& a, const Swing& b) { return a.t_lift < b.t_lift; });
      for (int k = 0; k < 2; ++k) {
        const RefStep& rs = k == 0 ? r.s1 : r.s2;
        if ((int)tk.sw.size() > k) {
          const Swing& q = tk.sw[k];
          printf(" %.2f(%.2f) %.2f(%.2f) %.2f(%.2f) %.2f(%.2f) %.0f(%.0f) |", q.t_lift - tk.sc.t_push, rs.lift, q.t_td - q.t_lift, rs.swing, q.len,
                 rs.len, q.tz, rs.tz, q.knee, rs.knee);
        } else {
          printf(" no step |");
        }
      }
      printf(" %.0f..%.0f (%.0f..%.0f) %.2f (%.2f) | %.2f\n", tk.lean_lo, tk.lean_hi, r.trunk_lo, r.trunk_hi, tk.drop, r.drop, PhasePenalty(tk));
    }
    delete w;
    return 0;
  }
  if (name == "test" || name == "eval") {  // one candidate = the current parameters
    const Gen g = MakeGen(1, cur_scale);
    double t0 = Now();
    Terms a = Evaluate(*wk[0], g, list, x0, W);
    printf("eval 1 (%.1f s): ", Now() - t0);
    PrintTerms(a);
    if (name == "test") {
      t0 = Now();
      Terms b = Evaluate(*wk[0], g, list, x0, W);
      printf("eval 2 same world (%.1f s): ", Now() - t0);
      PrintTerms(b);
      printf("difference of costs %.6f\n", std::fabs(a.cost - b.cost));
    } else {
      std::atomic<int> total(0);
      std::vector<Worker*> ws = wk;
      t0 = Now();
      ParallelFor(6, threads, ws, [&](Worker& w, int c) { total += SweepChunk(w, c, list, x0); });
      printf("full sweep %d/96 (%.1f s)\n", total.load(), Now() - t0);
    }
    return 0;
  }

  {  // the curriculum scale of an earlier run of this directory wins over --scale
    FILE* f = fopen((dir + "/scale.txt").c_str(), "r");
    double v;
    if (f) { if (fscanf(f, "%lf", &v) == 1) cur_scale = v; fclose(f); }
  }
  cma::Strategy cs;
  const std::string state = dir + "/state.txt";
  bool resumed = cs.Load(state.c_str(), n, lambda);
  if (!resumed) cs.Init(n, lambda, x0, sigma0);
  printf("opt '%s': %d parameters, lambda %d, mu %d, %d threads%s; output %s\n", name.c_str(), n, cs.lambda, cs.mu, threads,
         resumed ? (", resumed at generation " + std::to_string(cs.gen)).c_str() : "", dir.c_str());
  const std::string logp = dir + "/log.csv";
  {
    FILE* f = fopen(logp.c_str(), "r");
    if (f) fclose(f);
    else {
      f = fopen(logp.c_str(), "w");
      if (f) { fputs("gen,time_s,scale,best,mean,sigma,stag_fall,stag_phase,hold,steps,stand,bound,stag_stood,stag_n,sweep,ref_stag_stood,ref_stag_phase,ref_hold,ref_steps,ref_stand\n", f); fclose(f); }
    }
  }
  struct Top { double cost; int gen; std::vector<double> x; };
  std::vector<Top> top;  // best sets over the run by control cost (ascending), at most 3; file top.txt: cost gen x[0..n-1]
  const std::string topp = dir + "/top.txt", ctlp = dir + "/control.csv";
  if (FILE* f = fopen(topp.c_str(), "r")) {
    Top t;
    t.x.resize(n);
    while (fscanf(f, "%lf %d", &t.cost, &t.gen) == 2) {
      bool ok = true;
      for (int i = 0; i < n && ok; ++i) ok = fscanf(f, "%lf", &t.x[i]) == 1;
      if (!ok) break;
      top.push_back(t);
    }
    fclose(f);
  }
  {
    FILE* f = fopen(ctlp.c_str(), "r");
    if (f) fclose(f);
    else if ((f = fopen(ctlp.c_str(), "w"))) {
      fputs("gen,source,scale,key,cost,stag_stood,stag_n,stag_phase,hold,steps,stand,width,sway_mm,pfwd,sweep96\n", f);
      fclose(f);
    }
  }
  Terms ref_ctl;  // constraint baseline: the start set on the control set (ref.txt: hold steps stand)
  bool have_ref = false;
  if (FILE* f = fopen((dir + "/ref.txt").c_str(), "r")) {
    have_ref = fscanf(f, "%lf %lf %lf", &ref_ctl.hold, &ref_ctl.steps, &ref_ctl.stand) == 3;
    fclose(f);
  }
  // Score candidates on the control set, log them, merge into the top 3, rewrite top.txt / top1..3.params / best.params.
  auto control_round = [&](int gen, const std::vector<std::string>& names, const std::vector<std::vector<double>>& cands) {
    const double tc = Now();
    const std::vector<Terms> ct = ControlScores(wk, threads, list, cands, W, have_ref ? &ref_ctl : nullptr, cur_scale);
    if (!have_ref && !ct.empty()) {  // first call = the start set
      ref_ctl = ct[0];
      have_ref = true;
      if (FILE* rf = fopen((dir + "/ref.txt").c_str(), "w")) { fprintf(rf, "%.6f %.6f %.6f\n", ref_ctl.hold, ref_ctl.steps, ref_ctl.stand); fclose(rf); }
    }
    FILE* f = fopen(ctlp.c_str(), "a");
    for (size_t k = 0; k < ct.size(); ++k) {
      const Terms& t = ct[k];
      if (f)
        fprintf(f, "%d,%s,%.1f,%.4f,%.4f,%.0f,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f,%.0f\n", gen, names[k].c_str(), t.scale, t.key, t.cost, t.stag_stood,
                t.stag_n, t.stag_phase, t.hold, t.steps, t.stand, t.width, t.sway, t.pfwd, t.sweep_stood);
      printf("  control %s: ", names[k].c_str());
      PrintTerms(t);
      top.push_back({t.key, gen, cands[k]});  // rank by stagger stood, ties by phase distance
    }
    if (f) fclose(f);
    std::sort(top.begin(), top.end(), [](const Top& a, const Top& b) { return a.cost < b.cost; });
    if (top.size() > 3) top.resize(3);
    if (FILE* tf = fopen(topp.c_str(), "w")) {
      for (const Top& t : top) {
        fprintf(tf, "%.6f %d", t.cost, t.gen);
        for (double v : t.x) fprintf(tf, " %.12g", v);
        fprintf(tf, "\n");
      }
      fclose(tf);
    }
    for (size_t r = 0; r < top.size(); ++r) {
      const std::string head = "control key ((2 - scale) x 1000 + not stood + 0.1 phase) " + std::to_string(top[r].cost) + ", generation " + std::to_string(top[r].gen) + ", rank " + std::to_string(r + 1);
      WriteParams(dir + "/top" + std::to_string(r + 1) + ".params", list, top[r].x, head.c_str());
      if (r == 0) WriteParams(dir + "/best.params", list, top[r].x, head.c_str());
    }
    {  // curriculum: the best candidate of this round stood >= 60 % of the control stagger scenes -> harder pushes from now on
      double best_frac = 0;
      for (const Terms& t : ct) best_frac = std::fmax(best_frac, t.stag_n > 0 ? t.stag_stood / t.stag_n : 0.0);
      printf("  curriculum: best stood %.0f %% at scale %.1f", 100.0 * best_frac, cur_scale);
      if (best_frac >= 0.6 && cur_scale < 1.0 - 1e-9) {
        cur_scale = std::fmin(1.0, cur_scale + 0.1);
        if (FILE* sf = fopen((dir + "/scale.txt").c_str(), "w")) { fprintf(sf, "%.2f\n", cur_scale); fclose(sf); }
        printf(" -> scale %.1f", cur_scale);
      }
      printf("\n");
    }
    printf("control round (%.0f s): top 3 keys", Now() - tc);
    for (const Top& t : top) printf(" %.3f", t.cost);
    printf("\n");
    fflush(stdout);
  };
  if (!have_ref) control_round(-1, {"start"}, {x0});  // the starting point is a candidate too: best is never worse
  const double t_start = Now();
  for (int it = 0; it < gens; ++it) {
    const double t0 = Now();
    const int gen = cs.gen;
    const Gen g = MakeGen(1000 + gen, cur_scale);
    std::mt19937_64 rng(123456789ULL + 7919ULL * gen);
    std::vector<std::vector<double>> xs;
    cs.Ask(rng, &xs);
    std::vector<Terms> res(xs.size());
    std::vector<std::vector<double>> xe(xs.size(), std::vector<double>(n));
    for (size_t k = 0; k < xs.size(); ++k) {
      double bnd = 0;
      for (int i = 0; i < n; ++i) {
        double v = xs[k][i];
        bnd += std::fmax(0.0, std::fmax(-v, v - 1.0));
        v = std::fabs(std::fmod(v, 2.0));  // reflection into [0, 1]
        xe[k][i] = v > 1.0 ? 2.0 - v : v;
      }
      res[k].bound = bnd;
    }
    const int lam = (int)xs.size();
    res.resize(lam + 1);  // last = the start set on the same scenes (baseline of the constraints)
    xe.push_back(x0);
    ParallelFor(lam + 1, threads, wk, [&](Worker& w, int k) {
      const double bnd = res[k].bound;
      res[k] = Evaluate(w, g, list, xe[k], W);
      res[k].bound = bnd;
    });
    const Terms ref_gen = res[lam];
    for (int k = 0; k < lam; ++k) res[k].cost = Combine(res[k], ref_gen, W);
    res.resize(lam);
    xe.pop_back();
    std::vector<int> order(xs.size());
    for (size_t k = 0; k < order.size(); ++k) order[k] = (int)k;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return res[a].cost < res[b].cost; });
    double mean_cost = 0;
    for (const Terms& r : res) mean_cost += r.cost;
    mean_cost /= res.size();
    std::vector<std::vector<double>> sorted;
    for (int k : order) sorted.push_back(xs[k]);
    const int bi = order[0];
    // full sweep (10 s, fixed scenes) of the best candidate
    std::atomic<int> total(0);
    ParallelFor(6, threads, wk, [&](Worker& w, int c) { total += SweepChunk(w, c, list, xe[bi]); });
    Terms bt = res[bi];
    bt.sweep_stood = total.load();
    cs.Tell(sorted);
    cs.Save(state.c_str());
    WriteParams(dir + "/last.params", list, cs.mean, "mean of the search distribution");
    WriteParams(dir + "/gen_best.params", list, xe[bi], ("best of generation " + std::to_string(gen)).c_str());
    FILE* f = fopen(logp.c_str(), "a");
    if (f) {
      fprintf(f, "%d,%.0f,%.1f,%.4f,%.4f,%.5f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.0f,%.0f,%.0f,%.0f,%.4f,%.4f,%.4f,%.4f\n", gen, Now() - t_start, g.stag[0].scale, bt.cost,
              mean_cost, cs.sigma, bt.stag_fall, bt.stag_phase, bt.hold, bt.steps, bt.stand, bt.bound, bt.stag_stood, bt.stag_n, bt.sweep_stood,
              ref_gen.stag_stood, ref_gen.stag_phase, ref_gen.hold, ref_gen.steps, ref_gen.stand);
      fclose(f);
    }
    printf("gen %d scale %.1f (%.0f s) sigma %.4f mean cost %.3f; start set on the same scenes: stagger stood %.0f phase %.3f hold %.2f steps %.2f stand %.2f\n  best: ",
           gen, g.stag[0].scale, Now() - t0, cs.sigma, mean_cost, ref_gen.stag_stood, ref_gen.stag_phase, ref_gen.hold, ref_gen.steps, ref_gen.stand);
    PrintTerms(bt);
    {  // spread of every term over the candidates of the generation: do the terms tell candidates apart?
      double lo[5], hi[5];
      for (int q = 0; q < 5; ++q) { lo[q] = 1e9; hi[q] = -1e9; }
      for (const Terms& r : res) {
        const double v[5] = {r.stag_stood, r.stag_phase, r.hold, r.steps, r.stand};
        for (int q = 0; q < 5; ++q) { lo[q] = std::fmin(lo[q], v[q]); hi[q] = std::fmax(hi[q], v[q]); }
      }
      printf("  spread over candidates (min..max): stagger stood %.0f..%.0f phase %.3f..%.3f hold %.2f..%.2f steps %.2f..%.2f stand %.2f..%.2f cost %.2f..%.2f\n",
             lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3], lo[4], hi[4], res[order.front()].cost, res[order.back()].cost);
    }
    fflush(stdout);
    if (ctl_every > 0 && (gen + 1) % ctl_every == 0) {
      std::vector<std::string> names;
      std::vector<std::vector<double>> cands;
      for (int r = 0; r < 3 && r < (int)order.size(); ++r) {
        names.push_back("g" + std::to_string(gen) + "_rank" + std::to_string(r + 1));
        cands.push_back(xe[order[r]]);
      }
      std::vector<double> mx(n);
      for (int i = 0; i < n; ++i) mx[i] = Clamp(cs.mean[i], 0.0, 1.0);  // mean after the update
      names.push_back("g" + std::to_string(gen) + "_mean");
      cands.push_back(mx);
      control_round(gen, names, cands);
    }
    if (Now() - t_start > budget) break;
  }
  return 0;
}
