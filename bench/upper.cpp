// Upper-body bench: how the back, neck and arms move after the stagger pushes, with the upper layer on and off.
//   emil_bench upper [label] [threads=3] [scene]   (-p / --params apply to both runs; upper is set 1 and 0 here)
//   scene: per-frame trace of that scene (push at 1.5 s) -> $EMIL_TMP/upper/trace_u{1,0}.csv; label stand = stand only
// Scenes: the 20 chest pushes of the stagger bench (bench/stagger.h, same point and impulse), pushed at 1.5 .. 3.0 s
// every 0.25 s (140 runs per setting), 6 s after. Per scene and run: largest shoulder abduction of each arm over its
// value at the push [deg] and when [s after the push]; the time the larger one is back within a quarter of (peak -
// final value) of its final value (stood runs); largest hand separation [m]; elbow min and max [deg]; back twist
// (|lumbar + chest yaw|, chest vs pelvis) and trunk lean along the push (pelvis -> chest, relative), maxima [deg];
// ringing = trunk lean reversals larger than 1.5 deg in 0.3..3 s; coupling of shoulder flexion with (opposite - own)
// hip flexion over the frames the balancer steps (Pearson r and slope); neck yaw toward the push origin [deg] (side
// pushes: + = toward; others: largest |neck yaw|); stood (stagger criterion: pelvis above 0.5 m and no trunk-to-floor
// contact). Stand check: 10 s quiet stance, hand sway relative to the chest (rms, mm), hand jitter (rms second
// difference per frame, mm), mean elbow and abduction [deg]. Output: table on stdout, per scene
// $EMIL_TMP/upper/<label>/scenes.csv.
#include <mujoco/mujoco.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "tmp_dir.h"
#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"

using namespace emil;
using namespace stag;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

namespace {

constexpr int kChunk = 5;
int g_trace = -1;  // scene index traced (moment 1.5) to $EMIL_TMP/upper/trace_u{1,0}.csv

struct URes {
  int scene = 0, upper = 1;
  double moment = 1.5;
  bool stood = true;
  double abd[2] = {0, 0}, abd_t[2] = {0, 0}, ret_t = -1;
  double sep0 = 0, sep = 0, el_min = 1e9, el_max = -1e9, twist = 0, lean = 0;
  int ring = 0;
  double r = 0, slope = 0;
  int nstep = 0;      // frames in steps
  double head = 0;    // max head yaw toward the push origin (side pushes) or max |head yaw change| [deg]
  double neck = 0;    // max |neck yaw| [deg]
};

double Wrap(double a) {
  while (a > M_PI) a -= 2 * M_PI;
  while (a < -M_PI) a += 2 * M_PI;
  return a;
}
double Yaw(const double* R) { return std::atan2(R[3], R[0]); }

struct Tk {
  bool pushed = false, fell = false;
  double t_push = 0, ux = 0, uy = 0, lean0 = 0, yaw0 = 0, abd0[2] = {0, 0};
  double ext = 0;  // ringing: running extreme of the lean
  int dir = 0, dirn = 0;
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  std::vector<double> abd_hist;
};

double Lean(const mjData* d, const Bot& b, double ux, double uy) {
  const double* p = d->xpos + 3 * (b.body0 + B_PELVIS);
  const double* c = d->xpos + 3 * (b.body0 + B_CHEST);
  return std::atan2((c[0] - p[0]) * ux + (c[1] - p[1]) * uy, c[2] - p[2]) * kRad;
}

void RunChunk(int chunk, int upper, double kMoment, std::vector<URes>* out) {
  std::vector<BodyParams> bodies(kChunk);
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(BuildMutex());
    if (!w.Build(bodies, 3.0, 21, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return; }
  }
  ApplyOverrides(w);
  w.param[P_UPPER] = (float)upper;
  w.Reset();
  const mjModel* m = w.model();
  mjData* d = w.data();
  const int body00 = w.bot(0).body0;
  std::vector<Tk> tk(kChunk);
  std::vector<URes> res(kChunk);
  const int frames = (int)std::lround((kMoment + kTail) * 60.0) + 2;
  FILE* tr = nullptr;
  const int itr = g_trace - chunk * kChunk;
  if (itr >= 0 && itr < kChunk && kMoment == 1.5) {
    tr = fopen(upper ? TmpPath("upper/trace_u1.csv").c_str() : TmpPath("upper/trace_u0.csv").c_str(), "w");
    if (tr) fprintf(tr,
        "t,mode,fall,spread,elL,TelL,kpelL,elR,TelR,abdL,TabdL,abdR,TabdR,flexL,TflexL,flexR,TflexR,kpabdL,"
                        "lyaw,cyaw,Tcyaw,neck,Tneck,hipL,hipR,swing,tp,troll,pz,vy,ldL,ldR,vx,pp,shrot\n");
  }
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    for (int i = 0; i < kChunk; ++i) {
      Bot& b = w.bot(i);
      Tk& k = tk[i];
      URes& r = res[i];
      const double* q = d->qpos + b.qadr0;
      if (tr && i == itr && t > kMoment - 0.2 && t < kMoment + 3.0) {
        const double* T = b.target;
        fprintf(tr, "%.3f,%d,%d,%.3f,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,"
                    "%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%d,%d,%.3f,%.3f,%.3f\n", t, b.mode, b.fall_pred,
                        b.up_spread, q[J_ELBOW_L], T[J_ELBOW_L], b.kp[J_ELBOW_L],
                q[J_ELBOW_R], T[J_ELBOW_R], q[J_SH_ABD_L], T[J_SH_ABD_L], q[J_SH_ABD_R], T[J_SH_ABD_R], q[J_SH_FLEX_L],
                T[J_SH_FLEX_L], q[J_SH_FLEX_R], T[J_SH_FLEX_R], b.kp[J_SH_ABD_L], q[J_LUMBAR_YAW], q[J_CHEST_YAW],
                T[J_CHEST_YAW], q[J_NECK_YAW], T[J_NECK_YAW], q[J_HIP_FLEX_L], q[J_HIP_FLEX_R],
                    b.mode == 1 ? b.swing : -1,
                b.pc[b.cur].trunk_pitch, b.pc[b.cur].trunk_roll, d->xpos[3 * b.body0 + 2], b.pc[b.cur].com_vel[1],
                b.loaded[0], b.loaded[1], b.pc[b.cur].com_vel[0], b.pc[b.cur].pelvis_pitch, q[J_SH_ROT_L]);
      }
      const double* hl = d->xpos + 3 * (b.body0 + B_HAND_L);
      const double* hr = d->xpos + 3 * (b.body0 + B_HAND_R);
      const double sep = std::sqrt((hl[0] - hr[0]) * (hl[0] - hr[0]) + (hl[1] - hr[1]) * (hl[1] - hr[1]) +
                                   (hl[2] - hr[2]) * (hl[2] - hr[2]));
      const double* Rh = d->xmat + 9 * (b.body0 + B_HEAD);
      if (!k.pushed) {
        if (t < kMoment) continue;
        const Scene& sc = kScenes[chunk * kChunk + i];
        r.scene = chunk * kChunk + i;
        r.upper = upper;
        r.moment = kMoment;
        const double* R = d->xmat + 9 * b.body0;
        const double hn = std::hypot(R[0], R[3]) + 1e-9, hx = R[0] / hn, hy = R[3] / hn;
        const double dx[4] = {-hx, hx, -hy, hy}, dy[4] = {-hy, hy, hx, -hx};
        k.ux = dx[sc.dir];
        k.uy = dy[sc.dir];
        k.dir = sc.dir;
        k.pushed = true;
        k.t_push = t;
        k.lean0 = Lean(d, b, k.ux, k.uy);
        k.yaw0 = Yaw(Rh);
        k.abd0[0] = q[J_SH_ABD_L];
        k.abd0[1] = q[J_SH_ABD_R];
        r.sep0 = sep;
        const double J = sc.J * b.dims.total_mass / kRefMass;
        const double* o = d->xpos + 3 * (b.body0 + B_CHEST);
        const double pt[3] = {o[0] - 0.1 * k.ux, o[1] - 0.1 * k.uy, o[2] + 0.06};
        const double imp[3] = {k.ux * J, k.uy * J, 0};
        w.PushAt(i, imp, pt, B_CHEST);  // standard chest push point
        continue;
      }
      const double dt = t - k.t_push;
      if (dt > kTail || k.fell) continue;
      if (d->xpos[3 * (b.body0 + B_PELVIS) + 2] < 0.5) { k.fell = true; r.stood = false; continue; }
      for (int s = 0; s < 2; ++s) {
        const double a = (q[s ? J_SH_ABD_R : J_SH_ABD_L] - k.abd0[s]) * kRad;
        if (dt <= 3.0 && a > r.abd[s]) { r.abd[s] = a; r.abd_t[s] = dt; }
      }
      k.abd_hist.push_back(std::fmax(q[J_SH_ABD_L] - k.abd0[0], q[J_SH_ABD_R] - k.abd0[1]) * kRad);
      if (dt <= 3.0) {
        r.sep = std::fmax(r.sep, sep);
        for (int s = 0; s < 2; ++s) {
          const double e = q[s ? J_ELBOW_R : J_ELBOW_L] * kRad;
          r.el_min = std::fmin(r.el_min, e);
          r.el_max = std::fmax(r.el_max, e);
        }
        r.twist = std::fmax(r.twist, std::fabs(q[J_LUMBAR_YAW] + q[J_CHEST_YAW]) * kRad);
        const double l = Lean(d, b, k.ux, k.uy) - k.lean0;
        if (std::fabs(l) > std::fabs(r.lean)) r.lean = l;
        // ringing: reversals of the lean by more than 1.5 deg after 0.3 s (running extreme in the current direction)
        if (dt > 0.3) {
          if (k.dirn == 0) {
            if (std::fabs(l - k.ext) > 1.5) { k.dirn = l > k.ext ? 1 : -1; k.ext = l; }
          } else if ((l - k.ext) * k.dirn > 0) {
            k.ext = l;
          } else if ((k.ext - l) * k.dirn > 1.5) {
            ++r.ring;
            k.dirn = -k.dirn;
            k.ext = l;
          }
        } else {
          k.ext = l;
        }
        const double hy = q[J_NECK_YAW] * kRad;  // head against the chest (the whole body also turns)
        if (k.dir >= 2) {  // side push: the pusher is on the side the push comes from (left push from the right)
          const double toward = k.dir == 2 ? -hy : hy;  // left push (toward +y) comes from -y: turn right (-)
          r.head = std::fmax(r.head, toward);
        } else {
          r.head = std::fmax(r.head, std::fabs(hy));
        }
        r.neck = std::fmax(r.neck, std::fabs(q[J_NECK_YAW]) * kRad);
      }
      if (b.mode == 1) {  // coupling during steps, both arms pooled
        for (int s = 0; s < 2; ++s) {
          const double x = s ? q[J_HIP_FLEX_L] - q[J_HIP_FLEX_R] : q[J_HIP_FLEX_R] - q[J_HIP_FLEX_L];
          const double y = q[s ? J_SH_FLEX_R : J_SH_FLEX_L];
          k.sx += x; k.sy += y; k.sxx += x * x; k.syy += y * y; k.sxy += x * y;
          ++r.nstep;
        }
      }
    }
    for (int c = 0; c < d->ncon; ++c) {  // pelvis .. head on the floor
      const mjContact& cn = d->contact[c];
      const int g = cn.geom1 == 0 ? cn.geom2 : (cn.geom2 == 0 ? cn.geom1 : -1);
      if (g < 0) continue;
      const int rel0 = m->geom_bodyid[g] - body00;
      if (rel0 < 0) continue;
      const int i = rel0 / B_COUNT, rel = rel0 % B_COUNT;
      if (i < kChunk && rel >= B_PELVIS && rel <= B_HEAD && tk[i].pushed && t - tk[i].t_push <= kTail) {
        tk[i].fell = true;
        res[i].stood = false;
      }
    }
  }
  if (tr) fclose(tr);
  for (int i = 0; i < kChunk; ++i) {
    Tk& k = tk[i];
    URes& r = res[i];
    const int n = r.nstep;
    if (n > 10) {
      const double vx = k.sxx - k.sx * k.sx / n, vy = k.syy - k.sy * k.sy / n, cxy = k.sxy - k.sx * k.sy / n;
      r.r = cxy / std::sqrt(std::fmax(vx * vy, 1e-12));
      r.slope = cxy / std::fmax(vx, 1e-12);
    }
    const double pk = std::fmax(r.abd[0], r.abd[1]);
    const int ipk = (int)std::lround(std::fmax(r.abd_t[0], r.abd_t[1]) * 60.0);
    // back = the larger abduction within a quarter of (peak - final) of its final value (mean of the last second):
    // after steps the stance is wider and the arms follow the hip abduction, so the final value is not
    // the start
    const int nh = (int)k.abd_hist.size();
    if (pk > 3.0 && r.stood && nh > 120) {
      double fin = 0;
      for (int f = nh - 60; f < nh; ++f) fin += k.abd_hist[f];
      fin /= 60.0;
      for (int f = std::max(0, ipk); f < nh; ++f)
        if (k.abd_hist[f] - fin < 0.25 * (pk - fin)) { r.ret_t = (f + 1) / 60.0; break; }
    }
    if (r.el_min > 1e8) r.el_min = r.el_max = 0;
    out->push_back(r);
  }
}

struct Stand { double sway = 0, jit = 0, elbow = 0, abd = 0, sep = 0, pp_hand = 0, pp_head = 0, pp_pelvis = 0; };  // pp: world range [mm]

// 10 s quiet stance of one bot (after 3 s settling): hands in the chest frame.
Stand RunStand(int upper) {
  std::vector<BodyParams> bodies(1);
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(BuildMutex());
    if (!w.Build(bodies, 3.0, 21, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return Stand(); }
  }
  ApplyOverrides(w);
  w.param[P_UPPER] = (float)upper;
  w.Reset();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  std::vector<double> h[2][3];
  double lo[4][3], hi[4][3];  // world range of left hand, right hand, head, pelvis
  for (int i = 0; i < 4; ++i) for (int a = 0; a < 3; ++a) { lo[i][a] = 1e9; hi[i][a] = -1e9; }
  Stand s;
  FILE* tr = fopen(upper ? TmpPath("upper/stand_u1.csv").c_str() : TmpPath("upper/stand_u0.csv").c_str(), "w");
  if (tr) fprintf(tr, "t,spread,tp,tr,flexL,TflexL,abdL,TabdL,elL,TelL,neck_yaw,chest_yaw,mode\n");
  int n = 0;
  for (int fr = 0; fr < 13 * 60; ++fr) {
    w.StepFrame();
    if (fr < 180) continue;  // the spawn settles in about 2.5 s (measured on the trunk pitch)
    const double* c = d->xpos + 3 * (b.body0 + B_CHEST);
    const double* R = d->xmat + 9 * (b.body0 + B_CHEST);
    for (int side = 0; side < 2; ++side) {
      const double* p = d->xpos + 3 * (b.body0 + (side ? B_HAND_R : B_HAND_L));
      const double v[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
      for (int a = 0; a < 3; ++a) h[side][a].push_back(R[a] * v[0] + R[3 + a] * v[1] + R[6 + a] * v[2]);
    }
    const double* q = d->qpos + b.qadr0;
    if (tr) {
      const Percept& P = b.pc[b.cur];
      fprintf(tr, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n", d->time, b.up_spread,
          P.trunk_pitch,
              P.trunk_roll, q[J_SH_FLEX_L], b.target[J_SH_FLEX_L], q[J_SH_ABD_L], b.target[J_SH_ABD_L], q[J_ELBOW_L],
              b.target[J_ELBOW_L], q[J_NECK_YAW], q[J_CHEST_YAW], b.mode);
    }
    s.elbow += 0.5 * (q[J_ELBOW_L] + q[J_ELBOW_R]) * kRad;
    s.abd += 0.5 * (q[J_SH_ABD_L] + q[J_SH_ABD_R]) * kRad;
    const double* hl = d->xpos + 3 * (b.body0 + B_HAND_L);
    const double* hr = d->xpos + 3 * (b.body0 + B_HAND_R);
    s.sep += std::hypot(hl[0] - hr[0], hl[1] - hr[1]);
    const double* hd = d->xpos + 3 * (b.body0 + B_HEAD);
    const double* pw[4] = {hl, hr, hd, d->xpos + 3 * (b.body0 + B_PELVIS)};
    for (int i = 0; i < 4; ++i)
      for (int a = 0; a < 3; ++a) { lo[i][a] = std::fmin(lo[i][a], pw[i][a]); hi[i][a] = std::fmax(hi[i][a], pw[i][a]); }
    ++n;
  }
  if (tr) fclose(tr);
  double sw = 0, jt = 0;
  int nj = 0;
  for (int side = 0; side < 2; ++side)
    for (int a = 0; a < 3; ++a) {
      const std::vector<double>& x = h[side][a];
      double mean = 0;
      for (double v : x) mean += v;
      mean /= x.size();
      for (double v : x) sw += (v - mean) * (v - mean);
      for (size_t f = 2; f < x.size(); ++f) {
        const double dd = x[f] - 2 * x[f - 1] + x[f - 2];
        jt += dd * dd;
        ++nj;
      }
    }
  s.sway = 1000 * std::sqrt(sw / (2.0 * n));  // rms 3D displacement per hand
  for (int i = 0; i < 4; ++i)
    for (int a = 0; a < 3; ++a) {
      const double r = 1000 * (hi[i][a] - lo[i][a]);
      double& pp = i < 2 ? s.pp_hand : (i == 2 ? s.pp_head : s.pp_pelvis);
      pp = std::fmax(pp, r);
    }
  s.jit = 1000 * std::sqrt(jt / std::max(1, nj) * 3.0);
  s.elbow /= n;
  s.abd /= n;
  s.sep /= n;
  return s;
}

}  // namespace

int RunUpperBench(int argc, char** argv) {
  std::string label = argc > 2 ? argv[2] : "cur";
  int threads = argc > 3 ? std::atoi(argv[3]) : 3;
  if (argc > 4)
    for (int i = 0; i < 20; ++i)
      if (!std::strcmp(argv[4], kScenes[i].name)) g_trace = i;
  threads = std::max(1, std::min(threads, 3));
  if (label == "stand") {  // stand check only, traces in $EMIL_TMP/upper/stand_u{0,1}.csv
    mkdir(TmpPath("upper").c_str(), 0755);
    const Stand a = RunStand(1), c = RunStand(0);
    printf("stand: world range hand %.1f %.1f mm, head %.1f %.1f mm, pelvis %.1f %.1f mm (upper 1, 0)\n", a.pp_hand,
           c.pp_hand, a.pp_head, c.pp_head, a.pp_pelvis, c.pp_pelvis);
    printf("stand: sway %.1f %.1f mm, jitter %.2f %.2f mm, elbow %.1f %.1f, abd %.1f %.1f deg (upper 1, 0)\n", a.sway,
        c.sway,
           a.jit, c.jit, a.elbow, c.elbow, a.abd, c.abd);
    return 0;
  }
  struct Job { int chunk, upper; double moment; };
  std::vector<Job> jobs;
  // Push times 1.5 .. 3.0 s every 0.25 s: the balancer stands at the edge of its range here and single runs flip with
  // small changes, so 140 runs per setting are needed for a usable comparison.
  std::vector<double> moments;
  for (int k = 0; k < 7; ++k) moments.push_back(1.5 + 0.25 * k);
  if (g_trace >= 0) moments.assign(1, 1.5);
  for (int u = 1; u >= 0; --u)
    for (double mo : moments)
      for (int c = 0; c < 20 / kChunk; ++c)
        if (g_trace < 0 || c == g_trace / kChunk) jobs.push_back({c, u, mo});
  std::vector<std::vector<URes>> res(jobs.size());
  Stand st[2];
  std::atomic<int> next(0);
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t)
    th.emplace_back([&]() {
      for (;;) {
        const int j = next.fetch_add(1);
        if (j > (int)jobs.size() + 1) break;
        if (j == (int)jobs.size()) st[1] = RunStand(1);
        else if (j == (int)jobs.size() + 1) st[0] = RunStand(0);
        else RunChunk(jobs[j].chunk, jobs[j].upper, jobs[j].moment, &res[j]);
      }
    });
  for (auto& x : th) x.join();
  std::vector<URes> all[2];
  (void)0;
  for (size_t j = 0; j < jobs.size(); ++j) all[jobs[j].upper].insert(all[jobs[j].upper].end(), res[j].begin(),
      res[j].end());
  for (int u = 0; u < 2; ++u)
    std::sort(all[u].begin(), all[u].end(), [](const URes& a,
        const URes& b) { return a.scene != b.scene ? a.scene < b.scene : a.moment < b.moment; });

  mkdir(TmpDir().c_str(), 0755);
  mkdir(TmpPath("upper").c_str(), 0755);
  const std::string dir = TmpPath("upper/").c_str() + label;
  mkdir(dir.c_str(), 0755);
  FILE* f = fopen((dir + "/scenes.csv").c_str(), "w");
  if (f) fprintf(f,
      "scene,moment,upper,stood,abdL,abdL_t,abdR,abdR_t,ret_t,sep0,sep,el_min,el_max,twist,lean,ring,r,slope,head,neck\n");
  printf("stand 10 s      upper=1  upper=0\n");
  printf("  hand sway mm  %7.1f  %7.1f\n", st[1].sway, st[0].sway);
  printf("  jitter mm     %7.2f  %7.2f\n", st[1].jit, st[0].jit);
  printf("  elbow deg     %7.1f  %7.1f\n", st[1].elbow, st[0].elbow);
  printf("  abd deg       %7.1f  %7.1f\n", st[1].abd, st[0].abd);
  printf("  hands apart m %7.2f  %7.2f\n", st[1].sep, st[0].sep);
  printf("%-9s u st abdL@t     abdR@t     ret  sep  elbow    twist lean ring  r    slope head neck\n", "scene");
  int stood[2] = {0, 0};
  for (int u = 1; u >= 0; --u)
    for (const URes& r : all[u]) {
      stood[u] += r.stood;
      if (r.moment == 1.5) printf("%-9s %d %d %4.0f@%.2f %4.0f@%.2f %4.2f %.2f %3.0f-%3.0f %5.1f %5.1f %2d %5.2f %5.2f %4.0f %4.0f\n",
             kScenes[r.scene].name, u, (int)r.stood, r.abd[0], r.abd_t[0], r.abd[1], r.abd_t[1], r.ret_t, r.sep,
             r.el_min, r.el_max, r.twist, r.lean, r.ring, r.r, r.slope, r.head, r.neck);
      if (f)
        fprintf(f, "%s,%.1f,%d,%d,%.1f,%.2f,%.1f,%.2f,%.2f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%d,%.3f,%.3f,%.1f,%.1f\n",
                kScenes[r.scene].name, r.moment, u, (int)r.stood, r.abd[0], r.abd_t[0], r.abd[1], r.abd_t[1], r.ret_t,
                    r.sep0,
                r.sep, r.el_min, r.el_max, r.twist, r.lean, r.ring, r.r, r.slope, r.head, r.neck);
    }
  if (f) fclose(f);
  (void)0;
  printf("stood: upper=1 %d of %d, upper=0 %d of %d\n", stood[1], (int)all[1].size(), stood[0], (int)all[0].size());
  return 0;
}
