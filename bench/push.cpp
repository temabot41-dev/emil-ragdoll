// Push bench: 20 scenes x 6 push times, stood / swings / same / short / late / edge metrics, CSV per run.
#include <array>
#include <atomic>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <algorithm>

#include "tmp_dir.h"
#include "emil_world.h"
#include "footspin.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p name=value or --params file
bool HasOverrides();

double g_stag_dv = 0;  // > 0: push like the stagger bench (chest point 0.1 m behind, 0.06 m up; dv x mass)

namespace {

struct Scene {
  const char* name; double dx, dy, J;
  int cmd_foot = -1;  // >= 0 = commanded step instead of a push (dx, dy = step [m], dy outward)
  int cross = 0;      // commanded cross step: dy = 0.1 m past the other ankle
  int pt = 0;         // chest pushes (g_chest): 0 chest centre, 1 shoulder joint on the hit side, 2 head centre
};
const Scene kScenes[21] = {
    {"back30", -1, 0, 30},  {"back45", -1, 0, 45},  {"back60", -1, 0, 60},   {"back80", -1, 0, 80},
    {"back100", -1, 0, 100}, {"back120", -1, 0, 120}, {"fwd30", 1, 0, 30},    {"fwd45", 1, 0, 45},
    {"fwd60", 1, 0, 60},    {"fwd80", 1, 0, 80},    {"fwd100", 1, 0, 100},   {"left30", 0, 1, 30},
    {"left45", 0, 1, 45},   {"left60", 0, 1, 60},   {"left80", 0, 1, 80},    {"left100", 0, 1, 100},
    {"left120", 0, 1, 120}, {"right45", 0, -1, 45}, {"right60", 0, -1, 60},  {"right100", 0, -1, 100},
    {"right30", 0, -1, 30}};  // traced only (push set = first 20)
const double kMoments[6] = {1.5, 2.0, 2.5, 3.0, 3.5, 4.0};

bool g_ptrace = false;
bool g_chest = false;  // sweep: impulse on the chest body at its centre (like the editor push)
FILE* g_dbg = nullptr;  // push1: per-frame balancer state for tools/plot_push.py
FILE* g_yaw = nullptr;  // push1: yaw trace $EMIL_TMP/pk/one/yaw.csv
std::string g_an_dir;  // non-empty = write a per-frame analysis CSV per run (emil_bench fall)
std::mutex g_build_mu;  // model compile is not thread-safe here: serialise Build

struct Result { int fell = 0, swings = 0, same = 0, shorts = 0, late = 0, edge_n = 0, edge_hit = 0;
  double thw = 0, abd = 0;  // max thigh angle from vertical (world) and |hip abduction| [deg], pelvis > 0.6 m
  int thm = -1;  // mode at the max thigh angle (debug: 0 stand, 1 step, 2 fallen)
  // stance foot contact frames: [0 quiet stand, 1 single support, 2 double support][flat, edge, corner, none]
  double fc[3][4] = {};
  double fall_t = -1, step0 = -1;  // first fall frame time, first step start after the push [s]
  int tds = 0;  // touchdowns after the push
  double sw_max = 0;  // longest step (step start -> touchdown, or -> fall/end while stepping) [s]
  double dv_max = 0, ptilt = 0, yaw_all = 0;  // max CoM horizontal speed gain (command .. touchdown+0.3 s) [m/s],
                                              // max pelvis tilt [deg], max pelvis yaw while up [deg]
  // first touchdown error [m] and time [s], trunk tilt [deg], pelvis drop [m]
  double place_err = -1, td1 = -1, tilt = 0, drop = 0;
  // max pelvis heading change, chest-pelvis angle, thigh flexion [deg] until mode 2 or 2 s
  double yaw = 0, chest = 0, flex = 0;
  std::vector<double> e_lon, e_lat;     // every balancer step end, sole centre minus place along / across [m]
  std::vector<double> e_over, e_yaw, e_len;  // overshoot [m], pelvis yaw range [rad], step length [m]
};

// Foot box point in world: local (sx * fx, 0, -sz) with fx = +1 toe, -1 heel.
void FootPoint(const mjModel* m, const mjData* d, int g, double fx, double out[3]) {
  const double* R = d->geom_xmat + 9 * g;
  const double* s = m->geom_size + 3 * g;
  const double l[3] = {fx * s[0], 0, -s[2]};
  for (int i = 0; i < 3; ++i) out[i] = d->geom_xpos[3 * g + i] + R[3 * i] * l[0] + R[3 * i + 1] * l[1] + R[3 * i + 2] * l[2];
}

// One CSV row (format in docs/BENCH_FORMAT.md); contact = foot box touches the floor (raw contact).
void WriteRow(FILE* fp, const mjModel* m, const mjData* d, const Bot& b, double t) {
  if (!fp) return;
  const double* pp = d->xpos + 3 * b.body0;
  const double* hd = d->xipos + 3 * (b.body0 + B_HEAD);
  fprintf(fp, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", t, pp[0], pp[1], pp[2], hd[0], hd[1], hd[2]);
  double toe[2][3], heel[2][3], roll[2];
  for (int f = 0; f < 2; ++f) {
    const double* a = d->xanchor + 3 * (b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L));
    fprintf(fp, ",%.4f,%.4f,%.4f", a[0], a[1], a[2]);
    FootPoint(m, d, b.geom_foot[f], 1, toe[f]);
    FootPoint(m, d, b.geom_foot[f], -1, heel[f]);
    const double yz = d->geom_xmat[9 * b.geom_foot[f] + 7];
    roll[f] = std::asin(yz < -1 ? -1 : (yz > 1 ? 1 : yz)) * 57.29578 * (f == 0 ? 1 : -1);
  }
  for (int f = 0; f < 2; ++f) fprintf(fp, ",%.4f,%.4f,%.4f", toe[f][0], toe[f][1], toe[f][2]);
  for (int f = 0; f < 2; ++f) fprintf(fp, ",%.4f,%.4f,%.4f", heel[f][0], heel[f][1], heel[f][2]);
  int con[2] = {0, 0};
  for (int i = 0; i < d->ncon; ++i) {
    const mjContact& c = d->contact[i];
    for (int f = 0; f < 2; ++f)
      if ((c.geom1 == 0 && c.geom2 == b.geom_foot[f]) || (c.geom2 == 0 && c.geom1 == b.geom_foot[f])) con[f] = 1;
  }
  fprintf(fp, ",%.2f,%.2f,%d,%d\n", roll[0], roll[1], con[0], con[1]);
}

}  // namespace

// Push row plus chest (CoM), left and right hand (CoM) points: reaction bench CSV.
void WriteRowExtra(FILE* fp, const mjModel* m, const mjData* d, const Bot& b, double t) {
  if (!fp) return;
  WriteRow(fp, m, d, b, t);
  fseek(fp, -1, SEEK_CUR);  // overwrite the newline
  const int ids[3] = {B_CHEST, B_HAND_L, B_HAND_R};
  for (int k = 0; k < 3; ++k) {
    const double* x = d->xipos + 3 * (b.body0 + ids[k]);
    fprintf(fp, ",%.4f,%.4f,%.4f", x[0], x[1], x[2]);
  }
  const int js[4] = {J_KNEE_L, J_KNEE_R, J_HIP_FLEX_L, J_HIP_FLEX_R};  // degrees, flexion positive as in the model
  for (int k = 0; k < 4; ++k) fprintf(fp, ",%.1f", d->qpos[b.qadr0 + js[k]] * 57.29578);
  fprintf(fp, "\n");
}

namespace {

Result RunOne(const Scene& sc, double t_push, const std::string& dir) {
  Result r;
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(g_build_mu);
    if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) {
      fprintf(stderr, "build failed (%s@%.1f): %s\n", sc.name, t_push, err);
      r.fell = 1;
      return r;
    }
  }
  ApplyOverrides(w);
  if (HasOverrides()) w.Reset();  // spawn pose from the overridden parameters (as in the picker)
  const mjModel* m = w.model();
  mjData* d = w.data();
  Bot& b = w.bot(0);
  // Contact points of foot f on the floor with a normal force above 5 N: flat >= 3, edge 2, corner 1, none 0.
  auto foot_class = [&](int f) {
    int n = 0;
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      if (!((c.geom1 == 0 && c.geom2 == b.geom_foot[f]) || (c.geom2 == 0 && c.geom1 == b.geom_foot[f]))) continue;
      mjtNum fr6[6];
      mj_contactForce(m, d, i, fr6);
      if (std::fabs(fr6[0]) > 5.0) ++n;
    }
    return n >= 3 ? 0 : (n == 2 ? 1 : (n == 1 ? 2 : 3));
  };
  auto tally = [&](bool quiet) {
    if (b.mode == 2) return;
    const double W = b.dims.total_mass * 9.81;
    if (b.mode == 1 && b.phase != 3) {
      r.fc[quiet ? 0 : 1][foot_class(b.swing ^ 1)] += 1;
    } else {
      for (int f = 0; f < 2; ++f)
        if (b.foot_fn[f] > 0.1 * W) r.fc[quiet ? 0 : 2][foot_class(f)] += 1;
    }
  };
  FILE* an = nullptr;
  if (!g_an_dir.empty()) {
    char ap[512];
    snprintf(ap, sizeof ap, "%s/%s@%.1f.csv", g_an_dir.c_str(), sc.name, t_push);
    an = fopen(ap, "w");
    if (an)
      fprintf(an, "t,px,py,pz,cx,cy,cz,vx,vy,xix,xiy,hull,mode,swing,phase,ncL,ncR,fnL,fnR,axL,ayL,azL,axR,ayR,azR,"
                  "pitch,roll,yaw,prate,tpitch,kL,kR,tx,ty,onL,onR,wL,wR\n");
  }
  // Signed distance of point q outside the convex hull of n points (+ outside, - inside, the margin); n < 3 -> 0.
  auto hull_out = [](const double (*pts)[2], int n, const double q[2]) {
    if (n == 0) return 9.0;
    if (n == 1) return std::hypot(q[0] - pts[0][0], q[1] - pts[0][1]);
    int idx[16];
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int i = 1; i < n; ++i)  // sort by x then y
      for (int j = i; j > 0 && (pts[idx[j]][0] < pts[idx[j - 1]][0] ||
                                (pts[idx[j]][0] == pts[idx[j - 1]][0] && pts[idx[j]][1] < pts[idx[j - 1]][1])); --j)
        std::swap(idx[j], idx[j - 1]);
    int h[32], k = 0;
    auto cross = [&](int o, int a2, int b2) {
      return (pts[a2][0] - pts[o][0]) * (pts[b2][1] - pts[o][1]) - (pts[a2][1] - pts[o][1]) * (pts[b2][0] - pts[o][0]);
    };
    for (int i = 0; i < n; ++i) {
      while (k >= 2 && cross(h[k - 2], h[k - 1], idx[i]) <= 0) --k;
      h[k++] = idx[i];
    }
    for (int i = n - 2, t2 = k + 1; i >= 0; --i) {
      while (k >= t2 && cross(h[k - 2], h[k - 1], idx[i]) <= 0) --k;
      h[k++] = idx[i];
    }
    --k;
    if (k < 3) {  // collinear: distance to the segment
      double best = 1e9;
      for (int i = 0; i + 1 < n; ++i) {
        const double ax = pts[idx[i]][0], ay = pts[idx[i]][1], bx = pts[idx[n - 1]][0], by = pts[idx[n - 1]][1];
        const double dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy + 1e-12;
        double u = ((q[0] - ax) * dx + (q[1] - ay) * dy) / l2;
        u = u < 0 ? 0 : (u > 1 ? 1 : u);
        best = std::fmin(best, std::hypot(q[0] - ax - u * dx, q[1] - ay - u * dy));
      }
      return best;
    }
    double mx = -1e9;  // counter-clockwise hull: outside distance = max over edges of the left-normal offset
    for (int i = 0; i < k; ++i) {
      const int a2 = h[i], b2 = h[(i + 1) % k];
      const double ex = pts[b2][0] - pts[a2][0], ey = pts[b2][1] - pts[a2][1], el = std::hypot(ex, ey) + 1e-12;
      const double dd = (ey * (q[0] - pts[a2][0]) - ex * (q[1] - pts[a2][1])) / el;  // right normal: outside for CCW
      mx = std::fmax(mx, dd);
    }
    return mx;
  };
  const int pre = (int)std::lround(t_push / World::kFrameDt);
  const int pre_rec = 30;  // 0.5 s recorded before the push (t < 0): rest level for the swing metric
  for (int f = 0; f < pre - pre_rec; ++f) w.StepFrame();
  char path[512];
  snprintf(path, sizeof path, "%s/%s@%.1f.csv", dir.c_str(), sc.name, t_push);
  FILE* fp = fopen(path, "w");
  if (fp) {
    fprintf(fp, "t");
    const char* pts[] = {"pelvis", "head", "l_ankle", "r_ankle", "l_toe", "r_toe", "l_heel", "r_heel"};
    for (auto n : pts) fprintf(fp, ",%s_x,%s_y,%s_z", n, n, n);
    fprintf(fp, ",l_roll,r_roll,l_contact,r_contact\n");
  }
  for (int f = 0; f < pre_rec; ++f) {
    w.StepFrame();
    tally(true);
    WriteRow(fp, m, d, b, (f + 1 - pre_rec) * World::kFrameDt);
  }
  // Rest levels just before the push.
  double rest_ank[2], rest_toe[2], p[3];
  for (int f = 0; f < 2; ++f) {
    rest_ank[f] = d->xanchor[3 * (b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)) + 2];
    FootPoint(m, d, b.geom_foot[f], 1, p);
    rest_toe[f] = p[2];
  }
  const double* pp = d->xpos + 3 * b.body0;
  const double pt[3] = {pp[0], pp[1], pp[2] + 0.35};
  const double imp[3] = {sc.dx * sc.J, sc.dy * sc.J, 0};
  if (sc.cmd_foot >= 0) {
    double dyc = sc.dy;
    if (sc.cross) {  // 0.1 m past the other ankle, measured across the pelvis heading
      const double* R = d->xmat + 9 * b.body0;
      const double hn = std::hypot(R[0], R[3]) + 1e-9, lx = -R[3] / hn, ly = R[0] / hn;
      const double* a0 = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L);
      const double* a1 = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_R);
      const double sep = std::fabs((a0[0] - a1[0]) * lx + (a0[1] - a1[1]) * ly);
      dyc = -(sep + sc.dy);
    }
    w.CommandStep(0, sc.cmd_foot, sc.dx, dyc, sc.cross != 0);
  } else if (g_stag_dv > 0) {
    const double* o = d->xpos + 3 * (b.body0 + B_CHEST);
    const double ps[3] = {o[0] - 0.1 * sc.dx, o[1] - 0.1 * sc.dy, o[2] + 0.06};
    const double J = g_stag_dv * b.dims.total_mass;
    const double is[3] = {sc.dx * J, sc.dy * J, 0};
    w.PushAt(0, is, ps, B_CHEST);  // standard chest push point
  } else if (g_chest) {
    // Hit side = the side the push comes from: a push toward +y (left) hits the right shoulder.
    const int bd = sc.pt == 2 ? B_HEAD : sc.pt == 1 ? (sc.dy > 0 ? B_UARM_R : B_UARM_L) : B_CHEST;
    w.ApplyImpulse(0, b.body0 + bd, (sc.pt == 1 ? d->xpos : d->xipos) + 3 * (b.body0 + bd), imp);
  } else {
    w.PushAt(0, imp, pt, B_LUMBAR);  // standard push point
  }
  const double pz0 = pp[2], tcmd = d->time;
  double com33[2] = {d->subtree_com[3 * b.body0], d->subtree_com[3 * b.body0 + 1]}, v33 = 0;
  const double yaw33 = std::atan2(d->xmat[9 * b.body0 + 3], d->xmat[9 * b.body0]);
  double td_prev = b.td_time;

  auto yaw_of = [&](int body) { const double* R = d->xmat + 9 * (b.body0 + body); return std::atan2(R[3], R[0]); };
  auto wrap = [](double a) { while (a > 3.14159265) a -= 6.2831853; while (a < -3.14159265) a += 6.2831853; return a; };
  const double yaw0 = yaw_of(B_PELVIS);
  bool alive = true;
  int air_run[2] = {0, 0}, in_swing[2] = {0, 0}, last_swing = -1;
  double swing_toe0[2][2] = {{0, 0}, {0, 0}}, cand_toe0[2][2] = {{0, 0}, {0, 0}};
  double toe[2][3];
  for (int fr = 0; fr < 600; ++fr) {
    w.StepFrame();
    const double t = (fr + 1) * World::kFrameDt;
    double ank[2][3], heel[2][3], roll[2];
    int con[2] = {0, 0};
    for (int f = 0; f < 2; ++f) {
      const double* a = d->xanchor + 3 * (b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L));
      for (int k = 0; k < 3; ++k) ank[f][k] = a[k];
      FootPoint(m, d, b.geom_foot[f], 1, toe[f]);
      FootPoint(m, d, b.geom_foot[f], -1, heel[f]);
      const double yz = d->geom_xmat[9 * b.geom_foot[f] + 7];  // foot y axis, world z component
      roll[f] = std::asin(yz < -1 ? -1 : (yz > 1 ? 1 : yz)) * 57.29578 * (f == 0 ? 1 : -1);
    }
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
      if (g < 0) continue;
      if (g == b.geom_foot[0]) con[0] = 1;
      if (g == b.geom_foot[1]) con[1] = 1;
      const int bd = m->geom_bodyid[g] - b.body0;
      if (bd >= B_PELVIS && bd <= B_HEAD) r.fell = 1;
    }
    if (pp[2] < 0.5) r.fell = 1;
    if (r.fell && r.fall_t < 0) r.fall_t = (fr + 1) * World::kFrameDt;
    if (r.fall_t < 0 && b.td_time != td_prev) {
      ++r.tds;
      r.e_lon.push_back(b.td_err[0]);
      r.e_lat.push_back(b.td_err[1]);
      r.e_over.push_back(b.td_over);
      r.e_yaw.push_back(b.td_yaw);
      r.e_len.push_back(b.td_len);
      if (sc.cmd_foot >= 0 && r.td1 < 0) {
        r.td1 = d->time - tcmd;
        const double* a = d->xanchor + 3 * (b.jnt0 + (sc.cmd_foot ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L));
        r.place_err = std::hypot(a[0] - b.cmd_tgt[0], a[1] - b.cmd_tgt[1]);
      }
    }
    if (r.fall_t < 0) {
      const double* Rpl = d->xmat + 9 * b.body0;
      r.ptilt = std::fmax(r.ptilt, std::acos(std::fmax(-1.0, std::fmin(1.0, Rpl[8]))) * 57.29578);
      double dyw = std::atan2(Rpl[3], Rpl[0]) - yaw33;
      while (dyw > 3.14159265) dyw -= 6.2831853;
      while (dyw < -3.14159265) dyw += 6.2831853;
      r.yaw_all = std::fmax(r.yaw_all, std::fabs(dyw) * 57.29578);
      const double* cmn = d->subtree_com + 3 * b.body0;
      const double vh = std::hypot(cmn[0] - com33[0], cmn[1] - com33[1]) / World::kFrameDt;
      com33[0] = cmn[0]; com33[1] = cmn[1];
      if (fr == 0) v33 = vh;
      if (r.td1 < 0 || d->time - tcmd < r.td1 + 0.3) r.dv_max = std::fmax(r.dv_max, vh - v33);
      const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
      r.tilt = std::fmax(r.tilt, std::acos(std::fmax(-1.0, std::fmin(1.0, Rc[8]))) * 57.29578);
      r.drop = std::fmax(r.drop, pz0 - pp[2]);
    }
    if (r.fall_t < 0 && b.mode == 1) r.sw_max = std::fmax(r.sw_max, d->time - b.step_t0);
    td_prev = b.td_time;
    if (r.step0 < 0 && b.mode == 1) r.step0 = (fr + 1) * World::kFrameDt;
    if (g_dbg) {
      const Percept& P = b.pc[b.cur];
      const double om = std::sqrt(9.81 / std::fmax(P.com[2], 0.3));
      fprintf(g_dbg, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d", t, P.com[0], P.com[1], P.com[2],
              P.com[0] + P.com_vel[0] / om, P.com[1] + P.com_vel[1] / om, b.step_tgt[0], b.step_tgt[1], b.mode,
              b.swing, b.phase);
      // Extra trace: CoM velocity, virtual force, contacts, pelvis tilt, ankle and hip-abd torques (L, R).
      fprintf(g_dbg, ",%.3f,%.3f,%.1f,%.1f,%d,%d,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f\n", P.com_vel[0],
              P.com_vel[1], b.vf[0], b.vf[1], b.foot_on[0], b.foot_on[1], P.pelvis_pitch, P.pelvis_roll,
              d->ctrl[b.act0 + J_ANKLE_DORSI_L], d->ctrl[b.act0 + J_ANKLE_DORSI_R], d->ctrl[b.act0 + J_ANKLE_INV_L],
              d->ctrl[b.act0 + J_ANKLE_INV_R], d->ctrl[b.act0 + J_HIP_ABD_L], d->ctrl[b.act0 + J_HIP_ABD_R],
              b.foot_fn[0], b.foot_fn[1]);
      fseek(g_dbg, -1, SEEK_CUR);
      fprintf(g_dbg, ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", d->qpos[b.qadr0 + J_KNEE_R], b.target[J_KNEE_R],
              d->qpos[b.qadr0 + J_HIP_ABD_R], b.target[J_HIP_ABD_R], d->qpos[b.qadr0 + J_HIP_FLEX_R],
              b.target[J_HIP_FLEX_R]);
      fseek(g_dbg, -1, SEEK_CUR);
      fprintf(g_dbg, ",%.4f,%d,%.3f,%.3f,%.3f,%.3f\n", b.cp_out, b.sat_cnt, b.dbg_raw[0], b.dbg_raw[1], b.dbg_raw[2],
              b.dbg_raw[3]);
      fseek(g_dbg, -1, SEEK_CUR);
      fprintf(g_dbg, ",%.3f,%.3f,%.3f,%.3f\n", d->xpos[3 * (b.body0 + B_FOOT_L)], d->xpos[3 * (b.body0 + B_FOOT_R)],
              d->xpos[3 * (b.body0 + B_FOOT_L) + 1], d->xpos[3 * (b.body0 + B_FOOT_R) + 1]);
      fseek(g_dbg, -1, SEEK_CUR);
      fprintf(g_dbg, ",%d,%.3f,%.3f\n", b.cross, b.step_yaw, b.pc[b.cur].pelvis_yaw);
      {  // left knee, ankle joint centres (x, y, z) L and R, pelvis height, true CoM
        const int jl = b.jnt0 + J_ANKLE_DORSI_L, jr = b.jnt0 + J_ANKLE_DORSI_R;
        const double* cm = d->subtree_com + 3 * b.body0;
        fseek(g_dbg, -1, SEEK_CUR);
        fprintf(g_dbg, ",%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", d->qpos[b.qadr0 + J_KNEE_L],
                d->xanchor[3 * jl], d->xanchor[3 * jl + 1], d->xanchor[3 * jl + 2], d->xanchor[3 * jr],
                d->xanchor[3 * jr + 1], d->xanchor[3 * jr + 2], d->xpos[3 * b.body0 + 2], cm[0], cm[1]);
      }
      {  // balance numbers, ankle parts, centre of pressure of the floor contacts of both feet
        double sx = 0, sy = 0, sf = 0;
        mjtNum cf[6];
        for (int i = 0; i < d->ncon; ++i) {
          const mjContact& cn = d->contact[i];
          const int g = cn.geom1 == 0 ? cn.geom2 : (cn.geom2 == 0 ? cn.geom1 : -1);
          if (g != b.geom_foot[0] && g != b.geom_foot[1]) continue;
          mj_contactForce(m, d, i, cf);
          sx += cn.pos[0] * cf[0];
          sy += cn.pos[1] * cf[0];
          sf += cf[0];
        }
        fseek(g_dbg, -1, SEEK_CUR);
        fprintf(g_dbg, ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", b.cor_dbg[0], b.cor_dbg[1], b.cor_dbg[2], b.cor_dbg[3],
                sf > 1 ? sx / sf : 0.0, sf > 1 ? sy / sf : 0.0);
        // left hip flexion q, target, velocity, target velocity, net torque L and R, swing foot distance
        fprintf(g_dbg, ",%.4f,%.4f,%.4f,%.4f,%.2f,%.2f", b.dbg_h[0], b.dbg_h[1], b.dbg_h[2], b.dbg_h[3], b.al_dbg[0],
                b.al_dbg[1]);
        fprintf(g_dbg, ",%.4f,%.4f,%.3f,%.3f,%.1f,%.1f,%.4f,%.3f,%.3f,%.3f,%.3f\n", d->qpos[b.qadr0 + J_HIP_FLEX_L], b.target[J_HIP_FLEX_L],
                d->qvel[b.dof0 + J_HIP_FLEX_L], b.qd_tgt[J_HIP_FLEX_L], b.net_dbg[0], b.net_dbg[1], b.dbg_raw[3], b.dbg_lres, d->qpos[b.qadr0 + J_HIP_ROT_R],
                b.target[J_HIP_ROT_R], std::atan2(b.pc[b.cur ^ 1].foot_R[1][3], b.pc[b.cur ^ 1].foot_R[1][0]));
      }
      {  // spine pitch (lumbar, chest): angle, target, actuator torque; trunk line pelvis -> chest (+ = forward)
        const double* pp = d->xpos + 3 * (b.body0 + B_PELVIS);
        const double* pc = d->xpos + 3 * (b.body0 + B_CHEST);
        fseek(g_dbg, -1, SEEK_CUR);
        fprintf(g_dbg, ",%.4f,%.4f,%.1f,%.4f,%.4f,%.1f,%.4f\n", d->qpos[b.qadr0 + J_LUMBAR_PITCH],
                b.target[J_LUMBAR_PITCH], d->qfrc_actuator[b.dof0 + J_LUMBAR_PITCH], d->qpos[b.qadr0 + J_CHEST_PITCH],
                b.target[J_CHEST_PITCH], d->qfrc_actuator[b.dof0 + J_CHEST_PITCH], std::atan2(pc[0] - pp[0], pc[2] - pp[2]));
      }
      fseek(g_dbg, -1, SEEK_CUR);  // swing frame target (ankle x, z), pelvis x
      fprintf(g_dbg, ",%.4f,%.4f,%.4f\n", b.sw_ft[0], b.sw_ft[2], d->xpos[3 * (b.body0 + B_PELVIS)]);
      fseek(g_dbg, -1, SEEK_CUR);  // knee net torques and their limits L, R
      fprintf(g_dbg, ",%.1f,%.1f,%.1f,%.1f\n", b.net_dbg[4], b.net_dbg[5], b.net_dbg[6], b.net_dbg[7]);
    }
    if (d->xpos[3 * b.body0 + 2] > 0.6) {  // upright enough: thigh direction in the world is meaningful
      for (int f = 0; f < 2; ++f) {
        if (b.foot_fn[f] > 0.1 * b.dims.total_mass * 9.81) continue;  // free leg only (foot carries < 10 % weight)
        const double* R = d->xmat + 9 * (b.body0 + (f ? B_THIGH_R : B_THIGH_L));
        const double a = std::acos(std::fmax(-1.0, std::fmin(1.0, R[8]))) * 57.29578;
        if (a > r.thw) { r.thw = a; r.thm = b.mode * 10 + (b.single == f ? 1 : 0); }
        r.abd = std::fmax(r.abd, std::fabs(d->qpos[b.qadr0 + (f ? J_HIP_ABD_R : J_HIP_ABD_L)]) * 57.29578);
      }
    }
    if (b.mode == 2) alive = false;
    if (alive && t <= 3.0) tally(false);
    if (an && t <= 4.0) {
      mj_subtreeVel(m, d);
      const double* cm2 = d->subtree_com + 3 * b.body0;
      const double* cv = d->subtree_linvel + 3 * b.body0;
      const double om = std::sqrt(9.81 / std::fmax(cm2[2], 0.3));
      const double xi[2] = {cm2[0] + cv[0] / om, cm2[1] + cv[1] / om};
      double pts[16][2];
      int n = 0, ncf[2] = {0, 0};
      for (int i = 0; i < d->ncon && n < 16; ++i) {
        const mjContact& c = d->contact[i];
        const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
        if (g < 0) continue;
        const int f = g == b.geom_foot[0] ? 0 : (g == b.geom_foot[1] ? 1 : -1);
        if (f < 0) continue;
        mjtNum fr6[6];
        mj_contactForce(m, d, i, fr6);
        if (std::fabs(fr6[0]) < 5.0) continue;
        pts[n][0] = c.pos[0]; pts[n][1] = c.pos[1]; ++n; ++ncf[f];
      }
      const Percept& Pn = b.pc[b.cur];
      const double* ak[2] = {d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L), d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_R)};
      fprintf(an, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d,%.0f,%.0f,", t, pp[0], pp[1], pp[2],
              cm2[0], cm2[1], cm2[2], cv[0], cv[1], xi[0], xi[1], hull_out(pts, n, xi), b.mode, b.swing, b.phase, ncf[0],
              ncf[1], b.foot_fn[0], b.foot_fn[1]);
      fprintf(an, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.2f,%.2f\n", ak[0][0],
              ak[0][1], ak[0][2], ak[1][0], ak[1][1], ak[1][2], Pn.pelvis_pitch, Pn.pelvis_roll, Pn.pelvis_yaw,
              Pn.pelvis_rate[0], Pn.trunk_pitch, d->qpos[b.qadr0 + J_KNEE_L], d->qpos[b.qadr0 + J_KNEE_R], b.step_tgt[0],
              b.step_tgt[1], b.foot_on[0], b.foot_on[1], b.w_load[0], b.w_load[1]);
    }
    if (alive && t <= 2.0) {
      r.yaw = std::fmax(r.yaw, std::fabs(wrap(yaw_of(B_PELVIS) - yaw0)) * 57.29578);
      r.chest = std::fmax(r.chest, std::fabs(wrap(yaw_of(B_CHEST) - yaw_of(B_PELVIS))) * 57.29578);
      r.flex = std::fmax(r.flex, std::fmax(d->qpos[b.qadr0 + J_HIP_FLEX_L], d->qpos[b.qadr0 + J_HIP_FLEX_R]) * 57.29578);
    }
    if (g_yaw) {
      // Yaw trace: headings [deg], ground moment about vertical through the CoM per foot [N m], hip rot/flex of swing legs.
      auto yawof = [&](int body) {
        const double* R = d->xmat + 9 * (b.body0 + body);
        return std::atan2(R[3], R[0]) * 57.29578;
      };
      double mz[2] = {0, 0}, fz[2] = {0, 0}, cz[2] = {0, 0}, fxy[2][2] = {{0, 0}, {0, 0}}, fm[2] = {0, 0}, ft[2] = {0, 0};
      const double* cm = d->subtree_com + 3 * b.body0;
      mj_subtreeVel(m, d);
      for (int i = 0; i < d->ncon; ++i) {
        const mjContact& c = d->contact[i];
        const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
        int f = g == b.geom_foot[0] ? 0 : (g == b.geom_foot[1] ? 1 : -1);
        if (f < 0) continue;
        double fr6[6];
        mj_contactForce(m, d, i, fr6);
        double fw[3], tw[3];
        mju_mulMatTVec(fw, c.frame, fr6, 3, 3);  // contact frame rows -> world
        mju_mulMatTVec(tw, c.frame, fr6 + 3, 3, 3);
        if (c.geom1 == g) { fw[0] = -fw[0]; fw[1] = -fw[1]; fw[2] = -fw[2]; tw[0] = -tw[0]; tw[1] = -tw[1]; tw[2] = -tw[2]; }
        const double rx = c.pos[0] - cm[0], ry = c.pos[1] - cm[1];
        mz[f] += rx * fw[1] - ry * fw[0] + tw[2];
        fz[f] += fw[2];
        {
          const double* fc = d->xpos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
          fm[f] += (c.pos[0] - fc[0]) * fw[1] - (c.pos[1] - fc[1]) * fw[0];  // friction moment about the foot origin
          ft[f] += std::hypot(fw[0], fw[1]);
        }
        fxy[f][0] += fw[0]; fxy[f][1] += fw[1];  // force on the foot
        cz[f] += fw[0] * fw[0] + fw[1] * fw[1];
      }
      const double* hv = d->subtree_angmom + 3 * b.body0;
      auto thigh_flex = [&](int f) { return d->qpos[b.qadr0 + (f ? J_HIP_FLEX_R : J_HIP_FLEX_L)] * 57.29578; };
      fprintf(g_yaw, "%.4f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%.0f,%d,%d,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", t,
              yawof(B_PELVIS), yawof(B_CHEST), yawof(B_FOOT_L), yawof(B_FOOT_R), thigh_flex(0), thigh_flex(1),
              d->qpos[b.qadr0 + J_HIP_ROT_L] * 57.29578, d->qpos[b.qadr0 + J_HIP_ROT_R] * 57.29578, mz[0], mz[1], fz[0], fz[1],
              b.mode, b.swing, hv[2], std::sqrt(cz[0]), d->ctrl[b.act0 + J_HIP_ROT_L], d->ctrl[b.act0 + J_HIP_ROT_R],
              d->ctrl[b.act0 + J_HIP_FLEX_L], d->ctrl[b.act0 + J_HIP_FLEX_R], d->ctrl[b.act0 + J_LUMBAR_YAW],
              b.torso_tau[2], std::sqrt(cz[1]), 0.0);
      fseek(g_yaw, -1, SEEK_CUR);
      fprintf(g_yaw, ",%.1f,%.1f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.3f,%.3f\n", fm[0], fm[1], ft[0], ft[1], fxy[0][0], fxy[0][1],
              fxy[1][0], fxy[1][1], d->xpos[3 * (b.body0 + B_FOOT_L) + 1] - cm[1], d->xpos[3 * (b.body0 + B_FOOT_R) + 1] - cm[1]);
      // foot spin: contact count, vertical angular velocity [deg/s], horizontal speed [m/s] per foot
      int nc[2] = {0, 0};
      for (int i = 0; i < d->ncon; ++i)
        for (int f = 0; f < 2; ++f)
          if (d->contact[i].geom1 == b.geom_foot[f] || d->contact[i].geom2 == b.geom_foot[f]) ++nc[f];
      double vel[2][6];
      for (int f = 0; f < 2; ++f) mj_objectVelocity(m, d, mjOBJ_BODY, b.body0 + (f ? B_FOOT_R : B_FOOT_L), vel[f], 0);
      fseek(g_yaw, -1, SEEK_CUR);
      fprintf(g_yaw, ",%d,%d,%.0f,%.0f,%.2f,%.2f", nc[0], nc[1], vel[0][2] * 57.29578, vel[1][2] * 57.29578,
              std::hypot(vel[0][3], vel[0][4]), std::hypot(vel[1][3], vel[1][4]));
      // foot tilt: pitch (toe up +) and roll (outer edge up +) of the sole [deg], ankle angle, target, torque per foot
      for (int f = 0; f < 2; ++f) {
        const double* R = d->geom_xmat + 9 * b.geom_foot[f];
        const int jd = f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L, ji = f ? J_ANKLE_INV_R : J_ANKLE_INV_L;
        fprintf(g_yaw, ",%.1f,%.1f,%.1f,%.1f,%.0f,%.1f,%.1f,%.0f", std::asin(R[6]) * 57.29578 * -1.0,
                std::asin(R[7]) * 57.29578 * (f ? -1.0 : 1.0), d->qpos[b.qadr0 + jd] * 57.29578, b.target[jd] * 57.29578,
                d->ctrl[b.act0 + jd], d->qpos[b.qadr0 + ji] * 57.29578, b.target[ji] * 57.29578, d->ctrl[b.act0 + ji]);
      }
      {  // twist about the vertical since the foot took >= 15 % of the weight (swing-twist of Rf R0^T) [deg]
        static double tw_R0[2][9];
        static int tw_on[2] = {0, 0};
        for (int f = 0; f < 2; ++f) {
          const double* Rf = d->xmat + 9 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
          double tw = 0;
          if (fz[f] < 0.15 * b.dims.total_mass * 9.81) {
            tw_on[f] = 0;
          } else if (!tw_on[f]) {
            tw_on[f] = 1;
            for (int q = 0; q < 9; ++q) tw_R0[f][q] = Rf[q];
          } else {
            double Rr[4];
            for (int r = 0; r < 2; ++r)
              for (int c = 0; c < 2; ++c)
                Rr[2 * r + c] = Rf[3 * r] * tw_R0[f][3 * c] + Rf[3 * r + 1] * tw_R0[f][3 * c + 1] +
                                Rf[3 * r + 2] * tw_R0[f][3 * c + 2];
            tw = std::atan2(Rr[2] - Rr[1], Rr[0] + Rr[3]) * 57.29578;
          }
          fprintf(g_yaw, ",%.2f", tw);
        }
      }
      {  // foot spin measure inputs - ankle -> toe heading [deg], ankle and toe heights [m] per foot
        double an[2][3], he[2][3], to[2][3];
        for (int f = 0; f < 2; ++f) FootSpin::Points(m, d, b, f, an[f], he[f], to[f]);
        for (int f = 0; f < 2; ++f) fprintf(g_yaw, ",%.2f", std::atan2(to[f][1] - an[f][1], to[f][0] - an[f][0]) * 57.29578);
        for (int f = 0; f < 2; ++f) fprintf(g_yaw, ",%.4f", an[f][2]);
        for (int f = 0; f < 2; ++f) fprintf(g_yaw, ",%.4f", to[f][2]);
      }
      fprintf(g_yaw, "\n");
    }
    if (g_ptrace && fr % 3 == 0 && fr < 150) {
      const Percept& P = b.pc[b.cur];
      printf("t=%.2f mode=%d ph=%d sw=%d st=%d rec=%d pz=%.2f com=%.2f,%.2f v=%.2f,%.2f c=%d%d tgt=%.2f,%.2f aL=%.2f,%.2f,%.3f aR=%.2f,%.2f,%.3f pp=%.2f\n",
             t, b.mode, b.phase, b.swing, b.steps, b.recover, pp[2], P.com[0], P.com[1], P.com_vel[0], P.com_vel[1],
             con[0], con[1], b.step_tgt[0], b.step_tgt[1], ank[0][0], ank[0][1], ank[0][2], ank[1][0], ank[1][1],
             ank[1][2], P.pelvis_pitch);
      const int jr = J_HIP_ABD_R, jl = J_HIP_ABD_L;
      printf("   roll=%.2f abdT L/R %.2f/%.2f abdQ %.2f/%.2f flexT L/R %.2f/%.2f flexQ %.2f/%.2f fn %.0f/%.0f ctrl abd %.0f/%.0f flex %.0f/%.0f\n",
             P.pelvis_roll, b.target[jl], b.target[jr], d->qpos[b.qadr0 + jl], d->qpos[b.qadr0 + jr],
             b.target[J_HIP_FLEX_L], b.target[J_HIP_FLEX_R], d->qpos[b.qadr0 + J_HIP_FLEX_L],
             d->qpos[b.qadr0 + J_HIP_FLEX_R], b.foot_fn[0], b.foot_fn[1], d->ctrl[b.act0 + jl], d->ctrl[b.act0 + jr],
             d->ctrl[b.act0 + J_HIP_FLEX_L], d->ctrl[b.act0 + J_HIP_FLEX_R]);
    }
    // Swing detection: toe and ankle above rest + 0.015 m for >= 3 frames.
    for (int f = 0; f < 2; ++f) {
      const bool air = toe[f][2] > rest_toe[f] + 0.015 && ank[f][2] > rest_ank[f] + 0.015;
      if (air) {
        if (air_run[f] == 0) { cand_toe0[f][0] = toe[f][0]; cand_toe0[f][1] = toe[f][1]; }
        if (++air_run[f] == 3 && !in_swing[f]) {
          in_swing[f] = 1;
          r.swings++;
          if (last_swing == f) r.same++;
          last_swing = f;
          if (t - 2 * World::kFrameDt > 1.2) r.late++;
          swing_toe0[f][0] = cand_toe0[f][0];
          swing_toe0[f][1] = cand_toe0[f][1];
        }
      } else {
        air_run[f] = 0;
        if (in_swing[f]) {
          in_swing[f] = 0;
          if (std::hypot(toe[f][0] - swing_toe0[f][0], toe[f][1] - swing_toe0[f][1]) < 0.25) r.shorts++;
        }
      }
    }
    if ((fr + 1) % 15 == 0 && t >= 2.0 - 1e-9) {  // every 0.25 s after 2 s
      for (int f = 0; f < 2; ++f)
        if (con[f]) { r.edge_n++; if (roll[f] > 8) r.edge_hit++; }
    }
    WriteRow(fp, m, d, b, t);
  }
  for (int f = 0; f < 2; ++f)
    if (in_swing[f] && std::hypot(toe[f][0] - swing_toe0[f][0], toe[f][1] - swing_toe0[f][1]) < 0.25) r.shorts++;
  if (fp) fclose(fp);
  if (an) fclose(an);
  return r;
}

}  // namespace

int RunPushOne(const char* scene, double moment) {
  g_ptrace = true;
  for (const Scene& sc : kScenes)
    if (std::string(sc.name) == scene) {
      mkdir(TmpPath("pk").c_str(), 0755);
      mkdir(TmpPath("pk/one").c_str(), 0755);
      g_dbg = fopen(TmpPath("pk/one/dbg.csv").c_str(), "w");
      if (g_dbg) fprintf(g_dbg, "t,com_x,com_y,com_z,xi_x,xi_y,tgt_x,tgt_y,mode,swing,phase,vx,vy,vfx,vfy,onL,onR,pitch,roll,"
                            "dorL,dorR,invL,invR,abdL,abdR,fnL,fnR,qkR,TkR,qabR,TabR,qflR,TflR,out,sat,rawx,rawy,limx,rg,axL,axR,ayL,ayR,cross,yawT,yaw,qkL,anLx,anLy,anLz,anRx,anRy,anRz,pz,cx,cy,uf,ul,ankf,ankl,copx,copy,hdz,hmz,rlenR,dlen,alL,alR,qflL,TflL,wflL,dflL,nflL,nflR,dsw,lres,qrR,TrR,fyR,qlp,Tlp,flp,qcp,Tcp,fcp,trk,ftx,ftz,pelx,nkL,nkR,lkL,lkR\n");
      g_yaw = fopen(TmpPath("pk/one/yaw.csv").c_str(), "w");
      if (g_yaw) fprintf(g_yaw, "t,yawP,yawC,yawFL,yawFR,flexL,flexR,rotL,rotR,mzL,mzR,fzL,fzR,mode,swing,Lz,fhL,cRL,cRR,cFL,cFR,cLY,tauZ,fhR,z,fmL,fmR,ftL,ftR,fxL,fyL,fxR,fyR,dyL,dyR,ncL,ncR,wzL,wzR,vhL,vhR,ptL,rlL,qdL,tdL,cdL,qiL,tiL,ciL,ptR,rlR,qdR,tdR,cdR,qiR,tiR,ciR,twL,twR,atL,atR,azL,azR,tzL,tzR\n");
      const Result r = RunOne(sc, moment, TmpPath("pk/one").c_str());
      if (g_yaw) fclose(g_yaw);
      g_yaw = nullptr;
      if (g_dbg) fclose(g_dbg);
      g_dbg = nullptr;
      printf("fell=%d swings=%d same=%d short=%d late=%d edge=%d/%d\n", r.fell, r.swings, r.same, r.shorts, r.late,
             r.edge_hit, r.edge_n);
      return 0;
    }
  return 1;
}

int RunPushBench(const char* label, int threads);

namespace {
// Sweep: chest push 5..60 N s in 8 directions, pushed at 2.0 s, 10 s each.
const char* kSwDir[8] = {"fwd", "fl", "left", "bl", "back", "br", "right", "fr"};
const double kSwAng[8] = {0, 45, 90, 135, 180, 225, 270, 315};  // heading +x, +90 = left [deg]
constexpr int kSwN = 12;
Scene SweepScene(int di, int ji, char* name, size_t nl) {
  const double a = kSwAng[di] * 0.017453293, J = 5.0 * (ji + 1);
  snprintf(name, nl, "c%s%d", kSwDir[di], (int)J);
  return Scene{name, std::cos(a), std::sin(a), J};
}
// Oblique set (the editor camera looks at ~34 deg to the bot's side, the push goes away from it): 34 deg off the
// lateral axis toward forward / back, both sides; chest centre, shoulder joint (hit side), head; 30..80 N s.
const char* kObDir[4] = {"fl", "bl", "br", "fr"};
const double kObAng[4] = {56, 124, 236, 304};
const char kObPt[3] = {'c', 's', 'h'};
constexpr int kObN = 6;
Scene ObliqueScene(int i, char* name, size_t nl) {  // i = (dir * 3 + pt) * kObN + ji
  const int ji = i % kObN, pt = (i / kObN) % 3, di = i / (3 * kObN);
  const double a = kObAng[di] * 0.017453293, J = 30.0 + 10.0 * ji;
  snprintf(name, nl, "o%s_%c%d", kObDir[di], kObPt[pt], (int)J);
  Scene sc{name, std::cos(a), std::sin(a), J};
  sc.pt = pt;
  return sc;
}
}  // namespace

int RunObliqueBench(const char* label, int threads) {  // emil_bench obl [label] [threads]
  g_chest = true;
  std::string dir = std::string(TmpPath("pk").c_str());
  mkdir(dir.c_str(), 0755);
  dir += "/";
  dir += label;
  mkdir(dir.c_str(), 0755);
  const int n = 4 * 3 * kObN;
  std::vector<Result> res(n);
  std::vector<std::array<char, 32>> names(n);
  std::vector<Scene> sc(n);
  for (int i = 0; i < n; ++i) sc[i] = ObliqueScene(i, names[i].data(), 32);
  std::atomic<int> next(0);
  auto worker = [&]() {
    for (int i; (i = next++) < n;) res[i] = RunOne(sc[i], 2.0, dir);
  };
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k) th.emplace_back(worker);
  for (auto& t : th) t.join();
  printf("oblique '%s': 34 deg off the side, c chest / s shoulder / h head, at 2.0 s, 10 s; . no step | sN | xN fell\n", label);
  printf("%-8s", "N s");
  for (int ji = 0; ji < kObN; ++ji) printf("%4d", 30 + 10 * ji);
  printf("   stood\n");
  int tot = 0;
  for (int r0 = 0; r0 < 12; ++r0) {
    printf("%-5s %c  ", kObDir[r0 / 3], kObPt[r0 % 3]);
    int st = 0;
    for (int ji = 0; ji < kObN; ++ji) {
      const Result& r = res[r0 * kObN + ji];
      char c[8];
      if (!r.fell && r.step0 < 0) snprintf(c, sizeof c, ".");
      else snprintf(c, sizeof c, "%c%d", r.fell ? 'x' : 's', r.tds);
      printf("%4s", c);
      st += !r.fell;
    }
    tot += st;
    printf("   %d/%d\n", st, kObN);
  }
  printf("stood %d/%d\n", tot, n);
  return 0;
}

// Commanded steps from calm stance (no push): 8 kinds x 2 feet, command at 2.0 s, 10 s.
struct StepKind { const char* nm; double dx, dy; int cross; };
const StepKind kStepKinds[8] = {{"f15", 0.15, 0, 0}, {"f25", 0.25, 0, 0}, {"f35", 0.35, 0, 0}, {"b15", -0.15, 0, 0},
                                {"b25", -0.25, 0, 0}, {"s15", 0, 0.15, 0}, {"s25", 0, 0.25, 0}, {"x10", 0.15, 0.10, 1}};
  // x10: 0.15 m ahead: 0.30 m would put the target 0.42 m from the hip, the leg reaches ~0.29 m at stance height.
Scene StepScene(int i, char* name, size_t nl) {
  const StepKind& k = kStepKinds[i % 8];
  const int foot = i / 8;
  snprintf(name, nl, "%c_%s", foot ? 'R' : 'L', k.nm);
  Scene sc{name, k.dx, k.dy, 0};
  sc.cmd_foot = foot;
  sc.cross = k.cross;
  return sc;
}

// emil_bench cop <lean m> [x|y]: centre-of-pressure shift check. Calm stance; at 3.0 s the support middle is
// moved by <lean> along the heading (x) or to the left (y) for 2 s. Per frame -> $EMIL_TMP/cop.csv: CoM, centre of
// pressure from the floor contacts of both feet, balance numbers, ankle parts, and the CoP shift the ankle angle part asks
// for at the present stiffness (asked = -ank_f * (kp_dorsi_L + kp_dorsi_R) / Fn, + = forward) next to the measured shift
// of the CoP ahead of the CoM (got).
int RunCopTest(double lean, int axis) {
  World w;
  char err[512];
  if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) {
    fprintf(stderr, "build failed: %s\n", err);
    return 1;
  }
  ApplyOverrides(w);
  if (HasOverrides()) w.Reset();
  const mjModel* m = w.model();
  mjData* d = w.data();
  Bot& b = w.bot(0);
  FILE* fo = fopen(TmpPath("cop.csv").c_str(), "w");
  if (!fo) return 1;
  fprintf(fo, "t,com_f,com_l,v_f,v_l,cop_f,cop_l,fn,uf,ul,ank_f,ank_l,asked_f,got_f,got_l,tq_dL,tq_dR\n");
  double cprev[2] = {0, 0};
  for (int fr = 0; fr < 360; ++fr) {
    const double t = (fr + 1) * World::kFrameDt;
    b.in.lean[0] = b.in.lean[1] = 0;
    if (t >= 3.0 && t < 5.0) {
      const double* R = d->xmat + 9 * b.body0;
      const double hn = std::hypot(R[0], R[3]) + 1e-9, hx = R[0] / hn, hy = R[3] / hn;
      b.in.lean[0] = axis ? -hy * lean : hx * lean;
      b.in.lean[1] = axis ? hx * lean : hy * lean;
    }
    w.StepFrame();
    double sx = 0, sy = 0, sf = 0;
    mjtNum cf[6];
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& cn = d->contact[i];
      const int g = cn.geom1 == 0 ? cn.geom2 : (cn.geom2 == 0 ? cn.geom1 : -1);
      if (g != b.geom_foot[0] && g != b.geom_foot[1]) continue;
      mj_contactForce(m, d, i, cf);
      sx += cn.pos[0] * cf[0];
      sy += cn.pos[1] * cf[0];
      sf += cf[0];
    }
    const double* R = d->xmat + 9 * b.body0;
    const double hn = std::hypot(R[0], R[3]) + 1e-9, hx = R[0] / hn, hy = R[3] / hn, lx = -hy, ly = hx;
    const double* cm = d->subtree_com + 3 * b.body0;
    const double vx = fr ? (cm[0] - cprev[0]) / World::kFrameDt : 0, vy = fr ? (cm[1] - cprev[1]) / World::kFrameDt : 0;
    cprev[0] = cm[0];
    cprev[1] = cm[1];
    const double px = sf > 1 ? sx / sf : cm[0], py = sf > 1 ? sy / sf : cm[1];
    const double kd = b.kp[J_ANKLE_DORSI_L] + b.kp[J_ANKLE_DORSI_R];
    const double asked = sf > 1 ? -b.cor_dbg[2] * kd / sf : 0;
    // Net ankle torque applied by L0 = ctrl - passive damping * qd (ctrl compensates the joint damping).
    const int dl = b.dof0 + J_ANKLE_DORSI_L, dr = b.dof0 + J_ANKLE_DORSI_R;
    const double tl = d->ctrl[b.act0 + J_ANKLE_DORSI_L] - m->dof_damping[dl] * d->qvel[dl];
    const double tr = d->ctrl[b.act0 + J_ANKLE_DORSI_R] - m->dof_damping[dr] * d->qvel[dr];
    fprintf(fo, "%.4f,%.4f,%.4f,%.3f,%.3f,%.4f,%.4f,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f\n", t,
            cm[0] * hx + cm[1] * hy, cm[0] * lx + cm[1] * ly, vx * hx + vy * hy, vx * lx + vy * ly, px * hx + py * hy,
            px * lx + py * ly, sf, b.cor_dbg[0], b.cor_dbg[1], b.cor_dbg[2], b.cor_dbg[3], asked,
            (px - cm[0]) * hx + (py - cm[1]) * hy, (px - cm[0]) * lx + (py - cm[1]) * ly, tl, tr);
  }
  fclose(fo);
  printf("cop: lean %.3f m axis %s -> %s/cop.csv\n", lean, axis ? "y" : "x", TmpDir().c_str());
  return 0;
}

int RunStepsOne(const char* scene) {  // emil_bench steps1 L_f15: traced commanded step
  char nm[32];
  for (int i = 0; i < 16; ++i) {
    const Scene sc = StepScene(i, nm, sizeof nm);
    if (std::string(nm) != scene) continue;
    g_ptrace = true;
    mkdir(TmpPath("pk").c_str(), 0755);
    mkdir(TmpPath("pk/one").c_str(), 0755);
    g_dbg = fopen(TmpPath("pk/one/dbg.csv").c_str(), "w");
    if (g_dbg) fprintf(g_dbg, "t,com_x,com_y,com_z,xi_x,xi_y,tgt_x,tgt_y,mode,swing,phase,vx,vy,vfx,vfy,onL,onR,pitch,roll,"
                          "dorL,dorR,invL,invR,abdL,abdR,fnL,fnR,qkR,TkR,qabR,TabR,qflR,TflR,out,sat,rawx,rawy,limx,rg,axL,axR,ayL,ayR,cross,yawT,yaw,qkL,anLx,anLy,anLz,anRx,anRy,anRz,pz,cx,cy,uf,ul,ankf,ankl,copx,copy,hdz,hmz,rlenR,dlen,alL,alR,qflL,TflL,wflL,dflL,nflL,nflR,dsw,lres,qrR,TrR,fyR,qlp,Tlp,flp,qcp,Tcp,fcp,trk,ftx,ftz,pelx,nkL,nkR,lkL,lkR\n");
    const Result r = RunOne(sc, 2.0, TmpPath("pk/one").c_str());
    if (g_dbg) fclose(g_dbg);
    g_dbg = nullptr;
    printf("%s fell=%d fall_t=%.2f tds=%d first_td=%.2f err=%.1fcm yaw=%.0f tilt=%.0f drop=%.1fcm\n", nm, r.fell, r.fall_t,
           r.tds, r.td1, r.place_err * 100, r.yaw, r.tilt, r.drop * 100);
    return 0;
  }
  return 1;
}

int RunStepsBench(const char* label, int threads) {
  std::string dir = std::string(TmpPath("pk").c_str());
  mkdir(dir.c_str(), 0755);
  dir += "/";
  dir += label;
  mkdir(dir.c_str(), 0755);
  const int n = 16;
  std::vector<Result> res(n);
  std::vector<std::array<char, 32>> names(n);
  std::vector<Scene> sc(n);
  for (int i = 0; i < n; ++i) sc[i] = StepScene(i, names[i].data(), 32);
  std::atomic<int> next(0);
  auto worker = [&]() {
    for (int i; (i = next++) < n;) res[i] = RunOne(sc[i], 2.0, dir);
  };
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k) th.emplace_back(worker);
  for (auto& t : th) t.join();
  printf("steps '%s': commanded step from calm stance at 2.0 s, 10 s (err = first touchdown vs target)\n", label);
  printf("%-7s %5s %6s %4s %7s %6s %5s %5s %6s %6s %5s\n", "step", "stood", "fall_t", "tds", "td1[s]", "err", "yaw", "tilt",
         "drop", "dv", "ptilt");
  int st = 0, okc = 0;
  double dvm = 0, ptm = 0;
  for (int i = 0; i < n; ++i) {
    const Result& r = res[i];
    st += !r.fell;
    printf("%-7s %5s %6.2f %4d %7.2f %5.1fcm %5.0f %5.0f %5.1fcm %6.2f %5.1f\n", sc[i].name, r.fell ? "no" : "YES", r.fall_t,
           r.tds, r.td1, r.place_err * 100, r.yaw, r.tilt, r.drop * 100, r.dv_max, r.ptilt);
    okc += !r.fell && r.dv_max <= 0.15 && r.ptilt <= 10 && r.place_err >= 0 && r.place_err <= 0.05;
    dvm = std::fmax(dvm, r.dv_max); ptm = std::fmax(ptm, r.ptilt);
  }
  printf("step gate (stood, dv <= 0.15 m/s, pelvis tilt <= 10 deg, err <= 5 cm): %d/%d; max dv %.2f m/s, max tilt %.1f deg\n",
         okc, n, dvm, ptm);
  printf("stood %d/%d\n", st, n);
  {
    std::vector<double> el, ea;
    for (const Result& r : res)
      for (size_t i = 0; i < r.e_lon.size(); ++i) {
        el.push_back(std::fabs(r.e_lon[i]));
        ea.push_back(std::fabs(r.e_lat[i]));
      }
    auto med = [](std::vector<double> v) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    auto mx = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    printf("T33 place error (%d steps): along median %.3f worst %.3f, across median %.3f worst %.3f m "
           "[gate median <= 0.03, worst <= 0.06]\n", (int)el.size(), med(el), mx(el), med(ea), mx(ea));
    std::vector<double> ov, yw;
    int empty = 0;
    for (const Result& r : res)
      for (size_t i = 0; i < r.e_over.size(); ++i) {
        ov.push_back(r.e_over[i]);
        yw.push_back(r.e_yaw[i] * 57.29578);
        empty += r.e_len[i] < 0.08;
      }
    printf("T34 over median %.3f worst %.3f m | empty steps < 8 cm %d in %d scenes | yaw range median %.0f worst %.0f deg\n",
           med(ov), mx(ov), empty, (int)res.size(), med(yw), mx(yw));
  }
  return 0;
}

// weight shift onto one foot at 2.0 s (hold 4 s), and shift + lift the other foot
// 5 cm for 3 s + put it down (run 8 s). Measures from the command.
struct ShiftRes { int fell = 0; double t_over = -1, ld_max = 0, drag = 0, lift_min = 1, drop = 0, tilt = 0, slide = 0; };
ShiftRes RunShift(int foot, double lift_s, bool trace) {
  ShiftRes r;
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(g_build_mu);
    if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) { r.fell = 1; return r; }
  }
  ApplyOverrides(w);
  mjData* d = w.data();
  Bot& b = w.bot(0);
  const int pre = (int)std::lround(2.0 / World::kFrameDt);
  for (int f = 0; f < pre; ++f) w.StepFrame();
  const int o = foot ^ 1;
  const int ja_o = b.jnt0 + (o ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L), ja_s = b.jnt0 + (foot ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L);
  const double ao0[2] = {d->xanchor[3 * ja_o], d->xanchor[3 * ja_o + 1]};
  const double as0[2] = {d->xanchor[3 * ja_s], d->xanchor[3 * ja_s + 1]};
  const double az0 = d->xanchor[3 * ja_o + 2];
  const double pz0 = d->xpos[3 * b.body0 + 2];
  w.CommandShift(0, foot, lift_s);
  const int n = (int)std::lround((lift_s > 0 ? 8.0 : 4.0) / World::kFrameDt);
  for (int fr = 0; fr < n; ++fr) {
    w.StepFrame();
    const double t = (fr + 1) * World::kFrameDt;
    if (b.mode == 2 || d->xpos[3 * b.body0 + 2] < 0.6) { r.fell = 1; break; }
    const double* fc = d->geom_xpos + 3 * b.geom_foot[foot];
    const double cy = d->subtree_com[3 * b.body0 + 1];
    if (r.t_over < 0 && std::fabs(cy - fc[1]) < 0.02) r.t_over = t;
    const bool lifted = b.sh_state >= 1 && b.sh_foot >= 0;
    if (r.t_over >= 0 && t > r.t_over + 0.2 && !lifted && (lift_s <= 0 || b.sh_foot >= 0) && b.sh_state == 0)
      r.ld_max = std::fmax(r.ld_max, b.ld[o]);
    if (b.sh_state == 0 && b.sh_foot >= 0)
      r.drag = std::fmax(r.drag, std::hypot(d->xanchor[3 * ja_o] - ao0[0], d->xanchor[3 * ja_o + 1] - ao0[1]));
    if (b.sh_state == 1 && t > 0.0) {
      const double* R = d->geom_xmat + 9 * b.geom_foot[foot];
      r.tilt = std::fmax(r.tilt, std::acos(std::fmin(1.0, R[8])) * 57.2958);
      if (b.sh_up_t0 >= 0 && d->time - b.sh_up_t0 > 0.4)
        r.lift_min = std::fmin(r.lift_min, d->xanchor[3 * ja_o + 2] - az0);
    }
    r.drop = std::fmax(r.drop, pz0 - d->xpos[3 * b.body0 + 2]);
    r.slide = std::fmax(r.slide, std::hypot(d->xanchor[3 * ja_s] - as0[0], d->xanchor[3 * ja_s + 1] - as0[1]));
    if (trace && fr % 6 == 0)
      printf("t=%.2f st=%d com_y=%.3f foot_y=%.3f ldS=%.2f ldO=%.2f azO=%.3f pz=%.3f vf=%.0f sg=%d inv=%.0f "
             "kn=%.2f roll=%.3f pit=%.3f hf=%.2f ad=%.2f dx=%.3f vfx=%.0f tdo=%.0f thf=%.0f Tad=%.2f dr=%.3f md=%d fx=%.3f\n", t, b.sh_state, cy, fc[1], b.ld[foot], b.ld[o], d->xanchor[3 * ja_o + 2],
             d->xpos[3 * b.body0 + 2], b.vf[1], b.single,
             d->ctrl[b.act0 + (foot ? J_ANKLE_INV_R : J_ANKLE_INV_L)],
             d->qpos[w.model()->jnt_qposadr[b.jnt0 + (foot ? J_KNEE_R : J_KNEE_L)]], d->xmat[9 * b.body0 + 7], d->xmat[9 * b.body0 + 6],
             d->qpos[w.model()->jnt_qposadr[b.jnt0 + (foot ? J_HIP_FLEX_R : J_HIP_FLEX_L)]],
             d->qpos[w.model()->jnt_qposadr[b.jnt0 + (foot ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)]],
             d->subtree_com[3 * b.body0] - fc[0], b.vf[0], d->ctrl[b.act0 + (foot ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)],
             d->ctrl[b.act0 + (foot ? J_HIP_FLEX_R : J_HIP_FLEX_L)], b.target[foot ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L],
             std::hypot(d->xanchor[3 * ja_o] - ao0[0], d->xanchor[3 * ja_o + 1] - ao0[1]), b.mode, fc[0]);
  }
  if (lift_s <= 0) r.lift_min = 0;
  return r;
}

int RunShiftBench(const char* one) {  // emil_bench shift [scene]: weight shift, then shift and lift
  const char* nm[4] = {"L_shift", "R_shift", "L_lift3", "R_lift3"};
  ShiftRes res[4];
  std::vector<std::thread> th;
  for (int i = 0; i < 4; ++i) {
    if (one && std::string(one) != nm[i]) continue;
    th.emplace_back([&, i]() { res[i] = RunShift(i & 1, i >= 2 ? 3.0 : 0.0, one != nullptr); });
  }
  for (auto& t : th) t.join();
  printf("shift: weight onto the foot at 2.0 s (t_over: CoM within 2 cm sideways of the foot centre; ldO max: other foot\n"
         "load share from t_over+0.2 s while not lifted; drag: other ankle moved; lift: lifted ankle min height after\n"
         "0.4 s; tilt: stance sole max tilt while lifted; drop: pelvis; slide: stance ankle)\n");
  printf("%-8s %5s %6s %5s %6s %6s %5s %6s %6s\n", "scene", "stood", "t_over", "ldO", "drag", "lift", "tilt", "drop", "slide");
  for (int i = 0; i < 4; ++i) {
    if (one && std::string(one) != nm[i]) continue;
    const ShiftRes& r = res[i];
    printf("%-8s %5s %6.2f %5.2f %4.1fcm %4.1fcm %5.1f %4.1fcm %4.1fcm\n", nm[i], r.fell ? "no" : "YES", r.t_over,
           r.ld_max, r.drag * 100, r.lift_min * 100, r.tilt, r.drop * 100, r.slide * 100);
  }
  return 0;
}

int RunSweepOne(const char* scene) {  // emil_bench sweep1 c<dir><J>: traced chest push like push1
  char nm[32];
  for (int di = 0; di < 8; ++di)
    for (int ji = 0; ji < kSwN; ++ji) {
      if (scene[0] == 'o' && di * kSwN + ji >= 4 * 3 * kObN) continue;
      const Scene sc = scene[0] == 'o' ? ObliqueScene(di * kSwN + ji, nm, sizeof nm) : SweepScene(di, ji, nm, sizeof nm);
      if (std::string(nm) != scene) continue;
      g_chest = true;
      g_ptrace = true;
      mkdir(TmpPath("pk").c_str(), 0755);
      mkdir(TmpPath("pk/one").c_str(), 0755);
      g_dbg = fopen(TmpPath("pk/one/dbg.csv").c_str(), "w");
      if (g_dbg) fprintf(g_dbg, "t,com_x,com_y,com_z,xi_x,xi_y,tgt_x,tgt_y,mode,swing,phase,vx,vy,vfx,vfy,onL,onR,pitch,roll,"
                            "dorL,dorR,invL,invR,abdL,abdR,fnL,fnR,qkR,TkR,qabR,TabR,qflR,TflR,out,sat,rawx,rawy,limx,rg,axL,axR,ayL,ayR,cross,yawT,yaw,qkL,anLx,anLy,anLz,anRx,anRy,anRz,pz,cx,cy,uf,ul,ankf,ankl,copx,copy,hdz,hmz,rlenR,dlen,alL,alR,qflL,TflL,wflL,dflL,nflL,nflR,dsw,lres,qrR,TrR,fyR,qlp,Tlp,flp,qcp,Tcp,fcp,trk,ftx,ftz,pelx,nkL,nkR,lkL,lkR\n");
      g_yaw = fopen(TmpPath("pk/one/yaw.csv").c_str(), "w");
      if (g_yaw) fprintf(g_yaw, "t,yawP,yawC,yawFL,yawFR,flexL,flexR,rotL,rotR,mzL,mzR,fzL,fzR,mode,swing,Lz,fhL,cRL,cRR,cFL,cFR,cLY,tauZ,fhR,z,fmL,fmR,ftL,ftR,fxL,fyL,fxR,fyR,dyL,dyR,ncL,ncR,wzL,wzR,vhL,vhR,ptL,rlL,qdL,tdL,cdL,qiL,tiL,ciL,ptR,rlR,qdR,tdR,cdR,qiR,tiR,ciR,twL,twR,atL,atR,azL,azR,tzL,tzR\n");
      const Result r = RunOne(sc, 2.0, TmpPath("pk/one").c_str());
      if (g_yaw) fclose(g_yaw);
      g_yaw = nullptr;
      if (g_dbg) fclose(g_dbg);
      g_dbg = nullptr;
      printf("%s fell=%d fall_t=%.2f touchdowns=%d first_step=%.2f longest_step=%.2f\n", nm, r.fell, r.fall_t, r.tds,
             r.step0, r.sw_max);
      return 0;
    }
  return 1;
}

int RunSweepBench(const char* label, int threads) {
  g_chest = true;
  std::string dir = std::string(TmpPath("pk").c_str());
  mkdir(dir.c_str(), 0755);
  dir += "/";
  dir += label;
  mkdir(dir.c_str(), 0755);
  const int n = 8 * kSwN;
  std::vector<Result> res(n);
  std::vector<std::array<char, 32>> names(n);
  std::vector<Scene> sc(n);
  for (int i = 0; i < n; ++i) sc[i] = SweepScene(i / kSwN, i % kSwN, names[i].data(), 32);
  std::atomic<int> next(0);
  auto worker = [&]() {
    for (int i; (i = next++) < n;) res[i] = RunOne(sc[i], 2.0, dir);
  };
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k) th.emplace_back(worker);
  for (auto& t : th) t.join();
  FILE* sf = fopen((dir + "/sweep.txt").c_str(), "w");
  for (int i = 0; i < n && sf; ++i)
    fprintf(sf, "%s fell=%d fall_t=%.2f tds=%d step0=%.2f sw_max=%.2f\n", sc[i].name, res[i].fell, res[i].fall_t,
            res[i].tds, res[i].step0, res[i].sw_max);
  if (sf) fclose(sf);
  // Cell: "." stood without a step, "sN" stood after N touchdowns, "xN" fell after N touchdowns.
  printf("sweep '%s': chest push at 2.0 s, 10 s; cell = . stood no step | sN stood, N touchdowns | xN fell (N tds)\n", label);
  printf("%-6s", "N s");
  for (int ji = 0; ji < kSwN; ++ji) printf("%4d", 5 * (ji + 1));
  printf("   stood  no-step-max\n");
  int tot = 0;
  for (int di = 0; di < 8; ++di) {
    printf("%-6s", kSwDir[di]);
    int st = 0, ns = 0;
    bool ns_ok = true;
    for (int ji = 0; ji < kSwN; ++ji) {
      const Result& r = res[di * kSwN + ji];
      char c[8];
      if (!r.fell && r.step0 < 0) snprintf(c, sizeof c, ".");
      else snprintf(c, sizeof c, "%c%d", r.fell ? 'x' : 's', r.tds);
      printf("%4s", c);
      st += !r.fell;
      if (ns_ok && !r.fell && r.step0 < 0) ns = 5 * (ji + 1); else ns_ok = false;
    }
    tot += st;
    printf("   %2d/%d   %d\n", st, kSwN, ns);
  }
  int ns = 0, n4 = 0;
  double sm = 0;
  for (int i = 0; i < n; ++i)
    if (res[i].step0 >= 0) { ++ns; sm += res[i].sw_max; n4 += res[i].sw_max > 0.4; }
  double ym = 0;
  for (int i = 0; i < n; ++i) ym += res[i].yaw;
  printf("runs with steps %d: longest step mean %.2f s, > 0.4 s in %d; pelvis yaw max mean %.0f deg\n", ns,
         ns ? sm / ns : 0.0, n4, ym * (1.0 / n));
  {
    int up = 0, nx = 0;
    for (int i = 0; i < n; ++i) up += !res[i].fell && res[i].yaw_all <= 45;
    for (int di = 0; di < 8; ++di)
      for (int ji = 0; ji < 5; ++ji) nx += res[di * kSwN + ji].fell;  // <= 25 N s
    printf("stood without turn (pelvis yaw <= 45 deg): %d/%d; falls at <= 25 N s: %d\n", up, n, nx);
  }
  printf("stood %d/%d\n", tot, n);
  return 0;
}

int RunFallBench(const char* label, int threads) {  // push bench plus per-frame analysis CSVs in <label>/an
  g_an_dir = std::string(TmpPath("pk/").c_str()) + label + "/an";
  mkdir(TmpPath("pk").c_str(), 0755);
  mkdir((std::string(TmpPath("pk/").c_str()) + label).c_str(), 0755);
  mkdir(g_an_dir.c_str(), 0755);
  return RunPushBench(label, threads);
}

int RunPushBench(const char* label, int threads) {
  std::string dir = std::string(TmpPath("pk").c_str());
  mkdir(dir.c_str(), 0755);
  dir += "/";
  dir += label;
  mkdir(dir.c_str(), 0755);
  std::vector<Result> res(120);
  std::atomic<int> next(0);
  auto worker = [&]() {
    for (int i; (i = next++) < 120;) res[i] = RunOne(kScenes[i / 6], kMoments[i % 6], dir);
  };
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k) th.emplace_back(worker);
  for (auto& t : th) t.join();

  std::string sumpath = dir + "/summary.txt";
  FILE* sf = fopen(sumpath.c_str(), "w");
  for (int i = 0; i < 120 && sf; ++i) {
    const Result& r = res[i];
    fprintf(sf, "%s@%.1f fell=%d swings=%d same=%d short=%d late=%d edge=%d/%d yaw=%.0f chest=%.0f flex=%.0f thw=%.0f abd=%.0f thm=%d\n",
            kScenes[i / 6].name, kMoments[i % 6], r.fell, r.swings, r.same, r.shorts, r.late, r.edge_hit, r.edge_n, r.yaw,
            r.chest, r.flex, r.thw, r.abd, r.thm);
  }
  if (sf) fclose(sf);
  printf("push bench '%s' (per scene averages; target: stood 120/120, swings ~3.5 (<=4), late <=0.7)\n", label);
  printf("%-6s %7s %6s %6s %6s %6s %6s | %5s %5s %5s | %s\n", "moment", "stood", "swings", "same", "short", "late", "edge%", "yaw", "chest", "flex", "ok(yaw<=30,chest<=15,flex<=60) | thw abd ok(thw<=45)");
  for (int mo = 0; mo <= 6; ++mo) {
    int n = 0, stood = 0, sw = 0, sa = 0, sh = 0, la = 0, en = 0, eh = 0, oy = 0, oc = 0, of = 0, ot = 0;
    double my = 0, mc = 0, mf = 0, mt = 0, ma = 0;
    for (int i = 0; i < 120; ++i) {
      if (mo < 6 && i % 6 != mo) continue;
      const Result& r = res[i];
      ++n; stood += !r.fell; sw += r.swings; sa += r.same; sh += r.shorts; la += r.late; en += r.edge_n; eh += r.edge_hit;
      my += r.yaw; mc += r.chest; mf += r.flex; oy += r.yaw <= 30; oc += r.chest <= 15; of += r.flex <= 60;
      mt += r.thw; ma += r.abd; ot += r.thw <= 45;
    }
    char nm[16];
    if (mo < 6) snprintf(nm, sizeof nm, "%.1f", kMoments[mo]); else snprintf(nm, sizeof nm, "all");
    printf("%-6s %3d/%-3d %6.2f %6.2f %6.2f %6.2f %6.1f | %5.0f %5.0f %5.0f | %d %d %d of %d | %4.0f %4.0f %d\n", nm,
           stood, n, (double)sw / n, (double)sa / n, (double)sh / n, (double)la / n, en ? 100.0 * eh / en : 0.0, my / n,
           mc / n, mf / n, oy, oc, of, n, mt / n, ma / n, ot);
  }
  {  // share of frames by stance-foot contact class
    double tot[3][4] = {};
    for (int i = 0; i < 120; ++i)
      for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 4; ++k) tot[c][k] += res[i].fc[c][k];
    const char* nm[3] = {"quiet stand", "single supp.", "double supp."};
    printf("stance foot contacts, %% of frames (flat>=3 / edge 2 / corner 1 / none):\n");
    for (int c = 0; c < 3; ++c) {
      const double n = tot[c][0] + tot[c][1] + tot[c][2] + tot[c][3] + 1e-9;
      printf("  %-13s %5.1f %5.1f %5.1f %5.1f   (%.0f frames)\n", nm[c], 100 * tot[c][0] / n, 100 * tot[c][1] / n,
             100 * tot[c][2] / n, 100 * tot[c][3] / n, n);
    }
  }
  return 0;
}
