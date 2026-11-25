// Upper-body bench: upper-body measures after pushes (target "Э" and EMIL "E" columns).
//   emil_bench upper6 [threads] [-p ...]
// Chest pushes at 1.5 s (heading frame: back, forward, left), one bot per world, 2.5 s after the push. Angles in deg
// from the pose at the push:
//   trunk  = line pelvis -> head base in the push plane, + along the push (min..max)
//   back   = largest bend of the line pelvis -> chest -> head base
//   neck   = largest head-against-chest angle within 0.1 s after the push; nv = largest head-chest angular speed
//   fwd    = shoulder flexion of both arms (+ forward), side = abduction (+ out), elbow = largest elbow flexion
//   rest   = last time an arm is more than 5 deg (flexion or abduction) off its push-time pose [s after the push]
//   elrev  = elbow direction changes (1 deg hysteresis), both elbows summed
#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

namespace {

constexpr double kRad = 57.29578;
struct U6Scene { const char* name; int dir; double dv; };  // dir 0 back, 1 forward, 2 left; dv [m/s]
const U6Scene kU6[10] = {{"back0.73", 0, 0.73}, {"back0.97", 0, 0.97}, {"back1.30", 0, 1.30}, {"back1.62", 0, 1.62},
                         {"fwd0.73", 1, 0.73},  {"fwd0.97", 1, 0.97},  {"fwd1.62", 1, 1.62},  {"side0.49", 2, 0.49},
                         {"side0.97", 2, 0.97}, {"side1.30", 2, 1.30}};

struct U6Res {
  bool stood = true;
  int steps = 0, elrev = 0, rollrev = 0;
  double nr = 0;  // largest head-against-chest angle over the whole run [deg]
  double np_lo = 0, np_hi = 0, npv = 0;  // neck pitch (nod) range [deg] and its largest speed [deg/s]
  double tr_lo = 0, tr_hi = 0, back = 0, neck = 0, nv = 0, fw_lo = 0, fw_hi = 0, sd_lo = 0, sd_hi = 0, elbow = 0;
  double rest = 0;
  // arm swing (shoulder flexion, mean of both arms, against the push moment; + forward): forward max (first
  // 1 s) and its time, back max after it and its time, values at 0.35, 0.8, 1.4 s, elbow at the forward max
  double a_fmax = -1e9, t_fmax = 0, a_bmax = 1e9, t_bmax = 0, a35 = 0, a80 = 0, a140 = 0, el_f = 0;
  // same for the upper arm in the world (shoulder -> elbow against the vertical, + toward the facing direction)
  double w_fmax = -1e9, wt_fmax = 0, w_bmax = 1e9, wt_bmax = 0, w0 = 0, w35 = 0, w80 = 0, w140 = 0;
};

double Ang(const double a[3], const double b[3]) {
  const double c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) /
                   (std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) * std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) + 1e-12);
  return std::acos(std::fmax(-1.0, std::fmin(1.0, c)));
}

U6Res RunU6(const U6Scene& sc) {
  U6Res r;
  std::vector<BodyParams> bodies(1);
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(stag::BuildMutex());
    if (!w.Build(bodies, 3.0, 21, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return r; }
  }
  ApplyOverrides(w);
  w.Reset();
  const mjModel* m = w.model();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  const double kMoment = 1.5, kTail = 2.5;
  double ux = 0, uy = 0, t0 = -1, tr0 = 0, rel0[9], q0[J_COUNT], rest_bend = 0;
  int td0 = 0, elsgn[2] = {0, 0}, rsgn = 0;
  double rext = 0, fhx = 1, fhy = 0;
  double elext[2] = {0, 0};
  const int frames = (int)std::lround((kMoment + kTail) * 60.0) + 2;
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* q = d->qpos + b.qadr0;
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    const double* C = d->xpos + 3 * (b.body0 + B_CHEST);
    const double* H = d->xpos + 3 * (b.body0 + B_HEAD);
    const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
    const double* Rh = d->xmat + 9 * (b.body0 + B_HEAD);
    double rel[9];  // chest^T head
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) rel[3 * i + j] = Rc[i] * Rh[j] + Rc[3 + i] * Rh[3 + j] + Rc[6 + i] * Rh[6 + j];
    const double up[3] = {H[0] - P[0], H[1] - P[1], H[2] - P[2]};
    const double lo[3] = {C[0] - P[0], C[1] - P[1], C[2] - P[2]}, hi[3] = {H[0] - C[0], H[1] - C[1], H[2] - C[2]};
    const double trunk = std::atan2(up[0] * ux + up[1] * uy, up[2]);
    const double bend = Ang(lo, hi);
    if (t0 < 0) {
      if (t < kMoment) continue;
      const double* R = d->xmat + 9 * b.body0;
      const double hn = std::hypot(R[0], R[3]) + 1e-9, hx = R[0] / hn, hy = R[3] / hn;
      const double dx[3] = {-hx, hx, -hy}, dy[3] = {-hy, hy, hx};
      ux = dx[sc.dir];
      fhx = hx; fhy = hy;
      uy = dy[sc.dir];
      t0 = t;
      tr0 = std::atan2(up[0] * ux + up[1] * uy, up[2]);
      rest_bend = bend * kRad;
      for (int k = 0; k < 9; ++k) rel0[k] = rel[k];
      for (int j = 0; j < J_COUNT; ++j) q0[j] = q[j];
      td0 = b.td_cnt;
      if (getenv("U6_DBG") && sc.dir == 0 && sc.dv < 0.8)
        for (int j : {0, 1, 3, 6, 9, 10, 12})
          fprintf(stderr, "j %d I %.3f kp %.1f zm %.2f js.kp %.1f gsup %.2f\n", j, d->M[m->M_rowadr[b.dof0 + j] + m->M_rownnz[b.dof0 + j] - 1], b.kp[j],
                  b.zm[j], b.js[j].kp, b.gsup[j]);
      const double J = sc.dv * b.dims.total_mass;
      const double pt[3] = {C[0] - 0.1 * ux, C[1] - 0.1 * uy, C[2] + 0.06};
      const double imp[3] = {ux * J, uy * J, 0};
      w.PushAt(0, imp, pt, B_CHEST);  // standard chest push point
      rext = std::atan2(up[0] * (sc.dir == 2 ? ux : -uy) + up[1] * (sc.dir == 2 ? uy : ux), up[2]);
      elext[0] = q[J_ELBOW_L];
      elext[1] = q[J_ELBOW_R];
      continue;
    }
    const double dt = t - t0;
    if (P[2] < 0.5) r.stood = false;
    if (!r.stood) continue;
    const double tr = (trunk - tr0) * kRad;
    r.tr_lo = std::fmin(r.tr_lo, tr);
    r.tr_hi = std::fmax(r.tr_hi, tr);
    r.back = std::fmax(r.back, bend * kRad - rest_bend);
    // head against the chest, relative to the push-time relative attitude
    {
      double tr2 = 0;  // trace of rel0^T rel
      for (int k = 0; k < 9; ++k) tr2 += rel0[k] * rel[k];
      const double na = std::acos(std::fmax(-1.0, std::fmin(1.0, 0.5 * (tr2 - 1.0)))) * kRad;
      if (dt <= 0.1 + 1e-9) r.neck = std::fmax(r.neck, na);
      r.nr = std::fmax(r.nr, na);
      const double np = (q[J_NECK_PITCH] - q0[J_NECK_PITCH]) * kRad;
      r.np_lo = std::fmin(r.np_lo, np);
      r.np_hi = std::fmax(r.np_hi, np);
      r.npv = std::fmax(r.npv, std::fabs(d->qvel[b.dof0 + J_NECK_PITCH]) * kRad);
    }
    {  // trunk sideways (across the push for back and forward pushes, along it for side pushes): reversals, 1 deg
      const double sx = sc.dir == 2 ? ux : -uy, sy2 = sc.dir == 2 ? uy : ux;
      const double side = std::atan2(up[0] * sx + up[1] * sy2, up[2]);
      if (rsgn == 0) {
        if (std::fabs(side - rext) > 1.0 / kRad) { rsgn = side > rext ? 1 : -1; rext = side; }
      } else if ((side - rext) * rsgn > 0) {
        rext = side;
      } else if ((rext - side) * rsgn > 1.0 / kRad) {
        ++r.rollrev; rsgn = -rsgn; rext = side;
      }
    }
    const double* wh = d->cvel + 6 * (b.body0 + B_HEAD);
    const double* wc = d->cvel + 6 * (b.body0 + B_CHEST);
    r.nv = std::fmax(r.nv, std::sqrt((wh[0] - wc[0]) * (wh[0] - wc[0]) + (wh[1] - wc[1]) * (wh[1] - wc[1]) +
                                     (wh[2] - wc[2]) * (wh[2] - wc[2])) * kRad);
    for (int s = 0; s < 2; ++s) {
      const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
      const double fw = (q[sh] - q0[sh]) * kRad, sd = (q[sh + 1] - q0[sh + 1]) * kRad;
      r.fw_lo = std::fmin(r.fw_lo, fw);
      r.fw_hi = std::fmax(r.fw_hi, fw);
      r.sd_lo = std::fmin(r.sd_lo, sd);
      r.sd_hi = std::fmax(r.sd_hi, sd);
      r.elbow = std::fmax(r.elbow, q[sh + 3] * kRad);
      if (std::fabs(fw) > 5 || std::fabs(sd) > 5) r.rest = dt;
      const double e = q[sh + 3];
      if (elsgn[s] == 0) {
        if (std::fabs(e - elext[s]) > 1.0 / kRad) { elsgn[s] = e > elext[s] ? 1 : -1; elext[s] = e; }
      } else if ((e - elext[s]) * elsgn[s] > 0) {
        elext[s] = e;
      } else if ((elext[s] - e) * elsgn[s] > 1.0 / kRad) {
        ++r.elrev;
        elsgn[s] = -elsgn[s];
        elext[s] = e;
      }
    }
    r.steps = b.td_cnt - td0;
    {
      const double a = 0.5 * (q[J_SH_FLEX_L] - q0[J_SH_FLEX_L] + q[J_SH_FLEX_R] - q0[J_SH_FLEX_R]) * kRad;
      if (dt <= 1.0 && a > r.a_fmax) { r.a_fmax = a; r.t_fmax = dt; r.a_bmax = 1e9;
        r.el_f = 0.5 * (q[J_ELBOW_L] + q[J_ELBOW_R]) * kRad; }
      if (dt > r.t_fmax && dt <= r.t_fmax + 1.2 && a < r.a_bmax) { r.a_bmax = a; r.t_bmax = dt; }
      if (std::fabs(dt - 0.35) < 0.5 / 60) r.a35 = a;
      if (std::fabs(dt - 0.8) < 0.5 / 60) r.a80 = a;
      if (std::fabs(dt - 1.4) < 0.5 / 60) r.a140 = a;
      double aw = 0;
      for (int s = 0; s < 2; ++s) {
        const double* S = d->xpos + 3 * (b.body0 + (s ? B_UARM_R : B_UARM_L));
        const double* E = d->xpos + 3 * (b.body0 + (s ? B_FARM_R : B_FARM_L));
        aw += 0.5 * std::atan2((E[0] - S[0]) * fhx + (E[1] - S[1]) * fhy, S[2] - E[2]) * kRad;
      }
      if (dt < 0.5 / 60) r.w0 = aw;
      static const char* tr_sc = getenv("U6_TRACE");
      if (tr_sc && !std::strcmp(tr_sc, sc.name) && fr % 3 == 0) {
        const int j = J_SH_FLEX_L, dof = b.dof0 + j;
        fprintf(stderr, "TR %.2f aw %+4.0f q %+5.1f T %+5.1f qd %+6.1f kp %6.1f damp %5.2f gsup %.2f bias %+6.2f "
                "act %+6.2f trunk %+4.0f el %4.0f\n", dt, aw, q[j] * kRad, b.target[j] * kRad, d->qvel[dof] * kRad,
                b.kp[j], m->dof_damping[dof], b.gsup[j], d->qfrc_bias[dof], d->qfrc_actuator[dof] + d->qfrc_applied[dof],
                tr, q[J_ELBOW_L] * kRad);
        fprintf(stderr, "TR2 %.2f hip %+5.1f yaw %+5.1f wr %.2f\n", dt, b.up_dbg[0] * kRad, b.up_dbg[1] * kRad, b.up_dbg[2]);
      }
      if (dt <= 1.0 && aw > r.w_fmax) { r.w_fmax = aw; r.wt_fmax = dt; r.w_bmax = 1e9; }
      if (dt > r.wt_fmax && dt <= r.wt_fmax + 1.2 && aw < r.w_bmax) { r.w_bmax = aw; r.wt_bmax = dt; }
      if (std::fabs(dt - 0.35) < 0.5 / 60) r.w35 = aw;
      if (std::fabs(dt - 0.8) < 0.5 / 60) r.w80 = aw;
      if (std::fabs(dt - 1.4) < 0.5 / 60) r.w140 = aw;
    }
  }
  return r;
}

}  // namespace

int RunUpper6Bench(int argc, char** argv) {
  int threads = argc > 2 ? std::atoi(argv[2]) : 3;
  threads = std::max(1, std::min(threads, 3));
  std::vector<U6Res> res(10);
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&]() { for (int i; (i = next++) < 10;) res[i] = RunU6(kU6[i]); });
  for (auto& x : th) x.join();
  printf("U6 scene     stood steps  trunk        back  neck  nv    fwd          side         elbow rest elrev nr rollrev\n");
  double s_back = 0, s_neck = 0, s_nv = 0, s_fw = 0, s_sd = 0, s_el = 0;
  for (int i = 0; i < 10; ++i) {
    const U6Res& r = res[i];
    printf("U6 %-9s %5d %5d %5.0f..%-+5.0f %5.0f %5.0f %5.0f %5.0f..%-+5.0f %5.0f..%-+5.0f %5.0f %4.2f %3d\n",
           kU6[i].name, (int)r.stood, r.steps, r.tr_lo, r.tr_hi, r.back, r.neck, r.nv, r.fw_lo, r.fw_hi, r.sd_lo,
           r.sd_hi, r.elbow, r.rest, r.elrev);
    printf("U6A %-9s arm fwd max %+4.0f at %.2f s (elbow %3.0f), back max %+4.0f at %.2f s, swing %.2f s; "
           "at 0.35/0.8/1.4 s %+4.0f %+4.0f %+4.0f\n", kU6[i].name, r.a_fmax, r.t_fmax, r.el_f, r.a_bmax, r.t_bmax,
           r.t_bmax - r.t_fmax, r.a35, r.a80, r.a140);
    printf("U6W %-9s arm in world: start %+4.0f, fwd max %+4.0f at %.2f s, back max %+4.0f at %.2f s, swing %.2f s; "
           "at 0.35/0.8/1.4 s %+4.0f %+4.0f %+4.0f\n", kU6[i].name, r.w0, r.w_fmax, r.wt_fmax, r.w_bmax, r.wt_bmax,
           r.wt_bmax - r.wt_fmax, r.w35, r.w80, r.w140);
    printf("U6+ %-9s neck range %.0f, nod %.0f..%+.0f (%.0f deg/s), trunk sideways reversals %d\n", kU6[i].name, r.nr,
           r.np_lo, r.np_hi, r.npv, r.rollrev);
    s_back += r.back; s_neck = std::fmax(s_neck, r.neck); s_nv = std::fmax(s_nv, r.nv);
    s_fw += r.fw_hi - r.fw_lo; s_sd = std::fmax(s_sd, r.sd_hi); s_el += r.elbow;
  }
  printf("U6 summary: back mean %.1f, neck jerk worst %.1f deg, neck speed worst %.0f deg/s, arm fwd range mean %.1f, "
         "side worst %.1f, elbow mean %.1f deg\n", s_back / 10, s_neck, s_nv, s_fw / 10, s_sd, s_el / 10);
  return 0;
}
