// Fall bench: body measures during falls, compared with target bands.
//   emil_bench fall9 [threads] [-p ...]
// Scenes from 1.5 s of standing: chest push back dv 2.57 m/s, push to the left dv 1.93 m/s, burst of 5 chest bullets
// every 0.12 s. Per time after the event: trunk = line pelvis -> head base from the vertical, + along the push [deg];
// pz = pelvis height [m]; hip = mean hip flexion (trunk to thigh) [deg]; knee = mean knee flexion [deg];
// foot = highest foot rise over its start [cm]; hz = head height [m]. Events: first hand and first head floor touch
// [s], head speed at the head touch [m/s], steps (touchdowns).
#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

namespace {

constexpr double kRad = 57.29578;
struct F9Scene { const char* name; int kind; double dv; int seed = 21; double moment = 1.5; };  // kind 0 back push, 1 left push, 2 burst, 3 forward, 4 right push
constexpr int kF9N = 4;
const F9Scene kF9[kF9N] = {{"back2.57", 0, 2.57}, {"left1.93", 1, 1.93}, {"burst5", 2, 0}, {"fwd2.57", 3, 2.57}};
constexpr int kNT = 25;
const double kT[kNT] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.3,
                         2.6, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0};
constexpr int kNC = 16;  // trunk pz hip knee foot hz steps mode lying seat

struct F9Res {
  double v[kNT][kNC] = {};
  double t_hand = -1, t_head = -1, head_v = 0, foot_max = 0, t_m2 = -1, t_pred = -1, t_sit = -1;
  double tr_max = 0, t_pel = -1, pz_after = 0;  // largest trunk angle, first pelvis floor touch, pelvis max after it
  // Catch measure: head speed and time at 20 cm going down; first hand below 10 cm: time, its arm from the trunk axis
  // [deg], CoM speed, shoulder height and vertical speed then
  double hv20 = -1, t_h20 = -1, t_h10 = -1, arm_h10 = -1, v_h10 = 0, sz_h10 = 0, svz_h10 = 0;
  double head_1 = -1, el_max = 0, arm_min = 180;  // head z 1 s after the pelvis touch, elbow max with a hand
  // down, smallest arm-trunk angle while the CoM is faster than 0.5 m/s after the fall judgement [deg]
  // lie_d: pelvis distance from its place at the push, at the end [m]; feet 1.5 s after the touch, at the end
  double lie_d = 0, foot_15 = -1, foot_end = 0, knee_end = 0;
  double foot_late = 0, head_late = 0;  // after 2 s: largest foot rise [m], head height [m]
};

F9Res RunF9(const F9Scene& sc0) {
  F9Scene sc = sc0;  // F9_DV overrides the first scene (back push) dv or impulse
  if (sc.kind == 0 && getenv("F9_DV")) sc.dv = std::atof(getenv("F9_DV"));
  F9Res r;
  std::vector<BodyParams> bodies(1);
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(stag::BuildMutex());
    if (!w.Build(bodies, 3.0, sc.seed, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return r; }
  }
  ApplyOverrides(w);
  w.Reset();
  const mjModel* m = w.model();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  const double kMoment = sc.moment, kTail = 6.05;
  double t0 = -1, ux = -1, uy = 0, fx = 1, fy = 0, foot0[2] = {0, 0}, hz_prev = 0;
  double pp0[2] = {0, 0};
  int td0 = 0, shots = 0, ti = 0;
  const int frames = (int)std::lround((kMoment + kTail) * 60.0) + 2;
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    const double* C = d->xpos + 3 * (b.body0 + B_CHEST);
    const double* H = d->xpos + 3 * (b.body0 + B_HEAD);
    if (t0 < 0) {
      if (t < kMoment) continue;
      const double* R = d->xmat + 9 * b.body0;
      const double hn = std::hypot(R[0], R[3]) + 1e-9;
      fx = R[0] / hn; fy = R[3] / hn;
      if (sc.kind == 0) { ux = -fx; uy = -fy; }
      if (sc.kind == 1) { ux = -fy; uy = fx; }
      if (sc.kind == 4) { ux = fy; uy = -fx; }
      if (sc.kind == 2) { ux = -fx; uy = -fy; }
      if (sc.kind == 3) { ux = fx; uy = fy; }
      t0 = t; td0 = b.td_cnt; pp0[0] = P[0]; pp0[1] = P[1];
      for (int f = 0; f < 2; ++f) foot0[f] = d->xpos[3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L)) + 2];
      if (sc.kind != 2) {
        const double J = sc.dv > 20 ? sc.dv : sc.dv * b.dims.total_mass;  // > 20 = impulse [N s]
        const double pt[3] = {C[0] - 0.1 * ux, C[1] - 0.1 * uy, C[2] + 0.06};
        const double imp[3] = {ux * J, uy * J, 0};
        w.PushAt(0, imp, pt, B_CHEST);  // standard chest push point
      }
    }
    const double ts = t - t0;
    if (sc.kind == 2)
      while (shots < 5 && ts >= shots * 0.12 - 1e-9) {
        const double* c = d->xipos + 3 * (b.body0 + B_CHEST);
        const double pt[3] = {c[0] + 0.1 * fx - 0.03 * fy, c[1] + 0.1 * fy + 0.03 * fx, 1.39 * b.body.height / 1.78};
        const double dv[3] = {-fx, -fy, 0};
        w.Shoot(0, B_CHEST, pt, dv);
        ++shots;
      }
    const double up[3] = {H[0] - P[0], H[1] - P[1], H[2] - P[2]};
    const double* q = d->qpos + b.qadr0;
    double fmax = 0;
    for (int f = 0; f < 2; ++f) fmax = std::fmax(fmax, d->xpos[3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L)) + 2] - foot0[f]);
    r.foot_max = std::fmax(r.foot_max, fmax);
    if (t0 >= 0) r.lie_d = std::hypot(P[0] - pp0[0], P[1] - pp0[1]);
    // Trace: F9_TR=<kind> prints arms every 3 frames (shoulder flex q/T, abd q/T, elbow q, hand z, head z)
    if (t0 >= 0 && getenv("F9_TR") && std::atoi(getenv("F9_TR")) == sc.kind && fr % 3 == 0 && t - t0 < 1.6) {
      const double* hl = d->xpos + 3 * (b.body0 + B_HAND_L);
      const double* hr = d->xpos + 3 * (b.body0 + B_HAND_R);
      printf("TR %.2f m%d lie %.2f pred %d | L fl %4.0f/%4.0f ab %4.0f/%4.0f el %4.0f kp %5.1f hz %.2f | R fl %4.0f/%4.0f ab %4.0f/%4.0f "
             "el %4.0f kp %5.1f hz %.2f | head %.2f pel %.2f v %.2f lat %.2f vl %.2f | knee %3.0f/%3.0f %3.0f/%3.0f kp %4.0f hip %3.0f/%3.0f\n", t - t0, b.mode, b.lie_t0 >= 0 ? b.lie_t0 - t0 : -1.0,
             b.fall_pred, q[J_SH_FLEX_L] * kRad, b.target[J_SH_FLEX_L] * kRad, q[J_SH_ABD_L] * kRad,
             b.target[J_SH_ABD_L] * kRad, q[J_ELBOW_L] * kRad, b.kp[J_SH_FLEX_L], hl[2], q[J_SH_FLEX_R] * kRad,
             b.target[J_SH_FLEX_R] * kRad, q[J_SH_ABD_R] * kRad, b.target[J_SH_ABD_R] * kRad, q[J_ELBOW_R] * kRad,
             b.kp[J_SH_FLEX_R], hr[2], H[2], P[2], std::hypot(b.pc[b.cur ^ 1].com_vel[0], b.pc[b.cur ^ 1].com_vel[1]),
             (d->xmat[9 * (b.body0 + B_CHEST) + 2] * -fy + d->xmat[9 * (b.body0 + B_CHEST) + 5] * fx),
             b.pc[b.cur ^ 1].com_vel[0] * -fy + b.pc[b.cur ^ 1].com_vel[1] * fx, q[J_KNEE_L] * kRad, b.target[J_KNEE_L] * kRad,
             q[J_KNEE_R] * kRad, b.target[J_KNEE_R] * kRad, b.kp[J_KNEE_L], q[J_HIP_FLEX_L] * kRad, b.target[J_HIP_FLEX_L] * kRad);
    }
    if (r.t_pel >= 0 && std::fabs(t - t0 - r.t_pel - 1.5) < 0.5 / 60) r.foot_15 = fmax;
    r.foot_end = fmax;
    if (t0 >= 0 && r.t_h20 < 0 && H[2] < 0.20) {
      const double* hv = d->cvel + 6 * (b.body0 + B_HEAD);
      r.hv20 = std::sqrt(hv[3] * hv[3] + hv[4] * hv[4] + hv[5] * hv[5]);
      r.t_h20 = t - t0;
    }
    for (int f = 0; f < 2 && t0 >= 0 && r.t_h10 < 0; ++f) {
      const double* hw = d->xpos + 3 * (b.body0 + (f ? B_HAND_R : B_HAND_L));
      if (hw[2] >= 0.10) continue;
      const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
      const double* S = d->xpos + 3 * (b.body0 + (f ? B_UARM_R : B_UARM_L));
      const double* E = d->xpos + 3 * (b.body0 + (f ? B_FARM_R : B_FARM_L));
      const double u[3] = {E[0] - S[0], E[1] - S[1], E[2] - S[2]};
      const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 1e-9;
      const double c = -(u[0] * Rc[2] + u[1] * Rc[5] + u[2] * Rc[8]) / un;
      r.t_h10 = t - t0;
      const double* cv = b.pc[b.cur ^ 1].com_vel;
      r.v_h10 = std::sqrt(cv[0] * cv[0] + cv[1] * cv[1] + cv[2] * cv[2]);
      r.sz_h10 = S[2];
      r.svz_h10 = d->cvel[6 * (b.body0 + (f ? B_UARM_R : B_UARM_L)) + 5];
      r.arm_h10 = std::acos(std::fmax(-1.0, std::fmin(1.0, c))) * kRad;
    }
    if (r.t_pel >= 0 && std::fabs(t - t0 - r.t_pel - 1.0) < 0.5 / 60) r.head_1 = H[2];
    if (r.t_hand >= 0) r.el_max = std::fmax(r.el_max, std::fmax(q[J_ELBOW_L], q[J_ELBOW_R]) * kRad);
    if (b.fall_pred && b.fall_set && t > b.fall_t0 + 0.25 && std::hypot(b.pc[b.cur ^ 1].com_vel[0], b.pc[b.cur ^ 1].com_vel[1]) > 0.5 &&
        b.lie_t0 < 0) {  // from 0.25 s after the reach start, before lying: is an arm along the trunk?
      const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
      for (int f = 0; f < 2; ++f) {
        const double* S = d->xpos + 3 * (b.body0 + (f ? B_UARM_R : B_UARM_L));
        const double* E = d->xpos + 3 * (b.body0 + (f ? B_FARM_R : B_FARM_L));
        const double u[3] = {E[0] - S[0], E[1] - S[1], E[2] - S[2]};
        const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 1e-9;
        const double c = -(u[0] * Rc[2] + u[1] * Rc[5] + u[2] * Rc[8]) / un;  // against the chest z axis
        r.arm_min = std::fmin(r.arm_min, std::acos(std::fmax(-1.0, std::fmin(1.0, c))) * kRad);
      }
    }
    r.knee_end = 0.5 * (std::fabs(q[J_KNEE_L]) + std::fabs(q[J_KNEE_R])) * kRad;
    if (t0 >= 0 && t - t0 > 2.0) { r.foot_late = std::fmax(r.foot_late, fmax); r.head_late = std::fmax(r.head_late, H[2]); }
    for (int i = 0; i < d->ncon; ++i) {  // floor touches (geom 0 = floor)
      const mjContact& c = d->contact[i];
      const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
      if (g < 0) continue;
      const int bd = m->geom_bodyid[g] - b.body0;
      if ((bd == B_HAND_L || bd == B_HAND_R || bd == B_FARM_L || bd == B_FARM_R) && r.t_hand < 0) r.t_hand = ts;
      if (bd == B_PELVIS && r.t_pel < 0) r.t_pel = ts;
      if (bd == B_HEAD && r.t_head < 0) { r.t_head = ts; r.head_v = std::fabs(H[2] - hz_prev) * 60.0; }
    }
    hz_prev = H[2];
    r.tr_max = std::fmax(r.tr_max, std::atan2(up[0] * ux + up[1] * uy, up[2]) * kRad);
    if (r.t_pel >= 0) r.pz_after = std::fmax(r.pz_after, P[2]);
    if (b.mode == 2 && r.t_m2 < 0) r.t_m2 = ts;
    if (b.fall_pred && r.t_pred < 0) r.t_pred = ts;
    if (b.sit_t0 >= 0 && r.t_sit < 0) r.t_sit = ts;
    if (ti < kNT && ts >= kT[ti] - 1e-9) {
      double* v = r.v[ti];
      v[0] = std::atan2(up[0] * ux + up[1] * uy, up[2]) * kRad;
      v[1] = P[2];
      v[2] = 0.5 * (q[J_HIP_FLEX_L] + q[J_HIP_FLEX_R]) * kRad;
      v[3] = 0.5 * (std::fabs(q[J_KNEE_L]) + std::fabs(q[J_KNEE_R])) * kRad;
      v[4] = 100 * fmax;
      v[5] = H[2];
      v[6] = b.td_cnt - td0;
      v[7] = b.mode;
      {  // pelvis tilt along the push (+ back) [deg]
        const double* Rp = d->xmat + 9 * (b.body0 + B_PELVIS);
        v[8] = std::atan2(Rp[2] * ux + Rp[5] * uy, Rp[8]) * kRad;
      }
      {  // pelvis roll across the push (+ left side down) [deg]
        const double* Rp = d->xmat + 9 * (b.body0 + B_PELVIS);
        v[9] = std::asin(std::fmax(-1.0, std::fmin(1.0, Rp[7]))) * kRad;  // pelvis left axis up (+ = right side down)
      }
      for (int f = 0; f < 2; ++f) {  // per leg: hip, knee, foot rise
        v[10 + 3 * f] = q[f ? J_HIP_FLEX_R : J_HIP_FLEX_L] * kRad;
        v[11 + 3 * f] = std::fabs(q[f ? J_KNEE_R : J_KNEE_L]) * kRad;
        v[12 + 3 * f] = 100 * (d->xpos[3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L)) + 2] - foot0[f]);
      }
      ++ti;
    }
  }
  return r;
}

}  // namespace

int RunFall9Bench(int argc, char** argv) {
  // emil_bench fall9 j<kind> [threads]: pushes 110..260 N s every 10 (kind 0 back, 1 left, 3 forward)
  if (argc > 2 && argv[2][0] == 'j') {
    const int kind = std::atoi(argv[2] + 1);
    int threads = argc > 3 ? std::max(1, std::min(std::atoi(argv[3]), 3)) : 3;
    std::vector<F9Scene> sc;
    // F9_J=<J>: only that push (with F9_TR for a trace); F9_SEEDS=<n>: n bot seeds per push (outcomes are chaotic)
    const int ns = getenv("F9_SEEDS") ? std::max(1, std::atoi(getenv("F9_SEEDS"))) : 1;
    // F9_MOMS=2 adds the push moment 2.3 s (stance phase differs); rows from 100 N s
    const int nm = getenv("F9_MOMS") ? std::max(1, std::min(2, std::atoi(getenv("F9_MOMS")))) : 1;
    for (int J = 100; J <= 260; J += 10)
      for (int mo = 0; mo < nm; ++mo)
        for (int s = 0; s < ns; ++s)
          if ((!getenv("F9_J") || std::atoi(getenv("F9_J")) == J) && (!getenv("F9_S") || std::atoi(getenv("F9_S")) == s) &&
              (!getenv("F9_M") || std::atoi(getenv("F9_M")) == mo))
            sc.push_back({"j", kind, (double)J, 21 + 7 * s, mo ? 2.3 : 1.5});
    std::vector<F9Res> res(sc.size());
    std::atomic<int> next{0};
    std::vector<std::thread> th;
    for (int k = 0; k < threads; ++k)
      th.emplace_back([&]() { for (int i; (i = next++) < (int)sc.size();) res[i] = RunF9(sc[i]); });
    for (auto& x : th) x.join();
    for (size_t i = 0; i < sc.size(); ++i)
      printf("F9J kind %d J %3.0f pelvis touch %.2f s, feet 1.5 s after it %3.0f cm, at the end %3.0f cm, knees %3.0f deg, "
             "lies %.2f m, seat %.2f s | hand %.2f head %.2f s at %.1f m/s, head 1 s after the touch %.2f m, elbow %3.0f, "
             "arm-trunk min %3.0f | H10 %.2f arm %3.0f H20 %.2f v %.1f | seed %d mom %.1f | at hand com v %.2f shoulder z %.2f vz %.2f\n", kind, sc[i].dv, res[i].t_pel, 100 * res[i].foot_15, 100 * res[i].foot_end,
             res[i].knee_end, res[i].lie_d, res[i].t_sit, res[i].t_hand, res[i].t_head, res[i].head_v, res[i].head_1,
             res[i].el_max, res[i].arm_min, res[i].t_h10, res[i].arm_h10, res[i].t_h20, res[i].hv20, sc[i].seed, sc[i].moment, res[i].v_h10, res[i].sz_h10, res[i].svz_h10);
    return 0;
  }
  int threads = argc > 2 ? std::atoi(argv[2]) : 3;
  threads = std::max(1, std::min(threads, 3));
  std::vector<F9Res> res(kF9N);
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&]() { for (int i; (i = next++) < kF9N;) res[i] = RunF9(kF9[i]); });
  for (auto& x : th) x.join();
  for (int i = 0; i < kF9N; ++i) {
    const F9Res& r = res[i];
    printf("F9 %s hands %.2f s, head %.2f s at %.1f m/s, feet up to %.0f cm, gave up %.2f s, fall judged %.2f s, seat %.2f s\nF9 %s trunk max %.0f deg, pelvis touch %.2f s, pelvis max after it %.2f m; after 2 s feet up to %.0f cm, head up to %.2f m\n"
           "F9 t   trunk   pz   hip  knee foot   hz st\n", kF9[i].name, r.t_hand, r.t_head, r.head_v, 100 * r.foot_max, r.t_m2,
           r.t_pred, r.t_sit, kF9[i].name, r.tr_max, r.t_pel, r.pz_after, 100 * r.foot_late,
           r.head_late);
    printf("F9 %s lies %.2f m from the push point\n", kF9[i].name, r.lie_d);
    for (int k = 0; k < kNT; ++k) {
      const double* v = r.v[k];
      printf("F9 %3.1f %6.0f %5.2f %5.0f %5.0f %4.0f %5.2f %2.0f  m%.0f tp%.0f roll%.0f | L %.0f %.0f %.0f R %.0f %.0f %.0f\n", kT[k], v[0], v[1], v[2], v[3], v[4],
             v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
    }
  }
  return 0;
}
