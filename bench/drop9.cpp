// Drop bench: drops from height (editor tool "Height"): emil_bench drop9 [threads] [-p ...]
// Heights 0.9 1.0 1.1 | 1.8 2.0 2.2 | 3.6 4.0 4.4 | 5.6 6.0 6.4, 3 bot seeds each (D9_H=<h> one height,
// D9_S=<n> seeds).
// Per drop: feet touch time and CoM speed, deepest knee (mean of both) and lowest pelvis within 1 s, rebound (largest
// upward pelvis speed within 0.6 s after the first stop of the downward motion), lowest head within 2 s, standing at
// 2 s (pelvis above 0.7 m, balancer not given up), first hand and head floor touch and the head speed then.
// Group lines against the target bands.
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

struct D9Res {
  double t_land = -1, v_land = 0, knee_max = 0, pel_min = 9, t_hand = -1, t_head = -1, head_v = 0, pel_2 = 0;
  double rebound = 0, head_min = 9;
  double ank_min = 9, kn_over = -9, t_cross = 0;  // ankle gap min, knee gap minus ankle gap max, crossed [s]
  int mode_2 = 0, gave_up = 0;
  int first_bd = -1;                       // first body part (any) on the floor
  double tilt_pel = 0, tilt_chest = 0, first_t = -1, knee_first = 0;  // up-axis tilt [deg] and knees at that frame
};

// sideways gap (pelvis y axis, horizontal) between the left body bd and its right twin (bd + 3) [m]
double LatGap(const mjModel*, const mjData* d, const Bot& b, int bd) {
  const double* Rp = d->xmat + 9 * b.body0;
  const double ln = std::hypot(Rp[1], Rp[4]) + 1e-9;
  const double* l = d->xpos + 3 * (b.body0 + bd);
  const double* r = d->xpos + 3 * (b.body0 + bd + 3);
  return ((l[0] - r[0]) * Rp[1] + (l[1] - r[1]) * Rp[4]) / ln;
}

D9Res RunD9(double h, int seed) {
  D9Res r;
  std::vector<BodyParams> bodies(1);
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(stag::BuildMutex());
    if (!w.Build(bodies, 3.0, seed, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return r; }
  }
  ApplyOverrides(w);
  // D9_TILT=<deg> D9_DIR=<deg> D9_SPIN=<rad per s>: start tilted about the horizontal axis at that heading (edge fall)
  w.DropFrom(h, (getenv("D9_TILT") ? std::atof(getenv("D9_TILT")) : 0.0) * M_PI / 180,
             (getenv("D9_DIR") ? std::atof(getenv("D9_DIR")) : 0.0) * M_PI / 180,
             getenv("D9_SPIN") ? std::atof(getenv("D9_SPIN")) : 0.0);
  const mjModel* m = w.model();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  const double kRad = 57.29578;
  double hv_prev = 0, pz_prev = 0, t_stop = -1;
  const bool tr = getenv("D9_TR") && std::fabs(std::atof(getenv("D9_TR")) - h) < 0.01 &&
                  (!getenv("D9_TS") || std::atoi(getenv("D9_TS")) == seed);
  for (int fr = 0; fr < 60 * 6; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    const double* H = d->xpos + 3 * (b.body0 + B_HEAD);
    const double* q = d->qpos + b.qadr0;
    const double* hv = d->cvel + 6 * (b.body0 + B_HEAD);
    const double hs = std::sqrt(hv[3] * hv[3] + hv[4] * hv[4] + hv[5] * hv[5]);
    const double pvz = fr > 0 ? (P[2] - pz_prev) * 60.0 : 0.0;
    pz_prev = P[2];
    bool feet = false, hand = false, head = false;
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
      if (g < 0) continue;
      const int bd = m->geom_bodyid[g] - b.body0;
      if (r.first_bd < 0 && bd >= 0 && bd < 31 * 4) {  // the first body part to touch (pose at that frame)
        r.first_bd = bd;
        r.first_t = t;
        r.tilt_pel = std::acos(std::fmin(1.0, std::fmax(-1.0, d->xmat[9 * (b.body0 + B_PELVIS) + 8]))) * kRad;
        r.tilt_chest = std::acos(std::fmin(1.0, std::fmax(-1.0, d->xmat[9 * (b.body0 + B_CHEST) + 8]))) * kRad;
        r.knee_first = 0.5 * (std::fabs(q[J_KNEE_L]) + std::fabs(q[J_KNEE_R])) * kRad;
      }
      if (bd == B_FOOT_L || bd == B_FOOT_R) feet = true;
      if (bd >= B_UARM_L && bd <= B_HAND_R) hand = true;
      if (bd == B_HEAD) head = true;
    }
    if (r.t_land < 0 && feet) { r.t_land = t; r.v_land = std::fabs(b.pc[b.cur ^ 1].com_vel[2]); }
    if (r.t_land >= 0) {
      const double s = t - r.t_land;
      if (s <= 1.0) {
        r.knee_max = std::fmax(r.knee_max, 0.5 * (std::fabs(q[J_KNEE_L]) + std::fabs(q[J_KNEE_R])) * kRad);
        r.pel_min = std::fmin(r.pel_min, P[2]);
      }
      if (t_stop < 0 && pvz > -0.05) t_stop = t;
      if (t_stop >= 0 && t - t_stop < 0.6) r.rebound = std::fmax(r.rebound, pvz);
      if (s <= 2.0) r.head_min = std::fmin(r.head_min, H[2]);
      // Sideways (pelvis y axis, horizontal) gaps between the ankles and between the knees while
      // the body stands (pelvis above 0.4 m) in the first 2 s after the touch
      if (s <= 2.0 && P[2] > 0.4) {
        const double* Rp = d->xmat + 9 * b.body0;
        const double ly = Rp[1], lyy = Rp[4], ln = std::hypot(ly, lyy) + 1e-9;
        auto lat = [&](int bd) { const double* x = d->xpos + 3 * (b.body0 + bd); return (x[0] * ly + x[1] * lyy) / ln; };
        const double ag = lat(B_FOOT_L) - lat(B_FOOT_R), kg = lat(B_SHANK_L) - lat(B_SHANK_R);
        r.ank_min = std::fmin(r.ank_min, ag);
        r.kn_over = std::fmax(r.kn_over, kg - ag);
        if (ag < 0) r.t_cross += 1.0 / 60;
      }
      if (hand && r.t_hand < 0) r.t_hand = s;
      if (head && r.t_head < 0) { r.t_head = s; r.head_v = std::fmax(hs, hv_prev); }
      if (std::fabs(s - 2.0) < 0.5 / 60) { r.pel_2 = P[2]; r.mode_2 = b.mode; }
      if (b.mode == 2) r.gave_up = 1;
    }
    hv_prev = hs;
    // D9_LIE=<h>: pose after the touch every 0.5 s to 5 s: hips, knees, back (lumbar + chest pitch), neck [deg], pelvis z, CoM speed
    if (getenv("D9_LIE") && std::fabs(std::atof(getenv("D9_LIE")) - h) < 0.01 && r.t_land >= 0 && fr % 30 == 0 &&
        t - r.t_land < 5.0) {
      const double* cv = b.pc[b.cur ^ 1].com_vel;
      printf("LIE %.1f mode %d hip %4.0f %4.0f knee %4.0f %4.0f back %4.0f neck %4.0f sh %4.0f %4.0f pelvis %.2f v %.2f\n", t - r.t_land,
             b.mode, q[J_HIP_FLEX_L] * kRad, q[J_HIP_FLEX_R] * kRad, q[J_KNEE_L] * kRad, q[J_KNEE_R] * kRad,
             (q[J_LUMBAR_PITCH] + q[J_CHEST_PITCH]) * kRad, q[J_NECK_PITCH] * kRad, q[J_SH_FLEX_L] * kRad,
             q[J_SH_FLEX_R] * kRad, P[2], std::sqrt(cv[0] * cv[0] + cv[1] * cv[1] + cv[2] * cv[2]));
    }
    // D9_AIR=<h>: flight trace (arms: flex, abd, elbow L/R; legs: hip, knee L/R [deg]; chest tilt)
    if (getenv("D9_AIR") && std::fabs(std::atof(getenv("D9_AIR")) - h) < 0.01 && (r.t_land < 0) && b.airborne) {
      const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
      printf("AIR %.3f z %.2f vz %.1f cw %.2f sh %4.0f %4.0f abd %4.0f %4.0f el %4.0f %4.0f hip %4.0f %4.0f kn %4.0f %4.0f tilt %3.0f curl %d kp_sh %.0f kp_hip %.0f\n",
             t, P[2], d->qvel[b.free_dof + 2], b.air_cw, q[J_SH_FLEX_L] * kRad, q[J_SH_FLEX_R] * kRad, q[J_SH_ABD_L] * kRad,
             q[J_SH_ABD_R] * kRad, q[J_ELBOW_L] * kRad, q[J_ELBOW_R] * kRad, q[J_HIP_FLEX_L] * kRad, q[J_HIP_FLEX_R] * kRad,
             q[J_KNEE_L] * kRad, q[J_KNEE_R] * kRad, std::acos(std::fmin(1.0, std::fmax(-1.0, Rc[8]))) * kRad, b.air_curl,
             b.kp[J_SH_FLEX_L], b.kp[J_HIP_FLEX_L]);
    }
    // D9_RL=<h>: roll trace every 0.1 s after the touch: pelvis z, chest up-axis tilt and the pelvis forward axis z
    // (pitch: + = face down), lateral z (+ = left side down), head z and speed, bodies on the floor (mask letters)
    if (getenv("D9_RL") && std::fabs(std::atof(getenv("D9_RL")) - h) < 0.01 && r.t_land >= 0 && fr % 6 == 0 &&
        t - r.t_land < 3.0) {
      const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
      const double* Rpv = d->xmat + 9 * (b.body0 + B_PELVIS);
      char msk[40] = {0};
      int nm = 0;
      bool seen[B_COUNT] = {false};
      for (int i = 0; i < d->ncon && nm < 30; ++i) {
        const mjContact& c = d->contact[i];
        const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
        if (g < 0) continue;
        const int bd = m->geom_bodyid[g] - b.body0;
        if (bd < 0 || bd >= B_COUNT || seen[bd]) continue;
        seen[bd] = true;
        msk[nm++] = "PLCHuflUFTSOtsO"[bd < 15 ? bd : 14];
        msk[nm] = 0;
      }
      const double* cv = b.pc[b.cur ^ 1].com_vel;
      printf("RL %.1f m%d pel %.2f chestTilt %3.0f pelFwdZ %5.2f pelLatZ %5.2f head %.2f %.1f vh %.1f %.1f | %s\n", t - r.t_land, b.mode,
             P[2], std::acos(std::fmin(1.0, std::fmax(-1.0, Rc[8]))) * kRad, Rpv[6], Rpv[7], H[2], hs, cv[0], cv[1], msk);
    }
    // D9_TR=<h> (D9_TS=<seed>): trace every frame up to 1.5 s after the touch
    if (tr && (r.t_land < 0 ? t > 0.2 : t - r.t_land < 1.5)) {
      printf("DT %5.2f m%d air %d on %d%d | pel %.2f pvz %5.2f | knee %3.0f/%3.0f kp %5.0f hip %3.0f/%3.0f | ank %3.0f/%3.0f "
             "| head %.2f trunk %3.0f crouch %.2f hd %.2f hm %.2f zw %.2f land %d ikd %.3f\n",
             r.t_land < 0 ? -1.0 : t - r.t_land, b.mode, b.airborne, b.foot_on[0], b.foot_on[1], P[2], pvz,
             q[J_KNEE_L] * kRad, b.target[J_KNEE_L] * kRad, b.kp[J_KNEE_L], q[J_HIP_FLEX_L] * kRad,
             b.target[J_HIP_FLEX_L] * kRad, q[J_ANKLE_DORSI_L] * kRad, b.target[J_ANKLE_DORSI_L] * kRad, H[2],
             b.pc[b.cur ^ 1].trunk_pitch * kRad, b.in.crouch, b.dbg_h[0], b.dbg_h[1], b.land_z0, b.land_t >= 0, b.dbg_h[2]);
      // constraint (limit + contact) and muscle torque at the left knee and ankle [N m], dof damping torque
      const int dk = b.dof0 + J_KNEE_L, da = b.dof0 + J_ANKLE_DORSI_L;
      printf("DF %5.2f akp %5.0f knee con %6.0f mus %6.0f damp %6.0f qd %5.1f | ank con %6.0f mus %6.0f | ank range %.0f..%.0f\n",
             r.t_land < 0 ? -1.0 : t - r.t_land, b.kp[J_ANKLE_DORSI_L], d->qfrc_constraint[dk], d->actuator_force[b.act0 + J_KNEE_L],
             d->qfrc_passive[dk], d->qvel[dk], d->qfrc_constraint[da], d->actuator_force[b.act0 + J_ANKLE_DORSI_L],
             m->jnt_range[2 * (b.jnt0 + J_ANKLE_DORSI_L)] * kRad, m->jnt_range[2 * (b.jnt0 + J_ANKLE_DORSI_L) + 1] * kRad);
      // joint-limit forces (all bot joints that hit a limit this step) and the floor normal force on the feet [N]
      double fn = 0;
      printf("DA %5.2f abd %3.0f/%3.0f %3.0f/%3.0f rot %3.0f/%3.0f %3.0f/%3.0f inv %3.0f/%3.0f flex %3.0f/%3.0f sw %d on %d%d gapA %.2f gapK %.2f aL %.3f aR %.3f kL %.3f kR %.3f\n", r.t_land < 0 ? -1.0 : t - r.t_land,
             q[J_HIP_ABD_L] * kRad, b.target[J_HIP_ABD_L] * kRad, q[J_HIP_ABD_R] * kRad, b.target[J_HIP_ABD_R] * kRad,
             q[J_HIP_ROT_L] * kRad, b.target[J_HIP_ROT_L] * kRad, q[J_HIP_ROT_R] * kRad, b.target[J_HIP_ROT_R] * kRad,
             q[J_ANKLE_INV_L] * kRad, b.target[J_ANKLE_INV_L] * kRad, q[J_HIP_FLEX_L] * kRad, b.target[J_HIP_FLEX_L] * kRad, b.swing, b.foot_on[0], b.foot_on[1], LatGap(m, d, b, B_FOOT_L), LatGap(m, d, b, B_SHANK_L), d->xpos[3 * (b.body0 + B_FOOT_L) + 1], d->xpos[3 * (b.body0 + B_FOOT_R) + 1], d->xpos[3 * (b.body0 + B_SHANK_L) + 1], d->xpos[3 * (b.body0 + B_SHANK_R) + 1]);
      printf("DL %5.2f", r.t_land < 0 ? -1.0 : t - r.t_land);
      for (int i = 0; i < d->nefc; ++i) {
        if (d->efc_type[i] == mjCNSTR_LIMIT_JOINT && std::fabs(d->efc_force[i]) > 5)
          printf(" j%d:%.0f", d->efc_id[i] - b.jnt0, d->efc_force[i]);
      }
      for (int i = 0; i < d->ncon; ++i) {
        const mjContact& c = d->contact[i];
        const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
        if (g < 0) continue;
        const int bd = m->geom_bodyid[g] - b.body0;
        if (bd != B_FOOT_L && bd != B_FOOT_R) continue;
        mjtNum f6[6];
        mj_contactForce(m, d, i, f6);
        fn += f6[0];
      }
      printf(" | feet fn %.0f\n", fn);
      // left leg chain: hip, knee, ankle origins (x forward along the facing at the drop, z up) and the toe-heel tilt
      const double* Ph = d->xpos + 3 * (b.body0 + B_THIGH_L);
      const double* Pk = d->xpos + 3 * (b.body0 + B_SHANK_L);
      const double* Pa = d->xpos + 3 * (b.body0 + B_FOOT_L);
      const double* Rf = d->xmat + 9 * (b.body0 + B_FOOT_L);
      printf("DG %5.2f hip %.3f %.3f knee %.3f %.3f ankle %.3f %.3f footpitch %4.0f com_vz %5.2f\n",
             r.t_land < 0 ? -1.0 : t - r.t_land, Ph[0], Ph[2], Pk[0], Pk[2], Pa[0], Pa[2],
             std::atan2(-Rf[6], Rf[8]) * kRad, b.pc[b.cur ^ 1].com_vel[2]);
    }
  }
  return r;
}

}  // namespace

int RunDrop9Bench(int argc, char** argv) {
  int threads = argc > 2 ? std::atoi(argv[2]) : 3;
  threads = std::max(1, std::min(threads, 3));
  std::vector<double> hs = {0.9, 1.0, 1.1, 1.8, 2.0, 2.2, 3.6, 4.0, 4.4, 5.6, 6.0, 6.4};
  if (getenv("D9_HL")) {  // list of heights, comma separated
    hs.clear();
    for (const char* p = getenv("D9_HL"); *p;) { hs.push_back(std::atof(p)); while (*p && *p != 0x2c) ++p; if (*p) ++p; }
  }
  if (getenv("D9_H")) hs = {std::atof(getenv("D9_H"))};
  const int ns = getenv("D9_S") ? std::max(1, std::atoi(getenv("D9_S"))) : 3;
  struct Job { double h; int seed; };
  std::vector<Job> jobs;
  for (double h : hs)
    for (int s = 0; s < ns; ++s) jobs.push_back({h, 21 + 7 * s});
  std::vector<D9Res> res(jobs.size());
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&]() { for (int i; (i = next++) < (int)jobs.size();) res[i] = RunD9(jobs[i].h, jobs[i].seed); });
  for (auto& x : th) x.join();
  for (size_t i = 0; i < jobs.size(); ++i) {
    const D9Res& r = res[i];
    printf("D9 %.1f m s%d: feet %.2f s at %.1f m/s; knees to %3.0f deg, pelvis to %.2f m, rebound %.2f m/s, head min "
           "%.2f m; at 2 s pelvis %.2f m mode %d (%s); hand %.2f s, head %.2f s at %.1f m/s; ankles min %.0f cm, knees "
           "over %.0f cm, crossed %.2f s\n", jobs[i].h, jobs[i].seed,
           r.t_land, r.v_land, r.knee_max, r.pel_min, r.rebound, r.head_min, r.pel_2, r.mode_2,
           r.pel_2 > 0.7 && r.mode_2 != 2 ? "stands" : "down", r.t_hand, r.t_head, r.head_v, 100 * r.ank_min, 100 * r.kn_over, r.t_cross);
    printf("   first %d (%s) at %.2f s: pelvis tilt %.0f chest tilt %.0f knees %.0f\n", r.first_bd,
           r.first_bd == B_FOOT_L || r.first_bd == B_FOOT_R || r.first_bd == B_SHANK_L || r.first_bd == B_SHANK_R ? "feet" : "other",
           r.first_t, r.tilt_pel, r.tilt_chest, r.knee_first);
  }
  return 0;
}
