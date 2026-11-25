// Death bench: death and long relaxation against target rows: emil_bench death8 [-p ...]
// One standing bot; scenes: pistol bullet into the head from the front, from behind, from the left, from the right;
// the kill command; two rifle chest bullets 0.4 s apart (the second kills: damage 50 each); a pistol burst into the
// chest (5 x 25 every 0.15 s: the 5th wounds). Two moments each (1.6 s and 1.87 s of standing).
// Time zero = the start of the relaxation (b.dead_t). Rows per time: pelvis height [cm], pelvis shift [cm], trunk
// from vertical (pelvis -> head CoM, + forward of the facing) [deg], knees L R, hips L R [deg], arm from the trunk
// line (largest, upper arm against the chest -z axis) [deg] and its elbow [deg].
// Events: pelvis below 60 and 50 cm; at 60 cm: trunk, smaller knee, shift; head CoM below 20 cm: speed [m/s].
// Target, head bullet from the front: pelvis 90 87 78 66 52 38 25 13 21 cm at 0.1-0.8, 1.0 s; trunk 2 10 15
// 15 15 22 33 52 94; knees 20 38 65 95 120 140 140 118; arm 17 24 37 78 106 105 88 70; elbow 23 40 56 56 59 70 83 90.
// Targets: <= 85 cm at 0.3 s, <= 60 at 0.5 s, < 50 by 0.6 s; at 60 cm trunk <= 30 deg, knees >= 90, shift <= 20 cm;
// at 0.4 s an arm >= 45 deg with the elbow >= 30; head speed at 20 cm <= 2.5 m/s.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

namespace {

constexpr double kRad = 57.29578;
// Body bullets (pistol 15 damage, impulse bul_imp; the 5th wounds, the 7th kills): 5 into the chest 1.0 / 0.5 / 0.12 s
// apart, 5 into the belly 0.7 s apart, 5 into the back 0.5 s apart, 7 into the chest 0.5 s apart
enum { kHeadF, kHeadB, kHeadL, kHeadR, kKill, kRifle2, kBurst5, kC10, kC05, kC012, kB07, kCB05, kC7, kPF180, kPF200, kPL160, kPL180, kNS };
const char* kName[kNS] = {"head_f", "head_b", "head_l", "head_r", "kill", "rifle2", "burst5",
                          "c5_10", "c5_05", "c5_012", "b5_07", "cb5_05", "c7_05", "pf180", "pf200", "pl160", "pl180"};
constexpr int kNT = 12;
const double kT[kNT] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0};
constexpr int kNC = 12;
const char* kCol[kNC] = {"pelvis", "shift", "trunk", "kneeL", "kneeR", "hipL", "hipR", "arm", "elbow", "hgap", "hup", "hhead"};

struct DRes {
  double v[kNT][kNC] = {};
  int gu = 0;
  double lift = 0, fdist60 = 0, kmax60 = 0, foot2 = 0, sh50f = 0, sh50l = 0, ft50 = 0, t60 = -1, t50 = -1, tr60 = 0, kn60 = 0, sh60 = 0, t20 = -1, hv20 = 0, cv20 = 0, hpk40 = 0, gu_t = -1, gu_pz = 0, pel_end = 0, v_end = 0, knee_end = 0, foot_end = 0, t_hand = -1, t_headc = -1, el_hand = 0;
  // Measures 0-3 s after the wound: pelvis z per frame, ring of the last pelvis / chest / head z;
  // pelvis sink speed over 0.1 s, head below 40 cm and chest below 35 cm sink speed over 0.05 s, first pelvis < 25 cm
  // and chest < 35 cm, highest pelvis after it, largest |trunk| (all, 0.5-1.0 s), pelvis sink in its last 10 cm
  double pz3[184] = {}, zring[3][8] = {};
  int npz3 = 0, nring = 0;
  double vp10 = 0, vh40 = 0, vc35 = 0, t25 = -1, tc35 = -1, pafter = 0, trmax = 0, tr0510 = 0, vpfl = 0;
  double vpt = -1, vpz = 0, nofoot = 0, wend = -1, tm2 = -1;  // where the fastest sink is, feet, phases
  double nf_sw = 0, nf_up = 0;  // feet-off time with a swing on; with every foot box centre above 6 cm (flight)
  double v0 = -1, vwmax = 0;     // horizontal CoM speed at the wound and its top during the walk [m/s]
  double p_0 = 0, pmax02 = 0, pmin2 = 0, rise2 = 0;  // wound-moment pelvis, top to 0.2 s, rise after 0.2 s
  bool low30 = false;                                 // the pelvis has been below 30 cm (settled)
  // pelvis height per frame over the first 1.5 s; its lowest point (the floor) and the largest sink speed in
  // the last 10 cm before it
  double rz[96] = {}, rt[96] = {}, pfloor = 0, vfloor = 0;
  int nz = 0;
  // Legs measure: pelvis first below 25 cm [s], knees and hips at the end and at t25 [deg], largest knee
  // straightening over 0.5 s from t25 + 0.6 s on [deg] with its time and side
  double kn25[2] = {0, 0}, kn_end[2] = {0, 0}, hip_end[2] = {0, 0}, str_max = 0, str_t = -1;
  int str_side = -1;
  bool kn25_set = false;  // knees at t25 taken
};

DRes Run(int sc, double moment) {
  DRes r;
  std::vector<BodyParams> bodies(1);
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(stag::BuildMutex());
    if (!w.Build(bodies, 3.0, 21, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return r; }
  }
  ApplyOverrides(w);
  w.Reset();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  double fx = 1, fy = 0, p0[3] = {0, 0, 0}, f0[2][3] = {}, hz_prev = -1, hc_prev[3] = {0, 0, 0}, cc_prev[3] = {0, 0, 0};
  int shots = 0, ti = 0;
  // pf180, pf200, pl160, pl180: push scenes forward and to the left (no bullets)
  const bool pushsc = sc >= kPF180;
  const double pJ = sc == kPF180 || sc == kPL180 ? 180.0 : sc == kPF200 ? 200.0 : 160.0;
  const bool pleft = sc == kPL160 || sc == kPL180;  // push to the bot's left
  const int nshot = pushsc ? 0 : sc == kRifle2 ? 2 : sc == kC7 ? 7 : sc >= kBurst5 ? 5 : 1;
  double t_push = -1;
  const double gap = sc == kRifle2 ? 0.4 : sc == kC10 ? 1.0 : (sc == kC05 || sc == kCB05 || sc == kC7) ? 0.5
                     : sc == kC012 ? 0.12 : sc == kB07 ? 0.7 : 0.15;
  bool started = false;
  const double tend = getenv("D8_END") ? atof(getenv("D8_END")) : 3.0;  // record length after the last bullet
  const int frames = (int)std::lround((moment + std::max(nshot - 1, 0) * gap + tend) * 60.0);
  double khist[2][30] = {};  // knee angles of the last 0.5 s [deg]
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* Rp = d->xmat + 9 * b.body0;
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    if (pushsc && t_push < 0 && t >= moment - 1e-9) {
      const double h = std::hypot(Rp[0], Rp[3]) + 1e-9;
      fx = Rp[0] / h; fy = Rp[3] / h;
      const double* c = d->xpos + 3 * (b.body0 + B_CHEST);  // chest body frame origin
      const double ux = pleft ? -fy : fx, uy = pleft ? fx : fy;  // push direction
      // scene impulse J x (body mass / 61.6 kg) N s at the chest centre moved 10 cm against the push, 6 cm up
      const double Jn = pJ * b.dims.total_mass / 61.6;
      double pt[3] = {c[0] - 0.1 * ux, c[1] - 0.1 * uy, c[2] + 0.06};
      double J[3] = {Jn * ux, Jn * uy, 0};
      w.PushAt(0, J, pt, B_CHEST);  // standard chest push point
      t_push = t;
    }
    if (shots < nshot && t >= moment + shots * gap - 1e-9) {
      if (shots == 0) {
        const double h = std::hypot(Rp[0], Rp[3]) + 1e-9;
        fx = Rp[0] / h; fy = Rp[3] / h;
      }
      const double lx = -fy, ly = fx;
      if (sc == kKill) {
        w.Kill(0);
      } else {
        const bool head = sc <= kHeadR;
        const int body = head ? B_HEAD : sc == kB07 ? B_LUMBAR : B_CHEST;
        const double* c = d->xipos + 3 * (b.body0 + body);
        double ux = -fx, uy = -fy;  // bullet direction: shooter in front
        if (sc == kHeadB || sc == kCB05) { ux = fx; uy = fy; }
        if (sc == kHeadL) { ux = -lx; uy = -ly; }
        if (sc == kHeadR) { ux = lx; uy = ly; }
        const double pt[3] = {c[0] - 0.08 * ux, c[1] - 0.08 * uy, c[2]};
        const double dv[3] = {ux, uy, 0};
        if (sc == kRifle2) w.Shoot(0, body, pt, dv, 45.0, 50.0);
        else w.Shoot(0, body, pt, dv);
      }
      ++shots;
    }
    if (!started && (b.dead_t >= 0 || t_push >= 0)) {
      started = true;
      for (int k = 0; k < 3; ++k) p0[k] = P[k];
      for (int f = 0; f < 2; ++f)
        for (int k = 0; k < 3; ++k) f0[f][k] = d->xpos[3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L)) + k];
    }
    // D8_TP=<scene> (moment 0): per frame from the first shot to the wound - bullets, L1 mode and swing,
    // crouch input, pelvis z, desired pelvis height and hip middle, foot loads, knees q > target and kp, assist [cm, N]
    if (!started && shots > 0 && getenv("D8_TP") && !strcmp(getenv("D8_TP"), kName[sc]) && moment < 1.7) {
      const double* q = d->qpos + b.qadr0;
      printf("DP %.3f n %d md %d sw %2d cr %4.1f tone %.2f | pz %5.1f Hd %5.1f hm %5.1f | fn %4.0f %4.0f | kn %5.1f>%5.1f "
             "%5.1f>%5.1f kp %4.0f %4.0f | asst %4.0f %4.0f\n",
             t - moment, shots, b.mode, b.swing, b.in.crouch * 100, b.in.tone, P[2] * 100, b.dbg_h[0] * 100,
             b.dbg_h[1] * 100, b.foot_fn[0], b.foot_fn[1], q[J_KNEE_L] * kRad, b.target[J_KNEE_L] * kRad,
             q[J_KNEE_R] * kRad, b.target[J_KNEE_R] * kRad, b.kp[J_KNEE_L], b.kp[J_KNEE_R], b.assist_f[0], b.assist_f[1]);
    }
    if (!started) continue;
    const double ts = t - (pushsc ? t_push : b.dead_t);
    // first hand (or forearm) and first head floor contact; largest elbow flexion in 0.3 s after the hand touch
    for (int c = 0; c < d->ncon; ++c) {
      const mjModel* mm = w.model();
      const int b1 = mm->geom_bodyid[d->contact[c].geom1], b2 = mm->geom_bodyid[d->contact[c].geom2];
      if (b1 != 0 && b2 != 0) continue;
      const int bd = (b1 ? b1 : b2) - b.body0;
      if ((bd == B_HAND_L || bd == B_HAND_R || bd == B_FARM_L || bd == B_FARM_R) && r.t_hand < 0) r.t_hand = ts;
      if (bd == B_HEAD && r.t_headc < 0) r.t_headc = ts;
    }
    if (r.t_hand >= 0 && ts < r.t_hand + 0.3)
      r.el_hand = std::fmax(r.el_hand, std::fmax(d->qpos[b.qadr0 + J_ELBOW_L], d->qpos[b.qadr0 + J_ELBOW_R]) * kRad);
    // pelvis at the wound, its top over the first 0.2 s and the largest rise after 0.2 s above its lowest
    // point so far, until the pelvis first goes below 30 cm (then it lies; first second) [cm]
    if (r.p_0 <= 0) r.p_0 = P[2] * 100;
    if (r.gu_t < 0 && b.mode == 2) {  // first give-up after the wound and its reason bits
      r.gu_t = ts;
      r.gu = b.dbg_gu;
      r.gu_pz = P[2] * 100;
    }
    if (ts <= 1.5 && r.nz < 96) {
      r.rz[r.nz] = P[2];
      r.rt[r.nz++] = ts;
    }
    if (P[2] < 0.3) r.low30 = true;
    if (ts <= 0.2) {
      r.pmax02 = std::fmax(r.pmax02, P[2] * 100);
    } else if (ts <= 1.0 && !r.low30) {
      r.pmin2 = r.pmin2 > 0 ? std::fmin(r.pmin2, P[2] * 100) : P[2] * 100;
      r.rise2 = std::fmax(r.rise2, P[2] * 100 - r.pmin2);
    }
    const double* q = d->qpos + b.qadr0;
    const double* Hc = d->xipos + 3 * (b.body0 + B_HEAD);
    const double tx = Hc[0] - P[0], ty = Hc[1] - P[1], tz = Hc[2] - P[2];
    const double fw = tx * fx + ty * fy, hl = std::sqrt(tx * tx + ty * ty + tz * tz) + 1e-9;
    const double trunk = std::acos(std::fmin(1.0, tz / hl)) * kRad * (fw >= 0 ? 1 : -1);
    const double sh = std::hypot(P[0] - p0[0], P[1] - p0[1]) * 100;
    const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
    double arm = 0, elb = 0;
    for (int s = 0; s < 2; ++s) {
      const double* S = d->xpos + 3 * (b.body0 + (s ? B_UARM_R : B_UARM_L));
      const double* E = d->xpos + 3 * (b.body0 + (s ? B_FARM_R : B_FARM_L));
      const double u[3] = {E[0] - S[0], E[1] - S[1], E[2] - S[2]};
      const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 1e-9;
      const double c = -(u[0] * Rc[2] + u[1] * Rc[5] + u[2] * Rc[8]) / un;  // against the chest -z axis
      const double a = std::acos(std::fmax(-1.0, std::fmin(1.0, c))) * kRad;
      if (a > arm) { arm = a; elb = q[s ? J_ELBOW_R : J_ELBOW_L] * kRad; }
    }
    if (ts <= 3.0) {
      const double* Cc = d->xipos + 3 * (b.body0 + B_CHEST);
      const int k0 = r.nring % 8;
      r.zring[0][k0] = P[2];
      r.zring[1][k0] = Cc[2];
      r.zring[2][k0] = Hc[2];
      ++r.nring;
      if (r.npz3 < 184) r.pz3[r.npz3++] = P[2];
      if (r.nring > 6) {
        const double vs = (r.zring[0][(r.nring - 7) % 8] - P[2]) / 0.1;
        if (vs > r.vp10) { r.vp10 = vs; r.vpt = ts; r.vpz = P[2]; }
      }
      {
        const double* cvw = b.pc[b.cur].com_vel;
        const double vh = std::hypot(cvw[0], cvw[1]);
        if (r.v0 < 0) r.v0 = vh;
        if (b.dead_walk_t < 0) r.vwmax = std::fmax(r.vwmax, vh);
      }
      if (b.dead_t >= 0 && b.dead_walk_t < 0 && b.foot_fn[0] < 50 && b.foot_fn[1] < 50) {
        r.nofoot += 1.0 / 60.0;
        if (b.swing >= 0) r.nf_sw += 1.0 / 60.0;
        if (b.pc[b.cur].foot_c[0][2] > 0.06 && b.pc[b.cur].foot_c[1][2] > 0.06) r.nf_up += 1.0 / 60.0;
      }
      if (r.wend < 0 && b.dead_walk_t >= 0) r.wend = b.dead_walk_t - b.dead_t;
      if (r.tm2 < 0 && b.mode == 2) r.tm2 = ts;
      if (r.nring > 3) {
        const int k3 = (r.nring - 4) % 8;
        if (Hc[2] < 0.40) r.vh40 = std::fmax(r.vh40, (r.zring[2][k3] - Hc[2]) / 0.05);
        if (Cc[2] < 0.35) r.vc35 = std::fmax(r.vc35, (r.zring[1][k3] - Cc[2]) / 0.05);
      }
      if (r.t25 < 0 && P[2] < 0.25) r.t25 = ts;
      if (r.tc35 < 0 && Cc[2] < 0.35) r.tc35 = ts;
      if (r.t25 >= 0) r.pafter = std::fmax(r.pafter, P[2]);
      r.trmax = std::fmax(r.trmax, std::fabs(trunk));
      if (ts >= 0.5 && ts <= 1.0) r.tr0510 = std::fmax(r.tr0510, std::fabs(trunk));
    }
    // D8_TW=<scene> (moment 0; D8_M1 for moment 1): every 2nd frame to 1.8 s - tone, L1 mode, walk, curl
    // weight, crouch, pelvis z and its sink over 0.1 s, desired height, foot loads, L0 weight shares, knee and hip
    // flexion q > target with the knee kp, |trunk|, CoM horizontal speed [cm, m/s, N, deg]
    if (getenv("D8_TW") && !strcmp(getenv("D8_TW"), kName[sc]) && (moment < 1.7) == !getenv("D8_M1") && ts < 1.8 &&
        fr % 2 == 0) {
      const double* cv = b.pc[b.cur].com_vel;
      printf("DW %.3f tone %.2f md %d wk %d cw %.2f cr %4.1f | pz %5.1f vs %5.2f Hd %5.1f | fn %4.0f %4.0f al %.2f %.2f | "
             "kn %5.1f>%5.1f %5.1f>%5.1f kp %4.0f %4.0f | hp %5.1f>%5.1f %5.1f>%5.1f | tr %4.0f v %4.2f sw %2d\n",
             ts, b.in.tone, b.mode, b.dead_walk_t < 0 ? 1 : 0, b.curl_w, b.in.crouch * 100, P[2] * 100,
             r.nring > 6 ? (r.zring[0][(r.nring - 7) % 8] - P[2]) / 0.1 : 0.0, b.dbg_h[0] * 100, b.foot_fn[0],
             b.foot_fn[1], b.al_dbg[0], b.al_dbg[1], q[J_KNEE_L] * kRad, b.target[J_KNEE_L] * kRad, q[J_KNEE_R] * kRad,
             b.target[J_KNEE_R] * kRad, b.kp[J_KNEE_L], b.kp[J_KNEE_R], q[J_HIP_FLEX_L] * kRad,
             b.target[J_HIP_FLEX_L] * kRad, q[J_HIP_FLEX_R] * kRad, b.target[J_HIP_FLEX_R] * kRad, trunk,
             std::hypot(cv[0], cv[1]), b.swing);
      // along the CoM motion from the CoM: swing place (after reach), each foot's ankle [cm]
      const double* pcm = b.pc[b.cur].com;
      const double vn = std::hypot(cv[0], cv[1]) + 1e-9, ux = cv[0] / vn, uy = cv[1] / vn;
      const double* an = b.pc[b.cur].ankle[0];
      const double* an1 = b.pc[b.cur].ankle[1];
      printf("DX %.3f pl %+5.1f ftL %+5.1f ftR %+5.1f\n", ts,
             b.swing >= 0 ? ((b.sw_tg[0] - pcm[0]) * ux + (b.sw_tg[1] - pcm[1]) * uy) * 100 : 0.0,
             ((an[0] - pcm[0]) * ux + (an[1] - pcm[1]) * uy) * 100, ((an1[0] - pcm[0]) * ux + (an1[1] - pcm[1]) * uy) * 100);
    }
    if (getenv("D8_TA") && !strcmp(getenv("D8_TA"), kName[sc]) && moment < 1.7 && fr % 3 == 0) {
      printf("DA %.2f tone %.2f lay %.2f ff %5.2f fl %5.2f cg %.2f md %d fs %d |", ts, b.in.tone,
             b.dead_lay_t >= 0 ? t - b.dead_lay_t : -1.0, b.fall_fwd, b.fall_lat, b.catch_g, b.mode, b.fall_set);
      for (int j : {J_SH_FLEX_L, J_SH_ABD_L, J_ELBOW_L, J_SH_FLEX_R, J_ELBOW_R})
        printf(" q %5.1f T %5.1f kp %4.0f gs %.1f u %5.1f |", q[j] * kRad, b.target[j] * kRad, b.kp[j], b.gsup[j],
               d->ctrl[b.act0 + j]);
      printf("\n");
    }
    if (getenv("D8_TN") && !strcmp(getenv("D8_TN"), kName[sc]) && (moment < 1.7) == !getenv("D8_M1") && ts < 6.0 && fr % 2 == 0) {
      const double* hv = d->cvel + 6 * (b.body0 + B_HEAD);  // [ang lin] at the subtree com: speed via finite diff below
      (void)hv;
      printf("DN %.3f tone %.2f pz %.2f hz %.2f tr %4.0f |", ts, b.in.tone, P[2], Hc[2], trunk);
      for (int j : {J_NECK_PITCH, J_CHEST_PITCH, J_LUMBAR_PITCH, J_HIP_FLEX_L, J_HIP_FLEX_R, J_KNEE_L, J_KNEE_R})
        printf(" q %5.1f T %5.1f kp %4.0f |", q[j] * kRad, b.target[j] * kRad, b.kp[j]);
      const double* wv = d->cvel + 6 * (b.body0 + B_PELVIS);  // angular part first, world frame
      printf(" uc %5.2f wp %5.1f lay %.2f ncon %d\n", d->xmat[9 * (b.body0 + B_CHEST) + 6], -fy * wv[0] + fx * wv[1], b.dead_lay_t >= 0 ? t - b.dead_lay_t : -1.0,
             d->ncon);
    }
    if (getenv("D8_TR") && !strcmp(getenv("D8_TR"), kName[sc]) && moment < 1.7 && ts < 1.0) {
      const mjModel* m = w.model();
      printf("DT %.3f tone %.2f pz %.3f |", ts, b.in.tone, P[2]);
      for (int j : {J_HIP_FLEX_L, J_KNEE_L, J_ANKLE_DORSI_L}) {
        const int dof = b.dof0 + j;
        printf(" q %5.1f T %5.1f kp %5.0f u %6.1f bias %6.1f dmp %5.1f arm %.2f |", q[j] * kRad, b.target[j] * kRad,
               b.kp[j], d->ctrl[b.act0 + j], d->qfrc_bias[dof], m->dof_damping[dof], m->dof_armature[dof]);
      }
      double fmax = 0;
      int g1 = -1, g2 = -1;
      for (int c = 0; c < d->ncon; ++c) {
        mjtNum f6[6];
        mj_contactForce(m, d, c, f6);
        if (f6[0] > fmax) { fmax = f6[0]; g1 = m->geom_bodyid[d->contact[c].geom1] - b.body0; g2 = m->geom_bodyid[d->contact[c].geom2] - b.body0; }
      }
      printf(" asst %.0f %.0f mode %d | ncon %d fmax %.0f N bodies %d-%d\n", b.assist_f[0], b.assist_f[1], b.mode, d->ncon,
             fmax, g1, g2);
    }
    // D8_TL=<scene> (moment 0; D8_M1 for moment 1): per frame from the relaxation start - tone, swing, steps on,
    // crouch input, pelvis z and its speed, desired pelvis height (L1) and hip midpoint, foot loads, feet along the
    // motion (- facing) from the pelvis, knees and hip flexion q > target, head z and speed [cm, m/s, N, deg]
    if (getenv("D8_TL") && !strcmp(getenv("D8_TL"), kName[sc]) && (moment < 1.7) == !getenv("D8_M1") && ts < 1.2) {
      static thread_local double pz0 = 0, hz0[3] = {0, 0, 0}, t0 = -1;
      const double dtf = t0 >= 0 && t > t0 ? t - t0 : 1.0;
      const double vz = t0 >= 0 ? (P[2] - pz0) / dtf : 0.0;
      const double hv = t0 >= 0 ? std::sqrt((Hc[0] - hz0[0]) * (Hc[0] - hz0[0]) + (Hc[1] - hz0[1]) * (Hc[1] - hz0[1]) +
                                            (Hc[2] - hz0[2]) * (Hc[2] - hz0[2])) / dtf : 0.0;
      pz0 = P[2];
      for (int k = 0; k < 3; ++k) hz0[k] = Hc[k];
      t0 = t;
      double fa[2];
      for (int f = 0; f < 2; ++f) {
        const double* F = d->xpos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        fa[f] = -((F[0] - P[0]) * fx + (F[1] - P[1]) * fy) * 100;
      }
      double tga = 0, pla = 0;
      if (b.swing >= 0) {
        tga = -((b.sw_tg[0] - P[0]) * fx + (b.sw_tg[1] - P[1]) * fy) * 100;
        pla = -((b.place[b.swing][0] - P[0]) * fx + (b.place[b.swing][1] - P[1]) * fy) * 100;
      }
      printf("DS %.3f sw %2d t %.2f tg %5.1f pl %5.1f ftz %5.1f\n", ts, b.swing, b.sw_t0 >= 0 ? t - b.sw_t0 : -1.0, tga,
             pla, b.sw_ft[2] * 100);
      printf("DL %.3f tone %.2f sw %2d md %d walk %d cr %4.1f | pz %5.1f vz %5.2f Hd %5.1f hm %5.1f | fn %4.0f %4.0f | "
             "ft %5.1f %5.1f | kn %5.1f>%5.1f %5.1f>%5.1f hp %5.1f>%5.1f %5.1f>%5.1f | hz %5.1f hv %4.2f | kp %4.0f %4.0f "
             "kd %3.0f %3.0f\n",
             ts, b.in.tone, b.swing, b.mode, b.dead_walk_t < 0 ? 1 : 0, b.in.crouch * 100, P[2] * 100, vz,
             b.dbg_h[0] * 100, b.dbg_h[1] * 100, b.foot_fn[0], b.foot_fn[1], fa[0], fa[1], q[J_KNEE_L] * kRad,
             b.target[J_KNEE_L] * kRad, q[J_KNEE_R] * kRad, b.target[J_KNEE_R] * kRad, q[J_HIP_FLEX_L] * kRad,
             b.target[J_HIP_FLEX_L] * kRad, q[J_HIP_FLEX_R] * kRad, b.target[J_HIP_FLEX_R] * kRad, Hc[2] * 100, hv,
             b.kp[J_KNEE_L], b.kp[J_KNEE_R], w.model()->dof_damping[b.dof0 + J_KNEE_L], w.model()->dof_damping[b.dof0 + J_KNEE_R]);
    }
    while (ti < kNT && ts >= kT[ti] - 1e-9) {
      double* v = r.v[ti++];
      v[0] = P[2] * 100; v[1] = sh; v[2] = trunk;
      v[3] = q[J_KNEE_L] * kRad; v[4] = q[J_KNEE_R] * kRad; v[5] = q[J_HIP_FLEX_L] * kRad; v[6] = q[J_HIP_FLEX_R] * kRad;
      v[7] = arm; v[8] = elb;
      const double* HL = d->xpos + 3 * (b.body0 + B_HAND_L);
      const double* HR = d->xpos + 3 * (b.body0 + B_HAND_R);
      v[9] = std::sqrt((HL[0] - HR[0]) * (HL[0] - HR[0]) + (HL[1] - HR[1]) * (HL[1] - HR[1]) + (HL[2] - HR[2]) * (HL[2] - HR[2])) * 100;
      v[10] = (std::fmax(HL[2], HR[2]) - Hc[2]) * 100;  // higher hand above the head CoM
      double dh = 9;
      for (const double* Hh : {HL, HR})
        dh = std::fmin(dh, std::sqrt((Hh[0] - Hc[0]) * (Hh[0] - Hc[0]) + (Hh[1] - Hc[1]) * (Hh[1] - Hc[1]) + (Hh[2] - Hc[2]) * (Hh[2] - Hc[2])));
      v[11] = dh * 100;
    }
    for (int f = 0; f < 2 && r.t60 < 0; ++f)  // foot rise above its start before the pelvis passes 60 cm
      r.lift = std::fmax(r.lift, (d->xpos[3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L)) + 2] - f0[f][2]) * 100);
    if (r.t60 < 0 && P[2] < 0.60) {
      r.fdist60 = 9;
      for (int f = 0; f < 2; ++f) {
        const double* F = d->xpos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        r.fdist60 = std::fmin(r.fdist60, std::hypot(F[0] - P[0], F[1] - P[1]));
      }
      r.fdist60 *= 100;
      r.kmax60 = std::fmax(q[J_KNEE_L], q[J_KNEE_R]) * kRad;
      r.t60 = ts; r.tr60 = trunk; r.kn60 = std::fmin(q[J_KNEE_L], q[J_KNEE_R]) * kRad; r.sh60 = sh;
    }
    if (ts >= 2.0 && r.foot2 == 0)
      r.foot2 = std::fmax(d->xpos[3 * (b.body0 + B_FOOT_L) + 2], d->xpos[3 * (b.body0 + B_FOOT_R) + 2]) * 100;
    if (r.t50 < 0 && P[2] < 0.50) {
      r.t50 = ts;
      r.sh50f = ((P[0] - p0[0]) * fx + (P[1] - p0[1]) * fy) * 100;  // + forward of the facing at the start
      r.sh50l = (-(P[0] - p0[0]) * fy + (P[1] - p0[1]) * fx) * 100;  // + to the left
      for (int f = 0; f < 2; ++f) {
        const double* F = d->xpos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        r.ft50 = std::fmax(r.ft50, std::hypot(F[0] - f0[f][0], F[1] - f0[f][1]) * 100);
      }
    }
    // peak head speed while the head CoM is below 40 cm (3 s): a chest-first impact stops the head above 20 cm, so
    // the speed at 20 cm alone misses a hard slap
    if (hz_prev > 0 && Hc[2] < 0.40 && ts <= 3.0) {
      const double dx = Hc[0] - hc_prev[0], dy = Hc[1] - hc_prev[1], dz = Hc[2] - hc_prev[2];
      r.hpk40 = std::fmax(r.hpk40, std::sqrt(dx * dx + dy * dy + dz * dz) * 60.0);
    }
    if (r.t20 < 0 && hz_prev > 0 && Hc[2] < 0.20) {
      r.t20 = ts;
      const double dx = Hc[0] - hc_prev[0], dy = Hc[1] - hc_prev[1], dz = Hc[2] - hc_prev[2];
      r.hv20 = std::sqrt(dx * dx + dy * dy + dz * dz) * 60.0;
      const double* C = d->xipos + 3 * (b.body0 + B_CHEST);
      const double ex = C[0] - cc_prev[0], ey = C[1] - cc_prev[1], ez = C[2] - cc_prev[2];
      r.cv20 = std::sqrt(ex * ex + ey * ey + ez * ez) * 60.0;
    }
    for (int k = 0; k < 3; ++k) cc_prev[k] = d->xipos[3 * (b.body0 + B_CHEST) + k];
    hz_prev = Hc[2];
    for (int k = 0; k < 3; ++k) hc_prev[k] = Hc[k];
    r.pel_end = P[2];
    r.knee_end = std::fmax(q[J_KNEE_L], q[J_KNEE_R]) * kRad;  // larger knee at the end [deg]
    if (r.t25 < 0 && P[2] < 0.25) r.t25 = ts;  // also after 3 s (the 0-3 s measures stop there)
    if (!r.kn25_set && r.t25 >= 0) {
      r.kn25_set = true;
      for (int f = 0; f < 2; ++f) r.kn25[f] = q[f ? J_KNEE_R : J_KNEE_L] * kRad;
    }
    for (int f = 0; f < 2; ++f) {
      const double kq = q[f ? J_KNEE_R : J_KNEE_L] * kRad;
      double& old = khist[f][fr % 30];
      if (r.t25 >= 0 && ts - 0.5 >= r.t25 + 0.6 && old - kq > r.str_max) {
        r.str_max = old - kq;
        r.str_t = ts;
        r.str_side = f;
      }
      old = kq;
      r.kn_end[f] = kq;
      r.hip_end[f] = q[f ? J_HIP_FLEX_R : J_HIP_FLEX_L] * kRad;
    }
    r.foot_end = std::fmax(d->xpos[3 * (b.body0 + B_FOOT_L) + 2], d->xpos[3 * (b.body0 + B_FOOT_R) + 2]);
    const double* cv = d->qvel + b.dof0;
    double s2 = 0;
    for (int j = 0; j < J_COUNT; ++j) s2 += cv[j] * cv[j];
    r.v_end = std::sqrt(s2 / J_COUNT);
  }
  int im = 0;
  for (int k = 1; k < r.nz; ++k)
    if (r.rz[k] < r.rz[im]) im = k;
  r.pfloor = r.nz > 0 ? r.rz[im] * 100 : 0.0;
  for (int k = 1; k <= im; ++k)
    if (r.rz[k] <= r.rz[im] + 0.10 && r.rt[k] > r.rt[k - 1])
      r.vfloor = std::fmax(r.vfloor, (r.rz[k - 1] - r.rz[k]) / (r.rt[k] - r.rt[k - 1]));
  {
    double zf = 9;
    for (int k = 0; k < r.npz3; ++k) zf = std::fmin(zf, r.pz3[k]);
    for (int k = 3; k < r.npz3; ++k)
      if (r.pz3[k] <= zf + 0.10) r.vpfl = std::fmax(r.vpfl, (r.pz3[k - 3] - r.pz3[k]) / 0.05);
  }
  return r;
}

}  // namespace

int RunDeath8Bench(int argc, char** argv) {
  const char* only = getenv("D8_S");  // scene name filter
  double moments[2] = {1.6, 1.87};
  // D8_MOM=a,b sets the two moments [s]
  if (getenv("D8_MOM")) sscanf(getenv("D8_MOM"), "%lf,%lf", &moments[0], &moments[1]);
  printf("D8 time [s]        ");
  for (int k = 0; k < kNT; ++k) printf(" %5.1f", kT[k]);
  printf("\n");
  for (int s = 0; s < kNS; ++s) {
    if (only && !strstr(only, kName[s])) continue;
    for (int m = 0; m < 2; ++m) {
      const DRes r = Run(s, moments[m]);
      for (int c = 0; c < kNC; ++c) {
        printf("D8 %-7s m%d %-7s", kName[s], m, kCol[c]);
        for (int k = 0; k < kNT; ++k) printf(" %5.0f", r.v[k][c]);
        printf("\n");
      }
      printf("D8 %-7s m%d below60 %.2f s (trunk %.0f, knee %.0f, shift %.0f cm) below50 %.2f s (fwd %.0f left %.0f cm, feet moved %.0f cm); head20 %.2f s %.1f m/s (chest %.1f);"
             " feet at 2 s %.0f cm; lift %.0f cm, at 60: knee max %.0f foot %.0f cm; end pelvis %.2f m knee %.0f foot %.2f m joint rms %.2f rad/s\n",
             kName[s], m, r.t60, r.tr60, r.kn60, r.sh60, r.t50, r.sh50f, r.sh50l, r.ft50, r.t20, r.hv20, r.cv20, r.foot2, r.lift, r.kmax60, r.fdist60, r.pel_end, r.knee_end, r.foot_end, r.v_end);
      printf("D8 %-7s m%d touch hand %.2f head %.2f s, elbow max 0.3 s after the hand %.0f deg\n", kName[s], m, r.t_hand,
             r.t_headc, r.el_hand);
      printf("D8 %-7s m%d legs t25 %.2f s knees %.0f %.0f | end pelvis %.0f cm knees %.0f %.0f hips %.0f %.0f |"
             " straighten max %.0f deg in 0.5 s at %.2f s side %d\n", kName[s], m, r.t25, r.kn25[0], r.kn25[1],
             r.pel_end * 100, r.kn_end[0], r.kn_end[1], r.hip_end[0], r.hip_end[1], r.str_max, r.str_t, r.str_side);
      printf("D8 %-7s m%d t48 pelvis at the wound %.1f top to 0.2 s %.1f rise after 0.2 s %.1f cm; floor %.1f cm, last 10 cm "
             "%.2f m/s; head peak below 40 cm %.1f m/s\n", kName[s], m, r.p_0, r.pmax02, r.rise2, r.pfloor, r.vfloor, r.hpk40);
      printf("D8 %-7s m%d giveup %d at %.2f s pelvis %.1f cm\n", kName[s], m, r.gu, r.gu_t, r.gu_pz);
      printf("D8 %-7s m%d t50 vp10 %.2f below25 %.2f chest35 %.2f after %.1f trunk %.0f t0510 %.0f impact pelvis %.2f "
             "chest %.2f head %.2f | z", kName[s], m, r.vp10, r.t25, r.tc35, r.pafter * 100, r.trmax, r.tr0510, r.vpfl,
             r.vc35, r.vh40);
      for (int k = 0; k <= 15; ++k) {
        const int i = (int)std::lround(k * 0.1 * 60.0);
        printf(" %.0f", i < r.npz3 ? r.pz3[i] * 100 : -1.0);
      }
      printf("\n");
      printf("D8 %-7s m%d t50b vp10 at %.2f s pelvis %.1f cm | walk: both feet < 50 N %.2f s, end %.2f s; mode 2 %.2f s"
             " | feet off: swing on %.2f s, both up %.2f s | v wound %.2f walk top %.2f\n",
             kName[s], m, r.vpt, r.vpz * 100, r.nofoot, r.wend, r.tm2, r.nf_sw, r.nf_up, r.v0, r.vwmax);
    }
  }
  (void)argc; (void)argv;
  return 0;
}
