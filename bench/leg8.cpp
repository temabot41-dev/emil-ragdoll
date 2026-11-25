// Leg bullet bench: pistol bullet into a leg, measured against a target table: emil_bench leg8 [threads] [-p ...]
// One bot, bullets from the front at 1.5 s: left thigh, right thigh, left shin, left thigh twice (1 s apart), left
// thigh four times (0.8 s apart). Per time after the first hit: trunk forward (line pelvis -> head base against the
// vertical, in the facing plane) [deg], wounded / healthy knee [deg], pelvis height change [cm], pelvis shift toward
// the wounded side [cm], near / far hand distance to the wound point (moves with the leg) [cm], steps.
// Events: dips (pelvis down > 10 cm within 0.2 s), fall time (pelvis below 0.5 m).
// Target: trunk 15 11 35 36 36 34 40 33 at 0.4 1 2 3 3.4 3.6 3.8 4 s; wounded knee 36 36 46 60 75 93 82 64;
// healthy 37 34 36 33 77 100 44 47; pelvis -3.5 -3 -4.5 -6.2 -13 -25 -8 -10 cm; side 0 -3 8 29 48 57 72 91 cm; near
// hand 15 cm at 0.4 s, far 17-19 cm at 1.5 s; 1.5 m sideways in 6 s; shin: first dip about 5 s.
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
struct L8Scene { const char* name; int body; int n; double dt; };
constexpr int kN8 = 5;
const L8Scene kL8[kN8] = {{"thighL", B_THIGH_L, 1, 0}, {"thighR", B_THIGH_R, 1, 0}, {"shinL", B_SHANK_L, 1, 0},
                          {"thighL2", B_THIGH_L, 2, 1.0}, {"thighL4", B_THIGH_L, 4, 0.8}};
constexpr int kNT = 12;
const double kT[kNT] = {0.4, 1.0, 1.5, 2.0, 3.0, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 6.0};
constexpr int kNC = 13;  // trunk kneeW kneeH drop side handN handF steps, knee targets W H

struct L8Res {
  double v[kNT][kNC] = {};
  double t_fell = -1, t_dip[4] = {-1, -1, -1, -1};
  int n_dip = 0;
};

L8Res RunL8(const L8Scene& sc) {
  L8Res r;
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
  const double kMoment = 1.5;
  const int right = sc.body == B_THIGH_R || sc.body == B_SHANK_R;
  const int kw = right ? J_KNEE_R : J_KNEE_L, kh = right ? J_KNEE_L : J_KNEE_R;
  const int hn = right ? B_HAND_R : B_HAND_L, hf = right ? B_HAND_L : B_HAND_R;
  double t0 = -1, fx = 1, fy = 0, p0[3] = {0, 0, 0}, loc[3] = {0, 0, 0};
  int td0 = 0, ti = 0, shots = 0;
  std::vector<double> pz;  // pelvis height per frame since the hit (dips)
  // L8_DUR=<s> runs longer than 6 s after the hit (falls still counted; the table stops at 6 s)
  const int frames = (int)std::lround((kMoment + (getenv("L8_DUR") ? std::atof(getenv("L8_DUR")) : 6.05)) * 60.0);
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* Rp = d->xmat + 9 * b.body0;
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    const double* Rl = d->xmat + 9 * (b.body0 + sc.body);
    const double* Xl = d->xpos + 3 * (b.body0 + sc.body);
    if (t0 < 0 && t < kMoment) continue;
    if (shots < sc.n && t >= kMoment + shots * sc.dt - 1e-9) {
      if (t0 < 0) {
        const double h = std::hypot(Rp[0], Rp[3]) + 1e-9;
        fx = Rp[0] / h; fy = Rp[3] / h;
        t0 = t; td0 = b.td_cnt;
        for (int k = 0; k < 3; ++k) p0[k] = P[k];
      }
      const double* c = d->xipos + 3 * (b.body0 + sc.body);
      const double pt[3] = {c[0] + 0.1 * fx, c[1] + 0.1 * fy, c[2]};
      const double dv[3] = {-fx, -fy, 0};
      // wound point on the front of the leg (8 cm ahead of the segment axis), kept in the segment frame
      const double wp[3] = {c[0] + 0.06 * fx - Xl[0], c[1] + 0.06 * fy - Xl[1], c[2] - Xl[2]};
      if (shots == 0)
        for (int k = 0; k < 3; ++k) loc[k] = Rl[k] * wp[0] + Rl[3 + k] * wp[1] + Rl[6 + k] * wp[2];
      w.Shoot(0, sc.body, pt, dv);
      ++shots;
      continue;
    }
    const double ts = t - t0;
    pz.push_back(P[2]);
    const int n = (int)pz.size();
    if (n > 12 && pz[n - 13] - P[2] > 0.10) {
      const bool fresh = r.n_dip == 0 || ts - r.t_dip[std::min(r.n_dip, 4) - 1] > 0.5;
      if (fresh) { if (r.n_dip < 4) r.t_dip[r.n_dip] = ts; ++r.n_dip; }
    }
    if (r.t_fell < 0 && P[2] < 0.5) r.t_fell = ts;
    const double lx = -fy, ly = fx, sg = right ? -1.0 : 1.0;
    const double* H = d->xpos + 3 * (b.body0 + B_HEAD);
    const double tx = H[0] - P[0], ty = H[1] - P[1], tz = H[2] - P[2];
    const double trunk = std::atan2(tx * fx + ty * fy, tz) * kRad;
    double wq[3];
    for (int k = 0; k < 3; ++k) wq[k] = Xl[k] + Rl[3 * k] * loc[0] + Rl[3 * k + 1] * loc[1] + Rl[3 * k + 2] * loc[2];
    auto dist = [&](int bd) {
      const double* q = d->xpos + 3 * (b.body0 + bd);
      return std::sqrt((q[0] - wq[0]) * (q[0] - wq[0]) + (q[1] - wq[1]) * (q[1] - wq[1]) + (q[2] - wq[2]) * (q[2] - wq[2]));
    };
    const double* q = d->qpos + b.qadr0;
    if (getenv("L8_TR") && std::atoi(getenv("L8_TR")) == (int)(&sc - kL8) && fr % 3 == 0)
      printf("LT %5.2f pz %.2f on %d%d sw %d kW %3.0f/%3.0f kH %3.0f/%3.0f mode %d down %d lean %5.2f %5.2f weak %.2f %.2f give %.2f live %.2f crouch %.2f tl %.2f\n", ts, P[2], b.foot_on[0], b.foot_on[1], b.swing, q[kw] * kRad, b.target[kw] * kRad, q[kh] * kRad, b.target[kh] * kRad, b.mode, b.down,
             b.in.lean[0], b.in.lean[1], b.in.leg_weak[0], b.in.leg_weak[1], b.leg_give[0] - t0, b.bul_until - t0, b.in.crouch,
             b.in.trunk_lean);
    if (getenv("L8_TR") && std::atoi(getenv("L8_TR")) == (int)(&sc - kL8) && fr % 3 == 0) {  // trace of the hurt hip
      const int ha = (right ? J_HIP_FLEX_R : J_HIP_FLEX_L);
      printf("LA %5.2f hurt hip flex %5.1f/%5.1f abd %5.1f/%5.1f knee %5.1f/%5.1f | weak %.2f %.2f ld %.2f %.2f\n", ts,
             q[ha] * kRad, b.target[ha] * kRad, q[ha + 1] * kRad, b.target[ha + 1] * kRad, q[ha + 3] * kRad,
             b.target[ha + 3] * kRad, b.in.leg_weak[0], b.in.leg_weak[1], b.ld[0], b.ld[1]);
    }
    double fh[2] = {0, 0};  // normal contact force on each hand from the bot's own body (hand as a strut?)
    if (getenv("L8_TR")) {
      const mjModel* m = w.model();
      for (int c = 0; c < d->ncon; ++c) {
        const int b1 = m->geom_bodyid[d->contact[c].geom1], b2 = m->geom_bodyid[d->contact[c].geom2];
        for (int h = 0; h < 2; ++h) {
          const int hb = b.body0 + (h ? B_HAND_R : B_HAND_L);
          if (b1 == hb || b2 == hb) { mjtNum f6[6]; mj_contactForce(m, d, c, f6); fh[h] += f6[0]; }
        }
      }
    }
    if (getenv("L8_TR") && std::atoi(getenv("L8_TR")) == (int)(&sc - kL8) && fr % 3 == 0)
      printf("LH %5.2f hand contact L %5.1f R %5.1f N\n", ts, fh[0], fh[1]);
    if (getenv("L8_TR") && std::atoi(getenv("L8_TR")) == (int)(&sc - kL8) && fr % 3 == 0) {
      const double* FL = d->xpos + 3 * (b.body0 + B_FOOT_L);
      const double* FR = d->xpos + 3 * (b.body0 + B_FOOT_R);
      printf("LF %5.2f pel %6.2f %6.2f footL %6.2f %6.2f z %.2f footR %6.2f %6.2f z %.2f vg %5.2f %5.2f sw %d on %d%d plL %5.2f %5.2f plR %5.2f %5.2f tg %5.2f %5.2f\n", ts, P[0], P[1],
             FL[0], FL[1], FL[2], FR[0], FR[1], FR[2], b.in.vgoal[0], b.in.vgoal[1], b.swing, b.foot_on[0], b.foot_on[1], b.place[0][0], b.place[0][1], b.place[1][0], b.place[1][1], b.sw_tg[0], b.sw_tg[1]);
    }
    if (getenv("L8_TR") && std::atoi(getenv("L8_TR")) == (int)(&sc - kL8) && fr % 3 == 0)
      printf("LS %5.2f lum %5.1f/%5.1f kp %5.0f gs %.2f bias %6.1f qd %6.1f | ch %5.1f/%5.1f kp %5.0f bias %6.1f\n", ts,
             q[J_LUMBAR_PITCH] * kRad, b.target[J_LUMBAR_PITCH] * kRad, b.kp[J_LUMBAR_PITCH], b.gsup[J_LUMBAR_PITCH],
             d->qfrc_bias[b.dof0 + J_LUMBAR_PITCH], d->qvel[b.dof0 + J_LUMBAR_PITCH] * kRad, q[J_CHEST_PITCH] * kRad,
             b.target[J_CHEST_PITCH] * kRad, b.kp[J_CHEST_PITCH], d->qfrc_bias[b.dof0 + J_CHEST_PITCH]);
    while (ti < kNT && ts >= kT[ti] - 1e-9) {
      double* v = r.v[ti];
      v[0] = trunk;
      v[1] = q[kw] * kRad;
      v[2] = q[kh] * kRad;
      v[3] = (P[2] - p0[2]) * 100;
      v[4] = sg * ((P[0] - p0[0]) * lx + (P[1] - p0[1]) * ly) * 100;
      v[5] = dist(hn) * 100;
      v[6] = dist(hf) * 100;
      v[7] = b.td_cnt - td0;
      v[8] = b.target[kw] * kRad;
      v[9] = b.target[kh] * kRad;
      v[10] = (q[J_LUMBAR_PITCH] + q[J_CHEST_PITCH]) * kRad;
      v[11] = (b.target[J_LUMBAR_PITCH] + b.target[J_CHEST_PITCH]) * kRad;
      v[12] = std::atan2(-Rp[6], Rp[8]) * kRad;
      ++ti;
    }
  }
  return r;
}

}  // namespace

int RunLeg8Bench(int argc, char** argv) {
  int threads = argc > 2 ? std::atoi(argv[2]) : 3;
  threads = std::max(1, std::min(threads, 3));
  std::vector<L8Res> res(kN8);
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&]() { for (int i; (i = next++) < kN8;) res[i] = RunL8(kL8[i]); });
  for (auto& x : th) x.join();
  static const char* kc[kNC] = {"trunk", "kneeW", "kneeH", "drop", "side", "handN", "handF", "steps", "tgtW", "tgtH", "spine", "spineT", "pelP"};
  printf("L8 time [s]           ");
  for (int k = 0; k < kNT; ++k) printf(" %5.1f", kT[k]);
  printf("\n");
  for (int i = 0; i < kN8; ++i) {
    const L8Res& r = res[i];
    for (int c = 0; c < kNC; ++c) {
      printf("L8 %-8s %-6s     ", kL8[i].name, kc[c]);
      for (int k = 0; k < kNT; ++k) printf(" %5.0f", r.v[k][c]);
      printf("\n");
    }
    printf("L8 %-8s dips %d at", kL8[i].name, r.n_dip);
    for (int k = 0; k < std::min(r.n_dip, 4); ++k) printf(" %.2f", r.t_dip[k]);
    printf(" s; fell %.2f s\n", r.t_fell);
  }
  return 0;
}
