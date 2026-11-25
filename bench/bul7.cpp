// Bullet reaction bench: body measures after a pistol bullet, compared with target bands.
//   emil_bench bul7 [threads] [-p ...]
// One bot, pistol bullet from the front at 1.5 s (times below from the hit):
//   chest  1.39 m, 3 cm left of the middle;  belly 1.05 m;  thigh left 0.73 m;  burst 5 chest bullets every 0.12 s.
// Columns per time: drop = pelvis height change [cm]; knee L/R [deg]; trunk = line pelvis -> head base, + back (along
// the bullet) [deg]; bend = head-base line against the pelvis-chest line in the sagittal plane, + forward flexion
// [deg]; side = trunk lean toward the wound side [deg]; hand heights [cm]; hand distances to the wound point [cm];
// steps since the hit; travel = pelvis horizontal path from the hit pose [cm]. All angle changes from the hit pose.
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
struct B7Scene { const char* name; int body; double z, side; int n; double dt; };
constexpr int kNS = 6;
const B7Scene kB7[kNS] = {{"chest", B_CHEST, 1.39, 0.03, 1, 0}, {"belly", B_LUMBAR, 1.05, 0, 1, 0},
                          {"thigh", B_THIGH_L, 0.73, 0, 1, 0}, {"burst5", B_CHEST, 1.39, 0.03, 5, 0.12},
                          {"chestL", B_CHEST, 1.39, 0.10, 1, 0}, {"chestR", B_CHEST, 1.39, -0.10, 1, 0}};
constexpr int kNT = 13;
const double kT[kNT] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.2, 4.0};
constexpr int kNC = 17;  // + pose: line, spine, neck, head (see B7P)
constexpr int kNC0 = 13;  // drop kneeL kneeR trunk bend side handLz handRz dL dR steps travel yaw

struct B7Res {
  double v[kNT][kNC] = {};
  double t_fell = -1;
};

B7Res RunB7(const B7Scene& sc) {
  B7Res r;
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
  const double kMoment = 1.5, kTail = 4.05;
  double yaw0 = 0; double pose0[4] = {0, 0, 0, 0};
  double t0 = -1, fx = 1, fy = 0, z0 = 0, p0[2] = {0, 0}, tr0 = 0, bd0 = 0, sd0 = 0, wl[3] = {0, 0, 0};
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
      t0 = t; z0 = P[2]; p0[0] = P[0]; p0[1] = P[1]; td0 = b.td_cnt;
    }
    const double lx = -fy, ly = fx;  // left
    const double ts = t - t0;
    while (shots < sc.n && ts >= shots * sc.dt - 1e-9) {
      const int bd = b.body0 + sc.body;
      const double* c = d->xipos + 3 * bd;
      const double pt[3] = {c[0] + 0.1 * fx + sc.side * lx, c[1] + 0.1 * fy + sc.side * ly, sc.z * b.body.height / 1.78};
      const double dv[3] = {-fx, -fy, 0};
      if (shots == 0) {  // wound point in the hit body's frame
        const double* Rb = d->xmat + 9 * bd;
        const double* xb = d->xpos + 3 * bd;
        const double q[3] = {pt[0] - xb[0], pt[1] - xb[1], pt[2] - xb[2]};
        for (int a = 0; a < 3; ++a) wl[a] = Rb[a] * q[0] + Rb[3 + a] * q[1] + Rb[6 + a] * q[2];
      }
      w.Shoot(0, sc.body, pt, dv);
      ++shots;
    }
    const double up[3] = {H[0] - P[0], H[1] - P[1], H[2] - P[2]};
    const double lo[3] = {C[0] - P[0], C[1] - P[1], C[2] - P[2]}, hi[3] = {H[0] - C[0], H[1] - C[1], H[2] - C[2]};
    const double trunk = std::atan2(-(up[0] * fx + up[1] * fy), up[2]) * kRad;
    const double bend = (std::atan2(hi[0] * fx + hi[1] * fy, hi[2]) - std::atan2(lo[0] * fx + lo[1] * fy, lo[2])) * kRad;
    const double wsgn = sc.body == B_THIGH_R ? -1 : (sc.body == B_THIGH_L ? 1 : 1);
    const double side = wsgn * std::atan2(up[0] * lx + up[1] * ly, up[2]) * kRad;
    const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
    const double yaw = std::atan2(Rc[3], Rc[0]) * kRad;  // chest heading: + = left shoulder back
    // Pose in the sagittal plane (+ forward, deg): line pelvis -> neck from the vertical; spine = bend of the
    // lines pelvis -> mid-spine (chest joint) and mid-spine -> neck; neck = mid-spine -> neck against neck -> crown;
    // head = neck -> crown in the world (crown = 0.13 body height up the head axis)
    double pose[4];
    {
      const double* Rh = d->xmat + 9 * (b.body0 + B_HEAD);
      const double hh = 0.13 * b.body.height;
      const double cr[3] = {H[0] + hh * Rh[2], H[1] + hh * Rh[5], H[2] + hh * Rh[8]};
      auto ang = [&](const double* a, const double* c) {
        return std::atan2((c[0] - a[0]) * fx + (c[1] - a[1]) * fy, c[2] - a[2]) * kRad;
      };
      const double a1 = ang(P, C), a2 = ang(C, H), a3 = ang(H, cr);
      pose[0] = ang(P, H); pose[1] = a2 - a1; pose[2] = a3 - a2; pose[3] = a3;
    }
    if (ts <= 1e-9) { tr0 = trunk; bd0 = bend; sd0 = side; yaw0 = yaw; for (int k = 0; k < 4; ++k) pose0[k] = pose[k]; }
    if (r.t_fell < 0 && (P[2] < 0.5 || b.mode == 2)) r.t_fell = ts;
    if (ti < kNT && ts >= kT[ti] - 1e-9) {
      const int bd = b.body0 + sc.body;
      const double* Rb = d->xmat + 9 * bd;
      const double* xb = d->xpos + 3 * bd;
      double wp[3];
      for (int a = 0; a < 3; ++a) wp[a] = xb[a] + Rb[3 * a] * wl[0] + Rb[3 * a + 1] * wl[1] + Rb[3 * a + 2] * wl[2];
      const double* hl = d->xpos + 3 * (b.body0 + B_HAND_L);
      const double* hr = d->xpos + 3 * (b.body0 + B_HAND_R);
      const double* q = d->qpos + b.qadr0;
      double* v = r.v[ti];
      v[0] = 100 * (P[2] - z0);
      v[1] = std::fabs(q[J_KNEE_L]) * kRad;
      v[2] = std::fabs(q[J_KNEE_R]) * kRad;
      v[3] = trunk - tr0;
      v[4] = bend - bd0;
      v[5] = side - sd0;
      v[6] = 100 * hl[2];
      v[7] = 100 * hr[2];
      v[8] = 100 * std::sqrt((hl[0] - wp[0]) * (hl[0] - wp[0]) + (hl[1] - wp[1]) * (hl[1] - wp[1]) + (hl[2] - wp[2]) * (hl[2] - wp[2]));
      v[9] = 100 * std::sqrt((hr[0] - wp[0]) * (hr[0] - wp[0]) + (hr[1] - wp[1]) * (hr[1] - wp[1]) + (hr[2] - wp[2]) * (hr[2] - wp[2]));
      v[10] = b.td_cnt - td0;
      v[11] = 100 * std::hypot(P[0] - p0[0], P[1] - p0[1]);
      v[12] = std::remainder(yaw - yaw0, 360.0);
      for (int k = 0; k < 4; ++k) v[13 + k] = pose[k] - pose0[k];
      ++ti;
    }
  }
  return r;
}

}  // namespace

int RunBul7Bench(int argc, char** argv) {
  int threads = argc > 2 ? std::atoi(argv[2]) : 3;
  threads = std::max(1, std::min(threads, 4));
  std::vector<B7Res> res(kNS);
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&]() { for (int i; (i = next++) < kNS;) res[i] = RunB7(kB7[i]); });
  for (auto& x : th) x.join();
  for (int i = 0; i < kNS; ++i) {
    printf("B7 %s (fell at %.2f s)\nB7 t     drop kneeL kneeR trunk  bend  side  hLz  hRz   dL   dR st travel  yaw\n", kB7[i].name,
           res[i].t_fell);
    for (int k = 0; k < kNT; ++k) {
      const double* v = res[i].v[k];
      printf("B7 %3.1f %6.1f %5.0f %5.0f %5.1f %5.1f %5.1f %4.0f %4.0f %4.0f %4.0f %2.0f %6.0f %5.1f\n", kT[k], v[0], v[1], v[2],
             v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]);
    }
    printf("B7P %s pose [deg, + forward, from the stance]: t line spine neck head kneeL kneeR\n", kB7[i].name);
    for (int k = 0; k < kNT; ++k) {
      const double* v = res[i].v[k];
      printf("B7P %3.1f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f\n", kT[k], v[13], v[14], v[15], v[16], v[1], v[2]);
    }
  }
  return 0;
}
