// Upper-body twist after an off-centre bullet: emil_bench twist8 [threads] [-p ...]
// Bullets from the front into the left shoulder (19 cm to the side), the left half of the chest (10 cm), the left side
// of the belly (10 cm) and the chest centre. Per scene: shoulder-line turn from the stance [deg, - = left shoulder back]
// at fixed times, the pelvis turn, the steps; peak and its time, the largest turn speed back (after the peak).
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
struct T8Scene { const char* name; int body; double z, side; };
constexpr int kN8 = 4;
const T8Scene kT8[kN8] = {{"shoulderL19", B_CHEST, 1.42, 0.19}, {"chestL10", B_CHEST, 1.30, 0.10},
                          {"bellyL10", B_LUMBAR, 1.05, 0.10}, {"chest3", B_CHEST, 1.39, 0.03}};
constexpr int kNT = 9;
const double kT[kNT] = {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0};

struct T8Res {
  double line[kNT] = {}, pel[kNT] = {}, tw[kNT] = {}, peak = 0, t_peak = 0, back_v = 0, at2 = 0;
  int steps = 0;
};

T8Res RunT8(const T8Scene& sc) {
  T8Res r;
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
  double t0 = -1, fx = 1, fy = 0, l0 = 0, p0 = 0, prev = 0;
  int td0 = 0, ti = 0;
  const int frames = (int)std::lround((kMoment + 2.05) * 60.0);
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* SL = d->xpos + 3 * (b.body0 + B_UARM_L);
    const double* SR = d->xpos + 3 * (b.body0 + B_UARM_R);
    const double* Rp = d->xmat + 9 * b.body0;
    const double lh = std::atan2(SL[1] - SR[1], SL[0] - SR[0]) * kRad;  // shoulder line heading (R -> L)
    const double ph = std::atan2(Rp[3], Rp[0]) * kRad;
    if (t0 < 0) {
      if (t < kMoment) continue;
      const double hn = std::hypot(Rp[0], Rp[3]) + 1e-9;
      fx = Rp[0] / hn; fy = Rp[3] / hn;
      t0 = t; l0 = lh; p0 = ph; td0 = b.td_cnt;
      const double lx = -fy, ly = fx;
      const double* c = d->xipos + 3 * (b.body0 + sc.body);
      const double pt[3] = {c[0] + 0.1 * fx + sc.side * lx, c[1] + 0.1 * fy + sc.side * ly, sc.z * b.body.height / 1.78};
      const double dv[3] = {-fx, -fy, 0};
      w.Shoot(0, sc.body, pt, dv);
      continue;
    }
    const double ts = t - t0;
    const double line = -std::remainder(lh - l0, 360.0), pel = -std::remainder(ph - p0, 360.0);
    if (std::fabs(line) > std::fabs(r.peak)) { r.peak = line; r.t_peak = ts; }
    if (ts > r.t_peak + 0.02 && r.peak != 0) {  // speed back toward the stance after the peak [deg/s]
      const double v = (prev - line) * 60.0 * (r.peak > 0 ? 1 : -1);
      r.back_v = std::max(r.back_v, v);
    }
    prev = line;
    const double* q = d->qpos + b.qadr0;
    const double tw = -(q[J_LUMBAR_YAW] + q[J_CHEST_YAW]) * kRad;  // spine twist, same sign
    while (ti < kNT && ts >= kT[ti] - 1e-9) { r.line[ti] = line; r.pel[ti] = pel; r.tw[ti] = tw; ++ti; }
    r.at2 = line;
    r.steps = b.td_cnt - td0;
  }
  return r;
}

}  // namespace

int RunTwist8Bench(int argc, char** argv) {
  int threads = argc > 2 ? std::atoi(argv[2]) : 3;
  threads = std::max(1, std::min(threads, 3));
  std::vector<T8Res> res(kN8);
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&]() { for (int i; (i = next++) < kN8;) res[i] = RunT8(kT8[i]); });
  for (auto& x : th) x.join();
  printf("T8 shoulder line turn [deg, - = left shoulder back] at   0.05  0.1 0.15  0.2  0.3  0.4  0.6  0.8  1.0 | peak "
         "at | back speed | at 2 s | steps\n");
  for (int i = 0; i < kN8; ++i) {
    const T8Res& r = res[i];
    printf("T8 %-12s line", kT8[i].name);
    for (int k = 0; k < kNT; ++k) printf(" %4.0f", r.line[k]);
    printf(" | %4.0f %.2f | %4.0f deg/s | %4.0f | %d\nT8 %-12s pelvis", r.peak, r.t_peak, r.back_v, r.at2, r.steps,
           kT8[i].name);
    for (int k = 0; k < kNT; ++k) printf(" %4.0f", r.pel[k]);
    printf("\nT8 %-12s twist ", kT8[i].name);
    for (int k = 0; k < kNT; ++k) printf(" %4.0f", r.tw[k]);
    printf("\n");
  }
  return 0;
}
