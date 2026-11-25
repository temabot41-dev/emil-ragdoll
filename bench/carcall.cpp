// Car call bench, as the editor calls it. emil_bench carcall <kmh> <side> <start m> [wait s] [-p ...]
// Same build as the Godot editor (one bot, 1.78 m, 80 kg, with_car), the bot stands `wait` s after the reset (default
// 2 s, the avoid bench), then World::Car(0, kmh / 3.6, side, start) as EmilWorld.car(kmh, side, start). Every 0.1 s:
// the time from the call, the CoM's gap to the car front, the lateral distance of the CoM and of both feet from the
// car's centre line (car half width 0.9 m), pelvis height, dodge state and the time left at its decision; the first
// contact of the car with a body (any body, arms included) with its time.
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

int RunCarCall(int argc, char** argv) {
  if (argc < 5) { std::printf("usage: emil_bench carcall <kmh> <side 1 left, 0 front, -1 right> <start m> [wait s]\n"); return 1; }
  const double kmh = std::atof(argv[2]), start = std::atof(argv[4]);
  const int side = std::atoi(argv[3]);
  const double wait = argc > 5 && argv[5][0] != '-' ? std::atof(argv[5]) : 2.0;
  World w;
  w.with_car = true;
  std::vector<BodyParams> bp(1);
  char err[512];
  if (!w.Build(bp, 1.5, 1, err, sizeof err)) { std::printf("build failed: %s\n", err); return 1; }
  ApplyOverrides(w);
  w.Reset();
  for (int i = 0; i < (int)std::lround(wait * 60); ++i) w.StepFrame();
  const mjModel* m = w.model();
  const mjData* d = w.data();
  const Bot& b = w.bot(0);
  w.Car(0, kmh / 3.6, side, start);
  const int cb = mj_name2id(m, mjOBJ_BODY, "car");
  const double ux = side == 0 ? -1.0 : 0.0, uy = side == 0 ? 0.0 : (side > 0 ? -1.0 : 1.0);
  std::printf("carcall %.0f km/h side %d start %.2f m after %.2f s standing\n", kmh, side, start, wait);
  int hit = 0;
  for (int f = 1; f <= 240; ++f) {
    w.StepFrame();
    const double* Cc = d->xpos + 3 * cb;
    const double* c = d->subtree_com + 3 * b.body0;
    auto lat = [&](const double* p) { return -(p[0] - Cc[0]) * uy + (p[1] - Cc[1]) * ux; };
    for (int i = 0; i < d->ncon && !hit; ++i) {
      const int b1 = m->geom_bodyid[d->contact[i].geom1], b2 = m->geom_bodyid[d->contact[i].geom2];
      if (b1 != cb && b2 != cb) continue;
      const int rel = (b1 == cb ? b2 : b1) - b.body0;
      if (rel >= 0 && rel < B_COUNT) { hit = 1; std::printf("CONTACT t %.2f body %d\n", f / 60.0, rel); }
    }
    if (f % 6) continue;
    const double gap = (c[0] - Cc[0]) * ux + (c[1] - Cc[1]) * uy - 2.25;
    const double* q = d->qpos + b.qadr0;
    std::printf("t %.2f gap %6.2f com %6.2f footL %6.2f footR %6.2f pelvis_z %.2f dodge %d tc %.2f | "
                "shoulder %3.0f %3.0f elbow %3.0f %3.0f deg\n", f / 60.0, gap, lat(c), lat(d->xpos + 3 * (b.body0 + B_FOOT_L)),
                lat(d->xpos + 3 * (b.body0 + B_FOOT_R)), d->xpos[3 * b.body0 + 2], b.dodge, b.dodge_tc,
                q[J_SH_FLEX_L] * 57.3, q[J_SH_FLEX_R] * 57.3, q[J_ELBOW_L] * 57.3, q[J_ELBOW_R] * 57.3);
  }
  if (!hit) std::printf("CONTACT none\n");
  return 0;
}
