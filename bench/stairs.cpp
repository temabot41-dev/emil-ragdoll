// Stairs bench: pushed down the stairs (World::with_stairs).
//   emil_bench stairs [impulse ...] [-p k=v ...]    default impulses 100 120 140 [N s]; one thread each
// Bot 0 stands on the landing 0.3 m before the first edge facing down (+x); at 2 s a push from behind at the chest
// (impulse along +x). Per run, from the moment he loses balance (pelvis-head line over 45 deg from the vertical or the
// balancer judges the fall: fall_pred or mode 2) to the bottom (pelvis past the last riser and below 0.5 m over the floor):
//   tumble  = time moving (pelvis > 0.5 m/s or > 2 rad/s) with both feet unloaded or the body over 60 deg [s];
//   roll = time moving over 60 deg [s]; stop = pelvis x and height over the floor where he came to rest (1 s < 0.2 m/s)
//   stepair = share of the descent with a step commanded while tumbling (target 0); steps = step starts while tumbling
//   curl    = share of the roll time with mean hip flexion and mean knee flexion both over 60 deg
//   plank   = share of the roll time with both under 25 deg (straight body)
//   hand-head = mean distance of the nearer hand to the head while rolling [m] (arms near the head: protection)
//   head_v  = head speed peak [m/s]; bottom = time from the push to the bottom [s]
//   after   = pelvis travel and step starts in the 2 s after the bottom (lies as he landed: ~0)
// Env: ST_TR=<impulse> prints a line every 0.1 s for that run; ST_NR=1 no protective reaction (L2 no_react).
// ST_V=1 steep stairs scene: risers 0.18 x treads 0.28 (32.7 deg), push at 1.0 s at 0.45 m above the
// pelvis; ST_PV=<impulse> prints the pelvis x / height over the floor every 0.25 s of world time (t = 1.97, 2.22, ..).
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

namespace {

constexpr double kRad = 57.29578;

struct StRes {
  double J = 0, t_lost = -1, t_bot = -1, tumble = 0, roll = 0, step_air = 0, curl = 0, plank = 0, hh_sum = 0;
  double head_v = 0, x5 = -1, x8 = -1, after_d = 0, desc = 0, stop_x = -1, stop_z = 0, still = 0;
  int steps_air = 0, steps_after = 0, mode_end = 0;
  std::string log;
};

void RunStairs(double J, StRes* r) {
  r->J = J;
  std::vector<BodyParams> bodies(1);
  World w;
  w.with_stairs = true;
  const bool vid = getenv("ST_V") && std::atoi(getenv("ST_V"));
  if (vid) w.stair_rise = 0.18, w.stair_run = 0.28;  // steep stairs scene
  w.stairs_slant = getenv("ST_WALL") && std::atoi(getenv("ST_WALL"));  // editor walls
  w.stairs_body = getenv("ST_BODY") && std::atoi(getenv("ST_BODY"));  // stairs in a static body (editor scene)
  char err[512];
  {
    std::lock_guard<std::mutex> lk(stag::BuildMutex());
    if (!w.Build(bodies, 3.0, 1, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return; }
  }
  ApplyOverrides(w);
  w.Reset();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  // ST_GZ0=1 the stand floor starts at 0 as in the editor (spawned onto the landing with lift 0)
  if (getenv("ST_GZ0") && std::atoi(getenv("ST_GZ0"))) w.bot(0).gz = w.bot(0).gz_st = 0;
  if (getenv("ST_NR")) w.bot(0).no_react = std::atoi(getenv("ST_NR"));  // baseline: no protective reaction
  const bool tr = getenv("ST_TR") && std::fabs(std::atof(getenv("ST_TR")) - J) < 1e-6;
  const double kPush = getenv("ST_PT") ? std::atof(getenv("ST_PT")) : vid ? 1.0 : 2.0;  // ST_PT: push moment [s]
  const bool pvo = getenv("ST_PV") && std::fabs(std::atof(getenv("ST_PV")) - J) < 1e-6;
  const double kMax = 16.0, dt = 1.0 / 60;
  const double zf = w.FloorZ(), xb = w.StairsBottomX();
  double pb[2] = {0, 0}, hp[3] = {0, 0, 0};
  int st_prev = 0;
  bool pushed = false;
  char buf[400];
  for (int fr = 0;; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    const double* H = d->xpos + 3 * (b.body0 + B_HEAD);
    if (!pushed && t >= kPush) {
      pushed = true;
      const double* C = d->xpos + 3 * (b.body0 + B_CHEST);
      double pt[3] = {C[0] - 0.1, C[1], C[2] + 0.06};
      if (vid) pt[0] = C[0], pt[1] = C[1], pt[2] = P[2] + 0.45;  // steep stairs: chest point 0.45 m over pelvis
      const double imp[3] = {J, 0, 0};
      int hb = b.body0 + B_CHEST;
      if (getenv("ST_RAY")) {  // editor tool: a ray 3 m behind along +x at pelvis height + 0.45 m,
        // ST_RAY = its sideways offset from the pelvis [m]; the impulse at the hit point
        const double p0[3] = {P[0] - 3.0, P[1] + std::atof(getenv("ST_RAY")), P[2] + 0.45}, dv[3] = {1, 0, 0};
        double hit[3];
        int body = -1;
        if (w.Ray(p0, dv, &body, hit) >= 0) hb = body, pt[0] = hit[0], pt[1] = hit[1], pt[2] = hit[2];
      }
      w.PushAt(0, imp, pt, hb - b.body0);  // at the chest / ray point above
      st_prev = b.steps;
    }
    if (pvo && (fr + 1) % 15 == 13) {
      snprintf(buf, sizeof buf, "PV %5.2f %5.2f %5.2f\n", t, P[0], P[2] - zf);
      r->log += buf;
    }
    if (!pushed) { for (int k = 0; k < 3; ++k) hp[k] = H[k]; continue; }
    const double ts = t - kPush;
    if (r->x5 < 0 && ts >= 5.0) r->x5 = P[0];  // pelvis x 5 s / 8 s after the push
    if (r->x8 < 0 && ts >= 8.0) r->x8 = P[0];
    const double up[3] = {H[0] - P[0], H[1] - P[1], H[2] - P[2]};
    const double tilt = std::acos(std::fmax(-1.0, std::fmin(1.0, up[2] / (std::sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]) + 1e-9)))) * kRad;
    const double* q = d->qpos + b.qadr0;
    const double hip = 0.5 * (q[J_HIP_FLEX_L] + q[J_HIP_FLEX_R]) * kRad, knee = 0.5 * (q[J_KNEE_L] + q[J_KNEE_R]) * kRad;
    const double hv = std::sqrt((H[0] - hp[0]) * (H[0] - hp[0]) + (H[1] - hp[1]) * (H[1] - hp[1]) + (H[2] - hp[2]) * (H[2] - hp[2])) / dt;
    for (int k = 0; k < 3; ++k) hp[k] = H[k];
    const bool feet_off = !b.loaded[0] && !b.loaded[1];
    const double* cv = d->cvel + 6 * (b.body0 + B_PELVIS);
    const double pw = std::sqrt(cv[0] * cv[0] + cv[1] * cv[1] + cv[2] * cv[2]), pv = std::sqrt(cv[3] * cv[3] + cv[4] * cv[4] + cv[5] * cv[5]);
    const bool moving = pv > 0.5 || pw > 2.0;
    const int new_steps = b.steps - st_prev;
    st_prev = b.steps;
    if (r->t_lost < 0 && (tilt > 45 || b.mode == 2 || b.fall_pred)) r->t_lost = ts;
    r->still = moving || pv > 0.2 ? 0.0 : r->still + dt;
    if (r->t_lost >= 0 && r->t_bot < 0 && r->stop_x < 0 && r->still > 1.0) r->stop_x = P[0], r->stop_z = P[2] - zf;
    const bool desc = r->t_lost >= 0 && r->t_bot < 0 && r->stop_x < 0;
    if (desc) {
      r->desc += dt;
      r->head_v = std::fmax(r->head_v, hv);
      const bool tumb = (feet_off || tilt > 60) && moving;
      if (tumb) {
        r->tumble += dt;
        if (b.swing >= 0) r->step_air += dt;
        r->steps_air += new_steps;
      }
      if (tilt > 60 && moving) {
        r->roll += dt;
        if (hip > 60 && knee > 60) r->curl += dt;
        if (hip < 25 && knee < 25) r->plank += dt;
        double hh = 9;
        for (int s = 0; s < 2; ++s) {
          const double* A = d->xpos + 3 * (b.body0 + (s ? B_HAND_R : B_HAND_L));
          hh = std::fmin(hh, std::sqrt((A[0] - H[0]) * (A[0] - H[0]) + (A[1] - H[1]) * (A[1] - H[1]) + (A[2] - H[2]) * (A[2] - H[2])));
        }
        r->hh_sum += hh * dt;
      }
    }
    if (r->t_bot < 0 && P[0] > xb && P[2] < zf + 0.5) {
      r->t_bot = ts;
      pb[0] = P[0], pb[1] = P[1];
    }
    if (r->t_bot >= 0 && ts <= r->t_bot + 2.0) {
      r->after_d = std::hypot(P[0] - pb[0], P[1] - pb[1]);
      r->steps_after += new_steps;
    }
    if (tr && new_steps > 0) {
      snprintf(buf, sizeof buf, "STEP %5.2f foot %d ld %.2f %.2f tilt %3.0f md %d pred %d tumble %d feet z %+5.2f %+5.2f\n", ts, b.swing,
               b.ld[0], b.ld[1], tilt, b.mode, b.fall_pred, b.tumble, d->xpos[3 * (b.body0 + B_FOOT_L) + 2],
               d->xpos[3 * (b.body0 + B_FOOT_R) + 2]);
      r->log += buf;
    }
    if (tr && fr % 6 == 0) {
      snprintf(buf, sizeof buf,
               "ST %5.2f pel x %5.2f z %+5.2f tilt %3.0f hip %4.0f knee %4.0f ld %.2f %.2f sw %2d md %d air %d curl %d roll %.2f "
               "cw %.2f headv %4.1f w %4.1f lie %d pred %d tb %d\n",
               ts, P[0], P[2] - zf, tilt, hip, knee, b.ld[0], b.ld[1], b.swing, b.mode, b.airborne, b.curled, b.roll_w,
               b.curl_w, hv, std::sqrt(d->cvel[6 * (b.body0 + B_PELVIS)] * d->cvel[6 * (b.body0 + B_PELVIS)] +
                                   d->cvel[6 * (b.body0 + B_PELVIS) + 1] * d->cvel[6 * (b.body0 + B_PELVIS) + 1] +
                                   d->cvel[6 * (b.body0 + B_PELVIS) + 2] * d->cvel[6 * (b.body0 + B_PELVIS) + 2]),
               b.lie_t0 >= 0 ? 1 : 0, b.fall_pred, b.tumble);
      r->log += buf;
    }
    if (tr && fr % 6 == 0) {  // pelvis axes, head/feet offsets and ground contacts per part (head, trunk, arms, legs)
      const mjModel* m = w.model();
      const double* R = d->xmat + 9 * (b.body0 + B_PELVIS);
      const double* hp = d->xpos + 3 * (b.body0 + B_HEAD);
      const double* fl = d->xpos + 3 * (b.body0 + B_FOOT_L);
      int nc[4] = {0, 0, 0, 0};
      for (int i = 0; i < d->ncon; ++i) {
        const int b1 = m->geom_bodyid[d->contact[i].geom1], b2 = m->geom_bodyid[d->contact[i].geom2];
        const int own = b1 >= b.body0 && b1 < b.body0 + B_COUNT ? b1 : (b2 >= b.body0 && b2 < b.body0 + B_COUNT ? b2 : -1);
        const int oth = own == b1 ? b2 : b1;
        if (own < 0 || (oth >= b.body0 && oth < b.body0 + B_COUNT)) continue;
        const int k = own - b.body0;
        nc[k == B_HEAD ? 0 : k < B_HEAD ? 1 : k < B_THIGH_L ? 2 : 3]++;
      }
      snprintf(buf, sizeof buf, "STC %5.2f fwd %+5.2f %+5.2f %+5.2f left %+5.2f %+5.2f %+5.2f up %+5.2f | head dx %+5.2f dz %+5.2f"
               " | footL dx %+5.2f dz %+5.2f | con head %d trunk %d arms %d legs %d\n", ts, R[0], R[3], R[6], R[1], R[4], R[7],
               R[8], hp[0] - P[0], hp[2] - P[2], fl[0] - P[0], fl[2] - P[2], nc[0], nc[1], nc[2], nc[3]);
      r->log += buf;
    }
    if (ts > kMax || (r->t_bot >= 0 && ts > r->t_bot + 2.0) || (r->stop_x >= 0 && r->still > 3.0)) break;
  }
  r->mode_end = b.mode;
  const double xe = d->xpos[3 * (b.body0 + B_PELVIS)];  // run ended earlier (at rest or at the bottom): final x
  if (r->x5 < 0) r->x5 = xe;
  if (r->x8 < 0) r->x8 = xe;
}

}  // namespace

int RunStairsBench(int argc, char** argv) {
  std::vector<double> imps;
  for (int i = 2; i < argc; ++i) {
    if (argv[i][0] == '-') { ++i; continue; }  // -p k=v handled by ApplyOverrides
    imps.push_back(std::atof(argv[i]));
  }
  if (imps.empty()) imps = {100, 120, 140};
  std::vector<StRes> res(imps.size());
  std::vector<std::thread> th;
  for (size_t i = 0; i < imps.size(); ++i) th.emplace_back(RunStairs, imps[i], &res[i]);
  for (auto& x : th) x.join();
  for (const StRes& r : res) fputs(r.log.c_str(), stdout);
  printf("STAIRS    J  lost  bottom  desc tumble  roll | stepair steps | curl plank hand-head | head_v | after m steps | stop x  z | mode | x5 x8\n");
  for (const StRes& r : res) {
    const double rl = std::fmax(r.roll, 1e-9);
    printf("STAIRS %4.0f %5.2f %7.2f %5.2f %6.2f %5.2f | %7.2f %5d | %4.2f %5.2f %9.2f | %6.1f | %7.2f %5d | %5.2f %4.2f | %d | %5.2f %5.2f\n", r.J,
           r.t_lost, r.t_bot, r.desc, r.tumble, r.roll, r.desc > 0 ? r.step_air / r.desc : 0.0, r.steps_air, r.curl / rl,
           r.plank / rl, r.hh_sum / rl, r.head_v, r.after_d, r.steps_after, r.stop_x, r.stop_z, r.mode_end, r.x5, r.x8);
  }
  return 0;
}
