// Avoidance bench: car dodge (car 18/30/50 km/h from the front and the side, far start so there is time) and
// bullets that miss the head (flinch, gaze). One line per scene; AV_ONLY=<prefix> filter, AV_TRACE=<name> trace.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "emil_world.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p name=value

namespace {
std::mutex g_mu;

struct AScene {
  std::string name;
  bool car = true;
  double v = 0;     // car speed [m/s]
  int side = 0;     // car: 0 front, +1 left, -1 right
  double tc = 2.0;  // car: time to the first contact at the start [s]
  double ang = 0;   // miss: direction of the shooter (deg from the bot's front, + left)
  double off = 0;   // miss: the bullet passes this far beside the head centre [m] (+ left)
  double dist = 10; // miss: shooter distance [m]
};

struct AResult {
  double ft = 9;  // rear foot, lateral distance from the centre line toward the chest side, until the car touched the body
  double face = 9;  // chest forward . (toward the car) at the body touch: 1 face-on, 0 side, -1 back
  int fell = 0, hit = 0, hitb = 0, dodge = 0;  // hit: any touch of the car; hitb: touch by anything but the arms
  double t_hit = -1, lat_end = 0, sl_hit = 0, neck_max = 0, trunk_max = 0, err_gaze = 0, t_gaze = -1;
  double neck_pitch = 0, roll = 0, pelv = 0, err_seen = 0, neck_seen = 0, trunk_seen = 0;
  int steps = 0;
};

double Wrap(double a) { while (a > M_PI) a -= 2 * M_PI; while (a < -M_PI) a += 2 * M_PI; return a; }
double Yaw(const double* R) { return std::atan2(R[3], R[0]); }

AResult Run(const AScene& sc) {
  AResult r;
  World w;
  w.with_car = sc.car;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!w.Build({BodyParams()}, 2.0, std::getenv("AV_SEED") ? (uint64_t)std::atoi(std::getenv("AV_SEED")) : 1, err, sizeof err)) { r.fell = 1; return r; }
  }
  ApplyOverrides(w);
  const mjModel* m = w.model();
  mjData* d = w.data();
  Bot& b = w.bot(0);
  const bool trace = std::getenv("AV_TRACE") && sc.name == std::getenv("AV_TRACE");
  const int pre = 120;
  for (int f = 0; f < pre; ++f) w.StepFrame();
  const double p0[2] = {d->xpos[3 * b.body0], d->xpos[3 * b.body0 + 1]};
  double ev[3] = {0, 0, 0};  // shooter position
  double t_ev = 0;
  if (sc.car) {
    const double D = 2.25 + 0.2 + sc.v * sc.tc;
    w.Car(0, sc.v, sc.side, D);
  }
  const int frames = sc.car ? 300 : 240;
  const int cb = sc.car ? mj_name2id(m, mjOBJ_BODY, "car") : -1;
  bool shot = false;
  double prev_pz = 0;
  (void)prev_pz;
  int last_st = 0;
  for (int fr = 0; fr < frames; ++fr) {
    const double t = fr * World::kFrameDt;
    if (!sc.car && !shot && fr == 30) {
      shot = true;
      t_ev = t;
      const double* R = d->xmat + 9 * b.body0;
      const double hn = std::hypot(R[0], R[3]);
      const double hx = R[0] / hn, hy = R[3] / hn, lx = -hy, ly = hx;
      const double* h = d->xipos + 3 * (b.body0 + B_HEAD);
      const double a = sc.ang * M_PI / 180.0;
      const double dx = std::cos(a) * hx + std::sin(a) * lx, dy = std::cos(a) * hy + std::sin(a) * ly;  // to the shooter
      // the line passes sc.off to the left (pelvis frame, perpendicular to dx,dy) of the head centre
      const double px = -dy, py = dx;  // left of the shooter-to-bot direction seen from the bot (perp of dx,dy)
      ev[0] = h[0] + dx * sc.dist - px * sc.off;
      ev[1] = h[1] + dy * sc.dist - py * sc.off;
      ev[2] = h[2];
      const double dir[3] = {-dx, -dy, 0};  // parallel to the line shooter-head: passes sc.off beside the head
      w.Miss(0, ev, dir);
    }
    w.StepFrame();
    const double* R = d->xmat + 9 * b.body0;
    const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
    const double* Rh = d->xmat + 9 * (b.body0 + B_HEAD);
    const double* hd = d->xipos + 3 * (b.body0 + B_HEAD);
    const double neck = Wrap(Yaw(Rh) - Yaw(Rc)), trunk = Wrap(Yaw(Rc) - Yaw(R));
    if (!r.hit) {
      r.neck_max = std::fmax(r.neck_max, std::fabs(neck));
      r.trunk_max = std::fmax(r.trunk_max, std::fabs(trunk));
    }
    r.neck_pitch = std::fmax(r.neck_pitch, d->qpos[b.qadr0 + J_NECK_PITCH]);
    r.roll = std::fmax(r.roll, std::fabs(d->qpos[b.qadr0 + J_LUMBAR_ROLL]) + std::fabs(d->qpos[b.qadr0 + J_CHEST_ROLL]));
    if (d->xpos[3 * b.body0 + 2] < 0.5 && !r.hitb) r.fell = 1;  // a fall before the car touched
    if (b.steps != last_st) { last_st = b.steps; }
    double sl = 0, gap = 0, bearing_err = 0;
    if (sc.car) {
      const double* Cc = d->xpos + 3 * cb;
      const double* Ch = d->xipos + 3 * (b.body0 + B_CHEST);
      double ux = 0, uy = 0;
      if (sc.side == 0) { ux = -1; } else { uy = -sc.side; }
      const double rx = Ch[0] - Cc[0], ry = Ch[1] - Cc[1];
      gap = rx * ux + ry * uy - 2.25;
      sl = -rx * uy + ry * ux;
      if (!r.hitb) {
        double m = 9;
        for (int fo = 0; fo < 2; ++fo) {
          const double* Fp = d->xpos + 3 * (b.body0 + (fo ? B_FOOT_R : B_FOOT_L));
          m = std::fmin(m, (sl >= 0 ? 1.0 : -1.0) * (-(Fp[0] - Cc[0]) * uy + (Fp[1] - Cc[1]) * ux));
        }
        r.ft = m;
      }
      if (!r.hitb) {
        for (int i = 0; i < d->ncon; ++i) {
          const int b1 = m->geom_bodyid[d->contact[i].geom1], b2 = m->geom_bodyid[d->contact[i].geom2];
          if (b1 != cb && b2 != cb) continue;
          const int rel = (b1 == cb ? b2 : b1) - b.body0;
          if (rel >= 0 && rel < B_COUNT && (rel < B_UARM_L || rel > B_HAND_R)) {
            const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
            const double fn = std::hypot(Rc[0], Rc[3]) + 1e-9;
            r.face = -(Rc[0] * ux + Rc[3] * uy) / fn;
          }
          if (rel >= 0 && rel < B_COUNT && (rel < B_UARM_L || rel > B_HAND_R)) { r.hitb = 1; if (std::getenv("AV_HITB")) std::fprintf(stderr, "hitb body %d t %.2f sl %.2f\n", rel, t, sl); break; }
        }
      }
      if (!r.hit) {
        for (int i = 0; i < d->ncon; ++i) {
          const int b1 = m->geom_bodyid[d->contact[i].geom1], b2 = m->geom_bodyid[d->contact[i].geom2];
          if (b1 == cb || b2 == cb) { r.hit = 1; r.t_hit = t; r.sl_hit = sl; break; }
        }
      }
      if (!r.hit) { r.lat_end = sl; }
      if (b.dodge == 1 || b.dodge == 3) r.dodge = 1;
      // head bearing to the car front (error to the head heading), valid while the car is ahead
      const double fx = Cc[0] + ux * 2.25, fy = Cc[1] + uy * 2.25;
      bearing_err = Wrap(std::atan2(fy - hd[1], fx - hd[0]) - Yaw(Rh));
      if (b.car_seen_t >= 0 && r.t_gaze < 0 && std::fabs(bearing_err) < 0.2 && t > b.car_seen_t) r.t_gaze = t - b.car_seen_t;
      if (!r.hit && gap < 0.6 * sc.v + 0.3 && gap > 0.0) { r.err_seen = std::fabs(bearing_err); }  // ~0.6 s before the hit
    } else if (t >= t_ev + 1.0 && r.err_gaze == 0) {
      r.err_gaze = std::fabs(Wrap(std::atan2(ev[1] - hd[1], ev[0] - hd[0]) - Yaw(Rh)));
      r.t_gaze = r.err_gaze;
    }
    if (trace && fr % 6 == 0) {
      double ex = 0, ey = 1;  // lateral axis of the car scene (sl = r . e), positions relative to the start
      if (sc.car) { ex = sc.side == 0 ? 0.0 : 1.0; ey = sc.side == 0 ? -1.0 : 0.0; }
      const double* cm = w.bot(0).pc[w.bot(0).cur].com;
      const double* cvv = w.bot(0).pc[w.bot(0).cur].com_vel;
      const double* aL = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L);
      const double* aR = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_R);
      printf("   sw %d ph %d steps %d lat: ankL %.2f ankR %.2f com %.2f v %.2f tg %.2f rem %.2f zL %.2f zR %.2f vg %.2f\n", b.swing, b.phase, b.steps,
             (aL[0] - p0[0]) * ex + (aL[1] - p0[1]) * ey, (aR[0] - p0[0]) * ex + (aR[1] - p0[1]) * ey,
             (cm[0] - p0[0]) * ex + (cm[1] - p0[1]) * ey, cvv[0] * ex + cvv[1] * ey,
             (b.sw_tg[0] - p0[0]) * ex + (b.sw_tg[1] - p0[1]) * ey, b.sw_rem, aL[2], aR[2], std::hypot(b.in.vgoal[0], b.in.vgoal[1]));
      printf("gu %d ", b.dbg_gu); printf("t %.2f mode %d dodge %d sl %.2f gap %.2f neck %.0f trunk %.0f nq %.0f roll %.1f pelvx %.2f y %.2f z %.2f lean %.2f %.2f err %.0f\n", t, b.mode,
             b.dodge, sl, gap, neck * 57.3, trunk * 57.3, d->qpos[b.qadr0 + J_NECK_PITCH] * 57.3,
             d->qpos[b.qadr0 + J_CHEST_ROLL] * 57.3, d->xpos[3 * b.body0] - p0[0], d->xpos[3 * b.body0 + 1] - p0[1],
             d->xpos[3 * b.body0 + 2], b.in.lean[0], b.in.lean[1], bearing_err * 57.3);
    }
    if (sc.car && r.hit && t > r.t_hit + 2.5) break;
    r.pelv = std::hypot(d->xpos[3 * b.body0] - p0[0], d->xpos[3 * b.body0 + 1] - p0[1]);
  }
  r.steps = b.steps;
  return r;
}
}  // namespace

int RunAvoidBench(int threads) {
  std::vector<AScene> sc;
  for (double kmh : {18.0, 30.0, 50.0})
    for (int side : {0, 1})
      for (double tc : {2.0, 1.3, 0.9, 0.4}) {
        AScene s;
        char nm[48];
        std::snprintf(nm, sizeof nm, "car%d_%s_tc%.1f", (int)kmh, side ? "side" : "front", tc);
        s.name = nm; s.v = kmh / 3.6; s.side = side; s.tc = tc;
        sc.push_back(s);
      }
  const struct { const char* n; double ang, off, dist; } M[] = {
      {"miss_front_15", 0, 0.15, 10}, {"miss_front_40", 0, 0.4, 10}, {"miss_front_80", 0, 0.8, 10},
      {"miss_left_40", 90, 0.4, 10},  {"miss_back_40", 180, 0.4, 10}, {"miss_back_far", 180, 2.5, 20},
      {"miss_right_40", -90, 0.4, 10}, {"miss_fl45_40", 45, 0.4, 10}};
  for (auto& x : M) {
    AScene s; s.name = x.n; s.car = false; s.ang = x.ang; s.off = x.off; s.dist = x.dist;
    sc.push_back(s);
  }
  if (const char* only = std::getenv("AV_ONLY")) {
    std::vector<AScene> k;
    for (const AScene& x : sc) if (!std::strncmp(x.name.c_str(), only, std::strlen(only))) k.push_back(x);
    sc = k;
  }
  std::vector<AResult> res(sc.size());
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&] { for (int i; (i = next++) < (int)sc.size();) res[i] = Run(sc[i]); });
  for (auto& t : th) t.join();
  printf("%-20s fell hit t_hit dodge lat_end sl_hit neckmax trunkmax nPitch roll  gazeErr steps pelv hitb ftmin face\n", "scene");
  for (size_t i = 0; i < sc.size(); ++i) {
    const AResult& r = res[i];
    printf("%-20s %4d %3d %5.2f %5d %7.2f %6.2f %7.0f %8.0f %6.0f %5.1f %7.2f %5d %5.2f %4d %5.2f %5.2f\n", sc[i].name.c_str(), r.fell, r.hit,
           r.t_hit, r.dodge, r.lat_end, r.sl_hit, r.neck_max * 57.3, r.trunk_max * 57.3, r.neck_pitch * 57.3,
           r.roll * 57.3, sc[i].car ? r.err_seen * 57.3 : r.err_gaze * 57.3, r.steps, r.pelv, r.hitb, r.ft, r.face);
  }
  return 0;
}
