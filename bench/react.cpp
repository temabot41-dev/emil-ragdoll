// Reaction bench: bullets (12 scenes), falls (push 200/300 N s, reacting vs relaxed), drops from
// height, car hits, curl. CSV per scene (push format + chest, l_hand, r_hand) in $EMIL_TMP/rx/<label>/; one table.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "tmp_dir.h"
#include "emil_world.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p name=value

void WriteRowExtra(FILE* fp, const mjModel* m, const mjData* d, const Bot& b, double t);  // push.cpp

namespace {
std::mutex g_mu;

enum Kind { K_BULLET, K_FALL, K_RELAX, K_DROP, K_CAR, K_CURL, K_KILL };
struct Shot { double dt; int body; double dy, dz; };  // time after first shot, body, point offsets
struct RScene {
  const char* name;
  Kind kind;
  std::vector<Shot> shots;  // bullets
  double dx = 0, dy = 0, J = 0;  // fall push
  double h = 0;                  // drop height
  double v = 0; int side = 0;    // car
  int immortal = 0;              // param immortal
};

struct RResult {
  int fell = 0, swings = 0;
  double drift = 0, settle = -1;
  char first = '-';        // first floor touch: H hands, K knees, T trunk/head, '-' none
  double head_v = 0;       // head speed at first head-floor contact (0 = none)
  double vmax = 0;         // max part speed
  int through = 0, guard = 0;
  double t_pose = -1, kc = 0, jitter = 0;
  double t_m2 = -1, t_first = -1, t_head = -1;  // time of mode 2, first floor touch, first head touch
  int up = 0; double up_max = 0;  // limb parts (arms, legs) above 0.30 m after 4 s down, the highest
  double v_first = 0;      // speed of the touching part at the first floor touch
  // Bullet scenes: pelvis drop within 1.5 s after the first shot [m], largest trunk flexion (lumbar + chest pitch)
  // [deg], closest hand to the wound [m], back steps (pelvis moved along the bullet) [m]
  double drop = 0, flex = 0, hand_d = 9;
  double t_fell = -1, health = 0;  // time of the fall after the first shot [s], health at the end
  double path[5] = {0, 0, 0, 0, 0};  // pelvis path from the last shot at 0.4, 0.6, 0.8, 1.0, 2.0 s [m]
  double lie = 0;                    // pelvis distance from the start place at the end [m]
};

std::vector<RScene> Scenes() {
  std::vector<RScene> s;
  auto B = [&](const char* n, std::vector<Shot> sh) { RScene r{n, K_BULLET, sh}; s.push_back(r); };
  B("b_chest", {{0, B_CHEST, 0, 0.05}});
  B("b_belly", {{0, B_LUMBAR, 0, 0}});
  B("b_chest_x3", {{0, B_CHEST, 0, 0.05}, {1, B_CHEST, 0, 0.05}, {2, B_CHEST, 0, 0.05}});
  B("b_chest_belly", {{0, B_CHEST, 0, 0.05}, {1.5, B_LUMBAR, 0, 0}});
  B("b_belly_chest", {{0, B_LUMBAR, 0, 0}, {1.5, B_CHEST, 0, 0.05}});
  B("b_chest_lsh", {{0, B_CHEST, 0, 0.05}, {1.5, B_UARM_L, 0, 0}});
  B("b_chest_rthigh", {{0, B_CHEST, 0, 0.05}, {1.5, B_THIGH_R, 0, 0}});
  B("b_chest_belly_rsh", {{0, B_CHEST, 0, 0.05}, {1.2, B_LUMBAR, 0, 0}, {2.4, B_UARM_R, 0, 0}});
  B("b_lsh_bellyR", {{0, B_UARM_L, 0, 0}, {1.5, B_LUMBAR, -0.08, 0}});
  B("b_lthigh", {{0, B_THIGH_L, 0, 0}});
  B("b_lshin", {{0, B_SHANK_L, 0, 0}});
  B("b_rthigh", {{0, B_THIGH_R, 0, 0}});
  // Bullet scenes: one chest bullet from the front, a burst of 4 (0.15 s), one and four thigh bullets, head.
  B("w_chest1", {{0, B_CHEST, 0.06, 0.05}});
  // One chest bullet across the chest (dy [m] from the middle) and one in the belly; target: stands, <= 2 steps
  B("p_c0", {{0, B_CHEST, 0, 0.05}});
  B("p_cp3", {{0, B_CHEST, 0.03, 0.05}});
  B("p_cm3", {{0, B_CHEST, -0.03, 0.05}});
  B("p_cp6", {{0, B_CHEST, 0.06, 0.05}});
  B("p_cm6", {{0, B_CHEST, -0.06, 0.05}});
  B("p_cp10", {{0, B_CHEST, 0.10, 0.05}});
  B("p_cm10", {{0, B_CHEST, -0.10, 0.05}});
  B("p_belly", {{0, B_LUMBAR, 0, 0}});
  B("w_chest4", {{0, B_CHEST, 0.06, 0.05}, {0.15, B_CHEST, 0.06, 0.05}, {0.3, B_CHEST, 0.06, 0.05}, {0.45, B_CHEST, 0.06, 0.05}});
  B("w_chest2", {{0, B_CHEST, 0.06, 0.05}, {1.0, B_CHEST, 0.06, 0.05}});  // the 2nd bullet leans: steps back
  B("w_thigh1", {{0, B_THIGH_L, 0, 0}});
  B("w_thigh4", {{0, B_THIGH_L, 0, 0}, {0.4, B_THIGH_L, 0, 0}, {0.8, B_THIGH_L, 0, 0}, {1.2, B_THIGH_L, 0, 0}});
  B("w_head", {{0, B_HEAD, 0, 0.03}});
  // Health rules: n pistol bullets in the chest every dt; "im" = immortal
  for (int n : {1, 2, 3, 4, 5, 7, 12})
    for (double dt : {0.15, 1.0})
      for (int im : {0, 1}) {
        if ((im == 0 && n == 12) || (im == 1 && n < 7)) continue;
        static char names[64][24];
        static int nn = 0;
        char* nm = names[nn++ % 64];
        std::snprintf(nm, 24, "h_%d_%s%s", n, dt < 0.5 ? "f" : "s", im ? "_im" : "");
        std::vector<Shot> sh;
        for (int k = 0; k < n; ++k) sh.push_back({k * dt, B_CHEST, 0, 0.05});
        RScene r{nm, K_BULLET, sh};
        r.immortal = im;
        s.push_back(r);
      }
  B("h_head", {{0, B_HEAD, 0, 0.03}});
  { RScene k{"w_kill", K_KILL, {}}; s.push_back(k); }
  const struct { const char* n; double dx, dy, J; } F[] = {
      {"back200", -1, 0, 200}, {"back300", -1, 0, 300}, {"fwd200", 1, 0, 200},
      {"fwd300", 1, 0, 300},   {"left200", 0, 1, 200},  {"left300", 0, 1, 300}, {"fwd30", 1, 0, 30}, {"back30", -1, 0, 30}};
  for (auto& f : F) {
    RScene r{f.n, K_FALL, {}}; r.dx = f.dx; r.dy = f.dy; r.J = f.J; s.push_back(r);
    RScene q = r; q.kind = K_RELAX;
    static std::vector<std::string> names;  // stable storage for "relax_" names
    names.reserve(16);
    names.push_back(std::string("relax_") + f.n);
    q.name = names.back().c_str();
    s.push_back(q);
  }
  for (double h : {1.0, 2.0, 4.0}) {
    RScene r{h == 1.0 ? "drop1" : (h == 2.0 ? "drop2" : "drop4"), K_DROP, {}}; r.h = h; s.push_back(r);
  }
  const struct { const char* n; double v; int side; } C[] = {
      {"car18_side", 18, 1}, {"car18_front", 18, 0}, {"car50_side", 50, 1}, {"car50_front", 50, 0}};
  for (auto& c : C) { RScene r{c.n, K_CAR, {}}; r.v = c.v / 3.6; r.side = c.side; s.push_back(r); }
  RScene cu{"curl", K_CURL, {}}; s.push_back(cu);
  return s;
}

void Header(FILE* fp) {
  fprintf(fp, "t");
  const char* pts[] = {"pelvis", "head", "l_ankle", "r_ankle", "l_toe", "r_toe", "l_heel", "r_heel"};
  for (auto n : pts) fprintf(fp, ",%s_x,%s_y,%s_z", n, n, n);
  fprintf(fp, ",l_roll,r_roll,l_contact,r_contact,chest_x,chest_y,chest_z,l_hand_x,l_hand_y,l_hand_z,"
              "r_hand_x,r_hand_y,r_hand_z,l_knee,r_knee,l_hipf,r_hipf\n");
}

RResult RunR(const RScene& sc, const std::string& dir) {
  RResult r;
  World w;
  w.with_car = sc.kind == K_CAR;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) { r.fell = 1; return r; }
  }
  ApplyOverrides(w);
  if (sc.immortal) w.param[P_IMMORTAL] = 1;
  const mjModel* m = w.model();
  mjData* d = w.data();
  Bot& b = w.bot(0);
  const bool trace = std::getenv("RX_TRACE") && !std::strcmp(std::getenv("RX_TRACE"), sc.name);
  const double t_ev = 2.0;
  const int pre = (int)std::lround(t_ev / World::kFrameDt), pre_rec = 30;

  FILE* fp = fopen((dir + "/" + sc.name + ".csv").c_str(), "w");
  if (fp) Header(fp);
  for (int f = 0; f < pre; ++f) {
    w.StepFrame();
    if (f >= pre - pre_rec) WriteRowExtra(fp, m, d, b, (f + 1 - pre) * World::kFrameDt);
  }
  double rest_ank[2], rest_toe[2];
  auto toe_z = [&](int f) {
    const int g = b.geom_foot[f];
    const double* R = d->geom_xmat + 9 * g; const double* s = m->geom_size + 3 * g;
    return d->geom_xpos[3 * g + 2] + R[6] * s[0] - R[8] * s[2];
  };
  auto ank_z = [&](int f) { return d->xanchor[3 * (b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)) + 2]; };
  for (int f = 0; f < 2; ++f) { rest_ank[f] = ank_z(f); rest_toe[f] = toe_z(f); }
  const double p0[3] = {d->xpos[3 * b.body0], d->xpos[3 * b.body0 + 1], d->xpos[3 * b.body0 + 2]};
  int frames = sc.kind == K_CURL ? 600 : 360;
  if (!sc.shots.empty()) frames = std::max(frames, (int)std::lround(sc.shots.back().dt / World::kFrameDt) + 300);
  size_t next_shot = 0;
  double last_p[2] = {0, 0};
  int air[2] = {0, 0}, insw[2] = {0, 0};
  std::vector<double> prev(3 * B_COUNT);
  for (int k = 0; k < B_COUNT; ++k) for (int i = 0; i < 3; ++i) prev[3 * k + i] = d->xipos[3 * (b.body0 + k) + i];
  double last_move = 0, jit_s = 0; int jit_n = 0;
  bool head_hit = false;
  double hv = 0, hv_prev = 0;
  for (int fr = 0; fr <= frames; ++fr) {
    const double t = fr * World::kFrameDt;
    // Events at the start of the frame.
    if (sc.kind == K_BULLET && next_shot < sc.shots.size() && t >= sc.shots[next_shot].dt - 1e-9) {
      const Shot& s = sc.shots[next_shot++];
      const double* c = d->xipos + 3 * (b.body0 + s.body);
      const double pt[3] = {c[0] + 0.1, c[1] + s.dy, c[2] + s.dz}, dv[3] = {-1, 0, 0};
      w.Shoot(0, s.body, pt, dv);
    }
    if (fr == 0 && (sc.kind == K_FALL || sc.kind == K_RELAX || sc.kind == K_CURL)) {
      const double* pp = d->xpos + 3 * b.body0;
      const double pt[3] = {pp[0], pp[1], pp[2] + 0.35};
      const double J = sc.kind == K_CURL ? 200 : sc.J;
      const double imp[3] = {sc.kind == K_CURL ? -J : sc.dx * J, sc.dy * J, 0};
      w.PushAt(0, imp, pt, B_LUMBAR);  // standard push point
      if (sc.kind == K_RELAX) { b.no_react = 1; b.in.stiffness = 0.05; b.mode = 2; }
    }
    if (fr == 0 && sc.kind == K_CAR) w.Car(0, sc.v, sc.side);
    if (fr == 0 && sc.kind == K_KILL) w.Kill(0);
    if (fr > 0 && (sc.kind == K_BULLET || sc.kind == K_KILL)) {  // bullet metrics
      if (t <= 1.5) r.drop = std::fmax(r.drop, p0[2] - d->xpos[3 * b.body0 + 2]);
      if (t <= 1.5 && !r.fell)
        r.flex = std::fmax(r.flex, (d->qpos[b.qadr0 + J_LUMBAR_PITCH] + d->qpos[b.qadr0 + J_CHEST_PITCH]) * 57.29578);
      if (sc.kind == K_BULLET && sc.shots[0].body <= B_CHEST && !r.fell) {
        const double* cp = d->xpos + 3 * (b.body0 + B_CHEST);
        const double* R = d->xmat + 9 * (b.body0 + B_CHEST);
        double wp[3];
        for (int a = 0; a < 3; ++a) wp[a] = cp[a] + R[3 * a] * b.bul_c[0] + R[3 * a + 1] * b.bul_c[1] + R[3 * a + 2] * b.bul_c[2];
        for (int hb : {B_HAND_L, B_HAND_R}) {
          const double* h = d->xpos + 3 * (b.body0 + hb);
          r.hand_d = std::fmin(r.hand_d, std::sqrt((h[0] - wp[0]) * (h[0] - wp[0]) + (h[1] - wp[1]) * (h[1] - wp[1]) +
                                                   (h[2] - wp[2]) * (h[2] - wp[2])));
        }
      }
    }
    if (fr == 0 && sc.kind == K_DROP) w.DropFrom(sc.h);  // stand at h above the floor, then fall
    if (sc.kind == K_CURL && fr == 180) w.Curl(0, 1);
    if (fr > 0) {
      w.StepFrame();
      WriteRowExtra(fp, m, d, b, t);
    }
    // Swings and foot motion.
    for (int f = 0; f < 2; ++f) {
      const bool up = toe_z(f) > rest_toe[f] + 0.015 && ank_z(f) > rest_ank[f] + 0.015;
      if (up) { if (++air[f] == 3 && !insw[f]) { insw[f] = 1; if (!r.fell) r.swings++; } }  // not after a fall
      else { air[f] = 0; insw[f] = 0; }
    }
    // Part speeds, floor touches.
    double vb = 0, vfeet = 0;
    for (int k = 0; k < B_COUNT; ++k) {
      const double* x = d->xipos + 3 * (b.body0 + k);
      const double v = std::sqrt((x[0] - prev[3 * k]) * (x[0] - prev[3 * k]) + (x[1] - prev[3 * k + 1]) * (x[1] - prev[3 * k + 1]) +
                                 (x[2] - prev[3 * k + 2]) * (x[2] - prev[3 * k + 2])) / World::kFrameDt;
      if (fr > 0) { vb = std::fmax(vb, v); if (k == B_FOOT_L || k == B_FOOT_R) vfeet = std::fmax(vfeet, v); }
      if (k == B_HEAD) { hv_prev = hv; hv = v; }
      for (int i = 0; i < 3; ++i) prev[3 * k + i] = x[i];
    }
    if (fr > 0) r.vmax = std::fmax(r.vmax, vb);
    const double moving = sc.kind == K_BULLET ? vfeet : vb;
    if (moving > (sc.kind == K_BULLET ? 0.05 : 0.15)) last_move = t;
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
      if (g < 0) continue;
      const int bd = m->geom_bodyid[g] - b.body0;
      if (bd < 0 || bd >= B_COUNT) continue;
      const bool trunk = bd <= B_HEAD, hand = bd == B_HAND_L || bd == B_HAND_R || bd == B_FARM_L || bd == B_FARM_R;
      const bool knee = bd == B_SHANK_L || bd == B_SHANK_R || bd == B_THIGH_L || bd == B_THIGH_R;
      if (r.first == '-' && (trunk || hand || knee)) {
        r.first = trunk ? 'T' : (hand ? 'H' : 'K');
        r.t_first = t;
        if (trace) printf("first touch body %d t %.2f hands z %.2f %.2f\n", bd, t, d->xpos[3 * (b.body0 + B_HAND_L) + 2], d->xpos[3 * (b.body0 + B_HAND_R) + 2]);
        r.v_first = vb;
      }
      if (trunk) r.fell = 1;
      if (bd == B_HEAD && !head_hit) { head_hit = true; r.head_v = std::fmax(hv, hv_prev); r.t_head = t; }
    }
    if (d->xpos[3 * b.body0 + 2] < 0.5) r.fell = 1;
    if (r.fell && r.t_fell < 0) r.t_fell = t;
    if (sc.kind == K_BULLET && !sc.shots.empty()) {  // path after the last shot
      const double tl = sc.shots.back().dt;
      const double* pp = d->xpos + 3 * b.body0;
      if (std::fabs(t - tl) < 0.5 * World::kFrameDt) { last_p[0] = pp[0]; last_p[1] = pp[1]; }
      const double marks[5] = {0.4, 0.6, 0.8, 1.0, 2.0};
      for (int k = 0; k < 5; ++k)
        if (std::fabs(t - tl - marks[k]) < 0.5 * World::kFrameDt) r.path[k] = std::hypot(pp[0] - last_p[0], pp[1] - last_p[1]);
      r.lie = std::hypot(pp[0] - p0[0], pp[1] - p0[1]);
    }
    r.health = b.health;
    if (b.mode == 2 && r.t_m2 < 0) r.t_m2 = t;
    if (trace && fr % 3 == 0 && std::getenv("RX_ARM")) {
      for (int k : {B_PELVIS, B_CHEST, B_HEAD, B_UARM_L, B_FARM_L, B_HAND_L, B_UARM_R, B_HAND_R}) printf("arm t %.2f body %d (%.2f %.2f %.2f)\n", t, k, d->xpos[3 * (b.body0 + k)], d->xpos[3 * (b.body0 + k) + 1], d->xpos[3 * (b.body0 + k) + 2]);
    }
    if (trace && fr % 3 == 0) {  // RX_TRACE=<scene> prints the predictor inputs
      const Percept& P = b.pc[b.cur ^ 1];
      const double om = std::sqrt(9.81 * (1.0 / std::fmax(P.com[2], 0.3)));
      const double xd = std::hypot(P.com_vel[0], P.com_vel[1]) * (1.0 / om);
      printf("fall dir %.2f %.2f lie %.2f knee q %.2f tgt %.2f kp %.1f hip q %.2f tgt %.2f\n", b.fall_fwd, b.fall_lat, b.lie_t0,
             P.q[J_KNEE_L], b.target[J_KNEE_L], b.kp[J_KNEE_L], P.q[J_HIP_FLEX_L], b.target[J_HIP_FLEX_L]);
      printf("t %.2f mode %d comz %.2f speed %.2f vz %.2f xi_d %.2f cp_out %.2f pelvz %.2f ph %d pred %d tp %.2f dl %.2f fq %.2f ft %.2f aq %.2f at %.2f yaw %.2f roll %.2f pitch %.2f\n", t, b.mode, P.com[2],
             std::hypot(P.com_vel[0], P.com_vel[1]), P.com_vel[2], xd, b.cp_out, d->xpos[3 * b.body0 + 2], b.phase, b.fall_pred, b.fall_tp, b.fall_delay, P.q[J_SH_FLEX_L], b.target[J_SH_FLEX_L], P.q[J_SH_ABD_L], b.target[J_SH_ABD_L], P.pelvis_yaw, P.trunk_roll, P.trunk_pitch);
    }
    // Car: pass-through (part centre inside the car's lower box) and guarding hands at first car contact.
    if (sc.kind == K_CAR) {
      const int cb = mj_name2id(m, mjOBJ_BODY, "car");
      const double* cp = d->xpos + 3 * cb; const double* cR = d->xmat + 9 * cb;
      for (int k = 0; k < B_COUNT; ++k) {
        const double* x = d->xipos + 3 * (b.body0 + k);
        const double q[3] = {x[0] - cp[0], x[1] - cp[1], x[2] - cp[2]};
        const double lx = cR[0] * q[0] + cR[3] * q[1] + cR[6] * q[2], ly = cR[1] * q[0] + cR[4] * q[1] + cR[7] * q[2];
        if (std::fabs(lx) < 2.15 && std::fabs(ly) < 0.8 && q[2] > 0.2 && q[2] < 0.8) r.through = 1;
      }
      if (!r.guard) {
        for (int i = 0; i < d->ncon; ++i) {
          const mjContact& c = d->contact[i];
          const int b1 = m->geom_bodyid[c.geom1], b2 = m->geom_bodyid[c.geom2];
          if (b1 != cb && b2 != cb) continue;
          const int bd = (b1 == cb ? b2 : b1) - b.body0;
          if (bd >= 0 && bd < B_COUNT) {
            r.guard = (bd >= B_UARM_L && bd <= B_HAND_R) ? 2 : 1;
            if (trace) {  // hands toward the car at its first contact: hand ahead of the chest toward the car [m]
              const double* ch = d->xipos + 3 * (b.body0 + B_CHEST);
              const double* hl = d->xpos + 3 * (b.body0 + B_HAND_L);
              const double* hr = d->xpos + 3 * (b.body0 + B_HAND_R);
              printf("guard t %.2f body %d hand toward the car L %.2f R %.2f m, z %.2f %.2f\n", t, bd,
                     -((hl[0] - ch[0]) * cR[0] + (hl[1] - ch[1]) * cR[3]), -((hr[0] - ch[0]) * cR[0] + (hr[1] - ch[1]) * cR[3]),
                     hl[2], hr[2]);
            }
            break;
          }
        }
      }
    }
    if (sc.kind == K_CURL && fr >= 180) {
      double kc = 0;
      for (int s = 0; s < 2; ++s) {
        const double* kn = d->xpos + 3 * (b.body0 + (s ? B_SHANK_R : B_SHANK_L));
        const double* ch = d->xipos + 3 * (b.body0 + B_CHEST);
        kc += 0.5 * std::sqrt((kn[0] - ch[0]) * (kn[0] - ch[0]) + (kn[1] - ch[1]) * (kn[1] - ch[1]) + (kn[2] - ch[2]) * (kn[2] - ch[2]));
      }
      r.kc = kc;
      if (r.t_pose < 0 && kc < 0.40) r.t_pose = t - 3.0;
      if (fr >= frames - 60) {
        for (int j = 0; j < J_COUNT; ++j) { const double v = d->qvel[b.dof0 + j]; jit_s += v * v; ++jit_n; }
      }
    }
  }
  if (d->time > 0 && r.fell && sc.kind != K_DROP) {
    for (int k = B_UARM_L; k <= B_FOOT_R; ++k) {
      const double z = d->xipos[3 * (b.body0 + k) + 2];
      if (z > 0.30) { ++r.up; if (trace) printf("up body %d z %.2f\n", k, z); }
      r.up_max = std::fmax(r.up_max, z);
    }
  }
  if (fp) fclose(fp);
  r.drift = (p0[0] - d->xpos[3 * b.body0]);  // along the bullet (-x) [m]
  r.settle = last_move;
  if (jit_n) r.jitter = std::sqrt(jit_s / jit_n);
  return r;
}
}  // namespace

int RunReactBench(const char* label, int threads) {
  const std::string dir = std::string(TmpPath("rx/").c_str()) + label;
  mkdir(TmpPath("rx").c_str(), 0755);
  mkdir(dir.c_str(), 0755);
  std::vector<RScene> sc = Scenes();
  if (const char* only = std::getenv("RX_ONLY")) {  // RX_ONLY=<prefix> runs only the scenes starting with it
    std::vector<RScene> k;
    for (const RScene& x : sc) if (!std::strncmp(x.name, only, std::strlen(only))) k.push_back(x);
    sc = k;
  }
  std::vector<RResult> res(sc.size());
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int k = 0; k < threads; ++k)
    th.emplace_back([&] { for (int i; (i = next++) < (int)sc.size();) res[i] = RunR(sc[i], dir); });
  for (auto& t : th) t.join();
  printf("reaction bench '%s' (CSV in %s)\n", label, dir.c_str());
  printf("%-20s fell swings drift_m settle_s first head_v vmax  thru guard t_pose kc_m  jitter t_m2 t_1st t_head up umax\n", "scene");
  for (size_t i = 0; i < sc.size(); ++i) {
    const RResult& r = res[i];
    printf("%-20s %4d %6d %7.2f %8.2f %5c %6.2f %5.1f %4d %5d %6.2f %5.2f %6.3f %5.2f %5.2f %5.2f %2d %5.2f\n", sc[i].name, r.fell, r.swings,
           r.drift, r.settle, r.first, r.head_v, r.vmax, r.through, r.guard, r.t_pose, r.kc, r.jitter, r.t_m2, r.t_first, r.t_head, r.up, r.up_max);
  }
  printf("BUL %-10s fell steps drift_m settle_s first drop_cm flex_deg hand_cm t_1st\n", "scene");
  for (size_t i = 0; i < sc.size(); ++i) {
    if (std::strncmp(sc[i].name, "w_", 2)) continue;
    const RResult& r = res[i];
    printf("BUL %-10s %4d %5d %7.2f %8.2f %5c %7.1f %8.1f %7.1f %5.2f\n", sc[i].name, r.fell, r.swings, r.drift, r.settle,
           r.first, 100 * r.drop, r.flex, r.hand_d < 9 ? 100 * r.hand_d : -1.0, r.t_first);
  }
  printf("DTH %-10s fell t_fell steps head_v health | path after the last shot 0.4 0.6 0.8 1.0 2.0 s [cm] lie_m\n", "scene");
  for (size_t i = 0; i < sc.size(); ++i) {
    if (std::strncmp(sc[i].name, "h_", 2)) continue;
    const RResult& r = res[i];
    printf("DTH %-10s %4d %6.2f %5d %6.2f %6.0f | %4.0f %4.0f %4.0f %4.0f %4.0f %5.2f\n", sc[i].name, r.fell, r.t_fell, r.swings,
           r.head_v, r.health, 100 * r.path[0], 100 * r.path[1], 100 * r.path[2], 100 * r.path[3], 100 * r.path[4], r.lie);
  }
  return 0;
}
