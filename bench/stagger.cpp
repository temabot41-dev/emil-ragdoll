// Stagger bench: the recoil after chest pushes compared with the target table, repeatable.
//   emil_bench stagger [label] [threads=3] [--params file] [--moments 1.5,2,2.5] [--scale k] [--chunk 1]
// 20 chest pushes (impulses for a 61.6 kg body, scaled by our mass / 61.6 = same velocity change):
//   back 30 45 60 80 100 120, forward 30 45 60 80 100, left 30 45 60 80 100 120, right 45 60 100.
// Push point: chest body, 0.1 m toward the side the push comes from, 0.06 m above the body origin (env STAG_PUSH_DZ),
// horizontal.
// Per scene (6 s after the push, per 60 Hz frame): steps (ankle and toe both 0.015 m above their heights at the push
// for >= 3 frames; the step ends when one is lower again), of the first two steps lift time, swing time, ankle
// displacement along and across the push, capture point behind the new ankle at touchdown, knee flexion at touchdown;
// trunk lean (pelvis -> chest) along the push relative to the push moment (min and max over 1.5 s), pelvis drop over
// 3 s, number of steps, stood (pelvis above 0.5 m and no trunk-to-floor contact for 6 s). Output: medians per group
// (direction x velocity-change size) next to the target table (stagger.h), mark + = inside +-30 % (floors
// 0.05 s, 0.05 m, 0.1 m, 5 deg, 1 step, 0.03 m; trunk +-(30 % + 3 deg)).
// Details per scene: $EMIL_TMP/stagger/<label>/scenes.csv. Each (moment, chunk of --chunk scenes, default 1) is one
// world; work is threaded.
#include <mujoco/mujoco.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tmp_dir.h"
#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"
#include "footspin.h"

using namespace emil;
using namespace stag;

void ApplyOverrides(World& w);  // main.cpp: -p / --params
bool HasOverrides();

namespace {

// Scenes per world (--chunk, divides 20). Default 1: one bot per world; with 5 bots in one world the shared contact
// solver couples them numerically and marginal scenes flip.
int g_chunk = 1;
double g_scale = 1.0;              // --scale k: all impulses x k (sanity check of the stood criterion; the table needs 1)

struct Tracker {
  bool pushed = false, fell = false, done = false;
  double t_push = 0, t_fell = -1, ux = 0, uy = 0, lean0 = 0, z0 = 0, zmin = 1e9;
  double a0[2] = {0, 0}, toe0[2] = {0, 0};
  int cnt[2] = {0, 0};
  bool in[2] = {false, false};
  double t0[2] = {0, 0}, p0[2][2] = {{0, 0}, {0, 0}};
  double hz[2] = {0, 0};  // highest ankle rise of the current swing [m]
  double yl[2] = {0, 0}, ymax[2] = {0, 0}, shlo[2][2], shhi[2][2];  // per swing foot: yaw at lift, max change, arms
  double com_prev[3] = {0, 0, 0};
  double yaw0 = 0, yaw = 0;  // pelvis yaw at the push, max turn since [deg]
  int td_seen = 0;           // balancer step ends seen
  double t_fast = 0;         // last time the CoM speed was >= 0.15 m/s [s after the push]
  double t_abd = 0;          // last time a shoulder abduction was >= 15 deg [s after the push]
  double t_sw = 0;           // last time a step was going on [s after the push]
  double ref[5] = {0, 0, 0, 0, 0};          // flexL flexR abdL abdR twist at the push (rest pose) [rad]
  std::vector<std::array<double, 6>> hist;  // dt + the same five per frame
  double t_one = 0;          // time at rest (mode 0, no swing) with a foot under 5 % of the weight [s]
  double fsp_y0[2] = {0, 0};  // foot yaw when it took >= 15 % of the weight
  double fsp_R0[2][9] = {};    // foot orientation then (twist about the vertical is measured against it)
  FootSpin fs;                 // flat standing foot turn over 0.2 s
  int fsp_on[2] = {0, 0};
  double fspin = 0;  // largest turn of a loaded foot since it was loaded [deg]
  double fa_end = 0, lat_end = 0;  // feet split (right foot to left) along / across the pelvis heading at 2.4 s [m]
  std::vector<double> e_lon, e_lat, e_over, e_vpk, e_dur, e_len, e_yaw;
  std::vector<Step> steps;
  double lean_min = 0, lean_max = 0;
  double lt05 = 0, ut05 = 0, ut0 = 0, vr01 = -1, hvpk = 0, sq[3] = {0, 0, 0}, tq = -1;  // trunk tilts, speeds
  double tr_v[3] = {0, 0, 0}, tr_lt = 0, tr_ut = 0;  // last frame: pelvis, chest, head speed; tilts
};

std::mutex g_mu;

double Lean(const mjData* d, const Bot& b, double ux, double uy) {
  const double* p = d->xpos + 3 * (b.body0 + B_PELVIS);
  const double* c = d->xpos + 3 * (b.body0 + B_CHEST);
  return std::atan2((c[0] - p[0]) * ux + (c[1] - p[1]) * uy, c[2] - p[2]) * kRad;
}
// Upper trunk: chest joint -> neck joint (head body origin), tilt along (ux, uy) from the vertical [deg]
double UpLean(const mjData* d, const Bot& b, double ux, double uy) {
  const double* p = d->xpos + 3 * (b.body0 + B_CHEST);
  const double* c = d->xpos + 3 * (b.body0 + B_HEAD);
  return std::atan2((c[0] - p[0]) * ux + (c[1] - p[1]) * uy, c[2] - p[2]) * kRad;
}

// Ankle (joint anchor) and toe (front-bottom corner of the foot box) heights and ankle position of foot f.
void Foot(const mjModel* m, const mjData* d, const Bot& b, int f, double ank[3], double* toe_z) {
  const double* a = d->xanchor + 3 * (b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L));
  for (int k = 0; k < 3; ++k) ank[k] = a[k];
  const int g = b.geom_foot[f];
  const double* R = d->geom_xmat + 9 * g;
  const double* s = m->geom_size + 3 * g;
  *toe_z = d->geom_xpos[3 * g + 2] + R[6] * s[0] - R[8] * s[2];  // + x_half along the foot axis, - z_half (floor side)
}

void RunItems(const std::vector<Item>& items, const std::function<void(World&)>& setup, std::vector<Res>* out) {
  const int n = (int)items.size();
  std::vector<BodyParams> bodies;
  double moment_max = 0;
  for (const Item& it : items) { bodies.push_back(it.body); moment_max = std::fmax(moment_max, it.moment); }
  World w;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!w.Build(bodies, 3.0, 21, err, sizeof err)) {
      fprintf(stderr, "build failed: %s\n", err);
      return;
    }
  }
  if (setup) {
    setup(w);
    w.Reset();  // spawn pose from the parameters
  }
  const mjModel* m = w.model();
  mjData* d = w.data();
  const int body00 = w.bot(0).body0;
  std::vector<Tracker> tk(n);
  const int frames = (int)std::lround((moment_max + kTail) * 60.0) + 2;
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    for (int i = 0; i < n; ++i) {
      Bot& b = w.bot(i);
      Tracker& k = tk[i];
      const double* com = d->subtree_com + 3 * b.body0;
      const double com_v[3] = {(com[0] - k.com_prev[0]) * 60.0, (com[1] - k.com_prev[1]) * 60.0, (com[2] - k.com_prev[2]) * 60.0};
      for (int c = 0; c < 3; ++c) k.com_prev[c] = com[c];
      if (!k.pushed) {
        if (t < items[i].moment) continue;
        const Scene& sc = kScenes[items[i].scene];
        const double* R = d->xmat + 9 * b.body0;
        const double hn = std::hypot(R[0], R[3]) + 1e-9, hx = R[0] / hn, hy = R[3] / hn;
        const double dx[4] = {-hx, hx, -hy, hy}, dy[4] = {-hy, hy, hx, -hx};  // back, forward, left (+y), right
        k.ux = dx[sc.dir];
        k.uy = dy[sc.dir];
        k.pushed = true;
        k.t_push = t;
        k.fs.Rest(m, d, b);
        k.lean0 = Lean(d, b, k.ux, k.uy);
        k.ut0 = UpLean(d, b, k.ux, k.uy);
        k.yaw0 = std::atan2(R[3], R[0]);
        k.td_seen = b.td_cnt;
        k.z0 = d->xpos[3 * (b.body0 + B_PELVIS) + 2];
        {
          const double* qq = d->qpos + b.qadr0;
          const double r5[5] = {qq[J_SH_FLEX_L], qq[J_SH_FLEX_R], qq[J_SH_ABD_L], qq[J_SH_ABD_R],
                                qq[J_LUMBAR_YAW] + qq[J_CHEST_YAW]};
          for (int a = 0; a < 5; ++a) k.ref[a] = r5[a];
        }
        for (int f = 0; f < 2; ++f) {
          double a[3];
          Foot(m, d, b, f, a, &k.toe0[f]);
          k.a0[f] = a[2];
        }
        const double J = g_scale * items[i].scale * sc.J * b.dims.total_mass / kRefMass;
        const double* o = d->xpos + 3 * (b.body0 + B_CHEST);
        // STAG_PUSH_DZ = push point height over the chest body origin [m] (default 0.06: 0.43 m over the pelvis)
        static const double pdz = getenv("STAG_PUSH_DZ") ? std::atof(getenv("STAG_PUSH_DZ")) : 0.06;
        const double pt[3] = {o[0] - 0.1 * k.ux, o[1] - 0.1 * k.uy, o[2] + pdz};
        const double imp[3] = {k.ux * J, k.uy * J, 0};
        w.PushAt(i, imp, pt, B_CHEST);  // standard chest push point
        continue;
      }
      const double dt = t - k.t_push;
      if (dt > kTail || k.done) continue;
      const double zp = d->xpos[3 * (b.body0 + B_PELVIS) + 2];
      if (zp < 0.5 && !k.fell) { k.fell = true; k.t_fell = dt; }
      if (!k.fell && b.mode == 0 && b.swing < 0 && std::fmin(b.ld[0], b.ld[1]) < 0.05) k.t_one += World::kFrameDt;
      {
        const double* Rp = d->xmat + 9 * b.body0;
        double dy = std::atan2(Rp[3], Rp[0]) - k.yaw0;
        while (dy > 3.14159265) dy -= 6.2831853;
        while (dy < -3.14159265) dy += 6.2831853;
        if (!k.fell) k.yaw = std::fmax(k.yaw, std::fabs(dy) * 57.29578);
        if (!k.fell && dt <= 2.4) {  // feet split in the pelvis heading frame at 2.4 s (before late steps)
          const double* aL = d->xpos + 3 * (b.body0 + B_FOOT_L);
          const double* aR = d->xpos + 3 * (b.body0 + B_FOOT_R);
          const double ch = Rp[0], sh = Rp[3], nh = std::hypot(ch, sh) + 1e-9;
          k.fa_end = ((aL[0] - aR[0]) * ch + (aL[1] - aR[1]) * sh) / nh;
          k.lat_end = (-(aL[0] - aR[0]) * sh + (aL[1] - aR[1]) * ch) / nh;
        }
      }
      if (dt <= 3.0) k.zmin = std::fmin(k.zmin, zp);
      if (!k.fell && std::hypot(com_v[0], com_v[1]) >= 0.15) k.t_fast = dt;
      if (!k.fell && b.swing >= 0) k.t_sw = dt;
      if (!k.fell) k.fs.Step(m, d, b, dt);
      for (int f = 0; f < 2; ++f) {  // standing foot turn on the floor (target <= 5 deg)
        const double* Rf = d->xmat + 9 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        const double yf = std::atan2(Rf[3], Rf[0]);
        if (k.fell || b.foot_fn[f] < 0.15 * b.dims.total_mass * 9.81) { k.fsp_on[f] = 0; continue; }
        if (Rf[8] < 0.866) continue;  // sole tilted > 30 deg (on the toe): the heading is ill-defined, not standing
        if (!k.fsp_on[f]) {
          k.fsp_on[f] = 1;
          k.fsp_y0[f] = yf;
          for (int q = 0; q < 9; ++q) k.fsp_R0[f][q] = Rf[q];
          continue;
        }
        // Twist about the vertical of the rotation since loading, Rr = Rf R0^T (swing-twist): a foot rolling or
        // pitching onto its sole does not count (its heading angle atan2 changes by ~5 deg on a 15 deg pitch with a
        // 20 deg roll at touchdown).
        const double* R0 = k.fsp_R0[f];
        double Rr[4];
        for (int r = 0; r < 2; ++r)
          for (int c = 0; c < 2; ++c)
            Rr[2 * r + c] = Rf[3 * r] * R0[3 * c] + Rf[3 * r + 1] * R0[3 * c + 1] + Rf[3 * r + 2] * R0[3 * c + 2];
        const double e = std::atan2(Rr[2] - Rr[1], Rr[0] + Rr[3]);
        k.fspin = std::fmax(k.fspin, std::fabs(e) * 57.29578);
      }
      // STAG_TR="<scene> <moment>": per frame of that scene - swing, per foot load, twist since loading
      // (T43), sole roll (outer edge up +), ankle inversion q > target, hip twist q > target and net torque, sole axis
      // heading; pelvis heading [deg, N, N m]
      if (const char* tr = getenv("STAG_TR")) {
        char nm[32];
        double mo = -1;
        if (sscanf(tr, "%31s %lf", nm, &mo) == 2 && !std::strcmp(nm, kScenes[items[i].scene].name) && std::fabs(mo - items[i].moment) < 1e-3 &&
            dt <= 3.0 && d->xpos[3 * (b.body0 + B_PELVIS) + 2] > 0.3) {
          printf("ST %.3f sw %2d md %d kn %5.1f %5.1f kt %5.1f %5.1f hd %.3f hm %.3f", dt, b.swing, b.mode,
                 d->qpos[b.qadr0 + J_KNEE_L] * 57.29578, d->qpos[b.qadr0 + J_KNEE_R] * 57.29578,
                 b.target[J_KNEE_L] * 57.29578, b.target[J_KNEE_R] * 57.29578, b.dbg_h[0], b.dbg_h[1]);
          printf(" LT %.1f UT %.1f vP %.2f vC %.2f vH %.2f", k.tr_lt, k.tr_ut, k.tr_v[0], k.tr_v[1], k.tr_v[2]);
          printf(" SI %.2f CO %.2f", b.sit_t0 >= 0 ? d->time - b.sit_t0 : -1.0, b.cp_out);  // seat age, cp out [s, m]
          printf(" RC %.3f HL %d", b.sw_short, b.hpd_lock);  // swing place beyond the reach [m], hold lock
          {  // along the push from the pelvis start: CoM, capture point (CoM + v sqrt(z/g)), sole centres L/R [m]
            const double* cm = d->subtree_com + 3 * b.body0;
            double vc = 0;
            {
              static thread_local double cprev = 0, tprev = -1;
              const double cs = cm[0] * k.ux + cm[1] * k.uy;
              if (tprev >= 0 && t > tprev) vc = (cs - cprev) / (t - tprev);
              cprev = cs;
              tprev = t;
            }
            const double c0 = cm[0] * k.ux + cm[1] * k.uy;
            const double fl = d->xpos[3 * (b.body0 + B_FOOT_L)] * k.ux + d->xpos[3 * (b.body0 + B_FOOT_L) + 1] * k.uy;
            const double fr = d->xpos[3 * (b.body0 + B_FOOT_R)] * k.ux + d->xpos[3 * (b.body0 + B_FOOT_R) + 1] * k.uy;
            printf(" CX %.3f XI %.3f FL %.3f FR %.3f VC %.2f", c0, c0 + vc * std::sqrt(std::fmax(cm[2], 0.1) / 9.81), fl, fr,
                   vc);
            for (int f = 0; f < 2; ++f) {  // heel and toe along the push, heights [m]
              double an[3], he[3], to[3];
              FootSpin::Points(m, d, b, f, an, he, to);
              printf(" %cHT %.3f %.3f %.3f %.3f", "LR"[f], he[0] * k.ux + he[1] * k.uy, to[0] * k.ux + to[1] * k.uy,
                     he[2], to[2]);
            }
          }
          {  // hip flexion q > target and net torque per leg [deg, N m]; pelvis tilt with the push from the vertical [deg]
            const double* Rp = d->xmat + 9 * (b.body0 + B_PELVIS);
            printf(" HF %5.1f>%5.1f %5.1f>%5.1f hn %4.0f %4.0f PT %5.1f", d->qpos[b.qadr0 + J_HIP_FLEX_L] * 57.29578,
                   b.target[J_HIP_FLEX_L] * 57.29578, d->qpos[b.qadr0 + J_HIP_FLEX_R] * 57.29578,
                   b.target[J_HIP_FLEX_R] * 57.29578, b.net_dbg[0], b.net_dbg[1],
                   std::atan2(Rp[2] * k.ux + Rp[5] * k.uy, Rp[8]) * 57.29578);
          }
          printf(" ab %d at %.2f lt %.2f cr %.3f fo %d%d dr %.3f dl %.3f kf %d%d", b.airborne, b.air_t, b.land_t,
                 b.in.crouch, b.foot_on[0], b.foot_on[1], b.dbg_h[2], b.dbg_h[3], b.kf_fr[0], b.kf_fr[1]);
          printf(" tg %.3f %.3f fz %.3f %.3f px %.3f pl %.3f %.3f", b.sw_tg[0], b.sw_tg[1],
                 d->xpos[3 * (b.body0 + B_FOOT_L) + 2], d->xpos[3 * (b.body0 + B_FOOT_R) + 2],
                 d->xpos[3 * (b.body0 + B_PELVIS)], b.place[b.swing >= 0 ? b.swing : 0][0],
                 b.place[b.swing >= 0 ? b.swing : 0][1]);
          // knee R: actuator torque, passive damper torque, velocity [deg/s], torque limit
          printf(" kR %.0f %.0f %.0f %.0f", d->actuator_force[b.act0 + J_KNEE_R], d->qfrc_damper[b.dof0 + J_KNEE_R],
                 d->qvel[b.dof0 + J_KNEE_R] * 57.29578, b.js[J_KNEE_R].tau_max);
          for (int f = 0; f < 2; ++f) {
            const double* Rf = d->xmat + 9 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
            double e = 0;
            if (k.fsp_on[f]) {
              const double* R0 = k.fsp_R0[f];
              double Rr[4];
              for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c)
                  Rr[2 * r + c] = Rf[3 * r] * R0[3 * c] + Rf[3 * r + 1] * R0[3 * c + 1] + Rf[3 * r + 2] * R0[3 * c + 2];
              e = std::atan2(Rr[2] - Rr[1], Rr[0] + Rr[3]);
            }
            const double* Rg = d->geom_xmat + 9 * b.geom_foot[f];
            const int ji = f ? J_ANKLE_INV_R : J_ANKLE_INV_L, jr = f ? J_HIP_ROT_R : J_HIP_ROT_L;
            printf(" | %c fn %4.0f tw %5.1f roll %5.1f inv %5.1f>%5.1f ti %5.1f kp %5.0f rot %5.1f>%5.1f u %5.1f ax %6.1f",
                   "LR"[f], b.foot_fn[f], e * 57.29578, std::asin(Rg[7]) * 57.29578 * (f ? -1.0 : 1.0),
                   d->qpos[b.qadr0 + ji] * 57.29578, b.target[ji] * 57.29578,
                   d->qfrc_actuator[b.dof0 + ji] - m->dof_damping[b.dof0 + ji] * d->qvel[b.dof0 + ji], b.kp[ji],
                   d->qpos[b.qadr0 + jr] * 57.29578,
                   b.target[jr] * 57.29578,
                   d->qfrc_actuator[b.dof0 + jr] - m->dof_damping[b.dof0 + jr] * d->qvel[b.dof0 + jr],
                   std::atan2(Rg[3], Rg[0]) * 57.29578);
            // floor force along the push on this foot (25 ms low-pass, + = along the push) and the hip flexion torque [N m]
            printf(" fh %5.0f hq %5.0f", b.foot_fh[f][0] * k.ux + b.foot_fh[f][1] * k.uy, b.net_dbg[f]);
          }
          const double* Rp = d->xmat + 9 * b.body0;
          // CoM and its velocity, sole centres, pelvis height (world) [m, m/s]
          const double* cm = d->subtree_com + 3 * b.body0;
          mj_subtreeVel(m, d);
          const double* vc = d->subtree_linvel + 3 * b.body0;
          const double* gl = d->geom_xpos + 3 * b.geom_foot[0];
          const double* gr = d->geom_xpos + 3 * b.geom_foot[1];
          printf(" | pel %6.1f com %6.3f %6.3f v %5.2f %5.2f L %6.3f %6.3f R %6.3f %6.3f zp %.3f",
                 std::atan2(Rp[3], Rp[0]) * 57.29578, cm[0], cm[1], vc[0], vc[1], gl[0], gl[1], gr[0], gr[1],
                 d->xpos[3 * (b.body0 + B_PELVIS) + 2]);
          // floor CoP y of each foot, floor force y (25 ms filtered), whole-body angular momentum x about the CoM
          double cpy[2] = {0, 0}, fzs[2] = {0, 0};
          for (int c = 0; c < d->ncon; ++c) {
            const mjContact& cn = d->contact[c];
            for (int f = 0; f < 2; ++f) {
              if (cn.geom1 != b.geom_foot[f] && cn.geom2 != b.geom_foot[f]) continue;
              double f6[6];
              mj_contactForce(m, d, c, f6);
              const double fz = std::fabs(f6[0] * cn.frame[2]);
              cpy[f] += cn.pos[1] * fz;
              fzs[f] += fz;
            }
          }
          for (int f = 0; f < 2; ++f) cpy[f] = fzs[f] > 1 ? cpy[f] / fzs[f] : 0.0;
          // chest heading, lumbar + chest yaw joints, whole-body angular momentum z about the CoM [deg, N m s]
          const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
          printf(" cop %6.3f %6.3f Fy %5.0f %5.0f Lx %5.2f chest %6.1f byaw %5.1f Lz %5.2f\n", cpy[0], cpy[1],
                 b.foot_fh[0][1], b.foot_fh[1][1], d->subtree_angmom[3 * b.body0], std::atan2(Rc[3], Rc[0]) * 57.29578,
                 (d->qpos[b.qadr0 + J_LUMBAR_YAW] + d->qpos[b.qadr0 + J_CHEST_YAW]) * 57.29578,
                 d->subtree_angmom[3 * b.body0 + 2]);
        }
      }
      {
        const double* qq = d->qpos + b.qadr0;
        k.hist.push_back({dt, qq[J_SH_FLEX_L], qq[J_SH_FLEX_R], qq[J_SH_ABD_L], qq[J_SH_ABD_R],
                          qq[J_LUMBAR_YAW] + qq[J_CHEST_YAW]});
      }
      if (!k.fell && std::fmax(d->qpos[b.qadr0 + J_SH_ABD_L], d->qpos[b.qadr0 + J_SH_ABD_R]) * 57.29578 >= 15.0)
        k.t_abd = dt;
      if (b.td_cnt != k.td_seen && !k.fell) {
        k.td_seen = b.td_cnt;
        k.e_lon.push_back(b.td_err[0]);
        k.e_lat.push_back(b.td_err[1]);
        k.e_over.push_back(b.td_over);
        k.e_vpk.push_back(b.td_vpk);
        k.e_dur.push_back(b.td_dur);
        k.e_len.push_back(b.td_len);
        k.e_yaw.push_back(b.td_yaw);
      }
      if (dt <= 1.5) {
        const double l = Lean(d, b, k.ux, k.uy) - k.lean0;
        k.lean_min = std::fmin(k.lean_min, l);
        k.lean_max = std::fmax(k.lean_max, l);
      }
      {  // trunk tilts and speeds along the push (pelvis, chest, head body origins)
        static const int kb[3] = {B_PELVIS, B_CHEST, B_HEAD};
        double sq[3];
        for (int q = 0; q < 3; ++q) {
          const double* x = d->xpos + 3 * (b.body0 + kb[q]);
          sq[q] = x[0] * k.ux + x[1] * k.uy;
        }
        if (k.tq >= 0 && t > k.tq)
          for (int q = 0; q < 3; ++q) k.tr_v[q] = (sq[q] - k.sq[q]) / (t - k.tq);
        for (int q = 0; q < 3; ++q) k.sq[q] = sq[q];
        k.tq = t;
        k.tr_lt = Lean(d, b, k.ux, k.uy) - k.lean0;
        k.tr_ut = UpLean(d, b, k.ux, k.uy) - k.ut0;
        if (dt <= 0.5) {
          k.lt05 = std::fmax(k.lt05, k.tr_lt);
          k.ut05 = std::fmax(k.ut05, k.tr_ut);
        }
        if (k.vr01 < 0 && dt >= 0.1 - 1e-6 && std::fabs(k.tr_v[1]) > 1e-3) k.vr01 = k.tr_v[0] / k.tr_v[1];
        if (dt <= 1.0) k.hvpk = std::fmax(k.hvpk, k.tr_v[2]);
      }
      for (int f = 0; f < 2; ++f) {
        double a[3], tz;
        Foot(m, d, b, f, a, &tz);
        const bool up = a[2] > k.a0[f] + 0.015 && tz > k.toe0[f] + 0.015;
        const double* Ry = d->xmat + 9 * b.body0;
        const double yaw_now = std::atan2(Ry[3], Ry[0]);
        const double shq[2] = {d->qpos[b.qadr0 + J_SH_FLEX_L] * kRad, d->qpos[b.qadr0 + J_SH_FLEX_R] * kRad};
        if (k.in[f] || (up && k.cnt[f] > 0)) {
          double dy = yaw_now - k.yl[f];
          while (dy > 3.14159265) dy -= 6.2831853;
          while (dy < -3.14159265) dy += 6.2831853;
          k.ymax[f] = std::fmax(k.ymax[f], std::fabs(dy) * kRad);
          k.hz[f] = std::fmax(k.hz[f], a[2] - k.a0[f]);
          for (int s2 = 0; s2 < 2; ++s2) {
            k.shlo[f][s2] = std::fmin(k.shlo[f][s2], shq[s2]);
            k.shhi[f][s2] = std::fmax(k.shhi[f][s2], shq[s2]);
          }
        }
        if (!k.in[f]) {
          if (up) {
            if (k.cnt[f] == 0) {
              k.t0[f] = t; k.p0[f][0] = a[0]; k.p0[f][1] = a[1];
              k.hz[f] = 0;
              k.yl[f] = yaw_now; k.ymax[f] = 0;
              for (int s2 = 0; s2 < 2; ++s2) k.shlo[f][s2] = k.shhi[f][s2] = shq[s2];
            }
            if (++k.cnt[f] >= 3) k.in[f] = true;
          } else {
            k.cnt[f] = 0;
          }
        } else if (!up) {  // touchdown
          Step s;
          s.foot = f;
          s.t_lift = k.t0[f] - k.t_push;
          s.swing = t - k.t0[f];
          const double ex = a[0] - k.p0[f][0], ey = a[1] - k.p0[f][1];
          s.along = ex * k.ux + ey * k.uy;
          s.across = -ex * k.uy + ey * k.ux;
          const double om = std::sqrt(9.81 / std::fmax(com[2], 0.3));
          s.tz = (com[0] + com_v[0] / om - a[0]) * k.ux + (com[1] + com_v[1] / om - a[1]) * k.uy;
          s.knee = d->qpos[b.qadr0 + (f ? J_KNEE_R : J_KNEE_L)] * kRad;
          s.dyaw = k.ymax[f];
          s.shr = std::fmax(k.shhi[f][0] - k.shlo[f][0], k.shhi[f][1] - k.shlo[f][1]);
          s.ahead = (a[0] - com[0]) * k.ux + (a[1] - com[1]) * k.uy;
          s.hmax = k.hz[f];
          k.steps.push_back(s);
          k.in[f] = false;
          k.cnt[f] = 0;
        }
      }
    }
    for (int c = 0; c < d->ncon; ++c) {  // pelvis .. head on the floor
      const mjContact& cn = d->contact[c];
      const int g = cn.geom1 == 0 ? cn.geom2 : (cn.geom2 == 0 ? cn.geom1 : -1);
      if (g < 0) continue;
      const int rel0 = m->geom_bodyid[g] - body00;
      if (rel0 < 0) continue;
      const int i = rel0 / B_COUNT, rel = rel0 % B_COUNT;
      if (i < n && rel >= B_PELVIS && rel <= B_HEAD && tk[i].pushed && t - tk[i].t_push <= kTail && !tk[i].fell) {
        tk[i].fell = true;
        tk[i].t_fell = t - tk[i].t_push;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    Tracker& k = tk[i];
    Res r;
    r.scene = items[i].scene;
    r.moment = items[i].moment;
    r.dv = kScenes[items[i].scene].J / kRefMass;
    r.stood = k.pushed && !k.fell;
    r.yaw = k.yaw;
    r.t_fall = k.fell ? k.t_fell : -1;
    std::sort(k.steps.begin(), k.steps.end(), [](const Step& a, const Step& b) { return a.t_lift < b.t_lift; });
    r.nsteps = (int)k.steps.size();
    for (const Step& s : k.steps) r.late += s.t_lift > 2.5;
    r.t_rest = r.stood ? k.t_fast : -1;
    r.arms_dn = r.stood ? std::fmax(0.0, k.t_abd - k.t_fast) : -1;
    if (r.stood) {  // pose 0.6 s after rest (CoM < 0.15 m/s and no step) against the pose at the push [deg]
      const double tr = std::fmax(k.t_fast, k.t_sw) + 0.6;
      for (const auto& h : k.hist)
        if (h[0] >= tr) {
          r.rp_flex = std::fmax(std::fabs(h[1] - k.ref[0]), std::fabs(h[2] - k.ref[1])) * 57.29578;
          r.rp_abd = std::fmax(std::fabs(h[3] - k.ref[2]), std::fabs(h[4] - k.ref[3])) * 57.29578;
          r.rp_tw = std::fabs(h[5] - k.ref[4]) * 57.29578;
          break;
        }
    }
    r.t_one = k.t_one;
    r.fspin = k.fspin;
    r.fa_end = k.fa_end;
    r.spin02 = k.fs.worst;
    r.slide02 = k.fs.worst_slide;
    r.spin02_ip = k.fs.worst_ip;
    r.spin02_t = k.fs.worst_t;
    r.spin02_ax = k.fs.worst_ax;
    r.spin02_axip = k.fs.worst_ax_ip;
    r.roll_max = k.fs.roll_max;
    r.roll_frames = k.fs.roll_frames;
    r.lim_frames = k.fs.lim_frames;
    r.ip_single = k.fs.ip_single;
    r.ip_double = k.fs.ip_double;
    r.ip_lim = k.fs.ip_lim;
    r.ip_roll = k.fs.ip_roll;
    r.ip_noroll = k.fs.ip_noroll;
    r.ip_settle = k.fs.ip_settle;
    r.ip_tip = k.fs.ip_tip;
    r.lat_end = k.lat_end;
    r.e_lon = k.e_lon;
    r.e_lat = k.e_lat;
    r.e_over = k.e_over;
    r.e_vpk = k.e_vpk;
    r.e_dur = k.e_dur;
    r.e_len = k.e_len;
    r.e_yaw = k.e_yaw;
    r.have = std::min(2, r.nsteps);
    for (int q = 0; q < r.have; ++q) r.st[q] = k.steps[q];
    r.all_steps = k.steps;
    r.lean_min = k.lean_min;
    r.lean_max = k.lean_max;
    r.lt05 = k.lt05;
    r.ut05 = k.ut05;
    r.vr01 = k.vr01;
    r.hvpk = k.hvpk;
    r.drop = k.z0 - k.zmin;
    out->push_back(r);
  }
}

double Median(std::vector<double> v) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }), v.end());
  if (v.empty()) return kNan;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// '+' inside the band around the target, '-' outside, ' ' no target; band = max(rel * |target|, floor).
char Mark(double v, double target, double rel, double floor_) {
  if (std::isnan(v) || std::isnan(target)) return ' ';
  return std::fabs(v - target) <= std::fmax(rel * std::fabs(target), floor_) ? '+' : '-';
}

struct Cell { double v; char mark; };

int g_in = 0, g_all = 0;
void Print(const char* head, double v, double target, double rel, double floor_, const char* fmt) {
  char a[32], b[32];
  if (std::isnan(v)) snprintf(a, sizeof a, "  -  "); else snprintf(a, sizeof a, fmt, v);
  if (std::isnan(target)) snprintf(b, sizeof b, "-"); else snprintf(b, sizeof b, fmt, target);
  const char mk = Mark(v, target, rel, floor_);
  if (mk == '+') ++g_in;
  if (mk != ' ') ++g_all;
  printf(" %s %s[%s]%c", head, a, b, mk == ' ' ? ' ' : mk);
}

}  // namespace

namespace stag {

std::mutex& BuildMutex() { return g_mu; }

void RunWorld(const std::vector<Item>& items, const std::function<void(World&)>& setup, std::vector<Res>* out) {
  RunItems(items, setup, out);
}

int SizeGroup(double dv) { return dv < 0.6 ? 0 : (dv <= 1.1 ? 1 : 2); }

static double Band(double v, double target, double rel, double floor_) {  // violation beyond the band, in band widths, 0..3
  if (std::isnan(target)) return 0.0;
  const double band = std::fmax(rel * std::fabs(target), floor_);
  return std::fmin(3.0, std::fmax(0.0, std::fabs(v - target) / band - 1.0));
}

double PhasePenalty(const Res& r) {
  const int d = kScenes[r.scene].dir;
  const Ref& R = kRef[d == 0 ? 0 : (d == 1 ? 1 : 2)][SizeGroup(r.dv)];
  double sum = 0;
  int n = 0;
  auto add = [&](double v) { sum += v; ++n; };
  const double t1[2][4] = {{R.l1, R.s1, R.d1, R.t1}, {R.l2, R.s2, R.d2, R.t2}};
  for (int q = 0; q < 2; ++q) {
    if (std::isnan(t1[q][0])) continue;  // no step in the table (small sideways)
    if (q >= r.have) {
      for (int k = 0; k < 4; ++k) add(3.0);
      if (q == 1) add(3.0);
      continue;
    }
    const Step& s = r.st[q];
    add(Band(s.t_lift, t1[q][0], 0.3, 0.05));
    add(Band(s.swing, t1[q][1], 0.3, 0.05));
    add(Band(std::hypot(s.along, s.across), t1[q][2], 0.3, 0.05));
    add(Band(s.tz, t1[q][3], 0.0, 0.1));
    if (q == 1) add(Band(s.knee, R.k2, 0.3, 5.0));
  }
  // 2..3: half by the first two steps, half by how long it stayed up (1.5 s or more = 0; falls take 0.4-1.3 s)
  if (!r.stood) {
    const double m_steps = n ? sum / n : 3.0, m_fall = 3.0 * std::fmax(0.0, 1.0 - r.t_fall * (1.0 / 1.5));
    return 2.0 + (0.5 * m_steps + 0.5 * m_fall) * (1.0 / 3.0);
  }
  // stood: add trunk range (+-(30 % + 3 deg)), pelvis drop, number of steps
  const double lo = R.tr_lo - (0.3 * std::fabs(R.tr_lo) + 3), hi = R.tr_hi + (0.3 * std::fabs(R.tr_hi) + 3);
  add(std::fmin(3.0, std::fmax(0.0, std::fmax(lo - r.lean_min, r.lean_max - hi)) / 3.0));
  add(Band(r.drop, R.drop, 0.3, 0.03));
  add(Band(r.nsteps, R.steps, 0.3, 1.0));
  return std::fmin(2.0, sum / n);
}

}  // namespace stag

int RunStaggerBench(int argc, char** argv) {
  std::string label = "cur";
  int threads = 3, pos = 0;
  std::vector<double> moments = {1.5};
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) {
      g_scale = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) {
      g_chunk = std::atoi(argv[++i]);
      if (g_chunk < 1 || 20 % g_chunk) g_chunk = 1;
    } else if (!std::strcmp(argv[i], "--moments") && i + 1 < argc) {
      moments.clear();
      for (char* tok = std::strtok(argv[++i], ","); tok; tok = std::strtok(nullptr, ",")) moments.push_back(std::atof(tok));
    } else if (pos == 0) { label = argv[i]; ++pos; }
    else if (pos == 1) { threads = std::atoi(argv[i]); ++pos; }
  }
  if (moments.empty()) moments.push_back(1.5);
  threads = std::max(1, std::min(threads, 4));
  struct Job { int chunk; double moment; };
  std::vector<Job> jobs;
  for (double mo : moments)
    for (int c = 0; c < 20 / g_chunk; ++c) jobs.push_back({c, mo});
  std::vector<std::vector<Res>> res(jobs.size());
  std::atomic<int> next(0);
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t)
    th.emplace_back([&]() {
      for (;;) {
        const int j = next.fetch_add(1);
        if (j >= (int)jobs.size()) break;
        std::vector<Item> items;
        for (int q = 0; q < g_chunk; ++q) items.push_back({jobs[j].chunk * g_chunk + q, jobs[j].moment, BodyParams()});
        RunItems(items, [](World& w) { ApplyOverrides(w); }, &res[j]);
      }
    });
  for (auto& x : th) x.join();
  std::vector<Res> all;
  for (auto& r : res) all.insert(all.end(), r.begin(), r.end());

  mkdir(TmpDir().c_str(), 0755);
  mkdir(TmpPath("stagger").c_str(), 0755);
  const std::string dir = TmpPath("stagger/").c_str() + label;
  mkdir(dir.c_str(), 0755);
  if (FILE* f = fopen((dir + "/scenes.csv").c_str(), "w")) {
    fprintf(f, "scene,moment,dv,stood,steps,s1_lift,s1_swing,s1_along,s1_across,s1_tz,s1_knee,s2_lift,s2_swing,s2_along,s2_across,s2_tz,s2_knee,lean_min,lean_max,drop,t_fall,t_one,fspin,yaw,late,fa_end,lat_end,spin02,slide02,spin02_ip,spin02_t,spin02_ax,spin02_axip,roll_max,roll_frames,lim_frames,lt05,ut05,vr01,hvpk\n");
    for (const Res& r : all) {
      fprintf(f, "%s,%.2f,%.2f,%d,%d", kScenes[r.scene].name, r.moment, r.dv, (int)r.stood, r.nsteps);
      for (int q = 0; q < 2; ++q) {
        if (q < r.have) fprintf(f, ",%.3f,%.3f,%.3f,%.3f,%.3f,%.1f", r.st[q].t_lift, r.st[q].swing, r.st[q].along, r.st[q].across, r.st[q].tz, r.st[q].knee);
        else fprintf(f, ",,,,,,");
      }
      fprintf(f, ",%.1f,%.1f,%.3f,%.2f,%.2f,%.1f,%.0f,%d,%.3f,%.3f\n", r.lean_min, r.lean_max, r.drop, r.t_fall, r.t_one,
              r.fspin, r.yaw, (int)r.late, r.fa_end, r.lat_end);
      fseek(f, -1, SEEK_CUR);
      fprintf(f, ",%.1f,%.3f,%.1f,%.2f,%.1f,%.1f,%.1f,%d,%d\n", r.spin02, r.slide02, r.spin02_ip, r.spin02_t, r.spin02_ax,
              r.spin02_axip, r.roll_max, r.roll_frames, r.lim_frames);
      fseek(f, -1, SEEK_CUR);
      fprintf(f, ",%.1f,%.1f,%.2f,%.0f\n", r.lt05, r.ut05, r.vr01, 100 * r.hvpk);
    }
    fclose(f);
  }
  if (FILE* f = fopen((dir + "/steps.csv").c_str(), "w")) {  // one row per touchdown (step measures)
    fprintf(f, "scene,moment,stood,i,over,len,dur,vpk,yaw_deg,err_lon,err_lat\n");
    for (const Res& r : all)
      for (size_t i = 0; i < r.e_over.size(); ++i)
        fprintf(f, "%s,%.2f,%d,%d,%.3f,%.3f,%.3f,%.2f,%.1f,%.3f,%.3f\n", kScenes[r.scene].name, r.moment, (int)r.stood,
                (int)i, r.e_over[i], r.e_len[i], r.e_dur[i], r.e_vpk[i], r.e_yaw[i] * 57.29578,
                i < r.e_lon.size() ? r.e_lon[i] : 0.0, i < r.e_lat.size() ? r.e_lat[i] : 0.0);
    fclose(f);
  }

  printf("stagger '%s': 20 scenes x %d moment(s), medians per group [target table], + inside +-30 %%\n", label.c_str(),
         (int)moments.size());
  printf("step 1: lift/swing [s], displacement [m], capture point behind the ankle [m]; step 2 the same + knee [deg]; trunk lean "
         "min..max [deg] relative to the push; drop [m]\n");
  int tot_in = 0, tot_all = 0, stood_n = 0, scene_n = 0;
  double side_large_stood = 0;
  for (int dg = 0; dg < 3; ++dg)
    for (int sz = 0; sz < 3; ++sz) {
      std::vector<const Res*> g;
      for (const Res& r : all) {
        const int d = kScenes[r.scene].dir;
        const int dir_g = d == 0 ? 0 : (d == 1 ? 1 : 2);
        const int size_g = SizeGroup(r.dv);
        if (dir_g == dg && size_g == sz) g.push_back(&r);
      }
      if (g.empty()) continue;
      int stood = 0, steps_n[2] = {0, 0};
      std::vector<double> ns, v[2][6], lo, hi, dr;
      for (const Res* r : g) {
        stood += r->stood;
        ns.push_back(r->nsteps);
        for (int q = 0; q < 2; ++q)
          if (q < r->have) {
            ++steps_n[q];
            const Step& s = r->st[q];
            v[q][0].push_back(s.t_lift); v[q][1].push_back(s.swing);
            v[q][2].push_back(std::hypot(s.along, s.across)); v[q][3].push_back(s.tz); v[q][4].push_back(s.knee);
          }
        lo.push_back(r->lean_min); hi.push_back(r->lean_max); dr.push_back(r->drop);
      }
      stood_n += stood;
      scene_n += (int)g.size();
      if (dg == 2 && sz == 2) side_large_stood = (double)stood / g.size();
      const Ref& R = kRef[dg][sz];
      printf("%s, %s (n=%d, steps found %d/%d, stood %d/%d; table: stood all)\n ", kDirName[dg], kSizeName[sz], (int)g.size(), steps_n[0], steps_n[1],
             stood, (int)g.size());
      g_in = g_all = 0;
      Print("steps", Median(ns), R.steps, 0.3, 1.0, "%.1f");
      printf(" |");
      Print("lift", Median(v[0][0]), R.l1, 0.3, 0.05, "%.2f");
      Print("swing", Median(v[0][1]), R.s1, 0.3, 0.05, "%.2f");
      Print("len", Median(v[0][2]), R.d1, 0.3, 0.05, "%.2f");
      Print("tz", Median(v[0][3]), R.t1, 0.0, 0.1, "%.2f");
      printf(" |");
      Print("lift", Median(v[1][0]), R.l2, 0.3, 0.05, "%.2f");
      Print("swing", Median(v[1][1]), R.s2, 0.3, 0.05, "%.2f");
      Print("len", Median(v[1][2]), R.d2, 0.3, 0.05, "%.2f");
      Print("tz", Median(v[1][3]), R.t2, 0.0, 0.1, "%.2f");
      Print("knee", Median(v[1][4]), R.k2, 0.3, 5.0, "%.0f");
      printf(" |");
      const double tl = Median(lo), th_ = Median(hi);
      Print("trunk-", tl, R.tr_lo, 0.3, 3.0, "%.0f");
      Print("trunk+", th_, R.tr_hi, 0.3, 3.0, "%.0f");
      Print("drop", Median(dr), R.drop, 0.3, 0.03, "%.2f");
      printf("   [%d/%d in band]\n", g_in, g_all);
      tot_in += g_in;
      tot_all += g_all;
    }
  {
    int up = 0, nn = 0;
    for (const auto& v : res)
      for (const Res& r : v) { ++nn; up += r.stood && r.yaw <= 45; }
    printf("stood without turn (pelvis yaw <= 45 deg): %d/%d\n", up, nn);
  }
  // Per group stood / without turn / steps, late steps (> 2.5 s) and rest time (CoM speed stays < 0.15
  // m/s) as means over the stood scenes; placement error of every balancer step (sole centre vs place at touch).
  {
    const char* dn[3] = {"back", "fwd", "side"};
    const char* sn[3] = {"small", "medium", "large"};
    for (int dg = 0; dg < 3; ++dg)
      for (int sz = 0; sz < 3; ++sz) {
        int n = 0, st = 0, nt = 0;
        double s_steps = 0, s_late = 0, s_rest = 0;
        for (const Res& r : all) {
          const int d = kScenes[r.scene].dir;
          if ((d == 0 ? 0 : (d == 1 ? 1 : 2)) != dg || SizeGroup(r.dv) != sz) continue;
          ++n;
          if (!r.stood) continue;
          ++st;
          nt += r.yaw <= 45;
          s_steps += r.nsteps;
          s_late += r.late;
          s_rest += r.t_rest;
        }
        if (!n) continue;
        const double q = st ? 1.0 / st : 0.0;
        printf("T33 %-4s %-6s stood %2d/%2d noturn %2d steps %4.1f late %4.1f rest %4.2f s\n", dn[dg], sn[sz], st, n, nt,
               s_steps * q, s_late * q, s_rest * q);
      }
    std::vector<double> el, ea, ew;
    int st = 0, nt = 0;
    double s_steps = 0, s_late = 0, s_rest = 0;
    for (const Res& r : all) {
      for (size_t i = 0; i < r.e_lon.size(); ++i) {
        el.push_back(std::fabs(r.e_lon[i]));
        ea.push_back(std::fabs(r.e_lat[i]));
        ew.push_back(std::hypot(r.e_lon[i], r.e_lat[i]));
      }
      if (!r.stood) continue;
      ++st;
      nt += r.yaw <= 45;
      s_steps += r.nsteps;
      s_late += r.late;
      s_rest += r.t_rest;
    }
    const double q = st ? 1.0 / st : 0.0;
    printf("T33 all  stood %d/%d noturn %d steps %.1f late %.1f rest %.2f s\n", st, (int)all.size(), nt, s_steps * q,
           s_late * q, s_rest * q);
    auto mx = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    {  // pelvis heading change per step (lift -> touchdown) and shoulder flexion range per step
      std::vector<double> yw, sr;
      for (const Res& r : all)
        for (const Step& s : r.all_steps)  // only steps that end before a fall (rolling on the floor is not a step)
          if (r.stood || s.t_lift + s.swing < r.t_fall - 0.2) { yw.push_back(s.dyaw); sr.push_back(s.shr); }
      printf("YAW step yaw (%d steps): median %.1f worst %.1f deg; shoulder range per step median %.1f worst %.1f deg\n",
             (int)yw.size(), Median(yw), mx(yw), Median(sr), mx(sr));
      // arms down after rest (time from the last CoM speed >= 0.15 m/s to shoulder abduction < 15 deg)
      std::vector<double> ad;
      const Res* wr = nullptr;
      for (const Res& r : all)
        if (r.stood) {
          ad.push_back(r.arms_dn);
          if (!wr || r.arms_dn > wr->arms_dn) wr = &r;
        }
      if (wr)
        printf("ARM arms down after rest (%d stood): median %.2f worst %.2f s (%s %.2f) [gate 0.7]\n", (int)ad.size(),
               Median(ad), mx(ad), kScenes[wr->scene].name, wr->moment);
      {  // rest pose 0.6 s after rest
        std::vector<double> a1, a2, a3;
        const Res* wf = nullptr;
        for (const Res& r : all)
          if (r.stood && r.rp_flex >= 0) {
            a1.push_back(r.rp_flex); a2.push_back(r.rp_abd); a3.push_back(r.rp_tw);
            if (!wf || r.rp_flex > wf->rp_flex) wf = &r;
          }
        if (wf)
          printf("REST rest pose +0.6 s (%d): flex median %.1f worst %.1f (%s %.2f), abd %.1f %.1f, twist %.1f %.1f deg "
                 "[gates 8, 10, 3]\n", (int)a1.size(), Median(a1), mx(a1), kScenes[wf->scene].name, wf->moment,
                 Median(a2), mx(a2), Median(a3), mx(a3));
      }
      // the scenes of the 3 largest shoulder ranges (which push swings the arm most)
      std::vector<std::pair<double, std::pair<const Res*, const Step*>>> ws;
      for (const Res& r : all)
        for (const Step& s : r.all_steps)
          if (r.stood || s.t_lift + s.swing < r.t_fall - 0.2) ws.push_back({s.shr, {&r, &s}});
      std::sort(ws.begin(), ws.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      for (size_t i = 0; i < ws.size() && i < 3; ++i)
        printf("SHO shoulder worst %zu: %.1f deg, %s %.2f s, step lift %.2f s, foot %d, %s\n", i + 1, ws[i].first,
               kScenes[ws[i].second.first->scene].name, ws[i].second.first->moment, ws[i].second.second->t_lift,
               ws[i].second.second->foot, ws[i].second.first->stood ? "stood" : "fell");
    }
    printf("T33 place error (%d steps): along median %.3f worst %.3f, across median %.3f worst %.3f, total median %.3f "
           "worst %.3f m [gate median <= 0.03, worst <= 0.06]\n", (int)el.size(), Median(el), mx(el), Median(ea), mx(ea),
           Median(ew), mx(ew));
    // Overshoot, peak sole speed, swing time, empty steps (< 8 cm) per push, pelvis yaw range per step; per
    // group of the table and for all scenes.
    for (int g = -1; g < 9; ++g) {
      std::vector<double> ov, vp, du, yw;
      int n = 0, ns = 0, empty = 0, empty_s = 0;
      double one = 0, one_s = 0;
      for (const Res& r : all) {
        const int d = kScenes[r.scene].dir, dg = d == 0 ? 0 : (d == 1 ? 1 : 2), sz = SizeGroup(r.dv);
        if (g >= 0 && dg * 3 + sz != g) continue;
        ++n;
        ns += r.stood;
        one += r.t_one;
        if (r.stood) one_s += r.t_one;
        for (size_t i = 0; i < r.e_over.size(); ++i) {
          ov.push_back(r.e_over[i]);
          vp.push_back(r.e_vpk[i]);
          du.push_back(r.e_dur[i]);
          yw.push_back(r.e_yaw[i] * 57.29578);
          const int e = r.e_len[i] < 0.08;
          empty += e;
          if (r.stood) empty_s += e;
        }
      }
      if (!n || (g >= 0 && g % 3 == 2)) continue;  // large groups skipped (not stood in the table)
      const char* gn[9] = {"back small", "back medium", "", "fwd small", "fwd medium", "", "side small", "side medium", ""};
      printf("T34 %-12s over median %.3f worst %.3f m | vpeak %.2f m/s | swing %.2f s | empty <8cm per push %.2f "
             "(stood %.2f) | yaw range median %.0f worst %.0f deg | one foot at rest %.2f s per push (stood %.2f) "
             "[gates over 0.02/0.05, empty 0.5, one foot 0.15]\n",
             g < 0 ? "all" : gn[g], Median(ov), mx(ov), Median(vp), Median(du), (double)empty / n,
             ns ? (double)empty_s / ns : 0.0, Median(yw), mx(yw), one / n, ns ? one_s / ns : 0.0);
    }
  }
  printf("stood %d/%d (table: all, except part of the largest sideways, 6 of 24; ours sideways large %.0f %%); %d/%d medians inside the band\n",
         stood_n, scene_n, 100.0 * side_large_stood, tot_in, tot_all);
  // Large pushes: first step ankle ahead of the CoM along the push at touchdown (gate >= 0.10 m in 9 of 10),
  // knee at the first two touchdowns (median 20-40 deg), pelvis drop of the stood (<= 0.20 m), steps of the stood.
  for (int dg = 0; dg < 3; ++dg) {
    std::vector<double> ah, kn, dr;
    int n = 0, ns = 0, ok = 0, st_sum = 0;
    for (const Res& r : all) {
      const int dd = kScenes[r.scene].dir, g = dd == 0 ? 0 : (dd == 1 ? 1 : 2);
      if (g != dg || SizeGroup(r.dv) != 2) continue;
      ++n;
      if (r.have > 0) {
        ah.push_back(r.st[0].ahead);
        ok += r.st[0].ahead >= 0.10;
      }
      for (int q = 0; q < r.have; ++q) kn.push_back(r.st[q].knee);
      if (r.stood) {
        ++ns;
        dr.push_back(r.drop);
        st_sum += r.nsteps;
      }
    }
    if (!n) continue;
    const char* gn[3] = {"back", "fwd", "side"};
    double dmax = 0;
    for (double x : dr) dmax = std::fmax(dmax, x);
    printf("T35 %-4s large stood %2d/%2d | first step ahead of CoM median %.2f m, >= 0.10 m %d of %d | knee at "
           "touchdown median %.0f deg | stood: drop median %.2f worst %.2f m, steps %.1f\n", gn[dg], ns, n,
           ah.empty() ? 0.0 : Median(ah), ok, (int)ah.size(), kn.empty() ? 0.0 : Median(kn), dr.empty() ? 0.0 : Median(dr),
           dmax, ns ? (double)st_sum / ns : 0.0);
  }
  {  // Standing foot turn on the floor (target <= 5 deg in every scene)
    std::vector<double> sp;
    int over = 0;
    for (const Res& r : all) {
      sp.push_back(r.fspin);
      over += r.fspin > 5.0;
    }
    std::sort(sp.begin(), sp.end());
    printf("T43 standing foot turn: median %.1f max %.1f deg, > 5 deg in %d of %d scenes\n", sp.empty() ? 0.0 : Median(sp),
           sp.empty() ? 0.0 : sp.back(), over, (int)sp.size());
  }
  {  // Flat loaded foot, ankle -> toe heading turn over 0.2 s (target <= 5 deg)
    int o5 = 0, o10 = 0, ip5 = 0, ax5 = 0, rl8 = 0, lim = 0, ips = 0, ipd = 0, ipl = 0, ipr = 0, ipn = 0, ipse = 0, ipti = 0;
    double axmax = 0;
    double wmax = 0;
    for (const Res& r : all) {
      o5 += r.spin02 > 5.0;
      o10 += r.spin02 > 10.0;
      ip5 += r.spin02_ip > 5.0;
      ax5 += r.spin02_axip > 5.0;
      axmax = std::fmax(axmax, r.spin02_axip);
      rl8 += r.roll_frames > 0;
      ips += r.ip_single;
      ipd += r.ip_double;
      ipl += r.ip_lim;
      ipr += r.ip_roll;
      ipn += r.ip_noroll;
      ipse += r.ip_settle;
      ipti += r.ip_tip;
      lim += r.lim_frames > 0;
      wmax = std::fmax(wmax, r.spin02);
    }
    printf("T44 flat foot turn in 0.2 s: > 5 deg in %d, > 10 in %d of %d scenes (in place, heel and toe < 3 cm: > 5 in %d), "
           "worst %.1f deg; sole axis in place > 5 in %d, worst %.1f; loaded sole rolled > 8 deg in %d, ankle at its "
           "inversion range end in %d\n", o5, o10, (int)all.size(), ip5, wmax, ax5, axmax, rl8, lim);
    printf("T44 frames ending an in-place window > 5 deg: one foot standing %d, two %d; ankle at its inversion range end %d; "
           "sole roll changed > 4 deg in the window %d, less %d; toward flat %d, away %d\n", ips, ipd, ipl, ipr, ipn, ipse, ipti);
  }
  // Swing height by direction and size: highest ankle rise of the first and second step
  // (median, cm), knee at the first two touchdowns (median, deg). Target: 13-18 cm forward and
  // sideways, back 1.62: 13 then 22 cm.
  for (int dg = 0; dg < 3; ++dg) {
    for (int sg = 0; sg < 3; ++sg) {
      std::vector<double> h1, h2, kn;
      int n = 0, ns = 0;
      for (const Res& r : all) {
        const int dd = kScenes[r.scene].dir, g = dd == 0 ? 0 : (dd == 1 ? 1 : 2);
        if (g != dg || SizeGroup(r.dv) != sg) continue;
        ++n;
        ns += r.stood;
        if (r.have > 0) h1.push_back(100 * r.st[0].hmax);
        if (r.have > 1) h2.push_back(100 * r.st[1].hmax);
        for (int q = 0; q < r.have; ++q) kn.push_back(r.st[q].knee);
      }
      if (!n) continue;
      const char* gn[3] = {"back", "fwd", "side"};
      const char* sn[3] = {"small", "medium", "large"};
      printf("T38 %-4s %-6s stood %2d/%2d | ankle rise step 1 %4.1f cm, step 2 %4.1f cm | knee at touchdown %3.0f deg\n",
             gn[dg], sn[sg], ns, n, h1.empty() ? 0.0 : Median(h1), h2.empty() ? 0.0 : Median(h2),
             kn.empty() ? 0.0 : Median(kn));
    }
  }
  printf("details: %s/scenes.csv\n", dir.c_str());
  return 0;
}
