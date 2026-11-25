// L0 muscles: per-joint spring toward the balancer target with target-velocity feedforward, plus support torques.
// Net muscle torque = kp (q* - q) + d (qd* - qd) + support, limited to the muscle strength. The damping d is MuJoCo's
// passive dof_damping (integrated implicitly by Euler, stable at high gains); the motor adds d qd back, so the limit
// applies to the net torque and the joint viscosity never caps a fast swing (with the damping outside the limit the
// damping torque would eat the motor limit and cap the swing speed of the knee and the hip).
// Support: the joints carry the body weight only along the line support -> CoM. Per standing foot a
// force share * (W . u) u at its support point (CoP inside the sole), u = unit(CoM - share-weighted support point),
// turned into joint torques along the leg chain (J^T). It holds the weight and does not resist tipping; with the CoM
// over the soles u is vertical (no sideways squeeze of the feet).
// Upper body: qfrc_bias (load of the distal subtree).
#include <cmath>

#include "emil_world.h"

namespace emil {
namespace {
constexpr double kVfZmp = 0.8;        // share of the foot lever usable by VfTorque's ankle part (selftest formula)
// Passive end range: hip capsule torsional stiffness 0.8 N m/deg (cadaver study, PMC4655836, see SOURCES; used for
// the hip and the knee), engaged in the last 10 deg of the range (own choice), only below this leg tone (a living
// body is not affected)
constexpr double kEndK = 0.8 * 57.29578, kEndZone = 10.0 / 57.29578, kEndTone = 0.3;
constexpr double kPassiveDamp = 1.0;  // [N m s/rad] minimum joint damping (relaxed limbs must not chatter), bench-tuned
// Standing knee weakens from the angle where the full knee just holds the body on one leg (a deeply
// bent leg does not push the body back up, the body settles; a strut step below it holds the weight). Hold torque
// M g L sin(q/2) for thigh ~ shank = L: 220 N m, 79 kg, L 0.44 m -> 80 deg; with kKneeHold 0.9 the start is 71 deg
// (a weakening that starts much earlier would fold the knee of a 50-70 deg catching step). End: deep squat.
constexpr double kKneeHold = 0.9, kKneeWeak1 = 2.0;
// Force-velocity of the leg muscles: torque limit x m(w), w = joint speed in the torque's
// direction (> 0 shortening). Shortening: Hill (1 - r) / (1 + r / kFvA), r = w / wmax (Hill 1938; curvature 0.25 =
// Thelen 2003 Af, OpenSim default). Lengthening: kFvEcc - (kFvEcc - 1) kFvE / (kFvE + u), u = -w / wmax, plateau
// 1.4 (Thelen 2003 Flen, OpenSim default); kFvE: slope at zero twice the shortening one (own choice). wmax = 10
// optimal fibre lengths / s x fibre length / moment arm: ~20 rad/s hip, 18 knee, 10 ankle (see SOURCES).
constexpr double kFvA = 0.25, kFvEcc = 1.4;
constexpr double kFvE = (kFvEcc - 1) / (2 * (1 + 1 / kFvA));
constexpr double kFvWmax[6] = {20, 20, 20, 18, 10, 10};  // hip flex, abd, rot, knee, ankle dorsi, inv [rad/s]
// Anti-spin torque of a standing foot: stiffness and damping at 80 kg, cap = kSpinMu x normal force (sole torsional
// friction: friction 0.8 x effective radius ~0.06 m of a 0.26 x 0.10 m sole).
// A larger cap does not stop the turning of a foot standing on one corner: that turning torque comes through the
// tilted leg, not through the hip twist.
constexpr double kSpinK = 300.0, kSpinD = 30.0, kSpinMu = 0.05;
constexpr double kArm0 = 0.01;  // joint armature of the model default (emil_world.cpp); legs get b.arm when set
constexpr double kLowSole = 0.03;  // [m] a standing foot's sole this low counts for the weight share without load
                                   // (a foot hovering a few mm above the floor would otherwise carry no share)
constexpr double kSole = 0.8;  // support point at most this share of the half sole from its centre (edge margin)
// Whole-body inertia about a horizontal axis ~ kAirI M H^2 (radius of gyration ~0.28 H of a standing body: own
// estimate, 80 kg and 1.78 m give 19 kg m^2), natural frequency and angular acceleration cap of the flight torque (own
// choice: 90 deg turned in about 0.6 s)
constexpr double kAirI = 0.075, kAirWn = 6.0, kAirAlpha = 25.0, kAirDead = 0.5, kAirDeadInv = 1.0 / 0.3;
constexpr double kAirTqZ0 = 0.8, kAirTqZ1 = 1.2;  // [m] foot height where the flight torque fades in
constexpr double kRollDead = 0.1;  // roll landing: dead zone 6 deg around the leaned-back target
inline double Clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
}  // namespace

void World::Muscles(Bot& b) {
  {  // stay-upright pair force, computed in Balance: pelvis +F, feet -F by load share; sum zero
    double* fp = d_->xfrc_applied + 6 * (b.body0 + B_PELVIS);
    double* fl = d_->xfrc_applied + 6 * (b.body0 + B_FOOT_L);
    double* fr = d_->xfrc_applied + 6 * (b.body0 + B_FOOT_R);
    for (int i = 0; i < 2; ++i) {
      fp[i] = b.assist_f[i];
      fl[i] = -b.assist_f[i] * b.assist_w[0];
      fr[i] = -b.assist_f[i] * b.assist_w[1];
    }
    // asst_yaw: the pair acts at the pelvis CoM and the feet CoMs; a standing foot 10+ cm beside the pelvis makes
    // it a yaw couple (under the bullet pull it twists the standing foot and the sole creeps round). The help is
    // meant to right the body, not to turn it: a pure pelvis yaw torque cancels the pair's yaw moment (forces sum to
    // zero, so the moment is the same about any point).
    // Pushes keep the yaw moment: there it counters the yaw of braking on one standing foot beside the CoM (cancelled,
    // the body turns more and takes more steps); under the bullet pull (help x bul_assist) it twists the foot.
    if (param[P_ASST_YAW] > 1.5 || (param[P_ASST_YAW] > 0.5 && d_->time < b.pull_until)) {
      const double* xp = d_->xipos + 3 * (b.body0 + B_PELVIS);
      const double* xl = d_->xipos + 3 * (b.body0 + B_FOOT_L);
      const double* xr = d_->xipos + 3 * (b.body0 + B_FOOT_R);
      fp[5] = -((xp[0] * fp[1] - xp[1] * fp[0]) + (xl[0] * fl[1] - xl[1] * fl[0]) + (xr[0] * fr[1] - xr[1] * fr[0]));
    } else {
      fp[5] = 0;
    }
  }
  {  // In flight a torque on the pelvis turns it to feet down (up axis vertical), critically damped; the body is
     // not a rigid wheel, the joints carry the trunk and legs along. The torque is external (an own design choice for
     // a controlled landing); capped by an angular acceleration of the whole body.
    double* fp = d_->xfrc_applied + 6 * (b.body0 + B_PELVIS);
    fp[3] = fp[4] = 0;
    // only high above the floor (a drop or a fall off an edge): short flights of a push, a hop or a car hit keep their
    // own dynamics
    const double zf = std::fmin(d_->xpos[3 * (b.body0 + B_FOOT_L) + 2], d_->xpos[3 * (b.body0 + B_FOOT_R) + 2]) - b.gz;  // over the ground
    const double hg = Clamp((zf - kAirTqZ0) * (1.0 / (kAirTqZ1 - kAirTqZ0)), 0.0, 1.0);
    if (b.airborne && param[P_AIR_ROT] > 0 && hg > 0) {
      const double* R = d_->xmat + 9 * (b.body0 + B_PELVIS);
      // target up axis: vertical, or leaning back by kRollTilt along the heading for a roll landing
      double ux = 0, uy = 0, uz = 1;
      if (b.air_roll) {
        const double hn = std::hypot(R[0], R[3]) + 1e-9;
        ux = -std::sin(param[P_LAND_ROLL]) * R[0] * (1.0 / hn);
        uy = -std::sin(param[P_LAND_ROLL]) * R[3] * (1.0 / hn);
        uz = std::cos(param[P_LAND_ROLL]);
      }
      const double zx = R[2], zy = R[5], zz = R[8];
      // pelvis up axis cross target (x, y parts: the turn about horizontal axes), angle between them
      const double ax = zy * uz - zz * uy, ay = zz * ux - zx * uz;
      const double sn = std::hypot(ax, ay), th = std::acos(Clamp(zx * ux + zy * uy + zz * uz, -1.0, 1.0));
      const double* w = d_->cvel + 6 * (b.body0 + B_PELVIS);  // angular velocity (world axes)
      const double hh = b.body.height;
      const double inertia = kAirI * m_->body_subtreemass[b.body0] * hh * hh;
      const double wn = kAirWn * param[P_AIR_ROT];
      // dead zone kAirDead (14 deg): an upright body flies in its own pose (a torque there would upset the landing of
      // a drop it can stand); the error grows from zero at the edge and the damping fades in with it: no jerk at the
      // start
      const double e = std::fmax(th - (b.air_roll ? kRollDead : kAirDead), 0.0), g = Clamp(e * kAirDeadInv, 0.0, 1.0) * (1.0 - b.air_cw);  // the curl pose takes over
      double tx = -2 * wn * inertia * w[0] * g, ty = -2 * wn * inertia * w[1] * g;
      if (sn > 1e-6) {
        tx += inertia * wn * wn * e * (1.0 - b.air_cw) * ax * (1.0 / sn);
        ty += inertia * wn * wn * e * (1.0 - b.air_cw) * ay * (1.0 / sn);
      }
      tx *= hg;
      ty *= hg;
      const double cap = inertia * kAirAlpha * param[P_AIR_ROT], tn = std::hypot(tx, ty);
      const double k = tn > cap ? cap * (1.0 / tn) : 1.0;
      fp[3] = k * tx;
      fp[4] = k * ty;
    }
  }
  const double* com = d_->subtree_com + 3 * b.body0;
  const double M = m_->body_subtreemass[b.body0];
  const double* g = m_->opt.gravity;
  const double ratio = param[P_DAMP_RATIO];
  // Standing feet (touching, not the swing leg) and their weight shares by the CoM between the sole centres. Each
  // foot's support point is its sole centre shifted by the CoM offset from the share-weighted centre, clipped into the
  // sole (the CoP each foot can give): the weight is held where the CoM is while it is over the soles, and the force
  // line tilts (no resistance to tipping) once the CoM leaves them.
  double gp[2][3], al[2] = {0, 0}, sc[2][3];
  int st[2];
  for (int f = 0; f < 2; ++f) {
    const int gf = b.geom_foot[f];
    for (int k = 0; k < 3; ++k) sc[f][k] = d_->geom_xpos[3 * gf + k];
    sc[f][2] -= m_->geom_size[3 * gf + 2];  // sole bottom
    // A standing leg (not the swing) shares the weight by geometry even when its foot hangs just above the floor:
    // its support torque then reaches down to the floor and the loaded leg gives way to it.
    st[f] = b.stand_leg[f] && (b.ld[f] > kTouch || sc[f][2] - b.gz < kLowSole);  // over the support
  }
  if (st[0] && st[1]) {
    const double ax = sc[1][0] - sc[0][0], ay = sc[1][1] - sc[0][1];
    const double s = Clamp(((com[0] - sc[0][0]) * ax + (com[1] - sc[0][1]) * ay) / (ax * ax + ay * ay + 1e-9), 0.0, 1.0);
    al[0] = 1 - s;
    al[1] = s;
  } else if (st[0]) {
    al[0] = 1;
  } else if (st[1]) {
    al[1] = 1;
  }
  b.al_dbg[0] = al[0];
  b.al_dbg[1] = al[1];
  // cop_k: plus the balance numbers' shift from L1 (zero when off), clipped into the soles below
  const double ox = com[0] - (al[0] * sc[0][0] + al[1] * sc[1][0]) + b.cop_sh[0];
  const double oy = com[1] - (al[0] * sc[0][1] + al[1] * sc[1][1]) + b.cop_sh[1];
  for (int f = 0; f < 2; ++f) {
    const int gf = b.geom_foot[f];
    const double* R = d_->geom_xmat + 9 * gf;
    const double* hz = m_->geom_size + 3 * gf;
    double lx = R[0] * ox + R[3] * oy, ly = R[1] * ox + R[4] * oy;  // offset in the foot frame
    lx = Clamp(lx, -kSole * hz[0], kSole * hz[0]);
    ly = Clamp(ly, -kSole * hz[1], kSole * hz[1]);
    gp[f][0] = sc[f][0] + R[0] * lx + R[1] * ly;
    gp[f][1] = sc[f][1] + R[3] * lx + R[4] * ly;
    gp[f][2] = sc[f][2];
  }
  // Foot does not spin: a standing foot gets a torque about the vertical against turning away from
  // the heading it stood down with, within what the sole's torsional friction carries (condim 3 has none).
  for (int f = 0; f < 2; ++f) {
    double* xf = d_->xfrc_applied + 6 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
    const double* R = d_->xmat + 9 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
    if (!st[f] || b.mode == 2) { xf[5] = 0; b.foot_st[f] = 0; continue; }
    const double yaw = std::atan2(R[3], R[0]);
    if (!b.foot_st[f]) { b.foot_yaw0[f] = yaw; b.foot_st[f] = 1; }
    double e = yaw - b.foot_yaw0[f];
    while (e > 3.14159265) e -= 6.2831853;
    while (e < -3.14159265) e += 6.2831853;
    const double wz = d_->cvel[6 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L)) + 2];
    const double cap = kSpinMu * param[P_SPIN_CAP] * b.ld[f] * M * 9.81;
    xf[5] = Clamp(-param[P_SPIN_K] * M / 80.0 * e - kSpinD * M / 80.0 * wz, -cap, cap);
  }
  double F[2][3] = {{0, 0, 0}, {0, 0, 0}};
  if (al[0] + al[1] > 0) {
    double u[3];
    for (int k = 0; k < 3; ++k) u[k] = com[k] - (al[0] * gp[0][k] + al[1] * gp[1][k]);
    const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) + 1e-9;
    for (int k = 0; k < 3; ++k) u[k] /= un;
    if (param[P_SUP_LINE] < 0.5) { u[0] = u[1] = 0; u[2] = 1; }  // bench switch: vertical support force at the CoP
    double Wu = M * (g[0] * u[0] + g[1] * u[1] + g[2] * u[2]);
    // sup_brake: a leg damper along the CoM line (spring-loaded inverted pendulum, Blickhan 1989): while the CoM
    // moves toward the support point the legs push it back along that line, which brakes the body with no moment
    // about the CoM. Without it the stance hip's joint damping brakes a push by turning the trunk back, and without
    // that damping the landing leg carries too little of the weight and the body sits down.
    if (param[P_SUP_BRAKE] > 0 && b.mode != 2) {
      const double* cv = b.pc[b.cur].com_vel;
      const double vu = cv[0] * u[0] + cv[1] * u[1] + cv[2] * u[2];  // < 0: the CoM approaches the support
      if (vu < 0) Wu -= std::fmin(M * param[P_SUP_BRAKE] * -vu, param[P_SUP_BRAKE_MAX] * M * 9.81);
    }
    for (int f = 0; f < 2; ++f)
      for (int k = 0; k < 3; ++k) F[f][k] = al[f] * Wu * u[k];
  }
  for (int j = 0; j < J_COUNT; ++j) {
    const int dof = b.dof0 + j;
    const double q = d_->qpos[b.qadr0 + j], qd = d_->qvel[dof];
    const double tone = Clamp(j >= J_HIP_FLEX_L ? std::fmax(b.in.tone, b.leg_tone)
                                                : (j >= J_SH_FLEX_L ? std::fmax(b.in.tone, b.arm_tone) : b.in.tone),
                              0.0, 1.0);  // L2: muscles fade (death); scales drive, damping, support
    const double d = std::fmax(b.kp[j] * ratio * b.zm[j] * tone, kPassiveDamp);
    m_->dof_damping[dof] = d;
    m_->dof_armature[dof] = b.arm[j] > 0 ? b.arm[j] : kArm0;  // flywheel (legs: L1; 0 = model default)
    // upper layer: b.gsup = share of the weight support (gravity and bias) the muscle gives; arms hang.
    double tau = b.kp[j] * (b.target[j] - q) + d * b.qd_tgt[j] + b.gsup[j] * d_->qfrc_bias[dof];
    double lim = b.js[j].tau_max * tone * (b.limf[j] > 0 ? b.limf[j] : 1.0);  // x flywheel growth
    int f = -1, jl = -1;
    if (j >= J_HIP_FLEX_L) {
      f = j >= J_HIP_FLEX_R ? 1 : 0;
      jl = j - (f ? J_HIP_FLEX_R : J_HIP_FLEX_L);
      // asst_comp: with asst_split half the pair's reaction (x bul_assist under a bullet: up to 150 N) pushes a foot
      // in the air (~10 kg leg) off its swing target; the foot hangs up while the pelvis, pulled alone, turns over the
      // standing foot. The leg's muscles put the opposite
      // force on the foot (joint torque axis . (r x F w), r from the joint to the foot CoM where the pair acts): the
      // swing keeps its target and the whole body takes the pair, which has no yaw moment about the standing foot.
      // asst_comp 2: only a foot that is not the swing (a standing foot out of reach); the swing foot keeps the push,
      // which drives a backward step along (cancelled, the swings after a hit get slower).
      // asst_comp 3: any unloaded foot, only the part across the last hit's direction (the part along it drives the
      // backward step after a front hit and is kept)
      const double cm = param[P_ASST_COMP];
      if (cm > 0 && b.assist_w[f] > 0 && !b.loaded[f] && (cm < 1.5 || cm > 2.5 || f != b.swing)) {
        const double* o = d_->xanchor + 3 * (b.jnt0 + j);
        const double* ax = d_->xaxis + 3 * (b.jnt0 + j);
        const double* pf = d_->xipos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        const double rx = pf[0] - o[0], ry = pf[1] - o[1], rz = pf[2] - o[2];
        const double w = b.assist_w[f];
        double fx = b.assist_f[0] * w, fy = b.assist_f[1] * w;  // force of the leg on the foot (+F w), horizontal
        if (cm > 2.5) {
          const double hn = std::hypot(b.hit_dir[0], b.hit_dir[1]);
          if (hn > 1e-6) {
            const double ux = b.hit_dir[0] / hn, uy = b.hit_dir[1] / hn, al = fx * ux + fy * uy;
            fx -= al * ux;
            fy -= al * uy;
          }
        }
        tau += ax[0] * (-rz * fy) + ax[1] * (rz * fx) + ax[2] * (rx * fy - ry * fx);
      }
      lim *= b.in.leg_weak[f];
      if (al[f] > 0) {
        const double* p = d_->xanchor + 3 * (b.jnt0 + j);
        const double* a = d_->xaxis + 3 * (b.jnt0 + j);
        const double r[3] = {gp[f][0] - p[0], gp[f][1] - p[1], gp[f][2] - p[2]};
        const double* Ff = F[f];
        const double G[3] = {r[1] * Ff[2] - r[2] * Ff[1], r[2] * Ff[0] - r[0] * Ff[2], r[0] * Ff[1] - r[1] * Ff[0]};
        // down_supk: a body going down (wounded or dead) loses the weight support faster than the drive:
        // the support is scaled by the tone to the power down_supk on top of the tone itself
        const double sk = param[P_DOWN_SUPK] > 0 && b.dead_t >= 0 && b.down >= 1
                              ? std::pow(Clamp(tone, 0.0, 1.0), (double)param[P_DOWN_SUPK])
                              : 1.0;
        tau += sk * (a[0] * G[0] + a[1] * G[1] + a[2] * G[2]);
        if (jl == 3) {
          // fv on: weakening from the angle where the yielding (eccentric) knee no longer holds the body (109 deg)
          // to the knee limit; the catching range 60-100 deg keeps the full limit.
          const double Lm = 0.5 * (b.dims.thigh + b.dims.shank);
          const double ecc = param[P_FV] > 0 ? 1 + param[P_FV] * (kFvEcc - 1) : 1.0;
          const double q0 = 2 * std::asin(std::fmin(1.0, kKneeHold * ecc * b.js[j].tau_max / (M * 9.81 * Lm)));
          const double q1 = param[P_FV] > 0 ? b.js[j].hi : kKneeWeak1;
          const double w = Clamp((q - q0) / std::fmax(q1 - q0, 0.1), 0.0, 1.0);
          lim *= 1.0 - param[P_KNEE_WEAK] * w * w * (3 - 2 * w);
        }
      }
    }
    // act_t: muscle activation dynamics, a first-order lag of the leg drive torque (excitation -> activation,
    // Thelen 2003 rise 15 ms). Without it a swing start puts the full knee torque on in one substep: a huge knee
    // acceleration spike in a single frame.
    if (f >= 0 && param[P_ACT_T] > 0 && b.mode != 2) {
      b.act_tq[j] += (tau - b.act_tq[j]) * std::fmin(1.0, m_->opt.timestep / param[P_ACT_T]);
      tau = b.act_tq[j];
    } else {
      b.act_tq[j] = tau;
    }
    tau *= tone;
    // force-velocity: knee; hip flexion and abduction with fv_hip (lengthening only, fv_con 0)
    if ((jl == 3 || ((jl == 0 || jl == 1) && param[P_FV_HIP] > 0)) && param[P_FV] > 0) {
      const double net0 = tau - d * qd;
      const double w = net0 >= 0 ? qd : -qd;
      double mv;
      if (w >= 0) {
        const double r = std::fmin(w / kFvWmax[jl], 1.0);
        mv = 1 + param[P_FV_CON] * ((1 - r) / (1 + r / kFvA) - 1);
      } else {
        const double u = -w / kFvWmax[jl];
        mv = kFvEcc - (kFvEcc - 1) * kFvE / (kFvE + u);
      }
      lim *= 1 + param[P_FV] * (mv - 1);
    }
    if (jl == 2 && al[f] > 0 && b.mode != 2 && param[P_ST_TWIST] > 0) {
      // The standing hip twist passes to the foot at most what the sole's torsional friction holds (effective radius
      // 3-5 cm x normal force), so the body turns in the hip over the foot instead of spinning it.
      lim = std::fmin(lim, param[P_ST_TWIST] * std::fmax(b.foot_fn[f], 0.0));
    }
    double net = Clamp(tau - d * qd, -lim, lim);
    if (jl == 0 || jl == 1) b.net_dbg[(jl == 1 ? 2 : 0) + f] = net;
    if (jl == 3) {
      b.net_dbg[4 + f] = net;
      b.net_dbg[6 + f] = lim;
    }
    if (f >= 0 && (jl == 4 || jl == 5) && al[f] > 0 && b.mode != 2) {
      // Standing foot: ankle torque within what the sole carries (CoP inside the sole, Kajita ZMP); more only rolls
      // the foot onto its toe or edge.
      // zmp_fn: the sole carries what presses it, not the weight share: a falling body presses less and an ankle
      // torque from the share rolls the foot onto its toe.
      const double fw = al[f] * M * 9.81;
      // zmp_fn_inv: sideways only. A window from the weight share lets the CoP past the sole edge whenever the sole
      // presses less than that share, and the foot tips onto its edge. Fore-aft the full window is kept: applied to
      // both joints, it weakens the fore-aft ankle strategy.
      const bool meas = param[P_ZMP_FN] > 0 || (jl == 5 && param[P_ZMP_FN_INV] > 0);
      const double fn = meas ? std::fmin(fw, b.foot_fn[f]) : fw, kz = param[P_ZMP_FRAC];
      const BodyDims& dm = b.dims;
      double lo = jl == 4 ? -kz * fn * (dm.foot_len - dm.heel_back) : -kz * fn * 0.5 * dm.foot_w;
      double hi = jl == 4 ? kz * fn * dm.heel_back : kz * fn * 0.5 * dm.foot_w;
      // roll_guard: bullets, one foot standing: the lateral strategy keeps the CoP at the window edge; a load dip
      // tips the sole onto its edge and the same torque rolls it on (the restoring lever of the load shrinks as the
      // ankle comes over the edge). The side the sole already tilts to (about this joint's axis) fades to zero at the
      // guard tilt, so the load rolls it back flat.
      // pitch_guard: the same fore-aft, heel side only. Backing off under a bullet the brake puts the CoP at the front
      // foot's heel edge and rolls the sole onto its heel, toe up; a person rolls off the heel with the toe low. Live
      // bullet reactions only: under a car hit a clipped heel side would hold the body up longer and make the fall
      // that follows harder.
      const bool pg_on = b.hit_bullet && d_->time < b.bul_until;
      const double gd = jl == 5 ? param[P_ROLL_GUARD] : (pg_on ? (double)param[P_PITCH_GUARD] : 0.0);
      if (gd > 0) {
        const double* ax = d_->xaxis + 3 * (b.jnt0 + j);
        const double* Rg = d_->geom_xmat + 9 * b.geom_foot[f];
        const double dl = Rg[2] * ax[1] - Rg[5] * ax[0];  // sole tilt about the axis [rad], + = the way + torque turns it
        if (dl > 0) hi *= std::fmax(0.0, 1.0 - dl / gd);
        else if (jl == 5) lo *= std::fmax(0.0, 1.0 + dl / gd);  // fore-aft: heel side only (toe roll = push strategy)
      }
      // ank_fh: the floor's horizontal force acts at the sole, ankle_h below the ankle, and tips the foot by
      // ankle_h x force (a strong sideways brake rolls a trailing foot onto its inner edge even with the CoP limit
      // met). The window is centred on the ankle torque that holds that moment (CoP under the ankle):
      // tau_h = -(0, 0, -ankle_h) x F_h, projected on this joint's axis.
      double sh = 0;
      if (param[P_ANK_FH] > 0) {
        const double* ax = d_->xaxis + 3 * (b.jnt0 + j);
        sh = dm.ankle_h * (-b.foot_fh[f][1] * ax[0] + b.foot_fh[f][0] * ax[1]);
      }
      net = Clamp(net, lo + sh, hi + sh);
    }
    {
      // The three-number muscles raise their limit with the flywheel and add the larger damping back; the 4x motor
      // range would clip that (a saturated joint would move at a constant speed), so it widens with them. The table
      // muscles keep the 4x range (their fast swings are tuned with it).
      const double cr = ctrl0_[b.act0 + j] * (b.limf[j] > 1.0 ? 10.0 * b.limf[j] : 1.0);
      m_->actuator_ctrlrange[2 * (b.act0 + j)] = -cr;
      m_->actuator_ctrlrange[2 * (b.act0 + j) + 1] = cr;
    }
    // Passive end range of a limp leg (hip and knee): capsule and tissue push back in the last kEndZone of the range
    // (without it a dead body's leg can lie jammed at several joint limits at once, the foot far off the floor)
    if (jl >= 0 && jl <= 3 && tone < kEndTone) {
      const double eh = q - (b.js[j].hi - kEndZone), el = (b.js[j].lo + kEndZone) - q;
      if (eh > 0) net -= kEndK * eh;
      if (el > 0) net += kEndK * el;
    }
    d_->ctrl[b.act0 + j] = net + d * qd;
  }
}

// Joint torque realising a virtual CoM force F through the leg of joint j (J^T: -a.((c - p) x F)), for a leg
// carrying the whole force. Not used by the balancer (balance is made by positions); kept as the J^T formula the
// selftest checks. Ankle part capped by what the foot can hold (ZMP inside the sole, Kajita 2003).
// closed: both feet on the floor; hips act on the leg strut l = hip - ankle (tau = a.(l x F)), knees take no part.
double World::VfTorque(const Bot& b, int j, const double F[3], bool closed) const {
  const double* com = d_->subtree_com + 3 * b.body0;
  const int jid = b.jnt0 + j;
  const double* p = d_->xanchor + 3 * jid;
  const double* a = d_->xaxis + 3 * jid;
  const int jl = j - (j >= J_HIP_FLEX_R ? J_HIP_FLEX_R : J_HIP_FLEX_L);
  if (closed && jl <= 3) {
    if (jl == 3) return 0.0;
    const double* pa = d_->xanchor + 3 * (b.jnt0 + j - jl + 4);  // this leg's ankle
    const double l[3] = {p[0] - pa[0], p[1] - pa[1], p[2] - pa[2]};
    const double W[3] = {l[1] * F[2] - l[2] * F[1], l[2] * F[0] - l[0] * F[2], l[0] * F[1] - l[1] * F[0]};
    return a[0] * W[0] + a[1] * W[1] + a[2] * W[2];
  }
  const double r[3] = {com[0] - p[0], com[1] - p[1], com[2] - p[2]};
  const double V[3] = {r[1] * F[2] - r[2] * F[1], r[2] * F[0] - r[0] * F[2], r[0] * F[1] - r[1] * F[0]};
  double tau = -(a[0] * V[0] + a[1] * V[1] + a[2] * V[2]);
  if (j >= J_HIP_FLEX_L && jl >= 4) {
    const double fn = b.ld[j >= J_HIP_FLEX_R ? 1 : 0] * b.dims.total_mass * 9.81;  // smoothed (raw chatters)
    const BodyDims& dm = b.dims;
    double lo, hi;
    if (jl == 4) {  // dorsi: + moves the CoP to the heel, - to the toes
      lo = -kVfZmp * fn * (dm.foot_len - dm.heel_back);
      hi = kVfZmp * fn * dm.heel_back;
    } else {        // inversion: CoP to the foot edge, half width either way
      hi = kVfZmp * fn * 0.5 * dm.foot_w;
      lo = -hi;
    }
    tau = tau < lo ? lo : (tau > hi ? hi : tau);
  }
  return tau;
}

}  // namespace emil
