// L1 upper body while balancing, and its looseness after a hit. Runs every control
// step right after the leg balancer and before the reactions (L2). It reads the legs and the body (desired and actual
// hip angles, chest tilt and its rate, CoM velocity, foot contacts) and writes targets and stiffness of the back, neck
// and arms on top of what the balancer wrote. The arms and the back do not hold a pose: they follow the legs and the
// motion of the body. When the balancer gave up, a fall is predicted, the body is in the air or curled, the layer does
// nothing and L2 drives the upper body.
#include <cmath>

#include "emil_world.h"

namespace emil {

namespace {
// Arm in waiting: chest lean toward the CoM motion (sin) from-to, CoM speed from-to [m/s], arm toward the lean
// direction: flexion per forward share, abduction per side share, elbow [rad] (protective arm posture before a fall)
constexpr double kGuardLean0 = 0.35, kGuardLean1 = 0.65, kGuardV0 = 0.8, kGuardV1 = 1.4;
constexpr double kGuardFlex = 1.0, kGuardAbd = 1.2, kGuardElbow = 1.4;  // the soft elbow reaches ~0.65 of it

double Clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

double Uniform(uint64_t* s) {  // xorshift64, same generator as the rest of the bot
  *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
  return (*s >> 11) * (1.0 / 9007199254740992.0);
}

// Rest pose of the upper body (radians). Elbows slightly bent and arms slightly out: not "at attention".
constexpr double kRestShFlex = 0.05, kRestShAbd = 0.10, kRestElbow = 0.25;
constexpr double kPosture = 0.02;       // constant small flexion of each back joint (light curve of the spine)
constexpr double kElbowMin = 0.2;       // elbows never straight [rad]
constexpr double kElbowMax = 1.4;
constexpr double kAbdMax = 1.3;         // arms-out limit [rad]
constexpr double kFlexLo = -1.0, kFlexHi = 1.4;  // shoulder flexion target limits [rad] (joint -60..180 deg)
constexpr double kCalmV = 0.6;          // CoM speed at which the rest blend is down to up_rest0 [m/s]
// The quiet-stance arm lag (up_sway) acts up to this CoM speed [m/s] (quiet sway is ~1 cm/s); unbounded, the CoM
// speed of a push would drive the shoulder targets far forward and leave the arms hanging there
constexpr double kSwayV = 0.05;
// Arms out only at a real loss of balance (the arms stay in during a standing recovery and go out
// only when falling). Chest tilt [rad] and tilt rate [rad/s] thresholds; lower values spread the arms in ordinary
// recoveries, where the trunk tilts a lot but the body still stands. Tuned on the benches.
constexpr double kTiltThr = 0.45;
constexpr double kRateThr = 3.5;
constexpr double kYawRateThr = 3.5;     // pelvis yaw rate threshold [rad/s]
// Elbow flexion per rad of forward shoulder swing: the elbow flexes as the arm swings forward; the soft elbow alone
// only sags straight under the forearm weight.
constexpr double kElbowSwing = 2.2;  // tuned on the benches
constexpr double kHipAbdArm = 0.2;      // shoulder abduction per rad of own hip abduction (small: arms stay in)
constexpr double kTiltK = 2.5, kRateK = 0.12;  // arms out [rad] per rad of tilt and per rad/s of rate over thresholds
constexpr double kRiseT = 0.05, kFallT = 0.3;  // arms-out rise and release time constants [s]
constexpr double kSpreadMax = 0.8;      // arms-out part of the abduction [rad]
constexpr double kBaseT = 2.0;          // slow average of the chest tilt [s]
constexpr double kRestV = 0.15, kBaseRestT = 0.5;  // at rest the tilt average follows faster [m/s, s]
constexpr double kCounterMax = 0.25;  // counterweight after a hit: limit [rad] (gain up_cw)
constexpr double kReturn = 0.5;         // arms-out share while the tilt already decreases
constexpr double kCounterT = 1.0;       // after a hit: hips sideways -> arms the other way, this long [s]
constexpr double kGazeDelay = 0.15;     // reaction delay before the head turns to the hit [s] (+0..0.1 random)
constexpr double kGazeNeckK = 2.0;      // neck stiffness factor while looking at a point
// Look at the wound: start after the gaze delay [s], neck pitch for a trunk and a leg wound [rad], largest head turn
// to a side wound [rad]
constexpr double kWoundLookT = 0.1, kWoundLookTrunk = 0.7, kWoundLookLeg = 0.8, kWoundLookYaw = 0.5;
constexpr double kLookSlot = 0.4;       // during steps: look along the motion re-drawn every this long [s]
constexpr double kLookP = 0.35;         // ... with this probability
constexpr double kLookMax = 1.2;        // ... if the motion is within this of the chest heading [rad]
constexpr double kNeckYawMax = 1.0, kNeckPitchMax = 0.35;
constexpr double kNeckT = 0.12;
constexpr double kCarHalfLen = 2.25, kCarGazeEnd = 3.0;  // car box half length; gaze follows it until passed [m]
constexpr double kGazeThr = 0.5, kGazeTrunkRate = 4.0, kGazeTrunkMax = 0.7, kGazeTrunkBack = 0.6, kGazeLumbar = 0.4;
// Hands to the wound: target ramp in and out [s]; arm muscle frequency factor while reaching and holding (at the
// plain arm frequency the pendulum-like arm is far too slow to reach a chest wound in a fraction of a second)
constexpr double kWoundInT = 0.15, kWoundOutT = 1.5, kWoundArmW = 3.0;
// Frequency factors for a short time after a hit: a faster muscle lets the elbows, the neck and the spine follow the
// quick target pulses of a hit reaction instead of lagging behind them.
constexpr double kHitElbT = 0.8, kHitElbW = 2.0;  // elbow frequency x2 for 0.8 s after a hit (own choice)
constexpr double kWhipNeckW = 2.0, kWhipNeckT = 0.3;  // neck frequency factor and its time after a bullet
constexpr double kWhipBackW = 2.5, kWhipBackT = 1.0;  // spine pitch frequency factor and its time after a bullet
// Free-arm swing: dead zone and full-amplitude span of the chest speed [m/s], phase rate [rad/s] (slow: about
// 0.8 Hz), shoulder flexion, abduction and elbow amplitudes [rad]; elbow of a hanging wounded arm [rad]
constexpr double kSwingV0 = 0.3, kSwingV1 = 1.0, kSwingW = 5.0, kSwingFlex = 1.2, kSwingAbd = 0.5, kSwingElb = 0.8;
constexpr double kHangElbow = 0.8;
constexpr double kNeckRate = 4.0;  // neck target rate limit [rad/s] (229 deg/s: a human head-turn pace)
constexpr double kYawRefT = 0.5;   // rest heading set point follows the pelvis this fast [s]
constexpr double kYawTd = 0.1;     // ... rate term of the yaw damping [s]
constexpr double kYawFbMax = 0.6;  // ... twist limit (lumbar + chest yaw range is +-0.77 rad) [rad]
constexpr double kBulletLoose = 0.9, kBulletLooseT = 2.0;  // bullet looseness depth, time factor over up_loose_t

// Dying arms: face points ahead of the head CoM and either side of the midline [m at 1.78 m]; reach time [s];
// spread pose (head bullet from behind) [rad]
constexpr double kDeadFaceX = 0.15, kDeadFaceY = 0.03, kDeadFaceZ = 0.15, kDeadArmT = 0.45;
constexpr double kDeadSpreadFlex = 0.5, kDeadSpreadAbd = 1.1, kDeadSpreadElb = 1.4;  // fading arm reaches ~0.4 of it
// Hand to the wound: arm IK in the chest frame. Shoulder joints flex (0,-1,0), abd (sy,0,0), rot (0,0,sy) in this
// order at the shoulder, elbow flex (0,-1,0) at Lu below it (emil_body.cpp); palm at Lp below the elbow.
void Rot(const double a[3], double th, double R[9]) {  // Rodrigues, unit axis
  const double c = std::cos(th), s = std::sin(th), v = 1 - c;
  R[0] = c + a[0] * a[0] * v;        R[1] = a[0] * a[1] * v - a[2] * s; R[2] = a[0] * a[2] * v + a[1] * s;
  R[3] = a[1] * a[0] * v + a[2] * s; R[4] = c + a[1] * a[1] * v;        R[5] = a[1] * a[2] * v - a[0] * s;
  R[6] = a[2] * a[0] * v - a[1] * s; R[7] = a[2] * a[1] * v + a[0] * s; R[8] = c + a[2] * a[2] * v;
}
void Mul(const double A[9], const double B[9], double C[9]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
}
// Elbow and palm positions relative to the shoulder for x = (flex, abd, rot, elbow).
void ArmFk(const double x[4], double sy, double Lu, double Lp, double E[3], double H[3]) {
  const double af[3] = {0, -1, 0}, aa[3] = {sy, 0, 0}, ar[3] = {0, 0, sy};
  double Rf[9], Ra[9], Rr[9], T1[9], R[9];
  Rot(af, x[0], Rf); Rot(aa, x[1], Ra); Rot(ar, x[2], Rr);
  Mul(Rf, Ra, T1); Mul(T1, Rr, R);
  const double se = std::sin(x[3]), ce = std::cos(x[3]);
  const double u[3] = {Lp * se, 0, -Lu - Lp * ce};  // elbow flexion brings the forearm forward
  for (int k = 0; k < 3; ++k) { E[k] = -Lu * R[3 * k + 2]; H[k] = R[3 * k] * u[0] + R[3 * k + 1] * u[1] + R[3 * k + 2] * u[2]; }
}
// Damped Gauss-Newton on 4 joints: palm at tgt, elbow near its preferred place (out to the side, a little back).
// ef = preferred elbow place in units of Lu (x fwd, y out, z up); null = out to the side, a little back.
void ArmIk(double x[4], double sy, double Lu, double Lp, const double tgt[3], const double* ef = nullptr) {
  const double lo[4] = {-1.0, -0.4, -1.2, 0.2}, hi[4] = {2.4, 1.5, 1.5, 2.5};
  const double ep[3] = {(ef ? ef[0] : -0.35) * Lu, (ef ? ef[1] : 0.75) * sy * Lu, (ef ? ef[2] : -0.55) * Lu}, wE = 0.3;
  auto res = [&](const double* y, double* r) {
    double E[3], H[3];
    ArmFk(y, sy, Lu, Lp, E, H);
    for (int k = 0; k < 3; ++k) { r[k] = H[k] - tgt[k]; r[3 + k] = wE * (E[k] - ep[k]); }
  };
  for (int it = 0; it < 6; ++it) {
    double r0[6], J[6][4];
    res(x, r0);
    for (int c = 0; c < 4; ++c) {
      double y[4] = {x[0], x[1], x[2], x[3]}, r1[6];
      y[c] += 1e-4;
      res(y, r1);
      for (int k = 0; k < 6; ++k) J[k][c] = (r1[k] - r0[k]) * 1e4;
    }
    double A[4][5];
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        A[i][j] = i == j ? 1e-3 : 0.0;
        for (int k = 0; k < 6; ++k) A[i][j] += J[k][i] * J[k][j];
      }
      A[i][4] = 0;
      for (int k = 0; k < 6; ++k) A[i][4] -= J[k][i] * r0[k];
    }
    for (int i = 0; i < 4; ++i) {  // Gaussian elimination (A is symmetric positive definite)
      for (int r = i + 1; r < 4; ++r) {
        const double f = A[r][i] / A[i][i];
        for (int c = i; c < 5; ++c) A[r][c] -= f * A[i][c];
      }
    }
    double dx[4];
    for (int i = 3; i >= 0; --i) {
      double v = A[i][4];
      for (int c = i + 1; c < 4; ++c) v -= A[i][c] * dx[c];
      dx[i] = v / A[i][i];
    }
    for (int c = 0; c < 4; ++c) x[c] = Clamp(x[c] + Clamp(dx[c], -0.5, 0.5), lo[c], hi[c]);
  }
}

}  // namespace

// Arms of a dying body (called by L2 during the relaxation; the upper layer itself is off). mode 0: both palms to
// the face (head bullet from the front or side, kill): points kDeadFaceY either side of the face midline, kDeadFaceX
// ahead of the head CoM, so the hands meet before the face as the body drops; the second arm b.dead_arm_lag later.
// mode 1 (head bullet from behind): the arms spread out to the sides.
void World::DeadArms(Bot& b, double t, int mode) {
  const Percept& P = b.pc[b.cur ^ 1];
  double* T = b.target;
  const double s0 = t - b.dead_t;
  const double* xc = d_->xpos + 3 * (b.body0 + B_CHEST);
  const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
  const double* hc = d_->xipos + 3 * (b.body0 + B_HEAD);
  const double r[3] = {hc[0] - xc[0], hc[1] - xc[1], hc[2] - xc[2]};
  double fc[3];  // head CoM in the chest frame
  for (int a = 0; a < 3; ++a) fc[a] = Rc[a] * r[0] + Rc[3 + a] * r[1] + Rc[6 + a] * r[2];
  const double hs = b.body.height / 1.78;
  for (int s = 0; s < 2; ++s) {
    const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
    const double sy = s ? -1.0 : 1.0;
    const double lag = (s == 0) == (b.dead_arm_lag > 0) ? std::fabs(b.dead_arm_lag) : 0.0;
    const double a = Clamp((s0 - lag) / kDeadArmT, 0.0, 1.0);
    if (mode == 1) {
      T[sh] += a * (kDeadSpreadFlex - T[sh]);
      T[sh + 1] += a * (kDeadSpreadAbd - T[sh + 1]);
      T[sh + 3] += a * (kDeadSpreadElb - T[sh + 3]);
      continue;
    }
    const int ub = b.body0 + (s ? B_UARM_R : B_UARM_L);
    const double* S = m_->body_pos + 3 * ub;  // shoulder in the chest frame
    const double Lu = -m_->body_pos[3 * (ub + 1) + 2], Lp = -m_->body_pos[3 * (ub + 2) + 2] + 0.06 * hs;
    double* aq = s ? b.arm_q2 : b.arm_q;
    if (s0 < kFrameDt) {  // warm start from a raised arm (from the hanging arm the solver took a hyperextended branch
      aq[0] = 1.5; aq[1] = 0.3; aq[2] = 0.0; aq[3] = 1.5;  // when the head was thrown back by the bullet)
    }
    const double tgt[3] = {fc[0] + kDeadFaceX * hs - S[0], fc[1] + sy * kDeadFaceY * hs - S[1], fc[2] + kDeadFaceZ * hs - S[2]};
    static const double kFaceElbow[3] = {0.75, 0.2, -0.6};  // elbows before the chest, a little out (covering the face)
    ArmIk(aq, sy, Lu, Lp, tgt, kFaceElbow);
    for (int k = 0; k < 4; ++k) T[sh + k] += a * (aq[k] - T[sh + k]);
  }
}

void World::Upper(Bot& b, double t) {
  const float* pr = param;
  for (int j = J_SH_FLEX_L; j <= J_WRIST_R; ++j) b.gsup[j] = 1.0;  // L0 weight support of the arms (set below)
  // Neck strength (tuned, docs/SOURCES.md): a strong neck keeps the head close to the chest in the first 0.1 s of an
  // instant push; at the human 35 N m the head is thrown about by the push.
  for (int j = J_NECK_PITCH; j <= J_NECK_YAW; ++j) {
    float& tm = b.js[j].tau_max;
    if (b.neck_tau0[j - J_NECK_PITCH] <= 0) b.neck_tau0[j - J_NECK_PITCH] = tm;
    tm = (float)(b.neck_tau0[j - J_NECK_PITCH] * (pr[P_UPPER] > 0.5 ? pr[P_UP_NECK_STR] : 1.0));
  }
  if (pr[P_UPPER] < 0.5 || t <= 0) return;
  if (b.mode == 2 || b.fall_pred || b.airborne || b.in.curl || b.dead_t >= 0) return;  // L2 drives the upper body
  const Percept& P = b.pc[b.cur ^ 1];  // one control step old, like the balancer
  double* T = b.target;
  const double* q = P.q;
  const double dt = kFrameDt;

  // Rest blend: at the end every upper target is pulled to the rest pose by a share that grows as
  // the body slows: up_rest0 at speed (small, not zero), up_rest1 standing (almost all).
  const double v = std::hypot(P.com_vel[0], P.com_vel[1]);
  const double slow = Clamp(1.0 - v / kCalmV, 0.0, 1.0);
  const double wr = pr[P_UP_REST0] + (pr[P_UP_REST1] - pr[P_UP_REST0]) * slow;
  const double on_ground = (P.foot_contact[0] || P.foot_contact[1]) ? 1.0 : 0.0;

  // Heading frame (pelvis yaw): lateral CoM velocity, + = left.
  const double cy = std::cos(P.pelvis_yaw), sy = std::sin(P.pelvis_yaw);
  const double v_lat = -sy * P.com_vel[0] + cy * P.com_vel[1];
  const double v_fwd = cy * P.com_vel[0] + sy * P.com_vel[1];

  // ---- 5.2 back: posture + twist against the pelvis from the hip flexion difference (desired and actual).
  const double dh_des = T[J_HIP_FLEX_L] - T[J_HIP_FLEX_R], dh_act = q[J_HIP_FLEX_L] - q[J_HIP_FLEX_R];
  // + = left hip ahead -> right shoulder ahead = chest yaw to the left (+) relative to the pelvis, as in walking.
  double twist = pr[P_UP_TWIST] * 0.5 * (dh_des + dh_act) * on_ground * (1.0 - wr);  // rest blend: none standing
  double yaw_err = 0;  // pelvis heading off the set point + rate term [rad] (yaw damping, also for the arms)
  // Yaw damping: the swing leg's recoil turns the light pelvis at every step. The chest (with arms and head about 3x
  // the pelvis yaw inertia) turns further the same way as a reaction wheel: relative twist = k x (pelvis heading - set
  // point) + k Td x yaw rate; the pelvis then turns about I_p + (1 + k) I_c instead of I_p + I_c. Set point: the
  // heading at the step start (held by the balancer), slowly following it at rest.
  {
    auto wr = [](double a) { while (a > M_PI) a -= 2 * M_PI; while (a < -M_PI) a += 2 * M_PI; return a; };
    if (!b.up_yaw_init) { b.up_yaw_ref = P.pelvis_yaw; b.up_yaw_init = 1; }
    if (b.mode == 1) b.up_yaw_ref = b.yaw_hold;
    else b.up_yaw_ref += wr(P.pelvis_yaw - b.up_yaw_ref) * (dt / kYawRefT);
    const double dpsi = wr(P.pelvis_yaw - b.up_yaw_ref);
    yaw_err = (dpsi + kYawTd * P.pelvis_w[2]) * on_ground;
    twist += Clamp(pr[P_UP_YAW_FB] * yaw_err, -kYawFbMax, kYawFbMax);
  }
  T[J_LUMBAR_PITCH] += kPosture;
  T[J_CHEST_PITCH] += kPosture;
  T[J_LUMBAR_YAW] += 0.35 * twist;  // upper joints take more
  T[J_CHEST_YAW] += 0.65 * twist;

  // ---- 5.4 arms out when off balance: tilt and rotation rate of the chest beyond thresholds.
  // Tilt relative to the chest's own slow average (the stance posture leans ~0.07 rad forward, measured).
  if (!b.up_init) { b.up_base_p = P.trunk_pitch; b.up_base_r = P.trunk_roll; b.up_init = 1; }
  const double tilt = std::hypot(P.trunk_pitch - b.up_base_p, P.trunk_roll - b.up_base_r);
  // The average learns the posture only while calm: during a push it would follow the chest and the return to the
  // posture would re-trigger the arms.
  // At rest (CoM slower than kRestV, no step) the arms come down and the average takes the new posture fast, so a body
  // standing again in another trunk posture does not keep the tilt over the threshold and the arms out.
  const bool rest = v < kRestV && b.swing < 0 && on_ground > 0;
  if ((tilt < kTiltThr && b.up_spread < 0.05) || rest) {
    const double bt = rest ? kBaseRestT : kBaseT;
    b.up_base_p += (P.trunk_pitch - b.up_base_p) * (dt / bt);
    b.up_base_r += (P.trunk_roll - b.up_base_r) * (dt / bt);
  }
  const double rp = (P.trunk_pitch - b.up_prev_p) / dt, rr = (P.trunk_roll - b.up_prev_r) / dt;
  const double rate = std::hypot(rp, rr);
  double s = kTiltK * std::fmax(0.0, tilt - kTiltThr) + kRateK * std::fmax(0.0, rate - kRateThr) +
             kRateK * std::fmax(0.0, std::fabs(P.pelvis_rate[2]) - kYawRateThr);
  if (tilt < b.up_prev_tilt) s *= kReturn;  // already coming back
  if (rest) s = 0;  // standing still: no arms out
  s *= pr[P_UP_SPREAD];
  // fast rise, slower release
  const double tau = s > b.up_spread ? kRiseT : kFallT;
  b.up_spread += (s - b.up_spread) * (1.0 - std::exp(-dt / tau));
  b.up_prev_p = P.trunk_pitch;
  b.up_prev_r = P.trunk_roll;
  b.up_prev_tilt = tilt;
  const double spread = Clamp(b.up_spread, 0.0, kSpreadMax);

  // After a hit: hips go sideways -> the arms go the other way as a counterweight (decays over kCounterT).
  double cw = 0;
  if (b.hit_t >= 0 && t - b.hit_t < kCounterT)
    cw = Clamp(pr[P_UP_CW] * v_lat, -kCounterMax, kCounterMax) * (1.0 - (t - b.hit_t) / kCounterT);

  // ---- 5.3 arms follow the opposite hip.
  // During a live bullet reaction the arm lag stays unbounded: the arms act as a forward counterweight to the hit and
  // help the body absorb it without a step.
  const bool bul_live = b.hit_bullet && t < b.bul_until;
  for (int side = 0; side < 2; ++side) {
    const int sh = side ? J_SH_FLEX_R : J_SH_FLEX_L;
    const int hs = side ? J_HIP_FLEX_R : J_HIP_FLEX_L;   // own hip
    const int ho = side ? J_HIP_FLEX_L : J_HIP_FLEX_R;   // opposite hip
    const double lead = (T[ho] - q[ho]) * on_ground;     // opposite hip wants forward beyond where it is
    // target = the arm's own angle + a share of the opposite hip's target-minus-angle gap + a share
    // of the hip difference; no position coupling, no swing corridor. The arm swings by inertia and its weight.
    double flex = q[sh] + 0.5 * pr[P_UP_SWING] * lead + 0.25 * pr[P_UP_SWING] * (q[ho] - q[hs]) * on_ground;
    flex += (side ? 1.0 : -1.0) * pr[P_UP_ARM_FB] * yaw_err;  // arms as reaction masses for the yaw damping
    if (side == 0) {  // bench trace: left shoulder target parts (hip terms, yaw term, rest share)
      b.up_dbg[0] = 0.5 * pr[P_UP_SWING] * lead + 0.25 * pr[P_UP_SWING] * (q[ho] - q[hs]) * on_ground;
      b.up_dbg[1] = -pr[P_UP_ARM_FB] * yaw_err;
      b.up_dbg[2] = wr;
    }
    // elbow: its own angle + a share of how much the opposite hip lags its target
    // absolute target (a relative one lets the soft elbow sag straight under the forearm weight)
    // both ways (forward and side pushes swing the arms back first)
    const double swing_fwd = std::fabs(q[sh] - kRestShFlex);  // arm off the rest angle [rad]
    // sideways swing flexes the elbow too
    const double swing_side = std::fabs(q[sh + 1] - kRestShAbd);
    double elbow = kRestElbow + kElbowSwing * (swing_fwd + swing_side) + 0.6 * std::fabs(lead);
    // abduction: own hip side tilt + constant small spread + counterweight (left + = toward left); arms out on top
    const double away = side ? cw : -cw;  // hips left (cw > 0): right arm out, left arm in
    double abd = kRestShAbd + kHipAbdArm * q[hs + 1] * on_ground + away;
    // rest blend; the quiet-stance arm lag (up_sway) stays live
    flex += wr * (kRestShFlex - flex) - wr * pr[P_UP_SWAY] * (bul_live ? v_fwd : Clamp(v_fwd, -kSwayV, kSwayV));
    elbow += wr * (kRestElbow - elbow);
    abd += wr * (kRestShAbd - abd) + spread;
    // Brace on a moving support: arms out (deck_arm) and moving with the sway speed (deck_cw x the CoM speed across
    // the support; + = CoM going left -> right arm out as a counterweight; negative: CoM going left -> left arm out,
    // the arm's swing reaction turns the trunk back), elbows a little bent; held (gsup below)
    if (b.brace > 0.01) {
      const double vr = -sy * (P.com_vel[0] - b.sup_v[0]) + cy * (P.com_vel[1] - b.sup_v[1]);  // + = left
      const double c = Clamp(pr[P_DECK_CW] * vr, -0.5, 0.5);  // deck_cw < 0: the arm's swing reaction, not its weight
      abd += b.brace * (pr[P_DECK_ARM] + (side ? c : -c));
      elbow += b.brace * 0.25;
    }
    elbow -= 0.3 * spread;  // arms out straighten the elbows a little
    flex = Clamp(flex, kFlexLo, kFlexHi);
    T[sh + 0] = flex;
    T[sh + 1] = Clamp(abd, -0.1, kAbdMax);
    T[sh + 2] = 0;
    T[sh + 3] = Clamp(elbow, kElbowMin, kElbowMax);
    T[sh + 4] = 0;
  }
  // ---- Arm in waiting: while the chest leans along the CoM motion (a stumble that may end in a fall) the arms
  // already go toward that side, so the catch starts from there and not from the hands at the sides when the
  // balancer gives up late. Weight from the lean along the motion and the speed.
  {
    const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
    const double vx = P.com_vel[0], vy = P.com_vel[1], vh = std::hypot(vx, vy) + 1e-9;
    const double lean_v = (Rc[2] * vx + Rc[5] * vy) / vh;  // sin of the chest lean toward the motion
    const double g = Clamp((lean_v - kGuardLean0) / (kGuardLean1 - kGuardLean0), 0.0, 1.0) *
                     Clamp((vh - kGuardV0) / (kGuardV1 - kGuardV0), 0.0, 1.0) * pr[P_UP_GUARD];
    if (g > 0) {
      const double ln = std::hypot(Rc[2], Rc[5]) + 1e-9, lx = Rc[2] / ln, ly = Rc[5] / ln;
      const double cf = lx * Rc[0] + ly * Rc[3], cl = lx * Rc[1] + ly * Rc[4];  // lean direction in the chest frame
      for (int side = 0; side < 2; ++side) {
        const int sh = side ? J_SH_FLEX_R : J_SH_FLEX_L;
        const double sg = side ? -1.0 : 1.0;  // + = left
        const double fl = kGuardFlex * cf, ab = kGuardAbd * std::fmax(sg * cl, 0.0) + kRestShAbd;
        T[sh + 0] += g * (fl - T[sh + 0]);
        T[sh + 1] += g * (std::fmax(ab, T[sh + 1]) - T[sh + 1]);
        T[sh + 3] += g * (kGuardElbow - T[sh + 3]);
      }
    }
  }

  // ---- Hand to the wound: after the drawn delay the near arm puts the palm on the wound (chest
  // frame point of the hit, 4 cm out of the body), elbow out; held, then released. Ramps in and out.
  // Both hands may go: the near hand on the wound, the other over it (trunk) or beside it (leg, 15 cm); which lies on
  // top is drawn.
  int at_wound[2] = {0, 0};
  if (b.arm_side >= 0 && t >= b.arm_t0 && t < b.arm_t1) {
    if (b.bul_body >= B_THIGH_L) {
      // A leg wound moves with the leg, not with the chest: as the trunk bends over a thigh wound a fixed chest-frame
      // point would sink into the thigh and the palms pressing on it would hold the chest up. Recompute it from the
      // hit body.
      const double* bp = d_->xpos + 3 * (b.body0 + b.bul_body);
      const double* Rb = d_->xmat + 9 * (b.body0 + b.bul_body);
      const double* cp = d_->xpos + 3 * (b.body0 + B_CHEST);
      const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
      double r[3];
      for (int a = 0; a < 3; ++a)
        r[a] = bp[a] + Rb[3 * a] * b.bul_b[0] + Rb[3 * a + 1] * b.bul_b[1] + Rb[3 * a + 2] * b.bul_b[2] - cp[a];
      for (int a = 0; a < 3; ++a) b.bul_c[a] = Rc[a] * r[0] + Rc[3 + a] * r[1] + Rc[6 + a] * r[2];
    }
    const bool leg = b.bul_body >= B_THIGH_L;
    for (int n = 0; n < (b.arm_both ? 2 : 1); ++n) {
      const int side = n == 0 ? b.arm_side : 1 - b.arm_side, sh = side ? J_SH_FLEX_R : J_SH_FLEX_L;
      double* aq = n == 0 ? b.arm_q : b.arm_q2;
      const double sy = side ? -1.0 : 1.0;
      const int ub = b.body0 + (side ? B_UARM_R : B_UARM_L);
      const double* S = m_->body_pos + 3 * ub;  // shoulder in the chest frame
      const double Lu = -m_->body_pos[3 * (ub + 1) + 2], Lp = -m_->body_pos[3 * (ub + 2) + 2] + 0.06 * b.body.height / 1.78;
      const double hn = std::hypot(b.bul_c[0], b.bul_c[1]) + 1e-9;
      const double out = 0.04 + (n == b.arm_top ? 0.03 : 0.0);
      const double lat = n == 0 ? 0.0 : (leg ? 0.15 : 0.03) * sy;
      const double tgt[3] = {b.bul_c[0] + out * b.bul_c[0] / hn - S[0], b.bul_c[1] + out * b.bul_c[1] / hn + lat - S[1],
                             b.bul_c[2] - S[2]};
      if (t - b.arm_t0 < dt) for (int k = 0; k < 4; ++k) aq[k] = q[sh + (k < 3 ? k : 3)];
      ArmIk(aq, sy, Lu, Lp, tgt);
      const double w = Clamp(std::fmin((t - b.arm_t0) / kWoundInT, (b.arm_t1 - t) / kWoundOutT), 0.0, 1.0);
      for (int k = 0; k < 4; ++k) {
        const int j = sh + (k < 3 ? k : 3);
        T[j] += w * (aq[k] - T[j]);
      }
      at_wound[side] = t - b.arm_t0 < b.arm_t1 - b.arm_t0 - kWoundOutT ? 1 : 0;
    }
  }

  // ---- Free arms during a live bullet reaction. An arm hit by a bullet hangs after 1 s (abduction
  // halved, elbow bent, soft, does not reach). Other free arms swing: shoulder and elbow on a slow sine with their own
  // drawn phase and elbow bend, amplitude from the chest speed in body axes above a dead zone; at low speed the
  // targets blend to the current angles (free arm); the weight is not held (gsup 0 below).
  int swing_arm[2] = {0, 0}, hang_arm[2] = {0, 0};
  // Arm swing also outside bullets (arm_swing 1: while bracing on a moving support, 2: always); the speed is the
  // chest's over the support (a deck carrying the body is no motion of its own)
  const bool bul_sw = b.hit_bullet && t < b.bul_until && !b.down;
  const bool gen_sw = !bul_sw && !b.down && (pr[P_ARM_SWING] > 1.5 || (pr[P_ARM_SWING] > 0.5 && b.brace > 0.3));
  if (bul_sw || gen_sw) {
    const double* cv = d_->cvel + 6 * (b.body0 + B_CHEST);
    const double cvx = cv[3] - b.sup_v[0], cvy = cv[4] - b.sup_v[1];
    const double vs = bul_sw ? std::hypot(v_fwd, v_lat) : std::hypot(cvx, cvy);
    const double v0 = bul_sw ? kSwingV0 : pr[P_ARM_SW_V0];  // general: only a chest clearly losing the balance
    const double amp = (bul_sw ? pr[P_BUL_SWING] : 1.0) * Clamp((vs - v0) / kSwingV1, 0.0, 1.0);
    for (int s = 0; s < 2; ++s) {
      const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
      const bool wounded = b.arm_hit_t[s] >= 0 && t - b.arm_hit_t[s] > 1.0;
      if (wounded) {
        hang_arm[s] = 1;
        T[sh] = q[sh];
        T[sh + 1] = 0.5 * q[sh + 1];
        T[sh + 3] = kHangElbow;
        continue;
      }
      if (at_wound[s] || (b.arm_side >= 0 && t >= b.arm_t0 && t < b.arm_t1 && (s == b.arm_side || b.arm_both))) continue;
      b.swing_ph[s] += kSwingW * dt;
      const double sw = std::sin(b.swing_ph[s]);
      const double fl = kSwingFlex * amp * sw, el = b.swing_el[s] + kSwingElb * amp * (0.5 + 0.5 * std::sin(b.swing_ph[s] + 0.8));
      const double ab = kRestShAbd + kSwingAbd * amp * (0.5 + 0.5 * sw) + (gen_sw ? b.brace * pr[P_DECK_ARM] : 0.0);
      const double wsw = Clamp(amp * 2.0, 0.0, 1.0);  // low speed: follow the current angles (free arm)
      // general swing: below the swing speed the arms keep their pose (rest or braced) and their weight;
      // a bullet reaction lets them follow the present angles (free arms)
      const double b0 = gen_sw ? T[sh] : q[sh], b1 = gen_sw ? T[sh + 1] : q[sh + 1], b3 = gen_sw ? T[sh + 3] : q[sh + 3];
      T[sh] = b0 + wsw * (fl - b0);
      T[sh + 1] = b1 + wsw * (ab - b1);
      T[sh + 3] = b3 + wsw * (el - b3);
      swing_arm[s] = bul_sw || wsw > 0.5;
    }
  }

  // ---- 5.6 gaze: after a hit, with a reaction delay, look where it came from for 1-2 s; during steps sometimes along
  // the motion. Neck targets are smoothed; neck stiffer while looking at a point.
  double yaw_t = 0, pitch_t = 0;
  bool look = false;
  bool trunk_ok = false;  // the trunk helps only the newer gazes (car, shot that missed); hit gazes use the neck alone
  if (b.hit_t >= 0 && b.gaze_t0 < b.hit_t) {  // a new hit: draw the delay and the duration
    b.gaze_t0 = b.hit_t + kGazeDelay + 0.1 * Uniform(&b.rng);
    b.gaze_new = 0;
    b.gaze_t1 = b.gaze_t0 + 1.0 + Uniform(&b.rng);
    // the pusher 1 m back along the hit; a shooter farther away (5 m)
    const double back = b.hit_bullet ? 5.0 : 1.0;
    for (int k = 0; k < 3; ++k) b.gaze_pt[k] = b.hit_p[k] - back * b.hit_dir[k];
  }
  // the car is still ahead of the bot or alongside (its front face not passed by more than kCarGazeEnd)
  auto car_ahead = [&]() {
    const double* Cc = d_->xpos + 3 * car_body_;
    const double* Ch = d_->xipos + 3 * (b.body0 + B_CHEST);
    return (Ch[0] - Cc[0]) * car_dir_[0] + (Ch[1] - Cc[1]) * car_dir_[1] - kCarHalfLen > -kCarGazeEnd;
  };
  const double hx = 0.5 * (P.hip[0][0] + P.hip[1][0]), hy = 0.5 * (P.hip[0][1] + P.hip[1][1]);
  const double chest_yaw = P.pelvis_yaw + q[J_LUMBAR_YAW] + q[J_CHEST_YAW];
  auto wrap = [](double a) { while (a > M_PI) a -= 2 * M_PI; while (a < -M_PI) a += 2 * M_PI; return a; };
  double pitch_max = kNeckPitchMax;
  // Look at the wound (after a chest bullet the head goes back with the chest first, then down to the wound while
  // the hands hold it): a trunk or leg bullet of a live reaction, from kWoundLookT after the gaze delay until the
  // hands let go; a side wound turns the head to it.
  const bool body_wound = b.hit_bullet && b.bul_body >= 0 && (b.bul_body <= B_CHEST || b.bul_body >= B_THIGH_L);
  const bool wound_look = body_wound && b.gaze_t0 >= 0 && t >= b.gaze_t0 + kWoundLookT && t < b.arm_t1 &&
                          t < b.bul_until && pr[P_UP_GAZE] > 0.5;
  if (wound_look) {
    look = true;
    yaw_t = Clamp(std::atan2(b.bul_c[1], std::fmax(b.bul_c[0], 0.15)), -kWoundLookYaw, kWoundLookYaw);
    pitch_t = b.bul_body >= B_THIGH_L ? kWoundLookLeg : kWoundLookTrunk;
    pitch_max = pitch_t;
  } else if (pr[P_GAZE_ADV] > 0.5 && pr[P_UP_GAZE] > 0.5 && car_body_ >= 0 && car_speed_ > 0 && !car_hit_ &&
             b.car_gaze_t0 >= 0 && t >= b.car_gaze_t0 && car_ahead()) {
    // a car coming: the head follows its front (sight: after the orienting delay, drawn when it was seen)
    look = true;
    trunk_ok = true;
    const double* Cc = d_->xpos + 3 * car_body_;
    const double fx = Cc[0] + car_dir_[0] * kCarHalfLen, fy = Cc[1] + car_dir_[1] * kCarHalfLen;
    yaw_t = wrap(std::atan2(fy - hy, fx - hx) - chest_yaw);
    const double* hd = d_->xipos + 3 * (b.body0 + B_HEAD);
    pitch_t = Clamp(std::atan2(hd[2] - 0.7, std::hypot(fx - hd[0], fy - hd[1])), -0.2, kNeckPitchMax);  // at the bonnet
  } else if (b.gaze_t0 >= 0 && t >= b.gaze_t0 && t < b.gaze_t1 && pr[P_UP_GAZE] > 0.5) {
    look = true;
    trunk_ok = b.gaze_new != 0;
    {
      yaw_t = wrap(std::atan2(b.gaze_pt[1] - hy, b.gaze_pt[0] - hx) - chest_yaw);
    }
  } else if (b.mode == 1 && v > 0.3 && pr[P_UP_GAZE] > 0.5) {
    if (t >= b.look_t) {
      b.look_t = t + kLookSlot;
      b.look_on = Uniform(&b.rng) < kLookP;
    }
    const double along = wrap(std::atan2(P.com_vel[1], P.com_vel[0]) - chest_yaw);
    // only forward or sideways motion: stepping back, the head would turn to the neck limit
    if (b.look_on && std::fabs(along) < kLookMax) {
      yaw_t = along;
      pitch_t = 0.15;  // a little down, toward where the feet go
    }
  }
  // A large turn of the gaze: the trunk takes over what is beyond kGazeThr, so the neck stays soft inside its range
  // (in gaze shifts the head adds little below 25-40 deg and grows linearly beyond, docs/SOURCES.md; the model has no
  // eyes, so neck and trunk do all of it)
  if (look && trunk_ok && pr[P_GAZE_TRUNK] > 0.01) {
    const double ex = std::fabs(yaw_t) > kGazeThr ? yaw_t - (yaw_t > 0 ? kGazeThr : -kGazeThr) : 0.0;
    b.gaze_trunk = Clamp(b.gaze_trunk + kGazeTrunkRate * pr[P_GAZE_TRUNK] * ex * dt, -kGazeTrunkMax, kGazeTrunkMax);
  } else if (b.gaze_trunk != 0.0) {
    b.gaze_trunk *= std::exp(-dt * (1.0 / kGazeTrunkBack));
    if (std::fabs(b.gaze_trunk) < 1e-4) b.gaze_trunk = 0.0;
  }
  if (b.gaze_trunk != 0.0) {
    T[J_LUMBAR_YAW] += kGazeLumbar * b.gaze_trunk;
    T[J_CHEST_YAW] += (1.0 - kGazeLumbar) * b.gaze_trunk;
  }
  yaw_t = Clamp(yaw_t, -kNeckYawMax, kNeckYawMax);
  pitch_t = Clamp(pitch_t, -pitch_max, pitch_max);
  const double a = wound_look ? 1.0 : 1.0 - std::exp(-dt / kNeckT);  // to the wound: rate limit only
  // The head turns at a human pace: the target moves at most kNeckRate (the 0.12 s smoothing alone lets a large
  // gaze turn reach head rates against the chest far beyond a human head turn).
  const double mx = kNeckRate * dt;
  b.neck_yaw += Clamp((yaw_t - b.neck_yaw) * a, -mx, mx);
  b.neck_pitch += Clamp((pitch_t - b.neck_pitch) * a, -mx, mx);
  T[J_NECK_YAW] += b.neck_yaw;
  T[J_NECK_PITCH] += b.neck_pitch;

  // ---- 5.1 stiffness by frequency and damping ratio: kp = w^2 I, damping 2 z w I, I = the joint's
  // diagonal of the MuJoCo inertia matrix (what the joint carries), times the balancer's body stiffness scale.
  // L0 damping is kp x damp_ratio x zm, so zm = 2 z / (w ratio). The back twist keeps its kp scale (yaw damping).
  // ---- 6 looseness only after a bullet (none after a push): kp and with it the damping drop and come back.
  double loose = 1.0;
  if (b.hit_bullet && b.hit_t >= 0 && t >= b.hit_t)
    loose = 1.0 - kBulletLoose * std::exp(-(t - b.hit_t) / (std::fmax(0.02, (double)pr[P_UP_LOOSE_T]) * kBulletLooseT));
  const double ratio = std::fmax(1e-3, (double)pr[P_DAMP_RATIO]);
  for (int j = J_LUMBAR_PITCH; j <= J_WRIST_R; ++j) {
    if (j == J_LUMBAR_YAW || j == J_CHEST_YAW) {  // twist: yaw damping needs it right when the steps start
      const double lsy = 1.0 - pr[P_UP_LOOSE_YAW] * (1.0 - loose);
      const double wt = pr[P_UP_W_TW];
      if (wt <= 0) { b.kp[j] *= pr[P_UP_BACK] * lsy; continue; }
      // Muscle with three numbers: kp = w^2 x response inertia, damping 2 z w x it, flywheel
      // stf x inertia below x (w^2 h^2 + 2 z w h) and the limit grown as much. With the plain table kp an off-centre
      // bullet spins the light chest far within a few hundredths of a second and snaps it back; the flywheel makes
      // the turn slower and the return gradual.
      const int dof = b.dof0 + j;
      const double zt = pr[P_UP_Z_TW], mult = wt * wt * dt * dt + 2 * zt * wt * dt;
      const double Ib = d_->M[m_->M_rowadr[dof] + m_->M_rownnz[dof] - 1] - m_->dof_armature[dof];
      const double scale = b.kp[j] / std::fmax(b.js[j].kp, 1e-6);
      b.arm[j] = pr[P_UP_STF_TW] * std::fmax(Ib, 0.0) * mult;
      b.limf[j] = 1.0 + pr[P_UP_STF_TW] * mult;
      b.kp[j] = wt * wt * RespInertia(dof) * scale * lsy;
      b.zm[j] = 2.0 * zt / (wt * ratio);
      continue;
    }
    double wj, zj, ls = loose;
    if (j <= J_CHEST_YAW) {  // back: roll stiffer than pitch (a soft roll lets the trunk sway sideways
      // too often)
      const bool roll = j == J_LUMBAR_ROLL || j == J_CHEST_ROLL;
      wj = roll ? pr[P_UP_W_ROLL] : pr[P_UP_W_BACK];
      if (!roll && b.hit_bullet && b.hit_t >= 0 && t - b.hit_t < kWhipBackT) wj *= kWhipBackW;
      zj = pr[P_UP_Z_BACK];
      ls = 1.0 - pr[P_UP_LOOSE_BACK] * (1.0 - loose);
    }
    else if (j <= J_NECK_YAW) {
      wj = pr[P_UP_W_NECK] * (look ? kGazeNeckK : 1.0);
      zj = pr[P_UP_Z_NECK];
      // the head goes back with the chest right after a bullet; a briefly stiffer neck keeps the nod short and
      // small
      if (b.hit_bullet && b.hit_t >= 0 && t - b.hit_t < kWhipNeckT) wj *= kWhipNeckW;
    }
    else if (j == J_ELBOW_L || j == J_ELBOW_R) {
      wj = pr[P_UP_W_ELB] * pr[P_UP_ARM];
      zj = pr[P_UP_Z_ELB];
      // after a hit the elbows hold their bend: at 6 rad/s (kp 3 N m/rad) the arm swinging back on a strong
      // forward walk-off would straighten them
      if (b.hit_t >= 0 && t - b.hit_t < kHitElbT) wj *= kHitElbW;
    }
    else if (j == J_SH_ABD_L || j == J_SH_ABD_R) { wj = pr[P_UP_W_ABD] * pr[P_UP_ARM]; zj = 1.0; }  // arms kept in
    else { wj = pr[P_UP_W_ARM] * pr[P_UP_ARM]; zj = pr[P_UP_Z_ARM]; }
    if (j >= J_SH_FLEX_L && at_wound[j >= J_SH_FLEX_R]) { wj *= kWoundArmW; ls = 1.0; zj = std::fmax(zj, 0.9); }
    if (j >= J_SH_FLEX_L && hang_arm[j >= J_SH_FLEX_R]) wj *= 0.5;  // soft hanging wounded arm
    const int dof = b.dof0 + j;  // diagonal of the CSR inertia: the last entry of the row (lower triangle)
    double I = d_->M[m_->M_rowadr[dof] + m_->M_rownnz[dof] - 1];
    // back pitch and roll and the neck on the muscle with three numbers (flywheel, limit grown, kp on the response
    // inertia) when their stiffness multiplier is set
    // the shoulders (up_stf_arm) and the elbows and wrists (up_stf_elb) by the same switch (0 = off)
    const bool elb = j == J_ELBOW_L || j == J_ELBOW_R || j == J_WRIST_L || j == J_WRIST_R;
    const double stf = j <= J_CHEST_YAW ? pr[P_UP_STF_BACK]
                       : j <= J_NECK_YAW ? pr[P_UP_STF_NECK]
                       : (elb ? pr[P_UP_STF_ELB] : pr[P_UP_STF_ARM]);
    if (stf > 0) {
      const double mult = wj * wj * dt * dt + 2 * zj * wj * dt;
      b.arm[j] = stf * std::fmax(I - m_->dof_armature[dof], 0.0) * mult;
      b.limf[j] = 1.0 + stf * mult;
      I = RespInertia(dof);
    }
    const double scale = b.kp[j] / std::fmax(b.js[j].kp, 1e-6);  // body stiffness scale written by the balancer
    b.kp[j] = wj * wj * I * scale * ls;
    b.zm[j] = 2.0 * zj / (std::fmax(wj, 0.1) * ratio);
    // the arms hang on the shoulder by their weight (5.1); the elbow and the wrist hold the forearm and the hand
    // while a bullet reaction lives the arms hold their own weight (up_gsup_bul): arms hanging loose swing after the
    // hit and their swing drives the body into extra steps.
    if (j >= J_SH_FLEX_L && j != J_ELBOW_L && j != J_ELBOW_R && j != J_WRIST_L && j != J_WRIST_R)
      b.gsup[j] = swing_arm[j >= J_SH_FLEX_R] || hang_arm[j >= J_SH_FLEX_R] ? 0.0  // weight not held
                  : std::fmax(b.hit_bullet && t < b.bul_until ? pr[P_UP_GSUP_BUL] : pr[P_UP_GSUP], b.brace);
  }
}

}  // namespace emil
