// L2 avoidance: car dodge (side steps through the balancer's lean input), flinch after a bullet that missed close to
// the head, and the gaze trigger for both. Runs after Reactions; like every L2 piece it only changes BalanceInput and
// targets, the steps are the balancer's.
#include <algorithm>
#include <cmath>

#include "emil_world.h"

namespace emil {

namespace {

inline double Clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

double Rnd2(uint64_t* s) {  // xorshift64 of its own (the bot's main stream is not disturbed)
  *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
  return (*s >> 11) * (1.0 / 9007199254740992.0);
}

// Car (same geometry as in l2_reactions.cpp): half length, half width of the box, half width of the body with the
// arms at the sides [m]. Clear of the path = chest this far from the centre line.
constexpr double kCarHalfLen = 2.25, kCarHalfWid = 1.3, kClear = 0.9 + 0.3;
// Visual simple reaction time 0.213 s (Woods et al. 2015, docs/SOURCES.md): the dodge starts after it; the head
// orients earlier (0.15-0.25 s from the first sight, Hayhoe and Ballard 2005 saccade latencies ~0.2 s).
constexpr double kCarReactT = 0.21, kGazeD0 = 0.15, kGazeD1 = 0.25;
constexpr double kDodgeMinT = 0.5;    // time left to the hit that still allows a dodge [s] (own choice)
constexpr double kDodgeRamp = 0.15;   // the velocity goal rises over this [s]
constexpr double kStopAdd = 0.35;      // the speed goal reaches zero this far beyond the clear mark [m]
constexpr double kDodgeEnd = 0.3;     // the dodge ends when the braking taper is below this [m/s at 1.78 m]
constexpr double kPushT0 = 1.7, kPushT1 = 1.1;  // time left to the hit [s]: urgency 0 .. 1
constexpr double kLeanFa = 0.7;  // escape cosine to the heading above which the dodge leans (fore-aft)
constexpr double kCalmAdd = 0.4;      // extra distance of a calm dodge [m]
// Bullet miss: the crack is heard within kMissHear (the head turns to the origin), the startle comes when the bullet
// passed within kMissNear of the head. In the acoustic startle the neck (sternocleidomastoid) answers first and the
// muscles farther from the brainstem later (Brown et al. 1991, docs/SOURCES.md); the delays and the 0.5 s decay here
// are own choices, not measured values.
constexpr double kMissHear = 3.0, kMissNear = 0.9, kMissMin = 0.2;
constexpr double kFlT0 = 0.06, kFlRise = 0.08, kFlTau = 0.25;
constexpr double kFlNeck = 0.45, kFlChest = 0.18, kFlLumbar = 0.12, kFlRoll = 0.14, kFlNeckRoll = 0.2;
constexpr double kFlShFlex = 0.5, kFlElbow = 0.7;  // hands rise toward the face (arms: startle posture)
constexpr double kMissGazeT = 1.5;  // look toward the shooter for this long [s]

}  // namespace

void World::Miss(int bot, const double origin[3], const double dir[3]) {
  Bot& b = bots_[bot];
  if (param[P_GAZE_ADV] < 0.5 && param[P_FLINCH] < 0.5) return;
  if (b.dead_t >= 0 || b.health <= 0) return;
  const double t = d_->time;
  const double* h = d_->xipos + 3 * (b.body0 + B_HEAD);
  const double dn = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) + 1e-12;
  const double u[3] = {dir[0] / dn, dir[1] / dn, dir[2] / dn};
  const double r[3] = {h[0] - origin[0], h[1] - origin[1], h[2] - origin[2]};
  const double s = r[0] * u[0] + r[1] * u[1] + r[2] * u[2];
  if (s < 0.5) return;  // the head is not ahead of the shot
  const double c[3] = {r[0] - s * u[0], r[1] - s * u[1], r[2] - s * u[2]};  // head minus the closest point of the line
  const double d = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  if (d > kMissHear) return;
  // side of the pass seen from the bot: the bullet is left (+1) of the head when the head is right of the line
  // (head - line point = c points to the right of the bullet path as seen from the shooter...) in the pelvis frame
  const double* R = d_->xmat + 9 * b.body0;
  const double cl = c[0] * R[1] + c[1] * R[4];  // component of (head - line) on the pelvis left axis
  b.miss_side = cl > 0 ? -1.0 : 1.0;            // head left of the line -> the bullet passes on its right
  b.miss_t = d < kMissNear ? t : -1.0;
  b.miss_w = Clamp((kMissNear - d) / (kMissNear - kMissMin), 0.0, 1.0);
  if (param[P_GAZE_ADV] > 0.5 && b.hit_t < t - 0.2) {
    b.gaze_t0 = t + kGazeD0 + (kGazeD1 - kGazeD0) * Rnd2(&b.rng2);
    b.gaze_t1 = b.gaze_t0 + kMissGazeT;
    for (int k = 0; k < 3; ++k) b.gaze_pt[k] = origin[k];
    b.gaze_new = 1;
  }
}

// Calm dodge: the first step soon after the decision (L1 kDodgePt) and a gentler braking than dodge_ab, so the
// early step does not turn into a long run that cannot stop (own choice, tuned on the benches)
constexpr double kCalmUrg = 0.5, kCalmAb = 1.0;
constexpr double kTurnTc = 0.8;  // time left at the decision below which the body turns away [s] (tuned)
constexpr double kDiveClr = 1.05;  // a foot nearer the car line than this dives [m at 1.78 m] (0.95 needed + margin)
constexpr double kDiveShFlex = 0.6, kDiveShAbd = 0.1, kDiveElb = 2.1;  // arms tucked in a dive [rad]
constexpr double kJumpK = 3.0;  // leg stiffness x this during the jump push (own choice)
constexpr double kDiveAddK = 0.6;  // near hip adduction share of dive_abd (adduction range is smaller)
constexpr double kDiveExt = -0.4;  // hip extension of a backward dive [rad] (near the joint limit)
constexpr double kDiveHip = 1.4, kDiveKnee = 2.0, kDiveHold = 0.8;  // tuck [rad], held [s] (own choice)

void World::Avoid(Bot& b, double t) {
  const float* pr = param;
  double* T = b.target;
  const Percept& P = b.pc[b.cur ^ 1];
  const double hs = b.body.height / 1.78;
  const bool live = b.mode != 2 && !b.fall_pred && !b.airborne && b.dead_t < 0 && !b.in.curl && !b.down;

  // ---- car: gaze start and dodge
  double vg = 0, push = 0;
  const bool car_live = car_body_ >= 0 && car_speed_ > 0 && !car_hit_;
  if (car_live && b.car_seen_t >= 0) {
    if (b.car_gaze_t0 < 0) b.car_gaze_t0 = b.car_seen_t + kGazeD0 + (kGazeD1 - kGazeD0) * Rnd2(&b.rng2);
    const double* Cc = d_->xpos + 3 * car_body_;
    const double* Ch = d_->xipos + 3 * (b.body0 + B_CHEST);
    const double ux = car_dir_[0], uy = car_dir_[1];
    const double rx = Ch[0] - Cc[0], ry = Ch[1] - Cc[1];
    const double gap = rx * ux + ry * uy - kCarHalfLen;  // chest ahead of the car front [m]
    const double sl = -rx * uy + ry * ux;                // lateral position of the chest on the car's left [m]
    if (pr[P_DODGE] > 0.5 && live && b.dodge == 0 && t >= b.car_seen_t + kCarReactT) {
      const double tc = gap / car_speed_;
      if (tc > kDodgeMinT && std::fabs(sl) < kCarHalfWid) {
        double sg = sl > 0.05 ? 1.0 : (sl < -0.05 ? -1.0 : 0.0);
        const double* R = d_->xmat + 9 * b.body0;
        const double hn = std::hypot(R[0], R[3]) + 1e-9;
        const double nx = -uy, ny = ux;  // the car's left
        if (sg == 0.0) {                 // on the centre line: the side ahead of the body, else drawn
          const double fw = (nx * R[0] + ny * R[3]) / hn;
          sg = std::fabs(fw) > 0.5 ? (fw > 0 ? 1.0 : -1.0) : (Rnd2(&b.rng2) < 0.5 ? 1.0 : -1.0);
        }
        b.dodge = 1;
        b.dodge_t0 = t;
        b.dodge_urg = Clamp((kPushT0 - tc) * (1.0 / (kPushT0 - kPushT1)), 0.0, 1.0);
        b.dodge_tc = tc;
        b.dodge_dir[0] = sg * nx;
        b.dodge_dir[1] = sg * ny;
        b.dodge_fa = std::fabs(b.dodge_dir[0] * R[0] + b.dodge_dir[1] * R[3]) / hn > kLeanFa;  // fore-aft escape
      } else {
        b.dodge = 2;  // too late: arms toward the car (Reactions)
      }
    }
    if (b.dodge == 1) {
      const double sg = b.dodge_dir[0] * (-uy) + b.dodge_dir[1] * ux;  // +1 car's left
      // The dodge ends when the chest is past the clear mark and the braking taper has nearly stopped it (a body still
      // running fast at the mark cannot stop), or when the safe line is passed.
      // Urgency 0 (plenty of time) .. 1 (hurried), fixed at the decision: the push-off share and the lag cap of the
      // feet (L1) follow it, a calm dodge goes kCalmAdd farther (the hands reach beyond the chest, so the chest alone
      // clear is not enough).
      push = b.dodge_urg;
      const double kc = kClear + kCalmAdd * (1.0 - push);
      const double rem = kc + kStopAdd - sg * sl;
      // a calm dodge (urgency below kCalmUrg) steps at once (L1) and so brakes earlier: kCalmAb
      const double ab = b.dodge_urg < kCalmUrg ? kCalmAb : (double)pr[P_DODGE_AB];
      const double taper = std::sqrt(2.0 * ab * hs * std::fmax(rem, 0.0));
      const bool clear = (sg * sl > kc + 0.1 && taper < kDodgeEnd * hs) || rem < 0.0 || gap < -kCarHalfLen;
      if (!live) {
        b.dodge = 3;
      } else if (clear) {
        b.dodge = 3;
      } else {
        // Velocity goal along the dodge (L1 stepping law, `vgoal`): dodge_vg at most, falling with the distance left
        // to the safe line so that the last step can brake (a plain lean only makes the balancer catch it with one
        // short step).
        // The safe line is 0.2 m beyond the clear mark: the body stops before it.
        vg = Clamp((t - b.dodge_t0) * (1.0 / kDodgeRamp), 0.0, 1.0) *
             std::fmin(pr[P_DODGE_VG] * hs, taper);
        // A hurried dodge (kTurnTc .. throw_tc left) does not brake toward a stop at the safe line: it throws itself
        // on at thr_vg (braking would hold the CoM too close to the car line, and tucked feet trail behind the CoM)
        if (pr[P_THR_VG] > 0 && b.dodge_tc >= kTurnTc && b.dodge_tc < pr[P_THROW_TC])
          vg = Clamp((t - b.dodge_t0) * (1.0 / kDodgeRamp), 0.0, 1.0) * pr[P_THR_VG] * hs;
      }
    }
  } else if (b.dodge == 1) {
    b.dodge = 3;
  }
  // Turn away: a dodge decided with less than kTurnTc left turns the chest toward the dodge (pelvis heading for L1,
  // the back twisted toward it up to its range), so the car meets the side, not the face
  b.in.head_set = 0;
  if (b.dodge >= 1 && b.dodge_tc < kTurnTc && b.dead_t < 0) {
    b.in.head_set = 1;
    b.in.head_goal = std::atan2(b.dodge_dir[1], b.dodge_dir[0]);
    const double* R = d_->xmat + 9 * b.body0;
    const double cy = std::atan2(R[3], R[0]);
    double e = b.in.head_goal - cy;
    while (e > M_PI) e -= 2 * M_PI;
    while (e < -M_PI) e += 2 * M_PI;
    T[J_LUMBAR_YAW] += Clamp(e, -1.0, 1.0);
    T[J_CHEST_YAW] += Clamp(e, -1.0, 1.0);
  }
  // Dive: a hurried dodge (decided with kTurnTc .. throw_tc left) whose feet are still in the car's path dive_t
  // before the hit pulls the legs up (hips and knees flexed, no weight support, no new steps): the body flies on with
  // its sideways speed and the tucked feet go with the pelvis instead of standing in the path (stepping, the feet
  // trail behind the CoM, short of the 0.95 m needed). First the legs straighten hard for dive_jump (a jump up and out
  // along the leaning legs; a tuck alone does not carry the feet clear). Falling after it is accepted.
  if (pr[P_DIVE_T] > 0 && car_live && b.dodge >= 1 && b.dodge_tc >= kTurnTc && b.dodge_tc < pr[P_THROW_TC] &&
      b.dead_t < 0) {
    const double* Cc = d_->xpos + 3 * car_body_;
    const double ux = car_dir_[0], uy = car_dir_[1];
    const double* Ch = d_->xipos + 3 * (b.body0 + B_CHEST);
    const double tl = ((Ch[0] - Cc[0]) * ux + (Ch[1] - Cc[1]) * uy - kCarHalfLen) / car_speed_;
    const double sg = b.dodge_dir[0] * (-uy) + b.dodge_dir[1] * ux;  // +1: escape to the car's left
    double fm = 9;
    for (int fo = 0; fo < 2; ++fo) {
      const double* Fp = d_->xpos + 3 * (b.body0 + (fo ? B_FOOT_R : B_FOOT_L));
      fm = std::fmin(fm, sg * (-(Fp[0] - Cc[0]) * uy + (Fp[1] - Cc[1]) * ux));
    }
    if (b.dive_t0 < 0 && tl < pr[P_DIVE_T] && tl > 0 && fm < kDiveClr * hs) b.dive_t0 = t;
  }
  if (b.dive_t0 >= 0 && t - b.dive_t0 < pr[P_DIVE_JUMP]) {
    // jump: the standing legs straighten hard first (a push up and out along the leaning legs), then tuck
    for (int s = 0; s < 2; ++s) {
      const int hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
      T[hp] = 0.0;
      T[hp + 3] = 0.0;
      b.kp[hp] *= kJumpK;
      b.kp[hp + 3] *= kJumpK;
    }
    b.in.no_step = 1;
  } else if (b.dive_t0 >= 0 && t - b.dive_t0 < kDiveHold) {
    // dive_arms: the arms tuck to the chest too (elbows bent, close): an arm left out is the last part in the car's
    // path once the feet are clear
    if (pr[P_DIVE_ARMS] > 0) {
      for (int s = 0; s < 2; ++s) {
        const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
        T[sh] = kDiveShFlex;
        T[sh + 1] = kDiveShAbd;
        T[sh + 2] = 0.0;
        T[sh + 3] = kDiveElb;
      }
    }
    for (int s = 0; s < 2; ++s) {
      const int hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
      // the feet go toward the escape: sideways by the hips' abduction (the escape-side leg out, the near one in),
      // forward by hip flexion, backward by hip extension; the knees fold in every case
      const double* R = d_->xmat + 9 * b.body0;
      const double hn = std::hypot(R[0], R[3]) + 1e-9;
      const double ef = (b.dodge_dir[0] * R[0] + b.dodge_dir[1] * R[3]) / hn;  // escape along the heading
      const double el = (s ? -1.0 : 1.0) * (b.dodge_dir[0] * R[1] + b.dodge_dir[1] * R[4]);  // > 0: escape on this side
      const double al = std::fabs(el);
      T[hp] = al * kDiveHip + std::fmax(ef, 0.0) * kDiveHip + std::fmax(-ef, 0.0) * kDiveExt;
      T[hp + 3] = kDiveKnee;
      T[hp + 1] = el > 0 ? al * pr[P_DIVE_ABD] : -al * pr[P_DIVE_ABD] * kDiveAddK;
      b.stand_leg[s] = 0;
    }
    b.in.no_step = 1;
  }
  // dodge_lean: the body leans toward the escape before the feet move: the CoM target goes dodge_lean
  // toward it while the dodge runs (removed again here each frame: in.lean persists)
  b.in.lean[0] -= b.dg_lean[0];
  b.in.lean[1] -= b.dg_lean[1];
  b.dg_lean[0] = b.dg_lean[1] = 0;
  // only for a calm escape along the heading (car from the side); in a sideways escape or a hurried dodge the lean
  // delays the steps and costs foot clearance
  const bool lean_fa = b.dodge_fa;  // fixed at the decision (the body turns while it steps)
  if (pr[P_DODGE_LEAN] > 0 && b.dodge == 1 && vg > 0 && lean_fa && b.dodge_tc >= pr[P_THROW_TC]) {  // calm only
    b.dg_lean[0] = pr[P_DODGE_LEAN] * hs * b.dodge_dir[0];
    b.dg_lean[1] = pr[P_DODGE_LEAN] * hs * b.dodge_dir[1];
    b.in.lean[0] += b.dg_lean[0];
    b.in.lean[1] += b.dg_lean[1];
  }
  b.in.vgoal[0] = vg * b.dodge_dir[0] + b.hob_vg[0];  // + the hobble goal (0 outside)
  b.in.vgoal[1] = vg * b.dodge_dir[1] + b.hob_vg[1];
  b.in.vpush = push;
  b.in.vnox = (b.hob_vg[0] != 0 || b.hob_vg[1] != 0) ? pr[P_VG_NOX] : 0.0;  // the hobble only

  // ---- flinch: the bullet passed close to the head
  if (pr[P_FLINCH] > 0.5 && b.miss_t >= 0 && live) {
    const double s = t - b.miss_t - kFlT0;
    if (s > 0) {
      const double e = Clamp(s / kFlRise, 0.0, 1.0) * std::exp(-std::fmax(0.0, s - kFlRise) / kFlTau) * b.miss_w;
      if (e > 2e-3) {
        T[J_NECK_PITCH] += kFlNeck * e;
        T[J_NECK_ROLL] += kFlNeckRoll * e * b.miss_side;
        T[J_CHEST_PITCH] += kFlChest * e;
        T[J_LUMBAR_PITCH] += kFlLumbar * e;
        T[J_LUMBAR_ROLL] += kFlRoll * e * b.miss_side;
        T[J_CHEST_ROLL] += kFlRoll * e * b.miss_side;
        for (int sd = 0; sd < 2; ++sd) {
          const int sh = sd ? J_SH_FLEX_R : J_SH_FLEX_L;
          T[sh] += kFlShFlex * e;
          T[sh + 3] += kFlElbow * e;
        }
      } else if (s > 3.0) {
        b.miss_t = -1;
      }
    }
  }
  (void)P;
}

}  // namespace emil
