// L1 leg balancer: balance is made by positions. The balancer decides where and
// with which foot to step; the standing legs only prop the body: their angles are computed as if the pelvis were
// already at its desired pose (place, height, posture), so their muscles move the pelvis there.
// Once per frame on the percept one frame old (L3). Order: 3.2 give up, 3.3 need a step, 3.4 foot places, 3.5 which
// foot, 3.6 swing target and end of step, 3.7 balance numbers, 3.8 desired pelvis, 3.9 leg angles, 3.10 limits and
// stiffness. No step timer, no pause after a step, no "home step".
// Commanded step and weight shift (editor) use the same machinery: a command fixes the foot
// place or moves the support middle.
#include <cmath>

#include "emil_world.h"
#include "l1_leg_ik.h"

namespace emil {
namespace {

// Rotates x by the rotation vector th (Rodrigues).
void RotVec(const double th[3], const double x[3], double y[3]) {
  const double an = std::sqrt(th[0] * th[0] + th[1] * th[1] + th[2] * th[2]);
  if (an < 1e-9) {
    y[0] = x[0], y[1] = x[1], y[2] = x[2];
    return;
  }
  const double k[3] = {th[0] / an, th[1] / an, th[2] / an};
  const double c = std::cos(an), sn = std::sin(an), kd = (k[0] * x[0] + k[1] * x[1] + k[2] * x[2]) * (1 - c);
  const double kx[3] = {k[1] * x[2] - k[2] * x[1], k[2] * x[0] - k[0] * x[2], k[0] * x[1] - k[1] * x[0]};
  for (int i = 0; i < 3; ++i) y[i] = x[i] * c + kx[i] * sn + k[i] * kd;
}

constexpr double kPi = 3.14159265358979;
inline double Clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline double Wrap(double a) {
  while (a > kPi) a -= 2 * kPi;
  while (a < -kPi) a += 2 * kPi;
  return a;
}

// Signed distance of p outside the convex hull of pts (negative inside). Monotone chain, n <= 8.
double OutsideHull(double pts[][2], int n, const double p[2]) {
  for (int i = 1; i < n; ++i)  // insertion sort by x, then y
    for (int j = i; j > 0 && (pts[j][0] < pts[j - 1][0] ||
                              (pts[j][0] == pts[j - 1][0] && pts[j][1] < pts[j - 1][1])); --j) {
      double t0 = pts[j][0], t1 = pts[j][1];
      pts[j][0] = pts[j - 1][0]; pts[j][1] = pts[j - 1][1];
      pts[j - 1][0] = t0; pts[j - 1][1] = t1;
    }
  int h[17], k = 0;
  auto cross = [&](int o, int a, int b) {
    return (pts[a][0] - pts[o][0]) * (pts[b][1] - pts[o][1]) - (pts[a][1] - pts[o][1]) * (pts[b][0] - pts[o][0]);
  };
  for (int i = 0; i < n; ++i) {
    while (k >= 2 && cross(h[k - 2], h[k - 1], i) <= 0) --k;
    h[k++] = i;
  }
  for (int i = n - 2, t = k + 1; i >= 0; --i) {
    while (k >= t && cross(h[k - 2], h[k - 1], i) <= 0) --k;
    h[k++] = i;
  }
  double d = -1e9;
  for (int e = 0; e + 1 < k; ++e) {  // CCW hull: outward normal = (dy, -dx)
    const double* a = pts[h[e]];
    const double* b = pts[h[e + 1]];
    const double ex = b[0] - a[0], ey = b[1] - a[1], l = std::sqrt(ex * ex + ey * ey) + 1e-12;
    const double s = ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / l;
    if (s > d) d = s;
  }
  return d;
}

inline void SetLeg(double* T, int f, const Leg6& a) {
  const int j0 = f == 0 ? J_HIP_FLEX_L : J_HIP_FLEX_R;
  T[j0] = a.flex; T[j0 + 1] = a.abd; T[j0 + 2] = a.rot; T[j0 + 3] = a.knee; T[j0 + 4] = a.dorsi; T[j0 + 5] = a.inv;
}

constexpr double kH0 = 0.555;     // standing CoM height / body height (de Leva 1996 segment table; own model ~0.56)
constexpr double kCloseV = 0.12;  // swing foot "still approaching" when its distance shrinks faster [m/s] (tuned)
constexpr double kHobFollow = 1.5;  // hobble: the healthy foot lands this many stance widths from the hurt one (tuned)
// hobble: the hurt (goal-side) foot steps only this far ahead of the CoM [m at 1.78 m]; a longer step spreads the feet
// so far apart that the weak leg cannot carry the body until the healthy foot follows (tuned on the benches)
constexpr double kHobLead = 0.1;
// Push trunk laws (lean_spine_push, hpd_fast_push): full up to / gone at these values (own traces, standard push
// point); past them the push is beyond what a trunk law can save, so the trunk stops working against the fall
constexpr double kChFade0 = 0.26, kChFade1 = 0.44;  // chest tilt back [rad]: 15 / 25 deg
constexpr double kChV0 = 2.8, kChV1 = 3.1;          // peak backward CoM speed [m/s at 1.78 m] (tuned)
constexpr double kChDv0 = 2.45, kChDv1 = 2.65;      // hit velocity change [m/s]: edge of recoverable pushes (tuned)
constexpr double kChCos0 = 0.5, kChCos1 = 0.8;      // hit direction: cosine to straight back (60 / 37 deg off)
constexpr double kChHiT = 0.15, kChHiThr = 0.07;  // chest thrown back past 4 deg within 0.15 s of the hit: laws off
constexpr double kChOk = 0.05;                     // ... on once the chest is 3 deg forward of its tilt at the hit
constexpr double kLhTd = 0.0;  // hip fold from the first touchdown this long after the hit [s]
constexpr double kChHitT = 2.0;                     // ... counted this long after the hit [s] (a recovery lasts 1-2 s)
constexpr double kThrFa = 0.7;  // escape this much along the heading (cosine) counts as fore-aft
constexpr double kThrowTc0 = 0.8;  // below this time left the dodge turns away (L2 kTurnTc), no throw
constexpr double kDodgeCalm = 0.5, kDodgePt = 0.5;  // calm dodge urgency limit, true-speed share (tuned)
constexpr double kMinStepW = 0.08;  // shortest step of a weakened leg to its clipped target [m at 1.78 m] (own)
constexpr double kMinStep = 0.03; // a needed step shorter than this is not taken, at 1.78 m [m] (it would not help)
constexpr double kShiftLift = 0.05;  // commanded lift height [m]
constexpr double kPullFreeBand = 0.4;  // pull_free fades over this speed band below pull_free_v [m/s]
// pull_free_v for a wounded body: the low cap kPullFreeV0 [m/s] at health kPullFreeH0, pull_free_v from kPullFreeH1 up
// (pistol 15 per bullet: 70 after the 2nd bullet, 40 after the 4th)
constexpr double kPullFreeV0 = 1.4, kPullFreeH0 = 40, kPullFreeH1 = 70;
constexpr double kRest = 0.01;    // foot resting on the floor: ankle at most this above its standing height [m]
constexpr double kNearPlace = 0.08;  // a hanging foot this close to its place is put down, not stepped [m] (a step
                                     // shorter than 8 cm barely changes the support)
constexpr double kLowFoot = 0.03;    // ... if the ankle is this low above standing height [m] (feet bounce 1-2.5 cm)
constexpr double kAhead = 0.10;   // a stalled swing foot is put down only this far ahead of the CoM along its path [m]
constexpr double kBehindV = 0.5;  // ... when the CoM moves faster than this [m/s] (a foot behind it cannot catch it)
constexpr double kTdKnee = 0.38;    // knee at touchdown in strong stagger [rad], 22 deg: a bent knee takes the landing
constexpr double kSoleDown = 0.03;  // td_now 2: ankle at most this above standing height = sole down [m at 1.78]
constexpr double kTouchHard = 0.5;  // swing foot normal force that ends a swing past half its path [x body weight]
constexpr double kDropKeep = 0.14;  // pelvis drop to keep in strong stagger [m] (a lower pelvis lengthens the reach)
constexpr double kStandKnee = 0.28; // standing knee flexion [rad], 16 deg (own standing measurement)
constexpr double kHipMargin = 0.1;  // swing height: hip limit ellipse this far past the hard limits [rad] (3.6)
constexpr double kStallReach = 0.97;  // stalled swing: pelvis low enough for this x leg to the place (IK 0.995)
constexpr double kStall = 0.06;   // swing not getting closer this long -> foot down, step ends on touch [s] (tuned)
constexpr double kMidV = 0.15;    // commanded support middle speed [m/s] (faster: the CoM overshoots the stance foot)
constexpr double kMidV2 = 1.0;    // after a commanded step lands: the middle goes between the feet fast [m/s]

// The foot on the goal side (g = unit goal direction) steps out `ahead` in front of the CoM c (+ stance offset o)
// along the goal. A separate function: inlined, it changed the fused multiply-adds of the neighbouring code (aarch64).
constexpr double kGuSideV = 3.0;  // gu_side only below this CoM speed at the first step [m/s at 1.8 m]

__attribute__((noinline)) void VgLeadPlace(double* place, const double* sf, const double* so, const double* c, double ox,
                                           double oy, const double* g, double ahead, double tol) {
  if ((sf[0] - so[0]) * g[0] + (sf[1] - so[1]) * g[1] <= -tol) return;  // level feet: either may go first
  const double cg = (c[0] + ox) * g[0] + (c[1] + oy) * g[1];
  const double pg = place[0] * g[0] + place[1] * g[1];
  place[0] += (cg + ahead - pg) * g[0];
  place[1] += (cg + ahead - pg) * g[1];
}

// Push-off: before the first step he drops and shoves off the near foot; here a horizontal pair force along the goal
// (the helper force's reaction on the feet), fading as the CoM speed va (relative to the support) reaches the goal
// speed gn.
__attribute__((noinline)) void VgPushOff(double* f, const double* g, double gn, double va, double fmax) {
  const double k = std::fmin(std::fmax((gn - va) * (1.0 / gn), 0.0), 1.0) * fmax;
  f[0] += k * g[0];
  f[1] += k * g[1];
}

// The trailing foot (behind the other one along the goal g) does not land past the stance foot `so` (no crossing steps:
// a crossed leg blocks the next swing). Separate function: aarch64 float contraction of the neighbouring code.
__attribute__((noinline)) void VgNoCross(double* place, const double* sf, const double* so, const double* g, double k) {
  if ((sf[0] - so[0]) * g[0] + (sf[1] - so[1]) * g[1] > 0) return;
  const double pg = (place[0] - so[0]) * g[0] + (place[1] - so[1]) * g[1];
  if (pg <= 0) return;
  place[0] -= k * pg * g[0];
  place[1] -= k * pg * g[1];
}

}  // namespace

void World::CommandStep(int bot, int foot, double dx, double dy, bool cross) {
  Bot& b = bots_[bot];
  const double* R = d_->xmat + 9 * b.body0;  // pelvis
  const double hn = std::hypot(R[0], R[3]) + 1e-9, hx = R[0] / hn, hy = R[3] / hn, lx = -hy, ly = hx;
  const double side = foot == 0 ? 1.0 : -1.0;
  const double* a = d_->xanchor + 3 * (b.jnt0 + (foot ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L));
  b.cmd_tgt[0] = a[0] + dx * hx + side * dy * lx;  // ankle target (bench measures the ankle)
  b.cmd_tgt[1] = a[1] + dx * hy + side * dy * ly;
  b.cmd_foot = foot;
  b.cmd_cross = cross ? 1 : 0;  // the place decides the path; kept for the editor API
  b.cmd_on = 1;
  b.cmd_to = 2.5;
  b.cmd_started = 0;
  b.cmd_t0 = d_->time;
}

void World::CommandShift(int bot, int foot, double lift_s) {
  Bot& b = bots_[bot];
  b.sh_foot = foot;
  b.sh_state = 0;
  b.sh_t0 = d_->time;
  b.sh_lift = lift_s;
  b.sh_up_t0 = -1;
  b.sh_dn_t0 = -1;
}

// Whole-body centroidal inertia of a bot (world frame, row-major 3x3), its mass and CoM, from the bodies' inertial
// frames (a standing leg's joint moves the whole body above it).
static void BotInertia(const mjModel* m, const mjData* d, int b0, double Ic[9], double* M, double c[3]) {
  double mt = 0, cx = 0, cy = 0, cz = 0;
  for (int k = 0; k < B_COUNT; ++k) {
    const double mk = m->body_mass[b0 + k];
    const double* x = d->xipos + 3 * (b0 + k);
    mt += mk; cx += mk * x[0]; cy += mk * x[1]; cz += mk * x[2];
  }
  c[0] = cx / mt; c[1] = cy / mt; c[2] = cz / mt;
  *M = mt;
  for (int i = 0; i < 9; ++i) Ic[i] = 0;
  for (int k = 0; k < B_COUNT; ++k) {
    const double mk = m->body_mass[b0 + k];
    const double* x = d->xipos + 3 * (b0 + k);
    const double* R = d->ximat + 9 * (b0 + k);
    const double* I = m->body_inertia + 3 * (b0 + k);
    const double r[3] = {x[0] - c[0], x[1] - c[1], x[2] - c[2]};
    const double rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        const double rot = R[3 * i] * I[0] * R[3 * j] + R[3 * i + 1] * I[1] * R[3 * j + 1] + R[3 * i + 2] * I[2] * R[3 * j + 2];
        Ic[3 * i + j] += rot + mk * ((i == j ? rr : 0.0) - r[i] * r[j]);
      }
  }
}

// Inertia of the whole bot about the axis a (unit) through p.
static double AxisInertia(const double Ic[9], double M, const double c[3], const double* p, const double* a) {
  double s = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) s += a[i] * Ic[3 * i + j] * a[j];
  const double r[3] = {c[0] - p[0], c[1] - p[1], c[2] - p[2]};
  const double x[3] = {r[1] * a[2] - r[2] * a[1], r[2] * a[0] - r[0] * a[2], r[0] * a[1] - r[1] * a[0]};
  return s + M * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

// Inertia a standing knee moves: with the foot planted and the hip holding the trunk's orientation, the
// knee turns the thigh about its axis and carries the rest of the body with the hip point (translation):
// I = I_thigh(axis) + M_rest |(p_hip - p_knee) x a|^2, ~12 kg m^2. Treating the body above the knee as one rigid
// body would overstate it about 3x (33 kg m^2) and make the stance knee too stiff.
static double KneeStanceInertia(const mjModel* m, const mjData* d, int th, int jhip, int jknee, double Mb) {
  const double* a = d->xaxis + 3 * jknee;
  const double* pk = d->xanchor + 3 * jknee;
  auto perp2 = [&](const double* p) {
    const double r[3] = {p[0] - pk[0], p[1] - pk[1], p[2] - pk[2]};
    const double x[3] = {r[1] * a[2] - r[2] * a[1], r[2] * a[0] - r[0] * a[2], r[0] * a[1] - r[1] * a[0]};
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  };
  const double* R = d->ximat + 9 * th;  // inertial frame -> world, row-major
  const double* I = m->body_inertia + 3 * th;
  double s = 0;
  for (int k = 0; k < 3; ++k) {
    const double ak = R[k] * a[0] + R[3 + k] * a[1] + R[6 + k] * a[2];
    s += I[k] * ak * ak;
  }
  s += m->body_mass[th] * perp2(d->xipos + 3 * th);
  s += (Mb - m->body_subtreemass[th]) * perp2(d->xanchor + 3 * jhip);
  return s;
}

double World::RespInertia(int dof) {
  // (M^-1)_jj from the factor L'DL of the last substep: z = L^-T e_j touches the joint and its ancestors only.
  double* zb = mi_z_.data();
  zb[dof] = 1.0;
  double inv = 0;
  for (int r = dof; r >= 0; r = m_->dof_parentid[r]) {
    const double zr = zb[r];
    if (!m_->dof_simplenum[r])
      for (int adr = m_->M_rowadr[r], e = adr + m_->M_rownnz[r] - 1; adr < e; ++adr)
        zb[m_->M_colind[adr]] -= d_->qLD[adr] * zr;
    inv += zr * zr * d_->qLDiagInv[r];
    zb[r] = 0;
  }
  return 1.0 / std::fmax(inv, 1e-9);
}

void World::Balance(Bot& b, double t) {
  const Percept& P = b.pc[b.cur ^ 1];
  const float* pr = param;
  const BalanceInput& in = b.in;
  const double H = b.body.height, hs = H / 1.78, ts = std::sqrt(hs);
  const double thigh = b.dims.thigh, shank = b.dims.shank, leg = thigh + shank;
  const double dt = kFrameDt;
  double* T = b.target;
  double Tp[J_COUNT];  // place IK of the swing leg (target velocity source)
  double dsw_f = 0;    // swing foot distance to its place this frame (target velocity blend)

  // Upper body and arms (set in l1_upper.cpp): calm pose, trunk lean from L2.
  T[J_LUMBAR_PITCH] = 0.5 * in.trunk_lean;
  T[J_CHEST_PITCH] = 0.5 * in.trunk_lean;
  T[J_LUMBAR_ROLL] = T[J_LUMBAR_YAW] = T[J_CHEST_ROLL] = T[J_CHEST_YAW] = 0;
  T[J_NECK_PITCH] = T[J_NECK_ROLL] = T[J_NECK_YAW] = 0;
  for (int side = 0; side < 2; ++side) {
    const int j0 = side == 0 ? J_SH_FLEX_L : J_SH_FLEX_R;
    T[j0 + 0] = 0.05; T[j0 + 1] = 0.10; T[j0 + 2] = 0; T[j0 + 3] = 0.25; T[j0 + 4] = 0;
  }
  double stiff = pr[P_STIFF_SCALE] * in.stiffness;
  b.single = -1;
  b.assist_f[0] = b.assist_f[1] = 0;
  double kleg[2] = {pr[P_LEG_K], pr[P_LEG_K]};  // hip and knee stiffness multiplier per leg
  double zleg[2] = {1, 1};                      // damping multiplier per leg
  int ff_leg = -1;                              // leg with target velocity feedforward (the swing leg)
  b.stand_leg[0] = b.stand_leg[1] = 1;

  if (t <= 0) {  // reset: nominal stance pose in joint space
    const double half_w = 0.5 * pr[P_STANCE_W] * hs;
    const double abd0 = std::atan2(half_w - b.dims.hip_half_w, leg);
    for (int side = 0; side < 2; ++side) {
      const int j0 = side == 0 ? J_HIP_FLEX_L : J_HIP_FLEX_R;
      T[j0 + 0] = pr[P_KNEE_BEND] - pr[P_ANKLE_DORSI]; T[j0 + 1] = abd0; T[j0 + 2] = pr[P_TOE_OUT];
      T[j0 + 3] = pr[P_KNEE_BEND]; T[j0 + 4] = pr[P_ANKLE_DORSI]; T[j0 + 5] = abd0;
    }
    // Rest stance (a foot place is the prediction point + half the stance width): half the lateral
    // distance of the sole centres in the spawn pose, which is this reset pose.
    {
      const double* Rp = P.pelvis_R;
      const double hn0 = std::hypot(Rp[0], Rp[3]) + 1e-9, lx0 = -Rp[3] / hn0, ly0 = Rp[0] / hn0;
      const double rh = 0.5 * std::fabs((P.foot_c[0][0] - P.foot_c[1][0]) * lx0 + (P.foot_c[0][1] - P.foot_c[1][1]) * ly0);
      b.rest_half = rh > 0.05 * hs && rh < 0.3 * hs ? rh : half_w;
    }
    b.mode = 0;
    b.swing = -1;
    b.cmd_on = 0;
    b.sh_foot = -1;
    b.sh_state = 0;
    b.ld[0] = b.ld[1] = 0.5;
    b.loaded[0] = b.loaded[1] = 1;
    b.cor_i[0] = b.cor_i[1] = 0;
    b.pitch_acc = 0;
    b.ch_add = 0;
    b.ch_w = 1;
    b.budget = pr[P_STEP_BUDGET];
    b.legs_pass = -1;
    b.head_ref = 0;
    b.td_time = -1;
    b.kf_on = 0;
    b.hd_on = 0;
    b.hpd_lock = 0;
  } else if (b.mode == 2) {
    // rlx_m2: a relaxing (wounded or dying) body keeps the leg stiffness after the give-up; only the muscle tone fades
    // it, so muscle force and weight support fade together down to the floor instead of switching off at once.
    if (!(b.dead_t >= 0 && pr[P_RLX_M2] > 0)) {
      stiff *= 0.3;  // gave up: fall behaviour (L2) takes over; legs soft
      if (b.in.curl) stiff = 0.5;  // curl pose itself is set by L2 (Reactions)
    }
    b.swing = -1;
    kleg[0] = kleg[1] = 1;
  } else {
    // 3.1 Snapshot: body axes from the pelvis heading; feet on the floor (flat ground at z = 0).
    double hx = P.pelvis_R[0], hy = P.pelvis_R[3];
    {
      const double n = std::hypot(hx, hy) + 1e-9;
      hx /= n; hy /= n;
    }
    const double lx = -hy, ly = hx;
    const double* c = P.com;
    // On a moving support the capture point is taken with the CoM speed relative to the support (L2 sup_v)
    double v[2] = {P.com_vel[0] - in.sup_v[0], P.com_vel[1] - in.sup_v[1]};
    const double v_true[2] = {v[0], v[1]};  // before the goal-relative change below
    // Velocity goal (in.vgoal, set by L2; 0 = none, nothing below changes): the CoM speed counts relative to the goal
    // along it, so the foot places (prediction point = CoM + v x lead time) fall behind the CoM while it is slower than
    // the goal (the body keeps going over them) and ahead of it when faster (the capture step that stops it: the goal
    // is 0 again). Behind by at most vg_cap x lead time (the swing leg reaches about 0.4 m; a farther place cannot be
    // reached in time). The foot on the goal side steps first, vg_lead ahead of the CoM (a rear foot stepping first
    // goes wide behind and leaves the stance foot ahead of the CoM, braking it).
    {
      const double gn = std::hypot(in.vgoal[0], in.vgoal[1]);
      if (gn > 1e-6) {
        const double gx = in.vgoal[0] / gn, gy = in.vgoal[1] / gn;
        // thr_cap: a hurried dodge along the heading (car from the side) lets the feet lag farther behind the CoM,
        // so the steps push it on instead of braking it
        const bool thr_fa2 = pr[P_THR_CAP] > 0 && b.dodge == 1 && b.dodge_tc >= kThrowTc0 &&
                             b.dodge_tc < pr[P_THROW_TC] && std::fabs(gx * hx + gy * hy) > kThrFa;
        const double vcap = thr_fa2 ? pr[P_THR_CAP] * hs
                                    : (pr[P_VG_CAP] + (pr[P_VG_CAPH] - pr[P_VG_CAP]) * in.vpush) * hs;
        const double va = v[0] * gx + v[1] * gy, vr = std::fmax(va - gn, -vcap);
        v[0] += (vr - va) * gx;
        v[1] += (vr - va) * gy;
      }
    }
    const double speed = std::hypot(v[0], v[1]);
    const double vg_n = std::hypot(in.vgoal[0], in.vgoal[1]);
    const double vg_g[2] = {vg_n > 1e-6 ? in.vgoal[0] / vg_n : 0.0, vg_n > 1e-6 ? in.vgoal[1] / vg_n : 0.0};
    const double fx = 0.5 * b.dims.foot_len - b.dims.heel_back;  // sole centre ahead of the ankle [m]
    const double ah = b.dims.ankle_h;
    double S[2][2], fyaw[2];
    int on[2];
    for (int f = 0; f < 2; ++f) {
      S[f][0] = P.foot_c[f][0];
      S[f][1] = P.foot_c[f][1];
      fyaw[f] = std::atan2(P.foot_R[f][3], P.foot_R[f][0]);
      // Touching the floor: some load (25 ms low-pass) or resting on it (ankle within 1 cm of its standing height);
      // a foot resting without load is not "in the air" (otherwise unloaded resting feet are stepped again).
      on[f] = b.ld[f] > kTouch || P.ankle[f][2] < b.dims.ankle_h + kRest * hs;
      // 3.1: a standing foot stays support while its ankle is within on_keep of the standing height (hysteresis:
      // back only with load or within kRest); a foot lifted 2-3 cm by the body sway keeps the two-feet rule braking
      // the body instead of stepping (a small side push lifts the far foot without needing a step).
      if (pr[P_ON_KEEP] > 0 && b.on_prev[f] && f != b.swing && P.ankle[f][2] < b.dims.ankle_h + pr[P_ON_KEEP] * hs)
        on[f] = 1;
      // st_toe: a standing foot tilted toe down stays support while its toe is within kRest of the floor.
      // st_toe_side: only while the CoM moves more sideways than fore-aft; the toe-down far foot brakes a side push,
      // while a toe-down front foot in a back push would push the body on backwards.
      const bool toe_side = pr[P_ST_TOE_SIDE] <= 0 ||
                            std::fabs(v[0] * lx + v[1] * ly) > std::fabs(v[0] * hx + v[1] * hy);
      if (pr[P_ST_TOE] > 0 && toe_side && b.on_prev[f] && f != b.swing) {
        const double lt = std::fmax(b.dims.foot_len - b.dims.heel_back, 0.05);
        const double tz = P.ankle[f][2] + P.foot_R[f][6] * lt - P.foot_R[f][8] * b.dims.ankle_h;  // toe tip height
        if (tz < kRest * hs) on[f] = 1;
      }
      b.on_prev[f] = on[f];
    }
    // Place half width = half the rest stance (sole centres) x half_w: it follows the body's own stance, so steps do
    // not land narrower than it (feet in one line give a support only one foot wide).
    const double rest_h = b.rest_half > 0 ? b.rest_half : 0.5 * pr[P_STANCE_W] * hs;
    const double half_w = pr[P_HALF_W] * rest_h + std::fmax(in.wide, 0.0);  // a brace widens the places
    const double omega = std::sqrt(9.81 / std::fmax(c[2], 0.3));
    const double xi[2] = {c[0] + v[0] / omega, c[1] + v[1] / omega};  // capture point, for the assist only
    // pull_lean: under a bullet pull (a constant force F on the pelvis, the same test as World) the body can only rest
    // leaning into it, the support F / (M omega^2) = bul_pull x CoM height further along the pull (LIP with a constant
    // force: x'' = omega^2 (x - p) + F / M). The prediction point and the places move by that offset and the help
    // pulls toward the support middle minus it; with places for free motion the help (x bul_assist) would drive the
    // leaning body further along the pull.
    // pull_help: share of the offset for the help's target alone (0: the pull_lean share); the help (x bul_assist) at
    // the shifted target cancels most of the pull with the CoM over the support (the body braces).
    // rlx_pull: while the wounded body relaxes on the move the pull lean share is rlx_pull, so the bullet's pull does
    // not drive the fading body back faster than its feet follow (the stance foot would lift before the swing lands).
    const bool rlx_p = b.dead_t >= 0 && !in.no_step && pr[P_RLX_PULL] > 0;
    const double pl_k = rlx_p ? (double)pr[P_RLX_PULL] : (double)pr[P_PULL_LEAN];
    const bool pull_on = (pl_k > 0 || pr[P_PULL_HELP] > 0) && t < b.pull_until && b.mode != 2 &&
                         (b.foot_on[0] || b.foot_on[1]);
    const double dpu = pull_on ? b.pull_k * pr[P_BUL_PULL] * std::fmax(c[2], 0.3) : 0.0;
    const double dpl = std::fmin(pl_k, 1.0) * dpu;
    const double plx = dpl * b.pull_dir[0], ply = dpl * b.pull_dir[1];
    const double dph = pr[P_PULL_HELP] > 0 ? std::fmin(pr[P_PULL_HELP], 1.0) * dpu : dpl;

    // 3.3 Need a step: prediction point outside the quad around both soles, or the feet spread wide.
    // rlx_lead: while the wounded body relaxes on the move (L2: down, steps on) the prediction point and the places
    // use this share of the lead time: the steps go under the sinking pelvis instead of to the capture point of an
    // upright body (a foot far behind the pelvis lets the body vault over it and fall from the top of the vault).
    const bool rlx = b.dead_t >= 0 && !in.no_step;
    // rlx_ldir: the short lead time only for the backward part of the motion (share of the CoM velocity direction
    // against the heading): moving forward (bullets from behind) half steps do not brake the body, it falls on its
    // knees and the trunk slaps forward. rlx_ldir 2: the lowered swing reach (rlx_swhd) too: forward, a low swing
    // lands late and close, the rear foot stays far behind and the body kneels.
    double rlx_k = rlx && pr[P_RLX_LEAD] > 0 ? (double)pr[P_RLX_LEAD] : 1.0, rlx_wb = 1.0;
    if (rlx && pr[P_RLX_LDIR] > 0) {
      const double vn = std::hypot(v[0], v[1]);
      // full while the CoM moves back or sideways, none from 60 deg off the heading forward (cos 0.5)
      rlx_wb = vn > 1e-6 ? Clamp(1.0 - 2.0 * (v[0] * hx + v[1] * hy) / vn, 0.0, 1.0) : 1.0;
      rlx_k = 1.0 - (1.0 - rlx_k) * rlx_wb;
    }
    const double lead = pr[P_LEAD_T] * ts * rlx_k;
    // L2 lean (bullet: lean along its path) moves the prediction point too, so it can make a step.
    // A calm car dodge (urgency < kDodgeCalm) tests the need of a step with kDodgePt of the true CoM speed instead of
    // the goal-relative one (that point stays behind the CoM while it is slower than the goal, so the first step would
    // wait for the CoM to leave the support with the trailing foot still in the car's path); hurried dodges keep the
    // push-off. thr_pt: a hurried dodge (kThrowTc0 .. throw_tc left) steps at once too (no push-off on two feet first).
    const bool thr_win = pr[P_THR_PT] > 0 && b.dodge == 1 && b.dodge_tc >= kThrowTc0 && b.dodge_tc < pr[P_THROW_TC];
    const bool dodge_vg = vg_n > 1e-6 && in.vnox <= 0 && b.dodge == 1 && (b.dodge_urg < kDodgeCalm || thr_win);
    const double vpx = dodge_vg ? v[0] + kDodgePt * (v_true[0] - v[0]) : v[0];
    const double vpy = dodge_vg ? v[1] + kDodgePt * (v_true[1] - v[1]) : v[1];
    const double pt[2] = {c[0] + vpx * lead + in.lean[0] + plx, c[1] + vpy * lead + in.lean[1] + ply};
    double Q[2][2] = {{S[0][0], S[0][1]}, {S[1][0], S[1][1]}};
    const double sep = (S[0][0] - S[1][0]) * lx + (S[0][1] - S[1][1]) * ly;  // left foot left of the right one
    if ((!on[0] || !on[1]) && sep < 2 * half_w) {  // a foot in the air: not closer than the stance width
      const double add = 2 * half_w - sep, w0 = !on[0] ? (!on[1] ? 0.5 : 1.0) : 0.0, w1 = 1.0 - w0;
      Q[0][0] += w0 * add * lx; Q[0][1] += w0 * add * ly;
      Q[1][0] -= w1 * add * lx; Q[1][1] -= w1 * add * ly;
    }
    double pts[8][2];
    for (int f = 0; f < 2; ++f) {
      const double* R = P.foot_R[f];
      const double sx = 0.5 * b.dims.foot_len, sy = 0.5 * b.dims.foot_w;
      for (int k = 0; k < 4; ++k) {
        const double ax = (k & 1) ? sx : -sx, ay = (k & 2) ? sy : -sy;
        pts[4 * f + k][0] = Q[f][0] + R[0] * ax + R[1] * ay;
        pts[4 * f + k][1] = Q[f][1] + R[3] * ax + R[4] * ay;
      }
    }
    const double out = OutsideHull(pts, 8, pt);
    b.dbg_out = out;
    // Margin in foot widths, gathering width in rest stance widths: both follow this body, not a fixed number.
    // hob_ds: hobbling (goal with vnox), the next step starts once both feet have stood hob_ds: the goal-relative
    // prediction point leaves the stance only when the body has nearly stopped, which would stall the pace after each
    // pair of steps
    const bool hob_go = in.hob_ds > 0 && in.vnox > 0 && vg_n > 1e-6 && on[0] && on[1] && b.td_time >= 0 &&
                        t - b.td_time > in.hob_ds;
    // thr_ds: a hurried dodge (kThrowTc0 .. throw_tc left) steps as soon as both feet have stood thr_ds since the
    // decision or the last touchdown (otherwise the body leans on two feet and the escape starts late)
    const bool thr_w = b.dodge == 1 && b.dodge_tc >= kThrowTc0 && b.dodge_tc < pr[P_THROW_TC] && vg_n > 1e-6;
    // only for an escape along the heading (car from the side): for a sideways escape it would cost the dive its reach
    const bool thr_fa = std::fabs(vg_g[0] * hx + vg_g[1] * hy) > kThrFa;
    const bool thr_go = thr_w && thr_fa && pr[P_THR_DS] > 0 && on[0] && on[1] &&
                        t - std::fmax(b.dodge_t0, b.td_time) > pr[P_THR_DS];
    const bool need = out > pr[P_MARGIN] * b.dims.foot_w ||
                      std::fabs(sep) > 2 * half_w + pr[P_APART] * 2 * rest_h || hob_go || thr_go;
    // apart_fa: "feet spread noticeably wider than the stance" by their distance, so a long fore-aft
    // split gathers too (the CoM in the gap between the rear toe and the front heel drifts sideways). Only the step
    // choice uses it, not the give-up. The value is the CoM speed below which it acts [m/s]: faster, the prediction
    // point decides alone.
    const double fsp = (S[0][0] - S[1][0]) * hx + (S[0][1] - S[1][1]) * hy;
    // Bracing on a moving support, feet narrower than the widened stance step out (one foot at a time)
    const bool narrow = in.wide > 0.02 * hs && on[0] && on[1] && std::fabs(sep) < 2 * half_w - 0.04 * hs;
    const bool gather = (pr[P_APART_FA] > 0 && on[0] && on[1] && std::hypot(v[0], v[1]) < pr[P_APART_FA] &&
                         std::hypot(sep, fsp) > 2 * half_w + pr[P_APART] * 2 * rest_h) || narrow;
    b.cp_out = out;
    b.pt[0] = pt[0];
    b.pt[1] = pt[1];

    // 3.2 Give up: CoM too far from the nearest foot (or the feet middle) for its height, too low, steps spent.
    {
      // Switch gu_first (default off): until the first step from calm touches down, its foot counts at its
      // reach-clipped target instead of where it flies. In strong falls the place is always clipped (the target runs
      // with the hip), so this delays the give-up to the low-height rule.
      double Sg[2][2] = {{S[0][0], S[0][1]}, {S[1][0], S[1][1]}};
      if (b.swing >= 0 && (b.sw_first || vg_n > 1e-6)) {  // with a velocity goal the swing foot counts at its target
        Sg[b.swing][0] = b.sw_tg[0];
        Sg[b.swing][1] = b.sw_tg[1];
      }
      double off = std::hypot(c[0] - 0.5 * (Sg[0][0] + Sg[1][0]), c[1] - 0.5 * (Sg[0][1] + Sg[1][1]));
      for (int f = 0; f < 2; ++f) off = std::fmin(off, std::hypot(c[0] - Sg[f][0], c[1] - Sg[f][1]));
      const bool spent = ((pr[P_STEP_BUDGET] > 0 && b.budget <= 0) || in.steps_left == 0) && need && b.swing < 0;
      // gu_vz: during a swing the CoM, below 0.95 of its standing height, falls faster than gu_vz: the legs no longer
      // carry the body (1.2 m/s = 0.12 s of free fall; a recoverable step sinks more slowly).
      const bool sink = pr[P_GU_VZ] > 0 && b.swing >= 0 && P.com_vel[2] < -pr[P_GU_VZ] && c[2] < 0.95 * kH0 * H;
      const double lowk = pr[P_LOW];
      // rlx_gu: no give-up while the wounded body relaxes on the move: the balancer keeps stepping and the legs keep
      // their (fading) force and weight support down to the floor, with no moment where the support switches off
      const bool keep = rlx && pr[P_RLX_GU] > 0;
      // A strong sideways fall keeps stepping longer: the distance factor grows with the sideways share of the CoM
      // speed (side steps can still catch it; forward and back falls are not helped by it)
      const double vh = std::hypot(v[0], v[1]) + 1e-9;
      const double sidew = std::fabs(-v[0] * hy + v[1] * hx) * 1.0 * (1.0 / (vh * (std::hypot(hx, hy) + 1e-9)));
      // The share is the one at the first step of the recovery (min with the current one), held while steps are
      // needed: a back fall that turns the pelvis (there is no foot-yaw spring) soon looks sideways and would keep
      // stepping as a side fall.
      if (b.swing < 0 && !need) b.gu_sw = -1;
      // Faster than a step can catch at the first step (about leg length / step time, 0.9 m / 0.3 s = 3 m/s at
      // 1.8 m): no extra stepping, the give-up comes at the normal distance.
      else if (b.swing >= 0 && b.gu_sw < 0) b.gu_sw = vh < kGuSideV * hs ? sidew : 0.0;
      const double sw_k = b.gu_sw >= 0 ? std::fmin(sidew, b.gu_sw) : sidew;
      const double gu_k = 1.0 + (pr[P_GU_SIDE] - 1.0) * Clamp((sw_k - 0.5) * 3.0, 0.0, 1.0);
      // L2 tumble (going down the stairs out of control) gives up too: no step in the air or in a roll
      if (in.tumble ||
          (!keep && (off > pr[P_GIVE_UP] * gu_k * std::fmax(c[2], 0.2) || c[2] < lowk * kH0 * H || spent || sink))) {
        // Gave up: mode 2 holds until a reset. From this frame the leg balancer writes no pose and starts no steps,
        // the same as the mode-2 branch above (the jump skips the mode update below, which would clear it in the same
        // frame before L2 sees the fall).
        b.dbg_gu = (off > pr[P_GIVE_UP] * gu_k * std::fmax(c[2], 0.2) ? 1 : 0) | (c[2] < lowk * kH0 * H ? 2 : 0) |
                   (spent ? 4 : 0) | (sink ? 8 : 0) | (in.tumble ? 16 : 0);
        b.mode = 2;
        b.swing = -1;
        if (!(b.dead_t >= 0 && pr[P_RLX_M2] > 0)) stiff *= 0.3;  // rlx_m2: relaxing, only the tone fades the legs
        kleg[0] = kleg[1] = 1;
        goto gave_up;
      }
    }
    if (pr[P_STEP_BUDGET] > 0 && b.swing < 0 && speed < 0.1 && t - b.td_time > 1.0) b.budget = pr[P_STEP_BUDGET];

    // 3.4 Place of each foot: prediction point + half stance width to its side, small fore-aft correction by the
    // direction from the other foot to the prediction point (diagonal motion: the place does not slide under it).
    // Backing off, the place falls short of the lead_t point by at most place_back_d and by no more than
    // (1 - place_back) of its distance, so the body keeps going over the first foot instead of stopping on a foot far
    // behind; the need-a-step test keeps lead_t, so the next step comes at once. The absolute limit keeps strong pushes
    // braked (a share alone puts every foot too short). Not under a bullet pull: a force that keeps acting needs the
    // full place (the short place assumes free motion after a push).
    const double bkv = speed > 1e-6 && t >= b.pull_until ? std::fmax(-(v[0] * hx + v[1] * hy) / speed, 0.0) : 0.0;
    const double sl = speed * lead;
    const double fb = sl > 1e-6 ? std::fmax((double)pr[P_PLACE_BACK], 1.0 - pr[P_PLACE_BACK_D] * hs / sl) : 1.0;
    const double lp = lead * (1.0 - bkv * (1.0 - std::fmin(fb, 1.0)));
    const double pp[2] = {c[0] + v[0] * lp + in.lean[0] + plx, c[1] + v[1] * lp + in.lean[1] + ply};
    // A place for foot f stays on the support of the stance foot (the other one; f itself when only f is down):
    // along each support axis that ends in an edge (L3 deck_off / deck_half) its offset from the middle line shrinks
    // by in.deck_mid and is held deck_edge inside the edge, so a deck moving away under him does not carry the places
    // off its edge. The deck step keeps its stance width (mid share 0 there): a narrow stance on a moving deck tips.
    auto on_deck = [&](int f, double* p, double mid) {
      if (pr[P_DECK_EDGE] <= 0 && mid <= 0) return;
      const int sp = on[1 - f] ? 1 - f : (on[f] ? f : -1);
      if (sp < 0) return;
      for (int k = 0; k < 2; ++k) {
        const double hw = P.deck_half[sp][k];
        if (hw <= 0) continue;
        const double* ax = P.deck_ax[sp][k];
        const double off = P.deck_off[sp][k] + (p[0] - P.foot_c[sp][0]) * ax[0] + (p[1] - P.foot_c[sp][1]) * ax[1];
        double to = off * (1.0 - mid);
        const double lim = std::fmax(hw - pr[P_DECK_EDGE], 0.0);
        if (pr[P_DECK_EDGE] > 0) to = Clamp(to, -lim, lim);
        p[0] += (to - off) * ax[0];
        p[1] += (to - off) * ax[1];
      }
    };
    double dist[2];
    for (int f = 0; f < 2; ++f) {
      const int o = 1 - f;
      const double s = f == 0 ? 1.0 : -1.0;
      const double ux = pp[0] - S[o][0], uy = pp[1] - S[o][1], un = std::hypot(ux, uy) + 1e-9;
      const double cf = pr[P_PLACE_FWD] * hs * (ux * hx + uy * hy) / un;
      b.place[f][0] = pp[0] + s * half_w * lx + cf * hx;
      b.place[f][1] = pp[1] + s * half_w * ly + cf * hy;
      on_deck(f, b.place[f], in.deck_mid);
      dist[f] = std::hypot(b.place[f][0] - S[f][0], b.place[f][1] - S[f][1]);
    }
    if (vg_n > 1e-6 && pr[P_VG_LEAD] > 0) {  // the foot on the goal side steps out ahead of the CoM
      for (int f = 0; f < 2; ++f) {
        const double sg = f == 0 ? 1.0 : -1.0;
        // vg_nox: a swinging foot keeps the role (leading / trailing) it had at its lift-off; passing the stance foot
        // in the air does not make it the leading foot (its place would jump far ahead of the CoM)
        const double* sfe = (in.vnox > 0 && b.swing == f) ? b.sw_p0 : S[f];
        VgLeadPlace(b.place[f], sfe, S[1 - f], c, sg * half_w * lx, sg * half_w * ly, vg_g,
                    (in.vnox > 0 ? kHobLead : (double)pr[P_VG_LEAD]) * hs, 0.0 * hs);
        dist[f] = std::hypot(b.place[f][0] - S[f][0], b.place[f][1] - S[f][1]);
        if (in.vnox > 0 && on[1 - f]) {
          VgNoCross(b.place[f], sfe, S[1 - f], vg_g, (double)in.vnox);
          dist[f] = std::hypot(b.place[f][0] - S[f][0], b.place[f][1] - S[f][1]);
        }
      }
    }

    // thr_ahead: in a hurried dodge every step lands this far beyond the CoM along the escape (with its own side
    // offset), not under the body behind it (the velocity-goal lag)
    if (thr_w && pr[P_THR_AHEAD] > 0) {
      for (int f = 0; f < 2; ++f) {
        const double sg = f == 0 ? 1.0 : -1.0;
        const double ox = sg * half_w * lx, oy = sg * half_w * ly;
        const double pg = (b.place[f][0] - c[0] - ox) * vg_g[0] + (b.place[f][1] - c[1] - oy) * vg_g[1];
        const double want = pr[P_THR_AHEAD] * hs;
        if (pg < want) {
          b.place[f][0] += (want - pg) * vg_g[0];
          b.place[f][1] += (want - pg) * vg_g[1];
          dist[f] = std::hypot(b.place[f][0] - S[f][0], b.place[f][1] - S[f][1]);
        }
      }
    }
    // Throw: a dodge decided with little time left (kTurnTc .. throw_tc) does not walk out of the way: the escape-side
    // foot lunges first, at once, throw_lead beyond the CoM along the dodge (no crossover and no push-off on two feet,
    // which would take too long to clear the car's path)
    b.thr_on = pr[P_THROW] > 0 && b.dodge == 1 && b.dodge_tc >= kThrowTc0 && b.dodge_tc < pr[P_THROW_TC] && vg_n > 1e-6;
    const bool thr_first = b.thr_on && b.td_time < b.dodge_t0;  // no touchdown since the decision: the lunge
    if (b.thr_on) {
      if (b.thr_f < 0) {
        const double pj = (S[0][0] - S[1][0]) * vg_g[0] + (S[0][1] - S[1][1]) * vg_g[1];
        b.thr_f = pj > 0 ? 0 : 1;
      }
      if (thr_first) {
        const int f = b.thr_f;
        const double pl = pr[P_THROW_LEAD] * hs;
        b.place[f][0] = c[0] + pl * vg_g[0];
        b.place[f][1] = c[1] + pl * vg_g[1];
        dist[f] = std::hypot(b.place[f][0] - S[f][0], b.place[f][1] - S[f][1]);
      }
    } else {
      b.thr_f = -1;
    }
    // Swing target of foot f (sole centre): fresh place or a commanded one, clipped by the reach ellipse around the
    // point under the swing hip (hip mobility: extension and adduction limits, outward at most 0.38 m at 1.78 m).
    const double ext = -b.js[J_HIP_FLEX_L].lo, abd = b.js[J_HIP_ABD_L].hi, add = -b.js[J_HIP_ABD_L].lo;
    auto reach = [&](int f, double* p) {
      const double s = f == 0 ? 1.0 : -1.0, k = pr[P_STEP_LEN];
      const double cx = P.hip[f][0], cy = P.hip[f][1], rx = p[0] - cx, ry = p[1] - cy;
      const double a = rx * hx + ry * hy, l = rx * lx + ry * ly;
      double ax = k * leg * std::sin(ext);  // 30 deg either way: 0.43 m at 1.78 m
      double ay = s * l > 0 ? k * std::fmin(leg * std::sin(abd), 0.38 * hs) : k * leg * std::sin(add);
      if (pr[P_REACH_KEEP] > 0.5) {
        // Reach from the pelvis height to keep: the floor distance from the hip at which a leg with the touchdown
        // knee kTdKnee reaches the floor while the hip is kDropKeep below its standing height (a strong stagger lowers
        // the pelvis and spreads the feet wide). Fore-aft and outward; inward stays the adduction limit.
        const double le = std::sqrt(thigh * thigh + shank * shank + 2 * thigh * shank * std::cos(kTdKnee));
        const double l0 = std::sqrt(thigh * thigh + shank * shank + 2 * thigh * shank * std::cos(kStandKnee));
        const double vk = l0 - kDropKeep * hs;
        const double rk = k * std::sqrt(std::fmax(le * le - vk * vk, 0.0));
        ax = rk;
        if (s * l > 0) ay = rk;
      }
      // A weakened (wounded) leg reaches from its real hip height: a leg of length le (touchdown knee kTdKnee) over
      // the ankle height reaches the floor at most sqrt(le^2 - h^2) from the point under the hip (a farther place
      // leaves the hurt foot hanging in the air while the place runs on)
      if (in.leg_weak[f] < 1.0 && pr[P_REACH_HIP] > 0) {
        const double le = std::sqrt(thigh * thigh + shank * shank + 2 * thigh * shank * std::cos(kTdKnee));
        const double h = std::fmax(P.hip[f][2] - ah, 0.0);
        const double rh = std::sqrt(std::fmax(le * le - h * h, 0.0025));
        ax = std::fmin(ax, rh);
        ay = std::fmin(ay, rh);
      }
      if (thr_first && f == b.thr_f) return;  // the lunge is as long as the leg gets it
      const double rr = std::hypot(a / ax, l / ay);
      if (rr > 1) { p[0] = cx + rx / rr; p[1] = cy + ry / rr; }
    };
    auto target = [&](int f, double* p) {
      p[0] = b.place[f][0];
      p[1] = b.place[f][1];
      if (b.cmd_on && b.cmd_started == 1 && f == b.cmd_foot) {  // commanded step: the place is the command
        const double fy = fyaw[f];
        p[0] = b.cmd_tgt[0] + fx * std::cos(fy);
        p[1] = b.cmd_tgt[1] + fx * std::sin(fy);
        return;
      }
      if (b.sh_foot >= 0 && f != b.sh_foot) {  // commanded lift: the foot goes back to where it was
        p[0] = b.sw_p0[0];
        p[1] = b.sw_p0[1];
        return;
      }
      // Deck step: the capture place lies toward the stance foot while the pelvis lags the deck (the stance would
      // narrow and the next needed step would go off the deck), so the deck step keeps the stance width from the
      // standing foot and only takes the fore-aft of the place.
      if (b.dk_on && f == b.swing) {
        const int o = 1 - f;
        const double s = f == 0 ? 1.0 : -1.0;
        const double a = (p[0] - S[o][0]) * hx + (p[1] - S[o][1]) * hy;
        p[0] = S[o][0] + a * hx + s * 2 * half_w * lx;
        p[1] = S[o][1] + a * hy + s * 2 * half_w * ly;
      }
      if (b.dk_on && f == b.swing) on_deck(f, p, 0.0);
      {
        const double p0x = p[0], p0y = p[1];
        reach(f, p);
        if (f == b.swing) b.sw_short = std::hypot(p[0] - p0x, p[1] - p0y);  // place beyond the reach by [m]
      }
      // sw_commit: within |sw_commit| carrot lengths of its place the swing foot's place is frozen in the world for
      // the rest of the step, so the foot brakes to the floor instead of landing at the place's speed (a moving place
      // makes the toe touch down fast and slide). sw_commit > 0: only past half the step line start place -> place;
      // < 0: also at once, so a place near the foot at lift-off makes a short quick step under the body.
      if (pr[P_SW_COMMIT] != 0 && b.swing == f) {
        const double qx = p[0] - S[f][0], qy = p[1] - S[f][1], lx0 = p[0] - b.sw_p0[0], ly0 = p[1] - b.sw_p0[1];
        if (b.sw_cm) {
          p[0] = b.sw_cmp[0];
          p[1] = b.sw_cmp[1];
        } else if (std::hypot(qx, qy) < std::fabs(pr[P_SW_COMMIT]) * pr[P_CARROT] * hs &&
                   (pr[P_SW_COMMIT] < 0 || qx * lx0 + qy * ly0 < 0.5 * (lx0 * lx0 + ly0 * ly0))) {
          b.sw_cm = 1;
          b.sw_cmp[0] = p[0];
          b.sw_cmp[1] = p[1];
        }
      }
    };
    auto start = [&](int f) {
      b.swing = f;
      b.sw_cm = 0;
      b.sw_ft[0] = P.ankle[f][0];
      b.sw_ft[1] = P.ankle[f][1];
      b.sw_ft[2] = P.ankle[f][2];
      // Knee follower restarts at the knee: before the swing the soft unloaded knee lags its target, and a stiff
      // swing knee would close that gap at once with a huge acceleration.
      if (b.kf_on) {
        const int jk = f ? J_KNEE_R : J_KNEE_L;
        b.kf_q[f] = P.q[jk];
        b.kf_v[f] = (P.q[jk] - b.kq_prev[f]) / dt;
      }
      b.sw_p0[0] = S[f][0];
      b.sw_p0[1] = S[f][1];
      b.sw_air0 = !on[f];
      b.sw_ofar = dist[1 - f] > dist[f];
      double p[2];
      target(f, p);
      b.sw_first = pr[P_GU_FIRST] > 0.5 && (b.td_time < 0 || t - b.td_time > 1.0);  // first step from calm
      b.sw_tg[0] = p[0];
      b.sw_tg[1] = p[1];
      b.sw_far0 = b.sw_prev = b.sw_min = std::hypot(p[0] - S[f][0], p[1] - S[f][1]);
      b.sw_rem = b.sw_far0;
      b.sw_reach = 0;
      b.sw_stall = 0;
      b.sw_t0b = t;
      b.sw_omax = -1e9;
      b.tp_leg = -1;
      b.sw_vpk = 0;
      b.sw_y0 = P.pelvis_yaw;
      b.sw_ymin = b.sw_ymax = 0;
      b.sw_ps[0] = S[f][0];
      b.sw_ps[1] = S[f][1];
      b.steps++;
      b.step_t0 = t;
      b.step_start[0] = S[f][0];
      b.step_start[1] = S[f][1];
      b.yaw_hold = P.pelvis_yaw;
      if (pr[P_STEP_BUDGET] > 0) b.budget -= 1;
      if (b.in.steps_left > 0) b.in.steps_left -= 1;  // reaction reserve (L2 sets it)
    };
    // 3.5 Which foot, only while no step is going on.
    auto decide = [&]() {
      if (b.swing >= 0 || b.mode == 2 || b.cmd_on || b.sh_foot >= 0) return;
      b.dk_on = 0;
      if (pr[P_STEP_BUDGET] > 0 && b.budget <= 0) return;
      if (in.no_step || in.steps_left == 0) return;  // L2: death settles without steps; reaction reserve spent
      // bul_ds: a chest bullet series is braked in a wide double support: both feet stay down bul_ds after a
      // touchdown, so the trailing foot is not lifted at once and the body does not bounce off the straight leading
      // leg and run on.
      if (pr[P_BUL_DS] > 0 && b.hit_bullet && t < b.bul_until && on[0] && on[1] && b.td_time >= 0 &&
          t - b.td_time < pr[P_BUL_DS])
        return;
      int go = -1;
      for (int f = 0; f < 2; ++f) {
        const int o = 1 - f;
        const bool g = on[o] ? (((need || gather) && dist[f] > dist[o]) || !on[f]) : (!on[f] && dist[o] > dist[f]);
        if (g && (go < 0 || !on[f])) go = f;
      }
      b.dbg_dist[0] = dist[0], b.dbg_dist[1] = dist[1];
      if (thr_first && on[0] && on[1] && b.swing < 0) go = b.thr_f;  // the lunge at once
      if (vg_n > 1e-6 && pr[P_VG_LEAD] > 0 && on[0] && on[1] && go >= 0 && !thr_first) {  // leading foot first
        const double pj = (S[0][0] - S[1][0]) * vg_g[0] + (S[0][1] - S[1][1]) * vg_g[1];
        if (std::fabs(pj) > 0.0 * hs) go = pj > 0 ? 0 : 1;
      }
      if (go >= 0 && on[go] && dist[go] < kMinStep * hs) go = -1;
      // A weakened leg's step is measured to its reach-clipped target: when the clip leaves the hurt foot where it
      // stands, stepping it again only takes its weight support away (the knee folds under the load) while the healthy
      // foot never follows, so the other foot steps instead
      if (go >= 0 && on[go] && in.leg_weak[go] < 1.0 && pr[P_REACH_HIP] > 0) {
        double tc[2];
        target(go, tc);
        if (std::hypot(tc[0] - S[go][0], tc[1] - S[go][1]) < kMinStepW * hs) {
          const int o = 1 - go;
          // Hobble (goal with vnox): the healthy foot follows to kHobFollow stance widths from the hurt one against
          // the goal (short quick healthy steps after the long hurt one; its goal place would lie under it: no step)
          if (in.vnox > 0 && vg_n > 1e-6 && on[o]) {
            b.place[o][0] = S[go][0] - kHobFollow * 2 * half_w * vg_g[0];
            b.place[o][1] = S[go][1] - kHobFollow * 2 * half_w * vg_g[1];
            dist[o] = std::hypot(b.place[o][0] - S[o][0], b.place[o][1] - S[o][1]);
          }
          go = on[o] && dist[o] > kMinStepW * hs ? o : -1;  // the other (healthy) foot follows
        }
      }
      // Deck step: on a swaying bridge the prediction point stays well inside the stance, so no step is ever needed,
      // yet a person re-steps there. The foot the sway unloads (load < deck_ld, the CoM moving toward the other foot)
      // is re-placed under the pelvis, at most once per in.deck_step seconds.
      bool dk = false;
      if (go < 0 && in.deck_step > 0 && on[0] && on[1] && (b.td_time < 0 || t - b.td_time > in.deck_step)) {
        for (int f = 0; f < 2; ++f) {
          const int o = 1 - f;
          const double tox = S[o][0] - S[f][0], toy = S[o][1] - S[f][1];
          // and the deck moving toward the standing foot: the body lags toward the lifted foot, which lands under it
          // (with the deck moving the other way the pelvis would be left over the far edge)
          if (b.ld[f] < pr[P_DECK_LD] && v[0] * tox + v[1] * toy > 0 && in.sup_v[0] * tox + in.sup_v[1] * toy > 0) go = f;
        }
        dk = go >= 0;
      }
      // A foot hanging just above the floor at its place is put down by its standing leg, not stepped (after a
      // touchdown bounce it would otherwise be re-stepped by a few cm again and again).
      if (go >= 0 && !on[go] && dist[go] < kNearPlace * hs && P.ankle[go][2] < ah + kLowFoot * hs) go = -1;
      if (go >= 0) {
        b.dk_on = dk ? 1 : 0;
        b.dbg_why = dk ? 32 : (need ? 1 : 0) | (gather ? 2 : 0) | (on[go] ? 0 : 4) | (on[1 - go] ? 0 : 8) | (narrow ? 16 : 0);
        start(go);
      }
    };

    // Commands. Weight shift: the support middle is the chosen foot; with a lift the other foot rises 5 cm for
    // sh_lift seconds once the CoM is over the foot, then is put back and the middle returns between the feet.
    if (b.sh_foot >= 0) {
      const int sf = b.sh_foot, o = 1 - sf;
      const double el = std::hypot(c[0] - S[sf][0], c[1] - S[sf][1]);
      if (b.sh_state == 0 && b.sh_lift > 0 && b.swing < 0 && ((el < 0.02 * hs && speed < 0.1) || t - b.sh_t0 > 3.0)) {
        b.sh_state = 1;
        b.sh_up_t0 = t;
        start(o);
      }
      if (b.sh_state == 1 && t - b.sh_up_t0 >= b.sh_lift) b.sh_state = 2;
      if (b.sh_state == 3 && t - b.sh_dn_t0 > 0.6) {
        const double mx = 0.5 * (S[0][0] + S[1][0]), my = 0.5 * (S[0][1] + S[1][1]);
        if (std::hypot(c[0] - mx, c[1] - my) < 0.02 * hs || t - b.sh_dn_t0 > 2.0) b.sh_foot = -1;
      }
    }
    // Commanded step: the middle moves over the stance foot; the swing starts once the CoM is over it.
    if (b.cmd_on && !b.cmd_started && b.swing < 0) {
      const int sf = 1 - b.cmd_foot;
      const double el = (pt[0] - S[sf][0]) * lx + (pt[1] - S[sf][1]) * ly;  // predicted, not present CoM
      const double vl = v[0] * lx + v[1] * ly;
      if ((std::fabs(el) < 0.02 * hs && std::fabs(vl) < 0.08 && b.ld[b.cmd_foot] < 0.15) || t - b.cmd_t0 > b.cmd_to) {
        b.cmd_started = 1;
        start(b.cmd_foot);
      }
    }

    // 3.6 Swing: end of step (touching, no longer approaching or past the place, and past half the way or
    // "the other foot was farther and this one started in the air"; touching while the other foot is off). Then
    // the next decision at once.
    double tg[2] = {0, 0}, dsw = 0;
    if (b.swing >= 0) {
      const int f = b.swing, o = 1 - f;
      target(f, tg);
      const double dx = tg[0] - S[f][0], dy = tg[1] - S[f][1];
      dsw = std::hypot(dx, dy);
      const bool closing = dsw < b.sw_prev - kCloseV * hs * dt;
      const bool past = dx * (tg[0] - b.sw_p0[0]) + dy * (tg[1] - b.sw_p0[1]) < 0;
      const double el = t - b.sw_t0b;
      // Stall: the foot has not come closer to the (moving) place than before; then it is put down.
      // Not while the foot is short of kAhead ahead of the CoM along its motion (when the place runs with a fast body
      // a foot put down behind the CoM gives no support under it).
      const double vn = std::hypot(v[0], v[1]);
      const bool behind = vn > kBehindV && ((S[f][0] - c[0]) * v[0] + (S[f][1] - c[1]) * v[1]) < kAhead * hs * vn;
      if (dsw < b.sw_min - 0.002 * hs || el < 0.1 || behind) b.sw_stall = 0;
      else b.sw_stall += dt;
      b.sw_min = std::fmin(b.sw_min, dsw);
      bool end = false;
      if (b.sh_foot >= 0 && f != b.sh_foot) {
        end = b.sh_state == 2 && on[f] && el > 0.08;
        if (end) { b.sh_state = 3; b.sh_dn_t0 = t; }
      } else if (on[f] && el > 0.08 &&
                 (pr[P_TD_NOW] <= 0 || b.loaded[f] ||
                  (pr[P_TD_NOW] < 1.5 ? b.foot_fn[f] > 1.0 || P.ankle[f][2] < ah + kRest * hs
                                      : b.foot_fn[f] > 1.0 && P.ankle[f][2] < ah + kSoleDown * hs))) {
        // td_now: touching = floor force now or resting; the 25 ms load share alone stays above kTouch ~50 ms after
        // a one-frame brush, which would end the swing with the foot up again and lift the standing foot.
        // A foot that already carries weight stands (otherwise it drags on the floor after a receding place).
        end = !on[o] || b.loaded[f] || b.sw_stall > kStall ||
              ((!closing || past) && (dsw < 0.5 * b.sw_far0 || (b.sw_ofar && b.sw_air0)));
      } else if (el > 0.08 && b.foot_fn[f] > kTouchHard * b.dims.total_mass * 9.81 && dsw < 0.5 * b.sw_far0 &&
                 (pr[P_TD_NOW] < 1.5 || P.ankle[f][2] < ah + kSoleDown * hs)) {
        // td_now 2: a real touch only. The load share (25 ms) holds, or the floor pushes now with the sole down
        // (ankle within kSoleDown of standing); a toe brush with the ankle up is no touchdown.
        // A hard touch past half the path is a touchdown even before the 25 ms load filter sees it (otherwise the foot
        // bounces, the swing goes on with a bent knee and the second landing straightens the leg under load).
        end = true;
      }
      b.sw_prev = dsw;
      {  // bench measures: overshoot along the path, peak sole speed, pelvis yaw range
        const double px = tg[0] - b.sw_p0[0], py = tg[1] - b.sw_p0[1], pn = std::hypot(px, py);
        if (pn > 1e-3) b.sw_omax = std::fmax(b.sw_omax, ((S[f][0] - tg[0]) * px + (S[f][1] - tg[1]) * py) / pn);
        b.sw_vpk = std::fmax(b.sw_vpk, std::hypot(S[f][0] - b.sw_ps[0], S[f][1] - b.sw_ps[1]) / dt);
        b.sw_ps[0] = S[f][0];
        b.sw_ps[1] = S[f][1];
        const double dy = Wrap(P.pelvis_yaw - b.sw_y0);
        b.sw_ymin = std::fmin(b.sw_ymin, dy);
        b.sw_ymax = std::fmax(b.sw_ymax, dy);
      }
      if (end) {
        b.swing = -1;
        b.td_time = t;
        b.td_foot = f;
        b.sw_first = 0;
        b.td_over = std::fmax(b.sw_omax, 0.0);
        b.td_vpk = b.sw_vpk;
        b.td_dur = t - b.sw_t0b;
        b.td_len = std::hypot(S[f][0] - b.sw_p0[0], S[f][1] - b.sw_p0[1]);
        b.td_yaw = b.sw_ymax - b.sw_ymin;
        const double ex = S[f][0] - tg[0], ey = S[f][1] - tg[1];  // placement error (bench output)
        b.td_err[0] = ex * hx + ey * hy;
        b.td_err[1] = ex * lx + ey * ly;
        ++b.td_cnt;
        if (b.cmd_on && b.cmd_started == 1 && f == b.cmd_foot) { b.cmd_started = 2; b.cmd_ret_t0 = t; }
      }
    }
    if (b.cmd_on && b.cmd_started == 2) {  // commanded step landed: the middle returns between the feet, then free
      const double mx = 0.5 * (S[0][0] + S[1][0]), my = 0.5 * (S[0][1] + S[1][1]);
      const bool there = std::hypot(b.mid_c[0] - mx, b.mid_c[1] - my) < 0.03 * hs &&
                         std::hypot(c[0] - mx, c[1] - my) < 0.05 * hs && speed < 0.1;
      if (there || t - b.cmd_ret_t0 > 2.0) b.cmd_on = 0;
    }
    if (b.swing < 0) {
      decide();
      if (b.swing >= 0) {
        target(b.swing, tg);
        dsw = std::hypot(tg[0] - S[b.swing][0], tg[1] - S[b.swing][1]);
      }
    }
    b.mode = b.swing >= 0 ? 1 : 0;

    // 3.7 Balance numbers. Support middle: between the standing feet (plus posture and sway), the standing foot,
    // in a step where the CoM is flying; a command moves it over its foot.
    double mid[2];
    const bool two = on[0] && on[1] && b.swing < 0;
    if (b.sh_foot >= 0 && b.sh_state < 3) {
      mid[0] = S[b.sh_foot][0]; mid[1] = S[b.sh_foot][1];
    } else if (b.sh_foot >= 0) {  // returning between the feet over 0.6 s
      const double w = Clamp((t - b.sh_dn_t0) / 0.6, 0.0, 1.0), sf = b.sh_foot;
      mid[0] = (1 - w) * S[(int)sf][0] + w * 0.5 * (S[0][0] + S[1][0]);
      mid[1] = (1 - w) * S[(int)sf][1] + w * 0.5 * (S[0][1] + S[1][1]);
    } else if (b.cmd_on && b.cmd_started == 2) {
      mid[0] = 0.5 * (S[0][0] + S[1][0]) + pr[P_COM_FWD] * hs * hx;
      mid[1] = 0.5 * (S[0][1] + S[1][1]) + pr[P_COM_FWD] * hs * hy;
    } else if (b.cmd_on) {
      mid[0] = S[1 - b.cmd_foot][0]; mid[1] = S[1 - b.cmd_foot][1];
    } else if (b.swing >= 0) {
      // No lead time when backing off (the standing leg holds the pelvis where it is; a desired pelvis behind it would
      // flex the standing hip); full lead time sideways and forward, cosine in between.
      const double sp = std::hypot(v[0], v[1]), vb = sp > 1e-6 ? -(v[0] * hx + v[1] * hy) / sp : 0.0;
      // rlx_ldir 3: no support-middle lead time while the wounded body relaxes moving forward: a desired pelvis ahead
      // of the CoM straightens the knee over the just landed front foot and pushes the pelvis over it; without it the
      // standing legs fold under the pelvis.
      const double mk = rlx && pr[P_RLX_LDIR] > 2.5 ? rlx_wb : 1.0;
      const double mt = pr[P_MID_T] * ts * (1.0 - pr[P_MID_BACK] * Clamp(vb, 0.0, 1.0)) * mk;
      mid[0] = c[0] + v[0] * mt; mid[1] = c[1] + v[1] * mt;
    } else if (on[0] != on[1]) {
      const int f = on[0] ? 0 : 1;
      mid[0] = S[f][0]; mid[1] = S[f][1];
    } else {
      const double A = pr[P_SWAY_AMP] * hs, cf = pr[P_COM_FWD] * hs;
      const double sa = A * (std::sin(6.2831853 * b.sway_freq[0] * t + b.sway_phase[0]) +
                             0.5 * std::sin(6.2831853 * b.sway_freq[1] * t + b.sway_phase[1]));
      const double sl = 0.6 * A * std::sin(6.2831853 * b.sway_freq[2] * t + b.sway_phase[2]);
      mid[0] = 0.5 * (S[0][0] + S[1][0]) + (cf + sa) * hx + sl * lx;
      mid[1] = 0.5 * (S[0][1] + S[1][1]) + (cf + sa) * hy + sl * ly;
    }
    if (b.sh_foot >= 0 || b.cmd_on) {  // a command moves the middle at a limited speed (no jump of the target)
      if (!b.mid_c_on) {
        b.mid_c[0] = 0.5 * (S[0][0] + S[1][0]) + pr[P_COM_FWD] * hs * hx;
        b.mid_c[1] = 0.5 * (S[0][1] + S[1][1]) + pr[P_COM_FWD] * hs * hy;
        b.mid_c_on = 1;
      }
      const double ddx = mid[0] - b.mid_c[0], ddy = mid[1] - b.mid_c[1], dd = std::hypot(ddx, ddy);
      const double step = (b.cmd_on && b.cmd_started == 2 ? kMidV2 : kMidV) * hs * dt;
      if (dd > step) { b.mid_c[0] += ddx * step / dd; b.mid_c[1] += ddy * step / dd; }
      else { b.mid_c[0] = mid[0]; b.mid_c[1] = mid[1]; }
      mid[0] = b.mid_c[0];
      mid[1] = b.mid_c[1];
    } else {
      b.mid_c_on = 0;
    }
    mid[0] += in.lean[0];
    mid[1] += in.lean[1];
    const double ex = (c[0] - mid[0]) * hx + (c[1] - mid[1]) * hy, ey = (c[0] - mid[0]) * lx + (c[1] - mid[1]) * ly;
    const double vx = v[0] * hx + v[1] * hy, vy = v[0] * lx + v[1] * ly;
    const double imax = 0.02 * hs;
    if (two) {
      b.cor_i[0] = Clamp(b.cor_i[0] + ex * dt, -imax, imax);
      b.cor_i[1] = Clamp(b.cor_i[1] + ey * dt, -imax, imax);
    }
    const double td = pr[P_COR_TD] * ts, ki = pr[P_COR_KI] / ts;
    const double ul = ey + td * vy + ki * b.cor_i[1];  // lateral number [m]
    // bul_center: while a bullet reaction is live the position part counts this share (see the help below)
    // brk_after: braking and slow re-centring only once the pull has ended
    const bool brk_on = b.hit_bullet && t < b.bul_until && !(pr[P_BRK_AFTER] > 0 && t < b.pull_until);
    const double kxc = pr[P_BUL_CENTER] > 0 && brk_on ? (double)pr[P_BUL_CENTER] : 1.0;
    const double uf = kxc * ex + td * vx + ki * b.cor_i[0];  // fore-aft number [m] (two feet only)
    double flat = 1;  // fore-aft number fades when a standing foot is on its toes or heel
    for (int f = 0; f < 2; ++f)
      if (on[f]) flat = std::fmin(flat, 1.0 - Clamp((std::fabs(P.foot_R[f][6]) - 0.05) / 0.10, 0.0, 1.0));
    const double ank_f = two ? -pr[P_ANK_FB] * uf * flat : 0.0;
    const double ank_l = pr[P_ANK_LAT] * ul;
    b.cor_dbg[0] = uf, b.cor_dbg[1] = ul, b.cor_dbg[2] = ank_f, b.cor_dbg[3] = ank_l;
    // cop_k: the same numbers move the support point of the weight-holding torques (L0), so the standing feet put
    // the CoP beyond the CoM by the capture point error (LIP: CoM x'' = -w^2 (x + 0.3 v), damping ~0.5: a small push
    // stops within a few cm); the three-number ankle (kp 9 N m/rad) cannot do it through its angle.
    b.cop_sh[0] = b.cop_sh[1] = 0;
    const double ck = pr[P_COP_K];
    if (ck > 0) {
      const double sf = two ? uf * flat : 0.0;  // fore-aft only on two feet
      b.cop_sh[0] = ck * (sf * hx + ul * lx);
      b.cop_sh[1] = ck * (sf * hy + ul * ly);
    }
    // bul_brake: while a bullet reaction is live the support point goes fore-aft to the stopping point, the capture
    // point (LIP: CoP = x + v / omega gives x'' = -omega v), on one foot or two (L0 clips it into the soles it loads),
    // plus on two feet bul_center x the CoM offset from the middle (slow re-centring); it replaces the fore-aft part
    // of cop_k. The weight-holding torques alone hold the weight under the CoM while it is over the soles, so a rear
    // foot behind it brakes only from its toe and two feet not at all while the CoM moves toward the middle: the body
    // would coast on after each catch.
    if (pr[P_BUL_BRAKE] > 0 && brk_on && (on[0] || on[1])) {
      const double vh = v[0] * hx + v[1] * hy;
      const double sb = pr[P_BUL_BRAKE] * (vh / omega + (two ? kxc * ex : 0.0)) - ck * (two ? uf * flat : 0.0);
      b.cop_sh[0] += sb * hx;
      b.cop_sh[1] += sb * hy;
    }
    // pull_cop: under the bullet pull F the support point that keeps the CoM from accelerating lies F / (M w^2)
    // = bul_pull x CoM height further along the pull (LIP with a constant force: x'' = w^2 (x - p) + F / M). Without
    // it the weight-holding torques keep the CoP near the CoM, the floor does not resist the pull and the speed grows
    // until the pull ends.
    if (pr[P_PULL_COP] > 0 && t < b.pull_until && b.mode != 2 && (on[0] || on[1])) {
      const double dc = pr[P_PULL_COP] * pr[P_BUL_PULL] * std::fmax(c[2], 0.3);
      b.cop_sh[0] += dc * b.pull_dir[0];
      b.cop_sh[1] += dc * b.pull_dir[1];
    }
    // Leg length difference, pelvis and spine roll act kinematically only on two feet. On one foot a commanded roll
    // of the upper body pushes the CoM the other way while it turns (the CoM runs out over the sole edge with the
    // ankle at its limit).
    const double wl2 = on[0] && on[1] ? 1.0 : 0.0;
    const double dlen = pr[P_LEN_LAT] * ul * wl2;
    // 3.7: on one leg the desired pelvis pitches FORWARD in proportion to the backward CoM speed, with a limit, fresh
    // every frame; none moving forward or on two feet (a counter-lean against a back push). A sum of
    // pitch_v vx dt over time would lean the trunk back while backing off.
    // On two feet the counter-lean fades with the time constant lean_decay instead of vanishing (a reset at the first
    // touchdown would drop the standing hip target at once and throw the trunk back).
    // rlx_nolean: no one-leg counter-lean while the wounded body relaxes on the move (the L2 walk flexes the standing
    // hips itself; together the standing hip target runs far ahead of the joint and swings the leg off the floor).
    // Above 1.6 m/s backward the slope grows (pitch_v2) and the limit is higher (pitch_max): on strong back pushes the
    // hips bend more against the push and the trunk tilts back less.
    const double lean1 =
        rlx && pr[P_RLX_NOLEAN] > 0
            ? 0.0
            : Clamp(pr[P_PITCH_V] * std::fmax(-vx, 0.0) + pr[P_PITCH_V2] * std::fmax(-vx - 1.6, 0.0), 0.0,
                    (double)pr[P_PITCH_MAX]);
    b.pitch_acc = b.swing >= 0 ? lean1
                               : (pr[P_LEAN_DECAY] > 0 ? b.pitch_acc * std::exp(-dt / pr[P_LEAN_DECAY]) : 0.0);
    // A share of the counter-lean bends the spine forward (the chest push bends it back; the hips' part is held by
    // the strut budget).
    // lean_spine_push: after a push (not a bullet) the spine bends forward by the whole counter-lean: the chest stays
    // over the hips while the steps catch the body. Under bullets the series keeps pushing, so it stays off there.
    // Both push laws fade out between a chest tilt back of 15 and 25 deg: past that the body is falling (a recovering
    // chest stays near upright); kept on, the bent spine and the freer hips hold the pelvis up and the trunk topples
    // from higher.
    const double chb = -std::atan2(P.chest_up[0] * hx + P.chest_up[1] * hy, P.chest_up[2]);  // + = top back
    b.ch_w = Clamp((kChFade1 - chb) / (kChFade1 - kChFade0), 0.0, 1.0);
    // ... and after pushes too fast for the steps to catch (peak backward CoM speed of this recovery): there the
    // hip damping that turns the trunk back sits the body down first, a softer fall
    if (b.swing < 0 && speed < 0.15) b.ch_vpk = 0;
    b.ch_vpk = std::fmax(b.ch_vpk, -vx);
    const double sv = std::sqrt(hs);
    b.ch_w *= Clamp((kChV1 * sv - b.ch_vpk) / ((kChV1 - kChV0) * sv), 0.0, 1.0);
    // The speed peaks ~0.15 s after the hit, while the spine already bends: the hit's own velocity change fades the
    // laws from the first frame (any direction).
    if (b.hit_t >= 0 && t - b.hit_t < kChHitT) {
      b.ch_w *= Clamp((kChDv1 * sv - b.hit_dv) / ((kChDv1 - kChDv0) * sv), 0.0, 1.0);
      // sagittal laws: only for hits from the front (backward along the heading); side pushes keep the plain trunk
      // (a forward-bent spine does not help against a sideways fall)
      const double cb = -(b.hit_dir[0] * hx + b.hit_dir[1] * hy);
      b.ch_w *= Clamp((cb - kChCos0) / (kChCos1 - kChCos0), 0.0, 1.0);
      // A high push throws the chest itself back. Then bending the spine forward does not hold it and the steps
      // must come first: the laws are off, latched for this hit.
      // The laws wait for the sign that the push moved the hips: the chest goes forward first (a push at the hips
      // tips the trunk forward relative to them; a high push does not).
      if (t - b.hit_t < 0.02) { b.ch_hi = 0; b.ch_ok = 0; b.ch_t0 = chb; }
      if (t - b.hit_t < kChHiT) {
        if (chb > kChHiThr) b.ch_hi = 1;
        if (chb < b.ch_t0 - kChOk) b.ch_ok = 1;
      }
      if (b.ch_hi || !b.ch_ok) b.ch_w = 0;
    } else {
      b.ch_w = 0;  // push laws: only after a hit
    }
    const double lsp0 = pr[P_LEAN_SPINE];
    const double lsp = !b.hit_bullet && pr[P_LS_PUSH] >= 0 ? lsp0 + b.ch_w * (pr[P_LS_PUSH] - lsp0) : lsp0;
    T[J_LUMBAR_PITCH] += 0.5 * lsp * b.pitch_acc;
    T[J_CHEST_PITCH] += 0.5 * lsp * b.pitch_acc;
    T[J_LUMBAR_ROLL] = T[J_CHEST_ROLL] = 0.5 * pr[P_SPINE_LAT] * ul * wl2;  // + leans the trunk right (axis +x)
    // Chest hold: backing off, the chest is kept over the hips by pitching the desired pelvis forward (hip flexion),
    // in proportion to the chest's tilt back from vertical along the heading (plus a rate term); otherwise the spine
    // springs back after a push and the chest tilts far back.
    {
      const double rate = b.ch_prev != 0 ? (chb - b.ch_prev) / dt : 0.0;
      b.ch_prev = chb;
      double add = 0;
      if (pr[P_CH_HOLD] > 0 && b.mode != 2 && vx < -0.3)
        add = Clamp(pr[P_CH_HOLD] * (chb - pr[P_CH_THR] + pr[P_CH_TD] * rate), 0.0, (double)pr[P_CH_MAX]);
      b.ch_add = add;
      T[J_LUMBAR_PITCH] += 0.5 * pr[P_CH_SPINE] * add;
      T[J_CHEST_PITCH] += 0.5 * pr[P_CH_SPINE] * add;
    }

    // 3.8 Desired pelvis (hip midpoint): posture by heading with the accumulated pitch and the lateral roll, set
    // against the present pelvis rotation; place over the support middle keeping the present pelvis-CoM offset;
    // height from a compass over the standing feet (nearly straight legs), a bit higher on one leg.
    const double pitch_rate = P.pelvis_rate[0] * hx + P.pelvis_rate[1] * hy;  // + = top forward
    const double rollL_rate = P.pelvis_rate[1] * hx - P.pelvis_rate[0] * hy;  // + = top to the left
    // Heading: the calm-stance heading is held through a recovery (steps do not turn the body; holding the heading
    // each step started with lets the pelvis yaw ratchet step by step). On two feet the pelvis goes back to the feet's
    // mean heading slowly.
    if (b.swing < 0 && speed < 0.1 && (b.td_time < 0 || t - b.td_time > 0.5)) b.head_ref = P.pelvis_yaw;
    // Switch turn_go: turn along the motion. After a side push a person turns and moves off facing the motion
    // (forward steps reach farther than side steps): while stepping with the CoM moving more than 45 deg off the held
    // heading and faster than turn_v, the held heading becomes the motion direction; the standing hip's twist (+-45
    // deg) and the swing foot's heading carry the pelvis there over the next steps.
    // turn_bul: in bullet series the body turns slowly and little (each hit pushes again before a turn helps)
    const bool bul_live = b.hit_bullet && t < b.bul_until;
    if (pr[P_TURN_GO] > 0 && b.swing >= 0 && speed > pr[P_TURN_V] && !(pr[P_TURN_BUL] > 0 && bul_live)) {
      const double hm = std::atan2(v[1], v[0]), off = std::fabs(Wrap(hm - b.head_ref));
      if (off > 0.785 && off < 2.36) b.head_ref = hm;
    }
    if (in.head_set) b.head_ref = in.head_goal;  // L2's heading (turn away from a car)
    double yaw_t = P.pelvis_yaw;
    if (in.head_set) {  // L2's heading on one foot or two (turn away from a car; without it on two feet the feet's
      // mean heading would hold the pelvis)
      yaw_t += Clamp(Wrap(b.head_ref - P.pelvis_yaw), -0.17, 0.17);
    } else if (two) {
      const double fm = std::atan2(std::sin(fyaw[0]) + std::sin(fyaw[1]), std::cos(fyaw[0]) + std::cos(fyaw[1]));
      yaw_t += Clamp(0.5 * Wrap(fm - P.pelvis_yaw), -0.1, 0.1);
    } else {
      // One foot: toward the held heading (a rate-limited return to the calm heading did not place steps better).
      yaw_t += Clamp(Wrap(b.head_ref - P.pelvis_yaw), -0.17, 0.17);
    }
    const double Tr = pr[P_DAMP_RATIO] + dt;  // muscle response time + frame delay [s]
    // bul_yawd: a rate term for the heading under a bullet series (pitch and roll have one), so the pelvis does not
    // swing from side to side with each back step.
    if (pr[P_BUL_YAWD] > 0 && bul_live) yaw_t -= pr[P_BUL_YAWD] * Tr * P.pelvis_w[2];
    // rlx_pitch: relaxing on the move the desired pelvis pitches forward by rlx_pitch x (1 - tone) [rad] through the
    // standing legs' solution (feet stay where they stand): a squatting body keeps its trunk over the feet instead of
    // falling back (it needs the L2 standing hip flexion with it).
    const double rlx_th = b.dead_t >= 0 && !in.no_step && pr[P_RLX_PITCH] > 0
                              ? pr[P_RLX_PITCH] * (1.0 - Clamp((double)in.tone, 0.0, 1.0))
                              : 0.0;
    // lean_hip_push: after a push the hips take part of the forward fold (the pelvis tilts forward, the back stays
    // straighter), from the first touchdown on: during the first catching swing it would load the swing hip, and
    // over two feet only it would never act (the steps follow each other).
    const bool lh_on = !b.hit_bullet && b.hit_t >= 0 && b.td_time > b.hit_t + kLhTd;  // after the first catch
    const double lhp = lh_on ? pr[P_LH_PUSH] * b.ch_w * b.pitch_acc : 0.0;
    b.lh_add = lhp;
    const double th_y = b.pitch_acc + lhp + (1.0 - pr[P_CH_SPINE]) * b.ch_add + rlx_th + in.pel_pitch - Tr * pitch_rate;
    const double th_x = pr[P_ROLL_LAT] * ul * wl2 + Tr * rollL_rate;  // Rx angle: + = left side up, top right
    const Mat3 Rd = Mul(RotZ(yaw_t), Mul(RotY(th_y), RotX(th_x)));
    const double zw[3] = {0, 0, 1};
    double up_d[3];
    MulTV(Rd, zw, up_d);
    const double hm[3] = {0.5 * (P.hip[0][0] + P.hip[1][0]), 0.5 * (P.hip[0][1] + P.hip[1][1]),
                          0.5 * (P.hip[0][2] + P.hip[1][2])};
    const double hhw = 0.5 * std::sqrt((P.hip[0][0] - P.hip[1][0]) * (P.hip[0][0] - P.hip[1][0]) +
                                       (P.hip[0][1] - P.hip[1][1]) * (P.hip[0][1] - P.hip[1][1]) +
                                       (P.hip[0][2] - P.hip[1][2]) * (P.hip[0][2] - P.hip[1][2]));
    double Hd[3] = {mid[0] + hm[0] - c[0], mid[1] + hm[1] - c[1], hm[2]};
    // pull_lean 2: on two feet the desired pelvis rests in front of the support against the pull (over the middle the
    // legs only match the pull and the body keeps moving with it)
    if (b.swing < 0 && pr[P_PULL_LEAN] > 1.5) {
      Hd[0] -= plx;
      Hd[1] -= ply;
    }
    if (b.swing < 0 || b.cmd_on || b.sh_foot >= 0) {  // fixed middle: the place moves against the CoM velocity
      const double tp = pr[P_PLACE_TD] * ts;
      Hd[0] -= tp * v[0];
      Hd[1] -= tp * v[1];
    }
    double ho[2][3];  // desired hip offsets from the midpoint (world)
    // st_yawf: the standing leg is solved in the pelvis frame with its present heading, so only the hip twist (torque
    // limited by the sole friction, st_twist) carries the heading correction. In the desired heading frame the flexion
    // and abduction of a tilted standing leg would turn the pelvis too and the foot with it (a foot on its edge has
    // little friction against the turn).
    const bool syf = pr[P_ST_YAWF] > 0;
    const Mat3 Rds = syf ? Mul(RotZ(P.pelvis_yaw), Mul(RotY(th_y), RotX(th_x))) : Rd;
    double hos[2][3];  // standing-leg hip offsets in that frame
    for (int f = 0; f < 2; ++f) {
      const double off[3] = {0, (f == 0 ? 1.0 : -1.0) * hhw, 0};
      MulV(Rd, off, ho[f]);
      MulV(Rds, off, hos[f]);
    }
    {
      const double Lc = pr[P_LEG_FRAC] * leg, zlow = Lc - pr[P_DROP_MAX] * hs;
      double zt = 1e9;
      for (int f = 0; f < 2; ++f) {
        if (f == b.swing) continue;
        const double qx = Hd[0] + ho[f][0] - P.ankle[f][0], qy = Hd[1] + ho[f][1] - P.ankle[f][1];
        const double zf = std::fmax(std::sqrt(std::fmax(Lc * Lc - qx * qx - qy * qy, 0.0)), zlow);
        zt = std::fmin(zt, ah + zf - ho[f][2]);  // ankle at the floor height under the foot
      }
      // 3.8: in a step the compass also runs from the point the CoM flies to (CoM + v mid_t, without the backing-off
      // cut of the horizontal middle) to each foot: the standing foot where it stands, the swing foot at its frame
      // target of the last frame. Distance minus drop_c0 (about the stance half width: a plain stance gives no drop,
      // sideways less than fore-aft), times drop_k, into an arc of the leg's radius; limit drop_max. The leg reaches
      // its place from the lowered pelvis and lands bent; there is no touchdown knee rule.
      if (b.swing >= 0 && pr[P_DROP_K] > 0) {
        const double mt = pr[P_MID_T] * ts;
        const double fp[2] = {c[0] + v[0] * mt, c[1] + v[1] * mt};
        for (int f = 0; f < 2; ++f) {
          const double px = f == b.swing ? b.sw_ft[0] : P.ankle[f][0], py = f == b.swing ? b.sw_ft[1] : P.ankle[f][1];
          const double de = pr[P_DROP_K] * std::fmax(std::hypot(px - fp[0], py - fp[1]) - pr[P_DROP_C0] * hs, 0.0);
          const double zf = std::fmax(std::sqrt(std::fmax(Lc * Lc - de * de, 0.0)), Lc - pr[P_DROP_LIM] * hs);
          zt = std::fmin(zt, ah + zf - ho[f][2]);
        }
      }
      if (zt < 1e8) Hd[2] = zt + (b.swing >= 0 ? 0.01 * hs : 0.0) - std::fmax(in.crouch, 0.0);
      // rlx_tsink: relaxing on the move the desired pelvis goes down no faster than the weight support fades: at most
      // rlx_tsink x hs x (1 - tone) under the compass height (a fast crouch folds the standing legs ahead of the body)
      if (rlx && pr[P_RLX_TSINK] > 0 && zt < 1e8)
        Hd[2] = std::fmax(Hd[2], zt - pr[P_RLX_TSINK] * hs * (1.0 - Clamp((double)in.tone, 0.0, 1.0)));
      if (b.swing >= 0 && (b.sw_stall > kStall || b.sw_reach) && pr[P_STALL_REACH] > 0) {
        b.sw_reach = 1;  // held to the end of the swing (the stall resets on every 2 mm gain: the pelvis would bob)
        // A stalled swing whose place is out of reach at this pelvis height gets it by a lower pelvis (the standing
        // knee bends); otherwise a long commanded step leaves the foot hanging above the floor with the CoM held over
        // the standing foot. Not below the compass low.
        const int f = b.swing, o = 1 - f;
        const double fyd = yaw_t + (f == 0 ? 1.0 : -1.0) * pr[P_TOE_OUT];
        const double qx = Hd[0] + ho[f][0] - (tg[0] - fx * std::cos(fyd));
        const double qy = Hd[1] + ho[f][1] - (tg[1] - fx * std::sin(fyd));
        const double Lr = kStallReach * leg;
        const double zr = ah + std::sqrt(std::fmax(Lr * Lr - qx * qx - qy * qy, 0.0)) - ho[f][2];
        Hd[2] = std::fmax(std::fmin(Hd[2], zr), ah + zlow - ho[o][2]);
      }
      // bul_reach: under the bullet pull the swing leg would straighten to a far place from a high pelvis, hit the
      // floor on a straight knee and the body would bounce and vault over it. Here the pelvis is kept low enough for
      // the leg with the touchdown knee kTdKnee to reach the place, not below the compass low.
      if (pr[P_BUL_REACH] > 0 && b.swing >= 0 && b.hit_bullet && t < b.bul_until) {
        const int f = b.swing, o = 1 - f;
        const double fyd = yaw_t + (f == 0 ? 1.0 : -1.0) * pr[P_TOE_OUT];
        const double qx = Hd[0] + ho[f][0] - (tg[0] - fx * std::cos(fyd));
        const double qy = Hd[1] + ho[f][1] - (tg[1] - fx * std::sin(fyd));
        const double le = std::sqrt(thigh * thigh + shank * shank + 2 * thigh * shank * std::cos(kTdKnee));
        const double zr = ah + std::sqrt(std::fmax(le * le - qx * qx - qy * qy, 0.0)) - ho[f][2];
        Hd[2] = std::fmax(std::fmin(Hd[2], zr), ah + zlow - ho[o][2]);
      }
      // The desired pelvis rises at most drop_up (after a touchdown the lowered pelvis comes back up gradually).
      if (pr[P_DROP_UP] > 0 && b.hd_on) Hd[2] = std::fmin(Hd[2], b.hd_prev + pr[P_DROP_UP] * hs * dt);
      // rlx_gap: while the wounded body relaxes on the move the desired height stays at most rlx_gap below the present
      // hips (a target far below folds the standing leg faster than the body falls, the foot leaves the floor and the
      // body drops freely) and never rises (the compass over the standing foot would lift it as the body passes over).
      if (rlx && pr[P_RLX_GAP] > 0) {
        Hd[2] = std::fmax(Hd[2], hm[2] - pr[P_RLX_GAP] * hs);
        if (b.hd_on) Hd[2] = std::fmin(Hd[2], b.hd_prev);
      }
      // rlx_noext: and never above the present hips: a desired height from before the death can lie above them; the
      // standing leg would straighten toward it, lift the pelvis and shove the body on
      if (rlx && pr[P_RLX_NOEXT] > 0) Hd[2] = std::fmin(Hd[2], hm[2]);
      b.hd_prev = Hd[2];
      b.hd_on = 1;
      b.dbg_h[0] = Hd[2];
      b.dbg_h[1] = hm[2];
    }

    // 3.9 Legs. Standing leg: ankle in the desired pelvis frame, sole flat on the floor, hip twist from the foot
    // heading; plus the leg length difference and the ankle parts of the balance numbers. Swing leg: frame target
    // (place at its height, aimed past it, carrot), hip moved by the CoM velocity over the response time.
    Mat3 Ra;
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) Ra.m[i][k] = P.pelvis_R[3 * i + k];
    double up_a[3];
    MulTV(Ra, zw, up_a);
    // Swing damping multiplier (L0 damping = kp x damp_ratio x zm); also sets the swing response time Ts = damp_ratio x
    // zsw + dt. With leg_w_sw > 0: zm = 2 z / (w damp_ratio), so Ts = 2 z / w + dt.
    // The swing targeting keeps the table response time also with the three-number muscles: their own 2 z / w + dt
    // (0.083 s at 30 rad/s) is too short a preview for the carrot and makes back steps late.
    const double zsw = pr[P_LEG_W_SW] > 0
                           ? 2.0 * pr[P_LEG_Z_SW] / (pr[P_LEG_W_SW] * std::fmax(1e-3, (double)pr[P_DAMP_RATIO]))
                           : pr[P_SWING_Z] / std::sqrt(pr[P_SWING_K]);
    for (int f = 0; f < 2; ++f) {
      const double s = f == 0 ? 1.0 : -1.0;
      const int j0 = f == 0 ? J_HIP_FLEX_L : J_HIP_FLEX_R;
      Leg6 a;
      if (f != b.swing) {
        // The foot is taken at the floor height under it: a standing leg whose foot was lifted reaches down.
        // st_ank: a loaded foot whose heel is up (trailing leg past the 20 deg dorsiflexion limit) stands where it is;
        // solved to the floor the leg would stay straight and hold the pelvis above the desired height, so a catching
        // leg reaching from it would hang.
        const double az = pr[P_ST_ANK] > 0 && b.loaded[f] ? std::fmax(ah, P.ankle[f][2]) : ah;
        double rw[3] = {P.ankle[f][0] - (Hd[0] + hos[f][0]), P.ankle[f][1] - (Hd[1] + hos[f][1]),
                        az - (Hd[2] + hos[f][2])};
        // An unloaded standing foot on the far side of a sideways motion reaches the floor from the higher of the
        // desired and the present hip; a lower desired pelvis only lifts it while the other leg holds the pelvis (the
        // knee bends, the foot rises and gets stepped). Only that foot: fore-aft pushes need the lowered pelvis.
        const double vyl = v[0] * lx + v[1] * ly, sfl = (S[f][0] - c[0]) * lx + (S[f][1] - c[1]) * ly;
        // Sideways at least 0.15 m/s: settling sway stays below it, so the rule does not flicker on a just landed foot
        // while vx crosses zero.
        if (pr[P_FOOT_DOWN] > 0 && !b.loaded[f] && std::fabs(vyl) > std::fmax(std::fabs(vx), 0.15) && sfl * vyl < 0)
          rw[2] = ah - std::fmax(Hd[2] + ho[f][2], P.hip[f][2]);
        // rlx_trail: relaxing on the move forward, a standing foot behind its hip along the motion stays loaded: its
        // leg reaches the floor from at most rlx_trail below the present hip. A desired pelvis far under the hips folds
        // the trailing knee faster than the body falls, and the body drops freely before the swing lands. Forward share
        // of the motion (1 - backward share of rlx_ldir); behind by more than 0.1 hs, CoM faster than 0.3 m/s.
        if (rlx && pr[P_RLX_TRAIL] > 0) {
          const double vn = std::hypot(v[0], v[1]);
          const double bh = vn > 0.3 ? ((P.hip[f][0] - P.ankle[f][0]) * v[0] + (P.hip[f][1] - P.ankle[f][1]) * v[1]) / vn
                                     : 0.0;
          if (bh > 0.1 * hs) {
            const double hz = az - rw[2], hp = P.hip[f][2] - pr[P_RLX_TRAIL] * hs;
            const double wt = pr[P_RLX_LDIR] > 0 ? 1.0 - rlx_wb : 1.0;  // all directions without rlx_ldir
            if (hp > hz) rw[2] = az - (hz + wt * (hp - hz));
          }
        }
        if (pr[P_STAND_CLIP] > 0.5) {
          // 3.9: the standing foot is clipped by the reach ellipse around the desired hip, so the leg does not reach
          // toward the desired pelvis farther than its travel.
          const double a = rw[0] * hx + rw[1] * hy, l = rw[0] * lx + rw[1] * ly;
          const double ax = pr[P_STEP_LEN] * leg * std::sin(ext);
          const double ay = s * l > 0 ? pr[P_STEP_LEN] * std::fmin(leg * std::sin(abd), 0.38 * hs)
                                      : pr[P_STEP_LEN] * leg * std::sin(add);
          const double rr = std::hypot(a / ax, l / ay);
          if (rr > 1) {
            rw[0] /= rr;
            rw[1] /= rr;
          }
        }
        double r[3];
        MulTV(Rds, rw, r);
        // CoM to the left (ul > 0): the leg whose foot is more to the left gets longer. By the feet's lateral order,
        // not left / right (with crossed feet a left-right rule pushes the CoM further out); fades to 0 for feet in a
        // line.
        const double so = Clamp(((S[f][0] - S[1 - f][0]) * lx + (S[f][1] - S[1 - f][1]) * ly) / (0.05 * hs), -1.0, 1.0);
        // Only the leg on the error side lengthens (full difference); the other is not shortened: a shortened standing
        // leg lifts its foot, the foot hangs unloaded and the hanging-foot rule steps it in place again and again.
        r[2] -= std::fmax(so * dlen, 0.0);
        if (f == 1) {
          const double ex = P.hip[f][0] - P.ankle[f][0], ey = P.hip[f][1] - P.ankle[f][1], ez = P.hip[f][2] - P.ankle[f][2];
          b.dbg_h[2] = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) - std::sqrt(ex * ex + ey * ey + ez * ez);
          b.dbg_h[3] = dlen;
        }
        LegIK6(r, s * Wrap(fyaw[f] - yaw_t), s, thigh, shank, &a, 0.995, up_d);
        // st_toe: the standing leg is nearly straight (knee 16 deg: 4 mm of reach left), so a far hip that rises with
        // the body roll would lift the whole foot and the hanging-foot rule would step across. Instead the heel rises
        // with the toe on the floor: the sole is tilted toe down by the length the leg lacks (ankle-toe
        // lever).
        double toe_dn = 0;
        if (pr[P_ST_TOE] > 0 && !b.loaded[f] &&
            (pr[P_ST_TOE_SIDE] <= 0 || std::fabs(v[0] * lx + v[1] * ly) > std::fabs(v[0] * hx + v[1] * hy))) {
          const double lr = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) - 0.995 * (thigh + shank);
          const double lt = std::fmax(b.dims.foot_len - b.dims.heel_back, 0.05);
          if (lr > 0) toe_dn = std::asin(Clamp(pr[P_ST_TOE] * lr / lt, 0.0, 0.7));
        }
        if (pr[P_ST_INVLIM] > 0) {
          // st_invlim: the ankle inversion of the solution is set by where the hip is over the flat foot (the hip is a
          // ball joint). Past the range end (eversion -15 deg) the ankle target would be clipped while the hip and knee
          // still drive the pelvis there, and the foot would roll onto its edge. The ankle target is moved sideways in
          // the hip frame (secant on r_y) until the inversion is inside the range by the margin; the CoP part (ank_l)
          // is added after.
          const int ji = f ? J_ANKLE_INV_R : J_ANKLE_INV_L;
          const double ilo = b.js[ji].lo + pr[P_ST_INVLIM], ihi = b.js[ji].hi - pr[P_ST_INVLIM];
          if (a.inv < ilo || a.inv > ihi) {
            const double goal = a.inv < ilo ? ilo : ihi;
            double y0 = r[1], i0 = a.inv, y1 = r[1] + 0.02;
            Leg6 a1;
            for (int it = 0; it < 4; ++it) {
              const double r1[3] = {r[0], y1, r[2]};
              LegIK6(r1, s * Wrap(fyaw[f] - yaw_t), s, thigh, shank, &a1, 0.995, up_d);
              if (std::fabs(a1.inv - goal) < 0.002 || std::fabs(a1.inv - i0) < 1e-7) break;
              const double y2 = y1 + (goal - a1.inv) * (y1 - y0) / (a1.inv - i0);
              y0 = y1;
              i0 = a1.inv;
              y1 = Clamp(y2, r[1] - 0.3, r[1] + 0.3);
            }
            if (std::fabs(a1.inv - goal) < std::fabs(a.inv - goal)) a = a1;
          }
        }
        a.dorsi += ank_f - toe_dn;
        // Ankle lateral part: the CoP goes to the CoM error side, so the floor pushes the CoM back (the opposite sign
        // is positive feedback: the CoP moves inside and the CoM is pushed further out).
        a.inv += s * ank_l;
      } else {
        const int o = 1 - f;
        const double dx = tg[0] - S[f][0], dy = tg[1] - S[f][1];
        // Step height by direction: step_h forward and sideways, step_h_back for the backward part of the step
        // (forward and side steps are kept lower).
        const double dsh = std::hypot(dx, dy);
        const double bk = dsh > 1e-4 ? std::fmax(-(dx * hx + dy * hy) / dsh, 0.0) : 0.0;
        const double hstep = (pr[P_STEP_H] + (pr[P_STEP_H_BACK] - pr[P_STEP_H]) * bk) * hs;
        // Height (3.6). Base: half the path left along the step line (start place -> place) one response time ahead,
        // i.e. now minus what the foot gains on the place in that time; within +-step height.
        // Negative: the target is under the floor, the leg is pressed down and the step ends on touch. So the foot
        // comes down early instead of in the last frame. Plus the toe-down part; plus lift0 when it left the ground.
        // All capped by the hip's travel reserve toward the step: leg x (angle left from the present hip flexion /
        // abduction along the direction start-tilt -> place-tilt to the limit ellipse + kHipMargin); none left ->
        // height 0 or below, the foot goes to the floor where the leg reaches (back and cross steps come out low).
        double lift;
        {
          const double lx0 = tg[0] - b.sw_p0[0], ly0 = tg[1] - b.sw_p0[1], ln0 = std::hypot(lx0, ly0);
          const double ux = ln0 > 1e-6 ? lx0 / ln0 : (dsw > 1e-9 ? dx / dsw : 1.0);
          const double uy = ln0 > 1e-6 ? ly0 / ln0 : (dsw > 1e-9 ? dy / dsw : 0.0);
          const double along = dx * ux + dy * uy;
          const double gain = std::fmax((b.sw_rem - along) / dt, 0.0);  // closing speed on the place [m/s]
          b.sw_rem = along;
          const double Tl = pr[P_DAMP_RATIO] * zsw + dt;  // muscle response time (3.9)
          // lift0 (from the ground) joins the base before the step-height cap: the peak stays the step height.
          lift = Clamp(0.5 * (along - gain * Tl) + (b.sw_air0 ? 0.0 : pr[P_LIFT0] * hs), -hstep, hstep);
          // sw_press_b: in a backward step the press under the floor is scaled down: a deep target drives the knee to
          // full reach just before the touch, and the straight leg lands hard with no stroke left to brake.
          if (lift < 0) lift *= 1.0 - (1.0 - pr[P_SW_PRESS_B]) * bk;
          lift += std::fmax(-P.foot_R[f][6], 0.0) * fx;
          // Hip reserve: tilts (flexion, abduction) of the leg aimed at the start place and at the place, both from
          // the present hip in the present pelvis frame; the present tilt is the hip joint itself.
          const double fy0 = yaw_t + s * pr[P_TOE_OUT];
          const double cy0 = fx * std::cos(fy0), sy0 = fx * std::sin(fy0);
          const double w0[3] = {b.sw_p0[0] - cy0 - P.hip[f][0], b.sw_p0[1] - sy0 - P.hip[f][1], ah - P.hip[f][2]};
          const double w1[3] = {tg[0] - cy0 - P.hip[f][0], tg[1] - sy0 - P.hip[f][1], ah - P.hip[f][2]};
          double r0[3], r1[3];
          MulTV(Ra, w0, r0);
          MulTV(Ra, w1, r1);
          Leg6 a0, a1;
          LegIK6(r0, 0.0, s, thigh, shank, &a0, 0.995, up_a);
          LegIK6(r1, 0.0, s, thigh, shank, &a1, 0.995, up_a);
          double df = a1.flex - a0.flex, da = a1.abd - a0.abd;
          const double dn = std::hypot(df, da);
          if (dn > 1e-6) {
            df /= dn;
            da /= dn;
            // Limit region: each side its own semi-axis (hard limit + margin), so a convex quarter-ellipse patchwork.
            const double fhi = b.js[j0].hi + kHipMargin, flo = -b.js[j0].lo + kHipMargin;
            const double ahi = b.js[j0 + 1].hi + kHipMargin, alo = -b.js[j0 + 1].lo + kHipMargin;
            const double qf = P.q[j0], qa = P.q[j0 + 1];
            auto g = [&](double k) {  // > 0 outside
              const double xf = qf + k * df, xa = qa + k * da;
              const double nf = xf / (xf >= 0 ? fhi : flo), na = xa / (xa >= 0 ? ahi : alo);
              return nf * nf + na * na - 1;
            };
            double res;
            const double g0 = g(0);
            if (g0 > 0) {
              res = -(std::sqrt(g0 + 1) - 1);  // already past the edge: below the floor
            } else {  // the ray from inside leaves the convex region once: bisection
              double klo = 0, khi = 4.0;
              for (int it = 0; it < 24; ++it) {
                const double km = 0.5 * (klo + khi);
                if (g(km) > 0) khi = km; else klo = km;
              }
              res = klo;
            }
            b.dbg_lres = res;
            lift = std::fmin(lift, leg * res);
          }
        }
        lift *= lift > 0 ? Clamp(1.0 - b.sw_stall / kStall, 0.0, 1.0) : 1.0;  // not getting closer: put the foot down
        // lift_v: the height target rises from the lift-off at the foot's own speed; ordered at once, the full height
        // would jump the knee target by about a radian in one frame (a huge knee acceleration).
        if (pr[P_LIFT_V] > 0 && lift > 0) lift = std::fmin(lift, pr[P_LIFT_V] * hs / ts * (t - b.sw_t0b + dt));
        if (b.sh_foot >= 0 && f != b.sh_foot) lift = b.sh_state == 1 ? kShiftLift * hs : 0.0;
        // Aim past the place, clip by reach, then the carrot from the foot.
        double aim[2] = {tg[0], tg[1]};
        if (dsw > 1e-4 && !(b.sh_foot >= 0 && f != b.sh_foot)) {
          const double ov = std::fmin(pr[P_OVER] * hs, dsw);
          aim[0] += dx / dsw * ov;
          aim[1] += dy / dsw * ov;
          if (!(b.cmd_on && b.cmd_started == 1 && f == b.cmd_foot)) reach(f, aim);
        }
        // Carrot (circle). An ellipse shorter across by hip abduction / flexion strength would cut the side overshoot
        // but make side steps too slow.
        const double ax = aim[0] - S[f][0], ay = aim[1] - S[f][1], ad = std::hypot(ax, ay);
        // rlx_car: while relaxing on the move the carrot also covers the body's own travel toward the aim over the
        // swing response time, so the foot can get ahead of a body that moves faster than carrot / response (0.165 m
        // in 0.083 s ~ 2 m/s); otherwise the swing foot stays in front of a body falling back and both feet unload.
        double cl = pr[P_CARROT] * hs;
        if (rlx && pr[P_RLX_CAR] > 0 && ad > 1e-9)
          cl += pr[P_RLX_CAR] * std::fmax((v[0] * ax + v[1] * ay) / ad, 0.0) * (pr[P_DAMP_RATIO] * zsw + dt);
        double kc = ad > 1e-9 ? std::fmin(1.0, cl / ad) : 0.0;
        double ankz = ah + lift, lm = lift;  // frame target ankle height; lift measure for the angle overshoot
        if (pr[P_CAR3D] > 0) {
          // 3.9: the carrot runs in space from the foot to the frame target with its height, so the lift
          // and the descent are paced by it too; length = min(left to the target, response time x swing speed) + a
          // small constant (the foot moves at the swing speed and slows by itself near the target). Lift measure: how
          // far the target is above where the foot will be one response time on; the higher, the shorter the carrot
          // and the more the angles are pulled past the target (the leg lifts by bending first, not by reaching).
          const double Tr = pr[P_DAMP_RATIO] * zsw + dt;
          const double zF = P.ankle[f][2], fvz = (P.ankle[f][2] - b.ank_prev[f][2]) / dt;
          lm = std::fmax(ankz - (zF + fvz * Tr), 0.0);
          const double dz = ankz - zF, dist = std::sqrt(ad * ad + dz * dz);
          double L = std::fmin(dist, Tr * pr[P_SW_V] * ts) + pr[P_CAR_C] * hs;
          L /= 1.0 + lm / std::fmax(hstep, 1e-3);
          const double k3 = dist > 1e-9 ? std::fmin(1.0, L / dist) : 0.0;
          kc = k3;
          ankz = zF + dz * k3;
        }
        const double fyd = yaw_t + s * pr[P_TOE_OUT];  // swing foot heading
        const double sole[2] = {S[f][0] + ax * kc, S[f][1] + ay * kc};
        const double ank[3] = {sole[0] - fx * std::cos(fyd), sole[1] - fx * std::sin(fyd), ankz};
        b.sw_ft[0] = ank[0];
        b.sw_ft[1] = ank[1];
        b.sw_ft[2] = ank[2];
        b.sw_tg[0] = tg[0];
        b.sw_tg[1] = tg[1];
        b.step_tgt[0] = tg[0] - fx * std::cos(fyd);
        b.step_tgt[1] = tg[1] - fx * std::sin(fyd);
        // Preview (3.9): the joints reach the targets one response time later; by then the hip has moved and the
        // pelvis has turned. Default (swing_pv 0): the hip moves with the CoM velocity, no rotation. swing_pv 1: hip
        // velocity and the pelvis frame predicted for t + Ts (Rot(w Ts)); without an upper-body yaw counter-motion the
        // swing reaction feeds the pelvis turn and the prediction chases it, so it stays a switch.
        const double Ts = pr[P_DAMP_RATIO] * zsw + dt;
        // 3.8: the leg reaches its place from the lowered (desired) pelvis and so lands bent (sw_hd 1); from the
        // present hip the leg would straighten to the floor fast before landing while the pelvis stays up.
        // Only down and only the height of the hip midpoint (the present roll kept): the desired roll would move the
        // swing hip sideways and spoil side steps.
        // rlx_swhd: while relaxing on the move the swing leg reaches from the hips lowered by this share of the desired
        // drop: from the present hips (0) it lands straight and far off and the body vaults over it; from the desired
        // height (1) the foot hangs until the pelvis falls and lands late on a fully bent knee.
        const double swhd = rlx && pr[P_RLX_SWHD] > 0
                                ? (double)pr[P_RLX_SWHD] * (pr[P_RLX_LDIR] > 1.5 ? rlx_wb : 1.0)
                                : (double)pr[P_SW_HD];
        const double hipz = P.hip[f][2] + swhd * std::fmin(Hd[2] - hm[2], 0.0);
        const double kv = pr[P_SWING_PV];
        const double hv[3] = {kv * P.hip_vel[f][0] + (1 - kv) * v[0], kv * P.hip_vel[f][1] + (1 - kv) * v[1],
                              kv * P.hip_vel[f][2]};
        // pv_far: when the place lies beyond reach (strong push) and the foot is far from it, the carrot is taken
        // relative to the present hip, so the foot moves at body speed plus its own; with the full preview the hip's
        // travel v Ts (0.16 m at 2 m/s) cancels the 0.165 m carrot and the foot only keeps up with the body.
        // Weights: the velocity blend's (two carrots from the place) x the clipped part (one carrot). Only beyond
        // reach: for reachable places it makes the foot overshoot.
        const bool own = (b.cmd_on && b.cmd_started == 1 && f == b.cmd_foot) || (b.sh_foot >= 0 && f != b.sh_foot);
        const double cw = pr[P_CARROT] * hs;
        const double clip = own ? 0.0 : std::hypot(b.place[f][0] - tg[0], b.place[f][1] - tg[1]);
        const double kf = 1.0 - pr[P_PV_FAR] * Clamp((dsw - cw) / cw, 0.0, 1.0) * Clamp(clip / cw, 0.0, 1.0);
        const double rw[3] = {ank[0] - (P.hip[f][0] + kf * hv[0] * Ts), ank[1] - (P.hip[f][1] + kf * hv[1] * Ts),
                              ank[2] - (hipz + kf * hv[2] * Ts)};
        // swing_pvr: the pelvis rotation preview alone (hip at the CoM velocity). With the standing foot planted
        // (strut) the pelvis turns during the swing, and without the preview the swing foot follows the turn and lands
        // off to the side.
        const double kr = std::fmax(kv, pr[P_SWING_PVR]);
        const double th[3] = {-kr * P.pelvis_w[0] * Ts, -kr * P.pelvis_w[1] * Ts, -kr * P.pelvis_w[2] * Ts};
        double rp[3], upw[3], r[3], up_p[3];
        RotVec(th, rw, rp);  // world vector in the pelvis frame predicted for t + Ts = Ra^T Rot(-w Ts) x
        RotVec(th, zw, upw);
        MulTV(Ra, rp, r);
        MulTV(Ra, upw, up_p);
        LegIK6(r, s * Wrap(yaw_t - P.pelvis_yaw + th[2]) + pr[P_TOE_OUT], s, thigh, shank, &a, 0.995, up_p);
        const double ovr = pr[P_SWING_OVER] * Clamp(lm / hstep, 0.0, 1.0);  // angles pulled past the target
        a.flex += ovr * (a.flex - P.q[j0]);
        a.knee += ovr * (a.knee - P.q[j0 + 3]);
        const double back = dsw > 1e-4 ? std::fmax(-(dx * hx + dy * hy) / dsw, 0.0) : 0.0;
        a.dorsi -= pr[P_TOE_LAND] * back;  // backward step: toe down
        // Target velocity from the place: the joint angles that put the foot on the place at the present lift,
        // differenced over the frame. The frame target (carrot) moves with the foot; fed forward, its velocity would
        // cancel the damping and the foot would pass the place fast. So the damping acts against the foot velocity
        // relative to the place (a damped spring toward the target).
        {
          const double pw[3] = {b.step_tgt[0] - (P.hip[f][0] + hv[0] * Ts), b.step_tgt[1] - (P.hip[f][1] + hv[1] * Ts),
                                ah + lift - (hipz + hv[2] * Ts)};
          double pq[3], pl[3];
          RotVec(th, pw, pq);
          MulTV(Ra, pq, pl);
          Leg6 ap;
          LegIK6(pl, s * Wrap(yaw_t - P.pelvis_yaw + th[2]) + pr[P_TOE_OUT], s, thigh, shank, &ap, 0.995, up_p);
          ap.dorsi -= pr[P_TOE_LAND] * back;
          SetLeg(Tp, f, ap);
        }
        kleg[f] = pr[P_SWING_K];
        zleg[f] = zsw;
        ff_leg = f;
        b.stand_leg[f] = 0;
        if (!on[f]) b.single = o;
        b.phase = on[f] ? 0 : 1;
      }
      SetLeg(T, f, a);
    }
    b.dbg_raw[0] = pt[0];
    b.dbg_raw[1] = pt[1];
    b.dbg_raw[2] = out;
    b.dbg_raw[3] = dsw;
    dsw_f = dsw;

    // Stay-upright help: a pair force pelvis <-> loaded feet, sum zero. The pelvis is pulled toward the point
    // over the support middle using the capture point (CoM + v / omega); gain = assist * M omega^2 (assist 1 = a CoP
    // shift equal to the error). Only with a loaded foot and not fallen; fades linearly with the body tilt
    // over the support to zero at 45 deg; horizontal only; |F| <= 0.25 M g (friction margin).
    // asst_split: pulled toward the middle of both feet, the reaction half on each foot centre (standing or in the
    // air), on while any foot touches; the pair has almost no yaw moment (with the reaction on the one standing foot
    // beside the pelvis it would twist the body under the bullet pull)
    // asst_bold: under a live bullet reaction the loaded-feet rule instead of the split
    // asst_bold 2: the loaded-feet rule only while no swing is active; during a swing the split's push on the swing
    // foot drives the backward step (bullet bursts need many quick back steps)
    const bool bold = pr[P_ASST_BOLD] > 0 && b.hit_bullet && t < b.bul_until && (pr[P_ASST_BOLD] < 1.5 || b.swing < 0);
    const bool asplit = pr[P_ASST_SPLIT] > 0 && b.sh_foot < 0 && !bold;
    if (b.mode != 2 && pr[P_ASSIST] > 0 && (asplit ? (on[0] || on[1]) : (b.loaded[0] || b.loaded[1]))) {
      double sx = 0, sy = 0, sw = 0;
      if (asplit) {
        sx = P.foot_c[0][0] + P.foot_c[1][0];
        sy = P.foot_c[0][1] + P.foot_c[1][1];
        sw = 2;
      } else
      for (int f = 0; f < 2; ++f)
        // asst_on: a standing foot counts while it touches, not only while loaded: in a small side push the far foot
        // unloads while still down; with loaded feet only the middle would jump to the near foot, the capture point
        // would lie past it and the help would stop braking.
        if ((b.loaded[f] || (pr[P_ASST_ON] > 0 && on[f] && f != b.swing)) && (b.sh_foot < 0 || f == b.sh_foot)) {
          sx += P.foot_c[f][0];
          sy += P.foot_c[f][1];
          sw += 1;
        }
      if (sw < 1) { sx = P.foot_c[b.sh_foot][0]; sy = P.foot_c[b.sh_foot][1]; sw = 1; }
      sx /= sw; sy /= sw;
      sx -= dph * b.pull_dir[0];  // pull_lean: the rest point under the pull is in front of the support (against it)
      sy -= dph * b.pull_dir[1];
      const double M = b.dims.total_mass;
      const double tilt = std::atan2(std::hypot(P.com[0] - sx, P.com[1] - sy), std::fmax(P.com[2], 0.1));
      // asst_ref: the pulled point is CoM + v asst_t (velocity lead time and the CoM velocity damping together) and
      // the help fades as cos(tilt)^asst_n (8: 1 upright, 0.32 at 30 deg).
      const bool rf = pr[P_ASST_REF] > 0;
      const double fade = rf ? std::pow(std::cos(std::fmin(tilt, 1.5)), (double)pr[P_ASST_N])
                             : Clamp(1.0 - tilt * (1.0 / 0.785), 0.0, 1.0);
      const double k = pr[P_ASSIST] * M * omega * omega * fade;
      const double qx = rf ? P.com[0] + P.com_vel[0] * pr[P_ASST_T] : xi[0];
      const double qy = rf ? P.com[1] + P.com_vel[1] * pr[P_ASST_T] : xi[1];
      double ehx = qx - sx, ehy = qy - sy;
      // bul_center: while a bullet reaction is live the position part of the error (CoM from the support middle)
      // counts this share, the velocity part and the pull offset in full: the help brakes fully but re-centres
      // slowly (with tau 1 / omega = 0.3 s it would run the CoM to the middle after each catch and never let it rest)
      if (kxc < 1.0) {  // fore-aft only, like the number (sideways a slow return turns the body)
        const double mx = sx + dph * b.pull_dir[0], my = sy + dph * b.pull_dir[1];
        const double ef = (1.0 - kxc) * ((P.com[0] - mx) * hx + (P.com[1] - my) * hy);
        ehx -= ef * hx;
        ehy -= ef * hy;
      }
      double fxa = -k * ehx, fya = -k * ehy;
      const double fm = std::hypot(fxa, fya), lim = 0.25 * M * 9.81;
      if (fm > lim) { fxa *= lim / fm; fya *= lim / fm; }
      // pull_free: while the bullet pull acts the help keeps its part across the pull (stay upright) but drops this
      // share of its part against it: in full it would cancel the pull within a fraction of a second and the hit
      // would not move the body
      // pull_free_v: the share fades to zero as the CoM speed along the pull nears pull_free_v: yielding at walking
      // pace, not running away (without a cap a bullet series speeds the body up bullet after bullet)
      if (pr[P_PULL_FREE] > 0 && t < b.pull_until && b.pull_k > 0) {
        const double a = fxa * b.pull_dir[0] + fya * b.pull_dir[1];
        const double vp = v[0] * b.pull_dir[0] + v[1] * b.pull_dir[1];
        // The cap falls from pull_free_v at health kPullFreeH1 (2nd pistol bullet) to kPullFreeV0 at kPullFreeH0
        // (4th): a wounded body that will go down does not run into its fall faster
        const double vcap = kPullFreeV0 + (pr[P_PULL_FREE_V] - kPullFreeV0) *
                                              Clamp((b.health - kPullFreeH0) / (kPullFreeH1 - kPullFreeH0), 0.0, 1.0);
        // A leg-wounded body yields to the pull only as much as its weaker leg can step (a weak leg cannot catch a
        // body driven back fast)
        const double lw = std::fmin(in.leg_weak[0], in.leg_weak[1]);
        const double sh = lw * (pr[P_PULL_FREE_V] > 0 ? pr[P_PULL_FREE] * Clamp((vcap - vp) / kPullFreeBand, 0.0, 1.0)
                                               : (double)pr[P_PULL_FREE]);
        if (a < 0) {
          fxa -= sh * a * b.pull_dir[0];
          fya -= sh * a * b.pull_dir[1];
        }
      }
      b.assist_f[0] = fxa;
      b.assist_f[1] = fya;
      // A throw keeps shoving off (thr_po x the push-off) until the hit or the dodge ends
      const bool po_thr = b.thr_on && pr[P_THR_PO] > 0;
      if (vg_n > 1e-6 && pr[P_PO_F] > 0 && (t - b.dodge_t0 < pr[P_PO_T] * ts || po_thr))
        VgPushOff(b.assist_f, vg_g, vg_n, (v_true[0] * vg_g[0] + v_true[1] * vg_g[1]) * (po_thr ? 1.0 : 0.0) +
                                              (po_thr ? 0.0 : (v[0] * vg_g[0] + v[1] * vg_g[1])),
                  pr[P_PO_F] * in.vpush * M * 9.81 * (po_thr ? (double)pr[P_THR_PO] : 1.0));
      if (asplit) {
        b.assist_w[0] = b.assist_w[1] = 0.5;
      } else {
        const double l0 = b.loaded[0] ? std::fmax(b.ld[0], 0.05) : 0.0, l1 = b.loaded[1] ? std::fmax(b.ld[1], 0.05) : 0.0;
        b.assist_w[0] = l0 / (l0 + l1);
        b.assist_w[1] = l1 / (l0 + l1);
      }
    }
  }
gave_up:  // 3.2 give-up of this frame joins here (no pose, no steps)

  // 3.10 Limits and stiffness: joint targets inside the joint ranges; standing and swing legs have their own
  // stiffness; the swing leg gets the target velocity (feedforward in L0).
  for (int j = J_HIP_FLEX_L; j < J_COUNT; ++j) T[j] = Clamp(T[j], b.js[j].lo, b.js[j].hi);
  // Knee target smoothing: the raw knee target can jump by tens of degrees in a frame (the desired pelvis moves over
  // the new support, the reach clip pulls a foot in), and a stiff knee would follow with a huge acceleration; a human
  // knee lands bent and straightens slowly. A free knee (swing leg or unloaded foot) follows the raw target
  // time-optimally with acceleration kn_acc, restarted at the knee's angle and speed when it becomes free; a loaded
  // standing knee takes its raw target, straightening at most at kn_vst (soft weight acceptance). A loaded knee is
  // not filtered: it must react at once in strong pushes. Given up (mode 2): raw targets.
  if (b.mode == 2 || (pr[P_KN_ACC] <= 0 && pr[P_KN_VST] <= 0)) {
    b.kf_on = 0;
  } else {
    // A low acceleration limit makes the second step of a strong push late and short; the limit applies to the swing
    // knee and the unloaded standing knee alike.
    const double A = pr[P_KN_ACC];
    for (int f = 0; f < 2; ++f) {
      const int j = f ? J_KNEE_R : J_KNEE_L;
      const int fr = A > 0 && (f == b.swing || !b.loaded[f]);
      if (!b.kf_on) { b.kf_q[f] = T[j]; b.kf_v[f] = 0; b.kf_fr[f] = fr; }
      if (fr) {
        if (!b.kf_fr[f]) { b.kf_q[f] = P.q[j]; b.kf_v[f] = (P.q[j] - b.kq_prev[f]) / dt; }
        const double e = T[j] - b.kf_q[f];
        const double vd = std::copysign(std::fmin(std::fabs(e) / dt, std::sqrt(2.0 * A * std::fabs(e))), e);
        b.kf_v[f] += Clamp(vd - b.kf_v[f], -A * dt, A * dt);
        b.kf_q[f] = Clamp(b.kf_q[f] + b.kf_v[f] * dt, b.js[j].lo, b.js[j].hi);
      } else {
        double x = T[j];
        if (pr[P_KN_VST] > 0 && b.kf_on && f != b.swing && b.loaded[f]) x = std::fmax(x, b.kf_q[f] - pr[P_KN_VST] * dt);
        b.kf_v[f] = (x - b.kf_q[f]) / dt;
        b.kf_q[f] = x;
      }
      b.kf_fr[f] = fr;
      T[j] = b.kf_q[f];
    }
    b.kf_on = 1;
  }
  // Leg muscles by frequency and damping ratio: kp = w^2 I, damping 2 z w I (L0 damping = kp x
  // damp_ratio x zm, so zm = 2 z / (w damp_ratio)). Swing leg: I = the joint's diagonal of the MuJoCo inertia matrix
  // (the leg below the joint); standing leg: the body above the joint about its axis (whole-body inertia about the
  // axis minus the leg below). A fixed table gain gives very different frequencies and damping ratios for the light
  // swing leg and the heavy standing one. A frequency of 0 keeps the table model for that leg state; given up
  // (mode 2): table model.
  double Ic[9], Mb = 0, cb[3];
  const bool wz = b.mode != 2 && (pr[P_LEG_W_ST] > 0 || pr[P_LEG_W_SW] > 0);
  if (wz) BotInertia(m_, d_, b.body0, Ic, &Mb, cb);
  const double ratio = std::fmax(1e-3, (double)pr[P_DAMP_RATIO]);
  {  // hip_pd only for small pushes. A recovery whose CoM speed passed hip_pd_v keeps the plain joint damping until
     // rest: in a strong push the upright trunk sends the pelvis back faster and the one-leg stance foot spins.
    const double sp = std::hypot(P.com_vel[0], P.com_vel[1]);
    // The perceived speed ramps over ~3 frames after a push and the swing starts on the third, so judge it one frame
    // ahead: speed + its rise over the last frame.
    const double sp1 = sp + std::fmax(0.0, sp - b.hpd_sp);
    b.hpd_sp = sp;
    // hpd_bul: after a push the stance legs keep holding the pelvis (the lock would let the stance hip torque fall
    // to nothing while the hip flexes and the trunk tips); bullet hits and very fast impacts keep the lock.
    const bool lk = pr[P_HPD_BUL] <= 0 || b.hit_bullet || sp1 > pr[P_HPD_BUL] * std::sqrt(hs);
    if (pr[P_HIP_PD_V] > 0 && sp1 > pr[P_HIP_PD_V] * std::sqrt(hs) && lk) b.hpd_lock = 1;
    else if (b.swing < 0 && sp < 0.15) b.hpd_lock = 0;
  }
  // leg muscles with three numbers. Frequency w and
  // damping ratio z act on the joint's own response inertia (what a unit torque in the joint accelerates with the
  // other joints free and no contacts, 1 / (M^-1)_jj: the light side - leg below the hip, shank and foot below the
  // knee, foot below the ankle; the same for the standing and the swing leg), taken from the last substep with its
  // flywheel: kp = w^2 I, damping 2 z w I. The third number, muscle stiffness, is the flywheel (MuJoCo armature):
  // stf x inertia below the joint about its axis x (w^2 h^2 + 2 z w h), h = frame step (the inertia an implicit PD over
  // one frame adds; Tan, Liu and Turk 2011). External forces accelerate the joint that much less; the muscle limit
  // grows by the same factor as the joint's inertia below. The weight is carried by the support torques
  // (L0), so the weak spring only guides the pose. Hip axes each with their own inertia (a soft hip twist).
  const bool mus2 = pr[P_ARM] > 0 && b.mode != 2;
  double m2_k[12] = {}, m2_z[12] = {}, m2_a[12] = {}, m2_l[12] = {};
  if (mus2) {
    const double h = World::kFrameDt;
    for (int f = 0; f < 2; ++f) {
      const bool sw = f == b.swing;
      const double w = sw ? pr[P_MW_SW] : pr[P_MW_ST], zz = sw ? pr[P_MZ_SW] : pr[P_MZ_ST];
      const double mult = w * w * h * h + 2 * zz * w * h;
      const bool touch = b.foot_fn[f] > 0.02 * b.dims.total_mass * 9.81;
      for (int jl = 0; jl < 6; ++jl) {
        const int i = 6 * f + jl, dof = b.dof0 + (f ? J_HIP_FLEX_R : J_HIP_FLEX_L) + jl;
        const double a0 = m_->dof_armature[dof];  // flywheel of the last substep (inside M and its factor)
        const double Ib = d_->M[m_->M_rowadr[dof] + m_->M_rownnz[dof] - 1] - a0;
        const double stf = jl <= 2   ? pr[P_STF_HIP]
                           : jl == 3 ? (sw ? pr[P_STF_KSW] : pr[P_STF_KST])
                                     : (touch ? pr[P_STF_AST] : pr[P_STF_AAIR]);
        const double Im = RespInertia(dof);
        m2_k[i] = w * w * Im * stiff * in.leg_weak[f];
        m2_z[i] = 2.0 * zz / (w * ratio);
        m2_a[i] = stf * std::fmax(Ib, 0.0) * mult;
        // Limit growth = the joint's inertia growth from its flywheel (light side, inertia below): 1 + stf x mult. Not
        // from the response inertia: with every other joint free a hip torque mostly spins the light pelvis (its
        // response inertia is several times below the leg's), which would scale the limit far too much.
        m2_l[i] = 1.0 + stf * mult;
      }
    }
  }
  for (int j = 0; j < J_COUNT; ++j) {
    double k = b.js[j].kp * stiff, z = 1.0;
    b.arm[j] = 0;
    b.limf[j] = 1.0;
    if (j >= J_HIP_FLEX_L) {
      const int f = j >= J_HIP_FLEX_R ? 1 : 0, jl = j - (f ? J_HIP_FLEX_R : J_HIP_FLEX_L);
      k *= in.leg_weak[f];
      const bool sw = f == b.swing;
      const double w = sw ? pr[P_LEG_W_SW] : pr[P_LEG_W_ST];
      if (mus2) {
        k = m2_k[6 * f + jl];
        z = m2_z[6 * f + jl];
        b.arm[j] = m2_a[6 * f + jl];
        b.limf[j] = m2_l[6 * f + jl];
      } else if (jl <= 3 && wz && w > 0) {
        const int dof = b.dof0 + j;
        const double Ib = d_->M[m_->M_rowadr[dof] + m_->M_rownnz[dof] - 1];
        const int jh = b.jnt0 + j - jl;  // this leg's hip flexion joint
        const double I = sw        ? Ib
                         : jl == 3 ? KneeStanceInertia(m_, d_, b.body0 + (f ? B_THIGH_R : B_THIGH_L), jh, b.jnt0 + j, Mb)
                                   : std::fmax(AxisInertia(Ic, Mb, cb, d_->xanchor + 3 * (b.jnt0 + j),
                                                           d_->xaxis + 3 * (b.jnt0 + j)) - Ib, Ib);
        k = w * w * I * stiff * in.leg_weak[f];
        z = 2.0 * (sw ? pr[P_LEG_Z_SW] : pr[P_LEG_Z_ST]) / (w * ratio);
      } else if (jl <= 3) {
        k *= kleg[f];
        z = zleg[f];
      }
      // Strut (a torque budget): on one leg the standing hip pulls into flexion with at most strut x its muscle limit
      // x foot load share, so the pelvis lead-time and pitch parts cannot throw the planted leg forward and unload the
      // foot. A budget in torque, not an angle past the joint, holds for any stiffness. Only flexion is limited: the
      // hip must still extend against forward pushes. The one-leg counter-lean (3.7, pitch_acc) comes on top of the
      // budget, so the trunk still leans against the push.
      if (jl == 0 && !sw && b.swing >= 0 && b.mode != 2 && pr[P_STRUT] > 0 && k > 1e-6)
        T[j] = std::fmin(T[j], P.q[j] + pr[P_STRUT] * b.js[j].tau_max * std::fmin(b.ld[f], 1.0) / k +
                                   pr[P_LEAN_TOP] * b.pitch_acc + (1.0 - pr[P_CH_SPINE]) * b.ch_add + b.lh_add);
    }
    // deck_hold: a wide stance pushes the feet out (the floor pushes each foot inward) and the soft contact lets them
    // creep apart until gather steps are needed; while bracing on two loaded feet each hip adds the adduction torque
    // that cancels the measured outward push.
    if (in.wide > 0 && pr[P_DECK_HOLD] > 0 && (j == J_HIP_ABD_L || j == J_HIP_ABD_R) && b.swing < 0 && b.loaded[0] &&
        b.loaded[1] && k > 1e-6) {
      const int f = j == J_HIP_ABD_R ? 1 : 0;
      const double ln = std::hypot(P.pelvis_R[1], P.pelvis_R[4]) + 1e-9;
      const double fin = (f ? 1.0 : -1.0) * (b.foot_fh[f][0] * P.pelvis_R[1] + b.foot_fh[f][1] * P.pelvis_R[4]) / ln;
      T[j] -= Clamp(pr[P_DECK_HOLD] * fin * 0.5 * b.body.height / k, -0.1, 0.1);
    }
    b.kp[j] = k;
    b.zm[j] = z;
    b.qd_tgt[j] = 0;
    // hip_pd 2: only while the body moves back (heading speed below -0.3 m/s): side pushes with the turn need the
    // joint damping to keep the steps short.
    const bool hfp_on = !b.hit_bullet && pr[P_HPD_FAST_PUSH] > 0;  // push hip law
    if (pr[P_HIP_PD] > 0 && (!b.hpd_lock || pr[P_HIP_PD_FAST] > 0 || hfp_on) && b.swing >= 0 && b.mode != 2 &&
        (j == J_HIP_FLEX_L || j == J_HIP_FLEX_R) &&
        (pr[P_HIP_PD] < 1.5 || (P.com_vel[0] * P.pelvis_R[0] + P.com_vel[1] * P.pelvis_R[3]) <
                                   -0.3 * std::hypot(P.pelvis_R[0], P.pelvis_R[3]))) {
      // hip_pd: while the body falls back over the planted foot the leg turns under the pelvis, and the joint damping
      // (kp x damp_ratio x zm per rad/s) would pull the pelvis along and tip the trunk back instead of letting it bow
      // forward. The target velocity is the leg's own turn (joint rate minus pelvis pitch rate), so the damping acts
      // on the pelvis pitch rate only.
      const int f = j == J_HIP_FLEX_R ? 1 : 0;
      // hpd_ld: only a loaded leg holds the pelvis; a leg that touched and still swings back keeps the joint damping
      // that stops it (otherwise the landing foot slides on behind the CoM)
      // hpd_td: the leg that touched down last keeps the joint damping for hpd_td s: its hip brakes the leg turning
      // back under the body, so the landing foot stays put instead of sliding
      const bool fresh = pr[P_HPD_TD] > 0 && f == b.td_foot && t - b.td_time < pr[P_HPD_TD];
      if (f != b.swing && !fresh && (pr[P_HPD_LD] <= 0 || b.loaded[f])) {
        const double* R = P.pelvis_R;
        const double pr_w = P.pelvis_w[0] * R[1] + P.pelvis_w[1] * R[4] + P.pelvis_w[2] * R[7];  // + = top forward
        // hip_pd_fast: after a fast push (latched) only this share of the leg's own turn is let through: the
        // three-number stance hip (large response inertia with its flywheel, strong damping) would drag the pelvis
        // back with the planted leg.
        // hpd_fast_push: after a push half the leg's own turn is let through: the joint damping that turns the trunk
        // back with the planted leg is halved, so the chest stays nearer upright.
        const double hf0 = pr[P_HIP_PD_FAST];
        const double hfp = !b.hit_bullet && pr[P_HPD_FAST_PUSH] >= 0 ? hf0 + b.ch_w * (pr[P_HPD_FAST_PUSH] - hf0) : hf0;
        b.qd_tgt[j] = (b.hpd_lock ? hfp : 1.0) *
                      Clamp((P.q[j] - b.hq_prev[f]) / dt - pr_w, -10.0, 10.0);
      }
    }
  }
  if (ff_leg >= 0) {
    const int j0 = ff_leg == 0 ? J_HIP_FLEX_L : J_HIP_FLEX_R;
    const double fm = pr[P_FF_MAX];
    // Far from the place the target velocity is the frame target's (fast cruise: the place recedes with the body, and
    // a place-only velocity would brake the foot too early); within the carrot it is the place's (damped approach);
    // linear between one and two carrot lengths.
    const double c = pr[P_CARROT] * hs, w = Clamp((dsw_f - c) / c, 0.0, 1.0);
    for (int j = j0; j < j0 + 6; ++j) {
      Tp[j] = Clamp(Tp[j], b.js[j].lo, b.js[j].hi);
      const double qp = b.tp_leg == ff_leg ? (Tp[j] - b.tp_prev[j]) / dt : 0.0, qc = (T[j] - b.tq_prev[j]) / dt;
      b.qd_tgt[j] = Clamp(w * qc + (1 - w) * qp, -fm, fm);
      b.tp_prev[j] = Tp[j];
    }
    b.tp_leg = ff_leg;
  } else {
    b.tp_leg = -1;
  }
  // Knees of the swing leg and of an unloaded foot: the follower's own speed is the target velocity (the swing's
  // place-based part is not smoothed; with no feed an unloaded standing knee brakes the lift before the swing and the
  // feed's start at the swing jerks the knee). Loaded knees: none (fed, the double support chatters).
  if (b.kf_on)
    for (int f = 0; f < 2; ++f)
      if (b.kf_fr[f]) b.qd_tgt[f ? J_KNEE_R : J_KNEE_L] = Clamp(b.kf_v[f], -15.0, 15.0);
  for (int f = 0; f < 2; ++f)
    for (int k = 0; k < 3; ++k) b.ank_prev[f][k] = P.ankle[f][k];
  b.kq_prev[0] = P.q[J_KNEE_L];
  b.kq_prev[1] = P.q[J_KNEE_R];
  b.hq_prev[0] = P.q[J_HIP_FLEX_L];
  b.hq_prev[1] = P.q[J_HIP_FLEX_R];
  // rlx_ff: while relaxing on the move a standing knee gets its target's flexion rate (at most rlx_ff rad/s) as the
  // target velocity, so the joint damping (standing knee kd / kp ~0.1 s) does not brake the sink toward the lower
  // target. Off by default: with it the pelvis follows the crouch from the first frame instead of holding a moment.
  if (b.dead_t >= 0 && !b.in.no_step && pr[P_RLX_FF] > 0)
    for (int f = 0; f < 2; ++f) {
      const int j = f ? J_KNEE_R : J_KNEE_L;
      if (f != b.swing) b.qd_tgt[j] = Clamp((T[j] - b.tq_prev[j]) / dt, 0.0, (double)pr[P_RLX_FF]);
    }
  for (int j = 0; j < J_COUNT; ++j) b.tq_prev[j] = T[j];
  LegContacts(b, b.mode != 2);
}

}  // namespace emil
