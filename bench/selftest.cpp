// Foundations self-test: joint axis signs and limits, hip and leg kinematics vs the MuJoCo model,
// virtual CoM force sign, pelvis yaw torque sign. Test-only holds (pelvis kept in place) live here, never in core.
#include <cmath>
#include <cstdio>
#include <random>

#include "emil_world.h"
#include "l1_leg_ik.h"

using namespace emil;

namespace {
int g_fail = 0;
void Row(const char* name, bool ok, const char* fmt, double a = 0, double b = 0) {
  char msg[160];
  snprintf(msg, sizeof msg, fmt, a, b);
  printf("%-34s %-6s %s\n", name, ok ? "ok" : "ERROR", msg);
  if (!ok) ++g_fail;
}

bool Make(World& w) {
  char err[256];
  return w.Build({BodyParams()}, 2.0, 1, err, sizeof err);
}

// Hip anchor and pelvis rotation (world).
void PelvisFrame(World& w, double hip[3], const double** R, int side) {
  Bot& b = w.bot(0);
  const double* a = w.data()->xanchor + 3 * (b.jnt0 + (side ? J_HIP_FLEX_R : J_HIP_FLEX_L));
  for (int i = 0; i < 3; ++i) hip[i] = a[i];
  *R = w.data()->xmat + 9 * b.body0;
}
void ToPelvis(const double* R, const double v[3], double o[3]) {  // R^T v
  for (int i = 0; i < 3; ++i) o[i] = R[i] * v[0] + R[3 + i] * v[1] + R[6 + i] * v[2];
}

void TestAxes() {
  struct E { int j; int body; double dx, dy, dz; int side_y; int pre; };  // dy scaled by side if side_y
  // pre: 1 = elbow 90 deg before the test (shoulder rotation)
  // Expected world rotation axis of the child for a positive angle (anatomy, standing pose); x fwd, y left, z up.
  // flexion of trunk/neck: top forward (+y); lean right (+x); turn left (+z); limb flexion forward (-y);
  // abduction out (side * x); external rotation (side * z); knee flexion: shank back (+y); dorsiflexion: toes up (-y);
  // inversion: sole turns medially (-side * x). dy is multiplied by side for limbs where side_y = 1 (dx for abd/inv).
  const E t[] = {
      {J_LUMBAR_PITCH, B_HEAD, 0, 1, 0, 0, 0}, {J_LUMBAR_ROLL, B_HEAD, 1, 0, 0, 0, 0},
      {J_LUMBAR_YAW, B_UARM_L, 0, 0, 1, 0, 0}, {J_CHEST_PITCH, B_HEAD, 0, 1, 0, 0, 0},
      {J_CHEST_ROLL, B_HEAD, 1, 0, 0, 0, 0},   {J_CHEST_YAW, B_UARM_L, 0, 0, 1, 0, 0},
      {J_NECK_PITCH, B_HEAD, 0, 1, 0, 0, 0},   {J_NECK_ROLL, B_HEAD, 1, 0, 0, 0, 0},
      {J_SH_FLEX_L, B_HAND_L, 0, -1, 0, 0, 0}, {J_SH_ABD_L, B_HAND_L, 1, 0, 0, 1, 0},
      {J_SH_ROT_L, B_HAND_L, 0, 0, 1, 1, 1},   {J_ELBOW_L, B_HAND_L, 0, -1, 0, 0, 0},
      {J_WRIST_L, B_HAND_L, 0, -1, 0, 0, 0},   {J_SH_FLEX_R, B_HAND_R, 0, -1, 0, 0, 0},
      {J_SH_ABD_R, B_HAND_R, 1, 0, 0, 1, 0},   {J_SH_ROT_R, B_HAND_R, 0, 0, 1, 1, 1},
      {J_ELBOW_R, B_HAND_R, 0, -1, 0, 0, 0},   {J_WRIST_R, B_HAND_R, 0, -1, 0, 0, 0},
      {J_HIP_FLEX_L, B_FOOT_L, 0, -1, 0, 0, 0}, {J_HIP_ABD_L, B_FOOT_L, 1, 0, 0, 1, 0},
      {J_HIP_ROT_L, B_FOOT_L, 0, 0, 1, 1, 0},  {J_KNEE_L, B_FOOT_L, 0, 1, 0, 0, 0},
      {J_ANKLE_DORSI_L, B_FOOT_L, 0, -1, 0, 0, 0}, {J_ANKLE_INV_L, B_FOOT_L, -1, 0, 0, 1, 0},
      {J_HIP_FLEX_R, B_FOOT_R, 0, -1, 0, 0, 0}, {J_HIP_ABD_R, B_FOOT_R, 1, 0, 0, 1, 0},
      {J_HIP_ROT_R, B_FOOT_R, 0, 0, 1, 1, 0},  {J_KNEE_R, B_FOOT_R, 0, 1, 0, 0, 0},
      {J_ANKLE_DORSI_R, B_FOOT_R, 0, -1, 0, 0, 0}, {J_ANKLE_INV_R, B_FOOT_R, -1, 0, 0, 1, 0}};
  const char* names[] = {"lumbar_pitch +flex fwd", "lumbar_roll +lean right", "lumbar_yaw +turn left",
                         "chest_pitch +flex fwd", "chest_roll +lean right", "chest_yaw +turn left",
                         "neck_pitch +flex fwd", "neck_roll +lean right", "sh_flex_l +fwd", "sh_abd_l +out",
                         "sh_rot_l +external", "elbow_l +flex", "wrist_l +flex", "sh_flex_r +fwd", "sh_abd_r +out",
                         "sh_rot_r +external", "elbow_r +flex", "wrist_r +flex", "hip_flex_l +fwd",
                         "hip_abd_l +out", "hip_rot_l +external", "knee_l +flex", "ankle_dorsi_l +toes up",
                         "ankle_inv_l +sole in", "hip_flex_r +fwd", "hip_abd_r +out", "hip_rot_r +external",
                         "knee_r +flex", "ankle_dorsi_r +toes up", "ankle_inv_r +sole in"};
  int n = 0;
  for (const E& e : t) {
    World w;
    if (!Make(w)) return;
    mjModel* m = const_cast<mjModel*>(w.model());
    mjData* d = w.data();
    Bot& b = w.bot(0);
    m->opt.gravity[2] = 0;
    const bool left = e.j < J_SH_FLEX_R || (e.j >= J_HIP_FLEX_L && e.j < J_HIP_FLEX_R);
    const double sy = e.side_y ? (left ? 1.0 : -1.0) : 1.0;
    if (e.pre) d->qpos[b.qadr0 + e.j + 3] = 1.5708;  // elbow of the same arm
    d->qpos[b.free_qadr + 2] += 0.5;  // held in the air: no floor contact
    mj_forward(m, d);
    double qs[J_COUNT];
    for (int k = 0; k < J_COUNT; ++k) qs[k] = d->qpos[b.qadr0 + k];
    for (int k = 0; k < J_COUNT; ++k) m->dof_damping[b.dof0 + k] = 0.5;
    double q0[7];
    for (int i = 0; i < 7; ++i) q0[i] = d->qpos[b.free_qadr + i];
    double x0[3], p0[3];
    for (int i = 0; i < 3; ++i) x0[i] = d->xipos[3 * (b.body0 + e.body) + i];
    for (int i = 0; i < 3; ++i) p0[i] = d->xanchor[3 * (b.jnt0 + e.j) + i];
    const double tau = 0.2 * b.js[e.j].tau_max;  // small: light distal bodies must not reach the limit
    double qmax = 0;
    for (int s = 0; s < 480; ++s) {  // 1 s: first 40 substeps for the direction, all for the limit
      // Test hold: every other joint is locked by a stiff PD at its start angle (only the tested joint moves).
      for (int k = 0; k < J_COUNT; ++k) {
        const double lim = b.js[k].tau_max;
        double u = 2000.0 * (qs[k] - d->qpos[b.qadr0 + k]) - 40.0 * d->qvel[b.dof0 + k];
        d->ctrl[b.act0 + k] = u > lim ? lim : (u < -lim ? -lim : u);
      }
      d->ctrl[b.act0 + e.j] = s < 10 ? tau : 0.6 * b.js[e.j].tau_max;
      mj_step(m, d);
      for (int i = 0; i < 7; ++i) d->qpos[b.free_qadr + i] = q0[i];  // test hold: pelvis fixed
      for (int i = 0; i < 6; ++i) d->qvel[b.free_dof + i] = 0;
      if (s == 9) {
        mj_kinematics(m, d);
        // (1) +torque increases the angle; (2) world joint axis equals the anatomical axis from the table.
        const double ax[3] = {sy * e.dx, e.dy, sy * e.dz};
        const double* xa = d->xaxis + 3 * (b.jnt0 + e.j);
        const double c = ax[0] * xa[0] + ax[1] * xa[1] + ax[2] * xa[2];
        const double dq = d->qpos[b.qadr0 + e.j] - qs[e.j];
        char nm[64];
        snprintf(nm, sizeof nm, "axis %s", names[n]);
        Row(nm, c > 0.95 && dq > 0, "cos(axis, anatomical)=%.2f, dq %+.2f deg under +torque", c, dq * 57.3);
      }
      qmax = std::fmax(qmax, d->qpos[b.qadr0 + e.j]);
    }
    const double hi = m->jnt_range[2 * (b.jnt0 + e.j) + 1], lo = m->jnt_range[2 * (b.jnt0 + e.j)];
    char nm[64];
    snprintf(nm, sizeof nm, "limit %s", names[n]);
    Row(nm, qmax > 0.6 * hi && qmax < hi + 5 / 57.3, "q under 0.6 tau_max %.1f deg, range hi %.0f (soft limit)", qmax * 57.3, hi * 57.3);
    (void)lo;
    ++n;
  }
}

// Old decomposition (frontal first), kept to show the before/after error.
void OldHip(const double d[3], double side, double* flex, double* abd) {
  *abd = std::atan2(side * d[1], -d[2]);
  *flex = std::atan2(d[0], std::sqrt(d[1] * d[1] + d[2] * d[2]));
}

void TestKinematics() {
  World w;
  if (!Make(w)) return;
  mjModel* m = const_cast<mjModel*>(w.model());
  mjData* d = w.data();
  Bot& b = w.bot(0);
  const double thigh = b.dims.thigh, shank = b.dims.shank;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(0, 1);
  auto set_leg = [&](int side, const Leg6& a) {
    const int j0 = side ? J_HIP_FLEX_R : J_HIP_FLEX_L;
    const double v[6] = {a.flex, a.abd, a.rot, a.knee, a.dorsi, a.inv};
    for (int k = 0; k < 6; ++k) d->qpos[b.qadr0 + j0 + k] = v[k];
  };
  auto model_leg = [&](int side, double knee[3], double ankle[3], double* tilt) {
    mj_kinematics(m, d);
    double hip[3];
    const double* R;
    PelvisFrame(w, hip, &R, side);
    const int j0 = side ? J_HIP_FLEX_R : J_HIP_FLEX_L;
    double v[3];
    for (int i = 0; i < 3; ++i) v[i] = d->xanchor[3 * (b.jnt0 + j0 + 3) + i] - hip[i];
    ToPelvis(R, v, knee);
    for (int i = 0; i < 3; ++i) v[i] = d->xanchor[3 * (b.jnt0 + j0 + 4) + i] - hip[i];
    ToPelvis(R, v, ankle);
    const double* F = d->xmat + 9 * (b.body0 + (side ? B_FOOT_R : B_FOOT_L));
    const double fz[3] = {F[2], F[5], F[8]};
    double fzp[3];
    ToPelvis(R, fz, fzp);
    *tilt = std::acos(std::fmin(1.0, fzp[2])) * 57.29578;
  };
  // 1) FK formula vs model, random angles within the limits.
  double e_fk = 0, e_hip = 0, e_hip_old = 0, e_ik = 0, e_ik_old = 0, tilt_max = 0;
  for (int it = 0; it < 400; ++it) {
    const int side = it & 1;
    const double sd = side ? -1.0 : 1.0;
    Leg6 a;
    a.flex = (-20 + 110 * U(rng)) / 57.29578;  // -20..90
    a.abd = (-30 + 60 * U(rng)) / 57.29578;    // -30..30
    a.rot = (-30 + 60 * U(rng)) / 57.29578;
    a.knee = (12 + 108 * U(rng)) / 57.29578;  // reachable shell: IK clamps at 0.995 leg length (knee ~11.5 deg)
    a.dorsi = (-30 + 45 * U(rng)) / 57.29578;
    a.inv = (-10 + 30 * U(rng)) / 57.29578;
    mj_resetData(m, d);
    set_leg(side, a);
    double kn_m[3], an_m[3], tilt, kn_f[3], an_f[3];
    model_leg(side, kn_m, an_m, &tilt);
    LegFK(a, sd, thigh, shank, kn_f, an_f);
    for (int i = 0; i < 3; ++i) e_fk = std::fmax(e_fk, std::fabs(an_m[i] - an_f[i]));
    // 2) Hip: thigh direction -> angles -> model knee.
    const double L = std::sqrt(kn_f[0] * kn_f[0] + kn_f[1] * kn_f[1] + kn_f[2] * kn_f[2]);
    const double dir[3] = {kn_f[0] / L, kn_f[1] / L, kn_f[2] / L};
    for (int pass = 0; pass < 2; ++pass) {
      Leg6 h = a;
      h.knee = 0;
      if (pass == 0) HipFromDir(dir, a.rot, sd, &h.flex, &h.abd); else OldHip(dir, sd, &h.flex, &h.abd);
      mj_resetData(m, d);
      set_leg(side, h);
      double k2[3], a2[3], t2;
      model_leg(side, k2, a2, &t2);
      double err = 0;
      for (int i = 0; i < 3; ++i) err = std::fmax(err, std::fabs(k2[i] - kn_f[i]));
      if (pass == 0) e_hip = std::fmax(e_hip, err); else e_hip_old = std::fmax(e_hip_old, err);
    }
    // 3) Leg IK: ankle target -> angles (same twist) -> model ankle, sole tilt.
    Leg6 s;
    LegIK6(an_f, a.rot, sd, thigh, shank, &s);
    mj_resetData(m, d);
    set_leg(side, s);
    double k3[3], a3[3], t3;
    model_leg(side, k3, a3, &t3);
    for (int i = 0; i < 3; ++i) e_ik = std::fmax(e_ik, std::fabs(a3[i] - an_f[i]));
    tilt_max = std::fmax(tilt_max, t3);
    // Old IK form (frontal first, no twist) for comparison.
    {
      const double rx = an_f[0], ry = an_f[1], rz = an_f[2];
      Leg6 o = a;
      o.abd = std::atan2(sd * ry, -rz);
      const double down = std::sqrt(ry * ry + rz * rz);
      const double LL = std::sqrt(rx * rx + down * down);
      o.knee = 3.14159265 - std::acos(std::fmax(-1.0, std::fmin(1.0, (thigh * thigh + shank * shank - LL * LL) / (2 * thigh * shank))));
      const double al = std::acos(std::fmax(-1.0, std::fmin(1.0, (thigh * thigh + LL * LL - shank * shank) / (2 * thigh * LL))));
      o.flex = std::atan2(rx, down) + al;
      mj_resetData(m, d);
      set_leg(side, o);
      double k4[3], a4[3], t4;
      model_leg(side, k4, a4, &t4);
      for (int i = 0; i < 3; ++i) e_ik_old = std::fmax(e_ik_old, std::fabs(a4[i] - an_f[i]));
    }
  }
  Row("FK formula vs model (ankle)", e_fk < 1e-3, "max error %.3f mm", e_fk * 1000);
  Row("hip dir->angles->knee (new)", e_hip < 1e-3, "max error %.3f mm (old split %.1f mm)", e_hip * 1000, e_hip_old * 1000);
  Row("leg IK target->angles->ankle", e_ik < 1e-3, "max error %.3f mm (old IK %.1f mm)", e_ik * 1000, e_ik_old * 1000);
  Row("leg IK sole flat", tilt_max < 1.0, "max sole tilt %.2f deg", tilt_max);
}

// Standing bot; extra generalized forces on leg hinge dofs for 0.1 s vs a control run.
void RunExtra(int mode, double out[3]) {
  World w;
  if (!Make(w)) return;
  mjData* d = w.data();
  const mjModel* m = w.model();
  Bot& b = w.bot(0);
  for (int f = 0; f < 60; ++f) w.StepFrame();
  const double c0[3] = {d->subtree_com[3 * b.body0], d->subtree_com[3 * b.body0 + 1], 0};
  const double R0 = std::atan2(d->xmat[9 * b.body0 + 3], d->xmat[9 * b.body0]);
  for (int f = 0; f < 6; ++f) {
    mju_zero(d->qfrc_applied, m->nv);
    if (mode == 1 || mode == 2) {  // VF: F on the CoM via J^T of both legs (same formula as l0_muscles)
      const double F[3] = {mode == 1 ? 100.0 : 0.0, mode == 2 ? 100.0 : 0.0, 0};
      for (int j = J_HIP_FLEX_L; j < J_COUNT; ++j) {
        const int side = j >= J_HIP_FLEX_R ? 1 : 0;
        d->qfrc_applied[b.dof0 + j] = b.w_load[side] * w.VfTorque(b, j, F, true);  // core formula (ankle capped)
      }
    }
    if (mode == 3) {  // yaw: desired pelvis torque +30 N m about +z through the left hip rotation (as l0_muscles)
      const double* a = d->xaxis + 3 * (b.jnt0 + J_HIP_ROT_L);
      d->qfrc_applied[b.dof0 + J_HIP_ROT_L] = -a[2] * 30.0;
    }
    w.StepFrame();
  }
  mju_zero(d->qfrc_applied, m->nv);
  out[0] = d->subtree_com[3 * b.body0] - c0[0];
  out[1] = d->subtree_com[3 * b.body0 + 1] - c0[1];
  out[2] = std::atan2(d->xmat[9 * b.body0 + 3], d->xmat[9 * b.body0]) - R0;
}

void TestForces() {
  double c[3] = {0, 0, 0}, fx[3] = {0, 0, 0}, fy[3] = {0, 0, 0}, yw[3] = {0, 0, 0};
  RunExtra(0, c);
  RunExtra(1, fx);
  RunExtra(2, fy);
  RunExtra(3, yw);
  Row("VF +x on CoM -> CoM moves +x", fx[0] - c[0] > 0, "dx %.2f mm (dy %.2f mm)", (fx[0] - c[0]) * 1000, (fx[1] - c[1]) * 1000);
  Row("VF +y on CoM -> CoM moves +y", fy[1] - c[1] > 0, "dy %.2f mm (dx %.2f mm)", (fy[1] - c[1]) * 1000, (fy[0] - c[0]) * 1000);
  Row("yaw +30 N m via stance hip rot", yw[2] - c[2] > 0, "pelvis yaw change %.3f deg vs control", (yw[2] - c[2]) * 57.3);
}

// Virtual work check of the VF torque formula on one stance leg (foot fixed): tau_ok_j = F . dc/dq_j by finite
// differences (whole body re-anchored so the stance foot does not move), vs tau_code_j = -a.((c - p) x F).
void TestVfVirtualWork() {
  World w;
  if (!Make(w)) return;
  mjModel* m = const_cast<mjModel*>(w.model());
  mjData* d = w.data();
  Bot& b = w.bot(0);
  for (int f = 0; f < 30; ++f) w.StepFrame();
  const int foot = b.body0 + B_FOOT_L;
  auto com_fixed = [&](double c[3]) {  // CoM expressed with the left foot pose held at its base pose
    mj_kinematics(m, d);
    mj_comPos(m, d);
    const double* cc = d->subtree_com + 3 * b.body0;
    for (int i = 0; i < 3; ++i) c[i] = cc[i];
  };
  double q0[64];
  for (int i = 0; i < m->nq && i < 64; ++i) q0[i] = d->qpos[i];
  mj_kinematics(m, d);
  double Fp[3], FR[9];
  for (int i = 0; i < 3; ++i) Fp[i] = d->xpos[3 * foot + i];
  for (int i = 0; i < 9; ++i) FR[i] = d->xmat[9 * foot + i];
  double c0[3];
  com_fixed(c0);
  const double* com = d->subtree_com + 3 * b.body0;
  double cref[3] = {com[0], com[1], com[2]};
  for (int axis = 0; axis < 2; ++axis) {
    const double F[3] = {axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, 0};
    int agree = 0, n = 0;
    double G[3] = {0, 0, 0}, Gok[3] = {0, 0, 0};
    for (int j = J_HIP_FLEX_L; j <= J_ANKLE_INV_L; ++j) {
      for (int i = 0; i < m->nq && i < 64; ++i) d->qpos[i] = q0[i];
      mj_kinematics(m, d);
      mj_comPos(m, d);
      const double* p = d->xanchor + 3 * (b.jnt0 + j);
      const double* a = d->xaxis + 3 * (b.jnt0 + j);
      const double r[3] = {cref[0] - p[0], cref[1] - p[1], cref[2] - p[2]};
      const double V[3] = {r[1] * F[2] - r[2] * F[1], r[2] * F[0] - r[0] * F[2], r[0] * F[1] - r[1] * F[0]};
      const double tc = -(a[0] * V[0] + a[1] * V[1] + a[2] * V[2]);
      // Finite difference with the foot re-anchored: c' = FR FRn^T (c_n - Fp_n) + Fp.
      const double eps = 1e-4;
      d->qpos[b.qadr0 + j] += eps;
      mj_kinematics(m, d);
      mj_comPos(m, d);
      const double* pn = d->xpos + 3 * foot;
      const double* Rn = d->xmat + 9 * foot;
      const double* cn = d->subtree_com + 3 * b.body0;
      double loc[3], cfix[3];
      for (int i = 0; i < 3; ++i) loc[i] = Rn[i] * (cn[0] - pn[0]) + Rn[3 + i] * (cn[1] - pn[1]) + Rn[6 + i] * (cn[2] - pn[2]);
      for (int i = 0; i < 3; ++i) cfix[i] = Fp[i] + FR[3 * i] * loc[0] + FR[3 * i + 1] * loc[1] + FR[3 * i + 2] * loc[2];
      double dc[3];
      for (int i = 0; i < 3; ++i) dc[i] = (cfix[i] - c0[i]) / eps;
      const double tok = F[0] * dc[0] + F[1] * dc[1] + F[2] * dc[2];
      if (std::fabs(tok) > 0.02) { ++n; if (tok * tc > 0) ++agree; }
      for (int i = 0; i < 3; ++i) { G[i] += tc * dc[i]; Gok[i] += tok * dc[i]; }
      if (axis == 1) printf("  vw y %-14s tau_code %+7.3f  tau_ok %+7.3f\n", "", tc, tok);
    }
    for (int i = 0; i < m->nq && i < 64; ++i) d->qpos[i] = q0[i];
    const double gn = std::sqrt(G[0] * G[0] + G[1] * G[1] + G[2] * G[2]) + 1e-12;
    const double cosg = (G[0] * F[0] + G[1] * F[1]) / gn;
    char nm[64];
    snprintf(nm, sizeof nm, "VF %s one leg: J^T sign (virtual work)", axis ? "y" : "x");
    Row(nm, agree == n && cosg > 0.5, "joints agree %.0f of %.0f", agree, n);
    (void)Gok;
  }
}

// Same VF-y experiment on two feet, but the balancer is replaced by a plain pose-holding PD with gravity
// compensation (test only), to split "J^T sign" from "balancer reaction".
void TestVfNoBalancer() {
  double res[2] = {0, 0};
  for (int run = 0; run < 2; ++run) {
    World w;
    if (!Make(w)) return;
    mjModel* m = const_cast<mjModel*>(w.model());
    mjData* d = w.data();
    Bot& b = w.bot(0);
    for (int f = 0; f < 60; ++f) w.StepFrame();
    double qh[J_COUNT];
    for (int k = 0; k < J_COUNT; ++k) qh[k] = d->qpos[b.qadr0 + k];
    const double y0 = d->subtree_com[3 * b.body0 + 1];
    for (int s = 0; s < 48; ++s) {
      mj_step1(m, d);
      for (int k = 0; k < J_COUNT; ++k) {
        const double u = b.js[k].kp * (qh[k] - d->qpos[b.qadr0 + k]) - 0.1 * b.js[k].kp * d->qvel[b.dof0 + k] +
                         d->qfrc_bias[b.dof0 + k];
        const double lim = b.js[k].tau_max;
        d->ctrl[b.act0 + k] = u > lim ? lim : (u < -lim ? -lim : u);
      }
      mju_zero(d->qfrc_applied, m->nv);
      if (run == 1) {
        const double F[3] = {0, 100.0, 0};
        for (int j = J_HIP_FLEX_L; j < J_COUNT; ++j) d->qfrc_applied[b.dof0 + j] = 0.5 * w.VfTorque(b, j, F, true);
      }
      mj_step2(m, d);
    }
    mj_comPos(m, d);
    res[run] = d->subtree_com[3 * b.body0 + 1] - y0;
  }
  Row("VF +y two feet, pose PD (no balancer)", res[1] - res[0] > 0, "dy %.2f mm vs control", (res[1] - res[0]) * 1000);
}
// Pelvis yaw experiments. Pose PD without the balancer; a stance-hip rotation torque is applied the way
// Muscles does it (joint torque = -a_z * T for a wanted torque T on the pelvis about +z). Reports pelvis and stance
// foot yaw after 0.15 s. mode 0: two feet; 1: on the left foot (right foot lifted); 2: on the left foot, no T, the
// right leg swings forward and across (reaction of the swing leg).
double YawOf(const mjData* d, int body) {
  const double* R = d->xmat + 9 * body;
  return std::atan2(R[3], R[0]);
}
void YawRun(int mode, double T, double out[2]) {
  World w;
  if (!Make(w)) return;
  mjModel* m = const_cast<mjModel*>(w.model());
  mjData* d = w.data();
  Bot& b = w.bot(0);
  for (int f = 0; f < 60; ++f) w.StepFrame();
  double qh[J_COUNT];
  for (int k = 0; k < J_COUNT; ++k) qh[k] = d->qpos[b.qadr0 + k];
  auto pd_step = [&](const double* q) {
    mj_step1(m, d);
    for (int k = 0; k < J_COUNT; ++k) {
      const double u = b.js[k].kp * (q[k] - d->qpos[b.qadr0 + k]) - 0.1 * b.js[k].kp * d->qvel[b.dof0 + k] +
                       d->qfrc_bias[b.dof0 + k];
      const double lim = b.js[k].tau_max;
      d->ctrl[b.act0 + k] = u > lim ? lim : (u < -lim ? -lim : u);
    }
  };
  if (mode >= 1) {  // lift the right foot: hip and knee flexed (0.1 s)
    qh[J_HIP_FLEX_R] = 0.5;
    qh[J_KNEE_R] = 1.0;
    for (int s = 0; s < 48; ++s) { pd_step(qh); mj_step2(m, d); }
  }
  const int foot = b.body0 + B_FOOT_L;
  const double y0 = YawOf(d, b.body0), f0 = YawOf(d, foot);
  double q[J_COUNT];
  for (int k = 0; k < J_COUNT; ++k) q[k] = qh[k];
  if (mode == 2) { q[J_HIP_FLEX_R] = 1.0; q[J_HIP_ABD_R] = -0.4; }  // swing forward and across (adduction)
  for (int s = 0; s < 72; ++s) {
    pd_step(q);
    for (int side = 0; side < (mode == 0 ? 2 : 1); ++side) {
      const int j = side ? J_HIP_ROT_R : J_HIP_ROT_L;
      const double az = d->xaxis[3 * (b.jnt0 + j) + 2];
      d->ctrl[b.act0 + j] -= az * T / (mode == 0 ? 2 : 1);
    }
    mj_step2(m, d);
  }
  out[0] = YawOf(d, b.body0) - y0;
  out[1] = YawOf(d, foot) - f0;
}
void TestYaw() {
  double r[2], c[2];
  YawRun(0, 0, c);
  YawRun(0, 60, r);
  Row("yaw T=+60 two feet -> pelvis +", r[0] - c[0] > 0, "pelvis %+.3f rad, stance foot %+.3f", r[0] - c[0],
      r[1] - c[1]);
  YawRun(1, 0, c);
  YawRun(1, 60, r);
  Row("yaw T=+60 on left foot -> pelvis +", r[0] - c[0] > 0, "pelvis %+.3f rad, stance foot %+.3f",
      r[0] - c[0], r[1] - c[1]);
  YawRun(2, 0, r);
  printf("%-34s %-6s pelvis %+.3f rad, stance foot %+.3f (right leg forward+across, no T)\n", "yaw swing reaction", "info",
         r[0] - c[0], r[1] - c[1]);
}
}  // namespace

int RunSelfTest() {
  printf("%-34s %-6s %s\n", "check", "result", "details");
  TestAxes();
  TestKinematics();
  TestForces();
  TestVfVirtualWork();
  TestVfNoBalancer();
  TestYaw();
  printf("selftest: %d errors\n", g_fail);
  return g_fail ? 1 : 0;
}
