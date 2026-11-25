// L3 perception: snapshot of body state for the controllers. Velocities by finite difference over one
// control step (no extra MuJoCo passes).
#include <algorithm>
#include <cmath>

#include "emil_world.h"

namespace emil {

// Top face of a support geom at (x, y): a plane through its origin, a box's face most up; other shapes
// (none static so far) by their bounding sphere top.
double World::SupportTop(int g, double x, double y) const {
  const double* R = d_->geom_xmat + 9 * g;
  const double* gp = d_->geom_xpos + 3 * g;
  int k = 2;
  double s = 0;
  if (m_->geom_type[g] == mjGEOM_BOX) {
    for (int i = 0; i < 2; ++i)
      if (std::fabs(R[6 + i]) > std::fabs(R[6 + k])) k = i;
    s = m_->geom_size[3 * g + k];
  } else if (m_->geom_type[g] != mjGEOM_PLANE) {
    return gp[2] + m_->geom_rbound[g];
  }
  const double sg = R[6 + k] < 0 ? -1.0 : 1.0;
  const double ax = sg * R[k], ay = sg * R[3 + k], az = std::fmax(sg * R[6 + k], 0.2);
  const double tx = gp[0] + s * ax, ty = gp[1] + s * ay, tz = gp[2] + s * az;
  return tz - (ax * (x - tx) + ay * (y - ty)) / az;
}

// The ground under a point: the highest support top (plane, or a box whose footprint holds the point)
// not above z + 5 cm; -1e9 if none.
double World::GroundBelow(double x, double y, double z, bool* over) const {
  double best = -1e9;
  if (over) *over = false;
  for (int g : sup_geoms_) {
    if (m_->geom_type[g] == mjGEOM_BOX) {
      const double* R = d_->geom_xmat + 9 * g;
      const double* gp = d_->geom_xpos + 3 * g;
      const double* s = m_->geom_size + 3 * g;
      bool in = true;
      for (int k = 0; k < 3 && in; ++k) {  // horizontal extent along each local axis
        const double ax = R[k], ay = R[3 + k], n2 = ax * ax + ay * ay;
        if (n2 < 0.25) continue;  // the axis stands up
        const double u = (x - gp[0]) * ax + (y - gp[1]) * ay;
        in = std::fabs(u) <= s[k] * n2 + 1e-9;
      }
      if (!in) continue;
    }
    const double top = SupportTop(g, x, y);
    if (top <= z + 0.05 && top > best) best = top;
    if (over && top > z + 0.05) *over = true;  // a support rises above z here (a wall)
  }
  return best;
}

void World::Perceive(Bot& b) {
  Percept& p = b.pc[b.cur];
  const Percept& o = b.pc[b.cur ^ 1];
  // Support height under the feet, load weighted; held while neither foot stands on a support. Exact 0
  // on the floor plane (floor scenes unchanged).
  {
    double sw = 0, sz = 0;
    for (int f = 0; f < 2; ++f) {
      const int g = b.foot_sg[f];
      if (g < 0 || b.foot_fn[f] <= 0 || !geom_sup_[g]) continue;
      const int gf = b.geom_foot[f];
      const double wf = std::fmax(b.ld[f], 1e-3);
      sw += wf;
      sz += wf * SupportTop(g, d_->geom_xpos[3 * gf], d_->geom_xpos[3 * gf + 1]);
    }
    if (sw > 0) {
      b.gz = sz / sw;
      // The stand floor of the tumble test = the support under the standing feet while no fall is judged. Set
      // here, not in L2, so it follows the body from any spawn height (L2 returns early for a body that is not
      // falling).
      if (!(b.fall_pred || b.mode == 2)) b.gz_st = b.gz;
    } else {  // no foot on a support (in the air, lying, sitting): the ground under the CoM below the lower foot
      const double* cw = d_->subtree_com + 3 * b.body0;
      const double zf = std::fmin(d_->geom_xpos[3 * b.geom_foot[0] + 2], d_->geom_xpos[3 * b.geom_foot[1] + 2]);
      const double gb = GroundBelow(cw[0], cw[1], std::fmin(zf, cw[2]));
      if (gb > -1e8) b.gz = gb;
    }
  }
  const double gz = b.gz;
  p.gz = gz;
  const double* c = d_->subtree_com + 3 * b.body0;
  for (int k = 0; k < 3; ++k) {
    p.com[k] = c[k] - (k == 2 ? gz : 0.0);
    p.com_vel[k] = (c[k] - o.com[k] - (k == 2 ? o.gz : 0.0)) / kFrameDt;
  }
  if (d_->time == 0) p.com_vel[0] = p.com_vel[1] = p.com_vel[2] = 0;
  for (int f = 0; f < 2; ++f) {
    const int g = b.geom_foot[f];
    const int ja = b.jnt0 + (f == 0 ? J_ANKLE_DORSI_L : J_ANKLE_DORSI_R);
    for (int k = 0; k < 3; ++k) {
      p.foot_c[f][k] = d_->geom_xpos[3 * g + k];
      p.ankle[f][k] = d_->xanchor[3 * ja + k];
    }
    p.foot_c[f][2] -= gz;
    p.ankle[f][2] -= gz;
    p.foot_contact[f] = 0;
  }
  for (int f = 0; f < 2; ++f) p.foot_contact[f] = d_->time == 0 ? 1 : b.foot_on[f];
  // The support under each foot and the foot's place on it (geom frame: rows of xmat are world axes)
  for (int f = 0; f < 2; ++f) {
    const int g = b.foot_sg[f];
    const int msk = g >= 0 ? geom_edge_[g] : 0;
    for (int k = 0; k < 2; ++k) {
      p.deck_half[f][k] = 0, p.deck_off[f][k] = 0, p.deck_ax[f][k][0] = p.deck_ax[f][k][1] = 0;
      if (!(msk & (1 << k))) continue;
      const double* R = d_->geom_xmat + 9 * g;
      const double ax = R[k], ay = R[3 + k], n = std::hypot(ax, ay);  // column k = local axis k in world
      if (n < 0.5) continue;  // the axis stands up: no horizontal edge along it
      const double* gp = d_->geom_xpos + 3 * g;
      p.deck_ax[f][k][0] = ax / n, p.deck_ax[f][k][1] = ay / n;
      p.deck_off[f][k] = ((p.foot_c[f][0] - gp[0]) * ax + (p.foot_c[f][1] - gp[1]) * ay) / n;
      p.deck_half[f][k] = m_->geom_size[3 * g + k] * n;
    }
  }
  const double* Rp = d_->xmat + 9 * b.body0;
  p.pelvis_pitch = std::atan2(Rp[2], Rp[8]);
  p.pelvis_roll = std::atan2(Rp[5], Rp[8]);
  p.pelvis_rate[0] = (p.pelvis_pitch - o.pelvis_pitch) / kFrameDt;
  p.pelvis_rate[1] = (p.pelvis_roll - o.pelvis_roll) / kFrameDt;
  p.pelvis_yaw = std::atan2(Rp[3], Rp[0]);
  double dyaw = p.pelvis_yaw - o.pelvis_yaw;
  if (dyaw > 3.14159265) dyaw -= 6.2831853;
  if (dyaw < -3.14159265) dyaw += 6.2831853;
  p.pelvis_rate[2] = dyaw / kFrameDt;
  if (d_->time == 0) p.pelvis_rate[0] = p.pelvis_rate[1] = p.pelvis_rate[2] = 0;
  const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
  const double tp = std::atan2(Rc[2], Rc[8]), tr = std::atan2(Rc[5], Rc[8]);
  p.trunk_pitch = tp;
  p.trunk_roll = tr;
  for (int k = 0; k < 3; ++k) p.chest_up[k] = Rc[3 * k + 2];
  for (int j = 0; j < J_COUNT; ++j) p.q[j] = d_->qpos[b.qadr0 + j];
  for (int k = 0; k < 9; ++k) {
    p.pelvis_R[k] = Rp[k];
    p.foot_R[0][k] = d_->geom_xmat[9 * b.geom_foot[0] + k];
    p.foot_R[1][k] = d_->geom_xmat[9 * b.geom_foot[1] + k];
  }
  for (int f = 0; f < 2; ++f) {
    const int jh = b.jnt0 + (f == 0 ? J_HIP_FLEX_L : J_HIP_FLEX_R);
    for (int k = 0; k < 3; ++k) p.hip[f][k] = d_->xanchor[3 * jh + k];
    p.hip[f][2] -= gz;
  }
  // Pelvis angular velocity from the rotation over one control step (W = R R_old^T = I + [w dt]x) and the hip
  // velocities: the swing leg previews the pelvis motion over the muscle response time.
  {
    const double* a = p.pelvis_R;
    const double* q = o.pelvis_R;
    double Wm[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) Wm[i][j] = a[3 * i] * q[3 * j] + a[3 * i + 1] * q[3 * j + 1] + a[3 * i + 2] * q[3 * j + 2];
    const double k0 = d_->time == 0 ? 0.0 : 0.5 / kFrameDt;
    p.pelvis_w[0] = k0 * (Wm[2][1] - Wm[1][2]);
    p.pelvis_w[1] = k0 * (Wm[0][2] - Wm[2][0]);
    p.pelvis_w[2] = k0 * (Wm[1][0] - Wm[0][1]);
    for (int f = 0; f < 2; ++f)
      for (int k = 0; k < 3; ++k)
        p.hip_vel[f][k] = d_->time == 0 ? 0.0 : (p.hip[f][k] - o.hip[f][k] + (k == 2 ? gz - o.gz : 0.0)) / kFrameDt;
  }

}

// One pass over the contacts (order kept, so the sums equal the per-bot loops they replace): per foot the normal
// force, the horizontal force in world axes and the support velocity. Contacts with the bot's own body do not count
// as support (a crossing foot pressing on the stance shin must not count as load).
void World::FootContactsAll() {
  std::fill(foot_force_.begin(), foot_force_.end(), 0.0);
  std::fill(foot_sgf_.begin(), foot_sgf_.end(), 0.0);
  std::fill(foot_sgi_.begin(), foot_sgi_.end(), -1);
  mjtNum cf[6], ov[6];
  for (int i = 0; i < d_->ncon; ++i) {
    const mjContact& cn = d_->contact[i];
    for (int side = 0; side < 2; ++side) {
      const int g = side ? cn.geom2 : cn.geom1, og = side ? cn.geom1 : cn.geom2;
      const int ow = foot_bot_[g];
      if (ow < 0) continue;
      const Bot& b = bots_[ow >> 1];
      const int ob = m_->geom_bodyid[og] - b.body0;
      if (ob >= 0 && ob < B_COUNT) continue;
      mj_contactForce(m_, d_, i, cf);
      double* acc = &foot_force_[5 * ow];  // fn, fh x, fh y, support velocity x, y of this foot (bot * 10 + foot * 5)
      acc[0] += cf[0];
      // The support's own velocity under the foot (a moving deck; the floor is body 0 at rest),
      // normal-force weighted (divided by fn in FootContacts), at the contact point (bridge links turn); the normal
      // near vertical
      const int sb = m_->geom_bodyid[og];
      // support geom under the foot
      if (cf[0] > foot_sgf_[ow] && std::fabs(cn.frame[2]) > 0.7) foot_sgf_[ow] = cf[0], foot_sgi_[ow] = og;
      if (sb > 0 && cf[0] > 0 && std::fabs(cn.frame[2]) > 0.7) {  // under the foot only (not a car side)
        mj_objectVelocity(m_, d_, mjOBJ_BODY, sb, ov, 0);  // at the body origin: add w x r to the contact point
        const double* xo = d_->xpos + 3 * sb;
        const double rx = cn.pos[0] - xo[0], ry = cn.pos[1] - xo[1], rz = cn.pos[2] - xo[2];
        acc[3] += cf[0] * (ov[3] + ov[1] * rz - ov[2] * ry);
        acc[4] += cf[0] * (ov[4] + ov[2] * rx - ov[0] * rz);
      }
      // horizontal part in world axes; the contact force acts on geom2 along the frame (normal geom1 -> geom2)
      const double sg = side ? 1.0 : -1.0;
      for (int k = 0; k < 2; ++k) acc[1 + k] += sg * (cf[0] * cn.frame[k] + cf[1] * cn.frame[3 + k] + cf[2] * cn.frame[6 + k]);
    }
  }
}

// Foot contact state from the measured floor force (one hysteresis on the filtered load share).
void World::FootContacts(Bot& b) {
  const double* fa = &foot_force_[10 * (&b - &bots_[0])];
  const double fn[2] = {fa[0], fa[5]};
  const double fh[2][2] = {{fa[1], fa[2]}, {fa[6], fa[7]}};
  const double sv[2][2] = {{fa[3], fa[4]}, {fa[8], fa[9]}};
  const double W = b.dims.total_mass * 9.81;
  // One foot state for all layers: load share low-passed 25 ms with one hysteresis (kLoadOn / kLoadOff); the
  // contact flag of the percept is this state.
  const double a25 = m_->opt.timestep / (0.025 + m_->opt.timestep);
  for (int f = 0; f < 2; ++f) {
    b.foot_fn[f] = fn[f];
    b.foot_sg[f] = foot_sgi_[2 * (&b - &bots_[0]) + f];
    b.foot_sv[f][0] = fn[f] > 1e-6 ? sv[f][0] / fn[f] : 0.0;
    b.foot_sv[f][1] = fn[f] > 1e-6 ? sv[f][1] / fn[f] : 0.0;
    for (int k = 0; k < 2; ++k) b.foot_fh[f][k] += a25 * (fh[f][k] - b.foot_fh[f][k]);
    b.ld[f] += a25 * (std::fmin(fn[f] / W, 1.5) - b.ld[f]);
    if (b.ld[f] > kLoadOn) b.loaded[f] = 1;
    else if (b.ld[f] < kLoadOff) b.loaded[f] = 0;
    b.foot_on[f] = b.loaded[f];
  }
  // Foot load share for the muscle weight and virtual-force terms. Total = the weight the feet really carry
  // (measured normal force / body weight, low-passed 20 ms, capped at 1): zero off the ground and in mode 2
  // (fallen / lying: the legs do not carry the body; otherwise a lying leg gets the whole-body weight torque at
  // the knee and folds to its stop). Split between the feet: one foot loaded -> all to it;
  // both -> CoM projection between the foot centres (static split; the measured split is positive feedback:
  // the more loaded leg pushes harder and takes still more load until the stance collapses).
  const double dt = m_->opt.timestep, a = dt / (0.02 + dt);
  const double tot = b.mode == 2 ? 0.0 : std::fmin((fn[0] + fn[1]) / W, 1.0);
  b.w_tot += a * (tot - b.w_tot);
  double w = 0.5;
  const double* p0 = d_->geom_xpos + 3 * b.geom_foot[0];
  const double* p1 = d_->geom_xpos + 3 * b.geom_foot[1];
  const bool on0 = b.loaded[0], on1 = b.loaded[1];
  if (on0 && !on1) w = 1;
  else if (!on0 && on1) w = 0;
  else {
    const double* c = d_->subtree_com + 3 * b.body0;
    const double dx = p0[0] - p1[0], dy = p0[1] - p1[1];
    w = ((c[0] - p1[0]) * dx + (c[1] - p1[1]) * dy) / (dx * dx + dy * dy + 1e-9);
    w = w < 0 ? 0 : (w > 1 ? 1 : w);
  }
  b.w_load[0] = b.w_tot * w;
  b.w_load[1] = b.w_tot * (1 - w);
}

}  // namespace emil
