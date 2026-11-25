// Measure of a standing foot turning in place. Heading of the ankle -> toe line of a flat loaded foot (ankle and
// toe points no higher than 2.5 cm above their heights at rest, load >= 15 % of the weight); the worst turn over any
// 0.2 s window and how far the heel and the toe moved in it. Target: <= 5 deg per 0.2 s with heel and toe moving
// < 3 cm; a loaded flat foot should turn only slowly, about the toe.
#pragma once

#include <mujoco/mujoco.h>

#include <cmath>

#include "emil_body.h"
#include "emil_world.h"

namespace emil {

struct FootSpin {
  static constexpr int kW = 12;  // 0.2 s at 60 Hz
  static constexpr double kFlat = 0.025, kLoad = 0.15, kSlide = 0.03, kTurn = 5.0;
  double low_a[2] = {0, 0}, low_t[2] = {0, 0};
  double ring[2][kW + 1][7] = {};  // per frame: heading [rad], heel x y, toe x y, sole x-axis heading [rad], roll [rad]
  int n[2] = {0, 0};               // flat loaded frames in a row
  int pos[2] = {0, 0};             // ring write position
  double worst = 0, worst_slide = 0, worst_t = -1;  // worst turn [deg] over 0.2 s, its heel/toe travel [m], end time
  int worst_f = -1;
  double worst_ip = 0, worst_ip_t = -1;  // worst turn with heel and toe travel < 3 cm (in place)
  int frames_ip = 0;                     // frames ending a window > 5 deg in place
  double last_turn[2] = {0, 0}, last_slide[2] = {0, 0};  // window ending this frame (0 if none)
  // The ankle -> toe heading also changes when the sole rolls onto an edge (the ankle moves sideways over the toe):
  // the sole's long axis heading over the same windows separates a real turn about the vertical from that roll.
  double worst_ax = 0, worst_ax_ip = 0, last_ax[2] = {0, 0};
  // Loaded sole roll onto an edge (load >= 15 %): the worst roll [deg] and frames with the ankle inversion joint
  // within 1 deg of its range end (the leg's sideways tilt over the foot beyond the ankle range rolls the foot).
  double roll_max = 0;
  int roll_frames = 0, lim_frames = 0;
  int ip_single = 0, ip_double = 0, ip_lim = 0;  // in-place windows > kTurn: one foot standing / two; ankle at range end
  int ip_settle = 0, ip_tip = 0;  // ... the sole rolled back toward flat (landed on an edge) or away from flat
  int ip_roll = 0, ip_noroll = 0;  // in-place windows > kTurn whose sole roll changed > 4 deg / less

  static void Points(const mjModel* m, const mjData* d, const Bot& b, int f, double* ank, double* heel, double* toe) {
    const double* a = d->xanchor + 3 * (b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L));
    const int g = b.geom_foot[f];
    const double* c = d->geom_xpos + 3 * g;
    const double* R = d->geom_xmat + 9 * g;
    const double sx = m->geom_size[3 * g], sz = m->geom_size[3 * g + 2];
    for (int k = 0; k < 3; ++k) {
      ank[k] = a[k];
      toe[k] = c[k] + R[3 * k] * sx - R[3 * k + 2] * sz;
      heel[k] = c[k] - R[3 * k] * sx - R[3 * k + 2] * sz;
    }
  }
  void Rest(const mjModel* m, const mjData* d, const Bot& b) {
    for (int f = 0; f < 2; ++f) {
      double a[3], h[3], t[3];
      Points(m, d, b, f, a, h, t);
      low_a[f] = a[2];
      low_t[f] = t[2];
      n[f] = 0;
    }
  }
  // Returns the foot (0/1) whose window ending now turned over kTurn in place, else -1 (for printing).
  int Step(const mjModel* m, const mjData* d, const Bot& b, double t, double* turn_out = nullptr) {
    int hit = -1;
    for (int f = 0; f < 2; ++f) {
      double a[3], h[3], to[3];
      Points(m, d, b, f, a, h, to);
      const bool flat = a[2] <= low_a[f] + kFlat && to[2] <= low_t[f] + kFlat;
      const bool load = b.foot_fn[f] >= kLoad * b.dims.total_mass * 9.81;
      if (load) {
        const double* Rr = d->geom_xmat + 9 * b.geom_foot[f];
        const double rl = std::fabs(std::asin(Rr[7])) * 57.29578;
        if (rl > roll_max) roll_max = rl;
        roll_frames += rl > 8.0;
        const int jn = b.jnt0 + (f ? J_ANKLE_INV_R : J_ANKLE_INV_L);
        const double qi = d->qpos[m->jnt_qposadr[jn]], lo = m->jnt_range[2 * jn], hi = m->jnt_range[2 * jn + 1];
        lim_frames += qi < lo + 0.0175 || qi > hi - 0.0175;
      }
      last_turn[f] = last_slide[f] = last_ax[f] = 0;
      if (!flat || !load) { n[f] = 0; continue; }
      double* e = ring[f][pos[f]];
      e[0] = std::atan2(to[1] - a[1], to[0] - a[0]);
      e[1] = h[0]; e[2] = h[1]; e[3] = to[0]; e[4] = to[1];
      const double* Rg = d->geom_xmat + 9 * b.geom_foot[f];
      e[5] = std::atan2(Rg[3], Rg[0]);
      e[6] = std::asin(Rg[7]);
      const int old = (pos[f] + 1) % (kW + 1);  // the entry kW frames back
      pos[f] = old;
      if (++n[f] <= kW) continue;
      const double* o = ring[f][old];
      double dh = e[0] - o[0];
      while (dh > M_PI) dh -= 2 * M_PI;
      while (dh < -M_PI) dh += 2 * M_PI;
      const double turn = std::fabs(dh) * 57.29578;
      const double slide = std::fmax(std::hypot(e[1] - o[1], e[2] - o[2]), std::hypot(e[3] - o[3], e[4] - o[4]));
      double dx = e[5] - o[5];
      while (dx > M_PI) dx -= 2 * M_PI;
      while (dx < -M_PI) dx += 2 * M_PI;
      const double ax = std::fabs(dx) * 57.29578;
      last_turn[f] = turn;
      last_slide[f] = slide;
      last_ax[f] = ax;
      if (ax > worst_ax) worst_ax = ax;
      if (slide < kSlide && ax > worst_ax_ip) worst_ax_ip = ax;
      if (turn > worst) { worst = turn; worst_slide = slide; worst_t = t; worst_f = f; }
      if (slide < kSlide) {
        if (turn > worst_ip) { worst_ip = turn; worst_ip_t = t; }
        if (turn > kTurn) {
          ++frames_ip;
          hit = f;
          if (turn_out) *turn_out = turn;
          (b.swing >= 0 ? ip_single : ip_double)++;
          (std::fabs(e[6] - o[6]) > 0.07 ? ip_roll : ip_noroll)++;
          (std::fabs(e[6]) < std::fabs(o[6]) ? ip_settle : ip_tip)++;
          const int jn = b.jnt0 + (f ? J_ANKLE_INV_R : J_ANKLE_INV_L);
          const double qi = d->qpos[m->jnt_qposadr[jn]];
          ip_lim += qi < m->jnt_range[2 * jn] + 0.0175 || qi > m->jnt_range[2 * jn + 1] - 0.0175;
        }
      }
    }
    return hit;
  }
};

}  // namespace emil
