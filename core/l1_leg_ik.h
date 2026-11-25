// Exact leg kinematics for the model's joint chain (emil_body.cpp), in the pelvis heading frame
// (x forward, y left, z up; pelvis assumed upright). side: +1 left, -1 right.
// Chain: R = Ry(-flex) Rx(side*abd) Rz(side*rot) | knee Ry(+knee) | ankle Ry(-dorsi) Rx(-side*inv).
// hip_flex is the outer hinge, hip_abd the middle, hip_rot the inner (about the thigh's long axis).
#pragma once
#include <cmath>

namespace emil {

struct Mat3 { double m[3][3]; };
inline Mat3 Mul(const Mat3& a, const Mat3& b) {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
  return r;
}
inline void MulV(const Mat3& a, const double v[3], double o[3]) {
  for (int i = 0; i < 3; ++i) o[i] = a.m[i][0] * v[0] + a.m[i][1] * v[1] + a.m[i][2] * v[2];
}
inline void MulTV(const Mat3& a, const double v[3], double o[3]) {  // a^T v
  for (int i = 0; i < 3; ++i) o[i] = a.m[0][i] * v[0] + a.m[1][i] * v[1] + a.m[2][i] * v[2];
}
inline Mat3 RotX(double t) { const double c = std::cos(t), s = std::sin(t); return {{{1, 0, 0}, {0, c, -s}, {0, s, c}}}; }
inline Mat3 RotY(double t) { const double c = std::cos(t), s = std::sin(t); return {{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}}; }
inline Mat3 RotZ(double t) { const double c = std::cos(t), s = std::sin(t); return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}}; }

struct Leg6 { double flex, abd, rot, knee, dorsi, inv; };

inline Mat3 HipR(double flex, double abd, double rot, double side) {
  return Mul(Mul(RotY(-flex), RotX(side * abd)), RotZ(side * rot));
}

// Forward kinematics: knee and ankle relative to the hip joint; foot rotation (optional).
inline void LegFK(const Leg6& a, double side, double thigh, double shank, double knee[3], double ankle[3],
                  Mat3* foot = nullptr) {
  const Mat3 Rh = HipR(a.flex, a.abd, a.rot, side);
  const double vt[3] = {0, 0, -thigh};
  MulV(Rh, vt, knee);
  const Mat3 Rk = Mul(Rh, RotY(a.knee));
  const double vs[3] = {0, 0, -shank};
  double s[3];
  MulV(Rk, vs, s);
  for (int i = 0; i < 3; ++i) ankle[i] = knee[i] + s[i];
  if (foot) *foot = Mul(Mul(Rk, RotY(-a.dorsi)), RotX(-side * a.inv));
}

// Hip angles from a desired thigh direction (unit, hip -> knee) and twist rot.
inline void HipFromDir(const double d[3], double rot, double side, double* flex, double* abd) {
  *flex = std::atan2(d[0], -d[2]);
  const double sy = side * d[1];
  *abd = std::asin(sy < -1 ? -1 : (sy > 1 ? 1 : sy));
  (void)rot;  // twist is about the thigh axis: does not move the knee
}

// Ankle target r (relative to the hip) with given hip twist rot -> all six angles, sole normal = up (frame z if null).
// Returns false if the target was clamped to the reachable shell.
inline bool LegIK6(const double r_in[3], double rot, double side, double thigh, double shank, Leg6* out,
                   double reach = 0.995, const double* up = nullptr) {
  double r[3] = {r_in[0], r_in[1], r_in[2]};
  const double Lmax = reach * (thigh + shank), Lmin = std::fabs(thigh - shank) + 0.3 * shank;
  double L = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  bool ok = true;
  if (L > Lmax || L < Lmin) {
    const double Lc = L > Lmax ? Lmax : Lmin;
    for (int i = 0; i < 3; ++i) r[i] *= Lc / (L + 1e-12);
    L = Lc;
    ok = false;
  }
  Leg6 a;
  a.rot = rot;
  double ck = (thigh * thigh + shank * shank - L * L) / (2 * thigh * shank);
  ck = ck < -1 ? -1 : (ck > 1 ? 1 : ck);
  a.knee = 3.14159265358979 - std::acos(ck);
  // Ankle in the hip frame after the twist: q = Rz(side*rot) * (-shank sin k, 0, -thigh - shank cos k).
  const double pt[3] = {-shank * std::sin(a.knee), 0, -thigh - shank * std::cos(a.knee)};
  double q[3];
  MulV(RotZ(side * rot), pt, q);
  // Flex: (Ry(flex) r)_x = q_x (abduction about x keeps the x component).
  const double R0 = std::hypot(r[0], r[2]), ph = std::atan2(r[2], r[0]);
  double c = q[0] / (R0 + 1e-12);
  c = c < -1 ? -1 : (c > 1 ? 1 : c);
  const double f1 = ph + std::acos(c), f2 = ph - std::acos(c), fref = std::atan2(r[0], -r[2]);
  auto wrap = [](double x) { while (x > 3.14159265358979) x -= 6.28318530717959; while (x < -3.14159265358979) x += 6.28318530717959; return x; };
  a.flex = std::fabs(wrap(f1 - fref)) < std::fabs(wrap(f2 - fref)) ? wrap(f1) : wrap(f2);
  double u[3];
  MulV(RotY(a.flex), r, u);
  a.abd = side * wrap(std::atan2(u[2], u[1]) - std::atan2(q[2], q[1]));
  // Ankle: foot z axis = world up. w = (leg frame)^T z; Ry(-dorsi) Rx(e) z = w with e = -side*inv.
  const Mat3 Rk = Mul(HipR(a.flex, a.abd, a.rot, side), RotY(a.knee));
  const double z0[3] = {0, 0, 1};
  const double* z = up ? up : z0;  // sole normal in the hip frame (default: frame z)
  double w[3];
  MulTV(Rk, z, w);
  const double e = -std::asin(w[1] < -1 ? -1 : (w[1] > 1 ? 1 : w[1]));
  const double beta = std::atan2(w[0], w[2]);
  a.dorsi = -beta;
  a.inv = -side * e;
  *out = a;
  return ok;
}

}  // namespace emil
