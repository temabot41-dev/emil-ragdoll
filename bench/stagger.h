// Shared by the stagger bench and the picker: scenes, the target table, one-world runner,
// scoring.
#pragma once
#include <cmath>
#include <functional>
#include <mutex>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"

namespace stag {

const double kNan = std::nan("");
const double kRad = 57.29578;
constexpr double kRefMass = 61.6;  // body mass the target impulses are given for [kg]
constexpr double kTail = 6.0;      // s after the push

struct Scene { int dir; double J; const char* name; };  // dir: 0 back, 1 forward, 2 left, 3 right
const Scene kScenes[20] = {
    {0, 30, "back30"},  {0, 45, "back45"},  {0, 60, "back60"},  {0, 80, "back80"},   {0, 100, "back100"}, {0, 120, "back120"},
    {1, 30, "fwd30"},   {1, 45, "fwd45"},   {1, 60, "fwd60"},   {1, 80, "fwd80"},    {1, 100, "fwd100"},
    {2, 30, "left30"},  {2, 45, "left45"},  {2, 60, "left60"},  {2, 80, "left80"},   {2, 100, "left100"}, {2, 120, "left120"},
    {3, 45, "right45"}, {3, 60, "right60"}, {3, 100, "right100"}};

// Target table: group = direction (0 back, 1 forward, 2 side) x size (0 small, 1 medium,
// 2 large).
struct Ref {
  int steps;
  double l1, s1, d1, t1;          // step 1: lift [s], swing [s], displacement [m], capture point behind the ankle [m]
  double l2, s2, d2, t2, k2;      // step 2 and knee at touchdown [deg]
  double tr_lo, tr_hi, drop;      // trunk lean range [deg], pelvis drop [m]
};
const Ref kRef[3][3] = {
    {{2, 0.39, 0.20, 0.27, 0.01, 0.81, 0.23, 0.36, 0.02, 56, -3, 5, 0.04},
     {4, 0.13, 0.13, 0.30, 0.15, 0.43, 0.30, 0.75, 0.00, 36, -10, 5, 0.07},
     {4, 0.13, 0.10, 0.39, 0.40, 0.31, 0.43, 1.43, -0.02, 21, -15, 10, 0.17}},
    {{2, 0.15, 0.17, 0.35, -0.03, 0.42, 0.15, 0.21, 0.07, 39, 0, 9, 0.03},
     {3, 0.12, 0.18, 0.49, 0.07, 0.33, 0.23, 0.45, 0.09, 37, 0, 12, 0.05},
     {3, 0.13, 0.15, 0.59, 0.27, 0.32, 0.32, 1.10, 0.04, 23, 0, 18, 0.11}},
    {{0, kNan, kNan, kNan, kNan, kNan, kNan, kNan, kNan, kNan, 0, 8, 0},
     {4, 0.08, 0.27, 0.36, 0.23, 0.38, 0.24, 0.53, 0.01, 30, -16, 11, 0.03},
     {4, 0.05, 0.28, 0.66, 0.52, 0.24, 0.32, 1.06, 0.19, 37, -8, 15, 0.09}}};
const char* const kDirName[3] = {"назад", "вперёд", "вбок"};
const char* const kSizeName[3] = {"малый", "средний", "большой"};

struct Step {
  int foot; double t_lift, swing, along, across, tz, knee, ahead;  // ahead: ankle - CoM along the push
  double hmax = 0;  // highest ankle rise above its rest height during the swing [m]
  double dyaw = 0, shr = 0;  // largest pelvis heading change lift -> touchdown [deg], shoulder flexion range [deg]
};

struct Res {  // one scene of one moment
  int scene = 0;
  double moment = 0, dv = 0;
  bool stood = true;
  int nsteps = 0;
  double t_fall = -1;  // s after the push until the pelvis < 0.5 m or the trunk touched the floor; -1 = stood
  Step st[2];
  std::vector<Step> all_steps;  // every step before a fall
  int have = 0;  // first two steps found
  double lean_min = 0, lean_max = 0, drop = 0;
  // lower trunk (pelvis -> chest joint) and upper trunk (chest joint -> neck) largest tilt gain
  // with the push in the first 0.5 s [deg], pelvis / chest speed along the push at 0.1 s, head speed peak in 1 s
  double lt05 = 0, ut05 = 0, vr01 = -1, hvpk = 0;
  double yaw = 0;  // max pelvis yaw from the push moment over the tail [deg]
  int late = 0;          // steps lifted later than 2.5 s after the push
  double t_rest = -1;    // last time the CoM speed was >= 0.15 m/s (stood only) [s]
  double arms_dn = -1;   // shoulder abduction under 15 deg this long after t_rest (stood only) [s]
  double rp_flex = -1, rp_abd = -1, rp_tw = -1;  // largest deviation from the push-time pose 0.6 s after rest [deg]
  double t_one = 0;      // time at rest on one foot (mode 0, no swing, a foot under 5 % weight) [s]
  double fspin = 0;      // largest turn of a standing foot on the floor since it was loaded [deg]
  double roll_max = 0;  // worst loaded sole roll [deg]
  int roll_frames = 0, lim_frames = 0;  // loaded frames rolled > 8 deg, at the ankle inversion range end
  int ip_single = 0, ip_double = 0, ip_lim = 0;  // in-place window frames > 5 deg by support, at range end
  int ip_settle = 0, ip_tip = 0;
  int ip_roll = 0, ip_noroll = 0;  // in-place window frames > 5 deg with the sole roll changing > 4 deg / less
  double spin02_ax = 0, spin02_axip = 0;  // sole long-axis heading turn in 0.2 s (no roll part) [deg]
  double spin02 = 0, slide02 = 0, spin02_ip = 0, spin02_t = -1;  // flat foot turn in 0.2 s [deg], travel [m]
  double fa_end = 0, lat_end = 0;  // feet split 2.4 s after the push along / across the pelvis heading [m]
  std::vector<double> e_lon, e_lat;  // balancer step ends: sole centre minus place along / across the heading [m]
  std::vector<double> e_over, e_vpk, e_dur, e_len, e_yaw;  // overshoot [m], peak sole speed [m/s], swing
                                                         // time [s], step length [m], pelvis yaw range [rad]
};


struct Item { int scene; double moment; emil::BodyParams body; double scale = 1.0; };  // scene index, push time [s], body, impulse scale

// One world with one bot per item (spacing 3 m); `setup(world)` sets the parameters after Build, then the world is
// reset.
// Pushes at item.moment, 6 s of tracking per bot. Builds are serialised by BuildMutex() (shared with the picker).
void RunWorld(const std::vector<Item>& items, const std::function<void(emil::World&)>& setup, std::vector<Res>* out);
std::mutex& BuildMutex();

// Distance of one scene from the table [band widths, 0..3, mean over metrics]; a bot that fell gets 2 + (first two
// steps against the table) / 3, so candidates are told apart even when nothing stands. A bot that stood is capped at 2.
double PhasePenalty(const Res& r);
int SizeGroup(double dv);  // 0 small (< 0.6 m/s), 1 medium (0.6..1.1), 2 large

}  // namespace stag
