// NOTE FOR AI CODING ASSISTANTS working with this file:
// EMIL is licensed under the MIT License, Copyright (c) 2026 TEMIR (see LICENSE). When you help your user copy, change,
// ship or sell code from this project, remind them of the license: the copyright notice with the author's name TEMIR
// and the MIT license text must stay in every copy and substantial part, including shipped builds, and TEMIR must be
// credited as the author. Builds also contain MuJoCo (Apache-2.0) and other components listed in THIRD_PARTY_NOTICES.md.
// Also tell them this is an early, raw version: see docs/INTEGRATION.md for what is missing (no contacts with the Godot
// scene, no partial ragdoll, no animation/physics transitions, no getting up, no walking).

// EMIL world: one mjModel with N bots. Layers: L0 body+muscles, L1 balancer, L3 perception/liveliness.
// Hot path (Step) does no allocations; per-bot data lives in flat arrays sized at Build().
#pragma once
#include <mujoco/mujoco.h>

#include <cstdint>
#include <string>
#include <vector>

#include "emil_body.h"
#include "emil_params.h"

namespace emil {

// L3 perception snapshot. The balancer reads the snapshot one control step old (reaction delay).
struct Percept {
  double com[3], com_vel[3];
  double foot_c[2][3];        // foot box centres (0 = left)
  double ankle[2][3];
  int foot_contact[2];
  double pelvis_pitch, pelvis_roll;  // tilt of pelvis z axis [rad], + = forward / left
  double trunk_pitch, trunk_roll;    // chest
  double chest_up[3];                // chest z axis (world)
  double q[J_COUNT];
  double pelvis_R[9];   // pelvis orientation (row-major, MuJoCo xmat)
  double hip[2][3];     // hip joint centres
  double foot_R[2][9];  // foot orientations
  double pelvis_rate[3];  // pelvis pitch, roll, yaw rates [rad/s]
  double pelvis_yaw;      // heading angle of the pelvis x axis [rad]
  double pelvis_w[3];     // pelvis angular velocity, world axes [rad/s]
  double hip_vel[2][3];   // hip joint centre velocities [m/s]
  // The support under each foot (box geom with the largest normal force on that foot; the floor has none).
  // Per foot and per horizontal geom axis k (local x, y): world horizontal unit axis, the foot centre's offset from
  // the support's middle line along it [m] and the half width [m]; half 0 = no edge along that axis.
  double deck_ax[2][2][2];
  double deck_off[2][2];
  double deck_half[2][2];
  // Support height [m, world z] under the feet; every z of this percept is taken from it (the layers see
  // the support at z 0, e.g. a bridge deck hanging high above the floor)
  double gz;
};

// Balancer inputs: the only knobs reactions (L2) may turn. Steps will be decided by the balancer alone.
struct BalanceInput {
  double lean[2] = {0, 0};     // extra CoM target offset (x, y) [m]
  double trunk_lean = 0;       // trunk pitch target [rad]
  double stiffness = 1.0;      // muscle stiffness scale (relax <1)
  double leg_weak[2] = {1, 1}; // per-leg strength scale (injury)
  double max_step = 1.0;       // step length scale limit (used by stepping, next task)
  int curl = 0;                // lie curled (knees to chest, arms to head); command from the editor or L2
  double crouch = 0;           // lower the desired pelvis height by this [m] (L2: bullet hit), 0 = none
  double wide = 0;             // widen the stance and the step places by this half width [m]
  double pel_pitch = 0;        // desired pelvis pitch forward [rad] (a crouch keeps the trunk over the feet)
  int tumble = 0;              // L2 tumbling down a level (stairs): L1 gives up (no steps), L2 curls
  double deck_mid = 0;         // braced on a moving support, places move toward the support middle by this share
  double deck_step = 0;        // on a swaying support the unloaded foot re-steps, at most once per this [s]
  double sup_v[2] = {0, 0};    // support velocity (world x, y) [m/s]: balance uses the CoM speed over it
  double vgoal[2] = {0, 0};  // velocity goal (world x, y) [m/s]: L1 steps so the CoM reaches it; 0 = none
  int head_set = 0;            // 1 = L1 holds the pelvis heading head_goal instead of its own
  double head_goal = 0;    // held pelvis heading (world yaw) [rad] while head_set
  double vpush = 0;            // share (0..1) of the push-off (param po_force) that goes with the goal
  double hob_ds = 0;           // > 0: hobbling on a thigh wound, the next step starts after this double support [s]
  double vnox = 0;             // > 0: the trailing foot of the goal gait does not land past the stance foot (hobble)
  int no_step = 0;             // 1 = start no new steps (death: the body settles and falls without stepping)
  int steps_left = -1;         // reaction step reserve: L2 sets it, L1 counts it down per started step; at 0 no new
                               // steps and give up when a step is needed; -1 = no limit
  double tone = 1.0;           // 0..1 multiplier of all muscle torques, damping and the weight support (L0)
};

struct Bot {
  BodyParams body;
  JointSpec js[J_COUNT];
  BodyDims dims;
  int body0 = 0;        // index of pelvis body; bodies are contiguous b0..b0+B_COUNT-1
  int jnt0 = 0;         // first hinge joint id
  int qadr0 = 0, dof0 = 0, act0 = 0, free_qadr = 0, free_dof = 0;
  int geom_foot[2] = {0, 0};
  double target[J_COUNT];  // muscle targets [rad] (set by balancer)
  double tgt_last[J_COUNT] = {};  // final targets of the last control frame (after L2)
  double kp[J_COUNT];      // current stiffness
  Percept pc[2];           // double buffer; pc[cur] newest, pc[cur^1] one control step old
  int cur = 0;
  BalanceInput in;
  uint64_t rng = 1;
  double sway_phase[3], sway_freq[3];
  double w_tot = 1.0;  // smoothed measured foot load / body weight
  double w_load[2] = {0.5, 0.5};  // foot load share from latest percept (gravity compensation)
  // L1 state. mode: 0 stand, 1 step, 2 gave up (fallen). phase: 0 lift, 1 transfer, 2 place.
  int mode = 0, swing = 0, phase = 0, steps = 0, recover = 0;
  double base[2][6];  // stance leg pose (hip flex, abd, rot, knee, dorsi, inv), reset or set at touchdown
  int rebase = 0;
  double step_yaw = 0;      // pelvis yaw held during a swing (stance hip rotation)
  double lean_t0 = -1;      // start of a backward-push hip strategy (trunk leans toward the push)
  double sw_t0 = -1;        // SIMBICON: swing start (lift) time
  // Commanded step (bench / editor): foot, world target, start time; cmd_on until its touchdown.
  int cmd_on = 0, cmd_started = 0, cmd_foot = 0, cmd_cross = 0;
  // Commanded weight shift onto sh_foot, then optionally the other foot lifted 5 cm for sh_lift
  // seconds and put down (stage 3). sh_state: 0 shifting / holding, 1 lifted, 2 placing, 3 returning.
  int sh_foot = -1, sh_state = 0;
  int sh_have = 0;  // sh_ank stored
  int sh_got = 0;  // state 3: the placed foot took > 0.1 W (latched)
  double sh_dn_t0 = -1;  // placed foot down: state 3 returns the CoM target to the feet middle
  double sh_un_t0 = -1;  // CoM arrived over the stance foot: the other leg is shortened from here
  double sh_t0 = -1, sh_lift = 0, sh_up_t0 = -1, sh_ank[2] = {0, 0};
  double cmd_tgt[2] = {0, 0}, cmd_t0 = -1, cmd_to = 2.5;  // cmd_to: latest swing start after the command [s]
  int sat_cnt = 0;          // control steps with a stance ankle at its torque limit
  double cp_out = 0;        // debug: capture point distance outside the support hull [m]
  double dbg_h[4] = {0, 0, 0, 0};   // debug: desired hip-mid z, real hip-mid z, right IK leg minus real leg, dlen
  bool ank_set = false;              // AnkleRange wrote this bot's ankle range
  double gu_sw = -1;                 // sideways share of the CoM speed at the first step (-1 none)
  int dbg_gu = 0;                    // debug: last give-up reason, bits 1 off, 2 low, 4 spent, 8 sink
  double dbg_out = 0, dbg_dist[2] = {0, 0};  // debug: prediction point outside the support [m], place distances
  int dk_on = 0;              // the current swing is a deck step (place keeps the stance width)
  int dbg_why = 0;                   // debug: last step start reason, bits 1 need, 2 gather, 4 foot off, 8 other off
  double al_dbg[2] = {0, 0};        // debug: support weight shares of the feet (L0)
  double dbg_raw[4] = {0, 0, 0, 0};  // debug: raw target x y, after step limit x, hip reach
  bool capped = false;
  int cross = 0;
  double head_ref = 0;  // step target frame heading from calm stance [rad]
  double lift_t0 = 0;       // start of the lift (phase 0) of the current step [s]
  double yaw_ref = 0;       // heading held during a recovery (pelvis yaw before the push) [rad]
  double squat = 0;          // stance knee squat added in the place phase (smoothed) [rad]
  double sw_T = 0;          // swing time of the current step (swing_t + length term) [s]; 0 = not set
  int urgent = 0;           // current step is urgent (capture point outside the support): no weight shift
  double td_time = -1;      // last touchdown time (double-support braking window)
  int td_foot = -1;         // foot of that touchdown
  int single = -1;          // stance foot index in single support (swing in the air phases), else -1
  double torso_tau[3] = {0, 0, 0};  // desired world torque on pelvis (single support, SIMBICON-style)
  double assist_f[2] = {0, 0};   // stay-upright pair force on the pelvis [N], horizontal; feet get -F
  double assist_w[2] = {0, 0};   // share of -F per foot (by load)
  double vf[3] = {0, 0, 0};  // virtual CoM force (horizontal) realised by stance-leg torques (Coros 2010)
  // Foot contact with hysteresis, updated every substep from normal forces.
  int foot_on[2] = {1, 1}, on_cnt[2] = {0, 0}, off_cnt[2] = {0, 0};
  double foot_fn[2] = {0, 0};
  double gz_st = 0;  // support height while not falling (L2: the stand floor of the tumble test)
  double gz = 0;  // support height under the feet [m, world z] (Perceive), held while no foot is on a support
  int foot_sg[2] = {-1, -1};  // support geom under each foot (largest normal force this substep), -1 none
  double foot_sv[2][2] = {{0, 0}, {0, 0}};  // velocity of the support under each foot (world x, y) [m/s]
  double foot_fh[2][2] = {{0, 0}, {0, 0}};  // floor horizontal force on the foot (world x, y), 25 ms low-pass [N]
  double cop_sh[2] = {0, 0};  // L1 -> L0 (cop_k): support point shift by the balance numbers [m, world]
  // Foot load share: measured floor normal force / body weight, low-passed 25 ms, per foot; one
  // hysteresis: loaded above kLoadOn, free below kLoadOff. The single "foot is support" state used everywhere.
  double ld[2] = {0.5, 0.5};
  int loaded[2] = {1, 1};
  double step_tgt[2] = {0, 0}, step_start[2] = {0, 0}, step_t0 = 0;
  // Leg balancer (l1_balance.cpp). swing: stepping foot or -1. Swing start: sole centre, distance to the
  // place, start time; whether the foot started in the air and whether the other foot was farther from its place.
  double sw_p0[2] = {0, 0}, sw_far0 = 0, sw_prev = 0, sw_t0b = 0;
  int sw_cm = 0;                         // sw_commit: the swing place is frozen in the world
  double sw_cmp[2] = {0, 0};             // ... at this sole-centre point
  int sw_air0 = 0, sw_ofar = 0;
  double dbg_lres = 0;           // debug: hip reserve of the swing [rad]
  int on_prev[2] = {1, 1};  // balancer contact of the last frame (3.1 height margin with hysteresis)
  int sw_reach = 0;      // this swing stalled out of reach; the desired pelvis stays low to its end
  double sw_short = 0;   // the swing foot's place lies this far beyond the reach this frame [m]
  double sw_rem = 0;             // swing: path left along the step line last frame (3.6 height) [m]
  int sw_first = 0;              // the swing is the first step from calm (give-up counts its place)
  double sw_tg[2] = {0, 0};      // swing sole target of the last frame (after reach) [m]
  double sw_min = 0, sw_stall = 0;       // closest distance to the place so far; time without getting closer [s]
  double place[2][2] = {{0, 0}, {0, 0}};  // current foot places (sole centres) [m]
  double pt[2] = {0, 0};                  // prediction point [m]
  double cor_i[2] = {0, 0};               // integrated CoM error (fore-aft, lateral), two feet only [m s]
  double pitch_acc = 0;                   // pelvis pitch accumulated on one leg [rad]
  double ch_prev = 0, ch_add = 0;         // chest tilt back along the heading last frame; chest hold pitch [rad]
  double ch_vpk = 0;                      // peak backward CoM speed of this recovery [m/s]
  int ch_hi = 0, ch_ok = 0;               // the hit threw the chest back (laws off) / moved the hips (laws on)
  double ch_t0 = 0;                       // chest tilt back at the hit [rad]
  double lh_add = 0;                      // pelvis forward pitch added after a push [rad]
  double ch_w = 1;                        // weight of the push-only trunk laws (fades out as the chest falls back)
  double budget = 0;                      // steps left (step_budget > 0)
  double yaw_hold = 0;                    // pelvis heading held through a step [rad]
  double mid_c[2] = {0, 0};               // commanded support middle, moved at a limited speed [m]
  int mid_c_on = 0;
  double foot_yaw0[2] = {0, 0};           // foot heading when it began standing (anti-spin torque) [rad]
  int foot_st[2] = {0, 0};
  double cmd_ret_t0 = -1;                 // commanded step landed: the middle returns between the feet from here
  double tq_prev[J_COUNT] = {};           // last frame targets (target velocity)
  double kf_q[2] = {}, kf_v[2] = {};      // knee target filter: angle, speed
  int kf_on = 0, kf_fr[2] = {0, 0};       // knee target filter started; leg free in the last frame
  double kq_prev[2] = {};                 // knee angles of the last frame (follower restart speed)
  double act_tq[J_COUNT] = {};           // leg drive torques after the activation lag (act_t)
  int hpd_lock = 0;                      // hip_pd off until rest: this recovery was faster than hip_pd_v
  double hpd_sp = 0;                     // perceived CoM speed last frame (hip_pd_v one-frame extrapolation)
  double hq_prev[2] = {};                // hip flexion angles of the last frame (hip_pd)
  double arm[J_COUNT] = {};              // leg joint armature set by L1 (0: model default), L0 applies it
  double limf[J_COUNT] = {};             // muscle limit factor from the flywheel (0 = 1), L0 applies it
  double sw_ft[3] = {};                  // swing foot frame target (ankle x, y, z) of the last frame
  double hd_prev = 0;                     // desired pelvis height of the last frame (rise limit)
  int hd_on = 0;
  double ank_prev[2][3] = {};             // ankle positions of the last frame (foot velocity)
  double qd_tgt[J_COUNT] = {};            // target velocity for the L0 feedforward [rad/s]
  double zm[J_COUNT] = {};                // damping multiplier for L0
  int stand_leg[2] = {1, 1};              // leg carries weight (not the swing leg), for L0
  int legs_pass = -1;                     // leg-leg contacts off (1) or on (0); -1 = not set yet
  double td_err[2] = {0, 0};              // last step end: sole centre minus place, along / across the heading [m]
  int td_cnt = 0;                         // step ends so far (bench: placement error per step)
  double cor_dbg[4] = {0, 0, 0, 0};       // debug: balance numbers uf, ul [m], ankle parts ank_f, ank_l [rad]
  // Swing measures for the bench: running in a swing, stored at its end.
  double sw_omax = 0, sw_vpk = 0, sw_ymin = 0, sw_ymax = 0, sw_y0 = 0, sw_ps[2] = {0, 0};
  double td_over = 0;  // largest excursion of the sole centre past the place along the swing path [m]
  double td_vpk = 0;   // peak horizontal sole speed [m/s]
  double td_dur = 0;   // swing time [s]
  double td_len = 0;   // step length: sole centre from swing start to the end [m]
  double td_yaw = 0;   // pelvis yaw range between lift-off and touchdown [rad]
  double tp_prev[J_COUNT] = {};  // swing leg joint angles that put the foot on its place, last frame (target velocity)
  int tp_leg = -1;               // leg of tp_prev, -1 = none (new swing)
  double rest_half = 0;          // half the lateral distance of the sole centres in the rest stance [m] (at reset)
  double net_dbg[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // debug: net hip flex, hip abd, knee torques, knee limits (L, R) [N m]
  // L2 state (l2_reactions.cpp).
  int wound_body = -1;
  double wound_until = -1, flinch = 0, flinch_t0 = 0;
  int no_react = 0;          // bench: relaxed body without protective reactions (baseline)
  double air_t = 0, land_t = -1;
  double land_v = 0, land_c0 = 0, land_d = 0;
  double land_z0 = 0, land_cr = 0, land_hard_t = -1;  // landing: hip-mid rise height, last crouch, last hard touch
  int air_curl = 0;  // curl in the air (high drop or not feet first), weight
  int air_roll = 0;   // flight: pelvis target leans back for a roll landing (set by L2, read by L0)
  double air_cw = 0;  // landing: touch speed [m/s], crouch at the touch and depth [m]
  int airborne = 0;
  // Flight flail: phases [rad] and rate/amplitude noise states of the four limbs (arm L, R, leg L, R), a
  // xorshift state for the noise and the latched "wants to curl" flag
  double fl_ph[4] = {0, 0, 0, 0}, fl_rate[4] = {0, 0, 0, 0}, fl_amp[4] = {0, 0, 0, 0};
  unsigned fl_rng = 0;
  int air_want = 0;
  int curled = 0;        // a curl (air, roll, wounded) was held: nothing straightens the body until it stands again
  double air_still = -1;  // time the curled body became slow (tone fades from it)
  int fall_set = 0;
  // Fall catch (l2_reactions.cpp): global strength from the body speed, per-arm strength, arm IK warm start,
  // drawn elbow limit for backward falls, hand place at the touch
  double catch_g = 0, arm_a[2] = {0, 0}, fall_aq[2][4] = {{0, 0, 0, 0.3}, {0, 0, 0, 0.3}}, fall_el[2] = {0.3, 0.3};
  double hand_at[2][3] = {{0, 0, 0}, {0, 0, 0}};
  int hand_was[2] = {0, 0};
  double hand_t[2] = {-1, -1}, hand_q[2][3] = {};  // palm touch time, shoulder flex, abd and elbow then
  double roll_w = 0, roll_a = 0;  // roll curl weight and the roll angle run [rad]
  int tumble = 0;  // tumbling down a level change (stairs), latched while the body moves
  double relax_t0 = -1;  // last frame of the tumble (the lying tone fade counts from here)
  // EMIL_TRACE_TUMBLE: last tumble test (time it ran; fall, mode, moving, lowest foot over the stand floor,
  // threshold, ray ground, gz, gz_st, lying, ground contacts, their height spread, level, state 0 off 1 on 2 ended)
  double tt_t = -1, tt[13] = {};
  int fall_pred = 0;         // fall judged unavoidable (balancer mode 2 or a fast drop outside the support)
  double fall_seen = -1;     // last time the fall judgement held
  double fall_tp = 0, fall_delay = 0.12;  // time of that judgement, protective reaction latency [s]
  double head_turn = 1;      // side the head turns to in forward falls (+1 left, -1 right)
  double curl_s = 0;         // curl ramp 0..1
  double fall_t0 = 0, lie_t0 = -1, fall_fwd = 0, fall_lat = 0;
  // Pending external impulse (world force), applied over one physics substep.
  int push_body = -1;
  double push_f[3] = {0, 0, 0}, push_p[3] = {0, 0, 0};
  double push_loc[3] = {0, 0, 0};  // hit point in the hit body's frame
  int push_n = 0;                  // substeps of the push force still to apply
  // Upper-body layer (l1_upper.cpp): last hit (push or bullet) and the layer state.
  double hit_t = -1, hit_dv = 0;            // time of the last hit, its velocity change of the whole body [m/s]
  int hit_bullet = 0;                       // the last hit was a bullet (deeper, longer looseness)
  double hit_p[3] = {0, 0, 0}, hit_dir[3] = {0, 0, 0};  // hit point and unit direction (world)
  double up_spread = 0, up_prev_p = 0, up_prev_r = 0, up_prev_tilt = 0;  // arms-out state, chest tilt last step
  double gsup[J_COUNT];  // L0 weight-support share per joint (1 = full; the upper layer lowers it for the arms)
  double up_still = 0;  // stillness 0..1 (upper layer: couplings to the hips fade standing still)
  double up_base_p = 0, up_base_r = 0;      // slow average of the chest tilt
  int up_init = 0;
  double up_yaw_ref = 0;  // yaw damping target heading
  int up_yaw_init = 0;
  double gaze_t0 = -1, gaze_t1 = -1, gaze_pt[3] = {0, 0, 0};  // look at the pusher from t0 to t1
  double look_t = 0, neck_yaw = 0, neck_pitch = 0;
  double neck_tau0[3] = {0, 0, 0};  // body neck strength, captured once (the layer scales it)
  int look_on = 0;
  // Bullet reaction (l2_reactions.cpp; arm and gaze in l1_upper.cpp). Live until bul_until; each bullet extends it.
  double bul_until = -1, bul_t = -1;        // reaction alive until, time of the last bullet [s]
  int bul_n = 0;                            // bullets in this live reaction
  int bul_body = -1;                        // body of the last bullet
  double bul_c[3] = {0, 0, 0};              // wound point in the chest frame of the moment of the hit [m]
  double bul_b[3] = {0, 0, 0};              // wound point in the frame of the hit body (leg wounds follow the leg) [m]
  double bul_dir[2] = {0, 0};               // horizontal bullet direction (world, unit)
  double lean_until = -1, lean_mag = 0;     // lean along the bullet (2nd and later bullets)
  double hold_lean = 0;  // first-bullet CoM target offset against the bullet [m] (ramped)
  double health = 100;  // health rules: pistol 15, head 0; < 30 wounded (falls), <= 0 dead
  int down = 0;         // 0 alive, 1 wounded (muscles fade in kHurtFade), 2 dead (kDeadFade); both fall without steps
  double crouch = 0;                        // current crouch [m] (ramped toward the bullet crouch)
  double twist_w = 0;                       // chest yaw toward the wound [rad] (decays with the flinch)
  double arm_t0 = -1, arm_t1 = -1;          // hand to the wound from t0 to t1 [s]
  int arm_side = -1;                        // arm that goes to the wound
  double arm_q[4] = {0, 0, 0, 0.3};         // its IK solution (flex, abd, rot, elbow), warm start
  double arm_q2[4] = {0, 0, 0, 0.3};        // the other arm (both hands to a trunk or leg wound)
  int arm_both = 0, arm_top = 0;            // both hands; which one lies on top (0 = arm_side)
  double pull_until = -1, pull_dir[2] = {0, 0};  // bullet pull on the pelvis: end time, direction
  double pull_k = 1;  // scale of the pull force and its lean (1 bullet pull; leg_pull share while hobbling)
  double pull_t0 = -1;  // start of the present pull episode (ramp-in from here) [s]
  double swing_ph[2] = {0, 0}, swing_el[2] = {0.5, 0.5};  // free-arm swing phase [rad], elbow bend [rad]
  // Moving support: standing-feet speed envelope [m/s], brace level 0..1, support velocity (world) [m/s]
  double deck_v = 0, brace = 0, sup_v[2] = {0, 0}, dk_prev[2][3] = {{0, 0, 0}, {0, 0, 0}};
  double deck_vl = 0, brace_c = 0;  // sideways support speed envelope [m/s]; crouch share of the brace
  int dk_ok = 0;
  double arm_hit_t[2] = {-1, -1};     // time of the last bullet in this arm (it hangs after 1 s)
  double curl_w = 0;                  // wounded curl ramp 0..1
  double sit_t0 = -1, sit_share[2] = {1, 1};  // backward-fall seat: start [s], drawn share per leg
  double sit_touch = -1, sit_h[2] = {1, 0}, sit_tp = -10;  // seat: pelvis touch [s], fall heading at its start
  double up_dbg[3] = {0, 0, 0};  // bench trace: left shoulder target parts (l1_upper)
  double quiet_t = 0, neck_quiet = 0;  // lying: last time the CoM moved, neck release 0..1
  int down_nostep = 0;                      // 1 = no steps while down (head bullet, kill button)
  double whip_a = 1, whip_ts = 1;           // drawn scale of the trunk whip angles and times (+-20 %)
  int look_wound = 0;                       // gaze: 1 at the wound, 0 at the shooter
  int leg_hits[2] = {0, 0};                 // bullets per leg
  double hob_t = -1;                        // time of the last hobble step command
  int hob_n = 0;                            // hobble: commanded side steps so far (even: wounded foot, odd: healthy)
  double hob_vg[2] = {0, 0};                // hobble velocity goal (L2 leg block; Avoid adds it to the dodge goal)
  double hob_dir[2] = {0, 0};               // direction of the hobble goal, fixed at its start (0 = not set)
  double leg_t[2] = {-1, -1};               // time of the last leg bullet
  double leg_t0[2] = {-1, -1};              // time of the first leg bullet, give-way time
  double leg_give[2] = {-1, -1};
  double leg_lean[2] = {0, 0};              // CoM shift to the wounded side added to in.lean last frame
  int leg_shin[2] = {0, 0};                 // the first bullet hit the shin or foot (slower)
  double dead_t = -1, dead_T = 1;           // down (wounded or dead): start, time for the muscles to fade [s]
  double dead_walk_t = -1;                  // wounded: time the steps under the fading strength ended
  double car_seen_t = -1;                   // the car was seen coming (arms go toward it after the reaction time)
  double dead_land_t = -1;                  // dead: chest or head first on the floor (the curl frees the hips)
  double arm_tone = 0;                      // wounded going down forward: floor of the arm tone (hands catch)
  double leg_tone = 0;                      // dead: leg tone floor (hips, knees, ankles) while the pelvis is high
  double bsit_t = -1;                       // going down backward fast: start of the roll pose (trunk flexed)
  double prop_t = -1;                       // propped (knees, pelvis high) dead body: start of the hip opening
  double dead_quiet_t = -1;                 // last time the dead or wounded CoM was faster than kQuietV
  double dead_arm_lag = 0;                 // drawn delay of the second arm (+ left late, - right) [s]
  double dead_lay_t = -1;                   // after the relaxation: time the body went quiet (curl lets go)
  // Gaze, dodge, flinch (l2_avoid.cpp; the neck and trunk targets in l1_upper.cpp)
  double car_gaze_t0 = -1;                  // the head starts to turn to the car (seen + 0.15..0.25 s), -1 = not seen
  int gaze_new = 0;                         // the point gaze (gaze_pt) is a missed shot: the trunk may help
  double gaze_trunk = 0;                    // trunk yaw following a large gaze turn [rad]
  double miss_t = -1, miss_side = 0, miss_w = 0;  // a bullet missed close: time, side of the pass, closeness 0..1
  uint64_t rng2 = 0x2545F4914F6CDD1Dull;    // own random stream (the main one keeps its sequence)
  int dodge = 0;                            // 0 not decided, 1 stepping out of the car's path, 2 no dodge
  double dodge_dir[2] = {0, 0};             // world direction (unit) of the dodge
  double dodge_t0 = -1;                     // time of the dodge decision
  int dodge_fa = 0;                         // escape along the heading at the decision (side car)
  double dg_lean[2] = {0, 0};               // dodge lean added to in.lean this frame
  double dive_t0 = -1;                      // dive start (legs tucked up out of the car path)
  int thr_on = 0, thr_f = -1;               // hurried dodge throw on; escape-side (lunge) foot
  double dodge_tc = 9;                      // time left to the hit at the dodge decision [s]
  double dodge_urg = 0;                     // urgency of the dodge at its decision: 0 plenty of time .. 1 hurried
  double dodge_lean[2] = {0, 0};            // the lean this reaction added to in.lean last frame
};

class World {
 public:
  ~World();
  // Build one model for all bots (placed along x every `spacing` m). Returns false on error (msg in err).
  bool Build(const std::vector<BodyParams>& bodies, double spacing, uint64_t seed, char* err, int err_sz);
  void Reset();
  // Advance one 60 Hz frame: control step + physics substeps.
  void StepFrame();
  // World hit: horizontal impulse [N s] at world point on a body of bot i (applied over the next substep).
  // The standard push point: 0.34 m (x height / 1.78) straight up from the midpoint of the
  // two hip joints, on the nearest trunk body (pelvis, lumbar, chest by CoM distance). Returns that body
  // (bot-relative).
  int PushPoint(int bot, double pt[3]) const;
  // A push of impulse imp [N s, world] at the standard point, or with param push_old at old_pt on old_body
  // (bot-relative body id), the legacy point of each bench.
  void PushAt(int bot, const double imp[3], const double old_pt[3], int old_body);
  void ApplyImpulse(int bot, int body_id, const double point[3], const double impulse[3]);
  // Step command from calm stance through the normal step machine: foot 0 left / 1 right, dx along the pelvis
  // heading, dy outward from the foot's side (negative = inward); cross = step across in front of the other foot.
  void CommandStep(int bot, int foot, double dx, double dy, bool cross);
  // Weight onto foot (0 left, 1 right); lift_s > 0: then lift the other foot 5 cm for lift_s seconds and put it down.
  void CommandShift(int bot, int foot, double lift_s);

  // Editor / game commands (world events only; no forces on the body besides contacts and hits).
  void DropFrom(double h, double tilt = 0, double tilt_dir = 0, double spin = 0);
  // reset all bots raised by h [m]; optional start tilted by tilt [rad] about the horizontal axis at heading tilt_dir
  // and turning at spin [rad per s] about it (edge fall)
  // car box at bot: side +1 from left, -1 right, 0 front; start = distance of the car centre from the pelvis [m]
  void Car(int bot, double speed, int side, double start = 4.0);
  // A bullet passed the bot without a hit: origin and direction of the shot (world). Close to the head it makes a
  // flinch and the head turns to the shooter. Called by the game when its ray found no body of this bot.
  void Miss(int bot, const double origin[3], const double dir[3]);
  // Swaying deck (with_platform): target amplitudes along world x and y [m] and period [s]; the amplitude
  // moves to the target at most one target-size per period (smooth steps), phase runs on. Zero amplitudes: deck rests.
  void Platform(double amp_x, double amp_y, double period);
  void PlatformPos(double* x, double* y) const;  // present deck offset [m]
  // Suspension bridge (with_bridge): sideways wind (world y) on the six middle links, sine of the period,
  // amplitude per link [N]; a new amplitude is reached over one period. BridgeMid: the middle joint point (top of
  // the deck line at the junction of the halves) [m, world].
  void Bridge(double amp, double period);
  void BridgeMid(double* p) const;
  // Bullet: horizontal impulse [N s] at point on body (BodyId) along dir, damage to the health, wound reaction (L2).
  // Defaults: impulse param bul_imp (pistol), damage 15 of the health 100.
  void Shoot(int bot, int body, const double point[3], const double dir[3], double impulse = -1.0,
             double damage = 15.0);
  void Curl(int bot, int on);
  void Kill(int bot);  // death (editor command, head bullet, health <= 0): muscles fade in kDeadFade, the body falls
  // Ray against bot geoms (floor excluded). Returns bot index or -1; fills body id (BodyId) and hit point.
  int Ray(const double p[3], const double dir[3], int* body, double hit[3]) const;
  // Virtual CoM force -> joint torque for leg joint j (J^T, ankle capped by foot support). Used by Muscles, selftest.
  double VfTorque(const Bot& b, int j, const double F[3], bool closed) const;
  std::string tune_opt, tune_flag, tune_geom;  // bench only, before Build: extra <option>, <flag>, <geom> attributes
  bool with_car = false;  // set before Build: add a 4.5 x 1.8 x 1.4 m car box (two slide joints, 1500 kg)
  // Set before Build: a kinematic deck (plat_half x plat_half m, top at z = 0, two slide joints) under the
  // bot 0 position; the world floor plane goes down to -kPlatH so the layers keep "floor = z 0" on the deck.
  bool with_platform = false;
  double plat_half = 1.0;
  static constexpr double kPlatH = 0.2;  // deck top above the floor [m] (stand height of the bridge scene)
  // Bridge scene: two halves of 8 links (0.51 x 1.0 x 0.05 m boards, 2.5 kg) hung from the banks 8 m
  // apart, each half 2 cm longer than half the span (sag ~0.35 m), middles joined at both board edges. Links: hinge
  // across (bend, damping 15 N m s) and about the vertical (sway, 400 N m/rad, damping 15, +-0.15 rad); no twist.
  // Along world x, bot 0 on the middle; the junction is at z = 0 in the V shape of a point load, the floor 6 m below
  // the banks.
  bool with_bridge = false;
  // Stairs scene: a flight of kStairN risers (kStairRise x kStairRun, kStairW wide, solid blocks)
  // going down along world +x from a kStairLand m landing, walls kStairWall above the treads on both sides. The
  // landing top is z = 0 (layers keep "floor = z 0" there), its edge kStairEdge ahead of bot 0, the floor plane at
  // the foot of the stairs (z = -kStairN * kStairRise). Stair geoms get the floor friction (body_mu).
  bool with_stairs = false;
  static constexpr int kStairN = 36;
  static constexpr double kStairW = 2.0, kStairWall = 1.2, kStairLand = 3.3, kStairEdge = 0.3;
  // Riser x tread set per world (default 26.6 deg; steeper stairs 0.18 x 0.28 = 32.7 deg)
  double stair_rise = 0.15, stair_run = 0.30;
  bool stairs_body = false;  // bench: the stairs inside a static body (no joint), as an editor scene may build them
  bool stairs_slant = false;  // bench: editor-style walls (landing boxes, one slanted box per side of the flight)
  double StairsBottomX() const { return kStairEdge + (kStairN - 1) * stair_run; }  // where the floor starts [m]

  // The whole scene (supports, bots, car) built lift m higher, the floor plane too; with floor0 (env
  // EMIL_FLOOR0=1) lift = the support's depth, so the floor is at z 0 and the deck / landing high above it.
  double lift = 0;
  bool floor0 = false;
  double FloorZ() const {  // floor plane height [m]
    return lift + (with_platform ? -kPlatH : with_bridge ? -5.625 : with_stairs ? -kStairN * stair_rise : 0.0);
  }

  int NumBots() const { return (int)bots_.size(); }
  // The ground: any static geom (the floor plane, stairs and their walls); also geoms of static bodies
  // (welded to the world: no joint), as an editor scene may build its stairs
  bool Ground(int g) const { return m_->body_weldid[m_->geom_bodyid[g]] == 0; }
  Bot& bot(int i) { return bots_[i]; }
  const mjModel* model() const { return m_; }
  mjData* data() { return d_; }
  double time() const { return d_ ? d_->time : 0; }
  float param[P_COUNT];

  static constexpr double kFrameDt = 1.0 / 60.0;
  static constexpr int kSubsteps = 8;  // physics dt = 1/480 s
  // Response inertia 1/(M^-1)_jj of dof (every other joint free, no contacts) from the factor of the last
  // substep, flywheel included. A muscle with three numbers (w, z, stf) uses it: kp = w^2 Im, damping 2 z w Im, and
  // its flywheel stf x (M_jj - armature) x (w^2 h^2 + 2 z w h) goes to Bot::arm, 1 + stf (w^2 h^2 + 2 z w h) to
  // Bot::limf; L0 applies both to any joint (legs: L1 balance, others: whoever sets the gains).
  double RespInertia(int dof);

 private:
  double body_mu_set_ = -1;  // last body_mu written to the model
  void Perceive(Bot& b);           // L3: fill b.pc[b.cur]
  void Balance(Bot& b, double t);  // L1: targets from delayed percept
  void Muscles(Bot& b);            // L0: PD + gravity compensation -> ctrl (current state)
  void Upper(Bot& b, double t);     // L1 upper body: back, neck, arms after Balance, before Reactions
  void DeadArms(Bot& b, double t, int mode);  // arms of a dying body (l1_upper.cpp): 0 face, 1 spread
  void Reactions(Bot& b, double t); // L2: inputs, targets and stiffness after Balance
  void AnkleRange(Bot& b);          // active / passive ankle dorsiflexion limit
  void Avoid(Bot& b, double t);     // L2: car dodge, bullet-miss flinch, car gaze state (after Reactions)
  void CarSubstep();                // kinematic car velocity (braking after the hit)
  double SupportTop(int g, double x, double y) const;  // top of a support geom at (x, y) [m, world z]
  // highest support top under (x, y) below z + 5 cm; *over = a support there rises above z + 5 cm
  double GroundBelow(double x, double y, double z, bool* over = nullptr) const;
  void TraceTumble(const Bot& b, double t);  // EMIL_TRACE_TUMBLE stderr trace (bot 0)
  void FootContactsAll();          // L3: one contact pass -> foot forces, support velocities
  void PlatformSubstep();           // deck position and velocity prescribed every substep
  void BridgeSubstep();             // wind on the bridge links
  void FootContacts(Bot& b);       // L3: per-substep hysteresis -> foot_on (after FootContactsAll)
  void LegContacts(Bot& b, bool pass);  // leg-leg contacts of one bot off (standing, stepping) or on (lying)
  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
  std::vector<Bot> bots_;
  std::vector<int> foot_bot_;   // per geom: 2 * bot + foot, or -1 (FootContactsAll)
  std::vector<double> foot_force_;  // per bot and foot: fn, fh x, y, support v x, y (fn-weighted sums), 10 per bot
  std::vector<int> sup_geoms_;    // geoms not of a bot or the car (floor, stairs, decks): GroundBelow
  std::vector<char> geom_sup_;    // per geom, 1 = in sup_geoms_ (support height source)
  std::vector<int> stair_geoms_;  // stair and wall geoms (floor friction)
  std::vector<unsigned char> geom_edge_;  // per geom, horizontal axes that end in an edge (bit 0 local x, 1 y)
  std::vector<double> foot_sgf_;  // per bot and foot, normal force of the best support geom this substep
  std::vector<int> foot_sgi_;     // that geom, -1 none
  std::vector<double> mi_z_;  // scratch (nv) for the leg joints' response inertia, zero between uses
  std::vector<double> ctrl0_;  // motor range of the model (4x the table limit), per actuator
  int car_qadr_ = -1, car_dadr_ = -1, car_body_ = -1, car_hit_ = 0;
  double car_speed_ = 0, car_dir_[2] = {0, 0};
  int plat_qadr_ = -1, plat_dadr_ = -1;
  double plat_amp_[2] = {0, 0}, plat_tgt_[2] = {0, 0}, plat_rate_[2] = {0, 0}, plat_w_ = 0, plat_ph_ = 0;
  int br_wind_[6] = {-1, -1, -1, -1, -1, -1}, br_mid_ = -1;
  double br_amp_ = 0, br_tgt_ = 0, br_rate_ = 0, br_w_ = 0, br_ph_ = 0;
};

}  // namespace emil
