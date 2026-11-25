// EMIL tunable parameters: one table (name, default, min, max, description) shared by bench and editor.
// Lengths are given for a 1.78 m body and scaled by height; times of the leg balancer scale with sqrt(height).
#pragma once

namespace emil {
constexpr double kLoadOn = 0.4;   // foot load share above which a foot is loaded (support, cannot lift)
constexpr double kLoadOff = 0.1;  // ... below which it is free
constexpr double kTouch = 0.03;   // leg balancer and support torques: the foot touches the floor above this load share

enum ParamId {
  // L0 muscles
  P_STIFF_SCALE,      // global multiplier on all joint stiffness
  P_DAMP_RATIO,       // kd = kp * ratio [s] (muscle response time)
  P_ZMP_FRAC,         // loaded foot: ankle torque within this share of the foot lever (CoP inside the sole)
  P_KNEE_WEAK,        // standing knee: share of strength lost when deeply bent
  P_FF_MAX,           // target velocity feedforward limit [rad/s]
  P_SUP_LINE,         // 1: support force along CoP -> CoM; 0: vertical at the CoP (bench switch)
  // Reset pose
  P_KNEE_BEND,        // reset knee flexion [rad]
  P_ANKLE_DORSI,      // reset ankle dorsiflexion [rad]
  P_TOE_OUT,          // toe-out per foot [rad]
  P_STANCE_W,         // reset ankle separation [m]
  // L1 leg balancer
  P_LEG_K,            // standing leg stiffness multiplier
  P_SWING_K,          // swing leg hip and knee stiffness multiplier
  P_SWING_Z,          // swing leg damping multiplier (relative to critical-like base)
  P_STEP_H,           // step height [m]
  P_STEP_LEN,         // step length multiplier (reach ellipse)
  P_GIVE_UP,          // give up when the CoM is this far from the nearest foot, per metre of CoM height
  P_LOW,              // give up when the CoM is below this share of its standing height
  P_STEP_BUDGET,      // steps allowed before giving up (0 = unlimited)
  P_LEAD_T,           // prediction point: CoM + velocity * lead_t [s]
  P_HALF_W,           // foot place half width as a share of the rest stance half width (sole centres)
  P_MARGIN,           // prediction point this far outside the support quad -> step [foot widths]
  P_APART,            // feet farther apart than the place width + apart rest stances -> gathering step
  P_PLACE_FWD,        // fore-aft place correction by the direction other foot -> prediction point [m]
  P_CARROT,           // swing: frame target at most this far from the foot [m]
  P_OVER,             // swing: aim this far past the place [m]
  P_LIFT0,            // swing from the ground: minimum lift until near the place [m]
  P_MID_T,            // in a step the support middle is CoM + velocity * mid_t [s]
  P_LEG_FRAC,         // compass: standing leg length as a share of the full leg
  P_DROP_MAX,         // compass: pelvis drop limit [m]
  P_COM_FWD,          // posture: CoM ahead of the feet middle on two feet [m]
  P_COR_TD,           // balance numbers: velocity weight [s]
  P_PLACE_TD,         // standing: desired pelvis place shifted against the CoM velocity by this time [s]
  P_COR_KI,           // balance numbers: integral weight [1/s] (two feet only)
  P_ANK_FB,           // fore-aft number -> standing ankle angle [rad/m]
  P_ANK_LAT,          // lateral number -> ankle inversion [rad/m]
  P_LEN_LAT,          // lateral number -> leg length difference [m/m]
  P_ROLL_LAT,         // lateral number -> pelvis roll [rad/m]
  P_SPINE_LAT,        // lateral number -> spine roll [rad/m]
  P_PITCH_V,          // one leg: forward pelvis pitch per backward CoM speed [rad per m/s] (3.7)
  P_PITCH_MAX,        // one-leg counter-lean limit [rad]
  P_PITCH_V2,         // one-leg counter-lean: extra slope above 1.6 m/s backward CoM speed [rad per m/s]
  P_SWING_OVER,       // swing: joint targets pulled past the IK angles at full lift (share of the error)
  P_TOE_LAND,         // backward step: swing toe down [rad]
  P_SWAY_AMP,         // support middle sway amplitude (liveliness) [m]
  // World / L2
  P_FOOT_TORSION,     // foot-floor torsional friction [m] (torque = coefficient x normal force), 0 = none (condim 3)
  P_FALL_LEGS,        // leg stiffness scale kept while falling forward or backward (0 = limp legs)
  P_FALL_REACT,       // 1 = early fall judgement + arm and head protection (0 = reaction only after give-up)
  P_NOSLIP,           // MuJoCo noslip iterations: 0 = off, 1-3 = stick
  P_ASSIST,           // "stay upright" help: pelvis <-> loaded feet pair force, share of M omega^2 (0 = off)
  P_SWING_PV,         // swing preview: 0 hip moves with the CoM velocity; 1 hip velocity + pelvis rotation (3.9)
  // Upper-body layer (core/l1_upper.cpp)
  P_UPPER, P_UP_BACK, P_UP_ARM, P_UP_SWING, P_UP_TWIST, P_UP_SPREAD, P_UP_LOOSE, P_UP_LOOSE_T, P_UP_GAZE,
  P_UP_YAW_FB, P_BUL_LEAN, P_UP_LOOSE_YAW, P_UP_ARM_FB, P_BUL_CROUCH, P_UP_LOOSE_BACK,  // bullet crouch in joint space
  P_UP_SWAY, P_UP_ARM_D, P_BUL_HOLD, P_UP_LOOSE_BP,  // upper-body sway, arm damping, bullet hold
  // upper-body muscle frequencies and damping ratios
  P_UP_W_ARM, P_UP_Z_ARM, P_UP_W_ELB, P_UP_Z_ELB, P_UP_W_NECK, P_UP_Z_NECK, P_UP_W_BACK, P_UP_Z_BACK,
  P_UP_GSUP, P_UP_REST0, P_UP_REST1, P_UP_W_ABD, P_PUSH_T, P_UP_CW, P_UP_NECK_STR, P_UP_W_ROLL,
  P_IMMORTAL, P_UP_GSUP_BUL,  // no health loss; arm weight support under bullets
  // bullet pull, crouch, whip, hands, help, step reserve
  P_BUL_PULL, P_BUL_PULL_T0, P_BUL_PULL_T1, P_BUL_DEAD_NOSTEP, P_BUL_CROUCH_T, P_BUL_WHIP, P_BUL_HANDS, P_BUL_ASSIST, P_BUL_STEPS,
  P_BUL_SWING, P_BUL_STICK,  // bullet arm swing, sticky palms
  // push force limit, backward-fall seat
  P_PUSH_FMAX, P_FALL_SIT, P_FALL_SIT_LEGS, P_FALL_SIT_OUT, P_FALL_SIT_SHARE, P_FALL_SIT_SWING, P_FALL_SIT_KNEE_T, P_FALL_SIT_GSUP, P_FALL_SIT_ROLL,
  P_GU_FIRST,         // give-up: 1 the first step from calm counts at its (reach-clipped) target until touchdown
  P_PV_FAR,           // swing hip preview far from the place: 1 none (carrot taken relative to the hip), 0 full
  P_REACH_KEEP,       // reach: 1 from the pelvis height to keep (straight touchdown knee), 0 hip-angle ellipse
  P_GU_VZ,            // give up when the CoM falls faster than this during a swing [m/s]; 0 off
  P_GU_SIDE,          // give-up distance factor for a sideways fall (1 = as forward and back)
  P_FV,               // leg muscles force-velocity share (0 off, 1 full)
  P_FV_CON,           // share of the shortening (concentric) loss in fv
  P_STAND_CLIP,       // standing foot clipped by reach around the desired hip (3.9) (0 off, 1 on)
  P_STRUT,            // loaded standing hip flexion target within this of the present angle on one leg [rad]
  P_MID_BACK,         // share of the step support-middle shift dropped when backing off (cosine of the motion)
  P_FV_HIP,           // force-velocity also on hip flexion and abduction (0 off, 1 on)
  P_STALL_REACH,      // stalled swing lowers the desired pelvis until the swing leg reaches its place (0 off)
  P_ON_KEEP,          // a standing foot stays support while its ankle is this high above standing height [m]
  P_SWING_PVR,        // swing preview of the pelvis rotation only (hip at the CoM velocity) (0 off, 1 on)
  P_FOOT_DOWN,        // unloaded standing foot reaches the floor from the higher of desired / present hip (0 off)
  P_LEG_W_ST, P_LEG_Z_ST, P_LEG_W_SW, P_LEG_Z_SW,  // leg muscle frequency [rad/s] and damping ratio
  P_STEP_H_BACK,  // backward step height
  P_LEAN_TOP,  // one-leg counter-lean on top of the strut budget
  P_LEAN_DECAY,  // two-feet counter-lean fade time
  P_PLACE_BACK,  // backward place look-ahead share
  P_PLACE_BACK_D,  // backward place shortfall limit
  P_LEAN_SPINE,  // counter-lean share bent into the spine
  P_TURN_GO, P_TURN_V,  // held heading turns toward a sideways motion
  P_KN_ACC, P_KN_VST,  // knee target acceleration and straightening speed limits
  P_DROP_K, P_DROP_C0, P_DROP_UP, P_DROP_LIM,  // pelvis drop arc in a step
  P_CAR3D, P_SW_V, P_CAR_C, P_SW_HD,  // swing carrot in space
  // ankle torque window, pelvis hold, activation lag, toe stance, lift speed
  P_ZMP_FN, P_HIP_PD, P_ACT_T, P_ST_ANK, P_LIFT_V,
  P_HIP_PD_V,  // pelvis hold speed gate
  P_ARM, P_STF_HIP, P_STF_KST, P_STF_KSW, P_STF_AST, P_STF_AAIR,  // muscle stiffness (armature)
  P_ST_TWIST,  // standing hip twist limit
  P_MW_ST, P_MZ_ST, P_MW_SW, P_MZ_SW,  // leg muscle frequency / damping ratio (armature on)
  P_ASST_REF, P_ASST_T, P_ASST_N,  // alternative stay-upright help: pulled point, velocity look-ahead, tilt fade
  P_APART_FA,  // gathering step by the feet distance (fore-aft spread too)
  P_ST_YAWF,  // standing leg solved in the present-heading pelvis frame
  P_HIP_PD_FAST,  // share of hip_pd kept after a fast push (latched)
  P_ANK_FH,  // standing ankle torque window shifted by the floor's horizontal force at the ankle height
  P_SPIN_K,  // anti-spin stiffness of a standing foot (sole torsional stiffness)
  P_ST_INVLIM,  // standing leg solved for a hip position within the ankle inversion range (margin)
  P_SPIN_CAP,  // anti-spin cap multiplier (sole torsional friction)
  P_ZMP_FN_INV,  // ankle inversion torque window from the measured foot force
  P_ASST_YAW,  // the stay-upright pair force has no yaw moment
  P_ROLL_GUARD,  // a standing ankle does not push its sole further onto the edge it already tilts to
  P_ASST_ON,  // the stay-upright help's support middle counts the feet standing by the touch rule
  P_COP_K,  // balance numbers shift the support point of the weight-holding torques (ankle strategy)
  P_UP_W_TW, P_UP_Z_TW, P_UP_STF_TW,  // back twist muscle with three numbers
  P_UP_STF_BACK, P_UP_STF_NECK,  // back pitch and roll, neck: flywheel multipliers (0 = two numbers)
  P_UP_GUARD,  // arm in waiting during a stumble
  P_SW_PRESS_B,  // share of the below-floor swing height target kept on the backward part
  P_UP_STF_ARM, P_UP_STF_ELB,  // shoulders, elbows and wrists: flywheel multipliers (0 = two numbers)
  P_ST_TOE,  // an unloaded standing leg that cannot reach the floor keeps its toe down
  P_ASST_SPLIT,  // help pulls to the middle of both feet, reaction half on each foot
  P_TURN_BUL,  // no turn toward the motion while the bullet reaction lives
  P_ASST_COMP,  // the leg of an unloaded foot cancels its share of the help reaction
  P_ASST_BOLD,  // while a bullet reaction lives the help uses the loaded feet (rule before asst_split)
  P_HPD_BUL,  // the hip_pd fast-push lock only after bullet hits or impacts faster than this
  P_TD_NOW,  // a swing ends on touching only while the foot touches the floor now
  P_HPD_LD,  // hip_pd holds the pelvis only on a loaded leg
  P_SW_COMMIT,  // the swing place freezes in the world within this many carrot lengths
  P_HPD_TD,  // hip_pd spares the leg that touched down last for this long [s]
  P_ST_TOE_SIDE,  // st_toe only while the CoM moves more sideways than fore-aft
  P_PULL_LEAN,  // support offset along the bullet pull, share of bul_pull x CoM height
  P_BUL_BRAKE,  // one standing foot under a live bullet reaction: support point at the stopping point
  P_PULL_HELP,  // share of the pull offset for the help target alone
  P_BUL_CENTER,  // share of the position error in the help and fore-aft number under bullets
  P_RLX_LEAD,  // prediction look-ahead share while the wounded body relaxes on the move
  P_RLX_SWHD,  // swing leg from the desired height while relaxing
  P_RLX_GAP,  // desired pelvis at most this far below the hips while relaxing, no rise
  P_RLX_FF,  // standing knee target velocity = its target rate while relaxing (max rad/s)
  P_RLX_CAR,  // carrot + this x body travel toward the aim over the response time while relaxing
  P_RLX_LDIR,  // rlx_lead only for the backward part of the motion
  P_RLX_TRAIL,  // relaxing forward, a trailing standing leg reaches from at most this below the hip
  P_RLX_GU,  // no balancer give-up while the wounded body relaxes on the move
  P_RLX_TSINK,  // relaxing, the desired pelvis at most this x hs x (1 - tone) under the compass
  P_RLX_PITCH,  // relaxing, desired pelvis pitch forward x (1 - tone) [rad]
  P_RLX_M2,  // relaxing, the give-up keeps the leg stiffness (only the tone fades it)
  P_RLX_PULL,  // relaxing on the move, the pull lean share (instead of pull_lean)
  P_RLX_NOLEAN,  // no one-leg counter-lean while relaxing on the move
  P_BODY_MU,  // sliding friction of the floor and of every body part except the feet
  P_BRK_AFTER,   // bullet braking (bul_brake, bul_center) only after the pull ended
  P_PULL_RAMP, P_PULL_FADE,  // bullet pull ramp-in from the episode start, fade-out before its end
  P_PULL_COP,    // share of the pull rest offset added to the L0 support point
  P_BUL_DS,      // least double support after a touchdown while a bullet reaction lives
  P_BUL_REACH,   // in a bullet step the pelvis is low enough for a bent-knee touchdown
  P_BUL_YAWD,    // pelvis yaw rate damping of the desired heading while a bullet reaction lives
  P_LEG_HOB,     // hobble toward the wound after the give-way: side step length [m] (0 off)
  P_LEG_PULL,    // hobble pull toward the wound after the give-way: share of bul_pull (0 off)
  P_LEG_VG,      // hobble velocity goal toward the wound after the give-way [m/s at 1.78 m] (0 off)
  P_BUL_IMP,     // pistol bullet impulse [N s] (Shoot default)
  P_LAND_ROLL,   // medium landing (2.5-4.2 m): crouch, then curl and roll instead of the fall onto the hands
  P_AIR_ROT,     // flight torque on the pelvis towards feet down, upright; gain scale, 0 = off
  P_AIR_FLAIL,   // flight flail of arms and legs and a late curl before the touch (0: off)
  // brace on a moving support (deck), free-arm swing
  P_DECK, P_DECK_V0, P_DECK_V1, P_DECK_CROUCH, P_DECK_WIDE, P_DECK_ARM, P_DECK_CW, P_DECK_PITCH, P_ARM_SWING, P_DECK_REL, P_DECK_HOLD, P_ARM_SW_V0, P_DECK_STEP,
  P_DECK_EDGE, P_DECK_MID, P_DECK_LD,  // deck step place limits and load threshold
  P_WALK_END_T, P_FOLD_A, P_BUL_SIDE, P_BUL_GUARD,  // wounded walk end, death fold, trunk wound bend and guard
  P_GAZE_ADV, P_GAZE_TRUNK, P_DODGE,  // gaze at threats, trunk share of the gaze turn, car dodge
  P_DODGE_VG, P_DODGE_AB, P_VG_LEAD, P_VG_CAP,  // velocity goal: dodge speed, braking, first foot, rear lag cap
  // push-off: share of the body weight, duration; hurried lag cap, no crossing steps
  P_PO_F, P_PO_T, P_VG_CAPH, P_VG_NOX,
  P_FLINCH,  // startle after a close miss
  P_PULL_FREE,  // help ignores the bullet pull
  P_PITCH_GUARD, P_PULL_FREE_V,  // toe-up guard, pull_free speed limit
  // death arm and leg holds, pull scaled by tone, wounded walk speed limit
  P_DEAD_FWD_ARM, P_LEG_HOLD_BODY, P_PULL_TONE, P_DEAD_ARM_HOLD, P_WALK_VMAX,
  P_PROP_PEL, P_PROP_TONE, P_BSIT_V, P_BSIT_TONE,  // propped dead body, backward roll when going down
  P_ANKLE_ACT,  // active ankle dorsiflexion range
  P_FLAT_R,  // a tumble ends only on level ground
  P_PUSH_OLD,  // legacy push point switch
  // chest hold when backing off, support brake, push-specific lean and pelvis hold
  P_CH_HOLD, P_CH_THR, P_CH_TD, P_CH_MAX, P_CH_SPINE, P_SUP_BRAKE, P_SUP_BRAKE_MAX, P_LS_PUSH, P_HPD_FAST_PUSH,
  P_LH_PUSH,  // hip fold after a push
  P_HOB_EARLY, P_HOB_DS,  // hobble start and double support
  // hurried dodge: throw and dive
  P_THROW, P_THROW_TC, P_THROW_LEAD, P_THR_PO, P_DIVE_T, P_DIVE_ABD, P_THR_VG, P_THR_PT, P_DIVE_JUMP,
  P_THR_DS, P_THR_AHEAD, P_THR_CAP,  // hurried dodge stepping
  P_DODGE_LEAN, P_CAR_GUARD,  // calm dodge lean, car guard arms
  // dying and dodge arm options
  P_RLX_NOEXT, P_DEAD_BUCKLE, P_DEAD_KNEE_FIRST, P_HUR_SWING, P_GUARD_FRONT, P_DIVE_ARMS,
  // strength fade and leg support while going down
  P_CURL_FADE, P_CURL_NOSUP, P_HURT_FADE, P_DOWN_NOSUP, P_DOWN_SUPK, P_DEAD_UPRIGHT,
  P_CURL_CATCH, P_CURL_ARM_HOLD,  // catching arms in an early curl
  P_FALL_LEGS_FWD,  // leg stiffness while falling forward
  P_REACH_HIP,  // wounded leg reaches from its real hip height
  P_COUNT
};

struct ParamDef {
  const char* name;
  float value, min, max;
  const char* desc;
};

// Defaults. Leg balancer starting values follow the stepping model where the mechanism is
// the same (lead_t, half_w, margin, apart, step_h, carrot, over, lift0, leg_frac, give_up, low, leg_k, swing_k,
// swing_z); the rest from body geometry and tuning on the benches.
inline const ParamDef kParamDefs[P_COUNT] = {
    {"stiff_scale", 1.0f, 0.1f, 3.0f, "global joint stiffness multiplier"},
    {"damp_ratio", 0.08f, 0.01f, 0.3f, "joint damping = stiffness * ratio [s]"},
    {"zmp_frac", 0.8f, 0.2f, 1.0f, "loaded foot: ankle torque within this share of the foot lever (CoP inside the sole)"},
    {"knee_weak", 0.6f, 0.0f, 1.0f, "share of the standing knee strength lost when deeply bent"},
    {"ff_max", 3.0f, 0.0f, 20.0f, "target velocity feedforward limit [rad/s]"},
    {"sup_line", 1.0f, 0.0f, 1.0f, "support force direction: 1 along CoP -> CoM, 0 vertical at the CoP"},
    {"knee_bend", 0.21f, 0.0f, 0.6f, "reset knee flexion [rad]"},
    {"ankle_dorsi", 0.10f, -0.2f, 0.4f, "reset ankle dorsiflexion [rad]"},
    {"toe_out", 0.12f, 0.0f, 0.4f, "toe-out per foot [rad]"},
    {"stance_w", 0.28f, 0.15f, 0.35f, "reset ankle separation at 1.78 m [m]"},
    {"leg_k", 1.5f, 0.3f, 4.0f, "standing leg stiffness multiplier"},
    {"swing_k", 2.8f, 0.5f, 5.0f, "swing hip and knee stiffness multiplier"},
    {"swing_z", 0.72f, 0.2f, 2.0f, "swing leg damping multiplier"},
    {"step_h", 0.12f, 0.05f, 0.3f, "forward and sideways step height at 1.78 m [m]"},
    {"step_len", 1.0f, 0.5f, 1.5f, "step length multiplier (reach ellipse)"},
    {"give_up", 0.6f, 0.2f, 1.5f, "give up: CoM this far from the nearest foot per metre of CoM height [m/m]"},
    {"low", 0.55f, 0.3f, 0.9f, "give up: CoM below this share of its standing height"},
    {"step_budget", 0.0f, 0.0f, 30.0f, "steps before giving up, 0 = unlimited"},
    {"lead_t", 0.32f, 0.05f, 0.5f, "prediction point lead time at 1.78 m [s] (about lift-off plus swing time)"},
    {"half_w", 1.0f, 0.5f, 1.5f, "foot place half width: share of the rest stance half width (sole centres, ~0.135 m)"},
    {"margin", 0.3f, 0.0f, 1.0f, "prediction point outside the support quad by this share of the foot width -> step"},
    {"apart", 0.6f, 0.2f, 1.5f, "feet wider than the place width + this share of the rest stance -> gathering step"},
    {"place_fwd", 0.05f, 0.0f, 0.15f, "fore-aft place correction by direction other foot -> prediction point [m]"},
    {"carrot", 0.165f, 0.05f, 0.4f, "swing frame target at most this far from the foot [m]"},
    {"over", 0.0f, 0.0f, 0.15f, "swing aim past the place [m]"},
    {"lift0", 0.04f, 0.0f, 0.1f, "swing from the ground: minimum lift until near the place [m]"},
    {"mid_t", 0.1f, 0.0f, 0.4f, "in a step the support middle is CoM + velocity * mid_t [s]"},
    {"leg_frac", 0.99f, 0.9f, 1.0f, "compass standing leg length as a share of the full leg"},
    {"drop_max", 0.12f, 0.0f, 0.25f, "compass pelvis drop limit at 1.78 m [m]"},
    {"com_fwd", 0.005f, -0.08f, 0.08f, "posture: CoM ahead of the feet middle [m]"},
    {"cor_td", 0.3f, 0.0f, 0.5f, "balance numbers: velocity weight [s]"},
    {"place_td", 0.45f, 0.0f, 0.5f, "standing: desired pelvis place against the CoM velocity [s]"},
    {"cor_ki", 0.5f, 0.0f, 5.0f, "balance numbers: integral weight [1/s]"},
    {"ank_fb", 5.0f, 0.0f, 15.0f, "fore-aft number -> standing ankle [rad/m]"},
    {"ank_lat", 3.0f, 0.0f, 15.0f, "lateral number -> ankle inversion [rad/m]"},
    {"len_lat", 0.3f, 0.0f, 2.0f, "lateral number -> leg length difference [m/m]"},
    {"roll_lat", 0.5f, 0.0f, 5.0f, "lateral number -> pelvis roll [rad/m]"},
    {"spine_lat", 0.5f, 0.0f, 5.0f, "lateral number -> spine roll [rad/m]"},
    {"pitch_v", 0.19f, 0.0f, 1.0f, "one leg: forward pelvis pitch per backward CoM speed [rad per m/s]"},
    {"pitch_max", 0.30f, 0.0f, 1.0f, "one leg: counter-lean limit [rad]"},
    {"pitch_v2", 0.0f, 0.0f, 2.0f, "one leg: counter-lean extra slope above 1.6 m/s backward speed [rad per m/s]"},
    {"swing_over", 0.0f, 0.0f, 1.0f, "swing joint targets past the IK angles at full lift (share of the error)"},
    {"toe_land", 0.3f, 0.0f, 0.5f, "backward step: swing toe down [rad]"},
    {"sway_amp", 0.004f, 0.0f, 0.03f, "support middle sway amplitude [m]"},
    {"foot_torsion", 0.06f, 0.0f, 0.1f, "foot-floor torsional friction coefficient [m] (condim 4 on the feet); 0 = condim 3. 0.06 = sole friction 0.8 x mean radius 0.07-0.08 m of a 0.26 x 0.10 m sole; keeps a standing foot from twisting in place"},
    {"fall_legs", 3.0f, 0.0f, 4.0f, "leg stiffness multiplier while falling (the balancer gives 0.3 after give-up; 3 = ~0.9)"},
    {"fall_react", 0.0f, 0.0f, 1.0f, "1 = early fall judgement with arm and head protection, 0 = reaction only after give-up"},
    {"noslip", 0.0f, 0.0f, 5.0f, "MuJoCo noslip iterations: 0 off; 1 or more removes foot creep at a large frame-time cost"},
    {"assist", 0.45f, 0.0f, 1.0f, "stay-upright help: pelvis-feet pair force, share of M omega^2; 0 off"},
    {"swing_pv", 0.0f, 0.0f, 1.0f, "swing preview: 0 CoM velocity; 1 hip velocity + pelvis rotation (needs upper-body yaw)"},
    // Upper-body layer (core/l1_upper.cpp)
    {"upper", 1.0f, 0.0f, 1.0f, "upper-body layer on (1) or off (0, balancer pose only)"},
    {"up_back", 1.5f, 0.2f, 2.0f, "back twist (lumbar and chest yaw) and neck stiffness factor (the chest damps the pelvis yaw)"},
    {"up_arm", 1.0f, 0.3f, 2.0f, "arm stiffness factor (elbows are not loosened after a hit)"},
    {"up_swing", 0.5f, 0.0f, 1.5f, "shoulder flexion per rad of opposite-minus-own hip flexion (walking arm swing)"},
    {"up_twist", 0.3f, 0.0f, 1.0f, "chest yaw against the pelvis per rad of left-minus-right hip flexion"},
    {"up_spread", 1.0f, 0.0f, 2.0f, "arms-out gain on chest tilt and tilt rate beyond thresholds"},
    {"up_loose", 0.5f, 0.0f, 2.0f, "looseness depth after a hit per m/s of body velocity change (max 0.95)"},
    {"up_loose_t", 0.25f, 0.05f, 1.0f, "looseness recovery time constant [s]"},
    {"up_gaze", 1.0f, 0.0f, 1.0f, "look at the hit origin after a hit and sometimes along the steps (1 on)"},
    {"up_yaw_fb", 1.0f, 0.0f, 3.0f, "chest twist per rad of pelvis heading change (yaw damping) [rad/rad]"},
    {"bul_lean", 0.0f, 0.0f, 3.0f, "lean along the bullet (2nd and later bullets) scale; 0 = replaced by the pull force bul_pull"},
    {"up_loose_yaw", 1.0f, 0.0f, 1.0f, "share of the hit looseness applied to the back twist muscles"},
    {"up_arm_fb", 0.0f, 0.0f, 2.0f, "shoulder flexion per rad of pelvis heading error (arms help the yaw damping)"},
    {"bul_crouch", 1.0f, 0.0f, 2.5f, "bullet crouch scale (1 = 3.5 cm pelvis drop at 1.78 m); 0 = no crouch. A deeper crouch puts the compass pelvis far below the real one in a split stance and drives the CoP to the toes"},
    {"up_loose_back", 0.5f, 0.0f, 1.0f, "share of the hit looseness applied to the back pitch and roll muscles"},
    {"up_sway", 2.0f, 0.0f, 5.0f, "arm lag behind the quiet-stance sway [rad per m/s of CoM speed]"},
    {"up_arm_d", 3.0f, 0.0f, 6.0f, "shoulder damping as a multiple of the full-stiffness value, also while loose (not after a bullet)"},
    {"bul_hold", 0.03f, 0.0f, 0.1f, "first bullet: CoM target against the bullet while the reaction lives [m at 1.78 m]"},
    {"up_loose_bp", 0.0f, 0.0f, 1.0f, "share of the push looseness applied to the back pitch and roll"},
    {"up_w_arm", 4.0f, 0.5f, 15.0f, "shoulder muscle frequency [rad/s] (kp = w^2 I)"},
    {"up_z_arm", 0.3f, 0.05f, 2.0f, "shoulder damping ratio (low: the arm swings back after a push)"},
    {"up_w_elb", 6.0f, 0.5f, 15.0f, "elbow muscle frequency [rad/s] (softer than the shoulder: the forearm lags)"},
    {"up_z_elb", 0.6f, 0.05f, 2.0f, "elbow damping ratio"},
    {"up_w_neck", 15.0f, 2.0f, 60.0f, "neck muscle frequency [rad/s]"},
    {"up_z_neck", 2.7f, 0.1f, 5.0f, "neck damping ratio (high: the head does not whip after a push)"},
    {"up_w_back", 8.0f, 2.0f, 30.0f, "back pitch and roll muscle frequency [rad/s]"},
    {"up_z_back", 1.0f, 0.1f, 2.0f, "back pitch and roll damping ratio"},
    {"up_gsup", 0.1f, 0.0f, 1.0f, "share of the arm weight held by the arm muscles (0 = the arms hang as pendulums)"},
    {"up_rest0", 0.3f, 0.0f, 1.0f, "upper targets pulled to the rest pose by this share at speed (CoM >= 0.6 m/s)"},
    {"up_rest1", 0.95f, 0.0f, 1.0f, "... and by this share standing still"},
    {"up_w_abd", 8.0f, 0.5f, 20.0f, "shoulder abduction frequency [rad/s], damping ratio 1 (the arms stay at the body)"},
    {"push_t", 0.0f, 0.0f, 0.3f, "push contact time [s]; 0 = one substep (instant push)"},
    {"up_cw", 0.1f, 0.0f, 1.0f, "counterweight after a hit: arm abduction per m/s of lateral CoM velocity [rad s/m]"},
    {"up_neck_str", 4.0f, 0.5f, 10.0f, "neck strength factor over the body table"},
    {"up_w_roll", 16.0f, 2.0f, 30.0f, "back roll muscle frequency [rad/s] (stiffer than the pitch)"},
    {"immortal", 0.0f, 0.0f, 1.0f, "1 = health never drops: every bullet acts like the first (a head bullet still kills)"},
    {"up_gsup_bul", 1.0f, 0.0f, 1.0f, "arm weight share held by the arm muscles while a bullet reaction lives"},
    {"bul_pull", 0.31f, 0.0f, 1.0f, "pull on the pelvis along the bullet from the 2nd bullet of a live reaction [x weight]: fitted to the bullet travel of the standing body (3 chest hits 1 s apart: about 2.75 m, 2 hits: about 1.2 m) with the duration below"},
    {"bul_pull_t0", 0.65f, 0.0f, 3.0f, "bullet pull time per bullet, drawn from t0 [s] (on average 0.9 s, a little under the interval of a slow series, so the pull has ended before the next hit; +-28 %)"},
    {"bul_pull_t1", 1.15f, 0.0f, 3.0f, "... to t1 [s]"},
    {"bul_dead_nostep", 0.0f, 0.0f, 1.0f, "1 = no steps after a trunk bullet wounds or kills"},
    {"bul_crouch_t", 0.12f, 0.05f, 1.0f, "bullet crouch time constant [s]"},
    {"bul_whip", 1.0f, 0.0f, 2.0f, "trunk whip (chest) and fold (belly) scale after a bullet; 0 = off"},
    {"bul_hands", 2.0f, 1.0f, 2.0f, "hands to a trunk or leg wound: 2 both, 1 the near one"},
    {"bul_assist", 2.0f, 1.0f, 4.0f, "stay-upright help factor while a live bullet reaction lasts (too large lets the feet land far behind the CoM under the pull)"},
    {"bul_steps", 12.0f, 1.0f, 30.0f, "step reserve per bullet (enough to walk back under the pull of several bullets)"},
    {"bul_swing", 1.0f, 0.0f, 2.0f, "free-arm swing amplitude scale while a bullet reaction lives"},
    {"bul_stick", 1.0f, 0.0f, 3.0f, "sticky palm force scale"},
    {"push_fmax", 2000.0f, 0.0f, 10000.0f, "largest push force [N] (a hand); the push lasts impulse / force; 0 = instant"},
    {"fall_sit", 1.0f, 0.0f, 1.0f, "backward fall: sit down early (hips by the trunk angle, knees bent); 0 = off"},
    {"fall_sit_legs", 1.0f, 0.0f, 4.0f, "leg stiffness multiplier while seating in a backward fall after the give-up"},
// The seat starts once the capture point is far behind the support, so a step can still catch moderate
// backward pushes first.
    {"fall_sit_out", 0.9f, 0.0f, 2.0f, "seat starts when the capture point is this far out of the support backward [m]"},
    {"fall_sit_share", 0.9f, 0.0f, 1.0f, "backward seat: each leg draws its share of the seat from this to 1"},
    {"fall_sit_swing", 1.0f, 0.0f, 1.0f, "backward seat: 1 = the swing leg stays the balancer's until the give-up"},
    {"fall_sit_knee_t", 0.15f, 0.05f, 1.5f, "backward seat: knee ramp time [s] (hips ramp in 0.15 s)"},
    {"fall_sit_gsup", 0.0f, 0.0f, 1.0f, "backward seat after the pelvis touch: leg weight support share (with support the legs float up)"},
    {"fall_sit_roll", 0.0f, -3.0f, 3.0f, "backward seat: hip abduction per rad of pelvis roll (levels the pelvis) [rad/rad]"},
    {"gu_first", 1.0f, 0.0f, 1.0f, "give-up: 1 = the first step from calm counts at its target until touchdown"},
    {"pv_far", 1.0f, 0.0f, 1.0f, "swing: 1 = no hip preview beyond two carrots from the place (foot = body speed + own)"},
    {"reach_keep", 0.0f, 0.0f, 1.0f, "reach: 1 from the pelvis height to keep (straight touchdown knee), 0 hip angles"},
    {"gu_vz", 0.0f, 0.0f, 3.0f, "give up: CoM falling faster than this during a swing, below 0.95 standing height [m/s]; 0 off"},
    {"gu_side", 1.8f, 1.0f, 3.0f, "give up: distance factor when the CoM moves sideways (strong side falls keep stepping)"},
    {"fv", 1.0f, 0.0f, 1.0f, "leg muscles force-velocity: yielding holds up to 1.4x, fast shortening less (0 off)"},
    {"fv_con", 0.0f, 0.0f, 1.0f, "fv: share of the shortening loss (1 Hill, 0 none)"},
    {"stand_clip", 1.0f, 0.0f, 1.0f, "standing foot clipped by reach around the desired hip (0 off, 1 on)"},
    {"strut", 1.0f, 0.0f, 3.0f, "one leg: standing hip flexion pull at most this x its muscle limit x foot load share (target <= q + budget / kp; 0 off)"},
    {"mid_back", 1.0f, 0.0f, 1.0f, "step support-middle shift dropped by this x cosine of the backward motion"},
    {"fv_hip", 0.0f, 0.0f, 1.0f, "force-velocity also on hip flexion and abduction (0 off, 1 on)"},
    {"stall_reach", 1.0f, 0.0f, 1.0f, "stalled swing: desired pelvis lowered until the swing leg reaches the place (0 off, 1 on)"},
    {"on_keep", 0.03f, 0.0f, 0.1f, "standing foot stays support up to this ankle rise [m] (0 off)"},
    {"swing_pvr", 0.0f, 0.0f, 1.0f, "swing preview of the pelvis rotation only, hip at the CoM velocity (0 off, 1 on)"},
    {"foot_down", 1.0f, 0.0f, 1.0f, "far-side unloaded standing foot reaches the floor from the higher of desired and present hip (0 off, 1 on)"},
    {"leg_w_st", 0.0f, 0.0f, 40.0f, "standing hip and knee muscle frequency [rad/s] (kp = w^2 I, I = body above the joint; 0 = table x leg_k)"},
    {"leg_z_st", 0.4f, 0.05f, 3.0f, "standing hip and knee damping ratio"},
    {"leg_w_sw", 0.0f, 0.0f, 80.0f, "swing hip and knee muscle frequency [rad/s] (I = leg below the joint; 0 = table x swing_k)"},
    {"leg_z_sw", 0.45f, 0.05f, 3.0f, "swing hip and knee damping ratio"},
    {"step_h_back", 0.157f, 0.05f, 0.3f, "step height of the backward part of a step at 1.78 m [m] (step_h: forward, sideways)"},
    {"lean_top", 1.0f, 0.0f, 1.0f, "one leg: counter-lean pitch on top of the strut budget (1) or inside it (0)"},
    {"lean_decay", 0.3f, 0.0f, 2.0f, "two feet: counter-lean fades with this time constant [s] (0: gone at once)"},
    {"place_back", 0.43f, 0.2f, 1.0f, "backing off: place look-ahead = this x lead_t for the backward part of the motion (1 = as forward)"},
    {"place_back_d", 0.23f, 0.0f, 0.6f, "backing off: the place falls short of the lead_t point by at most this at 1.78 m [m]"},
    {"lean_spine", 0.0f, 0.0f, 2.0f, "share of the one-leg counter-lean bent into the spine (forward), on top of the pelvis pitch"},
    {"turn_go", 1.0f, 0.0f, 1.0f, "stepping sideways faster than turn_v, the held heading turns to the motion (0 off)"},
    {"turn_v", 0.5f, 0.0f, 3.0f, "least CoM speed for the turn [m/s]"},
    {"kn_acc", 0.0f, 0.0f, 1000.0f, "knee target acceleration limit [rad/s2] (0 off)"},
    {"kn_vst", 0.0f, 0.0f, 10.0f, "standing knee target straightening speed limit [rad/s] (0 off)"},
    {"drop_k", 0.0f, 0.0f, 5.0f, "in a step: pelvis drop arc gain on the CoM-flight-point to foot distance (0 off)"},
    {"drop_c0", 0.14f, 0.0f, 0.4f, "in a step: distance without drop at 1.78 m [m] (about the stance half width)"},
    {"drop_up", 0.0f, 0.0f, 2.0f, "desired pelvis rises at most this fast at 1.78 m [m/s] (0 off)"},
    {"drop_lim", 0.12f, 0.0f, 0.25f, "in a step: limit of that drop at 1.78 m [m]"},
    {"car3d", 0.0f, 0.0f, 1.0f, "swing carrot in space with the height, length from response time x swing speed (0 off)"},
    {"sw_v", 3.2f, 0.5f, 10.0f, "swing foot speed at 1.78 m [m/s] (car3d)"},
    {"car_c", 0.02f, 0.0f, 0.2f, "carrot length added at 1.78 m [m] (car3d)"},
    {"sw_hd", 0.0f, 0.0f, 1.0f, "swing leg reaches from the desired pelvis height (1) or the present one (0)"},
    {"zmp_fn", 0.0f, 0.0f, 1.0f, "standing ankle torque limit from the measured foot force (1) or the weight share (0)"},
    {"hip_pd", 2.0f, 0.0f, 2.0f, "one leg: standing hip flexion damping acts on the pelvis pitch rate, not on the leg turning under it (1; 2 only moving back)"},
    {"act_t", 0.0f, 0.0f, 0.1f, "leg muscle activation time constant: drive torque follows its command with this lag [s] (0 off; Thelen 2003 0.015)"},
    {"st_ank", 0.0f, 0.0f, 1.0f, "loaded standing foot on its toes: leg solved to the real ankle height (1) or the floor (0)"},
    {"lift_v", 0.0f, 0.0f, 5.0f, "swing height target rises from the lift-off at most this speed [m/s at 1.78 m] (0 off)"},
    {"hip_pd_v", 0.95f, 0.0f, 3.0f, "hip_pd only after pushes whose CoM speed (one frame ahead) stays below this [m/s at 1.78 m] (0: any)"},
    {"arm", 1.0f, 0.0f, 1.0f, "leg muscles with three numbers: frequency, damping, stiffness (flywheel = stf x inertia below x (w^2 h^2 + 2 z w h), h = frame); 0 = table muscles"},
    {"stf_hip", 4.0f, 0.0f, 100.0f, "hip muscle stiffness (flywheel multiplier; per hip axis)"},
    {"stf_kst", 4.0f, 0.0f, 100.0f, "standing knee muscle stiffness"},
    {"stf_ksw", 5.0f, 0.0f, 100.0f, "swing knee muscle stiffness"},
    {"stf_ast", 16.0f, 0.0f, 300.0f, "ankle muscle stiffness, foot touching the floor"},
    {"stf_aair", 1.0f, 0.0f, 100.0f, "ankle muscle stiffness, foot in the air"},
    {"st_twist", 0.04f, 0.0f, 0.2f, "standing hip twist torque <= this x foot normal force [m] (sole torsional friction 3-5 cm; 0 off)"},
    {"mw_st", 17.5f, 1.0f, 60.0f, "arm on: standing leg muscle frequency [rad/s] on the joint's own response inertia + flywheel"},
    {"mz_st", 1.0f, 0.1f, 3.0f, "arm on: standing leg muscle damping ratio"},
    {"mw_sw", 30.0f, 1.0f, 60.0f, "arm on: swing leg muscle frequency [rad/s]"},
    {"mz_sw", 1.0f, 0.1f, 3.0f, "arm on: swing leg muscle damping ratio"},
    {"asst_ref", 0.0f, 0.0f, 1.0f, "alternative stay-upright help: pulled point CoM + v asst_t (not the capture point), fade cos(tilt)^asst_n (0: capture point, linear fade to 45 deg)"},
    {"asst_t", 0.1f, 0.0f, 0.5f, "asst_ref: velocity look-ahead of the pulled point [s] (with the CoM velocity damping)"},
    {"asst_n", 8.0f, 1.0f, 20.0f, "asst_ref: tilt fade power (8: a third left at 30 deg)"},
    {"apart_fa", 0.3f, 0.0f, 9.0f, "> 0: below this CoM speed [m/s] the gathering step also tests the feet distance (fore-aft spread too); 0 = sideways spread only"},
    {"st_yawf", 0.0f, 0.0f, 1.0f, "1: the standing leg is solved in the pelvis frame with its present heading; only the hip twist (friction-limited, st_twist) turns the pelvis; 0 = desired heading frame"},
    {"hip_pd_fast", 0.0f, 0.0f, 1.0f, "share of the hip_pd target velocity kept after a fast push (latched; 0 = joint damping as before)"},
    {"ank_fh", 0.0f, 0.0f, 1.0f, "1: the standing ankle torque window is centred on the torque that holds the floor's horizontal force acting at the sole below the ankle (the CoP stays in the sole, the foot does not roll onto its edge)"},
    {"spin_k", 0.0f, 0.0f, 20000.0f, "anti-spin stiffness of a standing foot at 80 kg [N m/rad] (sole torsional stiffness before it slips; damping kSpinD)"},
    {"st_invlim", 0.0f, 0.0f, 0.3f, "> 0: the standing leg is solved for the hip position whose flat-foot ankle inversion stays this margin [rad] inside the joint range (keeps the foot off its edge); 0 = off"},
    {"spin_cap", 1.0f, 0.0f, 10.0f, "anti-spin cap multiplier over kSpinMu 0.05 x normal force (sole torsional friction)"},
    {"zmp_fn_inv", 1.0f, 0.0f, 1.0f, "1: the standing ankle inversion torque window from the measured foot force (min with the weight share; zmp_fn does both ankle joints); 0 = weight share"},
    {"asst_yaw", 0.0f, 0.0f, 2.0f, "the stay-upright pair (pelvis +F, feet -F) gets a pure pelvis yaw torque that cancels its yaw moment: 1 while a bullet pull lasts, 2 always, 0 never (the pair between the pelvis and a standing foot aside twists that foot; in pushes its yaw moment counters the braking lever of a single standing foot)"},
    {"roll_guard", 0.1f, 0.0f, 0.5f, "> 0: a standing ankle's inversion torque toward the edge its sole already tilts to fades to zero at this tilt [rad] (a foot that tipped onto its edge is not pushed further); 0 = off"},
    {"asst_on", 0.0f, 0.0f, 1.0f, "1: the stay-upright help pulls toward the middle of the feet standing by the balancer's touch rule (a foot lifted up to on_keep stays support), not only the loaded ones; 0 = loaded feet"},
    {"cop_k", 0.0f, 0.0f, 2.0f, "> 0: the balance numbers (lateral always, fore-aft on two feet; ~ capture point error) x this shift the support point of the weight-holding torques, clipped into the soles (ankle strategy: the CoP moves toward the outer edge); 0 = support point follows the CoM"},
    {"up_w_tw", 6.0f, 0.0f, 40.0f, "back twist (lumbar and chest yaw) muscle frequency [rad/s] on its response inertia; 0 = table stiffness x up_back"},
    {"up_z_tw", 1.0f, 0.1f, 3.0f, "back twist muscle damping ratio"},
    {"up_stf_tw", 24.0f, 0.0f, 100.0f, "back twist muscle stiffness (flywheel multiplier)"},
    {"up_stf_back", 0.0f, 0.0f, 100.0f, "back pitch and roll muscle stiffness (flywheel multiplier); > 0: kp on the response inertia; 0 = two numbers on the inertia carried"},
    {"up_stf_neck", 0.5f, 0.0f, 100.0f, "neck muscle stiffness (flywheel multiplier); > 0: kp on the response inertia; 0 = two numbers on the inertia carried"},
    {"up_guard", 1.0f, 0.0f, 1.0f, "arm in waiting: the arms go toward the side the chest leans and the body moves (stumble), 0 = off"},
    {"sw_press_b", 0.0f, 0.0f, 1.0f, "backward part of a step: share of the below-floor swing height target kept (1 = down to -step_h_back, 0 = the ankle aims at the floor height)"},
    {"up_stf_arm", 0.0f, 0.0f, 100.0f, "shoulder muscles (flex, abd, rot) stiffness (flywheel multiplier); > 0: kp on the response inertia; 0 = two numbers on the inertia carried"},
    {"up_stf_elb", 0.0f, 0.0f, 100.0f, "elbow and wrist muscles stiffness (flywheel multiplier); > 0: kp on the response inertia; 0 = two numbers on the inertia carried"},
    {"st_toe", 0.0f, 0.0f, 2.0f, "unloaded standing leg out of reach: the ankle plantarflexes by the shortfall over the ankle-toe length x this, so the toe stays on the floor (0 off)"},
    {"asst_split", 1.0f, 0.0f, 1.0f, "1: the stay-upright help pulls toward the middle of both feet and its reaction goes half to each foot (standing or in the air), on while any foot touches; 0: middle and shares of the loaded feet"},
    {"turn_bul", 0.0f, 0.0f, 1.0f, "1: turn_go (heading to the motion) is off while the bullet reaction lives"},
    {"asst_comp", 0.0f, 0.0f, 3.0f, "the leg of an unloaded foot cancels the help reaction on it (J^T F at the foot CoM), so the foot keeps its target and the pair keeps no yaw moment about the standing foot: 1 any unloaded foot, 2 not the swing foot, 3 any unloaded foot, only the part across the last hit (0 off)"},
// While a bullet reaction lives, pulling toward the loaded feet keeps a foot in the air from taking half of the
// pull, which would spin the body.
    {"asst_bold", 1.0f, 0.0f, 2.0f, "2: as 1 but only while no swing is active; 1: while a bullet reaction lives the help pulls toward the loaded feet with the reaction by load share (rule before asst_split); pushes keep the split"},
    {"hpd_bul", 0.0f, 0.0f, 9.0f, "> 0: the hip_pd_v lock latches only after a bullet hit or when the CoM speed (one frame ahead) passes this [m/s at 1.78 m]; pushes keep the pelvis hold (0: any fast push locks)"},
    {"td_now", 0.0f, 0.0f, 2.0f, "2: a swing ends only on a real touch (load share holds, or floor force with the sole down; also for the hard touch); 1: a swing ends on touching only while the foot has floor force now or rests (the filtered load share outlasts a one-frame brush of the floor)"},
    {"hpd_ld", 0.0f, 0.0f, 1.0f, "1: hip_pd (pelvis hold) only on a loaded leg; a leg that just touched keeps the joint damping that stops its swing"},
    {"sw_commit", 0.0f, -4.0f, 4.0f, "|v| > 0: when the swing foot comes within |v| carrot lengths of its place, the place freezes in the world for the rest of the step (the foot brakes to the floor instead of landing at the place speed); v > 0 only past half the step line, v < 0 also at once (a place near the foot at lift-off makes a short quick step)"},
    {"hpd_td", 0.0f, 0.0f, 1.0f, "hip_pd (pelvis hold) spares the leg that touched down last for this long [s]: its joint damping brakes the leg turning under the body (0 off)"},
    {"st_toe_side", 0.0f, 0.0f, 1.0f, "1: st_toe acts only while the CoM moves more sideways than fore-aft (a toe-down front foot pushes a back push on)"},
    {"pull_lean", 0.0f, 0.0f, 2.0f, "under the bullet pull the prediction point and places move along it by this share (up to 1) of bul_pull x CoM height (the support for resting against a constant force), the help pulls toward the support middle minus it; 2: full share and on two feet the desired pelvis too (0 off)"},
    {"bul_brake", 1.0f, 0.0f, 2.0f, "while a bullet reaction is live the weight-holding support point goes fore-aft to this x the capture point offset v / omega (+ bul_center x the offset from the middle on two feet), clipped into the soles; replaces the fore-aft part of cop_k (0 off)"},
    {"pull_help", 0.0f, 0.0f, 1.0f, "share of the pull offset for the help target alone (0: the pull_lean share)"},
    {"bul_center", 0.0f, 0.0f, 1.0f, "while a bullet reaction is live the help and the fore-aft balance number count this share of the CoM offset from the support middle, the velocity part in full (slow re-centring, same braking; 0 off = 1)"},
    {"rlx_lead", 0.0f, 0.0f, 1.0f, "while the wounded body relaxes on the move (L2: down, steps on) the prediction point and the places look ahead by this share of lead_t (0 off = full)"},
    {"rlx_swhd", 0.5f, 0.0f, 1.0f, "while relaxing on the move the swing leg reaches from the hips lowered by this share of the desired drop (sw_hd)"},
    {"rlx_gap", 0.15f, 0.0f, 0.3f, "while relaxing on the move the desired pelvis height never rises and stays at most this far below the present hips at 1.78 m [m] (0 off)"},
    {"rlx_ff", 0.0f, 0.0f, 20.0f, "while relaxing on the move a standing knee gets its target's flexion rate as the target velocity, at most this [rad/s] (0 off)"},
    {"rlx_car", 1.0f, 0.0f, 3.0f, "while relaxing on the move the swing carrot grows by this x the CoM travel toward the aim over the swing response time (0 off)"},
    {"rlx_ldir", 0.0f, 0.0f, 3.0f, "1: rlx_lead shortens the look-ahead only by the backward share of the CoM velocity direction (forward and sideways: full); 2: rlx_swhd too; 3: and no forward support-middle shift (mid_t)"},
    {"rlx_trail", 0.0f, 0.0f, 0.2f, "relaxing on the move: a standing foot behind its hip along the motion is reached from at most this x hs below the present hip (stays loaded); with rlx_ldir only the forward share"},
    {"rlx_gu", 1.0f, 0.0f, 1.0f, "1: the balancer does not give up while the wounded body relaxes on the move (L2 walk)"},
    {"rlx_tsink", 0.8f, 0.0f, 1.0f, "relaxing on the move the desired pelvis is at most this x hs x (1 - tone) under the compass height (sink no faster than the support fades; 0 = off)"},
    {"rlx_pitch", 0.0f, 0.0f, 1.5f, "relaxing on the move the desired pelvis pitches forward by this x (1 - tone) [rad] (trunk over the feet in the squat; 0 = off)"},
    {"rlx_m2", 0.0f, 0.0f, 1.0f, "1: a relaxing body keeps its leg stiffness after the balancer gives up (only the tone fades the legs)"},
    {"rlx_pull", 0.0f, 0.0f, 1.0f, "while relaxing on the move the prediction point and places lean into the bullet pull by this share (pull_lean for the relaxation; 0 = pull_lean)"},
    {"rlx_nolean", 0.0f, 0.0f, 1.0f, "1: no one-leg counter-lean (pelvis pitch forward when backing) while relaxing on the move"},
    {"body_mu", 0.5f, 0.1f, 1.5f, "sliding friction of the floor and of every body part except the feet (feet keep 1: MuJoCo takes the larger of the two geoms); a limp body is held by the floor friction and stops at the end of its fall"},
    {"brk_after", 0.0f, 0.0f, 1.0f, "1: bul_brake and bul_center act only once the bullet pull has ended (the body yields during the pull and brakes after it)"},
    {"pull_ramp", 0.0f, 0.0f, 1.5f, "bullet pull rises linearly over this time from the start of a pull episode (a bullet during a pull keeps the level) [s]; 0 = at once"},
    {"pull_fade", 0.0f, 0.0f, 1.0f, "bullet pull falls linearly to zero over this last time before its end [s]; 0 = step"},
    {"pull_cop", 0.0f, 0.0f, 1.5f, "share of the pull rest offset bul_pull x CoM height added to the L0 support point along the pull while it acts (1: the standing feet cancel the pull)"},
    {"bul_ds", 0.0f, 0.0f, 0.4f, "while a bullet reaction lives, no new step until this time after the last touchdown while both feet are on the floor [s] (the trailing foot stays down while the front one brakes)"},
    {"bul_reach", 0.0f, 0.0f, 1.0f, "1: while a bullet reaction lives, in a step the desired pelvis is no higher than the height from which the swing leg reaches its place with the touchdown knee kTdKnee (a bent-knee touchdown)"},
    {"bul_yawd", 0.0f, 0.0f, 3.0f, "while a bullet reaction lives the desired pelvis heading gets - bul_yawd x response time x pelvis yaw rate, as pitch and roll have (0 off)"},
    {"leg_hob", 0.0f, 0.0f, 0.6f, "after a leg bullet gave way, the body hobbles toward the wound in commanded side steps of this length (wounded foot out, healthy foot after it) [m]; 0 off"},
    {"leg_pull", 0.0f, 0.0f, 1.0f, "after a leg bullet gave way, the pelvis is pulled sideways toward the wound (bullet pull machinery) with this share of bul_pull for the hobble time; 0 off"},
    {"leg_vg", 0.8f, 0.0f, 1.5f, "after a leg bullet gave way, the velocity goal of the stepping law (L1 vgoal) toward the wound for the hobble time: the wounded foot steps out first, the healthy one follows with short steps [m/s at 1.78 m]; 0 off"},
    {"bul_imp", 29.0f, 10.0f, 100.0f, "pistol bullet impulse when Shoot gets none [N s]: 5/6 of the impulse at which one chest hit makes the standing body (80 kg, 1.78 m) take its first step (measured on the bench), so a single hit rocks the body without a step"},
    {"land_roll", 0.3f, 0.0f, 0.6f, "roll landing for a medium drop (2.5-4.2 m): curl in the air, pelvis leaning back by this angle [rad] at the touch; 0: fall onto the hands"},
    {"air_rot", 1.0f, 0.0f, 3.0f, "flight torque on the pelvis to right it (feet down) when the fall started tilted; 1 = natural frequency 6 rad/s, 0 = off"},
    {"air_flail", 1.0f, 0.0f, 1.0f, "1: drops above 2 m flail in the air (arms circle, legs pedal, left and right out of phase, noise) and the curl starts 0.45 s before the touch; 0: no flail, curl at once"},
    {"deck", 1.0f, 0.0f, 1.0f, "1 = brace on a moving support (L3 sees the standing feet ride, L2 lowers the pelvis, widens the stance, arms out against the sway)"},
    {"deck_v0", 0.06f, 0.0f, 0.5f, "support speed envelope where the brace starts [m/s]"},
    {"deck_v1", 0.18f, 0.05f, 1.0f, "support speed envelope of the full brace [m/s]"},
    {"deck_crouch", 0.0f, 0.0f, 0.2f, "full-brace pelvis drop at 1.78 m [m] (0 off: a crouch adds steps on a swaying stage and loads a light bridge more)"},
    {"deck_wide", 0.0f, 0.0f, 0.2f, "full-brace extra half width of the stance at 1.78 m [m] (feet about shoulder width: biacromial ~0.40 m; 0 off: a widening step can leave a narrow deck)"},
    {"deck_arm", 0.55f, 0.0f, 1.3f, "full-brace shoulder abduction over the rest [rad] (arms 30-70 deg from the body)"},
    {"deck_cw", -1.0f, -3.0f, 3.0f, "full-brace arm motion [rad per m/s] of the CoM speed across the support; + counterweight, - reaction"},
    {"deck_pitch", 0.0f, -0.5f, 0.8f, "full-brace desired pelvis pitch forward [rad] (pairs with deck_crouch: a lowered pelvis on an upright target falls back); 0 off"},
    {"arm_swing", 1.0f, 0.0f, 2.0f, "free-arm swing outside bullets: 1 while bracing on a moving support, 2 always (chest speed over the support); 0 = bullets only"},
    {"deck_rel", 0.5f, 0.0f, 1.0f, "while bracing the balance takes the CoM speed relative to this share of the measured support speed (a moving deck shifts the capture point; a light bridge moves with the body)"},
    {"deck_hold", 1.0f, 0.0f, 3.0f, "bracing: hip adduction torque per measured outward foot push x half height (keeps the feet from creeping apart in the wide stance)"},
    {"arm_sw_v0", 0.8f, 0.0f, 3.0f, "free-arm swing (arm_swing): chest speed over the support where the free arms start to swing [m/s]"},
    {"deck_step", 3.0f, 0.0f, 10.0f, "braced on a swaying support, the foot the sway unloads re-steps under the pelvis at most once per this [s] (0 off)"},
    {"deck_edge", 0.2f, 0.0f, 0.4f, "a step place stays this far [m] inside the edge of the support under the stance foot (foot centre; L3 deck_off / deck_half); 0 off"},
    {"deck_mid", 0.3f, 0.0f, 1.0f, "braced on a moving support, a step place's offset from the support middle line shrinks by this share"},
    {"deck_ld", 0.4f, 0.1f, 0.6f, "deck step when the foot's load share is below this (higher steps more often and risks the edge)"},
    {"walk_end_t", 0.08f, 0.0f, 0.5f, "target lag at the wounded walk end (curl on) [s]; also keeps the sink (0 off)"},
    {"fold_a", 0.0f, 0.0f, 1.0f, "death fold (hips 75 deg, ankle to the limit) with the passive ankle range; 0 = the 20 deg fold"},
    {"bul_side", 0.12f, 0.0f, 0.5f, "spine lateral bend toward a trunk wound at the side [rad] (own choice)"},
    {"bul_guard", 0.08f, 0.0f, 0.5f, "spine flexion over a chest wound held with the hands [rad] (own choice)"},
    {"gaze_adv", 1.0f, 0.0f, 1.0f, "1: the head turns to a car coming (after the reaction delay, follows it) and to the origin of a bullet that missed; hits already turned it"},
    {"gaze_trunk", 0.5f, 0.0f, 1.0f, "share of a large gaze turn (beyond gaze_thr) that the trunk takes over, so the neck stays inside its comfortable range"},
    {"dodge", 1.0f, 0.0f, 1.0f, "1: a car that would hit in more than 0.5 s after the reaction time makes a velocity goal along the side out of its path (stepping law with push-off and shuffle); 0 off"},
    {"dodge_vg", 3.5f, 0.5f, 6.0f, "dodge: CoM speed goal along the side out of the path [m/s at 1.78 m] (the goal is relative: the law places the feet behind the CoM by (goal - speed) x lead_t, capped by vg_cap)"},
    {"dodge_ab", 1.4f, 0.5f, 8.0f, "dodge: the goal falls as sqrt(2 x this x distance left to the safe line) [m/s2 at 1.78 m]: the braking the last step can give"},
    {"vg_lead", 0.3f, 0.0f, 0.6f, "velocity goal: the foot on the goal side steps first, this far ahead of the CoM along the goal [m at 1.78 m] (0 off)"},
    {"vg_cap", 1.2f, 0.2f, 4.0f, "velocity goal: CoM speed behind the goal that the foot places count at most [m/s at 1.78 m] (the leg reaches 0.4 m)"},
    {"po_force", 0.4f, 0.0f, 0.8f, "velocity goal push-off: a horizontal pair force pelvis - feet along the goal [x body weight] while the CoM is slower than the goal (0 off)"},
    {"po_t", 0.2f, 0.05f, 0.8f, "push-off lasts this long from the start of the dodge [s at 1.78 m]"},
    {"vg_caph", 0.5f, 0.2f, 4.0f, "velocity goal: the lag cap vg_cap of a hurried dodge (push share 1): the trailing foot lands only this far behind the CoM [m/s x lead_t at 1.78 m] (shuffle: it does not brake)"},
    {"vg_nox", 1.0f, 0.0f, 1.0f, "hobble (leg_vg): the trailing foot of the goal gait lands at most this share of the way back from a place past the stance foot along the goal (1: not past it, no crossing steps; 0 off). The dodge does not use it"},
    {"flinch", 1.0f, 0.0f, 1.0f, "1: a bullet that missed close to the head makes a short startle: head down, trunk and shoulders ducking away from the bullet"},
    {"pull_free", 1.0f, 0.0f, 1.0f, "share of the help's part against the bullet pull dropped while the pull acts (0 = the help cancels the pull)"},
    {"pitch_guard", 0.05f, 0.0f, 0.5f, "> 0: a standing ankle's dorsi torque fades to zero at this toe-up sole tilt [rad] (as roll_guard, heel side only); 0 = off"},
    {"pull_free_v", 1.8f, 0.0f, 3.0f, "> 0: pull_free fades to zero over 0.4 m/s up to this CoM speed along the pull [m/s] (the help returns at speed so the steps brake the body); 0 = no cap"},
    {"dead_fwd_arm", 0.0f, 0.0f, 3.0f, "> 0: a body killed while moving forward faster than 0.8 m/s (body hit) points the arms this far [rad] ahead of straight down (shoulder flexion = chest pitch + this), arms near the body, elbows almost straight (hands ahead of the head); 0 = the up-and-out pose"},
    {"leg_hold_body", 0.0f, 0.0f, 1.0f, "leg hold after a death from body hits: share of the leg strength kept while the pelvis is high (head bullet / kill: 0.8)"},
    {"pull_tone", 8.0f, 0.0f, 8.0f, "> 0: the bullet pull on a body going down (wounded or dead) is scaled by the muscle tone to this power (the help is off, the legs cannot answer); 0 = full pull"},
    {"dead_arm_hold", 0.0f, 0.0f, 1.0f, "with dead_fwd_arm: a wounded (not dead) body going down forward keeps this share of the arm strength (hands catch the floor)"},
    {"walk_vmax", 2.2f, 0.0f, 5.0f, "> 0: a wounded body moving forward faster than this [m/s] ends its relaxation steps (legs fold, kneels on the move); 0 = off"},
    {"prop_pel", 0.27f, 0.1f, 0.6f, "a quiet limp body with the pelvis above this x stance height [m at 1.78] and the head below the pelvis is propped (knees): its hips extend"},
    {"prop_tone", 0.2f, 0.0f, 1.0f, "leg tone floor for the hip opening of a propped dead body (0 = off)"},
    {"bsit_v", 1.5f, 0.0f, 5.0f, "> 0: a dying or wounded body going down backward faster than this [m/s] flexes the trunk and reaches back with the arms (rolls on the back); 0 = off"},
    {"bsit_tone", 0.6f, 0.0f, 1.0f, "strength share of the trunk, hips and arms in the bsit roll pose"},
    {"ankle_act", 0.0f, 0.0f, 40.0f, "> 0 = ankle dorsiflexion limit [deg] of a body that keeps its own balance (20 = active range), the passive body.ankle_dorsi after a bullet hit or death; 0 = always passive"},
    {"flat_r", 0.0f, 0.0f, 2.0f, "> 0 = a tumble also needs level ground this far around the pelvis [m] to end (no drop over 7.5 cm, so not on a stair tread); 0 = off"},
    {"push_old", 0.0f, 0.0f, 1.0f, "1 = pushes at each bench's legacy point (chest body + 0.06 m etc.); 0 = the standard point, the centre of mass of the body above the hip joints"},
    {"ch_hold", 0.0f, 0.0f, 3.0f, "backing off, the desired pelvis pitches forward by this x the chest tilt back past ch_thr [rad/rad] (0 off)"},
    {"ch_thr", 0.0f, -0.5f, 0.5f, "chest tilt back from vertical along the heading where the chest hold starts [rad]"},
    {"ch_td", 0.0f, 0.0f, 0.5f, "chest hold rate term [s]"},
    {"ch_max", 0.4f, 0.0f, 1.0f, "chest hold limit [rad]"},
    {"ch_spine", 0.0f, 0.0f, 1.0f, "share of the chest hold bent into the spine (forward) instead of the pelvis"},
    {"sup_brake", 0.0f, 0.0f, 20.0f, "standing legs' support force grows by M x this x the CoM speed toward the support point (leg damper along the CoM line) [1/s] (0 off)"},
    {"sup_brake_max", 1.0f, 0.0f, 3.0f, "limit of that extra support force [x body weight]"},
    {"lean_spine_push", 1.5f, -1.0f, 2.0f, "lean_spine after a push from the front (not a bullet, faded by kCh*); -1 = lean_spine"},
    {"hpd_fast_push", 0.5f, -1.0f, 1.0f, "hip_pd_fast after a push from the front (not a bullet, faded by kCh*); -1 = hip_pd_fast"},
    {"lean_hip_push", 1.0f, 0.0f, 3.0f, "after a push (same gates as lean_spine_push), from the first touchdown on, the desired pelvis pitches forward by this x the counter-lean on top (fold at the hips instead of the spine)"},
    {"hob_early", 2.5f, 0.0f, 5.0f, "the hobble goal starts this long after the last bullet in a thigh when that is before the give-way [s] (0: from the give-way)"},
    {"hob_ds", 0.1f, 0.0f, 2.0f, "thigh wounds only: hobbling, the next step starts after both feet have stood this long [s] (0: when the prediction point leaves the stance)"},
    {"throw", 0.0f, 0.0f, 1.0f, "hurried car dodge (decided with kTurnTc .. throw_tc left) throws the body: the escape-side foot lunges at once (1 on, 0 normal gait)"},
    {"throw_tc", 1.3f, 0.8f, 2.0f, "time left at the decision below which a dodge throws [s]"},
    {"throw_lead", 0.6f, 0.0f, 1.5f, "the lunge foot lands this far beyond the CoM along the dodge [m at 1.78 m]"},
    {"thr_po", 0.0f, 0.0f, 3.0f, "a throw keeps the push-off on (this x po_force, fading as the true CoM speed reaches the goal) until the dodge ends (0 off)"},
    {"dive_t", 0.4f, 0.0f, 1.0f, "a hurried dodge whose feet are still in the car path this long before the hit dives: legs tucked up, no support, no steps [s] (0 off)"},
    {"dive_abd", 0.6f, 0.0f, 1.0f, "dive: escape-side hip abduction [rad] (the near hip adducts kDiveAddK of it)"},
    {"thr_vg", 3.5f, 0.0f, 6.0f, "hurried dodge speed goal without the stop taper [m/s at 1.78 m] (0: as calm dodges)"},
    {"thr_pt", 0.0f, 0.0f, 1.0f, "1 = a hurried dodge tests the step need like a calm one (steps at once, no push-off pause)"},
    {"dive_jump", 0.1f, 0.0f, 0.5f, "dive: the legs first straighten hard this long (a jump) [s] (0: tuck at once)"},
    {"thr_ds", 0.0f, 0.0f, 1.0f, "hurried dodge: the next step starts once both feet stood this long since the decision or the last touchdown [s] (0 off)"},
    {"thr_ahead", 0.0f, 0.0f, 1.0f, "hurried dodge: every step lands at least this far beyond the CoM along the escape [m at 1.78 m] (0 off)"},
    {"thr_cap", 0.9f, 0.0f, 4.0f, "hurried dodge along the heading (car from the side): the lag cap vg_cap of the feet [m/s at 1.78 m] (0: vg_caph by urgency)"},
    {"dodge_lean", 0.03f, 0.0f, 0.5f, "while a calm car dodge along the heading runs the CoM target leans this far toward the escape [m at 1.78 m] (0 off)"},
    {"car_guard", 1.0f, 0.0f, 1.0f, "1 = a seen car: both forearms up in front (close) or the arms swing with the dodge (time left); 0 = the near arm reaches toward the car"},
    {"rlx_noext", 0.0f, 0.0f, 1.0f, "1 = relaxing on the move (dying) the desired pelvis is never above the present hips (no leg straightening push)"},
    {"dead_buckle", 0.0f, 0.0f, 2.0f, "dying, the legs fold to the squat within this time once the steps are over [s] (0: over the relaxation dead_T)"},
    {"dead_knee_first", 1.0f, 0.0f, 1.0f, "1 = a wounded body that stops stepping curls the hips no farther than its knees have folded"},
    {"hur_swing", 0.0f, 0.0f, 1.0f, "1 = a hurried dodge from the side swings the arms with the escape (as a calm one) instead of reaching toward the car"},
    {"guard_front", 0.0f, 0.0f, 1.0f, "1 = front cars (calm and hurried dodges) get the guard arms instead of the reach straight forward"},
    {"dive_arms", 0.0f, 0.0f, 1.0f, "1 = a diving dodge tucks the arms to the chest too"},
    {"curl_fade", 0.0f, 0.0f, 2.0f, "a wounded body whose relaxation steps end loses the rest of its strength within this time [s] (0: over the rest of the kHurtFade relaxation)"},
    {"curl_nosup", 0.0f, 0.0f, 1.0f, "1 = a dying body whose steps are over gets no weight support from its legs (they fold under it)"},
    {"hurt_fade", 0.0f, 0.0f, 3.0f, "a wounding hit fades the strength within this time [s] (0: kHurtFade, 4 x kDeadFade in l2_reactions.cpp)"},
    {"down_nosup", 1.0f, 0.0f, 1.0f, "1 = from a wounding hit (down 1; not dead) the legs carry no weight (L0 support off)"},
    {"down_supk", 0.0f, 0.0f, 8.0f, "a body going down: the legs' weight support x tone^this on top of the tone (0 off)"},
    {"dead_upright", 0.0f, 0.0f, 2.0f, "going down, the curl's hips reach full flexion only this long after the down-going hit [s] (0 off)"},
    {"curl_catch", 0.0f, 0.0f, 1.0f, "1 = a wounded body curling early keeps the catching arms while its pelvis is high"},
    {"curl_arm_hold", 0.0f, 0.0f, 1.0f, "a body-hit death keeps this share of the arm strength while the pelvis is high (0 off)"},
    {"fall_legs_fwd", 0.5f, 0.0f, 4.0f, "leg stiffness multiplier while falling forward (fall_legs for the other directions); softer legs fold under a forward fall and slow the head"},
    {"reach_hip", 1.0f, 0.0f, 1.0f, "1 = a weakened (wounded) leg reaches from its real hip height (sqrt(leg^2 - hip height^2)); 0 = the angle ellipse only"},
};

}  // namespace emil
