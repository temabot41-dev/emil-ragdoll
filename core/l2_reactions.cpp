// L2 reactions: change balancer inputs and muscle targets/stiffness only; no forces on the body
// except world events (bullet impulse, car contact). Runs after Balance (L1) every control step.
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "emil_world.h"

namespace emil {
namespace {
inline double Clamp(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
constexpr double kWoundT = 0.25;       // local muscle weakening after a hit [s]
constexpr double kWoundKp = 0.4;       // stiffness scale of joints next to the wound
// The back flinches from the pain, then straightens within 1-2 s: an exponential with a 0.6 s time constant leaves
// 5 % after 1.8 s.
constexpr double kFlinchT = 0.6;
constexpr double kHoldT = 0.3;  // ramp of the first-bullet CoM hold [s] (as the crouch, slower than the hit)
// Bullet reaction. All times drawn from the bot's RNG within these ranges. The crouch is held for about 4 s, the
// hands for about 3 s.
constexpr double kLive0 = 3.6, kLive1 = 4.6;      // reaction alive this long after the last bullet [s]
constexpr double kCrouch = 0.035;                 // pelvis lower while alive [m] (2-5 cm)
// Crouch in and out time constants [s]: a fast crouch (in.crouch) lets the legs drop the pelvis while the bullet
// pushes, and the body then needs extra steps; a slow one keeps the support firm.
constexpr double kCrouchOffT = 0.5;  // crouch in: param bul_crouch_t
constexpr double kLeanT0 = 0.15, kLeanT1 = 0.45;  // lean along the bullet (2nd and later bullets) duration [s]
constexpr double kLean0 = 0.10, kLean1 = 0.16;    // ... shift of the support middle [m] at 1.78 m
constexpr double kTwist = 0.0;  // chest yaw toward the wound side [rad]; off: the shoulders hardly turn
// Hands: both rise from 0.1-0.2 s, reach a chest wound at 0.4 s, hold about 3 s, then lower slowly (0.8 s ramp in
// l1_upper).
constexpr double kArmD0 = 0.04, kArmD1 = 0.1;     // reaction delay of the hands [s]
constexpr double kArmH0 = 4.4, kArmH1 = 5.0;      // the hands stay this long [s] (with the 1.5 s release)
// Trunk whip after a chest bullet: the back extends (about 19 deg at 0.2 s), then flexes 6-7 deg at 0.5-0.6 s and
// rests at 0.8 s. Target pulses [rad] at these times; drawn scale +-20 % (angles) and +-10 % (times). The back
// follows a target pulse about 0.1 s late, so the pulses come early.
constexpr double kWhipExt = 0.8, kWhipExtT = 0.07, kWhipExtW = 0.07;  // tuned on the benches
constexpr double kWhipFlex = 0.3, kWhipFlexT = 0.4, kWhipFlexW = 0.12;
// Belly bullet: the back folds forward about 18 deg at 0.4 s and stays while the hands hold the wound
constexpr double kBellyFlex = 0.45, kBellyT = 0.15, kBellyHold = 0.7;
// Thigh bullet: the wounded knee bends 36 -> 45 (2 s) -> 60 deg (3 s), the trunk leans to the wound 7 deg at 1.2 s
// to 28 deg at 3 s.
constexpr double kWLegKnee = 0.5, kWLegKneeT0 = 0.4, kWLegKneeT = 2.6;   // extra knee flexion [rad], start, ramp [s]
constexpr double kLegLeanW = 0.6, kLegLeanT0 = 0.3, kLegLeanT = 1.7;   // trunk lean to the wound [rad], start, ramp
// Hip flexion of both legs with the trunk lean [rad]; CoM shift to the wounded side [m at full], start, ramp
// [s]; healthy knee at most [rad] (33-36 deg)
// Give-way [s after the first bullet], shin factor, pulse length [s], strength share, extra knee [rad]; hobble
// CoM shift [m] (past the foot: sideways steps) and its length [s]
constexpr double kLegGiveT = 3.4, kLegShinF = 1.45, kLegGiveDur = 0.3, kLegGiveWeak = 0.5, kLegGiveKnee = 0.4;
constexpr double kLegDip = 0.15, kLegDipT0 = 0.2, kLegDipT1 = 0.3;  // give-way pelvis dip [m], down, up [s]
constexpr double kLegHobble = 0.0, kLegHobbleT = 4.0;
constexpr double kHobTo = 0.7;  // hobble: the swing starts at the latest this long after the command [s]
constexpr double kHobV = 0.5;  // hobble speed toward the wound [m/s] (0.5-0.9)
constexpr double kHobRamp = 0.3, kHobEnd = 3.0, kHobTaper = 0.8;  // hobble goal: ramp, end after give-way, taper [s]
constexpr double kHobBend = 0.5;  // share of the bent-over hip flexion kept while hobbling (tuned on the benches)
constexpr double kHobStart = 0.2, kHobGap = 0.15;  // hobble: first step after the give-way, pause between steps [s]
constexpr double kLegHipB = 0.3, kLegSide = 0.09, kLegSideT0 = 1.0, kLegSideT = 2.2, kLegHealthyKnee = 0;
constexpr double kLegDelay = 0.3;                 // a leg bullet weakens the leg after this [s]
constexpr double kLegLoss = 0.22;  // strength lost per leg bullet (1: 0.78 stands limping; 4: 0.12 gives way)
// Health 100, a pistol bullet takes 15: wounded below kHurtHp = 30, dead at 0 or below. The 5th chest bullet wounds
// (100 - 75 = 25 < 30; after the 4th it is 40) and the 7th kills (6th: 10, 7th: -5 -> 0). Muscles fade within
// kHurtFade and kDeadFade.
constexpr double kHealth = 100, kHurtHp = 30, kDeadHp = 0;
// Strength fade times [s]. Lethal: the shortest fade at which the pelvis falls fastest (shorter fades do not drop it
// sooner because the leg hold sits the body down, longer ones are slower). Wounded: four times the lethal one, so a
// wounded body keeps stepping a little while it goes down.
constexpr double kDeadFade = 0.35, kHurtFade = 4.0 * kDeadFade;
// Fold of the legs over the relaxation (head bullet, at 0.6 s: knees 140, hips 75 deg; ankle to the 20 deg
// dorsiflexion limit); start values = quiet stance
constexpr double kFoldKnee0 = 0.28, kFoldKnee = 2.45, kFoldHip0 = 0.07, kFoldHip = 2.0, kFoldAnkle = 0.35;
// With the passive ankle switch (>= kAnkleSw deg): hips fold to 75 deg and the ankle target goes to the limit - the
// knees go forward over the feet, which keeps the pelvis from drifting far behind the heels
constexpr double kFoldHipA = 1.3, kAnkleSw = 30.0;
// The side bend over a trunk wound is full at this hit offset from the mid-line [m] (half the chest breadth,
// ANSUR II men chest breadth 29 cm: docs/SOURCES.md anthropometry)
constexpr double kSideW = 0.15;
// Arms during the relaxation: to the face after a head bullet, else up and out; reached in kDeadArmT [s]
constexpr double kDeadArmT = 0.3, kDeadArmLag = 0.08;  // second arm up to this much later (not quite together)
constexpr double kDeadArmFlex = 1.0, kDeadArmAbd = 1.0, kDeadArmElb = 0.9;
constexpr double kDeadFwdV = 0.8, kDeadFwdAbd = 0.3, kDeadFwdElb = 0.3;  // forward-moving death [m/s, rad]
// Curl after the relaxation: strength, rise and fade times [s] (own choice; wounded roll curl 0.35)
constexpr double kDeadCurlTone = 0.3, kDeadCurlRise = 0.05, kDeadCurlFade = 0.5;
constexpr double kAirFootZ = 0.12;  // [m] airborne: both foot box centres this high (standing 0.03-0.04)
// Wounded steps: end below this pelvis height [m at 1.78 m] (half the stance); pelvis target sinks by this
// much [m] within kWalkSinkT [s]
constexpr double kLegHoldHead = 0.8, kLegHoldPel = 0.25, kLegHoldW = 0.1;  // tuned on the benches, see the dead branch
constexpr double kWalkEndPel = 0.47, kWalkSink = 0.45, kWalkSinkT = 0.4;  // a fast sink: pelvis 73 -> 43 cm in 0.3 s
// Settling on the move: trunk lean target, hip flexion added to the standing legs [rad]
constexpr double kWalkTrunk = 0.35, kWalkHip = 0.8;
// Curl: neck and spine get the table stiffness from chest forward axis z > -kBackUp0 over kBackUpW (own choice)
constexpr double kBackUp0 = 0.2, kBackUpW = 0.4;
// Face down: chest forward axis z below -kFaceDown and head CoM below kHeadLow [m at 1.78 m] (own choice)
constexpr double kFaceDown = 0.6, kHeadLow = 0.2, kHeadBelow = 0.2;
// Curl fallen forward: hip target [rad] (trunk 58-62 deg), weight by the trunk line cos (trunk 40 -> 60 deg) and by
// the chest axis z (-0.3 -> -0.7)
constexpr double kFwdHip = 1.05, kFwdCos0 = 0.77, kFwdCosW = 0.27, kFwdUp0 = 0.3, kFwdUpW = 0.4;
constexpr double kCatchPel = 0.45;  // catching arms kept above this pelvis height [m at 1.78 m] (own choice)
constexpr double kDeadCurlMax = 1.0;  // [s] longest curl before lying down (own choice)
// Back roll pose: chest leaning back more than kBsitLean [rad] to start; ends with the chest below
// kBsitChest x stance [m] (the back is down)
constexpr double kBsitLean = 0.15, kBsitChest = 0.25;
constexpr double kBsitPel = 0.8, kBsitRise = 0.1, kBsitMax = 1.5;  // back roll pose: start height, ramp, max [s]
constexpr double kPropMax = 2.0, kPropRise = 0.3;  // [s] hip opening of a propped body: at most, ramp
constexpr double kDeadQuietT = 0.25;  // [s] quiet this long: lie down (a longer wait holds the knees-up pose too long)
// The curl opens on the back from head below the pelvis by cos kRollCos0 (flat) over kRollCosW (15 deg), neck too;
// hips open to kOpenHip [rad] at table stiffness only in a back roll, trunk past 100 deg (cos kRollHipCos) over
// kRollCosW (own choice)
constexpr double kRollCos0 = 0.0, kRollCosW = 0.25, kOpenHip = 0.3, kRollHipCos = 0.17;
constexpr double kArmDead = 0.01;  // joint armature of a relaxed joint = the model default (L0 kArm0)
constexpr double kWoundCurlTone = 0.35;  // tone kept by a wounded body curling up (own choice: not a rag)
constexpr double kCurlV = 0.6;           // the wounded curl acts while the CoM is faster than this [m/s]
// Lying: the body counts as quiet when the CoM is slower than kQuietV for kQuietT; then the neck lets go
constexpr double kQuietV = 0.3, kQuietT = 0.5;

constexpr double kRelaxT = 0.8;        // lying: stiffness decay time constant [s]
constexpr double kRelaxMin = 0.06;     // lying: residual tone (damping keeps it still)
constexpr double kLieBlendT = 0.3;     // lying: fall pose -> rest pose time constant [s] (faster than the tone decay)
constexpr double kCurlT = 0.45;       // curl ramp time [s] (own choice within 0.3-0.5 s)
// Soft curl (not a tight ball): hips about 80 deg, knees about 92, muscles at kCurlKp of the full stiffness while it
// lasts, the support weight of legs and back eased by kCurlSup (off: easing it lets the head hit the floor hard in
// high drops); the pose is never straightened by a timer
constexpr double kCurlKp = 0.5, kCurlSup = 0.0, kCurlUpZ = 0.7, kCurlUpC = 0.85;
// Landing: the pelvis goes down after the touch at a constant deceleration (at most kLandA [m/s^2], about 2 g
// on top of gravity: two knees at their 220 N m with a 0.2 m lever carry ~2.5 body weights), at most kLandD deep
// [m at 1.78 m] (pelvis >= 0.35 m), held kLandHold [s], then up over kLandUp [s]. Standing pelvis kStandZ [m at
// 1.78 m] (measured). Stiff legs would turn the touch speed into a bounce instead of absorbing it.
constexpr double kLandA = 20.0, kLandD = 0.5, kLandHold = 0.3, kLandUp = 1.0, kStandZ = 0.92;
// Leg joint viscosity share while braking (L0's damping is implicit and outside the muscle limit: at full share it
// stops the body within a few centimetres, harder than muscles can; muscles brake within their eccentric limit)
constexpr double kLandZm = 0.1;
// A hop within this time after the touch keeps the landing profile [s]
constexpr double kLandKeepT = 0.6;
// Rise after the braking hold [m/s] (pelvis up at most 0.5 m/s after the stop); extra time allowed [s]
constexpr double kLandRise = 0.3, kLandRiseT = 1.5;
constexpr double kLandFb = 1.0;  // crouch per metre of the real hip over the wanted height in the rise
constexpr double kLandTrunkUp = 0.35;  // the rise waits for the trunk pitch below this [rad]
constexpr double kHardKeepT = 1.5;  // legs kept bent this long after a hard touch when the fall is sure [s]
// Flight pose [rad]: knee, hip, ankle dorsiflexion (its limit 20 deg), trunk lean, arm flexion
constexpr double kAirKnee = 1.15, kAirHip = 0.75, kAirAnkle = 0.10, kAirTrunk = 0.15, kAirArmFlex = 0.7;
// Braking ends with the pelvis this high at least [m at 1.78 m]: with the 20 deg ankle limit the thigh is level at
// knee 110 (pelvis 0.47 m), and a lower stop takes the knees to their 135 deg limit. In a deep squat the loaded shank
// turns the hip outward about the thigh axis (foot at its eversion limit, sliding in): knees out, feet together or
// crossed; this height stays clear of it.
constexpr double kLandPelMin = 0.62;
// Flight by the predicted touch speed sqrt(vz^2 + 2 g h_feet): above kAirCurlV (6 m drop) or with the
// chest tilted past kAirTilt [rad] (not feet first) the body curls in the air (ramp kAirCurlT [s]); above kWindV
// (2 m) the arms circle at kWindW [rad/s]. A touch at kLandHardV or more (2.9 m) ends in a forward fall onto the
// hands after the braking (lean kHardLean [m], trunk kHardTrunk [rad], no step).
constexpr double kAirCurlV = 10.8, kAirTilt = 1.05, kAirCurlT = 0.4, kWindV = 6.3, kWindW = 9.0;
// Flight flail: the arms circle (shoulder flexion and abduction a quarter turn apart, elbows follow,
// back and neck in time) and the legs pedal (knee a quarter turn behind the hip: hip flexed = knee flexing, thigh
// back = knee extended), left and right about half a turn apart, rate and size drift with noise. Rates [rad/s]
// (about 1.9 and 2.1 Hz, the cadence of a person flailing; own choice), centre and half-swing [rad] of hip and knee,
// arm flexion and abduction, elbow. The flail fades to the landing pose between kFlailEnd and kFlailFull [s] before
// the touch; a curl (very high or not feet first) starts kCurlLead [s] before the touch, so the flail shows while it
// is far.
constexpr double kFlailWa = 13.0, kFlailWl = 14.0, kFlailRateSd = 0.22, kFlailAmpSd = 0.18, kFlailTau = 0.5;
constexpr double kFlailHip = 0.80, kFlailHipA = 0.45, kFlailKnee = 1.0, kFlailKneeA = 0.55;
constexpr double kFlailShF = 0.9, kFlailShFA = 1.0, kFlailShAbd = 0.75, kFlailShAbdA = 0.4;
constexpr double kFlailEnd = 0.14, kFlailFull = 0.36, kCurlLead = 0.45, kFlailArmKp = 3.0;
constexpr double kLandHardV = 7.5, kHardLean = 0.25, kHardTrunk = 0.5;
// Medium touch (own choice, tuned on the benches): crouch to the stop, then the curl takes the body (roll)
constexpr double kRollV0 = 7.0, kRollV1 = 9.1, kRollCurlT = 0.0;
constexpr double kPredOut = 0.5;      // fall judgement: capture point outside the support by more than this [m] ...
constexpr double kPredXi = 0.7;        // ... and lies farther than this from the CoM (a 1.78 m body steps ~0.6 m) [m]
constexpr double kPredLean = 0.75, kPredFwdV = 1.5;  // forward stumble judged a fall: chest lean [rad], CoM [m/s]
// Side stumble judged a fall: chest lean to the side [rad], CoM speed that way [m/s]; set between the peaks of a side
// push that recovers and one that falls, so the fall is judged earlier than by the capture point
constexpr double kPredLatLean = 0.70, kPredLatV = 1.9;
constexpr double kPredHold = 0.5;      // the judgement lapses after this long in mode 0 without the fall condition [s]
constexpr double kLegHip = 0.15, kLegKnee = 0.2;  // leg targets of a stiff-legged fall [rad]
constexpr double kLieAbd = 0.6;       // lying: arm abduction [rad], the arms rest on the floor beside the body
constexpr double kLegRelaxT = 0.12;   // lying: leg tone decay time constant [s]
constexpr double kLimbTone = 0.25;    // lying: tone floor of arms and legs (lower: a limb stays propped up)
constexpr double kSitHip = 0.8, kSitKnee = 1.3;  // backward fall leg targets [rad]
// Side fall leg targets [rad]: fall-side hip, knee; far hip, knee
constexpr double kSideHip = 0.9, kSideKnee = 1.7, kSideHipFar = 0.6, kSideKneeFar = 1.0;
// Early seat: hips share x kSitHip1, knees share x kSitKnee1 ramped over kSitT, back round; per-leg share drawn
// from kSitShare0..1 (hips rise to 65-80 deg within 0.5 s and hold, knees 46-108 deg)
constexpr double kSitHip1 = 1.3, kSitKnee1 = 1.7, kSitBack = 0.3, kSitT = 0.15;  // per-leg share: fall_sit_share
// Lying after a backward fall: knees bent 100-130 deg with the feet on the floor
// Seat after the pelvis touch: thigh direction from the downward vertical [rad] (2.2 = knee 36 deg above the
// horizontal), knee [rad] (100-130 deg), leg stiffness share (heavy legs), blend time [s], trunk angle that
// counts as a touch [rad]
constexpr double kSitLieT0 = 0.2, kSitLieT = 0.5, kSitLieHip = 0.1, kSitLieKnee = 0.1;  // seat -> lying legs
constexpr double kThighDown = 2.2, kSitKneeLie = 1.95, kSitLegHeavy = 0.3, kSitWorldT = 0.1, kSitTouchTrunk = 0.785;
constexpr double kSitHoldT = 0.5, kSitLegRest = 0.1;  // lying seat: hold time [s], final leg stiffness share
constexpr double kPreArmKp = 2.0;     // arm stiffness in the reaction latency (the arms do not fly up at the shove)
constexpr double kReachT = 0.15;       // arm reach build-up [s]
constexpr double kArmFwd = 0.5, kArmBack = 0.2, kArmLat = 1.0, kArmBackAbd = 0.45, kElbow = 0.35;  // arm targets [rad]
// Catch: global strength from the CoM speed between kCatchV0
// and kCatchV1 [m/s], falling with kCatchT [s]; an arm off the floor gains strength with kArmRiseT, on the floor
// it yields by 1 per kArmYieldT [s] down to the global; the ray leaves the shoulder shifted kRayFwd x forwardness
// of the fall [m], with kRayDown of the speed added downward; the palm aims kLeadT ahead of the shoulder motion;
// shoulder-palm length kept within kReachMin..kReachMax of the arm length
constexpr double kCatchV0 = 0.3, kCatchV1 = 1.5, kCatchT = 0.4, kArmRiseT = 0.1, kArmYieldT = 1.0;
constexpr double kHoldShare = 0.25;  // after the touch the palm target moves this share to where it landed
// Roll curl: pelvis angular speed on / off [rad/s], ramp [s], hip and knee targets [rad], knee swing [rad]
// Tumble: lower foot below kTumbleZ (one stair rise 0.15 m below the stand floor less margin) [m at 1.78 m],
// moving CoM [m/s] (a walk is 1-1.5); guard arms: shoulder flexion, abduction, elbow [rad] (tuned on the hand-head
// distance)
// Flat rest place: ground contacts of the body within this height band [m] (half a stair rise of 0.15)
constexpr double kFlatDz = 0.075;
constexpr double kTumbleZ = -0.10, kTumbleV = 0.8, kGuardFlex = 2.3, kGuardAbd = 0.4, kGuardElbow = 2.1;
constexpr double kRollW1 = 4.0, kRollW0 = 2.0, kRollT = 0.15, kRollHip = 1.5, kRollKnee = 1.55, kRollKneeA = 0.3;
// Tumble tuck (thighs to the chest, back round, chin in; inside the AAOS ROM of the joint table: hip 120, knee
// 135, spine 40 + 40, neck 45 deg). The roll curl's 86 / 89 deg hip / knee is a chair: on steep stairs the body sits
// on a tread with the feet braced below and stops; the tuck keeps it rolling down
constexpr double kTuckHip = 2.0, kTuckKnee = 2.3, kTuckSpine = 0.6, kTuckNeck = 0.6;
// Far arm of a side fall: its ray starts this far in front of the chest [m] (reaches across the body, not over
// the head: from its own shoulder the target lies across and above, and the arm would swing up overhead)
constexpr double kFarFwd = 0.3;
constexpr double kHeadPitch = 0.5;  // pelvis x axis horizontal part above this (pitch < 60 deg): heading is it
// Elbow of a hand on the floor gives way at this rate [rad/s] (side fall: elbow 20 -> 122 deg over about 0.9 s
// while the body hangs on the arm)
constexpr double kElbowGive = 1.5;
constexpr double kFallCrouch = 0.25;  // side fall: pelvis lowered by the balancer while it steps [m at 1.78 m]
// Catch ray and reach limits: a reach of at most 0.8 of the arm keeps the elbow bent about 50 deg (46-90 deg in the
// first 0.6 s of a walk-off)
constexpr double kRayFwd = 0.15, kRayDown = 0.5, kLeadT = 0.1, kReachMin = 0.6, kReachMax = 0.8;
constexpr double kArmSide = 1.6;  // lateral falls: fall-side shoulder abduction [rad] (arm out, the body rolls onto it)
constexpr double kArmKp = 3.0, kElbowYield = 0.4, kNeckKp = 3.0;  // stiffness scales in a fall
constexpr double kChin = 0.6, kHeadTurn = 0.9;  // neck targets [rad]: chin to chest (back falls), head turn (forward)
// Forward falls: the neck extends (head up, chin out) instead of flexing: the head reaches the floor after the chest
// and much slower.
constexpr double kNeckFwd = -0.5;
// Car guard: seen when the time to contact is below kCarSeeT [s]; visual simple reaction time 0.213 s (Woods
// et al. 2015); arms rise over kCarRiseT [s]; car half length and the half width of its hitting band [m] (car 4.5 x
// 1.8 m plus a body half width); arm targets [rad]: flexion toward a car in front, abduction of the near arm, elbow
// (own choice)
constexpr double kCarSeeT = 2.0, kCarReactT = 0.21, kCarRiseT = 0.15, kCarHalfLen = 2.25, kCarHalfWid = 1.3;
constexpr double kCarFlex = 1.3, kCarAbd = 1.2, kCarElb = 0.5;
// car_guard: guard pose (forearms in front of the chest and face, elbows bent, wrists back) [rad], used with
// less than kCarGuardT [s] to the contact (own choices, not measured)
constexpr double kCarSwingFlex = 1.0, kCarSwingAbd = 0.6;  // calm dodge arm swing to the escape [rad] (tuned)
// elbows close to the body, upper arms low, forearms up: the forearms cover the face
constexpr double kCarGuardT = 0.5, kCarGuardFlex = 0.6, kCarGuardAbd = 0.15, kCarGuardElb = 2.1, kCarGuardWrist = -0.5;
constexpr double kTurnTc87 = 0.8;  // L2 Avoid's kTurnTc: a dodge decided later turns away and guards
constexpr double kCarBrake = 7.0;      // car deceleration after the hit [m/s^2] (~0.7 g, dry asphalt)
}  // namespace

namespace {
double Rnd(uint64_t* s) {  // xorshift64, the bot's generator
  *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
  return (*s >> 11) * (1.0 / 9007199254740992.0);
}
}  // namespace

namespace {
// The body goes down: the muscles fade within fade [s] from now. Already going down (wounded, then dead): the
// rest of the fade is shortened, the tone keeps its current value (same smoothstep position).
void GoDown(Bot& b, double t, int kind, double fade) {
  if (b.dead_t < 0) {
    b.dead_t = t;
    b.dead_T = fade;
  } else if (kind > b.down) {
    const double x = std::fmin(std::fmax((t - b.dead_t) / b.dead_T, 0.0), 1.0);
    if (x >= 1.0) {
      // Death comes to a body whose relaxation is over (a kill after a wound): it just goes limp - the curl fades
      // from now, no new relaxation (restarting it would jerk the curl from zero tone)
      if (b.dead_lay_t < 0) b.dead_lay_t = t;
    } else {
      b.dead_T = fade;
      b.dead_t = t - x * fade;
    }
  }
  if (kind > b.down) b.down = kind;
}
}  // namespace

void World::Kill(int bot) {  // head bullet or the editor button: no more steps
  Bot& b = bots_[bot];
  b.health = 0;
  b.down_nostep = 1;
  GoDown(b, d_->time, 2, kDeadFade);
}

void World::Shoot(int bot, int body, const double point[3], const double dir[3], double impulse, double damage) {
  Bot& b = bots_[bot];
  const double t = d_->time;
  if (impulse < 0) impulse = param[P_BUL_IMP];
  const double n = std::hypot(dir[0], dir[1]) + 1e-9;
  const double ux = dir[0] / n, uy = dir[1] / n;
  const double imp[3] = {impulse * ux, impulse * uy, 0};
  ApplyImpulse(bot, b.body0 + body, point, imp);
  // a bullet is instantaneous: the whole impulse in one substep (ApplyImpulse spread it over push_t)
  for (int k = 0; k < 3; ++k) b.push_f[k] *= b.push_n;
  b.push_n = 1;
  b.hit_bullet = 1;
  b.wound_body = body;
  b.wound_until = t + kWoundT;
  b.flinch_t0 = t;
  if (body == B_HEAD) { Kill(bot); return; }  // also when immortal
  // Live reaction: a bullet while alive is a "next" bullet (pull along it); the first after calm has no pull.
  const bool next = t < b.bul_until;
  // From the second bullet of a live reaction a horizontal force on the pelvis along the bullet
  // (bul_pull x weight) for a drawn bul_pull_t0..t1 s per bullet, also when the bullet wounds or kills; the legs
  // answer it with steps (World::StepFrame applies it).
  if (next && param[P_BUL_PULL] > 0) {
    // One pull; a new bullet sets its end anew (no sum, no extension beyond the drawn term)
    if (t >= b.pull_until) b.pull_t0 = t;  // a new pull episode (the ramp starts here)
    b.pull_until = t + param[P_BUL_PULL_T0] + (param[P_BUL_PULL_T1] - param[P_BUL_PULL_T0]) * Rnd(&b.rng);
    b.pull_dir[0] = ux;
    b.pull_dir[1] = uy;
    b.pull_k = 1.0;
  }
  // Health: immortal keeps it (every bullet acts like the first on a healthy body)
  if (param[P_IMMORTAL] < 0.5) b.health = std::fmax(0.0, b.health - damage);
  // A trunk bullet that wounds or kills leaves the steps on (the body falls on the move, pulled along the bullet)
  if (param[P_BUL_DEAD_NOSTEP] > 0.5) b.down_nostep = 1;
  if (b.health <= kDeadHp) { b.health = 0; GoDown(b, t, 2, kDeadFade); return; }
  // hurt_fade: the wounding hit fades the strength (one factor, weight support included) within hurt_fade instead of
  // kHurtFade; a short fade drops the pelvis quickly, while a long one keeps the body stepping upright under the pull
  // until it pivots over a heel
  if (b.health < kHurtHp || b.down) {
    GoDown(b, t, 1, param[P_HURT_FADE] > 0 ? (double)param[P_HURT_FADE] : kHurtFade);
    return;
  }  // wounded: settles and falls, not dead
  const double hs = b.body.height / 1.78;
  b.bul_n = next ? b.bul_n + 1 : 1;
  b.bul_t = t;
  b.bul_until = t + kLive0 + (kLive1 - kLive0) * Rnd(&b.rng);
  b.bul_body = body;
  b.bul_dir[0] = ux;
  b.bul_dir[1] = uy;
  b.budget = param[P_STEP_BUDGET];  // each bullet gives the step budget back
  // Step reserve of the reaction: bul_steps per bullet, every bullet gives them back; the balancer counts it down and
  // gives up when a step is needed and none is left.
  b.in.steps_left = (int)std::lround(param[P_BUL_STEPS]);
  b.steps = 0;
  if (next) {
    b.hold_lean = 0;  // a held first-bullet CoM offset on top of the lean would cost extra steps
    b.lean_until = t + kLeanT0 + (kLeanT1 - kLeanT0) * Rnd(&b.rng);
    b.lean_mag = (kLean0 + (kLean1 - kLean0) * Rnd(&b.rng)) * hs * param[P_BUL_LEAN];
  }
  // Wound point in the chest frame (the hand and the gaze go there; a trunk wound moves with the chest).
  const double* cp = d_->xpos + 3 * (b.body0 + B_CHEST);
  const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
  const double r[3] = {point[0] - cp[0], point[1] - cp[1], point[2] - cp[2]};
  for (int a = 0; a < 3; ++a) b.bul_c[a] = Rc[a] * r[0] + Rc[3 + a] * r[1] + Rc[6 + a] * r[2];
  if (body >= 0) {
    const double* bp = d_->xpos + 3 * (b.body0 + body);
    const double* Rb = d_->xmat + 9 * (b.body0 + body);
    const double rb[3] = {point[0] - bp[0], point[1] - bp[1], point[2] - bp[2]};
    for (int a = 0; a < 3; ++a) b.bul_b[a] = Rb[a] * rb[0] + Rb[3 + a] * rb[1] + Rb[6 + a] * rb[2];
  }
  const double* R = d_->xmat + 9 * b.body0;
  const double hx = R[0], hy = R[3];
  const bool trunk = body <= B_CHEST, arm = body >= B_UARM_L && body <= B_HAND_R, leg = body >= B_THIGH_L;
  // Flinch: the trunk flexes around the wound (toward the bullet's origin) for trunk hits.
  b.flinch = trunk ? Clamp(-(ux * hx + uy * hy), -1.0, 1.0) : 0.0;  // sign and share; shapes in Reactions
  // Twist toward the wound: the sign of the bullet's moment about the vertical through the chest.
  const double tz = r[0] * uy - r[1] * ux;
  b.twist_w = trunk ? kTwist * Clamp(tz / 0.08, -1.0, 1.0) : 0.0;
  // Hand to the wound (not every time): the near arm; a wound in an arm -> the other arm.
  // Both hands for a trunk or leg wound (the near one on it), the other hand for an arm wound
  {
    b.arm_side = arm ? (body >= B_UARM_R ? 0 : 1) : (b.bul_c[1] >= 0 ? 0 : 1);
    b.arm_both = !arm && param[P_BUL_HANDS] > 1.5;
    b.arm_top = Rnd(&b.rng) < 0.5;
    const bool holding = b.arm_t0 >= 0 && t >= b.arm_t0 && t < b.arm_t1;
    if (!holding) b.arm_t0 = t + kArmD0 + (kArmD1 - kArmD0) * Rnd(&b.rng);  // already there: keep holding
    b.arm_t1 = std::fmax(b.arm_t1, t + kArmH0 + (kArmH1 - kArmH0) * Rnd(&b.rng));
  }
  b.whip_a = (0.8 + 0.4 * Rnd(&b.rng)) * param[P_BUL_WHIP];
  for (int s = 0; s < 2; ++s) {  // free-arm swing: own phase and elbow bend per bullet
    b.swing_ph[s] = 2.0 * M_PI * Rnd(&b.rng);
    b.swing_el[s] = 0.3 + 0.6 * Rnd(&b.rng);
  }
  if (arm) b.arm_hit_t[body >= B_UARM_R ? 1 : 0] = t;  // hit arm: time of the hit
  b.whip_ts = 0.9 + 0.2 * Rnd(&b.rng);
  b.look_wound = Rnd(&b.rng) < 0.5;
  if (leg) {
    const int s = body >= B_THIGH_R ? 1 : 0;
    if (b.leg_hits[s] == 0) {
      b.leg_t0[s] = t;
      b.leg_shin[s] = body != B_THIGH_L && body != B_THIGH_R;
    }
    b.leg_hits[s]++;
    b.leg_t[s] = t;
    // The leg gives way kLegGiveT after the first bullet (shin x kLegShinF), each further bullet brings it earlier;
    // the reaction stays live through the hobble
    const double tg = b.leg_t0[s] + kLegGiveT * (b.leg_shin[s] ? kLegShinF : 1.0) / b.leg_hits[s];
    b.leg_give[s] = std::fmax(tg, t + kLegDelay);
    b.bul_until = std::fmax(b.bul_until, b.leg_give[s] + kLegHobbleT);
    // the hand stays on the leg wound through the hobble
    b.arm_t1 = std::fmax(b.arm_t1, b.leg_give[s] + kLegHobbleT);
  }
}

void World::Reactions(Bot& b, double t) {
  double* T = b.target;
  const Percept& P = b.pc[b.cur ^ 1];
  // L0 weight support of the legs back to full every frame (the seat lifts it after the pelvis touch); back and neck
  // too until the body lies (lying lowers them)
  for (int j = J_HIP_FLEX_L; j < J_COUNT; ++j) b.gsup[j] = 1.0;
  if (b.lie_t0 < 0) {
    for (int j = 0; j < J_SH_FLEX_L; ++j) b.gsup[j] = 1.0;
    b.neck_quiet = 0;
    b.quiet_t = t;
  }
  // Bullet: local weakening and flinch (trunk flexion), decaying.
  if (t < b.wound_until) {
    for (int j = 0; j < J_COUNT; ++j) {
      const int ch = m_->jnt_bodyid[b.jnt0 + j] - b.body0;
      const int pa = m_->body_parentid[b.body0 + ch] - b.body0;
      // not the spine yaw (a weak yaw lets an off-centre impulse turn the shoulders far)
      if ((ch == b.wound_body || pa == b.wound_body) && j != J_LUMBAR_YAW && j != J_CHEST_YAW) b.kp[j] *= kWoundKp;
    }
  }
  if (b.flinch != 0 || b.twist_w != 0) {
    const double s = t - b.flinch_t0, e = std::exp(-s / kFlinchT), x = s / b.whip_ts;
    auto bump = [](double v, double c, double wd) { const double u = (v - c) / wd; return std::exp(-u * u); };
    double f;
    bool done = e < 4e-3;
    if (b.bul_body == B_CHEST) {  // whip: back extension, then flexion (signs: flinch +1 = bullet from the front)
      f = b.flinch * b.whip_a * (-kWhipExt * bump(x, kWhipExtT, kWhipExtW) + kWhipFlex * bump(x, kWhipFlexT, kWhipFlexW));
    } else {  // belly: fold forward, held while the hands hold the wound, then released with the hands
      const double hold = t < b.arm_t1 ? 1.0 : std::exp(-(t - b.arm_t1) / 0.8);
      f = b.flinch * b.whip_a * kBellyFlex * (1.0 - std::exp(-x / kBellyT)) * (kBellyHold + (1 - kBellyHold) * std::exp(-std::fmax(0.0, x - 0.5) / 0.4)) * hold;
      done = hold < 4e-3;
    }
    const double y = b.twist_w * e;
    // The spine bends a little over the wound (guarding): toward its side (lateral flexion by the hit point's offset
    // from the mid-line, full at kSideW) and, for a chest wound, toward its face (the belly has its fold); in with
    // the belly fold time, held while the hands hold the wound, then released with them.
    if (param[P_BUL_SIDE] > 0 || param[P_BUL_GUARD] > 0) {
      const double hold = t < b.arm_t1 ? 1.0 : std::exp(-(t - b.arm_t1) / 0.8);
      const double env = (1.0 - std::exp(-x / kBellyT)) * hold;
      const double sd = param[P_BUL_SIDE] * Clamp(b.bul_c[1] / kSideW, -1.0, 1.0) * env;  // + = wound on the left
      T[J_LUMBAR_ROLL] -= 0.5 * sd;  // roll + leans right
      T[J_CHEST_ROLL] -= 0.5 * sd;
      if (b.bul_body == B_CHEST) f += b.flinch * param[P_BUL_GUARD] * env;
      done = done && hold < 4e-3;
    }
    T[J_LUMBAR_PITCH] += 0.5 * f;
    T[J_CHEST_PITCH] += 0.5 * f;
    T[J_LUMBAR_YAW] += 0.4 * y;
    T[J_CHEST_YAW] += 0.6 * y;
    if (done && s > 1.0) b.flinch = b.twist_w = 0;
  }
  // The backward seat as a step of its own: it runs for a live fall and for a body going down (wounded or dead)
  auto fall_seat = [&]() {
    const double om = std::sqrt(9.81 / std::fmax(P.com[2], 0.3));
    const double xi_d = std::hypot(P.com_vel[0], P.com_vel[1]) / om;
    // Backward fall: sit down early (hips flex within 0.1-0.3 s, the pelvis is seated in about 0.7 s while the body
    // steps back); straight hips at the give-up would throw the feet up. When the capture point is far out backward the
    // hips and knees blend toward a seat (each leg its own drawn share) on top of the balancer's targets, before and
    // after the give-up.
    {
      const double hx = P.pelvis_R[0], hy = P.pelvis_R[3], hn = std::hypot(hx, hy) + 1e-9;
      const double vb = -(P.com_vel[0] * hx + P.com_vel[1] * hy) / hn, vh = std::hypot(P.com_vel[0], P.com_vel[1]);
      const bool back_out = b.cp_out > param[P_FALL_SIT_OUT] && xi_d > kPredXi && vb > 0.7 * vh;
      if (param[P_FALL_SIT] > 0 && back_out && b.sit_t0 < 0 && b.lie_t0 < 0) {
        b.sit_t0 = t;
        b.sit_touch = -1;
        b.sit_tp = -10;
        b.sit_h[0] = hx / hn;
        b.sit_h[1] = hy / hn;
        for (int s = 0; s < 2; ++s) b.sit_share[s] = param[P_FALL_SIT_SHARE] + (1 - param[P_FALL_SIT_SHARE]) * Rnd(&b.rng);
      }
      if (b.sit_t0 >= 0 && b.mode == 0 && !back_out && t - b.sit_t0 > kPredHold) b.sit_t0 = -1;  // recovered
      // After the pelvis touches the floor (or the trunk passes 45 deg back) the thighs are led in the world: each
      // keeps its direction to the ground (knee up and forward), so the hip opens as the pelvis lies back; knees bent,
      // legs soft (heavy). Holding the hip flexion against the pelvis would lift the legs with the lying trunk (a
      // "candle"); the target is feet on the floor, knees 100-130 deg.
      if (b.sit_t0 >= 0) {
        const double* Rp = d_->xmat + 9 * (b.body0 + B_PELVIS);
        // pelvis tilt back from the vertical in the plane of the fall heading (pelvis z axis)
        const double tp = std::atan2(-(Rp[2] * b.sit_h[0] + Rp[5] * b.sit_h[1]), Rp[8]);
        const double* Pp = d_->xpos + 3 * (b.body0 + B_PELVIS);
        const double* Hh = d_->xpos + 3 * (b.body0 + B_HEAD);
        const double tb = std::atan2(-((Hh[0] - Pp[0]) * b.sit_h[0] + (Hh[1] - Pp[1]) * b.sit_h[1]), Hh[2] - Pp[2]);
        if (b.sit_touch < 0) {
          bool pel = tb > kSitTouchTrunk;
          for (int i = 0; i < d_->ncon && !pel; ++i) {
            const mjContact& c = d_->contact[i];
            const int g = Ground(c.geom1) ? c.geom2 : (Ground(c.geom2) ? c.geom1 : -1);
            if (g >= 0 && m_->geom_bodyid[g] == b.body0 + B_PELVIS) pel = true;
          }
          if (pel) b.sit_touch = t;
        }
        if (b.sit_touch >= 0) {
          const double wt = Clamp((t - b.sit_touch) / kSitWorldT, 0.0, 1.0);
          // the largest pelvis tilt since the touch: a hip target from the current tilt would rise when the trunk lifts
          // and flex the hip further (heavy legs: the trunk sits up, then falls back with the legs up)
          b.sit_tp = std::fmax(b.sit_tp, tp);
          for (int s = 0; s < 2; ++s) {
            const int hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
            const double sh = b.sit_share[s];
            const double hip_w = Clamp(sh * kThighDown - b.sit_tp, 0.0, 1.6);  // hip = thigh from down - pelvis tilt
            // From kSitLieT0 after the touch the seat pose blends into the lying pose over kSitLieT (the legs lie):
            // with the knees held at 1.95 rad and the thigh laid flat by a large pelvis tilt the folded legs would
            // stand up off the floor
            const double wl = Clamp((t - b.sit_touch - kSitLieT0) / kSitLieT, 0.0, 1.0);
            T[hp + 0] += wt * (hip_w + wl * (kSitLieHip - hip_w) - T[hp + 0]);
            // (a knee target from the actual thigh angle chases the legs into kicks; fixed knee here)
            T[hp + 3] += wt * (sh * kSitKneeLie * (1.0 - wl) + wl * kSitLieKnee - T[hp + 3]);
            T[hp + 1] += wt * (0.0 - T[hp + 1]);  // no abduction and rotation: a free leg would swing up sideways
            T[hp + 2] += wt * (0.0 - T[hp + 2]);
            T[hp + 4] += wt * (0.0 - T[hp + 4]);
            // heavy legs: the stiffness falls to kSitLegHeavy at the touch and to kSitLegRest after kSitHoldT (the legs
            // settle by their weight with the feet on the floor)
            const double rel = Clamp((t - b.sit_touch - kSitHoldT) / kSitHoldT, 0.0, 1.0);
            const double kl = kSitLegHeavy + (kSitLegRest - kSitLegHeavy) * rel;
            for (int k = 0; k < 6; ++k) b.kp[hp + k] *= 1.0 - wt * (1.0 - kl);
            // The legs are heavy for real: no L0 weight support after the touch (with it the seated legs float and the
            // feet bob up and down)
            for (int k = 0; k < 6; ++k) b.gsup[hp + k] = 1.0 - wt * (1.0 - param[P_FALL_SIT_GSUP]);
          }
        }
      }
      if (b.sit_t0 >= 0 && b.lie_t0 < 0 && b.sit_touch < 0) {
        // a held seat (hip flexion = thigh forward minus trunk back, so a hip target following the trunk would lift the
        // thighs and the feet as the trunk lies down): hips and knees ramp to a fixed seat
        const double x = Clamp((t - b.sit_t0) / kSitT, 0.0, 1.0);
        const double ws = param[P_FALL_SIT] * Clamp(x * 3.0, 0.0, 1.0);
        for (int s = 0; s < 2; ++s) {
          const int hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
          // Before the give-up only a loaded leg seats; the swing leg stays the balancer's (the body may step back
          // while it sits down)
          if (param[P_FALL_SIT_SWING] > 0.5 && b.mode != 2 && b.down == 0 && !b.loaded[s]) continue;  // live only
          const double sh = b.sit_share[s];
          T[hp + 0] += ws * (sh * kSitHip1 * x - T[hp + 0]);
          // knees slower than the hips (fast knees drop the pelvis to the floor too early)
          const double xk = Clamp((t - b.sit_t0) / param[P_FALL_SIT_KNEE_T], 0.0, 1.0);
          T[hp + 3] += ws * (sh * kSitKnee1 * xk - T[hp + 3]);
          // the legs lower the seat (eccentric): after the give-up the balancer leaves them at 0.3, too weak, and the
          // seat would drop to the floor early and roll the body over the shoulders
          if (b.mode == 2) for (int k = 0; k < 6; ++k) b.kp[hp + k] *= param[P_FALL_SIT_LEGS];
        }
        T[J_LUMBAR_PITCH] += ws * (kSitBack - T[J_LUMBAR_PITCH]);
        // Level the pelvis while seating: the balancer's backward step can leave the body falling over one leg with the
        // pelvis rolling far; the hip abduction of each leg lifts its pelvis side (roll r > 0 = left side up)
        {
          const double* Rp = d_->xmat + 9 * (b.body0 + B_PELVIS);
          const double r = std::asin(Clamp(Rp[7], -1.0, 1.0));
          const double k = param[P_FALL_SIT_ROLL] * ws;
          T[J_HIP_ABD_L] -= k * r;
          T[J_HIP_ABD_R] += k * r;
        }
      }
    }
  };
  // Death: the muscle tone (L0: torques, damping, weight support) fades to zero within dead_T and no new steps start;
  // no protective reaction, the body settles and falls by itself (the upper layer is off while dead).
  if (b.dead_t >= 0) {
    const double x = Clamp((t - b.dead_t) / b.dead_T, 0.0, 1.0);
    // One factor falls linearly from 1 to 0 over the relaxation, from the first frame, and scales the whole strength:
    // stiffness, damping, weight support (L0 tone) and the joints' added inertia (a dead joint is light and free); the
    // stay-upright help is off in the first frame; no steps. (A smoothstep keeps the tone high at first, the legs stand
    // stiff and the body topples straight.)
    b.in.tone = 1.0 - x;
    // Wounded (long relaxation): the balancer goes on stepping under the fading strength and the support of the legs
    // fades with it, its pelvis height sinks with the relaxation: the body settles on the move (a full step back after
    // the wounding bullet, then a fast sink). The steps end when the pelvis is below half the stance height or the
    // balancer gives up; then the legs fold and the body curls.
    const double hs = b.body.height / 1.78;
    const double pz = (d_->xpos[3 * (b.body0 + B_PELVIS) + 2] - b.gz);
    // walk_vmax: a wounded body carried forward faster than this [m/s] ends the steps (the legs fold): otherwise the
    // walk under the bullet pull speeds up and the body pivots over the knees instead of kneeling on the move
    const double vfw = (P.com_vel[0] * P.pelvis_R[0] + P.com_vel[1] * P.pelvis_R[3]) /
                       (std::hypot(P.pelvis_R[0], P.pelvis_R[3]) + 1e-9);
    const bool fast = param[P_WALK_VMAX] > 0 && vfw > param[P_WALK_VMAX];
    if (b.dead_walk_t < 0 && (b.down != 1 || b.mode == 2 || x >= 1.0 || pz < kWalkEndPel * hs || fast)) {
      b.dead_walk_t = t;
      // curl_fade: a wounded body whose steps end loses the rest of its strength (one factor: drive, damping and
      // weight support) within curl_fade s, not over the rest of the relaxation: with most of the tone left when the
      // curl flexes the hips, the planted knee holds and the body vaults over it; the strength fades together
      if (param[P_CURL_FADE] > 0 && b.down == 1 && x < 1.0 && b.dead_T > param[P_CURL_FADE]) {
        b.dead_T = param[P_CURL_FADE];
        b.dead_t = t - x * b.dead_T;
      }
    }
    const bool walk = b.dead_walk_t < 0;
    // The legs keep a share of their strength while the pelvis is high (L0 leg tone floor), so the body sits down
    // instead of dropping: the pelvis meets the floor at about 1-2 m/s instead of the free-fall speed, and the leg that
    // takes the impact can open. The share fades to zero over kLegHoldW above kLegHoldPel. Head bullet or
    // kill (the legs are the free part): kLegHoldHead; a body hit: less (the trunk collapse is sensitive to it).
    {
      const double hold = ((b.down_nostep || (b.wound_body == B_HEAD && b.hit_bullet)) ? kLegHoldHead
                                                                                         : (double)param[P_LEG_HOLD_BODY]);
      b.leg_tone = walk ? 0.0 : hold * Clamp((pz - kLegHoldPel * hs) / (kLegHoldW * hs), 0.0, 1.0);
    }
    // wounded, steps over: the curl starts at once on the remaining strength (the body sits, falls back and curls;
    // waiting for the end of the relaxation would let it lie back straight and hit the head hard)
    const bool curl = x >= 1.0 || (b.down == 1 && !walk);
    b.arm_tone = 0;
    // curl_arm_hold: a body hit going down keeps this share of its arm strength while the pelvis is above kCatchPel,
    // so the hands reach the floor; a kill soon after the wound would otherwise drop the tone to almost zero and leave
    // the arms near the trunk
    if (param[P_CURL_ARM_HOLD] > 0 && !(b.wound_body == B_HEAD && b.hit_bullet) && pz > kCatchPel * hs)
      b.arm_tone = param[P_CURL_ARM_HOLD];
    b.in.no_step = walk ? 0 : 1;
    b.in.steps_left = -1;
    // curl_nosup: once the steps are over the legs no longer carry the weight (L0 support torques off): the stance
    // knee folds under the body instead of holding it while the trunk curls (a held knee lets the body vault over the
    // just planted leg)
    if (param[P_CURL_NOSUP] > 0 && !walk && b.down >= 1) b.stand_leg[0] = b.stand_leg[1] = 0;
    // down_nosup: from the wounding hit on (steps or not) the legs carry no weight: the pelvis sinks from the first
    // frame. Wounded bodies only: on a dead one (head shot, kill) the leg hold above sits the body down, and without
    // it the head lands much harder
    if (param[P_DOWN_NOSUP] > 0 && b.down == 1) b.stand_leg[0] = b.stand_leg[1] = 0;
    b.assist_f[0] = b.assist_f[1] = 0;
    for (int j = 0; j < J_COUNT; ++j) {
      if (b.arm[j] > 0) b.arm[j] = std::fmax(b.arm[j] * b.in.tone, kArmDead);
      b.limf[j] = 1.0 + (b.limf[j] - 1.0) * b.in.tone;
    }
    fall_seat();  // a body going down backward sits too (else it falls almost like a board)
    // The legs fold under the body over the relaxation: the nearly straight standing knee (16 deg) carries the weight
    // on the bones (a few N m of knee torque, measured), so fading strength alone leaves the legs straight and the body
    // topples like a board. The leg targets go from the stance toward a deep squat over the relaxation time (head
    // bullet: knees 38 65 95 120 140 deg at 0.2-0.6 s, hips 6 17 36 56 75) while the same tone fades.
    // Arms to the face (head bullet from the front or side, kill) or spread (head bullet from behind)
    const bool head = b.wound_body == B_HEAD && b.hit_bullet;
    const double hn = std::hypot(b.hit_dir[0], b.hit_dir[1]) + 1e-9;
    const double hx = P.pelvis_R[0], hy = P.pelvis_R[3], fn = std::hypot(hx, hy) + 1e-9;
    const bool behind = head && (b.hit_dir[0] * hx + b.hit_dir[1] * hy) / (hn * fn) > 0.5;
    const bool face_arms = head || b.down_nostep;
    double spine_full = 0;  // wounded going down on the back: neck and spine at full strength (share)
    if (!curl) {
      // dead_buckle: once the steps are over the legs fold within dead_buckle s, not over the whole relaxation: a
      // slow fold keeps the planted leg as a strut and the body vaults over it
      const double s = param[P_DEAD_BUCKLE] > 0 && !walk && b.dead_walk_t >= 0
                           ? std::fmax(x, Clamp((t - b.dead_walk_t) / param[P_DEAD_BUCKLE], 0.0, 1.0))
                           : x;
      const bool ank = b.body.ankle_dorsi >= kAnkleSw && param[P_FOLD_A] > 0;
      const double fh = ank ? kFoldHipA : kFoldHip, fa = ank ? b.body.ankle_dorsi * M_PI / 180.0 : kFoldAnkle;
      for (int f = 0; f < 2 && !walk; ++f) {
        const int h = f ? J_HIP_FLEX_R : J_HIP_FLEX_L;
        T[h] = std::fmax(T[h], kFoldHip0 + (fh - kFoldHip0) * s);
        T[h + 3] = std::fmax(T[h + 3], kFoldKnee0 + (kFoldKnee - kFoldKnee0) * s);
        T[h + 4] = std::fmax(T[h + 4], fa * s);
      }
      // The bullet reaction goes on under the fading strength: the arms start to the face (head bullet) or up and out
      // (arm from the trunk line 17 24 37 78 106 deg at 0.1-0.5 s, elbow 23 40 56 56 59): they rise while there is
      // strength and fall with the body.
      if (t - b.dead_t < kFrameDt) b.dead_arm_lag = (Rnd(&b.rng) - 0.5) * 2.0 * kDeadArmLag;
      if (face_arms) {
        DeadArms(b, t, behind ? 1 : 0);
      } else {
        // dead_fwd_arm: a body dying on the move forward pivots over its knees; with the up-and-out pose the hands
        // stay at hip level and the head meets the floor first. The arms go forward-down instead, ahead of the head.
        const double vf = (P.com_vel[0] * hx + P.com_vel[1] * hy) / fn;
        const bool fwd = param[P_DEAD_FWD_ARM] > 0 && vf > kDeadFwdV;
        // dead_arm_hold: a wounded (still alive) body keeps this share of the arm strength, so the hands take the floor
        // before the head
        if (fwd && b.down == 1) b.arm_tone = param[P_DEAD_ARM_HOLD];
        // the shoulder angle is relative to the trunk: arm toward the floor = chest pitch, plus dead_fwd_arm ahead
        const double* Rcp = d_->xmat + 9 * (b.body0 + B_CHEST);
        const double cpit = std::atan2((Rcp[2] * hx + Rcp[5] * hy) / fn, Rcp[8]);  // + = chest top forward [rad]
        const double af = fwd ? Clamp(cpit + param[P_DEAD_FWD_ARM], 0.3, 2.6) : kDeadArmFlex,
                     aa = fwd ? kDeadFwdAbd : kDeadArmAbd,
                     ae = fwd ? kDeadFwdElb : kDeadArmElb;
        const double a = Clamp((t - b.dead_t) / kDeadArmT, 0.0, 1.0);
        for (int sd = 0; sd < 2; ++sd) {
          const int sh = sd ? J_SH_FLEX_R : J_SH_FLEX_L;
          T[sh] += a * (af - T[sh]);
          T[sh + 1] += a * (aa - T[sh + 1]);
          T[sh + 3] += a * (ae - T[sh + 3]);
        }
      }
      b.quiet_t = t;
    } else {
      // After the relaxation: not a rag at once - the roll curl (hips to the chest, knees bent, arms
      // bent before the chest and face, chin to the chest) on a small strength rising from zero, until the body is
      // quiet; then the strength goes and the body stays as it lies. No catching with the hands.
      const double vq = std::sqrt(P.com_vel[0] * P.com_vel[0] + P.com_vel[1] * P.com_vel[1] +
                                  P.com_vel[2] * P.com_vel[2]);
      // own quiet timer: the shared quiet_t restarts every frame until the lying detector fires, and a body on its
      // knees face down never counts as lying (the curl would hold that pose)
      if (b.dead_quiet_t < 0 || vq > kQuietV) b.dead_quiet_t = t;
      if (b.dead_lay_t < 0 && t - b.dead_quiet_t > kDeadQuietT) b.dead_lay_t = t;
      // face down with the head on the floor: no curl (it pulls the knees under the body: on the knees face down, or
      // rocking, never quiet) - lie down at once
      const double* Rch = d_->xmat + 9 * (b.body0 + B_CHEST);
      const double hzc = (d_->xipos[3 * (b.body0 + B_HEAD) + 2] - b.gz);
      // head on the floor and face down, or below the pelvis (on the knees with the spine curled the chest axis
      // points back, not down; not on the back: there it is a back roll, the hips open)
      if (b.dead_lay_t < 0 && hzc < kHeadLow * hs && (Rch[6] < -kFaceDown || (hzc < pz - kHeadBelow * hs && Rch[6] <= 0.0)))
        b.dead_lay_t = t;
      const double tc0 = x >= 1.0 && b.down == 2 ? b.dead_t + b.dead_T : std::fmin(b.dead_walk_t, b.dead_t + b.dead_T);
      // the curl rises over kDeadCurlRise: from a dead body's zero tone the tone rises; a wounded body that stops
      // stepping still has strength: its tone stays continuous and the targets move to the curl (a tone ramp from
      // zero would drop it to zero and back within a few frames: a jerk)
      const double cr = Clamp((t - tc0) / kDeadCurlRise, 0.0, 1.0);
      // the curl lasts at most kDeadCurlMax: a curled body on its side can keep rocking (CoM never quiet)
      if (b.dead_lay_t < 0 && t - tc0 > kDeadCurlMax) b.dead_lay_t = t;
      const bool early = tc0 < b.dead_t + b.dead_T - 1e-9;
      double c = early ? 1.0 : cr;
      static const int kj[] = {J_HIP_FLEX_L, J_HIP_FLEX_R, J_KNEE_L, J_KNEE_R, J_SH_FLEX_L, J_SH_FLEX_R,
                               J_SH_ABD_L, J_SH_ABD_R, J_ELBOW_L, J_ELBOW_R, J_NECK_PITCH, J_LUMBAR_PITCH, J_CHEST_PITCH};
      static const double kq[] = {1.6, 1.6, 1.8, 1.8, 1.6, 1.6, 0.3, 0.3, 1.8, 1.8, 0.9, 0.6, 0.6};
      // arms: table stiffness under the curl tone (lying, the balancer cuts the arms to 0.3: with the tone on top the
      // elbow is too weak and the arms lie along the body instead of bending about 100 deg)
      const double up_c = d_->xmat[9 * (b.body0 + B_CHEST) + 6];  // chest forward axis, z: > 0 facing up
      if (b.down == 1 && b.dead_lay_t < 0) spine_full = Clamp((up_c + kBackUp0) / kBackUpW, 0.0, 1.0);
      // dead_knee_first: the hips curl no farther than the knees have folded (share of the knee's way to the curl):
      // hips flexing over held knees pitch the trunk over the planted leg instead of letting the body sink
      double kf = 1.0;
      if (param[P_DEAD_KNEE_FIRST] > 0 && early)
        kf = Clamp(0.5 * (P.q[J_KNEE_L] + P.q[J_KNEE_R]) / kq[2], 0.0, 1.0);
      // dead_upright: and the trunk stays fairly upright for the first dead_upright s of the going down (the knees
      // fold first while the trunk is still upright)
      if (param[P_DEAD_UPRIGHT] > 0 && early)
        kf = std::fmin(kf, Clamp((t - b.dead_t) / param[P_DEAD_UPRIGHT], 0.0, 1.0));
      // curl_catch: a wounded body that went down early keeps the catching arms (forward-down moving forward, else up
      // and out) while the pelvis is still above kCatchPel: with the legs giving way at once the steps end sooner, and
      // the curl would pull the hands to the face on the way down
      const bool catching = param[P_CURL_CATCH] > 0 && early && pz > kCatchPel * hs;
      if (catching) {
        const double hx = P.pelvis_R[0], hy = P.pelvis_R[3], fn = std::hypot(hx, hy) + 1e-9;
        const double vf = (P.com_vel[0] * hx + P.com_vel[1] * hy) / fn;
        const bool fwd = param[P_DEAD_FWD_ARM] > 0 && vf > kDeadFwdV;
        if (fwd) b.arm_tone = std::fmax(b.arm_tone, (double)param[P_DEAD_ARM_HOLD]);
        const double* Rcp = d_->xmat + 9 * (b.body0 + B_CHEST);
        const double cpit = std::atan2((Rcp[2] * hx + Rcp[5] * hy) / fn, Rcp[8]);
        const double af = fwd ? Clamp(cpit + param[P_DEAD_FWD_ARM], 0.3, 2.6) : kDeadArmFlex,
                     aa = fwd ? kDeadFwdAbd : kDeadArmAbd, ae = fwd ? kDeadFwdElb : kDeadArmElb;
        for (int sd = 0; sd < 2; ++sd) {
          const int sh = sd ? J_SH_FLEX_R : J_SH_FLEX_L;
          T[sh] = af;
          T[sh + 1] = aa;
          T[sh + 3] = ae;
          b.kp[sh] = std::fmax(b.kp[sh], (double)b.js[sh].kp);
          b.kp[sh + 1] = std::fmax(b.kp[sh + 1], (double)b.js[sh + 1].kp);
          b.kp[sh + 3] = std::fmax(b.kp[sh + 3], (double)b.js[sh + 3].kp);
        }
      }
      for (int k = 0; k < 13; ++k) {
        if (catching && kj[k] >= J_SH_FLEX_L && kj[k] <= J_WRIST_R) continue;
        const double hk = (kj[k] == J_HIP_FLEX_L || kj[k] == J_HIP_FLEX_R) ? kf : 1.0;
        T[kj[k]] += (early ? cr : 1.0) * hk * (kq[k] - T[kj[k]]);
        if (kj[k] >= J_SH_FLEX_L && kj[k] <= J_WRIST_R) b.kp[kj[k]] = std::fmax(b.kp[kj[k]], (double)b.js[kj[k]].kp);
        // Chin and spine to the chest at table stiffness while the body goes down on its back (chest facing up):
        // with the balancer's lying cut (0.3) the neck is too soft and the head whips back as the back lands. Falling
        // forward the flexed spine only speeds the fall, so it is not stiffened there.
        if (kj[k] == J_NECK_PITCH || kj[k] == J_LUMBAR_PITCH || kj[k] == J_CHEST_PITCH)
          b.kp[kj[k]] += Clamp((up_c + kBackUp0) / kBackUpW, 0.0, 1.0) * std::fmax(b.js[kj[k]].kp - b.kp[kj[k]], 0.0);
      }
      // No back roll: on the back, as the head goes below the pelvis the curl opens (hips and spine extend). The curled
      // back (spine 40 + 40 deg, hips 92) rolls like a ball over the shoulders onto the knees face down; opened, the
      // legs swing down behind and stop the roll.
      const double* Hc = d_->xipos + 3 * (b.body0 + B_HEAD);
      const double* Pp = d_->xpos + 3 * (b.body0 + B_PELVIS);
      const double hx = Hc[0] - Pp[0], hy = Hc[1] - Pp[1], hz = Hc[2] - Pp[2];
      const double ctr = hz / (std::sqrt(hx * hx + hy * hy + hz * hz) + 1e-9);  // cos of the trunk from the vertical
      const double op = up_c > 0.0 ? Clamp((-ctr - kRollCos0) / kRollCosW, 0.0, 1.0) : 0.0;
      for (int j : {J_LUMBAR_PITCH, J_CHEST_PITCH, J_NECK_PITCH}) T[j] += op * (0.0 - T[j]);
      // Fallen forward (chest facing down): the hips hold the trunk at kFwdHip over the thighs, spine straight (the
      // body sits with the trunk about 60 deg forward; the curl hips at the lying cut would let the trunk fall over the
      // upright thighs onto the head)
      // forward: chest facing down and the trunk line past 40 deg (full at 60). The chest axis alone also fires on a
      // curled spine over an upright trunk; the pelvis axis flips on a body rolled back.
      // Only after a bullet from behind: a side head bullet folds forward too, then topples sideways; held hips would
      // make it an L and bring the head down harder.
      const double hdn = std::hypot(b.hit_dir[0], b.hit_dir[1]) + 1e-9;
      const double pfn = std::hypot(P.pelvis_R[0], P.pelvis_R[3]) + 1e-9;
      const bool from_behind = b.hit_bullet && (b.hit_dir[0] * P.pelvis_R[0] + b.hit_dir[1] * P.pelvis_R[3]) /
                                                       (hdn * pfn) > 0.5;
      const double fw = from_behind ? Clamp((kFwdCos0 - ctr) / kFwdCosW, 0.0, 1.0) *
                                          Clamp((-up_c - kFwdUp0) / kFwdUpW, 0.0, 1.0)
                                    : 0.0;
      for (int j : {J_LUMBAR_PITCH, J_CHEST_PITCH}) T[j] += fw * (0.0 - T[j]);
      for (int f = 0; f < 2; ++f) {
        const int h = f ? J_HIP_FLEX_R : J_HIP_FLEX_L;
        T[h] += fw * (kFwdHip - T[h]);
        b.kp[h] += fw * std::fmax(b.js[h].kp - b.kp[h], 0.0);
      }
      // A leg on the floor falls out under its own weight in 0.5-1 s, nothing holds it: the curl leaves the legs
      // alone: knees and ankles always, hips once the chest or head is on the floor (in the fall the curled hips slow
      // the trunk going back; held on the floor they would keep a tight ball); each target is the joint's angle, no
      // weight support. After a bullet from behind the hips still hold the trunk over the thighs fallen forward (fw
      // above). Free legs do not drive a back roll, so the knees need no opening.
      {
        if (b.dead_land_t < 0)
          for (int i = 0; i < d_->ncon; ++i) {
            const mjContact& cn = d_->contact[i];
            const int g = Ground(cn.geom1) ? cn.geom2 : (Ground(cn.geom2) ? cn.geom1 : -1);
            const int bd = g < 0 ? -1 : m_->geom_bodyid[g] - b.body0;
            if (bd == B_CHEST || bd == B_HEAD) b.dead_land_t = t;
          }
        for (int f = 0; f < 2; ++f) {
          const int h = f ? J_HIP_FLEX_R : J_HIP_FLEX_L;
          for (int j = b.dead_land_t >= 0 ? h : h + 3; j <= h + 5; ++j) {
            T[j] = P.q[j];
            if (j == h) T[j] += fw * (kFwdHip - T[j]);
            b.gsup[j] = 0.0;
          }
          // a back roll over the shoulders (trunk past 100 deg): the hips open and the legs swing down behind (free
          // hips let the legs fly over the head onto the 120 deg limit, knees straight, and stay up)
          const double opr = up_c > 0.0 ? Clamp((-ctr - kRollHipCos) / kRollCosW, 0.0, 1.0) : 0.0;
          T[h] += opr * (kOpenHip - T[h]);
          b.kp[h] += opr * std::fmax(b.js[h].kp - b.kp[h], 0.0);
        }
      }
      b.in.tone = std::fmax(1.0 - x, kDeadCurlTone) * c;
      // Quiet: the body goes limp where it lies. The tone fades to zero over kDeadCurlFade and stays zero; no targets
      // pull (each joint's target is its angle), only the passive joints hold. (Laying the body out by pulling hips,
      // knees, spine and neck to zero straightens both legs on the floor; a limp body keeps them uneven.)
      if (b.dead_lay_t >= 0) {
        for (int j = 0; j < J_COUNT; ++j) T[j] = P.q[j];
        const double u = Clamp((t - b.dead_lay_t) / kDeadCurlFade, 0.0, 1.0);
        b.in.tone *= 1.0 - u * u * (3 - 2 * u);
        // the leg hold is for the way down; quiet, it goes with the tone. Kept on, it would hold a killed body in a
        // squat on its hands (targets at the pose)
        b.leg_tone *= 1.0 - u * u * (3 - 2 * u);
        // Propped: limp on the knees face down (hips 120, knees 135 at their stops, pelvis about 40 cm, head and
        // chest on the floor) is a stable tripod, while a dead body should lie flat. Quiet with the pelvis above
        // prop_pel x stance and the head below it: the hips (only) extend on a leg tone floor of prop_tone until the
        // pelvis is down or kPropMax ran out; the knees and the rest stay limp.
        if (param[P_PROP_TONE] > 0 && b.prop_t < 0 && t - b.dead_quiet_t > kDeadQuietT && pz > param[P_PROP_PEL] * hs &&
            hzc < pz)
          b.prop_t = t;
        if (b.prop_t >= 0 && t - b.prop_t < kPropMax && pz > param[P_PROP_PEL] * hs) {
          T[J_HIP_FLEX_L] = T[J_HIP_FLEX_R] = 0.0;
          b.leg_tone = std::fmax(b.leg_tone, param[P_PROP_TONE] * Clamp((t - b.prop_t) / kPropRise, 0.0, 1.0));
        }
      }
    }
    // Wounded (not dead) and the balancer gave up: curl up (hips to the chest, arms to the body) with a moderate tone
    // kept (not a rag), while the body rolls or flies (pelvis faster than kCurlV); once curled it is held.
    const double vpel = std::sqrt(P.com_vel[0] * P.com_vel[0] + P.com_vel[1] * P.com_vel[1] + P.com_vel[2] * P.com_vel[2]);
    if (b.down == 1 && b.mode == 2 && b.sit_t0 < 0 && b.dead_lay_t < 0) {
      const double up = (vpel > kCurlV || b.curled) ? 1.0 : -1.0;  // held, not released by slowness
      b.curl_w = Clamp(b.curl_w + up * kFrameDt / kCurlT, 0.0, 1.0);
      static const int kj[] = {J_HIP_FLEX_L, J_HIP_FLEX_R, J_KNEE_L, J_KNEE_R, J_SH_FLEX_L, J_SH_FLEX_R,
                               J_SH_ABD_L, J_SH_ABD_R, J_ELBOW_L, J_ELBOW_R, J_NECK_PITCH, J_LUMBAR_PITCH, J_CHEST_PITCH};
      static const double kq[] = {1.45, 1.45, 1.6, 1.6, 0.9, 0.9, 0.1, 0.1, 1.9, 1.9, 0.5, 0.3, 0.3};  // soft curl
      const double cw = b.curl_w * b.curl_w * (3 - 2 * b.curl_w);
      for (int k = 0; k < 13; ++k) T[kj[k]] += cw * (kq[k] - T[kj[k]]);
      b.in.tone = std::fmax(b.in.tone, kWoundCurlTone * cw);
      if (cw > 0.5) b.curled = 1;
    }
    // after the steps end the sink stays where it was (dropping it at the walk end would jump the target)
    const double tw = walk || param[P_WALK_END_T] <= 0 ? t : b.dead_walk_t;
    b.in.crouch = walk || param[P_WALK_END_T] > 0 ? kWalkSink * hs * Clamp((tw - b.dead_t) / kWalkSinkT, 0.0, 1.0) : 0.0;
    b.in.trunk_lean = walk ? kWalkTrunk : 0.0;  // the trunk leans forward as the body sinks
    if (walk) {
      // the standing hips flex as the body sits down on the move (else the balancer keeps the trunk up by extending
      // them and the pelvis tips back with the trunk)
      const double hw = kWalkHip * Clamp((t - b.dead_t) / kWalkSinkT, 0.0, 1.0);
      for (int f = 0; f < 2; ++f)
        if (f != b.swing) T[f ? J_HIP_FLEX_R : J_HIP_FLEX_L] += hw;
    }
    // The wounded (alive) holds the chin and the upper back against the fall on the back at full strength: with
    // torque and stiffness scaled by the fading tone the chest and neck straighten as the back goes down and the head
    // hits the floor much faster than the chest.
    // Backing off fast as the body goes down (the pelvis meets the floor with the trunk still turning back and the
    // feet fly up): the trunk flexes forward (hips, lumbar, chest, chin to the chest) and the arms reach back to the
    // floor, so the back rolls instead of tipping flat. From the pelvis below kBsitPel x stance at more than bsit_v
    // back until lying or kBsitMax.
    // only with the trunk leaning back: backing off while falling forward the pose would take the hands off the floor
    // and the face would hit harder
    if (param[P_BSIT_V] > 0 && b.bsit_t == -1 && b.dead_lay_t < 0 && -vfw > param[P_BSIT_V] && pz < kBsitPel * hs) {
      const double* Rb = d_->xmat + 9 * (b.body0 + B_CHEST);
      const double hx = P.pelvis_R[0], hy = P.pelvis_R[3], hn = std::hypot(hx, hy) + 1e-9;
      const double lean = std::atan2((Rb[2] * hx + Rb[5] * hy) / hn, Rb[8]);  // + = chest top forward [rad]
      if (lean < -kBsitLean) b.bsit_t = t;
    }
    // the pose ends (for good: bsit_t = -2) once the back is on the floor (chest below kBsitChest x stance): held on,
    // the curled body rolls over its shoulders and stays curled on its back
    if (b.bsit_t >= 0 && (d_->xpos[3 * (b.body0 + B_CHEST) + 2] - b.gz) < kBsitChest * hs) b.bsit_t = -2;
    if (b.bsit_t >= 0 && b.dead_lay_t < 0 && t - b.bsit_t < kBsitMax) {
      const double w = Clamp((t - b.bsit_t) / kBsitRise, 0.0, 1.0);
      static const int bj[] = {J_HIP_FLEX_L, J_HIP_FLEX_R, J_LUMBAR_PITCH, J_CHEST_PITCH, J_NECK_PITCH,
                               J_SH_FLEX_L, J_SH_FLEX_R, J_SH_ABD_L, J_SH_ABD_R, J_ELBOW_L, J_ELBOW_R};
      // spine near its flexion stop (40 deg), chin 40 of 45 deg, arms 50 deg back (stop 60), slightly out, elbows soft
      static const double bq[] = {1.5, 1.5, 0.6, 0.6, 0.7, -0.9, -0.9, 0.3, 0.3, 0.3, 0.3};
      for (int k = 0; k < 11; ++k) T[bj[k]] += w * (bq[k] - T[bj[k]]);
      const double bt = param[P_BSIT_TONE] * w;
      b.arm_tone = std::fmax(b.arm_tone, bt);
      if (bt > b.in.tone) {
        const double sc = bt / std::fmax(b.in.tone, 0.05);
        for (int j : {J_NECK_PITCH, J_CHEST_PITCH, J_LUMBAR_PITCH, J_HIP_FLEX_L, J_HIP_FLEX_R}) {
          b.kp[j] *= sc;
          b.limf[j] = std::fmax(b.limf[j], sc);
        }
      }
    }
    if (spine_full > 0) {
      const double sc = 1.0 + spine_full * (1.0 / std::fmax(b.in.tone, 0.1) - 1.0);
      for (int j : {J_NECK_PITCH, J_CHEST_PITCH, J_LUMBAR_PITCH}) {  // (without the lumbar the body can back-roll)
        b.kp[j] *= sc;
        b.limf[j] = std::fmax(b.limf[j], sc);
      }
    }
    // The wounded body's walk end (steps over, curl on) is continuous: the targets follow the new pose with a
    // first-order lag (walk_end_t) for 5 lags. Without it the dead-arm pose is dropped and the curl comes within a
    // few frames: the shoulder targets jump.
    if (param[P_WALK_END_T] > 0 && b.dead_walk_t > b.dead_t && t - b.dead_walk_t < 5 * param[P_WALK_END_T]) {
      const double a = 1.0 - std::exp(-kFrameDt / param[P_WALK_END_T]);
      for (int j = 0; j < J_COUNT; ++j) T[j] = b.tgt_last[j] + a * (T[j] - b.tgt_last[j]);
    }
    b.in.lean[0] = b.in.lean[1] = 0;
    return;
  }
  // Moving support (bridge deck): the feet feel the support move (L3 foot_sv: velocity of the body under
  // each loaded foot). Speed envelope: fast rise, slow fall (holds the peak over a sway period); brace level from it
  // (deck_v0 .. deck_v1).
  {
    const double dt = kFrameDt;
    double w = 0, vx = 0, vy = 0;
    for (int f = 0; f < 2; ++f)
      if (b.loaded[f]) w += b.ld[f], vx += b.ld[f] * b.foot_sv[f][0], vy += b.ld[f] * b.foot_sv[f][1];
    if (w > 0.1 && b.dead_t < 0) {
      b.sup_v[0] = vx / w, b.sup_v[1] = vy / w;
      const double vs = std::hypot(b.sup_v[0], b.sup_v[1]);
      const double tau = vs > b.deck_v ? 0.3 : 3.0;
      b.deck_v += (vs - b.deck_v) * (1.0 - std::exp(-dt / tau));
      // sideways part (pelvis left axis): the crouch is for sideways sway only (on a fore-aft swaying deck the body
      // stands better without it)
      const double ln = std::hypot(P.pelvis_R[1], P.pelvis_R[4]) + 1e-9;
      const double vl = std::fabs(b.sup_v[0] * P.pelvis_R[1] + b.sup_v[1] * P.pelvis_R[4]) / ln;
      const double tl = vl > b.deck_vl ? 0.3 : 3.0;
      b.deck_vl += (vl - b.deck_vl) * (1.0 - std::exp(-dt / tl));
    } else {
      b.deck_v *= std::exp(-dt / 3.0);
      b.deck_vl *= std::exp(-dt / 3.0);
    }
    const double bt = param[P_DECK] > 0 && b.dead_t < 0
                          ? Clamp((b.deck_v - param[P_DECK_V0]) / std::fmax(param[P_DECK_V1] - param[P_DECK_V0], 1e-3), 0.0, 1.0)
                          : 0.0;
    b.brace += (bt - b.brace) * (1.0 - std::exp(-dt / 0.5));
    b.in.wide = b.brace * param[P_DECK_WIDE] * b.body.height / 1.78;
    const double lat = Clamp(b.deck_vl / std::fmax(b.deck_v, 1e-3), 0.0, 1.0);
    b.brace_c = b.brace * lat * lat;  // crouch share: sideways sway
    b.in.pel_pitch = b.brace_c * param[P_DECK_PITCH];
    b.in.deck_step = b.brace > 0.5 ? param[P_DECK_STEP] : 0.0;
    b.in.deck_mid = b.brace > 0.5 ? param[P_DECK_MID] : 0.0;
    const double sk = param[P_DECK_REL] * Clamp(b.brace / 0.3, 0.0, 1.0);
    b.in.sup_v[0] = sk * b.sup_v[0], b.in.sup_v[1] = sk * b.sup_v[1];
  }
  // Bullet reaction while alive: crouch, lean along the bullet (2nd+), leg wound.
  {
    const double dt = kFrameDt;
    const bool live = t < b.bul_until;
    if (!live && b.bul_until >= 0 && b.in.steps_left >= 0) b.in.steps_left = -1;  // reaction over: no limit
    // While the live bullet reaction lasts the stay-upright help is stronger (bul_assist x the pair force the balancer
    // wrote this frame; L0 applies it); the 0.3 ceiling of that help is for pushes.
    if (live && b.health >= kHurtHp) {
      b.assist_f[0] *= param[P_BUL_ASSIST];
      b.assist_f[1] *= param[P_BUL_ASSIST];
    }
    const double ct = live ? param[P_BUL_CROUCH_T] : kCrouchOffT;
    // The deep crouch (bul_crouch, knees 38-40 deg at 0.3-0.4 s after a chest bullet) only for the first bullet of a
    // reaction; under the pull a crouch shortens the steps and the body is carried farther
    const double csc = b.bul_n > 1 ? 1.0 : param[P_BUL_CROUCH];
    b.crouch += ((live ? kCrouch * csc * b.body.height / 1.78 : 0.0) - b.crouch) * (1.0 - std::exp(-dt / ct));
    b.in.crouch = b.crouch > 1e-4 ? b.crouch : 0.0;
    b.in.crouch = std::fmax(b.in.crouch, b.brace_c * param[P_DECK_CROUCH] * b.body.height / 1.78);  // deck brace
    b.hob_vg[0] = b.hob_vg[1] = 0;
    b.in.hob_ds = 0;
    b.in.lean[0] -= b.leg_lean[0];  // the leg-wound shift is set anew below (in.lean persists)
    b.in.lean[1] -= b.leg_lean[1];
    b.leg_lean[0] = b.leg_lean[1] = 0;
    if (t < b.lean_until) {
      b.in.lean[0] = b.lean_mag * b.bul_dir[0];
      b.in.lean[1] = b.lean_mag * b.bul_dir[1];
    } else if (b.hold_lean > 1e-4) {
      b.in.lean[0] = -b.hold_lean * b.bul_dir[0];
      b.in.lean[1] = -b.hold_lean * b.bul_dir[1];
      b.lean_until = -1;
    } else if (b.lean_until >= 0 || b.hold_lean > 0) {
      b.in.lean[0] = b.in.lean[1] = 0;
      b.lean_until = -1;
      b.hold_lean = 0;
    }
    double tl = 0, tr = 0;
    for (int f = 0; f < 2; ++f) {
      if (b.leg_hits[f] == 0 || t < b.leg_t[f] + kLegDelay) continue;
      b.in.leg_weak[f] = std::fmax(0.05, 1.0 - kLegLoss * b.leg_hits[f]);
      const int j0 = f ? J_HIP_FLEX_R : J_HIP_FLEX_L;
      b.kp[j0 + 3] *= b.in.leg_weak[f];  // softer knee
      // the wounded knee gives way slowly and the trunk leans to the wound (ramps over about 3 s)
      const double s = t - b.leg_t[f];
      // not while this leg swings (the swing IK reaches the floor with its own knee; more flexion would keep the hurt
      // foot off the floor)
      if (b.swing != f) T[j0 + 3] += kWLegKnee * Clamp((s - kWLegKneeT0) / kWLegKneeT, 0.0, 1.0);
      const double l = kLegLeanW * Clamp((s - kLegLeanT0) / kLegLeanT, 0.0, 1.0);
      tl = std::fmax(tl, l);
      tr += (f ? -1.0 : 1.0) * 0.3 * l;  // roll toward the wounded side (+ = left down)
      // Bent over the wound - the pelvis forward too (with the spine alone the balancer tilts the pelvis back and the
      // trunk line stays near upright instead of 35-40 deg); the body drifts toward the wounded side; the healthy knee
      // stays at 33-36 deg while the wounded one bends.
      const double hb = kLegHipB * Clamp((s - kLegLeanT0) / kLegLeanT, 0.0, 1.0);
      // not on this (wounded) leg while it swings: with the straight knee the bend would throw the foot far past its
      // place and up
      // and not on the healthy swing leg while the hobble runs (it would cross forward past the hurt foot)
      const bool hob_on = b.hob_dir[0] != 0 || b.hob_dir[1] != 0;
      // half the bent-over flexion while hobbling (with all of it the one-leg stance on the hurt leg folds at the hip
      // and the body falls forward)
      const double hb2 = hob_on ? hb * kHobBend : hb;
      if (b.swing != 0 || (f != 0 && !hob_on)) T[J_HIP_FLEX_L] += hb2;
      if (b.swing != 1 || (f != 1 && !hob_on)) T[J_HIP_FLEX_R] += hb2;
      const double u = Clamp((s - kLegSideT0) / kLegSideT, 0.0, 1.0);
      double sd = kLegSide * u * u * (3 - 2 * u);
      // Give-way: about 3.4 s after a thigh bullet the leg folds (pelvis down 13-28 cm in 0.2 s), the healthy leg
      // steps, then the body hobbles sideways to the wound (1.5 m in 6 s) dipping on the wounded leg: the CoM shift
      // goes past the foot (the balancer steps sideways), fades at the end of the hobble.
      // The give-way waits while the other (healthy) leg swings: the wounded leg is the only support then, and
      // weakening it mid-swing would leave no leg to stand on
      if (b.leg_give[f] >= 0 && t >= b.leg_give[f] && t - b.leg_give[f] < kFrameDt && b.swing == 1 - f)
        b.leg_give[f] = t;
      const double g = t - b.leg_give[f];
      if (b.leg_give[f] >= 0 && g >= 0) {
        if (g < kLegGiveDur) {
          b.in.leg_weak[f] *= kLegGiveWeak;
          b.kp[j0 + 3] *= kLegGiveWeak;
          T[j0 + 3] += kLegGiveKnee;
        }
        // the whole pelvis drops with the folding leg (on two feet the healthy leg alone would hold it: no dip) and
        // comes back up: down kLegDip in kLegDipT0, back over kLegDipT1
        const double dh = g < kLegDipT0 ? g / kLegDipT0 : Clamp(1 - (g - kLegDipT0) / kLegDipT1, 0.0, 1.0);
        b.in.crouch += kLegDip * b.body.height / 1.78 * dh;
        if (param[P_LEG_HOB] > 0 && g > kHobStart && g < kLegHobbleT && !b.cmd_on && b.swing < 0 && b.mode != 2 &&
            t > b.hob_t + kHobGap) {
          // Hobble: the wounded foot steps out toward the wound, then the healthy foot follows (command steps:
          // the balancer shifts the CoM over the stance foot first, the weak leg then carries the body and dips)
          const bool wounded = b.hob_n % 2 == 0;
          const int fs = wounded ? f : 1 - f;
          CommandStep(static_cast<int>(&b - bots_.data()), fs, 0.0, wounded ? param[P_LEG_HOB] : -param[P_LEG_HOB], false);
          b.cmd_to = kHobTo;
          ++b.hob_n;
          b.hob_t = t;
        }
        const double e = Clamp((kLegHobbleT - g) / 1.0, 0.0, 1.0);
        sd = std::fmax(sd, kLegHobble * e);
        if (param[P_LEG_PULL] > 0 && g < kLegHobbleT && b.mode != 2) {
          // hobble pull: toward the wound (pelvis y axis points left; leg 0 = left) while the reaction lives
          const double sy2 = f ? -1.0 : 1.0, hx = sy2 * P.pelvis_R[1], hy = sy2 * P.pelvis_R[4];
          const double hn = std::hypot(hx, hy) + 1e-9;
          b.pull_dir[0] = hx / hn;
          b.pull_dir[1] = hy / hn;
          // the pull fades out toward the hobble speed kHobV (a constant force accelerates the body past its steps)
          const double vs = P.com_vel[0] * b.pull_dir[0] + P.com_vel[1] * b.pull_dir[1];
          b.pull_k = param[P_LEG_PULL] * Clamp((kHobV - vs) * (1.0 / kHobV), 0.0, 1.0);
          b.pull_until = std::fmax(b.pull_until, t + 0.05);
        }
        if (param[P_LEG_HOB] > 0 && g < kLegHobbleT) sd = 0;  // the hobble steps move the body: no standing side shift
      }
      // hob_early: the hobble goal starts this long after the last bullet in the leg if that is before the give-way,
      // so the body drifts toward the wound before the leg folds instead of standing still until then
      // (thigh only: the shin gives way later and hobbling before it the body falls; counted from the last bullet, so
      // several quick bullets do not start the hobble at once). The goal window (kHobEnd) counts from the hobble's
      // start.
      const double hs0 = param[P_HOB_EARLY] > 0 && !b.leg_shin[f]
                             ? std::fmin(b.leg_give[f] + kHobStart, b.leg_t[f] + param[P_HOB_EARLY])
                             : b.leg_give[f] + kHobStart;
      const double gh = t - hs0 + kHobStart;  // = g without the early start
      if (param[P_LEG_VG] > 0 && b.leg_give[f] >= 0 && t > hs0 && gh < kHobEnd && b.mode != 2) {
        // Hobble: velocity goal toward the wound (pelvis y axis points left; leg 0 = left): the wounded foot, on the
        // goal side, steps out first, the healthy one follows; the goal tapers to zero, then the body stands.
        if (b.hob_dir[0] == 0 && b.hob_dir[1] == 0) {
          const double sy2 = f ? -1.0 : 1.0, hx = sy2 * P.pelvis_R[1], hy = sy2 * P.pelvis_R[4];
          const double hn = std::hypot(hx, hy) + 1e-9;
          b.hob_dir[0] = hx / hn;
          b.hob_dir[1] = hy / hn;
        }
        const double vh = param[P_LEG_VG] * b.body.height / 1.78 * Clamp((t - hs0) / kHobRamp, 0.0, 1.0) *
                          Clamp((kHobEnd - gh) / kHobTaper, 0.0, 1.0);
        b.hob_vg[0] = vh * b.hob_dir[0];
        b.hob_vg[1] = vh * b.hob_dir[1];
        // steady pace (hob_ds) for a thigh wound only: on a shin wound it makes the body fall
        if (!b.leg_shin[f]) b.in.hob_ds = param[P_HOB_DS];
        // the hobble is a walk, not the bullet reaction's step reserve (bul_steps): short hobble steps would use it up
        // and the balancer would give up while standing
        b.in.steps_left = -1;
      }
      const double sy = f ? -1.0 : 1.0;  // pelvis y axis = left
      b.leg_lean[0] += sy * sd * P.pelvis_R[1];
      b.leg_lean[1] += sy * sd * P.pelvis_R[4];
      const int jh = (f ? J_HIP_FLEX_L : J_HIP_FLEX_R) + 3;
      if (b.leg_hits[1 - f] == 0 && kLegHealthyKnee > 0) T[jh] = std::fmin(T[jh], kLegHealthyKnee);
    }
    b.in.lean[0] += b.leg_lean[0];
    b.in.lean[1] += b.leg_lean[1];
    if (tl > 0) {
      T[J_LUMBAR_PITCH] += 0.5 * tl;
      T[J_CHEST_PITCH] += 0.5 * tl;
      T[J_LUMBAR_ROLL] += 0.5 * tr;
      T[J_CHEST_ROLL] += 0.5 * tr;
    }
  }
  if (b.no_react) return;

  // Airborne (drop from height): both feet off for > 0.1 s while falling -> tuck, arms out, legs ready.
  // And both feet up off the floor (lower foot box centre above kAirFootZ): mid-step under a strong push both feet can
  // be unloaded for 0.1 s while still on the floor; a landing profile then would drop the pelvis target and fold the
  // catching leg
  const bool feet_off = !b.foot_on[0] && !b.foot_on[1] && std::fmin(P.foot_c[0][2], P.foot_c[1][2]) > kAirFootZ;
  b.air_t = feet_off ? b.air_t + kFrameDt : 0.0;
  if (b.curled && !b.air_curl && !b.in.curl && b.mode != 2 && b.lie_t0 < 0 && !b.airborne &&
      (d_->xpos[3 * (b.body0 + B_PELVIS) + 2] - b.gz) > kCurlUpZ * b.body.height / 1.78)
    b.curled = 0;  // standing again
  if (b.air_t > 0.1 && P.com_vel[2] < -0.5 && b.mode != 2) {
    b.airborne = 1;
    for (int s = 0; s < 2; ++s) {
      const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L, hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
      T[sh + 0] = kAirArmFlex; T[sh + 1] = 0.3; T[sh + 3] = 0.5;  // arms forward-down, elbows soft
      // Legs under the body already bent: knees kAirKnee, hips kAirHip, sole about level (ankle at its dorsiflexion
      // limit), feet at pelvis width. With nearly straight knees the touch force goes along the leg past the knee: the
      // body stops within a few centimetres and bounces up.
      T[hp + 0] = kAirHip; T[hp + 1] = 0; T[hp + 2] = 0; T[hp + 3] = kAirKnee; T[hp + 4] = kAirAnkle; T[hp + 5] = 0;
    }
    T[J_LUMBAR_PITCH] = T[J_CHEST_PITCH] = 0.5 * kAirTrunk;  // trunk slightly forward
    // (a hop inside a running landing keeps its profile: restarted by a slow re-touch the depth would be tiny, the
    // balancer would straighten the legs and throw the body up)
    if (b.land_t >= 0 && t - b.land_t > kLandKeepT) b.land_t = -1;
    // The balancer's pelvis height at the touch is the one of the flight pose (its standing target on the touch frame
    // would straighten the knee at full stiffness and bounce the body up)
    const double hv = (d_->xpos[3 * (b.body0 + B_PELVIS) + 2] - b.gz) - std::fmin(P.ankle[0][2], P.ankle[1][2]);
    b.in.crouch = std::fmax(b.in.crouch, kStandZ * b.body.height / 1.78 - b.dims.ankle_h - hv);
    const double hf = std::fmax(std::fmin(P.ankle[0][2], P.ankle[1][2]) - b.dims.ankle_h, 0.0);
    const double vi = std::sqrt(P.com_vel[2] * P.com_vel[2] + 2 * 9.81 * hf);
    const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
    // time to the touch from the lower of the feet and the pelvis (a tilted body lands on what is lowest)
    const double hl = std::fmax(std::fmin(hf, (d_->xpos[3 * (b.body0 + B_PELVIS) + 2] - b.gz) - 0.9 * b.body.height / 1.78), 0.0);
    const double vdn = std::fmax(-P.com_vel[2], 0.0);
    const double ttg = (-vdn + std::sqrt(vdn * vdn + 2 * 9.81 * hl)) / 9.81;
    // A medium touch (2.5-4.2 m) lands feet ahead of the CoM: the pelvis target in flight leans back by
    // kRollTilt, the touch then sits the body down and turns it over the back (a roll), not a board
    b.air_roll = param[P_LAND_ROLL] > 0 && vi >= kRollV0 && vi < kRollV1 ? 1 : 0;
    if (vi > kAirCurlV || Rc[8] < std::cos(kAirTilt) || b.air_roll) b.air_want = 1;
    const bool flail = param[P_AIR_FLAIL] > 0;
    if (b.air_want && (ttg < kCurlLead || !flail)) b.air_curl = 1;
    if (flail && !b.air_curl && vi > kWindV) {
      if (b.fl_rng == 0) {  // flail start: random phases (right half a turn off the left, legs off the arms), rates 0
        b.fl_rng = ((unsigned)(b.body0 * 2654435761u) ^ (unsigned)(t * 1e5)) | 1u;
        auto rnd = [&]() {  // uniform [-1, 1)
          b.fl_rng ^= b.fl_rng << 13; b.fl_rng ^= b.fl_rng >> 17; b.fl_rng ^= b.fl_rng << 5;
          return (b.fl_rng & 0xffffff) / 8388608.0 - 1.0;
        };
        b.fl_ph[0] = M_PI * rnd();
        b.fl_ph[1] = b.fl_ph[0] + M_PI + 0.8 * rnd();
        b.fl_ph[2] = M_PI * rnd();
        b.fl_ph[3] = b.fl_ph[2] + M_PI + 0.8 * rnd();
        for (int k = 0; k < 4; ++k) b.fl_rate[k] = kFlailRateSd * rnd(), b.fl_amp[k] = kFlailAmpSd * rnd();
      }
      auto rnd = [&]() {
        b.fl_rng ^= b.fl_rng << 13; b.fl_rng ^= b.fl_rng >> 17; b.fl_rng ^= b.fl_rng << 5;
        return (b.fl_rng & 0xffffff) / 8388608.0 - 1.0;
      };
      const double dtau = kFrameDt / kFlailTau;
      // Ornstein-Uhlenbeck drift of rate and size (uniform noise, unit variance x sqrt 3)
      for (int k = 0; k < 4; ++k) {
        b.fl_rate[k] += -b.fl_rate[k] * dtau + kFlailRateSd * std::sqrt(2 * dtau) * 1.73 * rnd();
        b.fl_amp[k] += -b.fl_amp[k] * dtau + kFlailAmpSd * std::sqrt(2 * dtau) * 1.73 * rnd();
        b.fl_ph[k] += (k < 2 ? kFlailWa : kFlailWl) * (1.0 + Clamp(b.fl_rate[k], -0.5, 0.5)) * kFrameDt;
      }
      const double u = Clamp((ttg - kFlailEnd) / (kFlailFull - kFlailEnd), 0.0, 1.0);
      const double wf = u * u * (3 - 2 * u);
      auto blend = [&](int j, double v) { T[j] += wf * (v - T[j]); };
      for (int s = 0; s < 2; ++s) {
        const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L, hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
        const double pa = b.fl_ph[s], pl = b.fl_ph[2 + s];
        const double aa = Clamp(1 + b.fl_amp[s], 0.6, 1.4), al = Clamp(1 + b.fl_amp[2 + s], 0.6, 1.4);
        blend(sh + 0, kFlailShF + aa * kFlailShFA * std::sin(pa));
        blend(sh + 1, kFlailShAbd + aa * kFlailShAbdA * std::cos(pa));
        blend(sh + 3, 0.6 + 0.4 * std::sin(pa - 1.0));
        blend(hp + 0, kFlailHip - al * kFlailHipA * std::cos(pl));
        blend(hp + 3, kFlailKnee + al * kFlailKneeA * std::sin(pl));
        // the shoulder and the leg muscles must follow 2 Hz: raise the stiffness to a fraction of the full one
        for (int k = 0; k < 4; ++k) b.kp[sh + k] += wf * std::fmax(kFlailArmKp * b.js[sh + k].kp - b.kp[sh + k], 0.0);
      }
      // back and neck in time with the arms: twist against the arm difference, small pitch and roll at the arm rate
      const double d01 = std::sin(b.fl_ph[0]) - std::sin(b.fl_ph[1]), s01 = std::sin(b.fl_ph[0]) + std::sin(b.fl_ph[1]);
      blend(J_LUMBAR_YAW, 0.12 * d01);
      blend(J_CHEST_YAW, 0.12 * d01);
      blend(J_LUMBAR_ROLL, 0.05 * std::cos(b.fl_ph[0]) - 0.05 * std::cos(b.fl_ph[1]));
      blend(J_LUMBAR_PITCH, 0.5 * kAirTrunk + 0.05 * s01);
      blend(J_NECK_PITCH, 0.08 * s01);
      blend(J_NECK_YAW, -0.12 * d01);
    } else {
      b.fl_rng = 0;
    }
  } else if (b.airborne && !feet_off) {
    b.airborne = 0;
    b.fl_rng = 0; b.air_want = 0;
    const double vt = std::fmax(-P.com_vel[2], 0.0);
    if (!(b.land_t >= 0 && vt <= b.land_v)) {  // a slower re-touch after a hop goes on with the running profile
      b.land_t = b.air_curl ? -1.0 : t;  // a curled body takes the touch in the curl
      const double hs = b.body.height / 1.78;
      b.land_v = vt;
      // c0 as in flight (pelvis over the lower ankle): from the absolute pelvis height it would differ from the last
      // flight frame (toes-down foot, ankle above the floor) and the knee target would jump at the touch
      const double hv0 = (d_->xpos[3 * (b.body0 + B_PELVIS) + 2] - b.gz) - std::fmin(P.ankle[0][2], P.ankle[1][2]);
      b.land_c0 = std::fmax(kStandZ * hs - b.dims.ankle_h - hv0, 0.0);
      b.land_d = std::fmin(b.land_v * b.land_v / (2 * kLandA),
                           std::fmax(hv0 + b.dims.ankle_h - kLandPelMin * hs, 0.05));
      b.land_cr = b.in.crouch;
      if (vt >= kLandHardV) b.land_hard_t = t;
    }
  }
  if (b.land_t >= 0) {  // landing: the balancer's pelvis height follows the braking profile (BalanceInput crouch)
    bool s_up = false;  // risen to the balancer's own height
    const double s = t - b.land_t, v0 = b.land_v, D = std::fmax(b.land_d, 1e-3);
    const double a = v0 * v0 / (2 * D), tb = v0 / std::fmax(a, 1e-3);  // deceleration, braking time
    double c;
    if (s < tb) c = v0 * s - 0.5 * a * s * s;
    else if (s < tb + kLandHold) c = D;
    else {
      const double u = Clamp((s - tb - kLandHold) / kLandUp, 0.0, 1.0);
      c = D * (1 - u * u * (3 - 2 * u));
    }
    double cr;
    if (b.hd_on && s >= tb + kLandHold) {
      // Rise: the wanted hip height (land_z0) goes up at kLandRise from the hip height at the hold end; crouch = the
      // balancer's compass height of the last frame (its desired height + the crouch then) minus it. The compass alone
      // rises as the hip comes back over the feet, so a fixed crouch profile would lift the pelvis too fast.
      if (s - kFrameDt < tb + kLandHold)
        b.land_z0 = 0.5 * (d_->xpos[3 * (b.body0 + B_THIGH_L) + 2] + d_->xpos[3 * (b.body0 + B_THIGH_R) + 2]);
      else if (P.trunk_pitch < kLandTrunkUp)  // the trunk righting itself lifts the hip by itself
        b.land_z0 += kLandRise * b.body.height / 1.78 * kFrameDt;
      // The balancer's leg reaches past its desired height while the desired hip is far from the standing ankle (IK
      // leg longer than the real one, real hip above the desired): crouch by the excess too.
      const double hz = 0.5 * (d_->xpos[3 * (b.body0 + B_THIGH_L) + 2] + d_->xpos[3 * (b.body0 + B_THIGH_R) + 2]);
      cr = std::fmax(b.hd_prev + b.land_cr - b.land_z0 + kLandFb * std::fmax(hz - b.land_z0, 0.0), 0.0);
      if (cr <= 0) s_up = true;
    } else {
      cr = b.land_c0 * (1 - Clamp(s / (tb + kLandHold + kLandUp), 0.0, 1.0)) + c;
    }
    b.in.crouch = std::fmax(b.in.crouch, cr);
    b.land_cr = b.in.crouch;
    if (s < tb) {
      for (int k = 0; k < 2; ++k) {
        const int hp = k ? J_HIP_FLEX_R : J_HIP_FLEX_L;
        b.zm[hp + 0] *= kLandZm; b.zm[hp + 3] *= kLandZm; b.zm[hp + 4] *= kLandZm;
      }
    }
    if (s < tb + kLandHold) {  // arms forward-down
      for (int k = 0; k < 2; ++k) {
        const int sh = k ? J_SH_FLEX_R : J_SH_FLEX_L;
        T[sh + 0] = 0.8; T[sh + 1] = 0.3; T[sh + 3] = 0.5;
      }
    }
    const bool roll_land = param[P_LAND_ROLL] > 0 && v0 >= kRollV0 && v0 < kRollV1;
    // roll: the curl (tucked back and chin, hips and knees up) after the touch
    if (roll_land && s >= kRollCurlT) b.air_curl = 1;
    if (v0 >= kLandHardV && !roll_land && s >= tb) {  // hard touch: forward onto the hands
      const double hx = P.pelvis_R[0], hy = P.pelvis_R[3], hn = std::hypot(hx, hy) + 1e-9;
      b.in.lean[0] = kHardLean * hx / hn;
      b.in.lean[1] = kHardLean * hy / hn;
      b.in.trunk_lean = kHardTrunk;
      b.in.steps_left = 0;
    }
    if (s_up || s > tb + kLandHold + kLandUp + kLandRiseT || b.mode == 2) {
      if (v0 >= kLandHardV && !roll_land) { b.in.lean[0] = b.in.lean[1] = 0; b.in.trunk_lean = 0; b.in.steps_left = -1; }
      b.land_t = -1;
    }
  }
  if (b.air_curl) {  // curl in the air and through the touch; it is held until the body stands again
    static const int kj[] = {J_HIP_FLEX_L, J_HIP_FLEX_R, J_KNEE_L, J_KNEE_R, J_SH_FLEX_L, J_SH_FLEX_R,
                             J_SH_ABD_L, J_SH_ABD_R, J_ELBOW_L, J_ELBOW_R, J_NECK_PITCH, J_LUMBAR_PITCH,
                             J_CHEST_PITCH};
    // (back round: after a curled high touch the body sits and rolls back, and a round back keeps the trunk from
    // falling onto the head)
    // Soft curl: hips 83 deg, knees 92, shoulder abduction lets the hands cover the head without a clamp
    static const double kq[] = {1.45, 1.45, 1.6, 1.6, 1.4, 1.4, 0.4, 0.4, 1.8, 1.8, 0.45, 0.5, 0.5};
    const double v3 = std::sqrt(P.com_vel[0] * P.com_vel[0] + P.com_vel[1] * P.com_vel[1] + P.com_vel[2] * P.com_vel[2]);
    const bool hold = b.airborne || v3 > 0.5;
    const double* Rcu = d_->xmat + 9 * (b.body0 + B_CHEST);
    const bool stands = !b.airborne && (d_->xpos[3 * (b.body0 + B_PELVIS) + 2] - b.gz) > kCurlUpZ * b.body.height / 1.78 &&
                        Rcu[8] > kCurlUpC;
    b.air_cw = Clamp(b.air_cw + ((hold || !stands) ? 1.0 : -1.0) * kFrameDt / kAirCurlT, 0.0, 1.0);
    const double w = b.air_cw * b.air_cw * (3 - 2 * b.air_cw);
    for (int k = 0; k < 13; ++k) T[kj[k]] += w * (kq[k] - T[kj[k]]);
    // soft: the muscles at kCurlKp, the legs and back give to the weight; slow on the floor, the tone fades and the
    // body stays as it lies (the targets stay, the limbs only sag under their weight)
    b.air_still = hold ? -1.0 : (b.air_still < 0 ? t : b.air_still);
    const double tone = b.air_still < 0 ? 1.0 : kRelaxMin + (1 - kRelaxMin) * std::exp(-(t - b.air_still) / kRelaxT);
    for (int j = 0; j < J_COUNT; ++j) {
      b.kp[j] *= (1.0 - (1.0 - kCurlKp) * w) * tone;
      if (j < J_NECK_PITCH || j >= J_HIP_FLEX_L) b.gsup[j] = std::fmin(b.gsup[j], 1.0 - kCurlSup * w) * tone;
    }
    if (w > 0.5) b.curled = 1;
    if (!hold && b.air_cw <= 0) b.air_curl = 0;
    b.fall_set = 0;
    return;
  }

  if (b.in.curl) {
    // Curl: knees to chest, arms to head and trunk; smooth ramp over kCurlT (~1-1.5 s).
    static const int kj[] = {J_HIP_FLEX_L, J_HIP_FLEX_R, J_KNEE_L, J_KNEE_R, J_SH_FLEX_L, J_SH_FLEX_R,
                             J_SH_ABD_L, J_SH_ABD_R, J_ELBOW_L, J_ELBOW_R, J_NECK_PITCH, J_LUMBAR_PITCH,
                             J_CHEST_PITCH};
    static const double kq[] = {1.45, 1.45, 1.6, 1.6, 1.4, 1.4, 0.4, 0.4, 1.8, 1.8, 0.45, 0.3, 0.3};  // soft curl
    b.curl_s = std::fmin(1.0, b.curl_s + kFrameDt / kCurlT);
    const double w = b.curl_s * b.curl_s * (3 - 2 * b.curl_s);
    for (int k = 0; k < 13; ++k) T[kj[k]] += w * (kq[k] - T[kj[k]]);
    for (int j = 0; j < J_COUNT; ++j) b.kp[j] *= 1.0 - (1.0 - kCurlKp) * w;
    if (w > 0.5) b.curled = 1;
    b.fall_set = 0;
    return;
  }
  b.curl_s = 0;
  // Fall judgement (percept only): the balancer gave up, or the capture point is far outside the support and
  // far from the CoM (beyond the longest step).
  const double om = std::sqrt(9.81/std::fmax(P.com[2], 0.3));
  const double xi_d = std::hypot(P.com_vel[0], P.com_vel[1])/om;  // capture point distance from the CoM
  // A forward stumble with the chest leaning far forward and the CoM running forward fast is judged a fall too
  // (judged only at the give-up, the hands would reach the floor together with the head)
  bool fwd_fall = false;
  {
    const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
    const double hx = P.pelvis_R[0], hy = P.pelvis_R[3], hn = std::hypot(hx, hy) + 1e-9;
    const double lean = (Rc[2] * hx + Rc[5] * hy) / hn;  // chest axis toward the heading: sin of the forward lean
    const double vf = (P.com_vel[0] * hx + P.com_vel[1] * hy) / hn;
    fwd_fall = b.mode == 1 && lean > std::sin(kPredLean) && vf > kPredFwdV;
    const double ll = (-Rc[2] * hy + Rc[5] * hx) / hn, vl = (-P.com_vel[0] * hy + P.com_vel[1] * hx) / hn;
    if (b.mode == 1 && ll * vl > 0 && std::fabs(ll) > std::sin(kPredLatLean) && std::fabs(vl) > kPredLatV) fwd_fall = true;
  }
  const bool imminent =
      b.mode == 2 || fwd_fall || (param[P_FALL_REACT] > 0 && b.cp_out > kPredOut && xi_d > kPredXi);
  if (imminent) b.fall_seen = t;
  if (b.mode == 0 && t - b.fall_seen > kPredHold) b.fall_pred = 0;  // recovered: the judgement lapses
  fall_seat();
  if (!b.fall_pred && imminent) {
    b.fall_pred = 1;
    b.fall_tp = t;
    // Latency of the protective arm response: onset 120 +- 18 ms (Borrelli et al. 2019, docs/SOURCES.md) ; the
    // muscle model adds the force build-up; spread from the bot's own RNG (differs between bots and events).
    b.rng ^= b.rng << 13; b.rng ^= b.rng >> 7; b.rng ^= b.rng << 17;
    const double u1 = (b.rng >> 11) * (1.0 / 9007199254740992.0);
    b.rng ^= b.rng << 13; b.rng ^= b.rng >> 7; b.rng ^= b.rng << 17;
    const double u2 = (b.rng >> 11) * (1.0 / 9007199254740992.0);
    b.rng ^= b.rng << 13; b.rng ^= b.rng >> 7; b.rng ^= b.rng << 17;
    const double u3 = (b.rng >> 11) * (1.0 / 9007199254740992.0);
    b.fall_delay = Clamp(0.09 + 0.018 * 2.0 * (u1 + u2 + u3 - 1.5), 0.08, 0.2);
    b.head_turn = (b.rng >> 40) & 1 ? 1.0 : -1.0;
    b.fall_set = 0;
  }
  // Car seen coming (not from behind, on a course that hits the chest within kCarSeeT): after the visual reaction
  // time the arms go toward it before the impact (the arms protect). The near arm abducts toward a side car;
  // elbows half bent; arm stiffness x kArmKp. The far arm stays: the car throws the body to its side, and an arm
  // reaching across the chest would not be out for the floor. Only before a fall is judged: then the catch takes the
  // arms (it first holds them where they are).
  if (!b.fall_pred && car_body_ >= 0 && car_speed_ > 0 && !car_hit_) {
    const double* Cc = d_->xpos + 3 * car_body_;
    const double* Ch = d_->xipos + 3 * (b.body0 + B_CHEST);
    const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
    const double ux = car_dir_[0], uy = car_dir_[1];
    const double rx = Ch[0] - Cc[0], ry = Ch[1] - Cc[1];
    const double gap = rx * ux + ry * uy - kCarHalfLen, lat = std::fabs(rx * uy - ry * ux);
    const double dx = -(ux * Rc[0] + uy * Rc[3]), dy = -(ux * Rc[1] + uy * Rc[4]);  // toward the car, chest frame
    const double dn = std::hypot(dx, dy) + 1e-9;
    if (b.car_seen_t < 0 && gap > 0 && gap < kCarSeeT * car_speed_ && lat < kCarHalfWid && dx / dn > -0.5)
      b.car_seen_t = t;
    const double a = b.car_seen_t < 0 ? 0.0 : Clamp((t - b.car_seen_t - kCarReactT) / kCarRiseT, 0.0, 1.0)
        * Clamp((kCarHalfWid + 0.15 - lat) * (1.0 / 0.3), 0.0, 1.0);  // only while the car still hits
    // One straight arm held toward the car looks like pointing. car_guard: when the car is close (no dodge possible,
    // or the late turn-away) both forearms come up in front of the chest and face, elbows bent, wrists back; while a
    // dodge still has time the arms are left to the upper layer (they swing with the steps)
    const double tl = gap / std::fmax(car_speed_, 0.1);
    // (a guard in the last kCarGuardT of a dodge disturbs its last steps, so a calm dodge does not guard)
    // calm dodge (decided with throw_tc or more left): arms swing with the escape; a hurried one (the dive) keeps the
    // near-arm reach (an arm swing there costs balance and the body is hit or falls more often); later than
    // kTurnTc87: guard (only an escape along the heading, a side car: the single near arm abducted toward it; front
    // cars keep both arms forward)
    const bool fa = b.dodge_fa;  // fixed at the decision (the chest turns while the body steps sideways)
    // hur_swing: a hurried dodge from the side swings the arms with the escape too (the near-arm reach would put the
    // near hand into the car even with the feet clear)
    const bool calm = (b.dodge == 1 || b.dodge == 3) && fa &&
                      (b.dodge_tc >= param[P_THROW_TC] || (param[P_HUR_SWING] > 0 && b.dodge_tc >= kTurnTc87));
    const bool hurried = (b.dodge == 1 || b.dodge == 3) && b.dodge_tc >= kTurnTc87 && !calm;  // and calm front cars
    (void)tl;
    if (a > 0 && param[P_CAR_GUARD] > 0 && calm) {
      // arms toward the escape (forward share flexes both, sideways share abducts the escape-side arm), as stiff as the
      // near-arm reach; the swing helps the escape steps, while arms held down or back slow them
      const double el = b.dodge_dir[0] * Rc[1] + b.dodge_dir[1] * Rc[4];  // escape toward the chest's left (+)
      const double ef = b.dodge_dir[0] * Rc[0] + b.dodge_dir[1] * Rc[3];  // escape forward (+)
      for (int s = 0; s < 2; ++s) {
        const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
        const double side = s ? -1.0 : 1.0, out = std::fmax(side * el, 0.0);
        T[sh + 0] += a * (kCarSwingFlex * ef - T[sh + 0]);
        T[sh + 1] += a * (0.2 + kCarSwingAbd * out - T[sh + 1]);
        T[sh + 2] += a * (0.0 - T[sh + 2]);
        T[sh + 3] += a * (kCarElb - T[sh + 3]);
        for (int k = 0; k < 4; ++k) b.kp[sh + k] *= 1.0 + a * (kArmKp - 1.0);
      }
    } else if (a > 0 && param[P_CAR_GUARD] > 0 &&
               (!hurried || (param[P_GUARD_FRONT] > 0 && b.dodge_tc >= param[P_THROW_TC]))) {
      // guard_front: calm dodges from front cars get the guard too: the near-arm reach would hold both arms straight
      // forward at shoulder height. Not the hurried ones: there the guard disturbs the dive and the car hits more often
      for (int s = 0; s < 2; ++s) {
        const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
        T[sh + 0] += a * (kCarGuardFlex - T[sh + 0]);
        T[sh + 1] += a * (kCarGuardAbd - T[sh + 1]);
        T[sh + 2] += a * (0.0 - T[sh + 2]);
        T[sh + 3] += a * (kCarGuardElb - T[sh + 3]);
        T[sh + 4] += a * (kCarGuardWrist - T[sh + 4]);
        for (int k = 0; k < 5; ++k) b.kp[sh + k] *= 1.0 + a * (kArmKp - 1.0);
      }
    } else if (a > 0) {
      const double cx = dx / dn, cy = dy / dn;
      for (int s = 0; s < 2; ++s) {
        const double side = s ? -1.0 : 1.0;
        const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
        const double near = std::fmax(side * cy, 0.0), fr = std::fmax(cx, 0.0), ar = a * std::fmax(near, fr);
        T[sh + 0] += ar * (Clamp(kCarFlex * fr + 0.3 * near, -0.5, 2.0) - T[sh + 0]);
        T[sh + 1] += ar * (0.2 + kCarAbd * near - T[sh + 1]);
        T[sh + 2] += ar * (0.0 - T[sh + 2]);
        T[sh + 3] += ar * (kCarElb - T[sh + 3]);
        for (int k = 0; k < 4; ++k) b.kp[sh + k] *= 1.0 + ar * (kArmKp - 1.0);
      }
    }
  }
  if (!b.fall_pred) return;
  // Legs and back stay the balancer's until it gives up (taking them earlier by capture point distance does not help).
  const bool doomed = b.mode == 2;
  if (t < b.fall_tp + b.fall_delay) {  // latency: only the stretch response acts, the arms stiffen where they are
    // (a raised arm is held where it is - the balancer's neutral would pull the arms raised in a side stumble back
    // down; an arm swung back by the steps goes to the neutral - held, a forward fall would be caught from far behind)
    // (not for a backward fall: there the neutral puts the hand down before the head more often than the hold)
    const double hx = P.pelvis_R[0], hy = P.pelvis_R[3], hn = std::hypot(hx, hy) + 1e-9;
    const double vf = (P.com_vel[0] * hx + P.com_vel[1] * hy) / hn;
    const bool hold = vf > -0.3 * std::hypot(P.com_vel[0], P.com_vel[1]);
    for (int j = J_SH_FLEX_L; j <= J_WRIST_R; ++j) {
      b.kp[j] *= kPreArmKp;
      const int k = (j - J_SH_FLEX_L) % 5;  // flex, abd, rot, elbow, wrist
      if (hold) T[j] = k <= 1 ? std::fmax(P.q[j], T[j]) : P.q[j];
    }
    return;
  }
  // Protective reaction toward the fall direction. The direction is followed in the body frame (the pelvis spins by
  // 30-50 deg during a fall, so a direction fixed at the start would put the arms in the wrong place).
  if (!b.fall_set) {
    b.fall_set = 1;
    b.fall_t0 = t;
    b.lie_t0 = -1;
    b.fall_fwd = b.fall_lat = 0;
    for (int s = 0; s < 2; ++s) {  // catch: warm start from the arm, drawn elbow limit (0.15-0.5 rad)
      const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
      for (int k = 0; k < 4; ++k) b.fall_aq[s][k] = P.q[sh + k];
      // not from an arm thrown behind: a push can swing it far back, and the solution choice (nearest to the last
      // target) would keep the arm up and back so that the hands land with the head
      b.fall_aq[s][0] = std::fmax(b.fall_aq[s][0], 0.0);
      b.fall_el[s] = 0.15 + 0.35 * Rnd(&b.rng);
      b.arm_a[s] = 0.5;
      b.hand_was[s] = 0;
      b.hand_t[s] = -1;
    }
  }
  if (b.lie_t0 < 0 && std::hypot(P.com_vel[0], P.com_vel[1]) > 0.5) {
    // Heading: the horizontal part of the pelvis x axis while it is long (pelvis pitched less than 60 deg), else the
    // twist of the pelvis about the vertical (R00 + R11, R10 - R01). The x axis flips as the pelvis pitches past 90 deg
    // in a dive, which would reverse fall_fwd and send the arms back. Only pitched forward: the twist would also change
    // side falls.
    const double* R = P.pelvis_R;
    const double xn = std::hypot(R[0], R[3]);
    const bool ax = xn > kHeadPitch || R[6] > 0.0;  // (pitched back: the axis as before)
    const double hx = ax ? R[0] : R[0] + R[4], hy = ax ? R[3] : R[3] - R[1], hn = std::hypot(hx, hy) + 1e-9;
    const double vx = P.com_vel[0], vy = P.com_vel[1];
    const double nf = (vx * hx + vy * hy) / hn, nl = (-vx * hy + vy * hx) / hn;  // + = to the bot's left
    // A rebound after the first impact reverses the velocity: keep the direction of the fall itself.
    if (nf * b.fall_fwd + nl * b.fall_lat >= 0) { b.fall_fwd = nf; b.fall_lat = nl; }
  }
  const double fv = std::hypot(b.fall_fwd, b.fall_lat) + 1e-9;
  const double cf = b.fall_fwd / fv, cl = b.fall_lat / fv;
  // A side fall while the balancer still steps: the pelvis goes down (the body tips over from lower, so the shoulder
  // is nearer the floor and the CoM slower when the hand meets it)
  if (!doomed && b.down == 0 && std::fabs(cl) > 0.5) b.in.crouch = std::fmax(b.in.crouch, kFallCrouch * b.body.height / 1.78);
  // Floor contacts: lying (trunk or head down) and which arm segments already touch.
  bool hand_down[2] = {false, false};
  for (int i = 0; i < d_->ncon; ++i) {
    const mjContact& c = d_->contact[i];
    const int g = Ground(c.geom1) ? c.geom2 : (Ground(c.geom2) ? c.geom1 : -1);
    if (g < 0) continue;
    const int bd = m_->geom_bodyid[g] - b.body0;
    if (b.lie_t0 < 0 && b.mode == 2 && bd >= B_PELVIS && bd <= B_HEAD) b.lie_t0 = t;
    if (bd >= B_UARM_L && bd <= B_HAND_R) hand_down[(bd - B_UARM_L) / 3] = true;
  }
  // Catch strength: global from the CoM speed (fast up, slow down); per arm: up while off the floor, yielding on it
  {
    const double v3 = std::sqrt(P.com_vel[0] * P.com_vel[0] + P.com_vel[1] * P.com_vel[1] + P.com_vel[2] * P.com_vel[2]);
    const double gt = Clamp((v3 - kCatchV0) / (kCatchV1 - kCatchV0), 0.0, 1.0);
    b.catch_g = gt > b.catch_g ? gt : b.catch_g + (gt - b.catch_g) * (1.0 - std::exp(-kFrameDt / kCatchT));
    for (int s = 0; s < 2; ++s) {
      if (!hand_down[s]) b.arm_a[s] += (1.0 - b.arm_a[s]) * (1.0 - std::exp(-kFrameDt / kArmRiseT));
      else b.arm_a[s] -= kFrameDt / kArmYieldT;
      b.arm_a[s] = Clamp(std::fmax(b.arm_a[s], b.catch_g), 0.0, 1.0);
    }
  }
  double tone = 1.0;
  double leg_tone = 1.0;  // legs let go faster: a stiff leg carries the pelvis rotation and flies up after the hip hits
  // Lying while tumbling down the stairs is not rest: the tone fade counts from the end of the tumble (a fade from the
  // first trunk touch would let the curl open and the body would stop on the treads)
  if (b.tumble) b.relax_t0 = t;
  if (b.lie_t0 >= 0) {
    const double t0 = std::fmax(b.lie_t0, b.relax_t0);
    tone = kRelaxMin + (1 - kRelaxMin) * std::exp(-(t - t0) / kRelaxT);
    leg_tone = kLimbTone + (1 - kLimbTone) * std::exp(-(t - t0) / kLegRelaxT);
  }
  // Forward and backward falls: legs stay stiff while no hand is down (the body pivots instead of collapsing, which
  // gives the arms time); they go limp once a hand touches or the body lies.
  const bool stiff_legs = param[P_FALL_LEGS] > 0 && doomed && b.lie_t0 < 0 && !hand_down[0] && !hand_down[1] &&
                          std::fabs(cf) > 0.5;
  const double ramp = Clamp((t - b.fall_t0) / kReachT, 0.0, 1.0);  // arm reach build-up
  for (int s = 0; s < 2; ++s) {
    const double side = s ? -1.0 : 1.0;  // +1 left
    const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L, hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
    // Arms toward the fall: forward falls reach forward-down; backward falls sweep back-down beside the hips;
    // lateral falls abduct the fall-side arm (the far arm goes across the chest, away from the floor).
    const double lat_this = Clamp((side * cl - 0.3) * 2.5, 0.0, 1.0);  // sectors: a diagonal fall blends two reactions
    const double fw = Clamp((cf - 0.3) * 2.5, 0.0, 1.0), bk = Clamp((-cf - 0.3) * 2.5, 0.0, 1.0);
    // World-angle targets: the arm hangs along the trunk, so its angle from the vertical is (flexion - trunk pitch);
    // the target angle (forward falls: ahead of the vertical, backward: slightly behind) is turned into a shoulder
    // angle.
    const double qp = P.trunk_pitch, qr = P.trunk_roll;
    T[sh + 0] = Clamp(1.4 * fw + (fw > bk ? 0.0 : 0.0) - kArmBack * bk + qp * bk, -1.0, 2.6);
    const double abd_fall = kArmSide + 0.0 * std::fabs(qr);  // feedforward: the body rolls over the outstretched arm
    T[sh + 1] = Clamp(0.25 + lat_this * (abd_fall - 0.25) * kArmLat + kArmBackAbd * bk, 0.1, 2.2);
    T[sh + 2] = 0;
    // far arm of a lateral fall: elbow bent, kept in
    T[sh + 3] = kElbow + 0.4 * Clamp((-side * cl - 0.3) * 2.5, 0.0, 1.0);
    // Hand by a ray to the floor (replaces the joint targets above while the catch lives): from the shoulder along
    // the fall direction (chest velocity plus a share downward); start shifted forward by the forwardness of the fall
    // (back for a backward fall); where it meets the floor the palm goes (else the end of the ray); after the touch
    // the palm stays midway between the ray point and where it landed.
    if (b.catch_g > 0.05 || b.lie_t0 < 0) {
      const int ub = b.body0 + (s ? B_UARM_R : B_UARM_L);
      const double* Sw = d_->xpos + 3 * ub;
      const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
      const double vx = P.com_vel[0], vy = P.com_vel[1], vz = P.com_vel[2];
      const double vn = std::sqrt(vx * vx + vy * vy + vz * vz) + 1e-6;
      double dir[3] = {vx / vn, vy / vn, vz / vn - kRayDown};
      const double dn = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
      for (int k = 0; k < 3; ++k) dir[k] /= dn;
      const double cfx = Rc[0], cfy = Rc[3], cfn = std::hypot(cfx, cfy) + 1e-9;  // chest forward (x axis) on the floor
      const double fwdness = (dir[0] * cfx + dir[1] * cfy) / cfn;
      const double far = Clamp((-side * cl - 0.3) * 2.5, 0.0, 1.0);
      const double sf = kRayFwd * fwdness + kFarFwd * far;
      const double o[3] = {Sw[0] + sf * cfx / cfn, Sw[1] + sf * cfy / cfn, Sw[2]};
      const double Lu = -m_->body_pos[3 * (ub + 1) + 2], Lp = -m_->body_pos[3 * (ub + 2) + 2] + 0.06 * b.body.height / 1.78;
      const double L = Lu + Lp;
      double p[3];
      const double hz = 0.04;  // palm thickness over the floor [m]
      const bool on_floor = dir[2] < -1e-3 && (o[2] - hz) / -dir[2] < 2.0 * L;
      if (on_floor) {
        const double sl = (o[2] - hz) / -dir[2];
        for (int k = 0; k < 3; ++k) p[k] = o[k] + sl * dir[k];
      } else {
        for (int k = 0; k < 3; ++k) p[k] = o[k] + L * dir[k];
      }
      if (hand_down[s]) {
        if (!b.hand_was[s]) {
          const double* hw = d_->xpos + 3 * (b.body0 + (s ? B_HAND_R : B_HAND_L));
          for (int k = 0; k < 3; ++k) b.hand_at[s][k] = hw[k];
          b.hand_t[s] = t;
          b.hand_q[s][0] = P.q[sh + 0]; b.hand_q[s][1] = P.q[sh + 1]; b.hand_q[s][2] = P.q[sh + 3];
        }
        // (midway: the body slides on over the held palm and the arm goes back along it; a quarter keeps the reach)
        for (int k = 0; k < 3; ++k) p[k] += kHoldShare * (b.hand_at[s][k] - p[k]);
      }
      b.hand_was[s] = hand_down[s];
      // palm relative to the shoulder ahead by the shoulder velocity, length clamped, into the chest frame
      // (only for a floor point; a ray end in the air moves with the shoulder, and the look-ahead would bend the arm
      // of a side fall to the reach minimum before the touch)
      const double lt = on_floor || hand_down[s] || cf < -0.3 ? kLeadT : 0.0;  // (backward falls always look ahead)
      const double* vs = P.com_vel;  // the shoulder moves with the body: CoM velocity predicts it
      double r[3] = {p[0] - Sw[0] - lt * vs[0], p[1] - Sw[1] - lt * vs[1], p[2] - Sw[2] - lt * vs[2]};
      const double rn = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) + 1e-9;
      const double rc = Clamp(rn, kReachMin * L, kReachMax * L);
      for (int k = 0; k < 3; ++k) r[k] *= rc / rn;
      double tg[3];
      for (int a = 0; a < 3; ++a) tg[a] = Rc[a] * r[0] + Rc[3 + a] * r[1] + Rc[6 + a] * r[2];
      // direct arm solution in the chest frame: elbow from the distance (law of cosines), abduction from the side part
      // of the direction, flexion from its sagittal part less the angle the bent elbow adds (the generic IK of the hand
      // to the wound jumps between solutions here)
      double* aq = b.fall_aq[s];
      {
        const double tn = std::sqrt(tg[0] * tg[0] + tg[1] * tg[1] + tg[2] * tg[2]) + 1e-9;
        const double ce = Clamp((Lu * Lu + Lp * Lp - tn * tn) / (2 * Lu * Lp), -1.0, 1.0);
        const double elb = M_PI - std::acos(ce);
        const double al = std::atan2(Lp * std::sin(elb), Lu + Lp * std::cos(elb));
        const double ab = std::asin(Clamp(side * tg[1] / tn, -1.0, 1.0));
        double fl = std::atan2(tg[0], -tg[2]);  // from hanging down toward forward; overhead stays positive
        // Behind and up (fl < -90 deg) is either an extension to the limit or a flexion over the head; the one nearer
        // the last target is kept (a fixed choice flips a backward fall's arm between the two frame by frame)
        if (fl < -0.5 * M_PI &&
            std::fabs(Clamp(fl + 2 * M_PI - al, -1.0, 2.4) - aq[0]) < std::fabs(Clamp(fl - al, -1.0, 2.4) - aq[0]))
          fl += 2 * M_PI;
        // A target out to the side and above the chest horizontal (side fall, trunk leaned far) is near the
        // flexion-abduction singularity, where the flexion can jump by a half turn. The same direction with the
        // abduction past 90 deg and the flexion turned by 180 is the second solution (there the bent elbow is offset
        // the other way); the one nearer the last target is kept.
        double fa = Clamp(fl - al, -1.0, 2.4), aa = Clamp(ab, -0.3, 1.5);
        if (side * tg[1] > 0 && tg[2] > 0) {
          const double fb = Clamp(std::atan2(-tg[0], tg[2]) + al, -1.0, 2.4), abb = Clamp(M_PI - ab, 1.5, 2.2);
          if (std::fabs(fb - aq[0]) + std::fabs(abb - aq[1]) < std::fabs(fa - aq[0]) + std::fabs(aa - aq[1])) {
            fa = fb;
            aa = abb;
          }
        }
        aq[1] = aa;
        aq[0] = fa;
        aq[2] = 0;
        aq[3] = Clamp(elb, 0.1, 2.0);
      }
      const double el = cf < -0.3 ? std::fmin(aq[3], b.fall_el[s]) : aq[3];  // backward: arms behind almost straight
      T[sh + 0] = aq[0];
      T[sh + 1] = aq[1];
      T[sh + 2] = aq[2];
      T[sh + 3] = el;
      // A hand on the floor makes the arm a strut: the shoulder keeps its angles of the touch, the elbow gives way at
      // kElbowGive. Following the held palm, the targets would turn behind and across once the shoulder passes over the
      // hand.
      if (hand_down[s] && b.hand_t[s] >= 0) {
        T[sh + 0] = b.hand_q[s][0];
        T[sh + 1] = b.hand_q[s][1];
        T[sh + 3] = std::fmin(b.hand_q[s][2] + kElbowGive * (t - b.hand_t[s]), 2.0);
      }
    }
    if (doomed && b.sit_t0 < 0) {  // a backward seat keeps its own legs
      // Legs and back are the balancer's while it still steps and the fall is not certain.
      // Backward falls sit down (hips and knees flexed: the seat lands low and the legs are a short lever when the body
      // rotates over it; straight legs would swing up to the vertical); forward falls keep straight legs.
      const bool sit = cf < -0.3;
      // Side falls fold the legs (fall-side knee 96-146 deg, far 45-86, hips 41-79 as the pelvis goes down; with less
      // flexion the body goes down on nearly straight knees like a board)
      const double far = Clamp((-side * cl - 0.3) * 2.5, 0.0, 1.0);
      T[hp + 0] = stiff_legs ? (sit ? kSitHip : kLegHip) : 0.5 + lat_this * (kSideHip - 0.5) + far * (kSideHipFar - 0.5);
      T[hp + 3] = stiff_legs ? (sit ? kSitKnee : kLegKnee)
                             : 0.8 + lat_this * (kSideKnee - 0.8) + far * (kSideKneeFar - 0.8);  // knees bend
      T[hp + 4] = 0.0;
      // After a hard touch the squatting legs do not straighten (the stiff-leg targets would drive the knees straight
      // from the squat: the body jumps up and falls on the face)
      if (b.land_hard_t >= 0 && t - b.land_hard_t < kHardKeepT) {
        T[hp + 0] = std::fmax(T[hp + 0], P.q[hp + 0]);
        T[hp + 3] = std::fmax(T[hp + 3], P.q[hp + 3]);
      }
    }
    (void)ramp;
  }
  // Curl on a roll (no separate reaction): while the pelvis turns faster than kRollW1 the hips come to the chest,
  // the knees follow a sine of the roll angle, neck and back round; the arms keep their catch. Fades below kRollW0.
  {
    const double* wp = d_->cvel + 6 * (b.body0 + B_PELVIS);
    const double wn = std::sqrt(wp[0] * wp[0] + wp[1] * wp[1] + wp[2] * wp[2]);
    // (not in the backward seat: it has its own legs; with the curl the seated body would lie with the feet up)
    // (only on the ground - in the tip-over itself the pelvis turns past 4 rad/s, and a curl there folds the legs and
    // drops the pelvis with the body, so the head lands hard)
    // Tumble: the fall is judged unavoidable and the lower foot is a stair rise below the stand floor (layers' z 0; on
    // flat ground the sole centre stays at +0.03 m), the body moving (CoM > kTumbleV or pelvis turn > kRollW0). Down
    // stairs a stepping body otherwise goes head first with loose legs until the trunk touches, then sits. From here L1
    // gives up and the curl comes at once (no trunk touch, no seat), latched while the body moves.
    {
      const double vc = std::sqrt(P.com_vel[0] * P.com_vel[0] + P.com_vel[1] * P.com_vel[1] + P.com_vel[2] * P.com_vel[2]);
      const bool mv = vc > kTumbleV || wn > kRollW0;
      // the ground under a foot (ray down from the sole, own foot excluded; only once the fall is judged): the feet
      // can still be over the first treads, not below them, when the step in the air starts
      // heights from the stand floor (b.gz_st, latched in L3 while not falling)
      const double dz = P.gz - b.gz_st;
      bool low = std::fmin(P.foot_c[0][2], P.foot_c[1][2]) + dz < kTumbleZ * b.body.height / 1.78;
      if (!b.tumble && !low && (b.fall_pred || b.mode == 2) && mv)
        for (int f = 0; f < 2 && !low; ++f) {
          const mjtNum pt[3] = {P.foot_c[f][0], P.foot_c[f][1], P.foot_c[f][2] + P.gz - 0.04}, dn[3] = {0, 0, -1};
          int g = -1;
          const mjtNum dist = mj_ray(m_, d_, pt, dn, nullptr, 1, b.body0 + (f ? B_FOOT_R : B_FOOT_L), &g, nullptr);
          low = g >= 0 && Ground(g) && pt[2] - dist - b.gz_st < kTumbleZ * b.body.height / 1.78;
        }
      if (!b.tumble && (b.fall_pred || b.mode == 2) && low && mv) b.tumble = 1;
      static const bool kTrace = getenv("EMIL_TRACE_TUMBLE") && std::atoi(getenv("EMIL_TRACE_TUMBLE"));
      if (kTrace) {  // trace (EMIL_TRACE_TUMBLE): what the tumble test saw this frame
        double rz = -9;  // ground under the lower foot by a ray (trace only; mj_ray reads the state)
        const int fl = P.foot_c[0][2] < P.foot_c[1][2] ? 0 : 1;
        const mjtNum pt[3] = {P.foot_c[fl][0], P.foot_c[fl][1], P.foot_c[fl][2] + P.gz - 0.04}, dn[3] = {0, 0, -1};
        int g = -1;
        const mjtNum dist = mj_ray(m_, d_, pt, dn, nullptr, 1, b.body0 + (fl ? B_FOOT_R : B_FOOT_L), &g, nullptr);
        if (g >= 0) rz = Ground(g) ? pt[2] - dist - b.gz_st : -8;  // -8: the ray hit a non-ground geom
        const double v[13] = {(double)b.fall_pred, (double)b.mode, (double)mv, std::fmin(P.foot_c[0][2], P.foot_c[1][2]) + dz,
                              kTumbleZ * b.body.height / 1.78, rz, P.gz, b.gz_st, b.lie_t0 >= 0 ? 1.0 : 0.0, 0, 0, 0,
                              (double)b.tumble};
        for (int k = 0; k < 13; ++k) b.tt[k] = v[k];
        int nc = 0;  // ground contacts of the body (any static geom)
        for (int i = 0; i < d_->ncon; ++i) {
          const int b1 = m_->geom_bodyid[d_->contact[i].geom1], b2 = m_->geom_bodyid[d_->contact[i].geom2];
          nc += ((b1 >= b.body0 && b1 < b.body0 + B_COUNT && Ground(d_->contact[i].geom2)) ||
                 (b2 >= b.body0 && b2 < b.body0 + B_COUNT && Ground(d_->contact[i].geom1)));
        }
        b.tt[9] = nc;
        b.tt_t = t;
      }
      // The tumble ends only at rest on a flat place: all ground contacts of the body within kFlatDz in height (on the
      // stairs a body lies over 2+ treads, 0.15 m apart; it can slow below kTumbleV there, and the lying fade and rest
      // would start on the slope)
      if (b.tumble && !mv && b.lie_t0 >= 0) {
        double zlo = 1e9, zhi = -1e9;
        for (int i = 0; i < d_->ncon; ++i) {
          const mjContact& c = d_->contact[i];
          const int b1 = m_->geom_bodyid[c.geom1], b2 = m_->geom_bodyid[c.geom2];
          const bool own1 = b1 >= b.body0 && b1 < b.body0 + B_COUNT, own2 = b2 >= b.body0 && b2 < b.body0 + B_COUNT;
          if (!((own1 && Ground(c.geom2)) || (own2 && Ground(c.geom1)))) continue;
          zlo = std::fmin(zlo, (double)c.pos[2]);
          zhi = std::fmax(zhi, (double)c.pos[2]);
        }
        // Switch flat_r: and the ground around the pelvis is level: no drop over kFlatDz within flat_r (a body lying
        // along one tread has all its contacts on it); a point under a support that rises above the pelvis (a wall) is
        // skipped. Off: ending the tumble there lets the limp body slide on
        bool level = zhi >= zlo && zhi - zlo < kFlatDz;
        const double fr = param[P_FLAT_R];
        const double* pc = d_->xpos + 3 * (b.body0 + B_PELVIS);
        const double g0 = GroundBelow(pc[0], pc[1], pc[2]);
        for (int k = 0; k < 4 && level && fr > 0; ++k) {
          bool over = false;
          const double dx = k == 0 ? fr : k == 1 ? -fr : 0.0, dy = k == 2 ? fr : k == 3 ? -fr : 0.0;
          const double g1 = GroundBelow(pc[0] + dx, pc[1] + dy, pc[2], &over);
          if (!over && g1 < g0 - kFlatDz) level = false;
        }
        if (level) b.tumble = 0;  // came to rest lying: the curl is held (curled)
        if (b.tt_t == t) b.tt[10] = zhi >= zlo ? zhi - zlo : -1, b.tt[11] = level, b.tt[12] = level ? 2 : 1;
      }
    }
    b.in.tumble = b.tumble;
    const bool roll_ok = (b.sit_t0 < 0 && b.lie_t0 >= 0) || b.tumble;
    const double up = b.tumble ? 1.0 : wn > kRollW1 && roll_ok ? 1.0 : (wn < kRollW0 || !roll_ok ? (b.curled && roll_ok && b.roll_w > 0 ? 0.0 : -1.0) : 0.0);
    b.roll_w = Clamp(b.roll_w + up * kFrameDt / kRollT, 0.0, 1.0);
    b.roll_a += wn * kFrameDt;
    if (b.roll_w > 0) {
      const double rw = b.roll_w * b.roll_w * (3 - 2 * b.roll_w);
      for (int s = 0; s < 2; ++s) {
        const int hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
        T[hp + 0] += rw * ((b.tumble ? kTuckHip : kRollHip) - T[hp + 0]);
        T[hp + 3] += rw * ((b.tumble ? kTuckKnee : kRollKnee) + kRollKneeA * std::sin(b.roll_a + s) - T[hp + 3]);
        b.gsup[hp + 0] = b.gsup[hp + 3] = std::fmax(b.gsup[hp + 0], rw);
      }
      if (rw > 0.5) b.curled = 1;
      if (b.tumble)  // forearms to the head (shoulders forward-up, elbows folded), the head between them
        for (int s = 0; s < 2; ++s) {
          const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L;
          T[sh + 0] += rw * (kGuardFlex - T[sh + 0]);
          T[sh + 1] += rw * (kGuardAbd - T[sh + 1]);
          T[sh + 2] += rw * (0.0 - T[sh + 2]);
          T[sh + 3] += rw * (kGuardElbow - T[sh + 3]);
        }
      T[J_NECK_PITCH] += rw * (0.5 - T[J_NECK_PITCH]);
      T[J_LUMBAR_PITCH] += rw * (0.3 - T[J_LUMBAR_PITCH]);
      T[J_CHEST_PITCH] += rw * (0.3 - T[J_CHEST_PITCH]);
    }
  }
  // Head: chin to chest for backward falls (neck flexors, Choi et al. 2017), turn to the side for forward falls.
  const double bkc = Clamp((-cf - 0.3) * 2.5, 0.0, 1.0), fwc = Clamp((cf - 0.3) * 2.5, 0.0, 1.0);
  T[J_NECK_PITCH] = kChin * bkc + kNeckFwd * fwc + 0.45 * (1 - bkc - fwc);
  T[J_NECK_YAW] = kHeadTurn * fwc * b.head_turn;
  // (no neck tilt away from the floor in a side fall: it makes the head land harder)
  if (doomed) T[J_LUMBAR_PITCH] = cf < 0 ? 0.3 : 0.1;  // round the back when falling backward
  if (b.tumble) T[J_LUMBAR_PITCH] = T[J_CHEST_PITCH] = kTuckSpine, T[J_NECK_PITCH] = kTuckNeck;  // tumble tuck
  // Lying: the protective pose is no longer needed; targets move to a rest pose (legs nearly straight, arms at
  // the sides, neck and back neutral) so the limbs lie on the floor (the 0.5/0.8 hip/knee fall pose would keep a
  // knee bent with the foot in the air).
  if (b.lie_t0 >= 0) {
    // A body that curled keeps its pose (no rest-pose targets for arms, hips, knees, back and neck: the tone fades
    // below and the limbs sag under their weight)
    const double w = 1 - std::exp(-(t - b.lie_t0) / kLieBlendT);
    const double wr = b.curled ? 0.0 : w;
    auto to = [&](int j, double v) { T[j] += wr * (v - T[j]); };
    for (int s = 0; s < 2; ++s) {
      const int sh = s ? J_SH_FLEX_R : J_SH_FLEX_L, hp = s ? J_HIP_FLEX_R : J_HIP_FLEX_L;
      // Arm abduction by the chest attitude: the angle that points the arm most downward (side lying: along the
      // body, the arm must not stand up on top; flat on the back or belly any angle lies on the floor).
      const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
      const double sy = s ? -1.0 : 1.0, vv = std::hypot(Rc[7], Rc[8]);
      const double down = Clamp(std::atan2(-sy * Rc[7], Rc[8]), -0.3, 1.6);
      const double wv = Clamp((vv - 0.2) * 3.0, 0.0, 1.0);
      // the arms go to rest only as the catch strength falls (else they leave the floor while the body still slides)
      const double wa = 1.0 - b.catch_g;
      T[sh + 0] += wa * wr * (0.0 - T[sh + 0]);
      T[sh + 1] += wa * wr * (kLieAbd + wv * (down - kLieAbd) - T[sh + 1]);
      T[sh + 3] += wa * wr * (0.2 - T[sh + 3]);
      if (b.sit_t0 < 0) { to(hp + 0, 0.1); to(hp + 3, 0.1); to(hp + 4, -0.6); }  // relaxed foot: plantar flexed
      // after a backward seat the legs stay with the seat block (thighs led in the world, feet on the floor)
    }
    to(J_NECK_PITCH, 0.0);
    // Lying, the legs and the back lie by their weight (with the L0 support the legs of a forward fall stay up); once
    // quiet the neck lets go too (else the head stays up, "chin to chest")
    const double vq = std::sqrt(P.com_vel[0] * P.com_vel[0] + P.com_vel[1] * P.com_vel[1] + P.com_vel[2] * P.com_vel[2]);
    if (vq > kQuietV) b.quiet_t = t;
    for (int j = 0; j < J_COUNT; ++j)
      if (j < J_NECK_PITCH || j >= J_HIP_FLEX_L) b.gsup[j] = std::fmin(b.gsup[j], 1.0 - w);
    if (t - b.quiet_t > kQuietT) {
      const double wn = Clamp((t - b.quiet_t - kQuietT) / kQuietT, 0.0, 1.0);
      for (int j = J_NECK_PITCH; j <= J_NECK_YAW; ++j) b.gsup[j] = 1.0 - wn;
      b.neck_quiet = wn;
    }
    to(J_NECK_YAW, 0.0);
    to(J_LUMBAR_PITCH, 0.0);
  }
  // Stiffness: arms brace (shoulders stiff, elbows give way once the hand is down), neck co-contracts to hold the
  // head, legs soft (balancer already scaled kp by 0.3); everything fades when lying.
  for (int j = 0; j < J_COUNT; ++j) {
    const bool arm = j >= J_SH_FLEX_L && j <= J_WRIST_R;
    if (arm) {
      b.kp[j] *= kArmKp;
      const int s = j >= J_SH_FLEX_R ? 1 : 0;
      if ((j == J_ELBOW_L || j == J_ELBOW_R) && hand_down[s]) b.kp[j] *= kElbowYield;
    }
    if (j >= J_NECK_PITCH && j <= J_NECK_YAW) b.kp[j] *= kNeckKp * (1.0 - 0.9 * b.neck_quiet);  // quiet: neck lets go
    // switch fall_legs_fwd: forward falls (see the param table)
    // only a living body falling from a push (no bullet reaction, not dying)
    const bool push_fall = b.dead_t < 0 && !(b.hit_bullet && t < b.bul_until) && !b.down;
    if (stiff_legs && j >= J_HIP_FLEX_L) b.kp[j] *= cf > 0 && push_fall ? param[P_FALL_LEGS_FWD] : param[P_FALL_LEGS];
    // an arm's stiffness follows its catch strength (not the time since lying)
    const double ta = arm ? std::fmax(std::fmax(tone, kLimbTone), b.arm_a[j >= J_SH_FLEX_R ? 1 : 0]) : tone;
    b.kp[j] *= j >= J_HIP_FLEX_L ? leg_tone : ta;
  }
}

void World::CarSubstep() {
  if (car_qadr_ < 0 || car_speed_ <= 0) return;
  if (!car_hit_) {
    const int cb = car_body_;
    for (int i = 0; i < d_->ncon; ++i) {
      const mjContact& c = d_->contact[i];
      if (m_->geom_bodyid[c.geom1] == cb || m_->geom_bodyid[c.geom2] == cb) { car_hit_ = 1; break; }
    }
  } else {
    car_speed_ = std::fmax(0.0, car_speed_ - kCarBrake * m_->opt.timestep);
  }
  // Kinematic car: velocity is prescribed every substep (contact reactions do not slow it).
  d_->qvel[car_dadr_] = car_speed_ * car_dir_[0];
  d_->qvel[car_dadr_ + 1] = car_speed_ * car_dir_[1];
  d_->qvel[car_dadr_ + 2] = 0;
}

}  // namespace emil
