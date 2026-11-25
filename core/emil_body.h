// L0 body: anthropometric segment model built from height/mass/sex/build, emitted as MJCF text.
#pragma once
#include <string>

namespace emil {

struct BodyParams {
  float height = 1.78f;  // [m]
  float mass = 80.0f;    // [kg]
  bool female = false;
  float build = 0.0f;    // -1 thin .. 0 average .. +1 heavy: scales thickness and trunk mass share
  int sizes = 0;         // segment sizes: 0 de Leva 1996, 1 alternative character proportions (1.75 m, T-pose joints)
  float knee_flex = 135.0f;  // model knee flexion limit [deg]: 135 = active limit (default), 150 = passive (switch:
                             // heel to buttock; the hip limit 120 still keeps a dead body from sitting on its heels)
  float ankle_dorsi = 40.0f;  // model ankle dorsiflexion limit [deg]: 40 = passive under body weight (default: with
                              // the active 20 a back step landing toe first keeps the heel up and the feet can leave
                              // the floor), 20 = active limit
  int shapes = 1;        // collision shapes: 1 detailed (skull mesh, neck, yoke, palm boxes, trunk
                         // ellipsoids), 0 simple (ANSUR ellipsoids, thin neck, hand sticks)
};

// Bodies per bot, fixed order (index = bot_body_base + id).
enum BodyId {
  B_PELVIS, B_LUMBAR, B_CHEST, B_HEAD,
  B_UARM_L, B_FARM_L, B_HAND_L, B_UARM_R, B_FARM_R, B_HAND_R,
  B_THIGH_L, B_SHANK_L, B_FOOT_L, B_THIGH_R, B_SHANK_R, B_FOOT_R,
  B_COUNT
};

// Hinge joints per bot, fixed order; each has one muscle (motor actuator) in the same order.
enum JointId {
  J_LUMBAR_PITCH, J_LUMBAR_ROLL, J_LUMBAR_YAW,
  J_CHEST_PITCH, J_CHEST_ROLL, J_CHEST_YAW,
  J_NECK_PITCH, J_NECK_ROLL, J_NECK_YAW,
  J_SH_FLEX_L, J_SH_ABD_L, J_SH_ROT_L, J_ELBOW_L, J_WRIST_L,
  J_SH_FLEX_R, J_SH_ABD_R, J_SH_ROT_R, J_ELBOW_R, J_WRIST_R,
  J_HIP_FLEX_L, J_HIP_ABD_L, J_HIP_ROT_L, J_KNEE_L, J_ANKLE_DORSI_L, J_ANKLE_INV_L,
  J_HIP_FLEX_R, J_HIP_ABD_R, J_HIP_ROT_R, J_KNEE_R, J_ANKLE_DORSI_R, J_ANKLE_INV_R,
  J_COUNT
};

// Per-joint static data computed from BodyParams (used by muscles).
struct JointSpec {
  float lo, hi;        // range [rad]
  float tau_max;       // muscle torque limit [N m]
  float kp;            // base stiffness [N m/rad] before global scale
};

struct BodyDims {      // key lengths for controllers and metrics
  float ankle_h, shank, thigh, hip_half_w, foot_len, foot_w, heel_back, total_mass;
};

// Compute joint specs and dims. Sign conventions: positive = flexion / abduction / dorsiflexion / inversion.
void ComputeBody(const BodyParams& p, JointSpec out_j[J_COUNT], BodyDims* dims);

// Append MJCF <body> tree for one bot named "b<idx>_*" standing at (x, y) to xml.
// Also appends actuator and contact-exclude XML to the given strings.
// Bench only: extra contact exclusions to size the cost of pair classes. Bits: 1 arm-leg, 2 arm-trunk, 4 leg-leg.
extern int g_body_tune;
extern std::string g_body_foot;  // bench only: extra attributes of the foot box geoms, empty by default
void AppendBotXml(int idx, const BodyParams& p, float x, float y, std::string* bodies,
                  std::string* actuators, std::string* excludes);

// Per-bot <asset> entries (the skull mesh, scaled to the body).
void AppendBotAssets(int idx, const BodyParams& p, std::string* assets);

const char* JointName(int j);
const char* BodyName(int b);

}  // namespace emil
