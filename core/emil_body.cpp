// L0 body builder. Anthropometry: de Leva 1996 (mass fractions, CoM, radii of gyration, mean lengths);
// vertical proportions not in de Leva (ankle, shoulder height) from Winter 2009. See docs/SOURCES.md.
#include "emil_body.h"
#include "skull_mesh.h"

#include <cmath>
#include <cstdio>

namespace emil {
namespace {

constexpr float kDeg = 3.14159265f / 180.0f;

// de Leva 1996 segment table (Table 4 male/female, joint-centre endpoints). Lengths: mean values [m],
// table statures 1.741 m (male), 1.735 m (female).
struct Seg { float mass, com, rs, rt, rl, len; };
enum { S_HEAD, S_UPT, S_MIDT, S_LOWT, S_UARM, S_FARM, S_HAND, S_THIGH, S_SHANK, S_FOOT, S_N };
// head: VERT-CERV row ("Head 2"; female CoM/radii borrowed from male row); trunk: SUPR-XYPH, XYPH-OMPH, OMPH-MIDH.
const Seg kMale[S_N] = {
    {0.0694f, 0.5002f, 0.303f, 0.315f, 0.261f, 0.2429f}, {0.1596f, 0.2999f, 0.716f, 0.454f, 0.659f, 0.1707f},
    {0.1633f, 0.4502f, 0.482f, 0.383f, 0.468f, 0.2155f}, {0.1117f, 0.6115f, 0.615f, 0.551f, 0.587f, 0.1457f},
    {0.0271f, 0.5772f, 0.285f, 0.269f, 0.158f, 0.2817f}, {0.0162f, 0.4574f, 0.276f, 0.265f, 0.121f, 0.2689f},
    {0.0061f, 0.7900f, 0.628f, 0.513f, 0.401f, 0.0862f}, {0.1416f, 0.4095f, 0.329f, 0.329f, 0.149f, 0.4222f},
    {0.0433f, 0.4395f, 0.251f, 0.246f, 0.102f, 0.4340f}, {0.0137f, 0.4415f, 0.257f, 0.245f, 0.124f, 0.2581f}};
const Seg kFemale[S_N] = {
    {0.0668f, 0.5002f, 0.303f, 0.315f, 0.261f, 0.2437f}, {0.1545f, 0.2077f, 0.746f, 0.502f, 0.718f, 0.1425f},
    {0.1465f, 0.4512f, 0.433f, 0.354f, 0.415f, 0.2053f}, {0.1247f, 0.4920f, 0.433f, 0.402f, 0.444f, 0.1815f},
    {0.0255f, 0.5754f, 0.278f, 0.260f, 0.148f, 0.2751f}, {0.0138f, 0.4559f, 0.261f, 0.257f, 0.094f, 0.2643f},
    {0.0056f, 0.7474f, 0.531f, 0.454f, 0.335f, 0.0780f}, {0.1478f, 0.3612f, 0.369f, 0.364f, 0.162f, 0.3685f},
    {0.0481f, 0.4352f, 0.267f, 0.263f, 0.092f, 0.4323f}, {0.0129f, 0.4014f, 0.299f, 0.279f, 0.139f, 0.2283f}};

struct JDef { const char* name; float lo_deg, hi_deg, tau80, kp80; };
// Ranges: AAOS normal ROM (Greene & Heckman 1994), thoracolumbar ROM split between 2 spine joints.
// tau80: max torque for 80 kg / 1.78 m male. kp80: base stiffness (tuned on the benches).
const JDef kJ[J_COUNT] = {
    {"lumbar_pitch", -12, 40, 200, 500}, {"lumbar_roll", -17, 17, 150, 400}, {"lumbar_yaw", -22, 22, 100, 200},
    {"chest_pitch", -12, 40, 200, 400}, {"chest_roll", -17, 17, 150, 300}, {"chest_yaw", -22, 22, 100, 150},
    {"neck_pitch", -45, 45, 35, 40}, {"neck_roll", -45, 45, 35, 40}, {"neck_yaw", -60, 60, 20, 20},
    {"sh_flex_l", -60, 180, 80, 40}, {"sh_abd_l", -30, 180, 80, 40}, {"sh_rot_l", -70, 90, 50, 15},
    {"elbow_l", 0, 150, 70, 20}, {"wrist_l", -70, 80, 15, 3},
    {"sh_flex_r", -60, 180, 80, 40}, {"sh_abd_r", -30, 180, 80, 40}, {"sh_rot_r", -70, 90, 50, 15},
    {"elbow_r", 0, 150, 70, 20}, {"wrist_r", -70, 80, 15, 3},
    {"hip_flex_l", -30, 120, 250, 700}, {"hip_abd_l", -30, 45, 150, 700}, {"hip_rot_l", -45, 45, 80, 150},
    {"knee_l", 0, 135, 220, 700}, {"ankle_dorsi_l", -50, 20, 180, 500}, {"ankle_inv_l", -15, 35, 60, 150},
    {"hip_flex_r", -30, 120, 250, 700}, {"hip_abd_r", -30, 45, 150, 700}, {"hip_rot_r", -45, 45, 80, 150},
    {"knee_r", 0, 135, 220, 700}, {"ankle_dorsi_r", -50, 20, 180, 500}, {"ankle_inv_r", -15, 35, 60, 150}};
// BodyParams.knee_flex 150 (switch): passive knee flexion up to ~150 deg, the calf meets the back of the thigh
// (knee-pain-explained.com knee range of motion); active 135 in the table above.
// BodyParams.ankle_dorsi 40 (switch): passive ankle dorsiflexion under body weight - weight-bearing lunge 34 deg
// is the low-risk cut-off, the population range 8-52 (PubMed 30321953); active 20 above.

const char* kBodyNames[B_COUNT] = {"pelvis", "lumbar", "chest", "head", "uarm_l", "farm_l", "hand_l",
                                   "uarm_r", "farm_r", "hand_r", "thigh_l", "shank_l", "foot_l",
                                   "thigh_r", "shank_r", "foot_r"};

struct Derived {
  Seg s[S_N];      // scaled: len in m, mass in kg
  float H, tf;     // height, thickness factor
  float ankle_h, sh_z, sh_half, hip_half, heel_back, foot_w;
  float foot_h, r_thigh, r_shank, r_uarm, r_farm;  // foot box height, limb capsule radii (before the build factor)
};

Derived Derive(const BodyParams& p) {
  Derived d;
  const Seg* t = p.female ? kFemale : kMale;
  const float ref = p.female ? 1.735f : 1.741f;
  const float sc = p.height / ref;
  // Build shifts mass share toward the trunk (+) or limbs (-), renormalised to total 1.
  const float b = p.build < -1 ? -1 : (p.build > 1 ? 1 : p.build);
  float sum = 0;
  for (int i = 0; i < S_N; ++i) {
    d.s[i] = t[i];
    d.s[i].len *= sc;
    if (i == S_UPT || i == S_MIDT || i == S_LOWT) d.s[i].mass *= 1.0f + 0.10f * b;
    const float mult = (i >= S_UARM && i != S_HEAD) ? 2.0f : 1.0f;  // limbs counted twice
    sum += d.s[i].mass * mult;
  }
  for (int i = 0; i < S_N; ++i) d.s[i].mass *= p.mass / sum;
  d.H = p.height;
  d.tf = 1.0f + 0.25f * b;
  d.ankle_h = 0.039f * p.height;      // Winter 2009 Fig 4.1
  d.sh_z = 0.818f * p.height;         // Winter 2009 Fig 4.1 (shoulder height)
  d.sh_half = 0.098f * p.height;      // biacromial 0.259H (Winter) minus acromion-to-joint offset (own estimate)
  d.hip_half = 0.050f * p.height;     // hip joint centres ~0.10H apart (own estimate, ~pelvis breadth share)
  d.heel_back = 0.20f * d.s[S_FOOT].len;
  d.foot_w = 0.38f * d.s[S_FOOT].len;
  // Limb radii = girth / 2 pi (ANSUR II, see AppendBotXml); foot box as high as the ankle.
  const float sa = p.height / 1.756f;
  d.foot_h = d.ankle_h;
  d.r_thigh = 0.082f * sa; d.r_shank = 0.056f * sa; d.r_uarm = 0.053f * sa; d.r_farm = 0.0387f * sa;
  if (p.sizes == 1) {
    // Alternative character proportions (T-pose joints, 1.75 m table), scaled by H / 1.75: segment lengths, joint
    // half distances and heights; the spine joints at the heights of Spine1, Spine2 and Neck (pelvis segment
    // hip -> Spine1, lumbar Spine1 -> Spine2, chest Spine2 -> Neck, head Neck -> vertex). Masses: de Leva shares;
    // CoM and radii of gyration: de Leva shares of the new lengths. Limb thicknesses from the same character.
    const float k = p.height / 1.75f;
    d.s[S_THIGH].len = 0.4285f * k;
    d.s[S_SHANK].len = 0.3938f * k;
    d.s[S_UARM].len = 0.2531f * k;
    d.s[S_FARM].len = 0.2317f * k;
    d.s[S_FOOT].len = 0.26f * k;                 // foot box 26 cm
    d.ankle_h = 0.1172f * k;
    d.s[S_LOWT].len = (1.2094f - 0.9357f) * k;
    d.s[S_MIDT].len = (1.3435f - 1.2094f) * k;
    d.s[S_UPT].len = (1.4942f - 1.3435f) * k;
    d.s[S_HEAD].len = p.height - (d.ankle_h + d.s[S_SHANK].len + d.s[S_THIGH].len + d.s[S_LOWT].len +
                                   d.s[S_MIDT].len + d.s[S_UPT].len);
    d.sh_z = 1.4291f * k;
    d.sh_half = 0.2107f * k;
    d.hip_half = 0.0982f * k;
    d.heel_back = 0.045f * k;                    // heel 4.5 cm behind the ankle, toe end 21.5 cm ahead (table: 23.6)
    d.foot_w = 0.095f * k;
    d.foot_h = 0.063f * k;
    d.r_thigh = 0.078f * k; d.r_shank = 0.056f * k; d.r_uarm = 0.0388f * k; d.r_farm = 0.0354f * k;
  }
  return d;
}

void Inertial(std::string* o, float x, float y, float z, float m, float ix, float iy, float iz) {
  char buf[200];
  snprintf(buf, sizeof buf, "<inertial pos='%.4f %.4f %.4f' mass='%.4f' diaginertia='%.5f %.5f %.5f'/>", x, y, z,
           m, ix, iy, iz);
  *o += buf;
}
void InertialSeg(std::string* o, const Seg& s, float x, float y, float z, bool foot = false) {
  const float L2 = s.len * s.len;
  const float is = s.mass * s.rs * s.rs * L2, it = s.mass * s.rt * s.rt * L2, il = s.mass * s.rl * s.rl * L2;
  if (foot) Inertial(o, x, y, z, s.mass, il, is, it);
  else Inertial(o, x, y, z, s.mass, is, it, il);
}

}  // namespace

const char* JointName(int j) { return kJ[j].name; }
const char* BodyName(int b) { return kBodyNames[b]; }

void ComputeBody(const BodyParams& p, JointSpec out_j[J_COUNT], BodyDims* dims) {
  const Derived d = Derive(p);
  const float strength = (p.mass / 80.0f) * (p.height / 1.78f);  // torque ~ m g L
  for (int j = 0; j < J_COUNT; ++j) {
    out_j[j].lo = kJ[j].lo_deg * kDeg;
    out_j[j].hi = kJ[j].hi_deg * kDeg;
    out_j[j].tau_max = kJ[j].tau80 * strength;
    out_j[j].kp = kJ[j].kp80 * strength;
  }
  dims->ankle_h = d.ankle_h;
  dims->shank = d.s[S_SHANK].len;
  dims->thigh = d.s[S_THIGH].len;
  dims->hip_half_w = d.hip_half;
  dims->foot_len = d.s[S_FOOT].len;
  dims->foot_w = d.foot_w;
  dims->heel_back = d.heel_back;
  dims->total_mass = p.mass;
}

// Detailed collision shapes (defined at 1.75 m): skull hull scaled by H / 1.75.
void AppendBotAssets(int idx, const BodyParams& p, std::string* o) {
  if (!p.shapes) return;
  char b[96];
  snprintf(b, sizeof b, "<mesh name='b%d_skull' scale='%.4f %.4f %.4f' vertex='", idx, p.height / 1.75f,
           p.height / 1.75f, p.height / 1.75f);
  *o += b;
  for (int i = 0; i < 3 * kSkullNV; ++i) {
    snprintf(b, sizeof b, "%.4f ", kSkullV[i]);
    *o += b;
  }
  *o += "'/>";
}

int g_body_tune = 0;
std::string g_body_foot;

void AppendBotXml(int idx, const BodyParams& p, float x, float y, std::string* o, std::string* act,
                  std::string* exc) {
  const Derived d = Derive(p);
  const Seg* s = d.s;
  const float H = d.H, tf = d.tf, sc = H / 1.78f;
  const float z_hip = d.ankle_h + s[S_SHANK].len + s[S_THIGH].len;
  const float z_omph = z_hip + s[S_LOWT].len, z_xyph = z_omph + s[S_MIDT].len;
  const float z_cerv = H - s[S_HEAD].len;
  char b[512];
  auto name = [&](int body) {
    static char nb[64];
    snprintf(nb, sizeof nb, "b%d_%s", idx, kBodyNames[body]);
    return nb;
  };
  auto joint = [&](int j, float ax, float ay, float az) {
    // Knee and ankle dorsiflexion: the model range can go to the passive limits (BodyParams switches) while the
    // controllers keep the active ones (JointSpec hi)
    const float hi = (j == J_KNEE_L || j == J_KNEE_R)                 ? p.knee_flex
                     : (j == J_ANKLE_DORSI_L || j == J_ANKLE_DORSI_R) ? p.ankle_dorsi
                                                                      : kJ[j].hi_deg;
    snprintf(b, sizeof b, "<joint name='b%d_%s' axis='%g %g %g' range='%.4f %.4f'/>", idx, kJ[j].name, ax, ay, az,
             kJ[j].lo_deg * kDeg, hi * kDeg);
    *o += b;
  };
  auto open = [&](int body, float px, float py, float pz) {
    snprintf(b, sizeof b, "<body name='%s' pos='%.4f %.4f %.4f'>", name(body), px, py, pz);
    *o += b;
  };
  auto capsule = [&](float x0, float y0, float z0, float x1, float y1, float z1, float r) {
    snprintf(b, sizeof b, "<geom type='capsule' fromto='%.4f %.4f %.4f %.4f %.4f %.4f' size='%.4f'/>", x0, y0, z0, x1,
             y1, z1, r);
    *o += b;
  };
  // Ellipsoid (semi-axes x fwd, y left, z up): smooth trunk parts that overlap into one solid body.
  auto ellipsoid = [&](float px, float py, float pz, float ax, float ay, float az) {
    snprintf(b, sizeof b, "<geom type='ellipsoid' pos='%.4f %.4f %.4f' size='%.4f %.4f %.4f'/>", px, py, pz, ax, ay, az);
    *o += b;
  };
  // Trunk sizes: ANSUR II 2012 male means (n 4082, stature 1756 mm), scaled to the body height; half of breadth
  // and depth -> semi-axes. Chest breadth 289, depth 254; waist 326 x 238; hip breadth 346, buttock depth 246;
  // neck circumference 398 mm (docs/SOURCES.md). tf scales the thickness with the build. Limb capsule radius =
  // girth / 2 pi: thigh mean of upper and lower thigh 625, 409; shank 0.75 calf 392 + 0.25 ankle 229; upper arm
  // biceps 358; forearm mean of forearm 310 and wrist 176. Head: length 200, breadth 154, height ~230.
  // Trunk parts are tall and overlap ~10 cm so the profile is flat (no "snowman" waists between them).
  const float sa = H / 1.756f;

  // Pelvis: lower trunk (OMPH-MIDH), origin at mid hip joint centre.
  open(B_PELVIS, x, y, z_hip);
  *o += "<freejoint/>";
  InertialSeg(o, s[S_LOWT], 0, 0, (1 - s[S_LOWT].com) * s[S_LOWT].len);
  // Hip region: wraps the hip joints and the top of the thighs; the buttocks reach 0.10 m below the hip axis.
  const float zp0 = -0.10f * sc, zp1 = s[S_LOWT].len + 0.08f * sc;
  // Detailed shapes, defined on a 1.75 m character: widths and depths x H / 1.75 (x build), heights mapped
  // between the hip joint (0.9357 m) and the neck joint (1.4942 m) of that character.
  const float so = H / 1.75f * tf, kv = (z_cerv - z_hip) / (1.4942f - 0.9357f);
  auto zo = [&](float z) { return z_hip + (z - 0.9357f) * kv; };  // character height -> this body's height
  auto sphere = [&](float px, float py, float pz, float r) {
    snprintf(b, sizeof b, "<geom type='sphere' pos='%.4f %.4f %.4f' size='%.4f'/>", px, py, pz, r);
    *o += b;
  };
  if (p.shapes) {
    // Hips_g: capsule across the hip joints and buttocks (ends at the hip breadth 346 mm, ANSUR II); Hips_pelvis_g:
    // ellipsoid over the iliac crests, the bridge from the buttocks to the waist.
    const float hy = std::fmax(0.173f * sa - 0.105f * so, 0.02f);
    capsule(-0.015f * sc, -hy, 0.0f, -0.015f * sc, hy, 0.0f, 0.105f * so);
    ellipsoid(0.0f, 0, zo(1.035f) - z_hip, 0.100f * so, 0.152f * so, 0.100f * kv);
  } else {
    ellipsoid(-0.01f * sc, 0, 0.5f * (zp0 + zp1), 0.123f * sa * tf, 0.173f * sa * tf, 0.5f * (zp1 - zp0));
  }
  // Lumbar: middle trunk (XYPH-OMPH). Trunk CoM sits anterior to the spine joints (soft spine).
  open(B_LUMBAR, 0, 0, s[S_LOWT].len);
  joint(J_LUMBAR_PITCH, 0, 1, 0); joint(J_LUMBAR_ROLL, 1, 0, 0); joint(J_LUMBAR_YAW, 0, 0, 1);
  InertialSeg(o, s[S_MIDT], 0.02f * sc, 0, (1 - s[S_MIDT].com) * s[S_MIDT].len);
  // Waist: overlaps the hip region and the chest by ~4 cm each so no gap opens when the spine bends.
  const float zl0 = -0.12f * sc, zl1 = s[S_MIDT].len + 0.10f * sc;
  if (p.shapes)  // Spine1_g: the waist, the narrowest part
    ellipsoid(0.005f * sc, 0, zo(1.120f) - z_omph, 0.095f * so, 0.137f * so, 0.190f * kv);
  else
    ellipsoid(-0.01f * sc, 0, 0.5f * (zl0 + zl1), 0.119f * sa * tf, 0.160f * sa * tf, 0.5f * (zl1 - zl0));
  // Chest: upper trunk (SUPR-XYPH), clavicles rigid.
  open(B_CHEST, 0, 0, s[S_MIDT].len);
  joint(J_CHEST_PITCH, 0, 1, 0); joint(J_CHEST_ROLL, 1, 0, 0); joint(J_CHEST_YAW, 0, 0, 1);
  InertialSeg(o, s[S_UPT], 0.03f * sc, 0, (1 - s[S_UPT].com) * s[S_UPT].len);
  // Chest: from below the xiphion up to the base of the neck.
  const float zt0 = -0.14f * sc, zt1 = 0.92f * (z_cerv - z_xyph);
  if (p.shapes) {
    // Spine2_g lower ribs, Spine3_g chest: large overlaps, no waists between; Spine3_yoke_g: flat ellipsoid from
    // shoulder to shoulder over the neck-to-shoulder line at the back and top, its ends in the deltoid spheres;
    // LeftShoulder_g: clavicle capsule from the neck base to the shoulder joint.
    ellipsoid(0.010f * sc, 0, zo(1.250f) - z_xyph, 0.104f * so, 0.147f * so, 0.190f * kv);
    ellipsoid(0.010f * sc, 0, zo(1.335f) - z_xyph, 0.112f * so, 0.162f * so, 0.175f * kv);
    ellipsoid(-0.020f * sc, 0, d.sh_z - z_xyph + 0.010f * sc, 0.092f * so, 0.185f * so * d.sh_half / (0.2107f * sa),
              0.082f * so);
    for (int k = -1; k <= 1; k += 2)
      capsule(0, k * 0.0705f * sa, zo(1.4726f) - z_xyph, 0, k * d.sh_half, d.sh_z - z_xyph, 0.042f * so);
  } else {
    ellipsoid(0.01f * sc, 0, 0.5f * (zt0 + zt1), 0.127f * sa * tf, 0.150f * sa * tf, 0.5f * (zt1 - zt0));
  }
  // Shoulder girdle across the chest top between the shoulder joints: bideltoid breadth 510 mm (ANSUR II) =
  // 2 (sh_half + r) + 2 upper-arm radii -> r ~ 0.05 m at 1.78 m (the arms hang beside the chest).
  if (!p.shapes) {
    const float zs = d.sh_z - z_xyph, rs = 0.050f * sa * tf;
    capsule(-0.01f * sc, -d.sh_half, zs, -0.01f * sc, d.sh_half, zs, rs);
  }
  // Head + neck (VERT-CERV).
  open(B_HEAD, 0, 0, z_cerv - z_xyph);
  joint(J_NECK_PITCH, 0, 1, 0); joint(J_NECK_ROLL, 1, 0, 0); joint(J_NECK_YAW, 0, 0, 1);
  InertialSeg(o, s[S_HEAD], 0.02f * sc, 0, (1 - s[S_HEAD].com) * s[S_HEAD].len);
  // Neck (thinner than the head) from inside the chest top to the skull base; head above it.
  const float rh = 0.09f * sc;
  // Neck (ANSUR II male means, mm, stature 1756): cervicale 1517, suprasternale 1439 (78 below C7),
  // tragion 1625 (108 above C7), base-of-neck circumference 435 (diameter 138). Base centre: 69 mm in front of the
  // C7 skin (x = -0.09 here, back of the upper chest), halfway between C7 and the sternal notch in height; axis 17 deg
  // forward (15-20 deg) up to the tragion level. Radius from the neck circumference 398.
  if (p.shapes) {
    // Detailed neck: capsule radius 5 cm at 1.75 m from the neck joint to just above the head joint (head joint
    // 7.4 cm above, 2.9 cm ahead: 21 deg forward); skull: the convex skull mesh (the back of the head hangs behind
    // the neck), top at the vertex. Mass and inertia stay the de Leva head above.
    const float k = H / 1.75f;
    capsule(-0.006f * k, 0, -0.020f * k, 0.033f * k, 0, 0.085f * k, 0.050f * so);
    const float hz = s[S_HEAD].len - 0.205f * k;  // skull top (0.205 above the head joint) at the vertex
    // The face midline (nose line) of the mesh data leans 4.6 deg top forward: turned 4.6 deg top back about the
    // head joint so the face is vertical on a level head link (otherwise the face looks down)
    snprintf(b, sizeof b, "<geom type='mesh' mesh='b%d_skull' pos='%.4f 0 %.4f' quat='0.99919 0 -0.04013 0'/>", idx,
             0.0285f * k, hz);
    *o += b;
  } else {
    const float bx = -0.021f * sa, bz = -0.039f * sa, tz = 0.070f * sa;
    const float tx = bx + (tz - bz) * 0.3057f;  // tan 17 deg
    capsule(bx, 0, bz, tx, 0, tz, 0.0633f * sa * tf);
    // Skull: smooth ellipsoid (no face features), top at the vertex, centre forward of the cervicale.
    ellipsoid(0.027f * sa, 0, s[S_HEAD].len - 0.115f * sa, 0.100f * sa, 0.077f * sa, 0.115f * sa);
  }
  (void)rh;
  *o += "</body>";
  // Arms. side +1 = left (+y).
  for (int side = 0; side < 2; ++side) {
    const float sy = side == 0 ? 1.0f : -1.0f;
    const int j0 = side == 0 ? J_SH_FLEX_L : J_SH_FLEX_R;
    const int b0 = side == 0 ? B_UARM_L : B_UARM_R;
    open(b0, 0, sy * d.sh_half, d.sh_z - z_xyph);
    joint(j0 + 0, 0, -1, 0); joint(j0 + 1, sy, 0, 0); joint(j0 + 2, 0, 0, sy);
    InertialSeg(o, s[S_UARM], 0, 0, -s[S_UARM].com * s[S_UARM].len);
    capsule(0, 0, 0, 0, 0, -s[S_UARM].len, d.r_uarm * tf);
    if (p.shapes) sphere(0, 0, 0, 0.054f * so);  // LeftShoulder_delt_g: deltoid ball on the shoulder joint
    open(b0 + 1, 0, 0, -s[S_UARM].len);
    joint(j0 + 3, 0, -1, 0);
    InertialSeg(o, s[S_FARM], 0, 0, -s[S_FARM].com * s[S_FARM].len);
    capsule(0, 0, 0, 0, 0, -s[S_FARM].len, d.r_farm * tf);
    open(b0 + 2, 0, 0, -s[S_FARM].len);
    joint(j0 + 4, 0, -1, 0);
    InertialSeg(o, s[S_HAND], 0, 0, -s[S_HAND].com * s[S_HAND].len);
    if (p.shapes) {  // LeftHand_g: palm plate 7.8 x 14.5 x 2.7 cm, palm toward the thigh in the stance
      const float k = H / 1.75f;
      snprintf(b, sizeof b, "<geom type='box' pos='0 0 %.4f' size='%.4f %.4f %.4f'/>", -0.0727f * k, 0.0391f * k,
               0.0136f * k, 0.0727f * k);
      *o += b;
    } else {
      capsule(0, 0, -0.02f * sc, 0, 0, -1.4f * s[S_HAND].len, 0.025f * sc);
    }
    *o += "</body></body></body>";
  }
  *o += "</body></body>";  // chest, lumbar
  // Legs.
  for (int side = 0; side < 2; ++side) {
    const float sy = side == 0 ? 1.0f : -1.0f;
    const int j0 = side == 0 ? J_HIP_FLEX_L : J_HIP_FLEX_R;
    const int b0 = side == 0 ? B_THIGH_L : B_THIGH_R;
    open(b0, 0, sy * d.hip_half, 0);
    joint(j0 + 0, 0, -1, 0); joint(j0 + 1, sy, 0, 0); joint(j0 + 2, 0, 0, sy);
    InertialSeg(o, s[S_THIGH], 0, 0, -s[S_THIGH].com * s[S_THIGH].len);
    // Capped below the hip half-width: thick builds otherwise overlap the two thighs at rest (the contact pushes
    // the legs apart and upsets the stance).
    capsule(0, 0, -0.03f * sc, 0, 0, -s[S_THIGH].len, std::fmin(d.r_thigh * tf, 0.95f * d.hip_half));
    open(b0 + 1, 0, 0, -s[S_THIGH].len);
    joint(j0 + 3, 0, 1, 0);
    InertialSeg(o, s[S_SHANK], 0, 0, -s[S_SHANK].com * s[S_SHANK].len);
    capsule(0, 0, 0, 0, 0, -(s[S_SHANK].len - 0.01f), d.r_shank * tf);
    open(b0 + 2, 0, 0, -s[S_SHANK].len);
    joint(j0 + 4, 0, -1, 0); joint(j0 + 5, -sy, 0, 0);
    const float fl = s[S_FOOT].len, fx = fl * 0.5f - d.heel_back;
    InertialSeg(o, s[S_FOOT], s[S_FOOT].com * fl - d.heel_back, 0, -0.5f * d.ankle_h, true);
    // sole on the floor: box from the sole up foot_h (alternative proportions: ankle 11.7 cm up, box 6.3 cm high)
    snprintf(b, sizeof b, "<geom type='box' pos='%.4f 0 %.4f' size='%.4f %.4f %.4f'%s/>", fx,
             -d.ankle_h + 0.5f * d.foot_h, 0.5f * fl, 0.5f * d.foot_w, 0.5f * d.foot_h, (std::string((g_body_tune & 8) ? " condim='1' " : " ") + g_body_foot).c_str());
    *o += b;
    *o += "</body></body></body>";
  }
  *o += "</body>";  // pelvis

  for (int j = 0; j < J_COUNT; ++j) {
    const float t = kJ[j].tau80 * (p.mass / 80.0f) * (p.height / 1.78f);
    snprintf(b, sizeof b, "<motor name='b%d_m_%s' joint='b%d_%s' ctrlrange='%.2f %.2f'/>", idx, kJ[j].name, idx,
             kJ[j].name, -t, t);
    *act += b;
  }
  // Non-adjacent pairs that overlap at rest by construction (parent-child pairs are filtered by MuJoCo).
  const int pairs[][2] = {{B_PELVIS, B_CHEST}, {B_LUMBAR, B_THIGH_L}, {B_LUMBAR, B_THIGH_R}, {B_CHEST, B_HEAD - 0},
                          {B_LUMBAR, B_HEAD}, {B_LUMBAR, B_UARM_L}, {B_LUMBAR, B_UARM_R},
                          {B_UARM_L, B_HAND_L}, {B_UARM_R, B_HAND_R},
                          {B_CHEST, B_HAND_L}, {B_CHEST, B_HAND_R}};  // hand-arm, hand-chest: limp arms stick on them
  for (auto& pr : pairs) {
    if (pr[0] == B_CHEST && pr[1] == B_HEAD) continue;  // parent-child already filtered
    char n0[64];
    snprintf(n0, sizeof n0, "%s", name(pr[0]));
    snprintf(b, sizeof b, "<exclude body1='%s' body2='%s'/>", n0, name(pr[1]));
    *exc += b;
  }
  if (g_body_tune) {
    auto ex = [&](int a, int c) {
      snprintf(b, sizeof b, "<exclude body1='%s' body2='%s'/>", name(a), name(c));
      *exc += b;
    };
    const int arm[] = {B_UARM_L, B_FARM_L, B_HAND_L, B_UARM_R, B_FARM_R, B_HAND_R};
    const int leg[] = {B_THIGH_L, B_SHANK_L, B_THIGH_R, B_SHANK_R};
    const int trunk[] = {B_PELVIS, B_LUMBAR, B_CHEST, B_HEAD};
    if (g_body_tune & 1) for (int a : arm) for (int l : leg) ex(a, l);
    if (g_body_tune & 2) for (int a : arm) for (int t : trunk) ex(a, t);
    if (g_body_tune & 4) ex(B_THIGH_L, B_THIGH_R), ex(B_SHANK_L, B_SHANK_R), ex(B_THIGH_L, B_SHANK_R), ex(B_THIGH_R, B_SHANK_L);
  }
}

}  // namespace emil
