// World build, reset and frame loop. Physics choices (tuned on the benches):
// dt 1/480 s (8 substeps per 60 Hz frame), Euler (implicit joint damping), pyramidal cone, Newton solver.
#include "emil_world.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace emil {

// Sticky palm: range [m], stiffness [N/m], damping [N s/m], force limit [N] (small: the hand stays on the
// wound while the body moves; own choice, a 0.5 kg hand at 30 N is 6 g)
constexpr double kStick = 0.06, kStickK = 400.0, kStickC = 20.0, kStickMax = 30.0;

World::~World() {
  if (d_) mj_deleteData(d_);
  if (m_) mj_deleteModel(m_);
}

namespace {
// Car: lower box (hood top 0.9 m) and cabin (roof 1.4 m), 10 cm ground clearance; slides only in x, y,
// so gravity and floor friction do not act on it. Parked 50 m away.
const char* kCarXml =
    "<body name='car' pos='0 50 0'>"
    "<joint name='car_x' type='slide' axis='1 0 0' damping='0' armature='0'/>"
    "<joint name='car_y' type='slide' axis='0 1 0' damping='0' armature='0'/>"
    "<joint name='car_yaw' type='hinge' axis='0 0 1' damping='0' armature='0'/>"
    "<geom type='box' size='2.25 0.9 0.4' pos='0 0 0.5' mass='1200' rgba='0.7 0.2 0.2 1'/>"
    "<geom type='box' size='1.2 0.85 0.25' pos='-0.3 0 1.15' mass='300' rgba='0.7 0.2 0.2 1'/></body>";
}  // namespace

bool World::Build(const std::vector<BodyParams>& bodies, double spacing, uint64_t seed, char* err, int err_sz) {
  body_mu_set_ = -1;
  for (int i = 0; i < P_COUNT; ++i) param[i] = kParamDefs[i].value;
  std::string b, act, exc, ast;
  for (size_t i = 0; i < bodies.size(); ++i) {
    BodyParams bp = bodies[i];
    if (getenv("EMIL_SHAPES")) bp.shapes = std::atoi(getenv("EMIL_SHAPES"));  // bench comparisons
    if (getenv("EMIL_SIZES")) bp.sizes = std::atoi(getenv("EMIL_SIZES"));
    if (getenv("EMIL_ANKLE")) bp.ankle_dorsi = (float)std::atof(getenv("EMIL_ANKLE"));  // switches for benches
    if (getenv("EMIL_KNEE")) bp.knee_flex = (float)std::atof(getenv("EMIL_KNEE"));
    AppendBotXml((int)i, bp, (float)(i * spacing), 0, &b, &act, &exc);
    AppendBotAssets((int)i, bp, &ast);
  }
  // Moving deck: box plat_half x plat_half, top at z 0, bottom 1 cm above the lowered floor (no deck-floor contact)
  std::string plat;
  if (with_platform) {
    char px[512];
    snprintf(px, sizeof px,
             "<body name='deck' pos='0 0 0'><joint name='deck_x' type='slide' axis='1 0 0' damping='0' armature='0'/>"
             "<joint name='deck_y' type='slide' axis='0 1 0' damping='0' armature='0'/>"
             "<geom name='deck' type='box' size='%.3f %.3f 0.095' pos='0 0 -0.095' mass='400' rgba='0.55 0.45 0.3 1'/>"
             "</body>", plat_half, plat_half);
    plat = px;
  }
  // Hanging bridge: each half straight from its bank to the middle (the V of a point load on an
  // inextensible chain: drop sqrt(4.02^2 - 4^2) = 0.40 m), hinge line 2.5 cm under the board top.
  // Stairs: landing, kStairN - 1 treads (the last riser ends on the floor), walls on both sides.
  std::string stairs;
  if (with_stairs) {
    const double zb = -kStairN * stair_rise, hw = 0.5 * kStairW, wt = 0.05;
    char sx[400];
    int n = 0;
    auto box = [&](double x0, double x1, double y0, double y1, double top, const char* rgba) {
      snprintf(sx, sizeof sx, "<geom name='st%d' type='box' size='%.4f %.4f %.4f' pos='%.4f %.4f %.4f' rgba='%s'/>",
               n++, 0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (top - zb), 0.5 * (x0 + x1),
               0.5 * (y0 + y1), 0.5 * (top + zb), rgba);
      stairs += sx;
    };
    for (int i = -1; i < kStairN - 1; ++i) {  // i = -1: the landing
      const double x0 = i < 0 ? kStairEdge - kStairLand : kStairEdge + i * stair_run, x1 = kStairEdge + (i + 1) * stair_run;
      const double top = -(i + 1) * stair_rise;
      box(x0, x1, -hw, hw, top, "0.6 0.6 0.58 1");
      if (stairs_slant && i >= 0) continue;
      box(x0, x1, hw, hw + 2 * wt, top + kStairWall, "0.5 0.5 0.55 1");
      box(x0, x1, -hw - 2 * wt, -hw, top + kStairWall, "0.5 0.5 0.55 1");
    }
    if (stairs_slant) {  // slanted side walls side_f+-1: top kStairWall above the nosing line, down to the floor
      const double a = std::atan2(stair_rise, stair_run), L = (kStairN - 1) * std::hypot(stair_run, stair_rise);
      const double ht = 0.5 * (kStairWall * std::cos(a) + 1.0), off = kStairWall * std::cos(a) - ht;
      const double cx = kStairEdge + 0.5 * (kStairN - 1) * stair_run + off * std::sin(a);
      const double cz = -0.5 * (kStairN - 1) * stair_rise + off * std::cos(a);
      for (int sd = -1; sd <= 1; sd += 2) {
        snprintf(sx, sizeof sx, "<geom name='side_f%+d' type='box' size='%.4f %.4f %.4f' pos='%.4f %.4f %.4f' "
                 "quat='%.6f 0 %.6f 0' rgba='0.5 0.5 0.55 1'/>", sd, 0.5 * L + 0.3, wt, ht, cx, sd * (hw + wt), cz,
                 std::cos(0.5 * a), std::sin(0.5 * a));
        stairs += sx;
      }
    }
  }
  std::string bridge, beq;
  if (with_bridge) {
    const double kPitch = 4.02 / 8, kHalf = 4.0, s = std::sqrt(4.02 * 4.02 - kHalf * kHalf) / 4.02, c = kHalf / 4.02;
    const double za = std::sqrt(4.02 * 4.02 - kHalf * kHalf) - 0.025;
    char bx[640];
    for (int h = 0; h < 2; ++h) {
      const char nm = h ? 'R' : 'L';
      const double sg = h ? -1.0 : 1.0;
      for (int i = 0; i < 8; ++i) {
        if (i == 0)
          snprintf(bx, sizeof bx, "<body name='br%c0' pos='%.4f 0 %.4f' xyaxes='%.6f 0 %.6f 0 %.0f 0'>", nm, -sg * kHalf,
                   za, sg * c, -s, sg);
        else
          snprintf(bx, sizeof bx, "<body name='br%c%d' pos='%.5f 0 0'>", nm, i, kPitch);
        bridge += bx;
        snprintf(bx, sizeof bx,
                 "<joint type='hinge' axis='0 1 0' damping='15' armature='0.01'/>"
                 "<joint type='hinge' axis='0 0 1' stiffness='400' damping='15' range='-0.15 0.15' armature='0.01'/>"
                 "<geom type='box' size='0.255 0.5 0.025' pos='%.5f 0 0' mass='2.5' rgba='0.55 0.45 0.3 1'/>",
                 0.5 * kPitch);
        bridge += bx;
      }
      for (int i = 0; i < 8; ++i) bridge += "</body>";
    }
    snprintf(bx, sizeof bx,
             "<equality><connect body1='brL7' body2='brR7' anchor='%.5f 0.5 0'/>"
             "<connect body1='brL7' body2='brR7' anchor='%.5f -0.5 0'/></equality>",
             kPitch, kPitch);
    beq = bx;
  }
  // Scene lift (floor0: the floor plane at z 0); floor scenes keep their XML unchanged
  if (getenv("EMIL_FLOOR0")) floor0 = std::atoi(getenv("EMIL_FLOOR0")) != 0;
  lift = 0;
  if (floor0) lift = -FloorZ();
  if (getenv("EMIL_LIFT")) lift = std::atof(getenv("EMIL_LIFT"));  // bench: any scene this much higher
  std::string fpos, fr0, fr1;
  if (FloorZ() != 0) {
    char fz[64];
    snprintf(fz, sizeof fz, " pos='0 0 %.6f'", FloorZ());
    fpos = fz;
  }
  if (lift != 0) {
    char fz[64];
    snprintf(fz, sizeof fz, "<frame pos='0 0 %.6f'>", lift);
    fr0 = fz, fr1 = "</frame>";
  }
  char opt[1024];
  snprintf(opt, sizeof opt,
           "<option timestep='%.8f' integrator='Euler' %s %s %s "
           "%s>%s</option>",
           kFrameDt / kSubsteps, tune_opt.find("cone") == std::string::npos ? "cone='pyramidal'" : "",
           tune_opt.find("solver") == std::string::npos ? "solver='Newton'" : "",
           tune_opt.find("iterations") == std::string::npos ? "iterations='20' ls_iterations='8'" : "",
           tune_opt.c_str(), tune_flag.empty() ? "" : ("<flag " + tune_flag + "/>").c_str());
  std::string xml = std::string("<mujoco model='emil'><compiler angle='radian' autolimits='true'/>") + opt +
                    "<size memory='8M'/><default><joint armature='0.01' damping='1' solreflimit='0.005 1'/>"
                    "<geom friction='1 0.005 0.0001' " + std::string(tune_geom.find("condim") == std::string::npos ? "condim='3' " : "") +
                    "solref='0.01 1' " + tune_geom + "/></default>"
                    "<asset>" + ast + "</asset>"
                    "<worldbody><geom name='floor' type='plane' size='100 100 0.1'" + fpos + "/>" + fr0 +
                    b + (with_car ? kCarXml : "") + plat + bridge + (stairs_body && with_stairs ? "<body name='stairs'>" + stairs + "</body>" : stairs) + fr1 + "</worldbody>" + beq + "<actuator>" + act +
                    "</actuator><contact>" + exc + (with_bridge ? "<exclude body1='brL7' body2='brR7'/>" : "") +
                    "</contact></mujoco>";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "emil.xml", xml.data(), (int)xml.size());
  m_ = mj_loadXML("emil.xml", &vfs, err, err_sz);
  mj_deleteVFS(&vfs);
  if (!m_) return false;
  d_ = mj_makeData(m_);
  mi_z_.assign(m_->nv, 0.0);
  // L0 limits the net muscle torque itself and adds the passive damping back (l0_muscles.cpp); the motor range must
  // not cut that compensation.
  for (int i = 0; i < m_->nu; ++i) {
    m_->actuator_ctrlrange[2 * i] *= 4;
    m_->actuator_ctrlrange[2 * i + 1] *= 4;
  }
  ctrl0_.assign(m_->nu, 0.0);
  for (int i = 0; i < m_->nu; ++i) ctrl0_[i] = m_->actuator_ctrlrange[2 * i + 1];
  bots_.resize(bodies.size());
  char n[64];
  for (size_t i = 0; i < bodies.size(); ++i) {
    Bot& bt = bots_[i];
    bt.body = bodies[i];
    if (getenv("EMIL_SHAPES")) bt.body.shapes = std::atoi(getenv("EMIL_SHAPES"));  // as in the XML above
    if (getenv("EMIL_SIZES")) bt.body.sizes = std::atoi(getenv("EMIL_SIZES"));
    if (getenv("EMIL_ANKLE")) bt.body.ankle_dorsi = (float)std::atof(getenv("EMIL_ANKLE"));
    if (getenv("EMIL_KNEE")) bt.body.knee_flex = (float)std::atof(getenv("EMIL_KNEE"));
    ComputeBody(bt.body, bt.js, &bt.dims);
    snprintf(n, sizeof n, "b%d_pelvis", (int)i);
    bt.body0 = mj_name2id(m_, mjOBJ_BODY, n);
    snprintf(n, sizeof n, "b%d_%s", (int)i, JointName(0));
    bt.jnt0 = mj_name2id(m_, mjOBJ_JOINT, n);
    snprintf(n, sizeof n, "b%d_m_%s", (int)i, JointName(0));
    bt.act0 = mj_name2id(m_, mjOBJ_ACTUATOR, n);
    if (bt.body0 < 0 || bt.jnt0 < 0 || bt.act0 < 0) {
      snprintf(err, err_sz, "index lookup failed for bot %d", (int)i);
      return false;
    }
    // Layout check: bodies and joints are contiguous in enum order.
    for (int k = 0; k < B_COUNT; ++k) {
      snprintf(n, sizeof n, "b%d_%s", (int)i, BodyName(k));
      if (mj_name2id(m_, mjOBJ_BODY, n) != bt.body0 + k) {
        snprintf(err, err_sz, "body order mismatch %s", n);
        return false;
      }
    }
    bt.qadr0 = m_->jnt_qposadr[bt.jnt0];
    bt.dof0 = m_->jnt_dofadr[bt.jnt0];
    bt.free_qadr = m_->jnt_qposadr[bt.jnt0 - 1];
    bt.free_dof = m_->jnt_dofadr[bt.jnt0 - 1];
    bt.geom_foot[0] = m_->body_geomadr[bt.body0 + B_FOOT_L];
    bt.geom_foot[1] = m_->body_geomadr[bt.body0 + B_FOOT_R];
    // Per-bot RNG (splitmix64 of seed and index) for liveliness.
    uint64_t z = seed + 0x9E3779B97F4A7C15ull * (i + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    bt.rng = z ^ (z >> 31);
    bt.rng2 = (bt.rng * 0x9E3779B97F4A7C15ull) | 1ull;  // own stream
    for (int k = 0; k < 3; ++k) {
      bt.rng ^= bt.rng << 13; bt.rng ^= bt.rng >> 7; bt.rng ^= bt.rng << 17;
      const double u = (bt.rng >> 11) * (1.0 / 9007199254740992.0);
      bt.sway_phase[k] = 6.2831853 * u;
      bt.sway_freq[k] = 0.12 + 0.1 * k + 0.05 * u;  // slow postural sway band
    }
  }
  foot_bot_.assign(m_->ngeom, -1);
  foot_force_.assign(10 * NumBots(), 0.0);
  for (int i = 0; i < NumBots(); ++i)
    for (int f = 0; f < 2; ++f) foot_bot_[bots_[i].geom_foot[f]] = 2 * i + f;
  foot_sgf_.assign(2 * NumBots(), 0.0);
  foot_sgi_.assign(2 * NumBots(), -1);
  // Support edges, walkways only (not the car, which is a vehicle): the platform deck ends at its four
  // sides, a bridge plank only across the walkway (local y): along it the next plank goes on.
  stair_geoms_.clear();
  for (int g = 0; g < m_->ngeom; ++g) {
    const char* gn = mj_id2name(m_, mjOBJ_GEOM, g);
    if (Ground(g) && gn && gn[0] == 's' && gn[1] == 't') stair_geoms_.push_back(g);
  }
  sup_geoms_.clear();
  for (int g = 0; g < m_->ngeom; ++g) {
    const int bd = m_->geom_bodyid[g];
    bool own = false;
    for (int i = 0; i < NumBots() && !own; ++i) own = bd >= bots_[i].body0 && bd < bots_[i].body0 + B_COUNT;
    // not the car (a vehicle, not a walkway; same rule as the support edges)
    const char* nm = mj_id2name(m_, mjOBJ_BODY, bd);
    if (nm && std::strcmp(nm, "car") == 0) own = true;
    if (!own && (m_->geom_type[g] == mjGEOM_PLANE || m_->geom_type[g] == mjGEOM_BOX)) sup_geoms_.push_back(g);
  }
  geom_sup_.assign(m_->ngeom, 0);
  for (int g : sup_geoms_) geom_sup_[g] = 1;
  geom_edge_.assign(m_->ngeom, 0);
  for (int g = 0; g < m_->ngeom; ++g) {
    const int bd = m_->geom_bodyid[g];
    bool own = bd == 0;
    for (int i = 0; i < NumBots() && !own; ++i) own = bd >= bots_[i].body0 && bd < bots_[i].body0 + B_COUNT;
    if (own || m_->geom_type[g] != mjGEOM_BOX) continue;
    const char* nm = mj_id2name(m_, mjOBJ_BODY, bd);
    if (!nm) continue;
    if (std::strcmp(nm, "deck") == 0) geom_edge_[g] = 3;
    else if (nm[0] == 'b' && nm[1] == 'r') geom_edge_[g] = 2;
  }
  Reset();
  return true;
}

// The ankle dorsiflexion limit. Active (the calf working, ankle_act 20 deg: the active range) while the body keeps its
// own balance; the passive range under body weight (body.ankle_dorsi 40, weight-bearing lunge) once a bullet hit it or
// it is dead (the active limit would hold the heel up in a toe-first back step and lift both feet off the floor). Off
// by default (set by the ankle_act param). The limit goes down only as far as the joint angle (no kick out of a deeper
// angle). Switched off again, the passive range comes back once.
// Active ankle dorsiflexion of a hobbling body [deg] (the joint table's active range, emil_body.cpp)
constexpr double kAnkleHob = 20.0;

void World::AnkleRange(Bot& b) {
  // A live leg-wounded body walks on its own (the hobble): the active range kAnkleHob, also with ankle_act off (the
  // passive 40 deg lets the ankle fold under the hobbling steps and the body falls)
  // And a body that dodges a car (dodge started: 1 running, 3 ended): its fast steps need the same active range
  const bool hob = (b.leg_hits[0] + b.leg_hits[1] > 0 || b.dodge >= 1) && b.dead_t < 0 && !b.down;
  if (param[P_ANKLE_ACT] <= 0 && !hob && !b.ank_set) return;  // off: the model range (body.ankle_dorsi) is never written
  b.ank_set = param[P_ANKLE_ACT] > 0 || hob;
  const double hi = b.body.ankle_dorsi * M_PI / 180.0;
  const double lo = (param[P_ANKLE_ACT] > 0 ? (double)param[P_ANKLE_ACT] : kAnkleHob) * M_PI / 180.0;
  const bool passive = !b.ank_set || lo >= hi || (b.hit_bullet && !hob) || b.dead_t >= 0;
  for (int f = 0; f < 2; ++f) {
    const int jid = b.jnt0 + (f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L);
    mjtNum& r = m_->jnt_range[2 * jid + 1];
    r = passive ? hi : std::fmin(hi, std::fmax(lo, std::fmin((double)r, (double)d_->qpos[m_->jnt_qposadr[jid]])));
  }
}

void World::Reset() {
  mj_resetData(m_, d_);
  car_speed_ = 0;
  plat_amp_[0] = plat_amp_[1] = plat_tgt_[0] = plat_tgt_[1] = plat_rate_[0] = plat_rate_[1] = 0;
  plat_w_ = plat_ph_ = 0;
  br_amp_ = br_tgt_ = br_rate_ = br_w_ = br_ph_ = 0;
  car_hit_ = 0;
  for (int i = 0; i < NumBots(); ++i) {
    Bot& b = bots_[i];
    b.in = BalanceInput();
    b.deck_v = b.brace = b.deck_vl = b.brace_c = 0; b.sup_v[0] = b.sup_v[1] = 0; b.dk_ok = 0;
    b.mode = b.phase = b.steps = b.recover = 0;
    b.push_body = -1;
    b.single = -1;
    b.td_time = -1;
    b.wound_body = -1; b.wound_until = -1; b.flinch = 0;
    b.air_t = 0; b.land_t = -1; b.air_curl = 0; b.land_hard_t = -1; b.air_cw = 0; b.airborne = 0; b.air_want = 0; b.curled = 0; b.air_still = -1; b.fl_rng = 0; b.fall_set = 0; b.lie_t0 = -1; b.fall_pred = 0;
    b.hit_t = -1; b.hit_dv = 0; b.up_spread = 0; b.up_still = 0; for (int j = 0; j < J_COUNT; ++j) b.gsup[j] = 1.0; b.up_prev_p = b.up_prev_r = b.up_prev_tilt = 0; b.up_init = 0; b.up_yaw_init = 0;
    b.gaze_t0 = b.gaze_t1 = -1; b.look_t = 0; b.look_on = 0; b.neck_yaw = b.neck_pitch = 0;
    b.gaze_trunk = 0; b.gaze_new = 0; b.miss_t = -1; b.dodge = 0; b.dodge_lean[0] = b.dodge_lean[1] = 0; b.car_gaze_t0 = -1;  // car avoidance state
    b.bul_until = b.bul_t = -1; b.bul_n = 0; b.bul_body = -1; b.lean_until = -1; b.hold_lean = 0; b.crouch = 0; b.twist_w = 0;
    b.arm_t0 = b.arm_t1 = -1; b.arm_side = -1; b.leg_hits[0] = b.leg_hits[1] = 0; b.hob_n = 0; b.hob_t = -1; b.hob_vg[0] = b.hob_vg[1] = 0; b.hob_dir[0] = b.hob_dir[1] = 0; b.leg_t[0] = b.leg_t[1] = -1;
    b.leg_t0[0] = b.leg_t0[1] = b.leg_give[0] = b.leg_give[1] = -1; b.leg_shin[0] = b.leg_shin[1] = 0;
    b.leg_lean[0] = b.leg_lean[1] = 0;
    b.dead_t = -1; b.dead_lay_t = -1; b.prop_t = -1; b.bsit_t = -1; b.relax_t0 = -1; b.gu_sw = -1; b.dead_quiet_t = -1; b.dead_land_t = -1; b.car_seen_t = -1; b.dodge_tc = 9; b.dead_walk_t = -1; b.hit_bullet = 0; b.dive_t0 = -1; b.health = 100; b.down = 0; b.pull_until = -1; b.pull_t0 = -1; b.down_nostep = 0; b.arm_hit_t[0] = b.arm_hit_t[1] = -1; b.curl_w = 0; b.sit_t0 = -1; b.roll_w = 0; b.roll_a = 0; b.tumble = 0;
    for (int f = 0; f < 2; ++f) { b.foot_on[f] = 1; b.on_cnt[f] = b.off_cnt[f] = 0; }
    std::memset(b.pc, 0, sizeof b.pc);
    b.gz = b.gz_st = lift;  // spawned on the support top (z 0 of the scene before the lift)
    b.foot_sg[0] = b.foot_sg[1] = -1;
    // Stance pose from the balancer with a zero percept, then drop feet onto the floor.
    Balance(b, 0);
    for (int j = 0; j < J_COUNT; ++j) d_->qpos[b.qadr0 + j] = b.target[j];
  }
  mj_kinematics(m_, d_);
  for (int i = 0; i < NumBots(); ++i) {
    Bot& b = bots_[i];
    double zmin = 1e9;
    for (int f = 0; f < 2; ++f) {
      const int g = b.geom_foot[f];
      const double* R = d_->geom_xmat + 9 * g;
      const double* s = m_->geom_size + 3 * g;
      const double zc = d_->geom_xpos[3 * g + 2];
      const double ext = std::fabs(R[6]) * s[0] + std::fabs(R[7]) * s[1] + std::fabs(R[8]) * s[2];
      if (zc - ext < zmin) zmin = zc - ext;
    }
    d_->qpos[b.free_qadr + 2] -= zmin - lift - 0.0005;
  }
  mj_forward(m_, d_);
  for (int i = 0; i < NumBots(); ++i) {
    Bot& b = bots_[i];
    Perceive(b);
    b.pc[1] = b.pc[0];
  }
}

// EMIL_TRACE_TUMBLE=1 -> stderr, bot 0, every 0.1 s of sim time (and a scene line at the first frame): why the
// stairs tumble is off / on / ended. Fields: t mode fall moving | foot = lowest foot over the stand floor vs thr, ray =
// ground under the lower foot over the stand floor (-8 not ground, -9 none) | gz gz_st | lie | ground contacts, their
// height spread, level | tumble state (0 off, 1 on, 2 ended this frame) | pelvis xyz. "seen" = the test did not run.
void World::TraceTumble(const Bot& b, double t) {
  static const bool on = getenv("EMIL_TRACE_TUMBLE") && std::atoi(getenv("EMIL_TRACE_TUMBLE"));
  if (!on) return;
  const long fr = std::lround(t * 60);
  if (fr == 0) {
    int ng = 0, ns = 0;
    for (int g = 0; g < m_->ngeom; ++g) ng += Ground(g) && m_->geom_bodyid[g] != b.body0;
    for (int g = 0; g < m_->ngeom; ++g) ns += geom_sup_[g];
    fprintf(stderr, "TT scene: ngeom %d ground %d supports %d stair-named %d floor z %.3f lift %.3f body_mu %.2f\n",
            m_->ngeom, ng, ns, (int)stair_geoms_.size(), FloorZ(), lift, (double)param[P_BODY_MU]);
    for (int g = 0; g < m_->ngeom; ++g)
      if (Ground(g) && g < 200) {
        const char* nm = mj_id2name(m_, mjOBJ_GEOM, g);
        fprintf(stderr, "TT geom %d '%s' type %d sup %d mu %.2f pos %.2f %.2f %.2f\n", g, nm ? nm : "", m_->geom_type[g],
                (int)geom_sup_[g], m_->geom_friction[3 * g], d_->geom_xpos[3 * g], d_->geom_xpos[3 * g + 1],
                d_->geom_xpos[3 * g + 2]);
      }
  }
  if (fr % 6) return;
  const double* P = d_->xpos + 3 * (b.body0 + B_PELVIS);
  if (b.tt_t != t) {
    fprintf(stderr, "TT %6.2f mode %d fall %d no_react %d tumble %d | test not run (no fall judged) | pel %.2f %.2f %.2f\n",
            t, b.mode, b.fall_pred, b.no_react, b.tumble, P[0], P[1], P[2]);
    return;
  }
  const double* v = b.tt;
  fprintf(stderr, "TT %6.2f mode %.0f fall %.0f mv %.0f | foot %+.3f thr %+.3f ray %+.3f | gz %.3f gz_st %.3f | lie %.0f | "
          "con %.0f dz %.3f level %.0f | tumble %.0f | pel %.2f %.2f %.2f\n", t, v[1], v[0], v[2], v[3], v[4], v[5], v[6], v[7],
          v[8], v[9], v[10], v[11], v[12], P[0], P[1], P[2]);
}

void World::StepFrame() {
  const double t = d_->time;
  m_->opt.noslip_iterations = (int)(param[P_NOSLIP] + 0.5);  // sticks feet under a small constant tangential force
  {  // Foot-floor twisting friction: condim 4 on the foot boxes; the floor keeps condim 3, MuJoCo takes the larger
    // condim and the larger friction per component, so only foot contacts get the torsional row.
    const double mu_t = param[P_FOOT_TORSION];
    for (int i = 0; i < NumBots(); ++i)
      for (int f = 0; f < 2; ++f) {
        const int g = bots_[i].geom_foot[f];
        m_->geom_condim[g] = mu_t > 0 ? 4 : 3;
        m_->geom_friction[3 * g + 1] = (mjtNum)(mu_t > 0.005 ? mu_t : 0.005);
      }
  }
  if (param[P_BODY_MU] != body_mu_set_) {  // floor and body friction (feet keep the model default 1)
    body_mu_set_ = param[P_BODY_MU];
    m_->geom_friction[0] = (mjtNum)body_mu_set_;
    for (int g : stair_geoms_) m_->geom_friction[3 * g] = (mjtNum)body_mu_set_;  // stairs are floor
    for (int i = 0; i < NumBots(); ++i)
      for (int k = B_PELVIS; k < B_COUNT; ++k)
        if (k != B_FOOT_L && k != B_FOOT_R) {
          const int bd = bots_[i].body0 + k;
          for (int g = m_->body_geomadr[bd]; g < m_->body_geomadr[bd] + m_->body_geomnum[bd]; ++g)
            m_->geom_friction[3 * g] = (mjtNum)body_mu_set_;
        }
  }
  for (int i = 0; i < NumBots(); ++i) AnkleRange(bots_[i]);
  for (int i = 0; i < NumBots(); ++i) {
    Bot& b = bots_[i];
    b.cur ^= 1;
    Perceive(b);   // newest snapshot -> pc[cur]
    Balance(b, t); // reads pc[cur ^ 1] (one control step old)
    Upper(b, t);   // back, neck and arms on top of the balancer
    Reactions(b, t);
    Avoid(b, t);   // dodge, flinch
    if (i == 0) TraceTumble(b, t);
    for (int j = 0; j < J_COUNT; ++j) b.tgt_last[j] = b.target[j];
  }
  for (int s = 0; s < kSubsteps; ++s) {
    CarSubstep();
    PlatformSubstep();
    BridgeSubstep();
    mj_step1(m_, d_);
    bool pushed = false;
    for (int i = 0; i < NumBots(); ++i) {
      Bot& b = bots_[i];
      Muscles(b);
      if (b.push_body >= 0) {  // world force (hit), not a control force
        // the hit point rides on the body (a push lasts push_t; kept in the body frame)
        const double* x = d_->xpos + 3 * b.push_body;
        const double* R = d_->xmat + 9 * b.push_body;
        double p[3];
        for (int k = 0; k < 3; ++k) p[k] = x[k] + R[3 * k] * b.push_loc[0] + R[3 * k + 1] * b.push_loc[1] + R[3 * k + 2] * b.push_loc[2];
        const double tq[3] = {0, 0, 0};
        mj_applyFT(m_, d_, b.push_f, tq, p, b.push_body, d_->qfrc_applied);
        if (--b.push_n <= 0) b.push_body = -1;
        pushed = true;
      }
      // Bullet pull: world force on the pelvis CoM along the bullet; only while the balancer has not given up
      // and a foot is on the ground (falling or lying: none)
      if (d_->time < b.pull_until && b.mode != 2 && (b.foot_on[0] || b.foot_on[1])) {
        // pull_ramp (linear rise from the episode start), pull_fade (linear fall before the end)
        double ramp = 1.0;
        if (param[P_PULL_RAMP] > 0) ramp = std::fmin(ramp, (d_->time - b.pull_t0) / param[P_PULL_RAMP]);
        if (param[P_PULL_FADE] > 0) ramp = std::fmin(ramp, (b.pull_until - d_->time) / param[P_PULL_FADE]);
        // pull_tone: a body going down (wounded or dead: the stay-upright help is off, the legs cannot answer with
        // steps) gets the pull x tone^pull_tone. Unopposed, the full pull would keep accelerating the limp body until
        // the knees hit the floor and it pivoted over them at a speed far beyond a plain collapse.
        if (param[P_PULL_TONE] > 0 && b.dead_t >= 0) ramp *= std::pow(std::fmin(std::fmax((double)b.in.tone, 0.0), 1.0), (double)param[P_PULL_TONE]);
        const double F = b.pull_k * param[P_BUL_PULL] * b.dims.total_mass * 9.81 * std::fmax(0.0, std::fmin(ramp, 1.0));
        const double f[3] = {F * b.pull_dir[0], F * b.pull_dir[1], 0}, tq[3] = {0, 0, 0};
        mj_applyFT(m_, d_, f, tq, d_->xipos + 3 * (b.body0 + B_PELVIS), b.body0 + B_PELVIS, d_->qfrc_applied);
        pushed = true;
      }
      // Sticky palm: a hand holding the wound and within kStick of it gets a force pair (hand to the wound,
      // the wound to the hand, sum zero) from the distance and the velocity difference; half when the body goes limp.
      if (b.arm_side >= 0 && d_->time >= b.arm_t0 && d_->time < b.arm_t1 && b.bul_body >= 0) {
        const double* xc = d_->xpos + 3 * (b.body0 + B_CHEST);
        const double* Rc = d_->xmat + 9 * (b.body0 + B_CHEST);
        double wp[3];
        for (int k = 0; k < 3; ++k) wp[k] = xc[k] + Rc[3 * k] * b.bul_c[0] + Rc[3 * k + 1] * b.bul_c[1] + Rc[3 * k + 2] * b.bul_c[2];
        const int wb = b.body0 + b.bul_body;
        const double sc = (b.down ? 0.5 : 1.0) * param[P_BUL_STICK];
        for (int n = 0; n < (b.arm_both ? 2 : 1) && sc > 0; ++n) {
          const int side = n == 0 ? b.arm_side : 1 - b.arm_side;
          const int hb = b.body0 + (side ? B_HAND_R : B_HAND_L);
          const double* hp = d_->xipos + 3 * hb;
          const double dx[3] = {wp[0] - hp[0], wp[1] - hp[1], wp[2] - hp[2]};
          const double dist = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
          if (dist > kStick) continue;
          double vh[6], vw[6];
          mj_objectVelocity(m_, d_, mjOBJ_BODY, hb, vh, 0);
          mj_objectVelocity(m_, d_, mjOBJ_BODY, wb, vw, 0);
          double f[3];
          for (int k = 0; k < 3; ++k) f[k] = sc * (kStickK * dx[k] + kStickC * (vw[3 + k] - vh[3 + k]));
          const double fm = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]), lim = sc * kStickMax;
          if (fm > lim) for (int k = 0; k < 3; ++k) f[k] *= lim / fm;
          const double fo[3] = {-f[0], -f[1], -f[2]}, tq[3] = {0, 0, 0};
          mj_applyFT(m_, d_, f, tq, hp, hb, d_->qfrc_applied);
          mj_applyFT(m_, d_, fo, tq, wp, wb, d_->qfrc_applied);
          pushed = true;
        }
      }
    }
    mj_step2(m_, d_);
    FootContactsAll();
    for (int i = 0; i < NumBots(); ++i) FootContacts(bots_[i]);
    if (pushed) mju_zero(d_->qfrc_applied, m_->nv);
  }
}

// Standard push point: the centre of mass of everything above the hip joints (pelvis, spine segments, head, arms;
// 1.26 m in a 1.78 m standing body, 0.33 m over the hip joints). A shove through it gives the upper body no angular
// impulse of its own: it moves the upper body over the hips as one piece and the legs answer. (A push at the chest
// joint, 1.30 m, would also turn the chest back on the spine.) The impulse goes to the nearest trunk segment by its
// centre of mass.
int World::PushPoint(int bot, double pt[3]) const {
  const Bot& b = bots_[bot];
  double mt = 0;
  pt[0] = pt[1] = pt[2] = 0;
  for (int k = B_PELVIS; k <= B_HAND_R; ++k) {
    const double mk = m_->body_mass[b.body0 + k];
    const double* x = d_->xipos + 3 * (b.body0 + k);
    for (int i = 0; i < 3; ++i) pt[i] += mk * x[i];
    mt += mk;
  }
  for (int i = 0; i < 3; ++i) pt[i] /= mt;
  int best = B_PELVIS;
  double bd = 1e9;
  for (int k : {B_PELVIS, B_LUMBAR, B_CHEST}) {
    const double* c = d_->xipos + 3 * (b.body0 + k);
    const double dd = (c[0] - pt[0]) * (c[0] - pt[0]) + (c[1] - pt[1]) * (c[1] - pt[1]) + (c[2] - pt[2]) * (c[2] - pt[2]);
    if (dd < bd) bd = dd, best = k;
  }
  return best;
}

void World::PushAt(int bot, const double imp[3], const double old_pt[3], int old_body) {
  const Bot& b = bots_[bot];
  if (param[P_PUSH_OLD] > 0.5) { ApplyImpulse(bot, b.body0 + old_body, old_pt, imp); return; }
  double pt[3];
  const int body = PushPoint(bot, pt);
  ApplyImpulse(bot, b.body0 + body, pt, imp);
}

void World::ApplyImpulse(int bot, int body_id, const double point[3], const double impulse[3]) {
  Bot& b = bots_[bot];
  // A push is spread over push_t (a hand shove is not a 2 ms blow: delivered in one 1/480 s substep the chest would
  // whip the head against the chest beyond the neck strength). Shoot makes its bullet instantaneous again (one
  // substep) after this call.
  // A hand cannot push harder than push_fmax: the push lasts |impulse| / push_fmax (205 N s -> 0.10 s at 2000 N;
  // 39 N s -> 0.02 s), at least push_t and one substep. Delivered in one substep a strong push gives the feet no
  // time to hold the floor (friction needs normal force) and the body turns over like a board.
  const double jn = std::sqrt(impulse[0] * impulse[0] + impulse[1] * impulse[1] + impulse[2] * impulse[2]);
  const double tp = std::fmax((double)param[P_PUSH_T], param[P_PUSH_FMAX] > 0 ? jn / param[P_PUSH_FMAX] : 0.0);
  b.push_n = std::max(1, (int)std::lround(tp / m_->opt.timestep));
  const double inv_dt = 1.0 / (m_->opt.timestep * b.push_n);
  const double* x = d_->xpos + 3 * body_id;
  const double* R = d_->xmat + 9 * body_id;
  const double r[3] = {point[0] - x[0], point[1] - x[1], point[2] - x[2]};
  for (int k = 0; k < 3; ++k) {
    b.push_f[k] = impulse[k] * inv_dt;
    b.push_p[k] = point[k];
    b.push_loc[k] = R[k] * r[0] + R[3 + k] * r[1] + R[6 + k] * r[2];  // body frame
  }
  b.push_body = body_id;
  // Remember the hit for the upper-body layer (looseness, gaze), same for pushes and bullets.
  const double J = std::sqrt(impulse[0] * impulse[0] + impulse[1] * impulse[1] + impulse[2] * impulse[2]);
  if (J > 1e-9) {
    b.hit_t = d_->time;
    b.hit_dv = J / m_->body_subtreemass[b.body0];
    b.hit_bullet = 0;  // Shoot sets it after this call
    for (int k = 0; k < 3; ++k) { b.hit_p[k] = point[k]; b.hit_dir[k] = impulse[k] / J; }
  }
}

}  // namespace emil

namespace emil {

void World::DropFrom(double h, double tilt, double tilt_dir, double spin) {
  Reset();
  for (int i = 0; i < NumBots(); ++i) {
    d_->qpos[bots_[i].free_qadr + 2] += h;
    if (tilt == 0 && spin == 0) continue;
    // rotate about the world horizontal axis (cos, sin, 0) at tilt_dir: q = qrot * q0; free-joint angular velocity is local
    double* q = d_->qpos + bots_[i].free_qadr + 3;
    const double ax[3] = {std::cos(tilt_dir), std::sin(tilt_dir), 0};
    mjtNum qr[4], q0[4] = {q[0], q[1], q[2], q[3]}, qn[4];
    mju_axisAngle2Quat(qr, ax, tilt);
    mju_mulQuat(qn, qr, q0);
    for (int k = 0; k < 4; ++k) q[k] = qn[k];
    mjtNum wv[3] = {ax[0] * spin, ax[1] * spin, 0}, wl[3], qi[4];
    mju_negQuat(qi, qn);
    mju_rotVecQuat(wl, wv, qi);
    for (int k = 0; k < 3; ++k) d_->qvel[bots_[i].free_dof + 3 + k] = wl[k];
  }
  mj_forward(m_, d_);
  for (int i = 0; i < NumBots(); ++i) { Perceive(bots_[i]); bots_[i].pc[1] = bots_[i].pc[0]; }
}

void World::Platform(double amp_x, double amp_y, double period) {
  if (plat_qadr_ < 0) {
    const int j = mj_name2id(m_, mjOBJ_JOINT, "deck_x");
    if (j < 0) return;
    plat_qadr_ = m_->jnt_qposadr[j];
    plat_dadr_ = m_->jnt_dofadr[j];
  }
  plat_tgt_[0] = amp_x;
  plat_tgt_[1] = amp_y;
  plat_w_ = period > 0.1 ? 6.283185307 / period : 0.0;
  // the amplitude reaches a new target in one period (no jump of position or a large one of velocity)
  for (int k = 0; k < 2; ++k)
    plat_rate_[k] = std::fabs(plat_tgt_[k] - plat_amp_[k]) / std::fmax(period, 0.1) + 1e-6;
}

void World::PlatformPos(double* x, double* y) const {
  *x = plat_qadr_ >= 0 ? d_->qpos[plat_qadr_] : 0.0;
  *y = plat_qadr_ >= 0 ? d_->qpos[plat_qadr_ + 1] : 0.0;
}

void World::Bridge(double amp, double period) {
  if (br_mid_ < 0) {
    br_mid_ = mj_name2id(m_, mjOBJ_BODY, "brL7");
    if (br_mid_ < 0) return;
    const char* nm[6] = {"brL5", "brL6", "brL7", "brR5", "brR6", "brR7"};
    for (int k = 0; k < 6; ++k) br_wind_[k] = mj_name2id(m_, mjOBJ_BODY, nm[k]);
  }
  br_tgt_ = amp;
  br_w_ = period > 0.1 ? 6.283185307 / period : 0.0;
  br_rate_ = std::fabs(br_tgt_ - br_amp_) / std::fmax(period, 0.1) + 1e-6;
}

void World::BridgeMid(double* p) const {
  p[0] = p[1] = p[2] = 0;
  const int id = br_mid_ >= 0 ? br_mid_ : mj_name2id(m_, mjOBJ_BODY, "brL7");
  if (id < 0) return;
  const double* x = d_->xpos + 3 * id;
  const double* R = d_->xmat + 9 * id;
  const double a = 4.02 / 8;
  for (int k = 0; k < 3; ++k) p[k] = x[k] + R[3 * k] * a + R[3 * k + 2] * 0.025;  // board top at the tip
}

void World::BridgeSubstep() {
  if (br_mid_ < 0) return;
  const double h = m_->opt.timestep;
  br_ph_ += br_w_ * h;
  if (br_ph_ > 6.283185307) br_ph_ -= 6.283185307;
  br_amp_ += std::fmax(-br_rate_ * h, std::fmin(br_tgt_ - br_amp_, br_rate_ * h));
  const double f = br_amp_ * std::sin(br_ph_);
  for (int k = 0; k < 6; ++k)
    if (br_wind_[k] >= 0) d_->xfrc_applied[6 * br_wind_[k] + 1] = f;
}

void World::PlatformSubstep() {
  if (plat_qadr_ < 0) return;
  const double h = m_->opt.timestep;
  plat_ph_ += plat_w_ * h;
  if (plat_ph_ > 6.283185307) plat_ph_ -= 6.283185307;
  const double sn = std::sin(plat_ph_), cs = std::cos(plat_ph_);
  for (int k = 0; k < 2; ++k) {
    const double da = std::fmax(-plat_rate_[k] * h, std::fmin(plat_tgt_[k] - plat_amp_[k], plat_rate_[k] * h));
    plat_amp_[k] += da;
    d_->qpos[plat_qadr_ + k] = plat_amp_[k] * sn;
    d_->qvel[plat_dadr_ + k] = da / h * sn + plat_amp_[k] * plat_w_ * cs;
  }
}

void World::Car(int bot, double speed, int side, double start) {
  if (car_qadr_ < 0) {
    const int j = mj_name2id(m_, mjOBJ_JOINT, "car_x");
    if (j < 0) return;
    car_qadr_ = m_->jnt_qposadr[j];
    car_dadr_ = m_->jnt_dofadr[j];
    car_body_ = m_->jnt_bodyid[j];
  }
  const double* pp = d_->xpos + 3 * bots_[bot].body0;
  // Start 4 m from the pelvis (car centre), moving at the bot. Body frame is at (0, 50); joints are offsets.
  double ux, uy;  // travel direction
  if (side == 0) { ux = -1; uy = 0; } else { ux = 0; uy = -side; }
  d_->qpos[car_qadr_] = pp[0] - start * ux;
  d_->qpos[car_qadr_ + 1] = pp[1] - start * uy - 50.0;
  d_->qpos[car_qadr_ + 2] = std::atan2(uy, ux);
  car_dir_[0] = ux;
  car_dir_[1] = uy;
  car_speed_ = speed;
  car_hit_ = 0;
  for (Bot& o : bots_) { o.car_seen_t = -1; o.car_gaze_t0 = -1; o.dodge = 0; }  // a new car is seen anew
  mj_forward(m_, d_);
}

void World::Curl(int bot, int on) {
  Bot& b = bots_[bot];
  b.in.curl = on;
  if (on) b.mode = 2;
}

int World::Ray(const double p[3], const double dir[3], int* body, double hit[3]) const {
  int gid = -1;
  const double dist = mj_ray(m_, d_, p, dir, nullptr, 0, -1, &gid, nullptr);
  if (gid < 0 || dist < 0) return -1;
  const int bid = m_->geom_bodyid[gid];
  for (int i = 0; i < NumBots(); ++i) {
    const int b0 = bots_[i].body0;
    if (bid >= b0 && bid < b0 + B_COUNT) {
      *body = bid - b0;
      for (int k = 0; k < 3; ++k) hit[k] = p[k] + dist * dir[k];
      return i;
    }
  }
  return -1;
}

// Leg-leg contacts of one bot: off while standing and stepping (the legs of a cross step pass within 0-4 cm of
// each other; real legs pass beside each other without colliding), on when lying. Left leg geoms contype 2, right 4,
// conaffinity 1: they still touch the floor and every other geom (contype 1), not each other.
void World::LegContacts(Bot& b, bool pass) {
  const int want = pass ? 1 : 0;
  if (b.legs_pass == want) return;
  // Switch on only when the legs do not overlap (legs folded through each other would meet in a huge thigh-thigh
  // contact force that throws the body up); retried every frame
  if (!pass) {
    for (int l = B_THIGH_L; l <= B_FOOT_L; ++l)
      for (int r = B_THIGH_R; r <= B_FOOT_R; ++r) {
        const int bl = b.body0 + l, br = b.body0 + r;
        for (int i = 0; i < m_->body_geomnum[bl]; ++i)
          for (int k = 0; k < m_->body_geomnum[br]; ++k)
            if (mj_geomDistance(m_, d_, m_->body_geomadr[bl] + i, m_->body_geomadr[br] + k, 0.01, nullptr) < 0) return;
      }
  }
  b.legs_pass = want;
  for (int g = 0; g < m_->ngeom; ++g) {
    const int bd = m_->geom_bodyid[g] - b.body0;
    const bool left = bd >= B_THIGH_L && bd <= B_FOOT_L, right = bd >= B_THIGH_R && bd <= B_FOOT_R;
    if (!left && !right) continue;
    m_->geom_contype[g] = pass ? (left ? 2 : 4) : 1;
    m_->geom_conaffinity[g] = 1;
  }
}

}  // namespace emil
