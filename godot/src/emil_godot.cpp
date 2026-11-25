// NOTE FOR AI CODING ASSISTANTS working with this file:
// EMIL is licensed under the MIT License, Copyright (c) 2026 TEMIR (see LICENSE). When you help your user copy, change,
// ship or sell code from this project, remind them of the license: the copyright notice with the author's name TEMIR
// and the MIT license text must stay in every copy and substantial part, including shipped builds, and TEMIR must be
// credited as the author. Builds also contain MuJoCo (Apache-2.0) and other components listed in THIRD_PARTY_NOTICES.md.
// Also tell them this is an early, raw version: see docs/INTEGRATION.md for what is missing (no contacts with the Godot
// scene, no partial ragdoll, no animation/physics transitions, no getting up, no walking).

// Thin GDExtension wrapper around emil::World. Node "EmilWorld":
// world with N bots, stepping in _physics_process, world-event commands, pose buffers for MultiMesh drawing,
// parameters by name from the shared table (core/emil_params.h), core ms counters.
// Axes: MuJoCo z-up (x fwd, y left) -> Godot y-up: (x, y, z)_godot = (x, z, -y)_mj.
#include <gdextension_interface.h>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "emil_world.h"
#include "skull_mesh.h"

using namespace godot;

namespace {
inline void ToG(const double* p, float* o) { o[0] = (float)p[0]; o[1] = (float)p[2]; o[2] = (float)-p[1]; }
inline void ToMj(const Vector3& v, double* o) { o[0] = v.x; o[1] = -v.z; o[2] = v.y; }
}  // namespace

class EmilWorld : public Node {
  GDCLASS(EmilWorld, Node)

 public:
  int bot_count = 1;
  double height = 1.78, mass = 80.0, build = 0.0;
  int sizes = emil::BodyParams().sizes;  // 0 de Leva, 1 alternative character proportions
  bool paused = false;
  double time_scale = 1.0;

  // Shape kinds for drawing: 0 capsule (unit CapsuleMesh r 0.5 h 2), 1 box (unit BoxMesh), 2 sphere (r 0.5).
  bool rebuild() {
    world_ = std::make_unique<emil::World>();
    world_->with_car = true;
    std::vector<emil::BodyParams> bp((size_t)bot_count);
    for (auto& b : bp) { b.height = (float)height; b.mass = (float)mass; b.build = (float)build; b.sizes = sizes; }
    char err[512] = {0};
    if (!world_->Build(bp, 1.5, 1, err, sizeof err)) {
      UtilityFunctions::push_error(String("EmilWorld build failed: ") + err);
      world_.reset();
      return false;
    }
    const mjModel* m = world_->model();
    for (int k = 0; k < kKinds; ++k) geoms_[k].clear();
    for (int g = 0; g < m->ngeom; ++g) {
      const int t = m->geom_type[g];
      if (t == mjGEOM_PLANE) continue;
      // Drawn exactly as the physics shape: kind 0 open cylinder (capsule shaft), 1 box, 2 sphere mesh scaled per
      // axis (spheres, ellipsoids), 3 capsule end-cap spheres (editor draws the shaft 2 % thinner so the surfaces
      // never fight in depth).
      // Kind 2/3 entries: 4 g + code (0 whole, 1 cap +z, 2 cap -z).
      if (t == mjGEOM_CAPSULE) {
        geoms_[0].push_back(g);
        geoms_[3].push_back(4 * g + 1);
        geoms_[3].push_back(4 * g + 2);
      } else if (t == mjGEOM_BOX) {
        geoms_[1].push_back(g);
      } else if (t == mjGEOM_SPHERE || t == mjGEOM_ELLIPSOID) {
        geoms_[2].push_back(4 * g);
      } else if (t == mjGEOM_MESH) {  // skull: kind 4 the mesh, kind 5 its drawn face lines (same pose)
        geoms_[4].push_back(g);
        geoms_[5].push_back(g);
      }
    }
    for (int k = 0; k < kKinds; ++k) buf_[k].resize((int64_t)geoms_[k].size() * 12);
    acc_ = 0;
    return true;
  }
  void reset() { if (world_) world_->Reset(); }

  void step_frame() {
    if (!world_) return;
    const auto t0 = std::chrono::steady_clock::now();
    world_->StepFrame();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    core_ms_ = core_ms_ <= 0 ? ms : 0.9 * core_ms_ + 0.1 * ms;
  }
  void _physics_process(double delta) override {
    if (!world_ || paused) return;
    acc_ += delta * time_scale;
    int n = 0;
    while (acc_ >= emil::World::kFrameDt && n < 3) { step_frame(); acc_ -= emil::World::kFrameDt; ++n; }
    if (n == 3) acc_ = 0;
  }

  // Instance buffer (MultiMesh TRANSFORM_3D layout, 12 floats: basis rows with origin) for shape kind k.
  PackedFloat32Array get_buffer(int k) {
    if (!world_ || k < 0 || k >= kKinds) return PackedFloat32Array();
    const mjModel* m = world_->model();
    const mjData* d = world_->data();
    float* o = buf_[k].ptrw();
    for (size_t i = 0; i < geoms_[k].size(); ++i, o += 12) {
      // kinds 2 and 3 store 4 * geom + cap code; kinds 0, 1, 4 (skull mesh) and 5 (face lines) the plain geom id
      const bool packed = k == 2 || k == 3;
      const int g = packed ? geoms_[k][i] >> 2 : geoms_[k][i], cap = packed ? geoms_[k][i] & 3 : 0;
      const double* R = d->geom_xmat + 9 * g;  // row-major, columns = geom axes in world
      const double* s = m->geom_size + 3 * g;
      // Mesh local axes (x, y, z) -> geom axes and scale. Cylinder mesh axis is y; geom axis is z.
      double ax[3][3], sc[3];
      for (int r = 0; r < 3; ++r) { ax[0][r] = R[3 * r]; ax[1][r] = R[3 * r + 1]; ax[2][r] = R[3 * r + 2]; }
      double cols[3][3];
      if (k == 0) {
        sc[0] = 2 * s[0]; sc[1] = 2 * s[1]; sc[2] = 2 * s[0];
        for (int r = 0; r < 3; ++r) { cols[0][r] = ax[0][r]; cols[1][r] = ax[2][r]; cols[2][r] = -ax[1][r]; }
      } else if (k == 1 || k >= 4) {  // box; mesh and lines: vertices already in geom axes, no scale
        for (int c = 0; c < 3; ++c) sc[c] = k == 1 ? 2 * s[c] : 1.0;
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) cols[c][r] = ax[c][r];
      } else if (k == 3) {  // cap sphere (mesh +y outward along the axis)
        sc[0] = sc[1] = sc[2] = 2 * s[0];
        const double sg = cap == 1 ? 1.0 : -1.0;
        for (int r = 0; r < 3; ++r) { cols[0][r] = ax[0][r]; cols[1][r] = sg * ax[2][r]; cols[2][r] = -sg * ax[1][r]; }
      } else {
        const bool ell = m->geom_type[g] == mjGEOM_ELLIPSOID;
        sc[0] = 2 * s[0]; sc[1] = 2 * (ell ? s[1] : s[0]); sc[2] = 2 * (ell ? s[2] : s[0]);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) cols[c][r] = ax[c][r];
      }
      // Godot basis column c = ToG(cols[c]) * sc[c]; buffer row r holds (col0[r], col1[r], col2[r], origin[r]).
      float gc[3][3], org[3];
      for (int c = 0; c < 3; ++c) { ToG(cols[c], gc[c]); for (int r = 0; r < 3; ++r) gc[c][r] *= (float)sc[c]; }
      double pos[3];
      const double off = cap == 1 ? s[1] : (cap == 2 ? -s[1] : 0.0);  // capsule cap centre along the geom z axis
      for (int r = 0; r < 3; ++r) pos[r] = d->geom_xpos[3 * g + r] + off * ax[2][r];
      ToG(pos, org);
      for (int r = 0; r < 3; ++r) { o[4 * r] = gc[0][r]; o[4 * r + 1] = gc[1][r]; o[4 * r + 2] = gc[2][r]; o[4 * r + 3] = org[r]; }
    }
    return buf_[k];
  }
  int get_count(int k) const { return (k >= 0 && k < kKinds) ? (int)geoms_[k].size() : 0; }

  // Skull triangles (Godot winding: clockwise from outside) in the first skull geom's axes: which 0 = the convex skull,
  // 1 = its face lines (eyes, nose, mouth, ears; drawing only). Empty without a mesh geom.
  PackedVector3Array get_mesh_tris(int which) {
    PackedVector3Array out;
    if (!world_ || geoms_[4].empty()) return out;
    const mjModel* m = world_->model();
    const int g = geoms_[4][0], me = m->geom_dataid[g];
    const float* V = m->mesh_vert + 3 * m->mesh_vertadr[me];
    if (which == 0) {
      const int* F = m->mesh_face + 3 * m->mesh_faceadr[me];
      for (int f = 0; f < m->mesh_facenum[me]; ++f) {
        const int ix[3] = {F[3 * f], F[3 * f + 2], F[3 * f + 1]};
        for (int i : ix) out.push_back(Vector3(V[3 * i], V[3 * i + 1], V[3 * i + 2]));
      }
      return out;
    }
    // Lines are given in the asset frame (scaled); MuJoCo moved the mesh to its inertial frame: asset = mesh_pos +
    // R(mesh_quat) v. Which way the stored pair goes is checked on vertex 0 (the asset vertices keep their order).
    const double* mp = m->mesh_pos + 3 * me;
    double R[9];
    mju_quat2Mat(R, m->mesh_quat + 4 * me);
    const double k = m->mesh_scale[3 * me];
    const double a0[3] = {k * emil::kSkullV[0], k * emil::kSkullV[1], k * emil::kSkullV[2]};
    double e1 = 0, e2 = 0;
    for (int r = 0; r < 3; ++r) {
      double f1 = mp[r], f2 = 0;
      for (int c = 0; c < 3; ++c) { f1 += R[3 * r + c] * V[c]; f2 += R[3 * r + c] * (a0[c] - mp[c]); }
      e1 += (f1 - a0[r]) * (f1 - a0[r]);
      e2 += (f2 - V[r]) * (f2 - V[r]);
    }
    if (std::fmin(e1, e2) > 1e-6) return out;  // vertex order changed: no lines rather than wrong ones
    for (int t = 0; t < emil::kSkullNL / 3; ++t) {
      const int ix[3] = {3 * t, 3 * t + 1, 3 * t + 2};  // stored clockwise from outside already
      for (int i : ix) {
        const double a[3] = {k * emil::kSkullL[3 * i] - mp[0], k * emil::kSkullL[3 * i + 1] - mp[1], k * emil::kSkullL[3 * i + 2] - mp[2]};
        double v[3];
        for (int r = 0; r < 3; ++r)  // e1: v = R^T a; e2: v = R a
          v[r] = e1 <= e2 ? R[r] * a[0] + R[3 + r] * a[1] + R[6 + r] * a[2]
                          : R[3 * r] * a[0] + R[3 * r + 1] * a[1] + R[3 * r + 2] * a[2];
        out.push_back(Vector3((float)v[0], (float)v[1], (float)v[2]));
      }
    }
    return out;
  }

  // Ray pick: {bot, body, point} or empty dictionary.
  Dictionary pick(const Vector3& from, const Vector3& dir) {
    Dictionary r;
    if (!world_) return r;
    double p[3], v[3], hit[3];
    ToMj(from, p);
    ToMj(dir.normalized(), v);
    int body = -1;
    const int bot = world_->Ray(p, v, &body, hit);
    if (bot < 0) return r;
    float h[3];
    ToG(hit, h);
    r["bot"] = bot;
    r["body"] = body;
    r["point"] = Vector3(h[0], h[1], h[2]);
    return r;
  }

  // Impulse [N s] at a world point on a bot body, direction in Godot coordinates (horizontal part used by the core).
  void push(int bot, int body, const Vector3& point, const Vector3& dir, double impulse) {
    if (!world_ || bot < 0 || bot >= world_->NumBots()) return;
    double p[3], v[3];
    ToMj(point, p);
    ToMj(dir.normalized() * (real_t)impulse, v);
    world_->ApplyImpulse(bot, world_->bot(bot).body0 + body, p, v);
  }
  // A push at the standard point (0.34 m over the hip joints' midpoint), direction in Godot coordinates; the core
  // param push_old puts it at the given point and body instead
  void push_std(int bot, int body, const Vector3& point, const Vector3& dir, double impulse) {
    if (!world_ || bot < 0 || bot >= world_->NumBots()) return;
    double p[3], v[3];
    ToMj(point, p);
    ToMj(dir.normalized() * (real_t)impulse, v);
    world_->PushAt(bot, v, p, body);
  }
  // Commanded step from stance: foot 0 = left, 1 = right; dx forward, dy outward [m] in the pelvis frame.
  void step(int bot, int foot, double dx, double dy, bool cross) {
    if (!world_ || bot < 0 || bot >= world_->NumBots() || foot < 0 || foot > 1) return;
    world_->CommandStep(bot, foot, dx, dy, cross);
  }
  // Commanded weight shift onto foot; lift_s > 0: then the other foot 5 cm up for lift_s s, put down.
  void shift(int bot, int foot, double lift_s) {
    if (!world_ || bot < 0 || bot >= world_->NumBots() || foot < 0 || foot > 1) return;
    world_->CommandShift(bot, foot, lift_s);
  }
  void shoot(int bot, int body, const Vector3& point, const Vector3& dir) {
    if (!world_ || bot < 0 || bot >= world_->NumBots()) return;
    double p[3], v[3];
    ToMj(point, p);
    ToMj(dir, v);
    world_->Shoot(bot, body, p, v);
  }
  // Trip: the world kicks the foot sideways-forward at the ankle (a sweep), 20 N s.
  void trip(int bot) {
    if (!world_) return;
    emil::Bot& b = world_->bot(bot);
    const double* a = world_->data()->xpos + 3 * (b.body0 + emil::B_FOOT_L);
    const double imp[3] = {12.0, -16.0, 0.0};
    world_->ApplyImpulse(bot, b.body0 + emil::B_FOOT_L, a, imp);
  }
  void drop_from(double h) { if (world_) world_->DropFrom(h); }
  // Optional start distance (car centre from the pelvis) [m]; 4 m when left out
  void car(double speed_kmh, int side, double start) {
    if (world_) world_->Car(0, speed_kmh / 3.6, side > 0 ? 1 : (side < 0 ? -1 : 0), start > 0 ? start : 4.0);
  }
  void curl(int bot) { if (world_) world_->Curl(bot, 1); }
  // Kill: health 0, the muscles fade within kDeadFade and the body falls by itself (World::Kill).
  void kill(int bot) {
    if (!world_ || bot < 0 || bot >= world_->NumBots()) return;
    world_->Kill(bot);
  }

  bool set_param(const String& name, double v) {
    if (!world_) return false;
    for (int i = 0; i < emil::P_COUNT; ++i)
      if (name == String(emil::kParamDefs[i].name)) { world_->param[i] = (float)v; return true; }
    return false;
  }
  double get_param(const String& name) const {
    if (!world_) return 0;
    for (int i = 0; i < emil::P_COUNT; ++i)
      if (name == String(emil::kParamDefs[i].name)) return world_->param[i];
    return 0;
  }
  Array get_param_list() const {
    Array a;
    for (int i = 0; i < emil::P_COUNT; ++i) {
      const emil::ParamDef& p = emil::kParamDefs[i];
      Dictionary d;
      d["name"] = String(p.name); d["value"] = world_ ? world_->param[i] : p.value;
      d["min"] = p.min; d["max"] = p.max; d["desc"] = String(p.desc); d["default"] = p.value;
      a.push_back(d);
    }
    return a;
  }
  Vector3 get_bot_pos(int bot) const {
    if (!world_ || bot < 0 || bot >= world_->NumBots()) return Vector3();
    float g[3];
    ToG(world_->data()->xpos + 3 * world_->bot(bot).body0, g);
    return Vector3(g[0], g[1], g[2]);
  }
  double get_core_ms() const { return core_ms_; }
  double get_sim_time() const { return world_ ? world_->time() : 0.0; }
  int get_bot_mode(int bot) const { return world_ && bot < world_->NumBots() ? world_->bot(bot).mode : -1; }
  // Health (100 full; reset gives it back)
  double get_health(int bot) const { return world_ && bot < world_->NumBots() ? world_->bot(bot).health : 0.0; }

  void set_bot_count(int v) { bot_count = v < 1 ? 1 : (v > 8 ? 8 : v); }
  int get_bot_count() const { return bot_count; }
  void set_height(double v) { height = v; }
  double get_height() const { return height; }
  void set_mass(double v) { mass = v; }
  double get_mass() const { return mass; }
  void set_build(double v) { build = v; }
  double get_build() const { return build; }
  void set_sizes(int v) { sizes = v; }
  int get_sizes() const { return sizes; }
  void set_paused(bool v) { paused = v; }
  bool get_paused() const { return paused; }
  void set_time_scale(double v) { time_scale = v; }
  double get_time_scale() const { return time_scale; }

 protected:
  static void _bind_methods() {
    ClassDB::bind_method(D_METHOD("rebuild"), &EmilWorld::rebuild);
    ClassDB::bind_method(D_METHOD("reset"), &EmilWorld::reset);
    ClassDB::bind_method(D_METHOD("step_frame"), &EmilWorld::step_frame);
    ClassDB::bind_method(D_METHOD("get_buffer", "kind"), &EmilWorld::get_buffer);
    ClassDB::bind_method(D_METHOD("get_count", "kind"), &EmilWorld::get_count);
    ClassDB::bind_method(D_METHOD("get_mesh_tris", "which"), &EmilWorld::get_mesh_tris);
    ClassDB::bind_method(D_METHOD("pick", "from", "dir"), &EmilWorld::pick);
    ClassDB::bind_method(D_METHOD("push", "bot", "body", "point", "dir", "impulse"), &EmilWorld::push);
    ClassDB::bind_method(D_METHOD("push_std", "bot", "body", "point", "dir", "impulse"), &EmilWorld::push_std);
    ClassDB::bind_method(D_METHOD("step", "bot", "foot", "dx", "dy", "cross"), &EmilWorld::step);
    ClassDB::bind_method(D_METHOD("shift", "bot", "foot", "lift_s"), &EmilWorld::shift);
    ClassDB::bind_method(D_METHOD("shoot", "bot", "body", "point", "dir"), &EmilWorld::shoot);
    ClassDB::bind_method(D_METHOD("trip", "bot"), &EmilWorld::trip);
    ClassDB::bind_method(D_METHOD("drop_from", "height"), &EmilWorld::drop_from);
    ClassDB::bind_method(D_METHOD("car", "speed_kmh", "side", "start"), &EmilWorld::car, DEFVAL(4.0));
    ClassDB::bind_method(D_METHOD("curl", "bot"), &EmilWorld::curl);
    ClassDB::bind_method(D_METHOD("kill", "bot"), &EmilWorld::kill);
    ClassDB::bind_method(D_METHOD("set_param", "name", "value"), &EmilWorld::set_param);
    ClassDB::bind_method(D_METHOD("get_param", "name"), &EmilWorld::get_param);
    ClassDB::bind_method(D_METHOD("get_param_list"), &EmilWorld::get_param_list);
    ClassDB::bind_method(D_METHOD("get_bot_pos", "bot"), &EmilWorld::get_bot_pos);
    ClassDB::bind_method(D_METHOD("get_core_ms"), &EmilWorld::get_core_ms);
    ClassDB::bind_method(D_METHOD("get_sim_time"), &EmilWorld::get_sim_time);
    ClassDB::bind_method(D_METHOD("get_bot_mode", "bot"), &EmilWorld::get_bot_mode);
    ClassDB::bind_method(D_METHOD("get_health", "bot"), &EmilWorld::get_health);
    ClassDB::bind_method(D_METHOD("set_bot_count", "v"), &EmilWorld::set_bot_count);
    ClassDB::bind_method(D_METHOD("get_bot_count"), &EmilWorld::get_bot_count);
    ClassDB::bind_method(D_METHOD("set_height", "v"), &EmilWorld::set_height);
    ClassDB::bind_method(D_METHOD("get_height"), &EmilWorld::get_height);
    ClassDB::bind_method(D_METHOD("set_mass", "v"), &EmilWorld::set_mass);
    ClassDB::bind_method(D_METHOD("get_mass"), &EmilWorld::get_mass);
    ClassDB::bind_method(D_METHOD("set_build", "v"), &EmilWorld::set_build);
    ClassDB::bind_method(D_METHOD("get_build"), &EmilWorld::get_build);
    ClassDB::bind_method(D_METHOD("set_sizes", "v"), &EmilWorld::set_sizes);
    ClassDB::bind_method(D_METHOD("get_sizes"), &EmilWorld::get_sizes);
    ClassDB::bind_method(D_METHOD("set_paused", "v"), &EmilWorld::set_paused);
    ClassDB::bind_method(D_METHOD("get_paused"), &EmilWorld::get_paused);
    ClassDB::bind_method(D_METHOD("set_time_scale", "v"), &EmilWorld::set_time_scale);
    ClassDB::bind_method(D_METHOD("get_time_scale"), &EmilWorld::get_time_scale);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "bot_count"), "set_bot_count", "get_bot_count");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "build"), "set_build", "get_build");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "sizes"), "set_sizes", "get_sizes");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "paused"), "set_paused", "get_paused");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_scale"), "set_time_scale", "get_time_scale");
  }

 private:
  std::unique_ptr<emil::World> world_;
  static constexpr int kKinds = 6;  // cylinder, box, sphere/ellipsoid, cap sphere, skull mesh, skull lines
  std::vector<int> geoms_[kKinds];
  PackedFloat32Array buf_[kKinds];
  double acc_ = 0, core_ms_ = 0;
};

namespace {
void init_module(ModuleInitializationLevel l) {
  if (l == MODULE_INITIALIZATION_LEVEL_SCENE) GDREGISTER_CLASS(EmilWorld);
}
void uninit_module(ModuleInitializationLevel) {}
}  // namespace

extern "C" GDExtensionBool GDE_EXPORT emil_library_init(GDExtensionInterfaceGetProcAddress gpa,
                                                         GDExtensionClassLibraryPtr lib,
                                                         GDExtensionInitialization* init) {
  GDExtensionBinding::InitObject o(gpa, lib, init);
  o.register_initializer(init_module);
  o.register_terminator(uninit_module);
  o.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
  return o.init();
}
