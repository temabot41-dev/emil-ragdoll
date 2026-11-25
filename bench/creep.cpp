// Foot creep test: a foot box under load on the floor, no balancer. A constant tangential force below the friction
// limit (mu N) should leave the foot at rest; MuJoCo's soft friction lets it creep. Reports the creep speed [mm/s]
// (last 2 s) and the displacement over 5 s [mm], for forces of 0.05-1.25 mu N, for several contact settings.
//   emil_bench creep [variant ...]       variant = name=o:<option attrs>+f:<flags>+g:<default geom attrs>+h:<foot-geom attrs>
// "heavy": the foot carries the body mass (rigid, 40.8 kg for N = 400 N); "light": a 1.2 kg foot pressed by an external
// vertical force (the load comes through a leg), which gives a smaller effective mass at the contact.
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"

void ApplyOverrides(emil::World& w);  // main.cpp
bool HasOverrides();                  // main.cpp

namespace {

struct Var { std::string name, o, f, g, h; };

Var Parse(const std::string& a) {
  Var v;
  const size_t eq = a.find('=');
  v.name = eq == std::string::npos ? a : a.substr(0, eq);
  if (eq == std::string::npos) return v;
  const std::string rest = a.substr(eq + 1);
  size_t pos = 0;
  while (pos < rest.size()) {
    size_t e = rest.find('+', pos);
    if (e == std::string::npos) e = rest.size();
    const std::string part = rest.substr(pos, e - pos);
    if (part.size() > 2 && part[1] == ':') {
      std::string* dst = part[0] == 'o' ? &v.o : part[0] == 'f' ? &v.f : part[0] == 'h' ? &v.h : &v.g;
      *dst += part.substr(2) + " ";
    }
    pos = e + 1;
  }
  return v;
}

bool Has(const std::string& s, const char* key) { return s.find(key) != std::string::npos; }

// Position [m] of the foot along x after the run; fills x(t) samples every 0.5 s.
struct Run { double speed_mm_s, disp_mm; };

Run RunOne(const Var& v, double N, double F, bool heavy, double mu) {
  const double dt = emil::World::kFrameDt / emil::World::kSubsteps;
  const double H = 1.78, fl = 0.152 * H, fw = 0.38 * fl, fh = 0.039 * H;  // foot of the 1.78 m body (emil_body)
  const double mass = heavy ? N / 9.81 : 1.2;
  std::string gd = std::string(Has(v.g, "friction") ? "" : "friction='1 0.005 0.0001' ") + (Has(v.g, "condim") ? "" : "condim='3' ") +
                   (Has(v.g, "solref=") ? "" : "solref='0.01 1' ") + v.g;
  std::string opt = std::string("timestep='") + std::to_string(dt) + "' integrator='Euler' " +
                    (Has(v.o, "cone") ? "" : "cone='pyramidal' ") + (Has(v.o, "solver") ? "" : "solver='Newton' ") +
                    (Has(v.o, "iterations") ? "" : "iterations='20' ls_iterations='8' ") + v.o;
  char body[1024];
  snprintf(body, sizeof body,
           "<body name='foot' pos='0 0 0.2'><joint name='sx' type='slide' axis='1 0 0'/><joint name='sy' type='slide' axis='0 1 0'/>"
           "<joint name='sz' type='slide' axis='0 0 1'/><joint name='yaw' type='hinge' axis='0 0 1'/>"
           "<inertial pos='0 0 0.1' mass='%g' diaginertia='0.5 0.5 0.1'/><geom type='box' size='%g %g %g' %s/></body>", mass,
           0.5 * fl, 0.5 * fw, 0.5 * fh, v.h.c_str());
  const std::string xml = "<mujoco><compiler angle='radian'/><option " + opt + ">" +
                          (v.f.empty() ? "" : "<flag " + v.f + "/>") + "</option><default><joint armature='0.01' damping='0'/><geom " +
                          gd + "/></default><worldbody><geom name='floor' type='plane' size='5 5 0.1'/>" + body +
                          "</worldbody></mujoco>";
  char err[512] = "";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "c.xml", xml.data(), (int)xml.size());
  mjModel* m = mj_loadXML("c.xml", &vfs, err, sizeof err);
  mj_deleteVFS(&vfs);
  if (!m) { fprintf(stderr, "xml error (%s): %s\n", v.name.c_str(), err); return {-1, -1}; }
  mjData* d = mj_makeData(m);
  const int steps_s = (int)std::lround(1.0 / dt);
  const double extra = heavy ? 0.0 : N - mass * 9.81;  // vertical press of the light foot [N]
  auto apply = [&](double fx) {
    d->qfrc_applied[0] = fx;
    d->qfrc_applied[2] = -extra;
  };
  for (int i = 0; i < steps_s; ++i) { apply(0); mj_step(m, d); }  // settle 1 s
  const double x0 = d->qpos[0];
  double x3 = 0, x5 = 0;
  for (int i = 0; i < 5 * steps_s; ++i) {
    apply(F);
    mj_step(m, d);
    if (i == 3 * steps_s - 1) x3 = d->qpos[0];
    if (i == 5 * steps_s - 1) x5 = d->qpos[0];
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  return {(x5 - x3) / 2.0 * 1e3, (x5 - x0) * 1e3};
}

}  // namespace

// Stand of 8 varied bots (as `ab`) for `secs` seconds with the variant settings: per bot pelvis height at the end, time
// of a fall (pelvis < 0.5 m), ankle gap, and the largest foot drift from the start.
int RunCreepStand(const Var& v, double secs) {
  using namespace emil;
  g_body_foot = v.h;
  World w;
  w.tune_opt = v.o; w.tune_flag = v.f; w.tune_geom = v.g;
  char err[512] = "";
  std::vector<BodyParams> bodies(8);
  for (int i = 0; i < 8; ++i) {
    bodies[i].height = 1.70f + 0.03f * (i % 4);
    bodies[i].mass = 70.0f + 5.0f * (i % 5);
    bodies[i].female = (i % 3) == 2;
    bodies[i].build = -0.5f + 0.25f * (i % 5);
  }
  if (!w.Build(bodies, 2.0, 7, err, sizeof err)) { fprintf(stderr, "build: %s\n", err); return 1; }
  ApplyOverrides(w);
  if (HasOverrides()) w.Reset();
  double fall_t[8], gap0[8][2], a0[8][2][3];
  for (int i = 0; i < 8; ++i) fall_t[i] = -1;
  const int frames = (int)(secs * 60);
  for (int f = 0; f <= frames; ++f) {
    w.StepFrame();
    for (int i = 0; i < 8; ++i) {
      Bot& b = w.bot(i);
      if (fall_t[i] < 0 && w.data()->xpos[3 * b.body0 + 2] < 0.5) fall_t[i] = w.data()->time;
      if (f == 30) for (int k = 0; k < 2; ++k) for (int c = 0; c < 3; ++c) a0[i][k][c] = w.data()->xanchor[3 * (b.jnt0 + (k ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)) + c];
    }
    (void)gap0;
    if (f % 30 == 15 && f > 20 && getenv("CREEP_FOOT")) {  // bot 0, left foot: velocity [mm/s] and contact forces [N]
      Bot& b = w.bot(0);
      const mjModel* m = w.model();
      const mjData* d = w.data();
      double fn = 0, ftx = 0, fty = 0;
      for (int c = 0; c < d->ncon; ++c) {
        const mjContact& cn = d->contact[c];
        if (cn.geom1 != b.geom_foot[0] && cn.geom2 != b.geom_foot[0]) continue;
        mjtNum cf[6];
        mj_contactForce(m, d, c, cf);
        fn += cf[0];
        ftx += cf[1] * cn.frame[3] + cf[2] * cn.frame[6];  // tangential force in world x (frame rows: normal, t1, t2)
        fty += cf[1] * cn.frame[4] + cf[2] * cn.frame[7];
      }
      mjtNum vel[6];
      mj_objectVelocity(m, d, mjOBJ_GEOM, b.geom_foot[0], vel, 0);
      printf("  hipabd L q %.3f tgt %.3f ctrl %.1f | ankinv L q %.3f tgt %.3f ctrl %.1f | R hipabd q %.3f tgt %.3f ctrl %.1f\n",
             d->qpos[b.qadr0 + J_HIP_ABD_L], b.target[J_HIP_ABD_L], d->ctrl[b.act0 + J_HIP_ABD_L], d->qpos[b.qadr0 + J_ANKLE_INV_L],
             b.target[J_ANKLE_INV_L], d->ctrl[b.act0 + J_ANKLE_INV_L], d->qpos[b.qadr0 + J_HIP_ABD_R], b.target[J_HIP_ABD_R],
             d->ctrl[b.act0 + J_HIP_ABD_R]);
      {
        const double* hp = d->xanchor + 3 * (b.jnt0 + J_HIP_ABD_L);
        const double* ap = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L);
        const double* cm = d->subtree_com + 3 * b.body0;
        const double Wt = b.dims.total_mass * 9.81;
        printf("  lateral levers (L leg, y): com-hip %.3f, ankle-hip %.3f m; half-weight torque with com lever %.1f N m, with foot lever %.1f N m\n",
               cm[1] - hp[1], ap[1] - hp[1], 0.5 * Wt * (cm[1] - hp[1]), 0.5 * Wt * (ap[1] - hp[1]));
      }
      printf("t=%.2f foot L: v(x,y) %.2f %.2f mm/s, yaw rate %.3f rad/s, N %.0f, Ft(x,y) %.1f %.1f N\n", d->time, vel[3] * 1e3, vel[4] * 1e3,
             vel[2], fn, ftx, fty);
    }
    if (f == 0 && getenv("CREEP_GEOM")) {  // spawn geometry of every bot: hip separation, ankle separation, abduction angles
      for (int i = 0; i < 8; ++i) {
        Bot& b = w.bot(i);
        const mjData* d = w.data();
        const double* hl = d->xanchor + 3 * (b.jnt0 + J_HIP_ABD_L);
        const double* hr = d->xanchor + 3 * (b.jnt0 + J_HIP_ABD_R);
        const double* al = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L);
        const double* ar = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_R);
        printf("bot %d H %.2f: hip sep %.3f (dims %.3f), ankle sep %.3f, hip z %.3f ankle z %.3f, abd q %.4f / %.4f, commanded %.3f\n", i,
               b.body.height, std::hypot(hl[0] - hr[0], hl[1] - hr[1]), 2 * b.dims.hip_half_w, std::hypot(al[0] - ar[0], al[1] - ar[1]),
               hl[2], al[2], d->qpos[b.qadr0 + J_HIP_ABD_L], d->qpos[b.qadr0 + J_HIP_ABD_R],
               w.param[P_STANCE_W] * b.body.height * (1.0 / 1.78));
      }
    }
    if (f % 30 == 0 && getenv("CREEP_TRACE")) {  // ankle gap of every bot twice a second
      printf("t=%.1f gap", w.data()->time);
      for (int i = 0; i < 8; ++i) {
        Bot& b = w.bot(i);
        const double* l = w.data()->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L);
        const double* r = w.data()->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_R);
        printf(" %.3f", std::hypot(l[0] - r[0], l[1] - r[1]));
      }
      printf("\n");
    }
  }
  printf("%-10s secs %.1f: bot  fall_t  pelvis_z  ankle_gap  drift L/R from 0.5 s [mm]  mode\n", v.name.c_str(), secs);
  for (int i = 0; i < 8; ++i) {
    Bot& b = w.bot(i);
    double a[2][3];
    for (int k = 0; k < 2; ++k) for (int c = 0; c < 3; ++c) a[k][c] = w.data()->xanchor[3 * (b.jnt0 + (k ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)) + c];
    const double gap = std::hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]);
    const double dl = std::hypot(a[0][0] - a0[i][0][0], a[0][1] - a0[i][0][1]) * 1e3, dr = std::hypot(a[1][0] - a0[i][1][0], a[1][1] - a0[i][1][1]) * 1e3;
    printf("  %d  %6.2f  %6.3f  %6.3f  %6.1f %6.1f  %d\n", i, fall_t[i], w.data()->xpos[3 * b.body0 + 2], gap, dl, dr, b.mode);
  }
  return 0;
}

// 8 bots, optional commanded step forward by dx [m] (left foot) at 2.0 s; window of 10 s starting 2.5 s after the step
// (or at 3 s without a step): foot drift, ankle gap at the window start and end, mean outward push of the legs on the
// floor [N] and mean foot loads.
int RunCreepWindow(const Var& v, double dx) {
  using namespace emil;
  g_body_foot = v.h;
  World w;
  w.tune_opt = v.o; w.tune_flag = v.f; w.tune_geom = v.g;
  char err[512] = "";
  std::vector<BodyParams> bodies(8);
  for (int i = 0; i < 8; ++i) {
    bodies[i].height = 1.70f + 0.03f * (i % 4);
    bodies[i].mass = 70.0f + 5.0f * (i % 5);
    bodies[i].female = (i % 3) == 2;
    bodies[i].build = -0.5f + 0.25f * (i % 5);
  }
  if (!w.Build(bodies, 2.0, 7, err, sizeof err)) { fprintf(stderr, "build: %s\n", err); return 1; }
  ApplyOverrides(w);
  if (HasOverrides()) w.Reset();  // spawn pose from the overridden parameters
  const double t_step = 2.0, ta = (dx != 0 ? t_step + 2.5 : 3.0), tb = ta + 10.0;
  const int fa = (int)std::lround(ta * 60), fb = (int)std::lround(tb * 60);
  double lat_a[8] = {}, lat_b[8] = {}, anc_a[8][2][3] = {}, anc_b[8][2][3] = {}, out_sum[8] = {}, fn_sum[8][2] = {}, fall_t[8];
  int nwin = 0;
  for (int i = 0; i < 8; ++i) fall_t[i] = -1;
  auto ankle = [&](int i, int k, double* o) {
    Bot& b = w.bot(i);
    for (int c = 0; c < 3; ++c) o[c] = w.data()->xanchor[3 * (b.jnt0 + (k ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L)) + c];
  };
  for (int f = 0; f <= fb; ++f) {
    if (dx != 0 && f == (int)std::lround(t_step * 60)) for (int i = 0; i < 8; ++i) w.CommandStep(i, 0, dx, 0.0, false);
    w.StepFrame();
    const mjModel* m = w.model();
    const mjData* d = w.data();
    for (int i = 0; i < 8; ++i)
      if (fall_t[i] < 0 && d->xpos[3 * w.bot(i).body0 + 2] < 0.5) fall_t[i] = d->time;
    auto lateral = [&](int i) {  // ankle separation across the pelvis heading
      const double* R = d->xmat + 9 * w.bot(i).body0;
      const double hn = std::hypot(R[0], R[3]) + 1e-9;
      double al[3], ar[3];
      ankle(i, 0, al); ankle(i, 1, ar);
      return std::fabs(-(al[0] - ar[0]) * R[3] / hn + (al[1] - ar[1]) * R[0] / hn);
    };
    if (f == fa) for (int i = 0; i < 8; ++i) { for (int k = 0; k < 2; ++k) ankle(i, k, anc_a[i][k]); lat_a[i] = lateral(i); }
    if (f == fb) for (int i = 0; i < 8; ++i) { for (int k = 0; k < 2; ++k) ankle(i, k, anc_b[i][k]); lat_b[i] = lateral(i); }
    if (f > fa && f <= fb) {
      ++nwin;
      for (int i = 0; i < 8; ++i) {
        Bot& b = w.bot(i);
        double al[3], ar[3];
        ankle(i, 0, al); ankle(i, 1, ar);
        double ux = al[0] - ar[0], uy = al[1] - ar[1];
        const double un = std::hypot(ux, uy) + 1e-9;
        ux = ux * (1.0 / un); uy = uy * (1.0 / un);
        double push = 0;
        for (int k = 0; k < 2; ++k) {
          double fn = 0, fx = 0, fy = 0;
          for (int c = 0; c < d->ncon; ++c) {
            const mjContact& cn = d->contact[c];
            if (cn.geom1 != b.geom_foot[k] && cn.geom2 != b.geom_foot[k]) continue;
            mjtNum cf[6];
            mj_contactForce(m, d, c, cf);
            fn += cf[0];
            fx += cf[1] * cn.frame[3] + cf[2] * cn.frame[6];
            fy += cf[1] * cn.frame[4] + cf[2] * cn.frame[7];
          }
          fn_sum[i][k] += fn;
          // force on the foot along (left - right); outward push of the leg = -along (left foot), +along (right foot)
          const double along = fx * ux + fy * uy;
          push += (k == 0 ? -along : along) * 0.5;
        }
        out_sum[i] += push;
      }
    }
  }
  printf("%s dx %.2f window %.1f-%.1f s: bot fall_t lat_gap_a lat_gap_b  drift L/R [mm]  outward push/foot [N]  load L/R [N]\n", v.name.c_str(), dx, ta, tb);
  double gsum = 0, osum = 0, dsum = 0;
  int nok = 0;
  const double inw = 1.0 * (nwin > 0 ? nwin : 1);
  for (int i = 0; i < 8; ++i) {
    const double ga = lat_a[i], gb = lat_b[i];
    const double dl = std::hypot(anc_b[i][0][0] - anc_a[i][0][0], anc_b[i][0][1] - anc_a[i][0][1]) * 1e3;
    const double dr = std::hypot(anc_b[i][1][0] - anc_a[i][1][0], anc_b[i][1][1] - anc_a[i][1][1]) * 1e3;
    printf("  %d %6.2f %.3f %.3f  %6.1f %6.1f  %6.1f  %5.0f %5.0f\n", i, fall_t[i], ga, gb, dl, dr, out_sum[i] / inw, fn_sum[i][0] / inw,
           fn_sum[i][1] / inw);
    if (fall_t[i] < 0) { ++nok; gsum += gb; osum += out_sum[i] / inw; dsum += 0.5 * (dl + dr); }
  }
  const double dn = 1.0 * (nok > 0 ? nok : 1);
  printf("  standing %d of 8: mean lateral gap at window end %.3f m, mean outward push %.1f N per foot, mean foot drift %.1f mm in 10 s\n", nok,
         gsum / dn, osum / dn, dsum / dn);
  return 0;
}

int RunCreepBench(int argc, char** argv) {
  if (argc >= 4 && !std::strcmp(argv[2], "win")) return RunCreepWindow(Parse(argv[3]), argc >= 5 ? std::atof(argv[4]) : 0.0);
  if (argc >= 4 && !std::strcmp(argv[2], "stand")) return RunCreepStand(Parse(argv[3]), argc >= 5 ? std::atof(argv[4]) : 5.0);
  std::vector<Var> vs;
  for (int i = 2; i < argc; ++i) vs.push_back(Parse(argv[i]));
  if (vs.empty()) vs.push_back(Parse("base"));
  const double N = 400.0, mu = 1.0;  // one foot of an 80 kg body in double support; mu = foot-floor friction
  const double fs[] = {20, 50, 100, 200, 500};
  for (bool heavy : {true, false}) {
    printf("%s foot, N = %.0f N, mu N = %.0f N: creep speed [mm/s] / displacement over 5 s [mm]\n%-18s", heavy ? "heavy" : "light", N,
           mu * N, "variant");
    for (double f : fs) printf(" %12.0f N", f);
    printf("\n");
    for (const Var& v : vs) {
      printf("%-18s", v.name.c_str());
      for (double f : fs) {
        const Run r = RunOne(v, N, f, heavy, mu);
        printf(" %7.2f/%-6.1f", r.speed_mm_s, r.disp_mm);
      }
      printf("\n");
    }
  }
  return 0;
}
