// emil_bench: headless scenes, metrics, timing.
// Usage: emil_bench [--poses FILE]
//   Scene "stand": 10 s, prints rest metrics for bot 0, then ms/frame for 1 and 8 bots.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include "emil_world.h"
#include "skull_mesh.h"

using namespace emil;

int RunPushBench(const char* label, int threads);
int RunReactBench(const char* label, int threads);
int RunAvoidBench(int threads);  // car dodge, flinch, gaze
int RunCstat(int nbots, const char* scene);
int RunPairs(int nbots, bool fall);
int RunSpeed(int nbots, const char* scene, int reps, int frames);
int RunCreepBench(int argc, char** argv);
int RunOptBench(int argc, char** argv);
int RunUpper6Bench(int argc, char** argv);  // upper-body measures after pushes
int RunBul7Bench(int argc, char** argv);    // bullet reaction measures
int RunFall9Bench(int argc, char** argv);
int RunBul3Bench(int argc, char** argv);  // n bullets every dt, traced
int RunDrop9Bench(int argc, char** argv);  // drops from 1, 2, 4, 6 m
int RunSwayBench(int argc, char** argv);  // swaying deck
int RunStairsBench(int argc, char** argv);  // pushed down the stairs
int RunTwist8Bench(int argc, char** argv);  // upper-body twist after an off-centre bullet
int RunLeg8Bench(int argc, char** argv);    // bullet into a leg vs the target rows
int RunCarCall(int argc, char** argv);  // bench/carcall.cpp
int RunDeath8Bench(int argc, char** argv);  // death and relaxation
int RunBul3Bench(int argc, char** argv);  // n bullets every dt, traced
int RunDrop9Bench(int argc, char** argv);  // drops from 1, 2, 4, 6 m
int RunUpperBench(int argc, char** argv);    // upper body after pushes
int RunStaggerBench(int argc, char** argv);  // recoil against the target table
int RunAB(int nbots, const char* scene, int rounds, const std::vector<std::string>& args);
int RunSelfTest();
int RunPushOne(const char* scene, double moment);
int RunSweepBench(const char* label, int threads);
int RunStepsBench(const char* label, int threads);
int RunCopTest(double lean, int axis);
int RunStepsOne(const char* scene);
int RunShiftBench(const char* one);
int RunSweepOne(const char* scene);
int RunObliqueBench(const char* label, int threads);

void ApplyOverrides(emil::World& w);  // -p name=value
bool HasOverrides();
int RunTorsionBench(int argc, char** argv);
int RunFallBench(const char* label, int threads);

namespace {

// Allocation counter: MuJoCo heap allocations go through mju_user_malloc; C++ ones through operator new.
long g_allocs = 0;
bool g_trace = false;
bool g_count = false;
void* CountingMalloc(size_t n) {
  if (g_count) ++g_allocs;
  return std::malloc(n);
}

struct StandStats {
  double ankle_sep = 0, pelvis_h = 0, pelvis_fwd_ankle = 0, pelvis_fwd_foot = 0, knee = 0;
  double sway_rms = 0, sway_p2p = 0;
  bool fell = false;
  int self_contacts = 0;
  double lift = 0;    // largest ankle rise above its lowest height after 0.5 s [m] (a foot lift)
  int air_frames = 0; // frames after 0.5 s with a foot out of floor contact
  double one_leg = 0; // time after 0.5 s at rest (mode 0, no swing) with a foot under 5 % of the weight [s]
};

StandStats RunStand(World& w, int bot, FILE* poses, double secs) {
  StandStats st;
  const mjModel* m = w.model();
  mjData* d = w.data();
  const Bot& b = w.bot(bot);
  const int frames = (int)std::lround(secs / World::kFrameDt);
  double sx = 0, sy = 0, sxx = 0, syy = 0, minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
  int n = 0;
  double az_lo[2] = {1e9, 1e9};
  for (int f = 0; f < frames; ++f) {
    w.StepFrame();
    const double* pp = d->xpos + 3 * b.body0;
    if (g_trace && f % 6 == 0 && f < 240) {
      const Percept& P = b.pc[b.cur];
      printf("t=%.2f mode=%d ph=%d sw=%d steps=%d pz=%.3f com=%.3f,%.3f v=%.2f,%.2f c=%d%d aL=%.3f,%.3f,%.3f aR=%.3f,%.3f,%.3f pp=%.2f\n",
             d->time, b.mode, b.phase, b.swing, b.steps, pp[2], P.com[0], P.com[1], P.com_vel[0], P.com_vel[1],
             P.foot_contact[0], P.foot_contact[1], P.ankle[0][0], P.ankle[0][1], P.ankle[0][2], P.ankle[1][0],
             P.ankle[1][1], P.ankle[1][2], P.pelvis_pitch);
      printf("   flexT %.3f %.3f Q %.3f %.3f kneeT %.3f %.3f Q %.3f %.3f dorT %.3f %.3f Q %.3f %.3f\n",
             b.target[J_HIP_FLEX_L], b.target[J_HIP_FLEX_R], d->qpos[b.qadr0 + J_HIP_FLEX_L], d->qpos[b.qadr0 + J_HIP_FLEX_R],
             b.target[J_KNEE_L], b.target[J_KNEE_R], d->qpos[b.qadr0 + J_KNEE_L], d->qpos[b.qadr0 + J_KNEE_R],
             b.target[J_ANKLE_DORSI_L], b.target[J_ANKLE_DORSI_R], d->qpos[b.qadr0 + J_ANKLE_DORSI_L],
             d->qpos[b.qadr0 + J_ANKLE_DORSI_R]);
      printf("   abdT %.3f %.3f abdQ %.3f %.3f invT %.3f %.3f invQ %.3f %.3f roll %.3f\n", b.target[J_HIP_ABD_L], b.target[J_HIP_ABD_R], d->qpos[b.qadr0+J_HIP_ABD_L], d->qpos[b.qadr0+J_HIP_ABD_R], b.target[J_ANKLE_INV_L], b.target[J_ANKLE_INV_R], d->qpos[b.qadr0+J_ANKLE_INV_L], d->qpos[b.qadr0+J_ANKLE_INV_R], P.pelvis_roll);
    }
    if (pp[2] < 0.5) st.fell = true;
    if (d->time >= 0.5) {
      const Percept& P = b.pc[b.cur];
      for (int k = 0; k < 2; ++k) {
        az_lo[k] = std::fmin(az_lo[k], P.ankle[k][2]);
        st.lift = std::fmax(st.lift, P.ankle[k][2] - az_lo[k]);
        if (!P.foot_contact[k]) ++st.air_frames;
      }
      if (b.mode == 0 && b.swing < 0 && std::fmin(b.ld[0], b.ld[1]) < 0.05) st.one_leg += World::kFrameDt;
    }
    for (int i = 0; i < d->ncon; ++i) {  // self/other-body contacts not involving the floor
      const mjContact& c = d->contact[i];
      if (c.geom1 != 0 && c.geom2 != 0) st.self_contacts++;
      const int g = c.geom1 == 0 ? c.geom2 : (c.geom2 == 0 ? c.geom1 : -1);
      if (g >= 0) {
        const int bd = m->geom_bodyid[g] - b.body0;
        if (bd >= 0 && bd <= B_HEAD) st.fell = true;
      }
    }
    if (poses) {
      fprintf(poses, "%.4f", d->time);
      for (int k = 1; k < m->nbody; ++k)
        fprintf(poses, ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", d->xpos[3 * k], d->xpos[3 * k + 1], d->xpos[3 * k + 2],
                d->xquat[4 * k], d->xquat[4 * k + 1], d->xquat[4 * k + 2], d->xquat[4 * k + 3]);
      fputc('\n', poses);
    }
    if (d->time >= 2.0) {
      const double x = pp[0], y = pp[1];
      sx += x; sy += y; sxx += x * x; syy += y * y; ++n;
      minx = std::fmin(minx, x); maxx = std::fmax(maxx, x); miny = std::fmin(miny, y); maxy = std::fmax(maxy, y);
    }
  }
  const double* pp = d->xpos + 3 * b.body0;
  const double* aL = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_L);
  const double* aR = d->xanchor + 3 * (b.jnt0 + J_ANKLE_DORSI_R);
  const double* fL = d->geom_xpos + 3 * b.geom_foot[0];
  const double* fR = d->geom_xpos + 3 * b.geom_foot[1];
  st.ankle_sep = std::hypot(aL[0] - aR[0], aL[1] - aR[1]);
  st.pelvis_h = pp[2];
  st.pelvis_fwd_ankle = pp[0] - 0.5 * (aL[0] + aR[0]);
  st.pelvis_fwd_foot = pp[0] - 0.5 * (fL[0] + fR[0]);
  st.knee = 0.5 * (d->qpos[b.qadr0 + J_KNEE_L] + d->qpos[b.qadr0 + J_KNEE_R]);
  if (n > 1) {
    const double vx = sxx / n - (sx / n) * (sx / n), vy = syy / n - (sy / n) * (sy / n);
    st.sway_rms = std::sqrt(std::fmax(0, vx + vy));
    st.sway_p2p = std::hypot(maxx - minx, maxy - miny);
  }
  return st;
}

static mjtNum BenchClockMs() {  // MuJoCo timers need a clock callback
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double TimeFrames(int nbots, int frames, int* fallen, long* allocs) {
  mjcb_time = getenv("EMIL_NOTIMER") ? nullptr : BenchClockMs;  // the stage timers cost clock reads per step
  World w;
  char err[512] = "";
  std::vector<BodyParams> bodies(nbots);
  for (int i = 0; i < nbots; ++i) {  // varied bodies in one world
    bodies[i].height = 1.70f + 0.03f * (i % 4);
    bodies[i].mass = 70.0f + 5.0f * (i % 5);
    bodies[i].female = (i % 3) == 2;
    bodies[i].build = -0.5f + 0.25f * (i % 5);
  }
  if (getenv("EMIL_TOPT")) w.tune_opt = getenv("EMIL_TOPT");  // <option> attributes for speed trials
  if (!w.Build(bodies, 2.0, 7, err, sizeof err)) {
    fprintf(stderr, "build failed: %s\n", err);
    return -1;
  }
  ApplyOverrides(w);
  if (HasOverrides()) w.Reset();  // spawn pose from the overridden parameters
  for (int f = 0; f < 30; ++f) w.StepFrame();  // warm-up
  g_allocs = 0;
  g_count = true;
  const auto t0 = std::chrono::steady_clock::now();
  double ncon = 0, nit = 0, nefc = 0;
  for (int f = 0; f < frames; ++f) {
    w.StepFrame();
    ncon += w.data()->ncon;
    nit += w.data()->solver_niter[0];
    nefc += w.data()->nefc;
  }
  const auto t1 = std::chrono::steady_clock::now();
  fprintf(stderr, "diag %d bots: contacts %.1f, constraint rows %.1f, solver iterations %.1f per step\n", nbots,
          ncon / frames, nefc / frames, nit / frames);
  {  // MuJoCo internal timers: share of the step spent per stage (load-independent ratio).
    const mjData* d = w.data();
    const double st = d->timer[mjTIMER_POSITION].duration + d->timer[mjTIMER_VELOCITY].duration +
                      d->timer[mjTIMER_ACTUATION].duration + d->timer[mjTIMER_CONSTRAINT].duration + 1e-12;
    fprintf(stderr, "diag %d bots: share of position+velocity+actuation+constraint: collision %.0f%% (broad %.0f%%, narrow %.0f%%), solver %.0f%%\n", nbots,
            100 * d->timer[mjTIMER_POS_COLLISION].duration / st, 100 * d->timer[mjTIMER_COL_BROAD].duration / st,
            100 * d->timer[mjTIMER_COL_NARROW].duration / st, 100 * d->timer[mjTIMER_CONSTRAINT].duration / st);
  }
  {  // Physics ms per frame from MuJoCo's stage timers (mean per call x substeps), rest = controllers + glue
    const mjData* d = w.data();
    const int st[5] = {mjTIMER_POSITION, mjTIMER_VELOCITY, mjTIMER_ACTUATION, mjTIMER_CONSTRAINT, mjTIMER_ADVANCE};
    const char* nm[5] = {"position", "velocity", "actuation", "constraint", "advance"};
    double tot = 0, part[5];
    for (int k = 0; k < 5; ++k) {
      part[k] = d->timer[st[k]].number > 0 ? d->timer[st[k]].duration / d->timer[st[k]].number * World::kSubsteps : 0;
      tot += part[k];
    }
    const double fr = std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;
    fprintf(stderr, "diag %d bots: physics %.3f ms/frame (", nbots, tot);
    for (int k = 0; k < 5; ++k) fprintf(stderr, "%s%s %.3f", k ? ", " : "", nm[k], part[k]);
    fprintf(stderr, "), frame %.3f, controllers and glue %.3f\n", fr, fr - tot);
    const int ps[5] = {mjTIMER_POS_KINEMATICS, mjTIMER_POS_INERTIA, mjTIMER_POS_COLLISION, mjTIMER_POS_MAKE,
                       mjTIMER_POS_PROJECT};
    const char* pn[5] = {"kinematics", "inertia", "collision", "make", "project"};
    fprintf(stderr, "diag %d bots: position parts ms/frame:", nbots);
    for (int k = 0; k < 5; ++k)
      fprintf(stderr, " %s %.3f", pn[k],
              d->timer[ps[k]].number > 0 ? d->timer[ps[k]].duration / d->timer[ps[k]].number * World::kSubsteps : 0.0);
    fprintf(stderr, "; islands %d, nv %d\n", (int)d->nisland, (int)w.model()->nv);
  }
  g_count = false;
  *allocs = g_allocs;
  *fallen = 0;
  for (int i = 0; i < nbots; ++i)
    if (w.data()->xpos[3 * w.bot(i).body0 + 2] < 0.5) ++*fallen;
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / frames;
}

}  // namespace

// Parameter overrides from the command line: -p name=value (any number, any position), applied after Build.
static std::vector<std::pair<int, float>> g_over;
bool HasOverrides() { return !g_over.empty(); }
bool OverrideValue(int idx, float* v) {  // last override of parameter idx
  bool has = false;
  for (auto& o : g_over) if (o.first == idx) { *v = o.second; has = true; }
  return has;
}
void ApplyOverrides(World& w) {
  for (auto& o : g_over) w.param[o.first] = o.second;
}


// Balance inputs for L2 (tone, no_step, steps_left, lean). One bot per case, the input set at 1.0 s.
int RunInputTest() {
  const char* names[4] = {"tone 0", "no_step + push 0.5 m/s", "steps_left 1 + push 0.9 m/s", "lean 0.15 m fwd for 0.3 s"};
  for (int c = 0; c < 4; ++c) {
    World w;
    char err[1024] = "";
    if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) {
      fprintf(stderr, "build failed: %s\n", err);
      return 1;
    }
    ApplyOverrides(w);
    mjData* d = w.data();
    Bot& b = w.bot(0);
    int steps0 = 0, gave = 0;
    double zmin = 9, x0 = 0;
    const int frames = (int)std::lround(4.0 / World::kFrameDt), f1 = (int)std::lround(1.0 / World::kFrameDt);
    for (int f = 0; f < frames; ++f) {
      if (f == f1) {
        steps0 = b.steps;
        x0 = d->subtree_com[3 * b.body0];
        if (c == 0) b.in.tone = 0;
        if (c == 1) b.in.no_step = 1;
        if (c == 2) b.in.steps_left = 1;
        if (c == 3) b.in.lean[0] = 0.15;
        if (c == 1 || c == 2) {  // stagger-like chest push forward
          const double* o = d->xpos + 3 * (b.body0 + B_CHEST);
          const double J = (c == 1 ? 0.5 : 0.9) * b.dims.total_mass;
          const double p[3] = {o[0] - 0.1, o[1], o[2] + 0.06}, imp[3] = {J, 0, 0};
          w.PushAt(0, imp, p, B_CHEST);  // standard chest push point
        }
      }
      if (c == 3 && f == f1 + (int)std::lround(0.3 / World::kFrameDt)) b.in.lean[0] = 0;  // a short lean (bullet)
      w.StepFrame();
      if (f >= f1) {
        zmin = std::fmin(zmin, d->xpos[3 * b.body0 + 2]);
        gave |= b.mode == 2;
      }
    }
    printf("intest %-28s steps %d, pelvis min %.3f m, gave up %d, CoM x moved %.2f m, steps_left %d\n", names[c],
           b.steps - steps0, zmin, gave, d->subtree_com[3 * b.body0] - x0, b.in.steps_left);
  }
  return 0;
}

int main(int argc, char** argv) {
  {  // strip "-p name=value" pairs
    int k = 1;
    for (int i = 1; i < argc; ++i) {
      if (!std::strcmp(argv[i], "-p") && i + 1 < argc) {
        const char* a = argv[++i];
        const char* eq = std::strchr(a, '=');
        bool ok = false;
        for (int j = 0; eq && j < P_COUNT; ++j)
          if ((size_t)(eq - a) == std::strlen(kParamDefs[j].name) && !std::strncmp(a, kParamDefs[j].name, eq - a)) {
            g_over.push_back({j, (float)std::atof(eq + 1)});
            ok = true;
          }
        if (!ok) { fprintf(stderr, "unknown param override: %s\n", a); return 2; }
      } else if (!std::strcmp(argv[i], "--params") && i + 1 < argc) {  // file of name=value lines (picker output)
        FILE* pf = fopen(argv[++i], "r");
        if (!pf) { fprintf(stderr, "cannot open %s\n", argv[i]); return 2; }
        char ln[256];
        while (fgets(ln, sizeof ln, pf)) {
          char* eq = std::strchr(ln, '=');
          if (ln[0] == '#' || !eq) continue;
          *eq = 0;
          bool ok = false;
          for (int j = 0; j < P_COUNT; ++j)
            if (!std::strcmp(kParamDefs[j].name, ln)) { g_over.push_back({j, (float)std::atof(eq + 1)}); ok = true; }
          if (!ok) { fprintf(stderr, "unknown param in file: %s\n", ln); return 2; }
        }
        fclose(pf);
      } else {
        argv[k++] = argv[i];
      }
    }
    argc = k;
  }
  mju_user_malloc = CountingMalloc;
  if (argc >= 2 && !std::strcmp(argv[1], "push"))  // emil_bench push [label] [threads]
    return RunPushBench(argc >= 3 ? argv[2] : "cur", argc >= 4 ? std::atoi(argv[3]) : 8);
  if (argc >= 2 && !std::strcmp(argv[1], "torsion")) return RunTorsionBench(argc, argv);  // foot twisting test
  if (argc >= 2 && !std::strcmp(argv[1], "fall"))  // emil_bench fall [label] [threads]: push bench + per-frame CSVs
    return RunFallBench(argc >= 3 ? argv[2] : "fall", argc >= 4 ? std::atoi(argv[3]) : 4);
  if (argc >= 2 && !std::strcmp(argv[1], "selftest")) return RunSelfTest();
  if (argc >= 2 && !std::strcmp(argv[1], "ab")) {  // emil_bench ab bots stand|fall rounds variant...: interleaved A/B CPU time
    std::vector<std::string> v;
    for (int i = 5; i < argc; ++i) v.push_back(argv[i]);
    return RunAB(argc >= 3 ? std::atoi(argv[2]) : 8, argc >= 4 ? argv[3] : "stand", argc >= 5 ? std::atoi(argv[4]) : 5, v);
  }
  if (argc >= 2 && !std::strcmp(argv[1], "opt")) return RunOptBench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "stagger")) return RunStaggerBench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "upper6")) return RunUpper6Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "bul7")) return RunBul7Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "fall9")) return RunFall9Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "bul3")) return RunBul3Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "drop9")) return RunDrop9Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "sway")) return RunSwayBench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "stairs")) return RunStairsBench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "twist8")) return RunTwist8Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "leg8")) return RunLeg8Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "carcall")) return RunCarCall(argc, argv);  // car call as from the editor
  if (argc >= 2 && !std::strcmp(argv[1], "death8")) return RunDeath8Bench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "geomdump")) {  // shapes after t s of standing, for frame sheets
    World w;
    char err[512];
    if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const double ts = argc > 2 ? std::atof(argv[2]) : 2.0;
    while (w.data()->time < ts) w.StepFrame();
    const mjModel* m = w.model();
    const mjData* d = w.data();
    for (int g = 0; g < m->ngeom; ++g) {
      if (m->geom_bodyid[g] < w.bot(0).body0 || m->geom_bodyid[g] >= w.bot(0).body0 + B_COUNT) continue;
      const double* p = d->geom_xpos + 3 * g;
      const double* R = d->geom_xmat + 9 * g;
      const double* z = m->geom_size + 3 * g;
      printf("G %d %d %.5f %.5f %.5f %.5f %.5f %.5f", g, m->geom_type[g], z[0], z[1], z[2], p[0], p[1], p[2]);
      for (int k = 0; k < 9; ++k) printf(" %.5f", R[k]);
      printf("\n");
      if (m->geom_type[g] == mjGEOM_MESH) {
        const int me = m->geom_dataid[g];
        const float* V = m->mesh_vert + 3 * m->mesh_vertadr[me];
        for (int i = 0; i < m->mesh_vertnum[me]; ++i) printf("V %.5f %.5f %.5f\n", V[3 * i], V[3 * i + 1], V[3 * i + 2]);
        const int* F = m->mesh_face + 3 * m->mesh_faceadr[me];
        for (int f = 0; f < m->mesh_facenum[me]; ++f) printf("F %d %d %d\n", F[3 * f], F[3 * f + 1], F[3 * f + 2]);
      }
    }
    return 0;
  }
  if (argc >= 2 && !std::strcmp(argv[1], "skull")) {  // skull mesh as compiled (editor line transform check)
    World w;
    char err[512];
    if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    const mjModel* m = w.model();
    for (int g = 0; g < m->ngeom; ++g) {
      if (m->geom_type[g] != mjGEOM_MESH) continue;
      const int me = m->geom_dataid[g];
      const float* V = m->mesh_vert + 3 * m->mesh_vertadr[me];
      const double* mp = m->mesh_pos + 3 * me;
      double R[9];
      mju_quat2Mat(R, m->mesh_quat + 4 * me);
      const double k = m->mesh_scale[3 * me];
      double e1 = 0, e2 = 0;
      for (int i = 0; i < 3; ++i) {
        const double a0[3] = {k * kSkullV[3 * i], k * kSkullV[3 * i + 1], k * kSkullV[3 * i + 2]};
        for (int r = 0; r < 3; ++r) {
          double f1 = mp[r], f2 = 0;
          for (int c = 0; c < 3; ++c) { f1 += R[3 * r + c] * V[3 * i + c]; f2 += R[3 * r + c] * (a0[c] - mp[c]); }
          e1 += (f1 - a0[r]) * (f1 - a0[r]);
          e2 += (f2 - V[3 * i + r]) * (f2 - V[3 * i + r]);
        }
      }
      printf("skull geom %d: verts %d faces %d (asset %d), residual asset=p+Rv %.2g, v=R(a-p) %.2g; body %d pos %.3f %.3f %.3f\n",
             g, m->mesh_vertnum[me], m->mesh_facenum[me], kSkullNV, e1, e2, m->geom_bodyid[g], m->geom_pos[3 * g],
             m->geom_pos[3 * g + 1], m->geom_pos[3 * g + 2]);
    }
    return 0;
  }
  if (argc >= 2 && !std::strcmp(argv[1], "upper")) return RunUpperBench(argc, argv);
  if (argc >= 2 && !std::strcmp(argv[1], "inertia")) {  // per joint inertia below and response inertia
    World w;                                               // after 1 s of standing (three-number muscle numbers)
    char err[1024] = "";
    if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) return 1;
    ApplyOverrides(w);
    for (int f = 0; f < 60; ++f) w.StepFrame();
    const mjModel* m = w.model();
    const mjData* d = w.data();
    const Bot& b = w.bot(0);
    printf("joint            Ib(M_jj-arm)  Im(1/Minv_jj)  arm  [kg m^2]\n");
    for (int j = 0; j < J_COUNT; ++j) {
      const int dof = b.dof0 + j;
      const double a0 = m->dof_armature[dof], Mjj = d->M[m->M_rowadr[dof] + m->M_rownnz[dof] - 1];
      printf("%-16s %10.4f %12.4f %8.4f\n", JointName(j), Mjj - a0, w.RespInertia(dof), a0);
    }
    return 0;
  }
  if (argc >= 2 && !std::strcmp(argv[1], "creep")) return RunCreepBench(argc, argv);  // foot creep, tangential force
  if (argc >= 2 && !std::strcmp(argv[1], "speed"))  // emil_bench speed [bots] [stand|fall] [reps] [frames]
    return RunSpeed(argc >= 3 ? std::atoi(argv[2]) : 8, argc >= 4 ? argv[3] : "stand",
                    argc >= 5 ? std::atoi(argv[4]) : 7, argc >= 6 ? std::atoi(argv[5]) : 600);
  if (argc >= 2 && !std::strcmp(argv[1], "pairs")) return RunPairs(argc >= 3 ? std::atoi(argv[2]) : 1, argc >= 4);
  if (argc >= 2 && !std::strcmp(argv[1], "cstat"))  // emil_bench cstat [bots] [stand|fall|both]: cost counters
    return RunCstat(argc >= 3 ? std::atoi(argv[2]) : 8, argc >= 4 ? argv[3] : "both");
  if (argc >= 2 && !std::strcmp(argv[1], "avoid"))  // emil_bench avoid [threads]: dodge, flinch, gaze
    return RunAvoidBench(argc >= 3 ? std::atoi(argv[2]) : 4);
  if (argc >= 2 && !std::strcmp(argv[1], "rx"))  // emil_bench rx [label] [threads]: reaction bench
    return RunReactBench(argc >= 3 ? argv[2] : "cur", argc >= 4 ? std::atoi(argv[3]) : 8);
  if (argc >= 2 && !std::strcmp(argv[1], "shift"))  // emil_bench shift [scene]: weight shift / one-leg stance
    return RunShiftBench(argc >= 3 ? argv[2] : nullptr);
  if (argc >= 2 && !std::strcmp(argv[1], "steps"))  // emil_bench steps [label] [threads]: commanded steps
    return RunStepsBench(argc >= 3 ? argv[2] : "st", argc >= 4 ? std::atoi(argv[3]) : 6);
  if (argc >= 3 && !std::strcmp(argv[1], "cop"))  // emil_bench cop <lean m> [x|y]: centre-of-pressure shift check
    return RunCopTest(std::atof(argv[2]), argc >= 4 && argv[3][0] == 'y');
  if (argc >= 3 && !std::strcmp(argv[1], "steps1"))  // emil_bench steps1 L_f15: traced commanded step
    return RunStepsOne(argv[2]);
  if (argc >= 2 && !std::strcmp(argv[1], "sweep"))  // emil_bench sweep [label] [threads]: chest push sweep
    return RunSweepBench(argc >= 3 ? argv[2] : "sw", argc >= 4 ? std::atoi(argv[3]) : 8);
  if (argc >= 2 && !std::strcmp(argv[1], "obl"))  // emil_bench obl [label] [threads]: oblique pushes
    return RunObliqueBench(argc >= 3 ? argv[2] : "ob", argc >= 4 ? std::atoi(argv[3]) : 4);
  if (argc >= 3 && !std::strcmp(argv[1], "sweep1"))  // emil_bench sweep1 cleft10: traced sweep run
    return RunSweepOne(argv[2]);
  if (argc >= 2 && !std::strcmp(argv[1], "intest")) return RunInputTest();  // L2 balance inputs
  if (argc >= 5 && !std::strcmp(argv[1], "push1s")) {  // emil_bench push1s <scene> <moment> <dv>: stagger-like push
    extern double g_stag_dv;
    g_stag_dv = std::atof(argv[4]);
    return RunPushOne(argv[2], std::atof(argv[3]));
  }
  if (argc >= 4 && !std::strcmp(argv[1], "push1"))  // emil_bench push1 <scene> <moment>: traced single run
    return RunPushOne(argv[2], std::atof(argv[3]));
  const char* pose_path = nullptr;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--poses") && i + 1 < argc) pose_path = argv[++i];
    else if (!std::strcmp(argv[i], "--trace")) g_trace = true;

  World w;
  char err[1024] = "";
  if (!w.Build({BodyParams()}, 2.0, 1, err, sizeof err)) {
    fprintf(stderr, "build failed: %s\n", err);
    return 1;
  }
  ApplyOverrides(w);
  const mjModel* m = w.model();
  printf("model: nbody=%d njnt=%d nu=%d dt=%.5f mass=%.2f\n", (int)m->nbody, (int)m->njnt, (int)m->nu, m->opt.timestep,
         m->body_subtreemass[w.bot(0).body0]);
  FILE* poses = pose_path ? fopen(pose_path, "w") : nullptr;
  const StandStats s = RunStand(w, 0, poses, 10.0);
  if (poses) fclose(poses);
  printf("stand10s: fell=%d ankle_sep=%.3f pelvis_h=%.3f pelvis_fwd(ankle_mid)=%.3f pelvis_fwd(foot_mid)=%.3f "
         "knee=%.1fdeg sway_rms=%.1fmm sway_p2p=%.1fmm self_contacts=%d\n",
         s.fell, s.ankle_sep, s.pelvis_h, s.pelvis_fwd_ankle, s.pelvis_fwd_foot, s.knee * 57.2958,
         s.sway_rms * 1000, s.sway_p2p * 1000, s.self_contacts);
  printf("stand10s: balancer steps %d, foot lift %.1f mm, frames with a foot in the air %d (all must be ~0), "
         "one foot at rest %.2f s, rest stance half width %.3f m\n", w.bot(0).steps, s.lift * 1000, s.air_frames, s.one_leg,
         w.bot(0).rest_half);
  {  // Forward tilt of body z axes from vertical after the stand (deg, + = top forward): posture of trunk and head.
    const mjData* d = w.data();
    const int ids[4] = {B_PELVIS, B_LUMBAR, B_CHEST, B_HEAD};
    printf("posture: tilt fwd deg pelvis/lumbar/chest/head");
    for (int k = 0; k < 4; ++k) {
      const double* R = d->xmat + 9 * (w.bot(0).body0 + ids[k]);
      const double* P = d->xmat + 9 * w.bot(0).body0;  // pelvis heading = its x axis on the floor
      const double hx = P[0], hy = P[3], hn = std::hypot(hx, hy) + 1e-9;
      printf(" %.1f", std::atan2((R[2] * hx + R[5] * hy) / hn, R[8]) * 57.2958);  // body z axis along heading
    }
    printf("\n");
  }
  int f1 = 0, f8 = 0;
  long a1 = 0, a8 = 0;
  const double t1 = TimeFrames(1, 600, &f1, &a1), t8 = TimeFrames(8, 600, &f8, &a8);
  printf("timing: 1 bot %.3f ms/frame, 8 bots %.3f ms/frame (60 Hz, %d substeps); fallen after 10.5 s: %d/1, %d/8; "
         "allocs in timed frames: %ld, %ld\n", t1, t8, World::kSubsteps, f1, f8, a1, a8);
  const Bot& b0 = w.bot(0);
  const mjData* d = w.data();
  printf("diag: hip_abd L/R act %.2f/%.2f tgt %.2f/%.2f deg; foot load L/R %.2f/%.2f\n",
         d->qpos[b0.qadr0 + J_HIP_ABD_L] * 57.2958, d->qpos[b0.qadr0 + J_HIP_ABD_R] * 57.2958,
         b0.target[J_HIP_ABD_L] * 57.2958, b0.target[J_HIP_ABD_R] * 57.2958, b0.w_load[0], b0.w_load[1]);
  return 0;
}

// Count C++ heap allocations during timed frames as well.
void* operator new(size_t n) {
  if (g_count) ++g_allocs;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
