// Load-independent cost counters. `emil_bench cstat [bots] [scene]` (scene: stand | fall | both).
// Per control frame (read after the last of 8 substeps): contacts by class, constraint rows by type, solver iterations;
// per substep: narrow-phase candidate pairs by geometry-type pair (mjcb_contactfilter, counted before the narrow phase) and how many
// of them produced a contact. Time is not measured here (see `emil_bench ab`).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp

namespace {
const mjModel* g_m = nullptr;
int g_b0 = -1, g_nb = 0;
std::map<std::string, double> g_cand, g_body;  // by type pair, summed over substeps; g_body: bot 0 by body names
const char* GT(int t) {
  switch (t) {
    case mjGEOM_PLANE: return "plane";
    case mjGEOM_SPHERE: return "sphere";
    case mjGEOM_CAPSULE: return "capsule";
    case mjGEOM_ELLIPSOID: return "ellipsoid";
    case mjGEOM_BOX: return "box";
    default: return "other";
  }
}
int Filter(const mjModel* m, mjData*, int g1, int g2) {
  int a = m->geom_type[g1], b = m->geom_type[g2];
  if (a > b) std::swap(a, b);
  g_cand[std::string(GT(a)) + "-" + GT(b)] += 1;
  const int b1 = m->geom_bodyid[g1], b2 = m->geom_bodyid[g2];
  if (g_b0 >= 0 && b1 >= g_b0 && b2 >= g_b0 && g_nb > 0) {
    const int k1 = (b1 - g_b0)/B_COUNT, k2 = (b2 - g_b0)/B_COUNT;
    if (k1 == k2 && k1 < g_nb) {
      int x = (b1 - g_b0) % B_COUNT, y = (b2 - g_b0) % B_COUNT;
      if (x > y) std::swap(x, y);
      g_body[std::string(BodyName(x)) + "-" + BodyName(y) + " (" + GT(a) + "-" + GT(b) + ")"] += 1;
    }
  }
  return 0;  // keep the default behaviour
}

// Body role inside a bot (-1 = world, -2 = car or unknown).
int Role(World& w, int nbots, int body, int* bot) {
  if (body == 0) return -1;
  const int b0 = w.bot(0).body0;
  const int k = (body - b0) / B_COUNT;
  if (body < b0 || k >= nbots) return -2;
  *bot = k;
  return (body - b0) % B_COUNT;
}
const char* Cls(int r) {  // coarse part class
  if (r >= B_THIGH_L && r <= B_SHANK_R && r != B_FOOT_L && r != B_FOOT_R) return "leg";
  if (r == B_FOOT_L || r == B_FOOT_R) return "foot";
  if (r >= B_UARM_L) return "arm";
  return "trunk";
}

void Run(int nbots, bool fall) {
  World w;
  char err[512] = "";
  std::vector<BodyParams> bodies(nbots);
  for (int i = 0; i < nbots; ++i) {
    bodies[i].height = 1.70f + 0.03f * (i % 4);
    bodies[i].mass = 70.0f + 5.0f * (i % 5);
    bodies[i].female = (i % 3) == 2;
    bodies[i].build = -0.5f + 0.25f * (i % 5);
  }
  if (!w.Build(bodies, 2.0, 7, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return; }
  ApplyOverrides(w);
  const mjModel* m = w.model();
  const mjData* d = w.data();
  g_m = m;
  for (int f = 0; f < 120; ++f) w.StepFrame();
  if (fall) {
    for (int i = 0; i < nbots; ++i) {
      Bot& b = w.bot(i);
      const double* pp = w.data()->xpos + 3 * b.body0;
      const double pt[3] = {pp[0], pp[1], pp[2] + 0.35}, imp[3] = {-200.0, 0, 0};
      w.PushAt(i, imp, pt, B_LUMBAR);  // standard push point
    }
  }
  const int frames = fall ? 180 : 180;
  g_cand.clear();
  g_body.clear();
  g_b0 = w.bot(0).body0;
  g_nb = nbots;
  mjcb_contactfilter = Filter;
  std::map<std::string, double> con, rows;
  double ncon = 0, nefc = 0, nit = 0, nmax_con = 0, nnz = 0, nislands = 0;
  double found = 0;
  for (int f = 0; f < frames; ++f) {
    w.StepFrame();
    ncon += d->ncon; nefc += d->nefc; nit += d->solver_niter[0]; nnz += d->solver_nnz[0]; nislands += d->nisland;
    nmax_con = std::fmax(nmax_con, d->ncon);
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& c = d->contact[i];
      const int b1 = m->geom_bodyid[c.geom1], b2 = m->geom_bodyid[c.geom2];
      int k1 = -1, k2 = -1;
      const int r1 = Role(w, nbots, b1, &k1), r2 = Role(w, nbots, b2, &k2);
      std::string cl;
      if (r1 == -1 || r2 == -1) {
        const int r = r1 == -1 ? r2 : r1;
        cl = r < 0 ? "floor-other" : std::string(Cls(r)) + "-floor";
      } else if (r1 < 0 || r2 < 0) {
        cl = "car";
      } else if (k1 != k2) {
        cl = "bot-bot";
      } else {
        const std::string a = Cls(r1), b = Cls(r2);
        cl = (a == "trunk" || b == "trunk") ? "limb-trunk (self)" : (a + "-" + b + " (self)");
        if (a == "trunk" && b == "trunk") cl = "trunk-trunk (self)";
      }
      con[cl] += 1;
    }
    for (int i = 0; i < d->nefc; ++i) {
      const int t = d->efc_type[i];
      rows[t == mjCNSTR_CONTACT_PYRAMIDAL ? "contact (pyramidal)" : (t == mjCNSTR_LIMIT_JOINT ? "joint limit" :
           (t == mjCNSTR_FRICTION_DOF ? "dof friction" : (t == mjCNSTR_EQUALITY ? "equality" : "other")))] += 1;
    }
    found += d->ncon;
  }
  mjcb_contactfilter = nullptr;
  printf("== %d bots, %s, %d frames (%d substeps)\n", nbots, fall ? "fall (back 200 N s each)" : "quiet stand", frames, frames * 8);
  printf("per frame (end of frame): contacts %.1f (max %.0f), rows %.1f, solver iterations %.1f, nnz %.0f, islands %.1f\n", ncon / frames,
         nmax_con, nefc / frames, nit / frames, nnz / frames, nislands / frames);
  printf("contacts by class (per frame):\n");
  for (auto& kv : con) printf("  %-24s %7.2f\n", kv.first.c_str(), kv.second / frames);
  printf("constraint rows by type (per frame):\n");
  for (auto& kv : rows) printf("  %-24s %7.1f\n", kv.first.c_str(), kv.second / frames);
  printf("narrow-phase candidate pairs by geometry types (per substep), total contacts per substep ~ %.1f:\n", found / frames);
  double tot = 0;
  for (auto& kv : g_cand) tot += kv.second;
  for (auto& kv : g_cand) printf("  %-22s %8.1f\n", kv.first.c_str(), kv.second / (frames * 8.0));
  printf("  %-22s %8.1f\n", "total", tot / (frames * 8.0));
  printf("candidate pairs inside bots by body pair (per substep, all bots):\n");
  for (auto& kv : g_body) printf("  %-44s %6.2f\n", kv.first.c_str(), kv.second * (1.0 / (frames * 8.0)));
  printf("model: %d geoms, %d bodies, %d dofs, %d exclude pairs\n", m->ngeom, m->nbody, m->nv, m->nexclude);
}
}  // namespace

int RunCstat(int nbots, const char* scene) {
  if (!std::strcmp(scene, "stand") || !std::strcmp(scene, "both")) Run(nbots, false);
  if (!std::strcmp(scene, "fall") || !std::strcmp(scene, "both")) Run(nbots, true);
  return 0;
}

// ---- A/B timing: `emil_bench ab bots scene rounds variant...` -------------------------------------------------------------------
// A variant is "name=o:<option attrs>+f:<flag attrs>+g:<default geom attrs>" (any part optional; "base" = no change).
// Runs interleave the variants (A B C A B C ...), each run on a fresh world: 120 warm-up frames, then 120 timed frames of quiet
// stand or fall (back 200 N s per bot). Per variant the smallest thread CPU time per frame over the rounds is reported (less
// sensitive to a busy machine than wall time, still not exact); /proc/loadavg is printed before every round.
#include <ctime>

namespace {
double CpuMs() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
double g_tim[mjNTIMER];  // MuJoCo timers (thread CPU ms per frame) of the last timed run
struct Variant { std::string name, o, f, g, h; int b = 0; };
Variant ParseVariant(const std::string& a) {
  Variant v;
  const size_t eq = a.find('=');
  v.name = eq == std::string::npos ? a : a.substr(0, eq);
  if (eq == std::string::npos) return v;
  std::string rest = a.substr(eq + 1);
  size_t pos = 0;
  while (pos < rest.size()) {
    size_t e = rest.find('+', pos);
    if (e == std::string::npos) e = rest.size();
    const std::string part = rest.substr(pos, e - pos);
    if (part.size() > 2 && part[1] == ':') { if (part[0] == 'b') v.b = std::atoi(part.c_str() + 2); else (part[0] == 'o' ? v.o : (part[0] == 'f' ? v.f : (part[0] == 'h' ? v.h : v.g))) += part.substr(2) + " "; }
    pos = e + 1;
  }
  return v;
}
double TimeOne(const Variant& v, int nbots, bool fall, int frames, double* ncon, double* nefc, double* nit, int* fallen) {
  g_body_tune = v.b;
  g_body_foot = v.h;
  World w;
  w.tune_opt = v.o; w.tune_flag = v.f; w.tune_geom = v.g;
  char err[512] = "";
  std::vector<BodyParams> bodies(nbots);
  for (int i = 0; i < nbots; ++i) {
    bodies[i].height = 1.70f + 0.03f * (i % 4);
    bodies[i].mass = 70.0f + 5.0f * (i % 5);
    bodies[i].female = (i % 3) == 2;
    bodies[i].build = -0.5f + 0.25f * (i % 5);
  }
  if (!w.Build(bodies, 2.0, 7, err, sizeof err)) { fprintf(stderr, "build failed (%s): %s\n", v.name.c_str(), err); return -1; }
  ApplyOverrides(w);
  for (int f = 0; f < 120; ++f) w.StepFrame();
  if (fall) {
    for (int i = 0; i < nbots; ++i) {
      Bot& b = w.bot(i);
      const double* pp = w.data()->xpos + 3 * b.body0;
      const double pt[3] = {pp[0], pp[1], pp[2] + 0.35}, imp[3] = {-200.0, 0, 0};
      w.PushAt(i, imp, pt, B_LUMBAR);  // standard push point
    }
  }
  double tim0[mjNTIMER];
  for (int k = 0; k < mjNTIMER; ++k) tim0[k] = w.data()->timer[k].duration;
  double c = 0, e = 0, it = 0;
  const double t0 = CpuMs();
  for (int f = 0; f < frames; ++f) {
    w.StepFrame();
    c += w.data()->ncon; e += w.data()->nefc; it += w.data()->solver_niter[0];
  }
  const double t1 = CpuMs();
  for (int k = 0; k < mjNTIMER; ++k) g_tim[k] = (w.data()->timer[k].duration - tim0[k]) / frames;
  *ncon = c / frames; *nefc = e / frames; *nit = it / frames;
  *fallen = 0;
  for (int i = 0; i < nbots; ++i) if (w.data()->xpos[3 * w.bot(i).body0 + 2] < 0.5) ++*fallen;
  return (t1 - t0) / frames;
}
}  // namespace

int RunAB(int nbots, const char* scene, int rounds, const std::vector<std::string>& args) {
  mjcb_time = CpuMs;
  std::vector<Variant> vs;
  for (auto& a : args) vs.push_back(ParseVariant(a));
  if (vs.empty()) vs.push_back(ParseVariant("base"));
  const bool fall = !std::strcmp(scene, "fall");
  std::vector<double> best(vs.size(), 1e9), ncon(vs.size()), nefc(vs.size()), nit(vs.size());
  std::vector<int> fallen(vs.size());
  std::vector<std::vector<double>> tim(vs.size(), std::vector<double>(mjNTIMER, 1e9));
  for (int r = 0; r < rounds; ++r) {
    FILE* fl = fopen("/proc/loadavg", "r");
    char la[64] = "";
    if (fl) { if (!fgets(la, sizeof la, fl)) la[0] = 0; fclose(fl); }
    la[strcspn(la, "\n")] = 0;
    printf("round %d load %s:", r, la);
    for (size_t q = 0; q < vs.size(); ++q) {
      const size_t i = (q + r) % vs.size();  // the starting variant rotates: the first run after a pause is slower
      const double t = TimeOne(vs[i], nbots, fall, 120, &ncon[i], &nefc[i], &nit[i], &fallen[i]);
      printf(" %s %.3f", vs[i].name.c_str(), t);
      if (t >= 0) best[i] = std::fmin(best[i], t);
      for (int k = 0; k < mjNTIMER; ++k) tim[i][k] = std::fmin(tim[i][k], g_tim[k]);
    }
    printf("\n");
    fflush(stdout);
  }
  printf("== %d bots, %s: smallest CPU ms per frame over %d rounds (ratio to %s)\n", nbots, scene, rounds, vs[0].name.c_str());
  for (size_t i = 0; i < vs.size(); ++i)
    printf("  %-14s %.3f ms  x%.3f   contacts %.1f rows %.1f iterations %.2f fallen %d\n", vs[i].name.c_str(), best[i], best[i] / best[0],
           ncon[i], nefc[i], nit[i], fallen[i]);
  printf("MuJoCo stage timers, smallest CPU ms per frame (8 substeps):\n");
  for (size_t i = 0; i < vs.size(); ++i) {
    const std::vector<double>& t = tim[i];
    printf("  %-14s step %.3f | position %.3f (kinematics %.3f inertia %.3f collision %.3f [broad %.3f narrow %.3f] make %.3f project %.3f)"
           " velocity %.3f actuation %.3f constraint(solve) %.3f advance %.3f\n", vs[i].name.c_str(), t[mjTIMER_STEP], t[mjTIMER_POSITION],
           t[mjTIMER_POS_KINEMATICS], t[mjTIMER_POS_INERTIA], t[mjTIMER_POS_COLLISION], t[mjTIMER_COL_BROAD], t[mjTIMER_COL_NARROW],
           t[mjTIMER_POS_MAKE], t[mjTIMER_POS_PROJECT], t[mjTIMER_VELOCITY], t[mjTIMER_ACTUATION], t[mjTIMER_CONSTRAINT], t[mjTIMER_ADVANCE]);
  }
  return 0;
}

// emil_bench speed [bots] [stand|fall] [reps] [frames]: thread CPU ms per frame without MuJoCo timer callbacks
// (clock reads inflate the step), min and median over reps. Pin to one core (taskset -c 3) for stable numbers.
int RunSpeed(int nbots, const char* scene, int reps, int frames) {
  mjcb_time = nullptr;
  const bool fall = !std::strcmp(scene, "fall");
  Variant v = ParseVariant("base");
  std::vector<double> t;
  for (int r = 0; r < reps; ++r) {
    double nc, ne, ni;
    int fl;
    t.push_back(TimeOne(v, nbots, fall, frames, &nc, &ne, &ni, &fl));
  }
  std::sort(t.begin(), t.end());
  printf("speed %d bots %s: min %.3f median %.3f max %.3f ms per frame (CPU, %d reps x %d frames)\n", nbots, scene,
         t[0], t[t.size() * 1 / 2], t.back(), reps, frames);
  return 0;
}

// emil_bench pairs [fall]: geom pairs that reach the narrow phase (mjcb_contactfilter counts them) per substep.
namespace {
std::map<std::pair<int, int>, long> g_pairs;
long g_pair_total = 0;
int CountPair(const mjModel*, mjData*, int g1, int g2) {
  ++g_pairs[{std::min(g1, g2), std::max(g1, g2)}];
  ++g_pair_total;
  return 0;
}
}  // namespace
int RunPairs(int nbots, bool fall) {
  World w;
  char err[512] = "";
  std::vector<BodyParams> bodies(nbots);
  if (!w.Build(bodies, 2.0, 7, err, sizeof err)) return 1;
  for (int f = 0; f < 120; ++f) w.StepFrame();
  if (fall) {
    Bot& b = w.bot(0);
    const double* pp = w.data()->xpos + 3 * b.body0;
    const double pt[3] = {pp[0], pp[1], pp[2] + 0.35}, imp[3] = {-200.0, 0, 0};
    w.PushAt(0, imp, pt, B_LUMBAR);  // standard push point
  }
  mjcb_contactfilter = CountPair;
  const int frames = 120;
  double nc = 0;
  for (int f = 0; f < frames; ++f) {
    w.StepFrame();
    nc += w.data()->ncon;
  }
  mjcb_contactfilter = nullptr;
  const double steps = (double)frames * World::kSubsteps;
  printf("pairs per substep (narrow-phase candidates, bot 0 + world): %.1f, contacts %.1f\n", g_pair_total / steps,
         nc / frames);
  std::vector<std::pair<long, std::pair<int, int>>> v;
  for (auto& kv : g_pairs) v.push_back({kv.second, kv.first});
  std::sort(v.rbegin(), v.rend());
  for (size_t i = 0; i < v.size() && i < 40; ++i) {
    const int g1 = v[i].second.first, g2 = v[i].second.second;
    const char* n1 = mj_id2name(w.model(), mjOBJ_BODY, w.model()->geom_bodyid[g1]);
    const char* n2 = mj_id2name(w.model(), mjOBJ_BODY, w.model()->geom_bodyid[g2]);
    printf("  %5.2f per substep  %s (type %d) - %s (type %d)\n", v[i].first / steps, n1 ? n1 : "?",
           w.model()->geom_type[g1], n2 ? n2 : "?", w.model()->geom_type[g2]);
  }
  return 0;
}
