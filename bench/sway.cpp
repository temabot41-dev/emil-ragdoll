// Sway bench: standing on a swaying deck (World::with_platform) or on a suspension bridge (SW_BR=1,
// World::with_bridge: sideways wind on the six middle links, 45 -> 110 N per link by 5, 20 s each; period = wind).
// emil_bench sway [period ...] [-p k=v ...]   default periods 3.2 2.5; one thread per period.
// Amplitude 0.05 -> 0.35 m in 0.05 steps, 20 s each (sideways = world y, the bot faces +x); stops 3 s after a fall.
// Env: SW_A=<amp> one stage only (20 s); SW_AX=1 fore-aft (world x); SW_TR="t0 t1" trace every 0.1 s (one period).
// Per stage: fall time, trunk (chest axis) and head tilt from vertical max/mean, upper-arm angle from the trunk
// (mean of both, "shoulder away from the body"), pelvis height change vs calm standing, stance width (ankle gap
// across the pelvis) mean/min, frames with the feet crossed (gap < 5 cm), steps, first body part on the deck/floor.
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"
#include "stagger.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

namespace {

struct SwStage {
  double amp = 0, t_fall = -1, trunk_max = 0, trunk_sum = 0, head_max = 0, arm_sum = 0, arm_max = 0, pel_sum = 0;
  double w_sum = 0, w_min = 9, hand_t = -1;
  int n = 0, crossed = 0, steps = 0, off = 0;
  const char* first = "-";
};

double Deg(double c) { return std::acos(std::fmax(-1.0, std::fmin(1.0, c))) * 57.29578; }

// angle between the upper arm (shoulder -> elbow) and the chest's down axis [deg]
double ArmAngle(const mjData* d, const Bot& b, int up) {
  const double* s = d->xpos + 3 * (b.body0 + up);
  const double* e = d->xpos + 3 * (b.body0 + up + 1);
  const double* R = d->xmat + 9 * (b.body0 + B_CHEST);
  const double a[3] = {e[0] - s[0], e[1] - s[1], e[2] - s[2]};
  const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) + 1e-9;
  return Deg(-(a[0] * R[2] + a[1] * R[5] + a[2] * R[8]) / n);
}

void RunSway(double period, std::vector<SwStage>* out, std::string* log) {
  std::vector<BodyParams> bodies(1);
  World w;
  const bool br = getenv("SW_BR") && std::atoi(getenv("SW_BR")) == 1;
  w.with_platform = !br;
  w.with_bridge = br;
  char err[512];
  {
    std::lock_guard<std::mutex> lk(stag::BuildMutex());
    if (!w.Build(bodies, 3.0, 1, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return; }
  }
  ApplyOverrides(w);
  w.Reset();
  const mjModel* m = w.model();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  const bool fa = getenv("SW_AX") && std::atoi(getenv("SW_AX")) == 1;
  const double one = getenv("SW_A") ? std::atof(getenv("SW_A")) : -1;
  double tr0 = -1, tr1 = -1;
  if (getenv("SW_TR")) sscanf(getenv("SW_TR"), "%lf %lf", &tr0, &tr1);
  const int nst = one > 0 ? 1 : (br ? 14 : 7);
  // deck origin: platform offset (top at z 0) or the bridge middle (junction top)
  auto deck = [&](double* px, double* py, double* pz) {
    if (br) {
      double m3[3];
      w.BridgeMid(m3);
      *px = m3[0], *py = m3[1], *pz = m3[2];
    } else {
      w.PlatformPos(px, py);
      *pz = w.lift;
    }
  };
  const double kSettle = 2.0, kStage = 20.0;
  double pel0 = 0, t_fall = -1;
  int steps0 = 0;
  char buf[512];
  for (int fr = 0;; ++fr) {
    const double t = fr / 60.0;
    const int st = t < kSettle ? -1 : (int)((t - kSettle) / kStage);
    if (st >= nst) break;
    if (t_fall >= 0 && t > t_fall + 3.0) break;
    if (std::fabs(t - kSettle) < 0.5 / 60) {
      double qx, qy, qz;
      deck(&qx, &qy, &qz);
      pel0 = d->xpos[3 * (b.body0 + B_PELVIS) + 2] - qz;
      steps0 = b.steps;
    }
    if (st >= 0 && (int)out->size() <= st && t_fall >= 0) break;  // no new stage after a fall
    if (st >= 0 && (int)out->size() <= st) {  // new stage
      SwStage s;
      s.amp = one > 0 ? one : (br ? 45.0 + 5.0 * st : 0.05 * (st + 1));
      out->push_back(s);
      if (br) w.Bridge(s.amp, period);
      else w.Platform(fa ? s.amp : 0.0, fa ? 0.0 : s.amp, period);
      if (st > 0) (*out)[st - 1].steps = b.steps - steps0;
      steps0 = b.steps;
    }
    const int st0 = b.steps;
    w.StepFrame();
    if (st < 0) continue;
    if (tr0 >= 0 && t >= tr0 && t <= tr1 && b.steps != st0) {
      double px, py;
      double pz_;
      deck(&px, &py, &pz_);
      const Percept& Q = b.pc[b.cur ^ 1];
      snprintf(buf, sizeof buf, "STEP %.2f foot %d why %d deck %+5.2f com-deck %+5.2f v %+5.2f supv %+5.2f gap %4.2f\n", t,
               b.swing, b.dbg_why, fa ? px : py, (fa ? Q.com[0] - px : Q.com[1] - py), fa ? Q.com_vel[0] : Q.com_vel[1],
               fa ? b.in.sup_v[0] : b.in.sup_v[1], 0.0);
      *log += buf;
      // both feet's offset from the plank middle across the walkway (axis y of the plank) and its half width
      snprintf(buf, sizeof buf, "     deck off L %+5.2f R %+5.2f half %4.2f %4.2f tgt %+5.2f %+5.2f\n", Q.deck_off[0][1],
               Q.deck_off[1][1], Q.deck_half[0][1], Q.deck_half[1][1], b.sw_tg[0], b.sw_tg[1]);
      *log += buf;
    }
    SwStage& s = (*out)[st];
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
    const double* Rh = d->xmat + 9 * (b.body0 + B_HEAD);
    const double* Rp = d->xmat + 9 * (b.body0 + B_PELVIS);
    const double trunk = Deg(Rc[8]), head = Deg(Rh[8]);
    const double ln = std::hypot(Rp[1], Rp[4]) + 1e-9;
    const double* fl = d->xpos + 3 * (b.body0 + B_FOOT_L);
    const double* frr = d->xpos + 3 * (b.body0 + B_FOOT_R);
    const double gap = ((fl[0] - frr[0]) * Rp[1] + (fl[1] - frr[1]) * Rp[4]) / ln;
    const double arm = 0.5 * (ArmAngle(d, b, B_UARM_L) + ArmAngle(d, b, B_UARM_R));
    double dx_, dy_, dz_;
    deck(&dx_, &dy_, &dz_);
    // a foot below the deck top (stepped off) or the pelvis beyond the bridge edge
    const bool off = br ? (fl[2] < dz_ - 0.3 || frr[2] < dz_ - 0.3 || std::fabs(P[1] - dy_) > 0.6)
                        : (fl[2] < w.lift - 0.08 || frr[2] < w.lift - 0.08);
    if (t_fall < 0 && (P[2] - dz_ < 0.45 || b.mode == 2 || off)) {
      t_fall = t;
      s.t_fall = t - kSettle - st * kStage;
      if (off) s.off = 1;
      snprintf(buf, sizeof buf, "FALL %.2f pz-dz %.2f feet z %.2f %.2f dz %.2f py-dy %+.2f md %d\n", t, P[2] - dz_, fl[2],
               frr[2], dz_, P[1] - dy_, b.mode);
      *log += buf;
    }
    if (t_fall < 0) {
      s.n++;
      s.trunk_max = std::fmax(s.trunk_max, trunk);
      s.trunk_sum += trunk;
      s.head_max = std::fmax(s.head_max, head);
      s.arm_sum += arm;
      s.arm_max = std::fmax(s.arm_max, arm);
      s.pel_sum += P[2] - dz_ - pel0;
      s.w_sum += gap;
      s.w_min = std::fmin(s.w_min, gap);
      if (gap < 0.05) s.crossed++;
    }
    if (std::strcmp(s.first, "-") == 0) {  // first body part other than the feet on the deck or floor
      for (int i = 0; i < d->ncon; ++i) {
        const mjContact& c = d->contact[i];
        const int b1 = m->geom_bodyid[c.geom1] - b.body0, b2 = m->geom_bodyid[c.geom2] - b.body0;
        const bool own1 = b1 >= 0 && b1 < B_COUNT, own2 = b2 >= 0 && b2 < B_COUNT;
        if (own1 == own2) continue;
        const int bd = own1 ? b1 : b2;
        if (bd == B_FOOT_L || bd == B_FOOT_R) continue;
        s.first = (bd == B_HAND_L || bd == B_HAND_R || bd == B_FARM_L || bd == B_FARM_R) ? "hand"
                  : (bd == B_HEAD ? "head" : (bd == B_SHANK_L || bd == B_SHANK_R ? "knee" : "trunk"));
        s.hand_t = t - kSettle - st * kStage;
        break;
      }
    }
    if (tr0 >= 0 && t >= tr0 && t <= tr1 && fr % 6 == 0) {
      double px, py;
      double pz_;
      deck(&px, &py, &pz_);
      const double* vc = b.pc[b.cur ^ 1].com_vel;
      snprintf(buf, sizeof buf,
               "SW %.2f deck %+5.2f pel %+5.2f v %+5.2f trunk %4.1f (fx %+5.1f sy %+5.1f pp %+5.1f) head %4.1f arm %4.0f %4.0f gap %4.2f pz %4.2f sw %2d "
               "md %d ld %.2f %.2f fy %+5.3f %+5.3f fh %+4.0f %+4.0f br %.2f %.2f out %+.3f d %.3f %.3f\n",
               t, fa ? px : py, (fa ? P[0] - px : P[1] - py), fa ? vc[0] : vc[1], trunk, 57.3 * std::asin(Rc[2]), 57.3 * std::asin(Rc[5]), 57.3 * std::asin(Rp[2]), head,
               ArmAngle(d, b, B_UARM_L), ArmAngle(d, b, B_UARM_R), gap, P[2], b.swing, b.mode, b.ld[0], b.ld[1],
               (fa ? fl[0] - px : fl[1] - py), (fa ? frr[0] - px : frr[1] - py), fa ? b.foot_fh[0][0] : b.foot_fh[0][1],
               fa ? b.foot_fh[1][0] : b.foot_fh[1][1], b.brace, b.brace_c, b.dbg_out, b.dbg_dist[0], b.dbg_dist[1]);
      *log += buf;
    }
  }
  if (!out->empty() && out->back().steps == 0) out->back().steps = b.steps - steps0;
}

}  // namespace

int RunSwayBench(int argc, char** argv) {
  std::vector<double> periods;
  for (int i = 2; i < argc; ++i) {
    if (argv[i][0] == '-') { ++i; continue; }
    const double p = std::atof(argv[i]);
    if (p > 0.5) periods.push_back(p);
  }
  if (periods.empty()) periods = {3.2, 2.5};
  std::vector<std::vector<SwStage>> res(periods.size());
  std::vector<std::string> logs(periods.size());
  std::vector<std::thread> th;
  for (size_t i = 0; i < periods.size(); ++i) th.emplace_back(RunSway, periods[i], &res[i], &logs[i]);
  for (auto& t : th) t.join();
  for (size_t i = 0; i < periods.size(); ++i) {
    fputs(logs[i].c_str(), stdout);
    int last_ok = -1;
    for (size_t k = 0; k < res[i].size(); ++k) {
      const SwStage& s = res[i][k];
      const double n = std::fmax(s.n, 1);
      printf("SWAY T %.1f A %.2f | %s | trunk max %4.1f mean %4.1f head max %4.1f | arm mean %4.0f max %4.0f | pelvis %+5.1f cm "
             "| width %4.1f min %4.1f cm crossed %d | steps %d | first %s\n",
             periods[i], s.amp, s.t_fall >= 0 ? (s.off ? "OFF " : "FELL") : "stood", s.trunk_max, s.trunk_sum / n, s.head_max, s.arm_sum / n,
             s.arm_max, 100 * s.pel_sum / n, 100 * s.w_sum / n, 100 * s.w_min, s.crossed, s.steps, s.first);
      if (s.t_fall >= 0) printf("SWAY T %.1f A %.2f fell at +%.1f s, first contact %s at +%.1f s\n", periods[i], s.amp,
                                s.t_fall, s.first, s.hand_t);
      if (s.t_fall < 0) last_ok = (int)k;
    }
    printf("SWAY T %.1f stood up to A %.2f m\n", periods[i], last_ok >= 0 ? res[i][last_ok].amp : 0.0);
  }
  return 0;
}
