// Bullet series bench: n chest bullets every dt seconds (live reaction), traced every 0.1 s.
//   emil_bench bul3 [n=3] [dt=1.0] [-p ...]
// From 1.5 s of standing. Columns per time after the first bullet: path = pelvis from its place at the first bullet
// [cm]; vb = CoM speed along the bullet [m/s]; mode; steps (touchdowns); pull = bullet pull on (1); asF = assist force
// [N]; crouch; hold = bul_hold lean [m]; sl = steps_left; cpo = capture point outside the support [m]; sw = swing foot
// (-1 none); lift/len of the last step: foot travel along the bullet at its touchdown [cm].
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "emil_body.h"
#include "emil_world.h"
#include "footspin.h"

using namespace emil;

void ApplyOverrides(World& w);  // main.cpp: -p / --params

int RunBul3Bench(int argc, char** argv) {
  const int n = argc > 2 ? std::atoi(argv[2]) : 3;
  const double gap = argc > 3 ? std::atof(argv[3]) : 1.0;
  // hit point left of the middle [m] (a negative number is a value, not an option)
  const double lat = argc > 4 && (argv[4][0] != '-' || (argv[4][1] >= '0' && argv[4][1] <= '9')) ? std::atof(argv[4]) : 0.0;
  // hit height above the floor at 1.78 m [m], picked by a ray (see the shot); 0 = chest centre + 5 cm.
  // Belly: 1.05 m, lat -0.016 (halves +-0.10)
  const double hz = argc > 5 && argv[5][0] != '-' ? std::atof(argv[5]) : 0.0;
  std::vector<BodyParams> bodies(1);
  World w;
  char err[512];
  if (!w.Build(bodies, 3.0, 21, err, sizeof err)) { fprintf(stderr, "build failed: %s\n", err); return 1; }
  ApplyOverrides(w);
  w.Reset();
  mjData* d = w.data();
  const Bot& b = w.bot(0);
  // moment of the first bullet after the reset (6th argument; typical: 1.5 and 2.5 s)
  const double kMoment = argc > 6 && argv[6][0] != '-' ? std::atof(argv[6]) : 1.5, kTail = 6.0;
  double t0 = -1, fx = 1, fy = 0, p0[2] = {0, 0}, fz[2] = {0, 0}, path_max = 0;
  // pelvis turn about the vertical from the first bullet (largest over the scene and over any 0.5 s) and the pelvis
  // travel length; the foot-turn measure alone misses a body spun around by its steps
  double yaw_prev = 0, yaw_u = 0, yaw_max = 0, yaw_win = 0, travel = 0, pp[2] = {0, 0}, yring[31] = {0};
  int yn = 0;
  int shots = 0, td0 = 0, td_prev = 0;
  // settle time after each bullet from the 2nd: first time after it that the pelvis speed over 0.1 s is below 0.3 m/s
  // once it was above 0.5 [s] (target <= 1.3 s)
  double prg[7][2] = {}, settle[16], stand[16], cand[16];
  bool moved[16] = {};  // the pelvis went above 0.5 m/s after that bullet (the hit moves it later than the chest)
  int prn = 0;
  for (double& x : settle) x = -1;
  for (int k = 0; k < 16; ++k) stand[k] = cand[k] = -1;
  // both feet off the ground over 4 s from the first bullet (target: none):
  // no normal force on either foot / both soles more than 2 cm over the support; the longest such stretch
  double off_c = 0, off_h = 0, off_run = 0, off_long = 0;
  double air0[2] = {-1, -1}, air_max[2] = {0, 0};  // longest time a foot is unloaded while up [s]
  FootSpin fs;  // flat standing foot turn over 0.2 s
  const int frames = (int)std::lround((kMoment + kTail) * 60.0) + 2;
  printf("B3 %d bullets every %.2f s\nB3 t     path   vb  mode st pull  asF  crouch  hold  sl  cpo   sw  step\n", n, gap);
  for (int fr = 0; fr < frames; ++fr) {
    w.StepFrame();
    const double t = d->time;
    const double* P = d->xpos + 3 * (b.body0 + B_PELVIS);
    if (t0 < 0) {
      if (t < kMoment) continue;
      const double* R = d->xmat + 9 * b.body0;
      const double hn = std::hypot(R[0], R[3]) + 1e-9;
      fx = R[0] / hn; fy = R[3] / hn;
      t0 = t; td0 = td_prev = b.td_cnt;
      fs.Rest(w.model(), d, b);
      p0[0] = P[0]; p0[1] = P[1];
      const double* Rq = d->xmat + 9 * (b.body0 + B_PELVIS);
      yaw_prev = std::atan2(Rq[3], Rq[0]);
      pp[0] = P[0]; pp[1] = P[1];
      for (int f = 0; f < 2; ++f) {
        const double* F = d->xpos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        fz[f] = -(F[0] * fx + F[1] * fy);
      }
    }
    const double ts = t - t0;
    if (ts >= 0 && ts < 4.0) {
      const bool nc = b.foot_fn[0] <= 0 && b.foot_fn[1] <= 0;
      const double* m = w.model()->geom_size;
      double zs = 1e9;
      for (int f = 0; f < 2; ++f) {  // sole bottom of the box (lowest corner)
        const int g = b.geom_foot[f];
        const double* R = d->geom_xmat + 9 * g;
        const double ext = std::fabs(R[6]) * m[3 * g] + std::fabs(R[7]) * m[3 * g + 1] + std::fabs(R[8]) * m[3 * g + 2];
        zs = std::fmin(zs, d->geom_xpos[3 * g + 2] - ext - b.gz);
      }
      if (nc) off_c += 1.0 / 60;
      if (zs > 0.02) off_h += 1.0 / 60;
      off_run = nc ? off_run + 1.0 / 60 : 0.0;
      off_long = std::fmax(off_long, off_run);
    }
    for (int f = 0; f < 2; ++f) {  // a foot in the air (unloaded, ankle > 3 cm over standing) while up
      const double az = d->xpos[3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L)) + 2];
      if (!b.loaded[f] && az > b.dims.ankle_h + 0.03 && P[2] > 0.5) {
        if (air0[f] < 0) air0[f] = t;
        air_max[f] = std::fmax(air_max[f], t - air0[f]);
      } else
        air0[f] = -1;
    }
    while (shots < n && ts >= shots * gap - 1e-9) {
      const double dv[3] = {-fx, -fy, 0};
      if (hz > 0) {
        // picked by a ray as in the editor: from 2 m ahead along the start heading at the hit
        // height (x body height / 1.78), lat to the left, flying back; the first body met is hit (belly 1.05 m: the
        // first bullet meets the pelvis shape 18 cm before its centre, the second the lumbar body)
        const double ro[3] = {p0[0] + 2.0 * fx - lat * fy, p0[1] + 2.0 * fy + lat * fx, hz * b.body.height / 1.78};
        int hb = -1;
        double hp[3];
        if (w.Ray(ro, dv, &hb, hp) >= 0) {
          const double* c = d->xipos + 3 * (b.body0 + hb);
          printf("B3 shot %d t %.2f ray hit body %d, %.0f cm before its centre, %.0f cm left, z %.2f\n", shots + 1, ts,
                 hb, 100 * ((hp[0] - c[0]) * fx + (hp[1] - c[1]) * fy), 100 * (-(hp[0] - c[0]) * fy + (hp[1] - c[1]) * fx),
                 hp[2]);
          w.Shoot(0, hb, hp, dv);
        } else {
          printf("B3 shot %d t %.2f ray missed\n", shots + 1, ts);
        }
      } else {
        const double* c = d->xipos + 3 * (b.body0 + B_CHEST);
        const double pt[3] = {c[0] + 0.1 * fx - lat * fy, c[1] + 0.1 * fy + lat * fx, c[2] + 0.05};
        w.Shoot(0, B_CHEST, pt, dv);
      }
      ++shots;
    }
    // B3_PL="t0 t1" per frame along the bullet (+ = further along its flight) [cm, m/s, N]: CoM from the
    // feet middle, CoM speed, floor CoP from the CoM, floor horizontal force, pull on, help, desired / real hip height,
    // knees, ankles dorsi, swing, crouch input
    {
      static double pl0 = -1, pl1 = -1;
      if (pl0 < 0 && getenv("B3_PL")) sscanf(getenv("B3_PL"), "%lf %lf", &pl0, &pl1);
      if (pl0 >= 0 && ts >= pl0 && ts <= pl1) {
        const mjModel* m = w.model();
        const double* cm = d->subtree_com + 3 * b.body0;
        const double* vc = b.pc[b.cur ^ 1].com_vel;  // perception snapshot (subtree_linvel is not computed)
        const double ux = -fx, uy = -fy;
        double fzs = 0, cpx = 0, fh = 0;
        for (int i = 0; i < d->ncon; ++i) {
          const mjContact& c = d->contact[i];
          const int bb1 = m->geom_bodyid[c.geom1], bb2 = m->geom_bodyid[c.geom2];
          const bool in1 = bb1 >= b.body0 && bb1 < b.body0 + B_COUNT, in2 = bb2 >= b.body0 && bb2 < b.body0 + B_COUNT;
          if (in1 == in2) continue;
          double f6[6], F[3];
          mj_contactForce(m, d, i, f6);
          const double sg = in2 ? 1.0 : -1.0;
          for (int k = 0; k < 3; ++k) F[k] = sg * (f6[0] * c.frame[k] + f6[1] * c.frame[3 + k] + f6[2] * c.frame[6 + k]);
          fzs += F[2];
          cpx += F[2] * ((c.pos[0] - cm[0]) * ux + (c.pos[1] - cm[1]) * uy);
          fh += F[0] * ux + F[1] * uy;
        }
        const double* aL = d->xpos + 3 * (b.body0 + B_FOOT_L);
        const double* aR = d->xpos + 3 * (b.body0 + B_FOOT_R);
        const double fm = 0.5 * ((aL[0] + aR[0]) * ux + (aL[1] + aR[1]) * uy);
        const double* q = d->qpos + b.qadr0;
        printf("PL %.3f fL %+5.1f fR %+5.1f ld %.2f %.2f c-f %+5.1f v %5.2f cop %+5.1f Fh %+5.0f Fz %4.0f pull %d help %+5.0f Hd %5.1f hm %5.1f kn %3.0f %3.0f an %3.0f %3.0f sw %2d cr %.3f\n",
               ts, 100 * ((aL[0] - cm[0]) * ux + (aL[1] - cm[1]) * uy), 100 * ((aR[0] - cm[0]) * ux + (aR[1] - cm[1]) * uy),
               b.ld[0], b.ld[1], 100 * ((cm[0] * ux + cm[1] * uy) - fm), vc[0] * ux + vc[1] * uy, fzs > 1 ? 100 * cpx / fzs : 0.0, fh, fzs,
               (int)(d->time < b.pull_until), b.assist_f[0] * ux + b.assist_f[1] * uy, 100 * b.dbg_h[0], 100 * b.dbg_h[1],
               q[J_KNEE_L] * 57.29578, q[J_KNEE_R] * 57.29578, q[J_ANKLE_DORSI_L] * 57.29578, q[J_ANKLE_DORSI_R] * 57.29578,
               b.mode == 1 ? b.swing : -1, b.in.crouch);
      }
    }
    // B3_JK="t0 t1" per frame: the largest joint target jump [deg] (joint), the largest stiffness ratio
    // change, stiffness scale, crouch, pelvis acceleration [m/s2]
    if (getenv("B3_JK")) {
      static double tp[J_COUNT], kp0[J_COUNT], vp[3];
      static bool have = false;
      double a0 = 0, a1 = 0;
      sscanf(getenv("B3_JK"), "%lf %lf", &a0, &a1);
      const double* vv = d->cvel + 6 * (b.body0 + B_PELVIS);  // rotation, translation (com-based frame)
      if (ts >= a0 && ts <= a1 && have) {
        int jm = 0, km = 0;
        double dm = 0, dk = 0;
        for (int j = 0; j < J_COUNT; ++j) {
          const double dj = std::fabs(b.target[j] - tp[j]);
          if (dj > dm) dm = dj, jm = j;
          const double r = std::fabs(b.kp[j] - kp0[j]) / (std::fabs(kp0[j]) + 1e-6);
          if (r > dk) dk = r, km = j;
        }
        const double ac = std::sqrt((vv[3] - vp[0]) * (vv[3] - vp[0]) + (vv[4] - vp[1]) * (vv[4] - vp[1]) +
                                    (vv[5] - vp[2]) * (vv[5] - vp[2])) * 60;
        printf("JK %.3f md %d sw %2d dT %5.1f j%2d dK %4.2f j%2d stiff %.2f tone %.2f cr %.3f acc %5.1f dn %d fp %d dt %.3f dw %.3f\n", ts,
               b.mode, b.swing, 57.3 * dm, jm, dk, km, b.in.stiffness, b.in.tone, b.in.crouch, ac, b.down, b.fall_pred,
               b.dead_t, b.dead_walk_t);
      }
      for (int j = 0; j < J_COUNT; ++j) tp[j] = b.target[j], kp0[j] = b.kp[j];
      for (int k = 0; k < 3; ++k) vp[k] = vv[3 + k];
      have = true;
    }
    // B3_AK="t0 t1" per frame stance-foot heel trace: swing, loads, ankle / toe height over the support [cm],
    // foot pitch [deg, + toe down], ankle dorsi q / target [deg], ankle torque [N m], knee q / target, pelvis rise [cm]
    if (getenv("B3_AK")) {
      double a0 = 0, a1 = 0;
      sscanf(getenv("B3_AK"), "%lf %lf", &a0, &a1);
      if (ts >= a0 && ts <= a1) {
        const double* q = d->qpos + b.qadr0;
        char ln[600];
        int k = snprintf(ln, sizeof ln, "AK %.3f sw %2d md %d ld %.2f %.2f", ts, b.swing, b.mode, b.ld[0], b.ld[1]);
        for (int f = 0; f < 2; ++f) {
          const int g = b.geom_foot[f], ja = f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L, jk = f ? J_KNEE_R : J_KNEE_L;
          const double* R = d->geom_xmat + 9 * g;
          const double* gs = w.model()->geom_size + 3 * g;
          const double* gp = d->geom_xpos + 3 * g;
          const double heel = gp[2] - R[6] * gs[0] - R[8] * gs[2] - b.gz, toe = gp[2] + R[6] * gs[0] - R[8] * gs[2] - b.gz;
          k += snprintf(ln + k, sizeof ln - k, " | %c heel %4.1f toe %4.1f pit %+5.1f an %+5.1f/%+5.1f tq %+5.0f kn %3.0f/%3.0f",
                        f ? 'R' : 'L', 100 * heel, 100 * toe, 57.3 * std::asin(std::fmax(-1.0, std::fmin(1.0, -R[6]))),
                        57.3 * q[ja], 57.3 * b.target[ja], d->ctrl[b.act0 + ja], 57.3 * q[jk], 57.3 * b.target[jk]);
        }
        snprintf(ln + k, sizeof ln - k, " | pz %+5.1f\n", 100 * (P[2] - b.gz - 0.93));
        fputs(ln, stdout);
      }
    }
    {
      double turn = 0;
      const int hf = fs.Step(w.model(), d, b, ts, &turn);
      (void)hf;
      // B3_TR="t0 t1": per frame foot heading (ankle -> toe), loads, travel, pelvis yaw, hip twist q/target/torque,
      // anti-spin torque, swing leg and mode
      static double tr0 = -1, tr1 = -1;
      if (tr0 < 0 && getenv("B3_TR")) sscanf(getenv("B3_TR"), "%lf %lf", &tr0, &tr1);
      if (tr0 >= 0 && ts >= tr0 && ts <= tr1) {
        const mjModel* m = w.model();
        printf("TR %.3f md %d sw %2d", ts, b.mode, b.swing);
        for (int f = 0; f < 2; ++f) {
          double an[3], he[3], to[3];
          FootSpin::Points(m, d, b, f, an, he, to);
          const int jr = f ? J_HIP_ROT_R : J_HIP_ROT_L, bf = b.body0 + (f ? B_FOOT_R : B_FOOT_L);
          printf(" | %c fn %4.0f h %6.1f heel %6.3f %6.3f toe %6.3f %6.3f z %.3f/%.3f rot %5.1f>%5.1f u %5.1f xf %5.1f", "LR"[f],
                 b.foot_fn[f], std::atan2(to[1] - an[1], to[0] - an[0]) * 57.29578, he[0], he[1], to[0], to[1], an[2], to[2],
                 d->qpos[b.qadr0 + jr] * 57.29578, b.target[jr] * 57.29578,
                 d->ctrl[b.act0 + jr] - m->dof_damping[b.dof0 + jr] * d->qvel[b.dof0 + jr], d->xfrc_applied[6 * bf + 5]);
          printf(" fw %.0f %.0f", b.foot_fh[f][0], b.foot_fh[f][1]);  // floor force on the foot, world x y (25 ms)
          printf(" qd %5.2f", d->qvel[b.dof0 + jr]);  // u above = net hip twist torque (motor + joint damping) [N m]
          {  // sole roll (outer edge up +) and ankle inversion q > target [deg]
            const double* Rg = d->geom_xmat + 9 * b.geom_foot[f];
            const int ji = f ? J_ANKLE_INV_R : J_ANKLE_INV_L;
            printf(" roll %5.1f inv %5.1f>%5.1f", std::asin(Rg[7]) * 57.29578 * (f ? -1.0 : 1.0),
                   d->qpos[b.qadr0 + ji] * 57.29578, b.target[ji] * 57.29578);
          }
        }
        const double* Rp = d->xmat + 9 * b.body0;
        const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
        const double* hv = d->subtree_angmom + 3 * b.body0;  // whole body angular momentum about its CoM (z) [N m s]
        mj_subtreeVel(m, d);
        printf(" | pel %6.1f chest %6.1f lyaw %5.1f cyaw %5.1f Lz %5.2f pull %d\n", std::atan2(Rp[3], Rp[0]) * 57.29578,
               std::atan2(Rc[3], Rc[0]) * 57.29578, d->qpos[b.qadr0 + J_LUMBAR_YAW] * 57.29578,
               d->qpos[b.qadr0 + J_CHEST_YAW] * 57.29578, hv[2], (int)(d->time < b.pull_until));
        // TQ: external yaw torques about the whole-body CoM [N m] (frame-end snapshot): ground contacts of the left
        // foot, the right foot and any other body part, anti-spin wrench on the feet, bullet pull, impact twist
        {
          const double* cm = d->subtree_com + 3 * b.body0;
          double cz[3] = {0, 0, 0}, fz[3] = {0, 0, 0};
          for (int i = 0; i < d->ncon; ++i) {
            const mjContact& c = d->contact[i];
            const int bb1 = m->geom_bodyid[c.geom1], bb2 = m->geom_bodyid[c.geom2];
            const bool in1 = bb1 >= b.body0 && bb1 < b.body0 + B_COUNT, in2 = bb2 >= b.body0 && bb2 < b.body0 + B_COUNT;
            if (in1 == in2) continue;  // floor-bot contacts only
            double f6[6], F[3];
            mj_contactForce(m, d, i, f6);
            const double sg = in2 ? 1.0 : -1.0;  // force on geom2 along the normal (geom1 -> geom2)
            for (int k = 0; k < 3; ++k) F[k] = sg * (f6[0] * c.frame[k] + f6[1] * c.frame[3 + k] + f6[2] * c.frame[6 + k]);
            const int g = in2 ? c.geom2 : c.geom1;
            const int slot = g == b.geom_foot[0] ? 0 : g == b.geom_foot[1] ? 1 : 2;
            cz[slot] += (c.pos[0] - cm[0]) * F[1] - (c.pos[1] - cm[1]) * F[0];
            fz[slot] += F[2];
          }
          double xz[2];
          for (int f = 0; f < 2; ++f) {
            const int bf = b.body0 + (f ? B_FOOT_R : B_FOOT_L);
            const double* x = d->xfrc_applied + 6 * bf;
            const double* p = d->xipos + 3 * bf;
            xz[f] = x[5] + (p[0] - cm[0]) * x[1] - (p[1] - cm[1]) * x[0];
          }
          double pz = 0, tz = 0;
          if (d->time < b.pull_until && b.mode != 2 && (b.foot_on[0] || b.foot_on[1])) {
            const double F = w.param[P_BUL_PULL] * b.dims.total_mass * 9.81;
            const double* p = d->xipos + 3 * (b.body0 + B_PELVIS);
            pz = (p[0] - cm[0]) * F * b.pull_dir[1] - (p[1] - cm[1]) * F * b.pull_dir[0];
          }
          static double lz_prev = 0;
          printf("TQ %.3f Lz %5.2f dLz %6.1f | cL %6.1f cR %6.1f cO %6.1f | xL %6.1f xR %6.1f | pull %6.1f twist %6.1f | sum %6.1f"
                 " | fzL %4.0f fzR %4.0f\n", ts, hv[2], (hv[2] - lz_prev) * 60.0, cz[0], cz[1], cz[2], xz[0], xz[1], pz, tz,
                 cz[0] + cz[1] + cz[2] + xz[0] + xz[1] + pz + tz, fz[0], fz[1]);
          lz_prev = hv[2];
          // YB: yaw balance of each loaded foot about the vertical through its CoP [N m]. Hf Ha Ht: vertical
          // part of the torque the pelvis puts on the thigh by hip flexion, abduction, twist (motor + joint damping),
          // Hl: hip joint limits; lev: lever of the floor's horizontal force, ((hip - CoP) x F)_z. A quasi-static leg
          // needs the free moment pred = lev - (Hf + Ha + Ht + Hl) from the floor; free: measured (contact forces about
          // the CoP and torsional moments, tor) plus the anti-spin wrench xs; wz: foot yaw rate [rad/s].
          printf("YB %.3f", ts);
          for (int f = 0; f < 2; ++f) {
            const double* o = d->geom_xpos + 3 * b.geom_foot[f];
            double F[3] = {0, 0, 0}, mo = 0, tor = 0, px = 0, py = 0;
            for (int i = 0; i < d->ncon; ++i) {
              const mjContact& c = d->contact[i];
              if (c.geom1 != b.geom_foot[f] && c.geom2 != b.geom_foot[f]) continue;
              double f6[6], Fc[3];
              mj_contactForce(m, d, i, f6);
              const double sg = c.geom2 == b.geom_foot[f] ? 1.0 : -1.0;
              for (int k = 0; k < 3; ++k) Fc[k] = sg * (f6[0] * c.frame[k] + f6[1] * c.frame[3 + k] + f6[2] * c.frame[6 + k]);
              for (int k = 0; k < 3; ++k) F[k] += Fc[k];
              mo += (c.pos[0] - o[0]) * Fc[1] - (c.pos[1] - o[1]) * Fc[0];
              if (c.dim >= 4) {
                tor += sg * f6[3] * c.frame[2];
                mo += sg * f6[3] * c.frame[2];
              }
              px += c.pos[0] * Fc[2];
              py += c.pos[1] * Fc[2];
            }
            if (F[2] < 30) continue;
            const double cop[2] = {px / F[2], py / F[2]};
            const double free_c = mo - ((cop[0] - o[0]) * F[1] - (cop[1] - o[1]) * F[0]);
            const int bf = b.body0 + (f ? B_FOOT_R : B_FOOT_L);
            const double* x = d->xfrc_applied + 6 * bf;
            const double* xp = d->xipos + 3 * bf;
            const double xs = x[5] + (xp[0] - cop[0]) * x[1] - (xp[1] - cop[1]) * x[0];
            const int j0 = f ? J_HIP_FLEX_R : J_HIP_FLEX_L;
            const double* hp = d->xanchor + 3 * (b.jnt0 + j0);
            const double lev = (hp[0] - cop[0]) * F[1] - (hp[1] - cop[1]) * F[0];
            double H[4] = {0, 0, 0, 0};
            for (int jl = 0; jl < 3; ++jl) {
              const int j = j0 + jl, dof = b.dof0 + j, jid = b.jnt0 + j;
              const double az = d->xaxis[3 * jid + 2];
              H[jl] = (d->qfrc_actuator[dof] - m->dof_damping[dof] * d->qvel[dof]) * az;
              const double mid = 0.5 * (m->jnt_range[2 * jid] + m->jnt_range[2 * jid + 1]);
              for (int e = 0; e < d->nefc; ++e)
                if (d->efc_type[e] == mjCNSTR_LIMIT_JOINT && d->efc_id[e] == jid)
                  H[3] += (d->qpos[m->jnt_qposadr[jid]] < mid ? 1.0 : -1.0) * d->efc_force[e] * az;
            }
            // ac: yaw moment of the stay-upright pair (pelvis +F at its CoM, feet -F share at theirs) about this CoP
            double ac = 0;
            {
              const int bl[3] = {B_PELVIS, B_FOOT_L, B_FOOT_R};
              for (int q = 0; q < 3; ++q) {
                const double* xq = d->xfrc_applied + 6 * (b.body0 + bl[q]);
                const double* pq = d->xipos + 3 * (b.body0 + bl[q]);
                ac += (pq[0] - cop[0]) * xq[1] - (pq[1] - cop[1]) * xq[0];
              }
            }
            printf(" | %c fn %4.0f Hf %6.1f Ha %6.1f Ht %6.1f Hl %6.1f lev %6.1f pred %6.1f free %6.1f tor %6.1f xs %6.1f"
                   " wz %5.2f ac %6.1f asF %4.0f", "LR"[f], F[2], H[0], H[1], H[2], H[3], lev, lev - H[0] - H[1] - H[2] - H[3],
                   free_c + xs, tor, xs, d->cvel[6 * bf + 2], ac, std::hypot(b.assist_f[0], b.assist_f[1]));
          }
          printf("\n");
          // CM: CoM and its velocity (world), feet sole centres, swing flag, ankle inversion torque (net) per foot
          const double* vc = d->subtree_linvel + 3 * b.body0;
          // sagittal joints per foot: ankle dorsi, knee, hip flex q > target [deg] and net ankle torque [N m];
          // pelvis and chest forward tilt (body z axis toward world x) [deg]
          printf("SG %.3f", ts);
          for (int f = 0; f < 2; ++f) {
            const int ja = f ? J_ANKLE_DORSI_R : J_ANKLE_DORSI_L, jk = f ? J_KNEE_R : J_KNEE_L;
            const int jh = f ? J_HIP_FLEX_R : J_HIP_FLEX_L;
            printf(" | %c an %5.1f>%5.1f %5.0f kn %5.1f>%5.1f hp %5.1f>%5.1f", "LR"[f], d->qpos[b.qadr0 + ja] * 57.29578,
                   b.target[ja] * 57.29578, d->ctrl[b.act0 + ja] - m->dof_damping[b.dof0 + ja] * d->qvel[b.dof0 + ja],
                   d->qpos[b.qadr0 + jk] * 57.29578, b.target[jk] * 57.29578, d->qpos[b.qadr0 + jh] * 57.29578,
                   b.target[jh] * 57.29578);
          }
          {
            const double* Rp = d->xmat + 9 * b.body0;
            const double* Rc = d->xmat + 9 * (b.body0 + B_CHEST);
            printf(" | pp %5.1f cp %5.1f pz %.3f\n", std::atan2(Rp[2], Rp[8]) * 57.29578, std::atan2(Rc[2], Rc[8]) * 57.29578,
                   d->xpos[3 * b.body0 + 2]);
          }
          // swing target and prediction point (world x y), help force on the pelvis (world x y)
          printf("CX %.3f tg %6.3f %6.3f pt %6.3f %6.3f asf %5.0f %5.0f pl %6.3f %6.3f\n", ts, b.sw_tg[0], b.sw_tg[1], b.pt[0],
                 b.pt[1], b.assist_f[0], b.assist_f[1], b.place[b.swing >= 0 ? b.swing : 0][0], b.place[b.swing >= 0 ? b.swing : 0][1]);
          printf("CM %.3f com %6.3f %6.3f v %5.2f %5.2f | L %6.3f %6.3f | R %6.3f %6.3f | invL %5.1f invR %5.1f\n", ts, cm[0],
                 cm[1], vc[0], vc[1], d->geom_xpos[3 * b.geom_foot[0]], d->geom_xpos[3 * b.geom_foot[0] + 1],
                 d->geom_xpos[3 * b.geom_foot[1]], d->geom_xpos[3 * b.geom_foot[1] + 1],
                 d->ctrl[b.act0 + J_ANKLE_INV_L] - m->dof_damping[b.dof0 + J_ANKLE_INV_L] * d->qvel[b.dof0 + J_ANKLE_INV_L],
                 d->ctrl[b.act0 + J_ANKLE_INV_R] - m->dof_damping[b.dof0 + J_ANKLE_INV_R] * d->qvel[b.dof0 + J_ANKLE_INV_R]);
        }
      }
      for (int f = 0; f < 2; ++f)
        if (getenv("B3_SPIN") && fs.last_turn[f] > FootSpin::kTurn)
          printf("SPIN %.2f foot %c turned %.1f deg in 0.2 s (sole axis %.1f), heel/toe moved %.1f cm\n", ts, "LR"[f],
                 fs.last_turn[f], fs.last_ax[f], 100 * fs.last_slide[f]);
    }
    const double path = std::hypot(P[0] - p0[0], P[1] - p0[1]);
    path_max = std::fmax(path_max, path);
    if (ts >= 0) {
      prg[prn % 7][0] = P[0];
      prg[prn % 7][1] = P[1];
      if (prn >= 6) {
        const double* q = prg[(prn - 6) % 7];
        const double vp = std::hypot(P[0] - q[0], P[1] - q[1]) * 10.0;  // 6 frames = 0.1 s
        for (int k = 1; k < shots && k < 16; ++k)
          if (ts >= k * gap) {
            if (vp > 0.5) moved[k] = true;
            if (settle[k] < 0 && moved[k] && vp < 0.3) settle[k] = ts - k * gap;
            // stands: below 0.3 m/s and staying below for 0.25 s (or until the next bullet)
            if (stand[k] < 0 && moved[k]) {
              if (vp >= 0.3) {
                cand[k] = -1;
              } else {
                if (cand[k] < 0) cand[k] = ts;
                if (ts - cand[k] >= 0.25 - 1e-9 || (k + 1 < shots && ts >= (k + 1) * gap)) stand[k] = cand[k] - k * gap;
              }
            }
          }
      }
      ++prn;
    }
    {
      const double* Rq = d->xmat + 9 * (b.body0 + B_PELVIS);
      const double yw = std::atan2(Rq[3], Rq[0]);
      double dyw = yw - yaw_prev;
      while (dyw > M_PI) dyw -= 2 * M_PI;
      while (dyw < -M_PI) dyw += 2 * M_PI;
      yaw_u += dyw;
      yaw_prev = yw;
      yaw_max = std::fmax(yaw_max, std::fabs(yaw_u));
      yring[yn % 31] = yaw_u;
      if (yn >= 30) yaw_win = std::fmax(yaw_win, std::fabs(yaw_u - yring[(yn - 30) % 31]));
      ++yn;
      travel += std::hypot(P[0] - pp[0], P[1] - pp[1]);
      pp[0] = P[0]; pp[1] = P[1];
    }
    const Percept& pc = b.pc[b.cur ^ 1];
    const double vb = -(pc.com_vel[0] * fx + pc.com_vel[1] * fy);
    static double step_len = 0;
    static double pl_rel = 0, tg_rel = 0;
    if (b.mode == 1 && b.swing >= 0) {  // place and the reach-clipped swing target behind the CoM along the bullet [cm]
      const double cz0 = -(pc.com[0] * fx + pc.com[1] * fy);
      pl_rel = 100 * (-(b.place[b.swing][0] * fx + b.place[b.swing][1] * fy) - cz0);
      tg_rel = 100 * (-(b.sw_tg[0] * fx + b.sw_tg[1] * fy) - cz0);
    }
    if (b.td_cnt != td_prev) {  // a touchdown: travel of the foot that landed along the bullet since its last touchdown
      td_prev = b.td_cnt;
      for (int f = 0; f < 2; ++f) {
        const double* F = d->xpos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        const double z = -(F[0] * fx + F[1] * fy);
        if (std::fabs(z - fz[f]) > 0.02) { step_len = 100 * (z - fz[f]); fz[f] = z; }
      }
      // touchdown line: both feet behind the CoM along the bullet [cm], CoM speed along it
      const double cz = -(pc.com[0] * fx + pc.com[1] * fy);
      double rel[2];
      for (int f = 0; f < 2; ++f) {
        const double* F = d->xpos + 3 * (b.body0 + (f ? B_FOOT_R : B_FOOT_L));
        rel[f] = 100 * (-(F[0] * fx + F[1] * fy) - cz);
      }
      if (getenv("B3_TD"))
        printf("TD %.2f feet behind the CoM L %+4.0f R %+4.0f vb %.2f; last frame of the swing: place %+4.0f target %+4.0f\n", ts, rel[0], rel[1], vb, pl_rel, tg_rel);
    }
    const int k = (int)std::lround(ts * 60);
    if (k % 6 == 0) {
      const bool pull = t < b.pull_until;
      printf("B3 %4.1f %5.0f %5.2f %4d %3d %3d %5.0f %6.2f %6.3f %3d %5.2f %3d %5.0f\n", ts, 100 * path, vb, b.mode,
             b.td_cnt - td0, (int)pull, -(b.assist_f[0] * fx + b.assist_f[1] * fy), b.in.crouch, b.hold_lean,
             b.in.steps_left, b.cp_out, b.mode == 1 ? b.swing : -1, step_len);
    }
  }
  printf("B3 flat foot turn in 0.2 s: worst %.1f deg at %.2f s (%c, heel/toe moved %.1f cm); in place (< 3 cm) worst %.1f "
         "at %.2f s, %d frames over 5 deg\n", fs.worst, fs.worst_t, fs.worst_f >= 0 ? "LR"[fs.worst_f] : '-',
         100 * fs.worst_slide, fs.worst_ip, fs.worst_ip_t, fs.frames_ip);
  printf("B3 sole axis turn in 0.2 s: worst %.1f deg, in place %.1f deg\n", fs.worst_ax, fs.worst_ax_ip);
  printf("B3 loaded sole roll: worst %.1f deg, frames > 8 deg %d, ankle inversion at range end %d\n", fs.roll_max,
         fs.roll_frames, fs.lim_frames);
  printf("B3 path max %.0f cm, steps %d\n", 100 * path_max, b.td_cnt - td0);
  printf("B3 settle after bullets 2..%d (pelvis < 0.3 m/s over 0.1 s) [s]:", shots);
  for (int k = 1; k < shots && k < 16; ++k) printf(" %.2f", settle[k]);
  printf("\n");
  printf("B3 stands after bullets 2..%d (pelvis < 0.3 m/s over 0.1 s, staying 0.25 s or to the next bullet) [s]:", shots);
  for (int k = 1; k < shots && k < 16; ++k) printf(" %.2f", stand[k]);
  printf("\n");
  printf("B3 pelvis turn max %.0f deg, in 0.5 s max %.0f deg, travel %.0f cm\n", yaw_max * 57.29578,
         yaw_win * 57.29578, 100 * travel);
  printf("B3 foot in the air longest: L %.2f s, R %.2f s\n", air_max[0], air_max[1]);
  printf("B3 both feet off 0-4 s: no contact %.2f s (longest %.2f), soles over 2 cm %.2f s\n", off_c, off_long, off_h);
  return 0;
}
