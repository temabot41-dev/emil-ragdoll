// Foot twisting test: a foot box under a load on the floor, no balancer. The foot body can slide in x, y, z and
// turn about the vertical only; a vertical torque is stepped up until the foot spins. Reports the breakaway torque for
// several loads and contact layouts (flat, lateral edge, toe edge, heel edge), for the model's foot contact settings.
//   emil_bench torsion [condim] [mu_torsion] [mu_roll]     (defaults: 3 0.005 0.0001 = current world)
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "emil_body.h"
#include "emil_world.h"

namespace {

struct Layout { const char* name; double roll, pitch; };  // tilt of the foot box [rad] about its x and y axes

// Breakaway torque [N m] of one foot under mass m [kg]; also returns the contact count at rest.
double Breakaway(int condim, double mu_t, double mu_r, double fl, double fw, double fh, double mass, const Layout& lay,
                 int* ncon, double hold_s, double mu) {
  char xml[2048];
  const double dt = emil::World::kFrameDt / emil::World::kSubsteps;
  // The box is tilted inside the body (geom euler), so the body keeps only yaw freedom and the contact layout is fixed.
  snprintf(xml, sizeof xml,
           "<mujoco><compiler angle='radian'/><option timestep='%.8f' integrator='Euler' cone='pyramidal' solver='Newton' "
           "iterations='20' ls_iterations='8'/><default><joint armature='0.01' damping='0'/>"
           "<geom friction='%g %g %g' condim='%d' solref='0.01 1'/></default><worldbody>"
           "<geom name='floor' type='plane' size='5 5 0.1'/>"
           "<body name='foot' pos='0 0 0.2'><joint name='sx' type='slide' axis='1 0 0'/><joint name='sy' type='slide' axis='0 1 0'/>"
           "<joint name='sz' type='slide' axis='0 0 1'/><joint name='yaw' type='hinge' axis='0 0 1'/>"
           "<inertial pos='0 0 0.1' mass='%g' diaginertia='0.5 0.5 0.1'/>"
           "<geom type='box' size='%g %g %g' euler='%g %g 0'/></body></worldbody></mujoco>",
           dt, mu, mu_t, mu_r, condim, mass, 0.5 * fl, 0.5 * fw, 0.5 * fh, lay.roll, lay.pitch);
  char err[512] = "";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "t.xml", xml, (int)std::strlen(xml));
  mjModel* m = mj_loadXML("t.xml", &vfs, err, sizeof err);
  mj_deleteVFS(&vfs);
  if (!m) { fprintf(stderr, "xml: %s\n", err); return -1; }
  mjData* d = mj_makeData(m);
  // settle: let the foot sink to the floor, no torque
  for (int i = 0; i < 480; ++i) mj_step(m, d);
  *ncon = d->ncon;
  const int yaw = m->jnt_dofadr[3];
  double breakaway = -1, T = 0;
  for (T = 1.0; T <= 150.0; T += 1.0) {  // stepped torque, each level held hold_s
    const double y0 = d->qpos[m->jnt_qposadr[3]];
    const int n = (int)(hold_s / dt);
    for (int i = 0; i < n; ++i) {
      d->qfrc_applied[yaw] = T;
      mj_step(m, d);
    }
    if (std::fabs(d->qpos[m->jnt_qposadr[3]] - y0) > 0.1) { breakaway = T; break; }  // > 6 deg per level
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  return breakaway;
}

}  // namespace

int RunTorsionBench(int argc, char** argv) {
  const int condim = argc >= 3 ? std::atoi(argv[2]) : 3;
  const double mu_t = argc >= 4 ? std::atof(argv[3]) : 0.005, mu_r = argc >= 5 ? std::atof(argv[4]) : 0.0001;
  const double mu = argc >= 6 ? std::atof(argv[5]) : 1.0;
  // Foot of the 1.78 m body: length 0.152 H, width 0.38 length (de Leva 1996 / emil_body), box height = ankle height.
  const double H = 1.78, fl = 0.152 * H, fw = 0.38 * fl, fh = 0.039 * H;
  const Layout lays[] = {{"flat", 0, 0}, {"lat edge", 0.15, 0}, {"toe edge", 0, 0.15}, {"heel edge", 0, -0.15}, {"corner", 0.15, 0.15}};
  printf("foot torsion test: condim %d, mu %.2f, mu_torsion %.4f, mu_roll %.4f; foot %.3f x %.3f m\n", condim, mu, mu_t,
         mu_r, fl, fw);
  printf("breakaway torque [N m] (contacts at rest), step 1 N m per 0.25 s\n%-10s", "load N");
  for (const Layout& l : lays) printf(" %12s", l.name);
  printf("\n");
  for (double N : {200.0, 400.0, 800.0, 1000.0}) {
    printf("%-10.0f", N);
    for (const Layout& l : lays) {
      int nc = 0;
      const double b = Breakaway(condim, mu_t, mu_r, fl, fw, fh, N / 9.81, l, &nc, 0.25, mu);
      printf("   %5.0f (%d)  ", b, nc);
    }
    printf("\n");
  }
  return 0;
}
