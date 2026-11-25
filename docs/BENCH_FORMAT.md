# Bench commands and output formats

All runs are headless and deterministic for a given build. Output files go under `$EMIL_TMP` (default `/tmp/emil`).
Any command takes parameter overrides `-p name=value` (names in `core/emil_params.h`).

## Main commands
- `emil_bench`: stand scene of 10 s (rest metrics: stance width, sway, pelvis position), timing for 1 and 8 bots,
  allocation count. `--poses FILE` writes all body poses (CSV: time, then x, y, z, qw, qx, qy, qz per body).
- `emil_bench selftest`: joint axes and limits, leg IK, sign conventions; prints `selftest: N errors`.
- `emil_bench push <label> [threads]`: push set, 20 scenes x 6 push times (below). Prints a table per push time
  and `all`, CSV per run in `$EMIL_TMP/pk/<label>/`.
- `emil_bench push1 <scene> <moment>`: one traced push run; CSV in `$EMIL_TMP/pk/one/` plus `dbg.csv` (t, com_x/y/z,
  xi_x/y capture point, tgt_x/y step target, mode, swing, phase) and `yaw.csv` (headings, thigh flexion and rotation,
  foot torques and contact data per frame). Frame sheet: `python3 -I tools/plot_push.py RUN.csv DBG.csv OUT.png 0 2`.
- `emil_bench rx <label> [threads]`: reaction set (bullets, falls, drops, car, curl). One table row per scene: fell,
  swings, drift, settle time, first body to touch the floor, head speed, and the timing columns t_m2 (give-up),
  t_1st (first floor touch), t_head. Prefixed tables: `BUL` (bullet reactions), `DTH` (deaths). CSV in
  `$EMIL_TMP/rx/<label>/`.
- `emil_bench stagger [--scale k] [--moments a,b,c]`: 20 push scenes grouped by direction and size, step timing
  and lengths, trunk lean and drop next to the target table, stood / no-turn counts. Extra lines: `YAW` (pelvis yaw
  per step), `ARM` (arms down after rest), `REST` (rest pose), `SHO` (largest shoulder angles).
- `emil_bench death8`: body deaths (head shots, kill, bullet series, pushes into death): pelvis height over time,
  give-up, impact speeds of pelvis, chest and head.
- `emil_bench leg8`: a bullet into a leg: trunk, knees, pelvis drop and sideways travel, hand on the wound, steps.
- `emil_bench avoid`: car dodges (cars at 18 / 30 / 50 km/h from the front and the side, 2.0 / 1.3 / 0.9 / 0.4 s to
  the hit) and near-miss bullets; `AV_SEED`, `AV_ONLY`, `AV_TRACE` environment variables.
- `emil_bench carcall <kmh> <side> <start m> [wait s]`: one editor-style car call with a trace of the CoM and feet
  distance from the car line and the first contact.
- Others (see `bench/main.cpp`): `push1s`, `shift`, `cop`, `torsion`, `cstat`, `ab`, `upper`, `upper6`, `bul3`,
  `bul7`, `fall9`, `drop9`, `twist8`, `sway`, `stairs`, `creep`, `opt`, `sweep`, `steps`, `speed`, `inertia`.

## Push set
- Scenes: `back30 back45 back60 back80 back100 back120 fwd30 fwd45 fwd60 fwd80 fwd100 left30 left45 left60 left80
  left100 left120 right45 right60 right100` (direction and impulse in N s); push times `1.5 2.0 2.5 3.0 3.5 4.0` s.
- Push: horizontal impulse at the standard push point (the centre of mass of the body above the hip joints), applied
  to the nearest trunk segment; a hand push is spread over at least `push_t` and at most `push_fmax` of force.
- Run CSV `$EMIL_TMP/pk/<label>/<scene>@<moment>.csv`: first line column names, then one row per 60 Hz frame from
  0.5 s before the push to 10 s after. Columns: `t` (s, 0 = the push frame), then `_x,_y,_z` (m, world: X forward
  from the bot, Y left, Z up) of `pelvis` (pelvis body origin), `head` (head centre of mass), `l_ankle`, `r_ankle`,
  `l_toe`, `r_toe` (front bottom edge centre of the foot box), `l_heel`, `r_heel` (back bottom edge centre); `l_roll`,
  `r_roll` (foot roll, deg, + = outer edge up); `l_contact`, `r_contact` (0/1).
- `summary.txt` in the same directory: one line per run (fell, swings, same, short, late, edge, yaw, chest, flex).

## Push metrics
- swing: toe and ankle both above their rest heights (at the push frame) by 0.015 m for at least 3 frames; it starts
  at the first frame of that run.
- same: a swing of the same foot as the previous swing. short: toe displacement over the swing below 0.25 m (an
  unfinished swing counts by its displacement so far). late: swing start more than 1.2 s after the push.
- edge: from 2 s after the push, every 0.25 s, each foot in contact with a roll over 8 deg (share of samples).
- fell: pelvis below 0.5 m or a trunk or head body touching the floor.
- yaw, chest, flex: largest pelvis heading change, chest-pelvis heading difference and thigh flexion [deg] until the
  give-up or 2 s after the push; the last column counts runs with yaw <= 30, chest <= 15, flex <= 60.
- The push table also prints the share of frames by stance-foot contact count (flat >= 3 contacts, edge 2, corner 1,
  none) for quiet stand, single and double support.
