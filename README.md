# EMIL: active ragdoll for Godot 4 (C++17, MuJoCo)

Video: [EMIL 1.0 on YouTube](https://youtu.be/uTP37frIiLM)

EMIL is a physics-based active-ragdoll character core for games, written in C++17 on top of MuJoCo 3.17.0. The body
moves only through joint torques (muscles); external forces come only from the world (floor, hits, bullets, vehicles).
The core keeps the character standing, steps to recover from pushes, and reacts to hits, trips, drops and cars.

**Status: a very raw, early version.** Much of it works poorly and much is missing: no contacts with the Godot
scene yet (the core only knows its floor and a few built-in test props), no partial ragdoll, no transitions
between animation and physics, no getting up, no walking on command. It has never been used in a real game level.
Details and a plan for adding scene contacts: `docs/INTEGRATION.md`.

The behaviour targets (how a body should react) were set by observing and measuring motion from the outside;
EMIL contains no code or data of any other character controller. Physical values come from the open literature
(`docs/SOURCES.md`) and from tuning on the benches in `bench/`.

All code in this repository was written by Claude, an AI model by Anthropic, under the direction of TEMIR.

Documentation: `docs/ARCHITECTURE.md` (code structure), `docs/BENCH_FORMAT.md` (bench output), `docs/EDITOR.md`
(editor), `docs/INTEGRATION.md` (status, what is missing, how to add scene contacts), `docs/SOURCES.md` (literature and data sources). License: `LICENSE`; third-party components:
`THIRD_PARTY_NOTICES.md`.

## Stay-upright assist
The body is moved by joint torques only, with one deliberate exception: an optional "stay upright" pair force between
the pelvis and the loaded feet (sums to zero, horizontal, active only with a loaded foot, fades to zero at 45 deg body
tilt, off when fallen). Parameter `assist` (editor slider; 0 = off, default 0.3).

## Key physical values and where they come from
- Standard push point: the centre of mass of the body above the hip joints (1.26 m in a 1.78 m body); a push through
  it moves the upper body over the hips without turning it.
- Pistol bullet impulse `bul_imp` 29 N s: 5/6 of the impulse at which one chest hit makes the standing body take its
  first step on the bench (35 N s), so one hit rocks the body without a step.
- Strength fade: lethal 0.35 s (the shortest fade at which the pelvis falls fastest), wounded 4 x that (1.4 s).
- Bullet pull `bul_pull` 0.31 x weight for 0.65-1.15 s per hit: about 0.9 s (ends before the next hit of a slow series), strength
  fitted to the bullet travel of the standing body (3 hits 1 s apart about 2.75 m, 2 hits about 1.2 m).

## Layout
- `core/`: library `emil` (C++17, depends only on MuJoCo 3.17.0): body model, muscles (L0), balance (L1),
  reactions (L2), perception (L3), world and frame loop, parameter table (`core/emil_params.h`).
- `bench/`: headless bench `emil_bench`: standing, timing, selftest, push set, reaction set, traced single runs.
- `godot/`: GDExtension `EmilWorld` (Godot 4.7.2, prebuilt godot-cpp); `editor/`: the EMIL Editor app.
- `tools/`: dependency setup and check scripts, Android toolchain, APK build, minSdk patch, plotting and trace helpers.
- `third_party/`: MuJoCo build patch and build script, MuJoCo notice.

## Dependencies
On a fresh Ubuntu x86_64 machine:
```
bash tools/setup_deps.sh   # toolchain, MuJoCo 3.17.0 source + third_party/mujoco-3.17.0-emil-static.patch, static libs without LTO
bash tools/check.sh   # build, selftest, push + reaction sets; prints summary lines
```
Environment: `EMIL_DEPS` (dependency root, CMake reads it too; `setup_deps.sh` defaults to `$HOME/.cache/emil_deps`,
CMake to `$HOME/.cache/emil_deps` as well), `EMIL_TMP` (bench output and traces, default `/tmp/emil`), `JOBS` (build and bench
threads). If GitHub is unreachable, `setup_deps.sh` falls back to the shared library of the official PyPI wheel and
`check.sh` passes it to CMake as `MUJOCO_SO`.

On other hosts, `third_party/build_mujoco.sh` builds the MuJoCo static libraries.

## Build
Linux:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEMIL_DEPS=$EMIL_DEPS && cmake --build build -j$(nproc)
```
The MuJoCo static libraries are expected in `$EMIL_DEPS/mujoco-3.17.0/build-linux-nolto/lib` (built without LTO, so a
bench relink takes seconds; override with `-DMUJOCO_LIBDIR=...`).

Android arm64 (API 28), cross-compiled with the NDK r27c sysroot (`ANDROID_NDK`):
```
cmake -S . -B build-android -DCMAKE_TOOLCHAIN_FILE=tools/android-arm64.cmake \
  -DMUJOCO_LIBDIR=$EMIL_DEPS/mujoco-3.17.0/build-android/lib && cmake --build build-android -j8
```

## Bench (`./build/emil_bench ...`)
| command | what |
|---|---|
| (none) `[--poses file.csv]` | stand 10 s (rest metrics), ms per frame for 1 and 8 bots, contacts, allocations |
| `selftest` | joint axes and limits, leg IK, virtual force J^T sign, yaw sign (prints `selftest: N errors`) |
| `push <label> [threads]` | 20 scenes x 6 moments: stood, swings, same, short, late, edge; CSV in `$EMIL_TMP/pk/<label>/` |
| `push1 <scene> <moment>` | one traced run, per-frame balancer state in `$EMIL_TMP/pk/one/dbg.csv` |
| `rx <label> [threads]` | bullets, falls, drops (height), car, curl; table + CSV in `$EMIL_TMP/rx/<label>/` |
| `stagger`, `death8`, `sway`, `avoid`, ... | further scenario benches (see `bench/main.cpp`) |
| `-p name=value` | parameter override (any command), names in `core/emil_params.h` |

Helpers:
```
python3 -I tools/push_dirs.py <label>                 # push results per direction
python3 -I tools/trace_dbg.py <t0> <t1> <k> <cols>    # columns of the push1 trace
python3 -I tools/touchdowns.py <name> [x|y]           # touchdowns of the push1 trace
python3 -I tools/plot_push.py RUN.csv [DBG.csv] OUT.png [t0 t1]   # frame sheet of a push run (PIL)
```

## Editor
Godot 4.7.2 project in `editor/`: a bot on a floor with an orbit camera, Material 3 UI, tools (push, bullet, trip,
drop height, car, commanded steps), parameter sliders, reset, x0.25 slow motion, pause, fps and core timing.
See `docs/EDITOR.md`.

Build the GDExtension and the APK (`com.temir.emil.editor`, minSdk 28, arm64-v8a, debug key):
```
sh tools/make_apk.sh [tuned.params]   # Linux + Android extension libs, patched template, build-apk/emil_editor.apk
adb install -r build-apk/emil_editor.apk
```
The script reads `EMIL_DEPS`, `GODOT` (editor binary) and `TPL` (Android debug export template). Headless editor
self-test: `godot --headless --path editor --fixed-fps 60 --quit-after 2400 -- --selftest`.

## License
EMIL is released under the MIT License (`LICENSE`), copyright (c) 2026 TEMIR. Binaries also contain
MuJoCo (Apache-2.0) and other third-party components listed with their license texts in `THIRD_PARTY_NOTICES.md`.

## Keywords
active ragdoll, ragdoll physics, physics-based character animation, procedural animation, character controller,
balance control, push recovery, stepping, hit and bullet reactions, death falls, biped simulation, Godot 4, GDExtension,
MuJoCo, C++17, Android, game physics, physically simulated characters.
