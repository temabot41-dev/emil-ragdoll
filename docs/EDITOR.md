# EMIL Editor and the Godot module

## Module (`godot/src/emil_godot.cpp`)
GDExtension built by the root CMake with `-DEMIL_GODOT=ON` (core linked statically, `-static-libstdc++`), against a
prebuilt godot-cpp static library (template_debug). Outputs go to `editor/bin/` (`libemil.linux.arm64.so`,
`libemil.android.arm64.so`), entry symbol `emil_library_init`.

Node `EmilWorld`:
- properties: `bot_count` (1..8), `height`, `mass`, `build`, `sizes` (0 de Leva proportions, 1 alternative character
  proportions; the body properties apply on `rebuild()`), `paused`, `time_scale`;
- `rebuild()`, `reset()`, `step_frame()`; `_physics_process` steps the core at 60 Hz x `time_scale` (at most 3 frames
  per tick);
- commands:
  - `push(bot, body, point, dir, impulse)` at the given point;
  - `push_std(...)` at the standard push point (the centre of mass of the body above the hip joints);
  - `shoot(bot, body, point, dir)` (pistol bullet, impulse `bul_imp`);
  - `trip(bot)`, `drop_from(height)`, `car(speed_kmh, side, start = 4.0)` (side 0 from the front, +1 left, -1 right;
    start = distance of the car centre from the pelvis [m]);
  - `step(bot, foot, dx, dy, cross)`, `shift(bot, foot, lift_s)` (commanded step and weight shift), `curl(bot)`,
    `kill(bot)`;
- `pick(from, dir)` -> `{bot, body, point}` by ray cast (floor excluded);
- drawing: `get_buffer(kind)` (MultiMesh TRANSFORM_3D buffer, 12 floats per instance), `get_count(kind)`,
  `get_mesh_tris(which)`;
- parameters: `set_param` / `get_param(name)`, `get_param_list()` (the table of `core/emil_params.h`);
- state: `get_core_ms()` (smoothed time of one core frame), `get_sim_time()`, `get_bot_pos(bot)`, `get_bot_mode(bot)`,
  `get_health(bot)`;
- axes: Godot (x, y, z) = MuJoCo (x, z, -y).

## Editor (`editor/`, Godot 4.7.2)
`main.gd` builds the scene in code: floor with a 1 m grid, the bots as MultiMeshes, orbit camera (one-finger drag,
pinch or mouse-wheel zoom), Material 3 light theme.
- Tools (bottom): Толчок (push), Пуля (bullet; Не умирает = immortal switch, Убить = kill), Подножка (trip),
  Высота (drop from 1 / 2 / 4 m), Машина (cars at 18 / 50 km/h from the left or right, and dodge cars at 35 km/h from
  the front or the left: calm, hurried and late, i.e. 2.0 / 1.3 / 0.9 s to the hit), Шаги (commanded steps). A tap
  on the bot sets the hit point of the push and the bullet.
- Top right: Сброс (reset), x0.25 (slow motion), Пауза (pause), Настройки (settings panel: body height, mass, build
  and proportions with "Применить тело", balance parameter sliders, "По умолчанию" for the defaults).
- Top left: fps, core ms per frame, simulated time.
- Log lines: `EMIL: module loaded`, `EMIL: t=... s pelvis_y=... mode=...` (standing: pelvis about 0.93 m, mode 0).

## Steps tool
Upper row: foot choice (left / right / alternate) and the parameter-set switch. Lower row:
- "Вес на левую" / "Вес на правую" (`shift`, weight onto one foot, no lift);
- "На одну ногу" (`shift` with a 3 s lift of the other foot);
- "Шаг вперёд / назад / вбок" (`step`, 0.25 / -0.25 / 0.20 m);
- "Накрест" (`step`, crossover: 0.10 m forward, 0.34 m across).

All start from a calm stance and run through the normal step machine.

"Числа: обычные / подобранные" applies the parameters of `res://params/best.params` (`name=value` lines, written by
the opt bench) to the core and respawns the bot. Without the file the button is inactive. The file is packed by
`sh tools/make_apk.sh path/to/best.params`.

## Build, export, self-test
`sh tools/make_apk.sh` builds the Linux and Android libraries, the minSdk-28 template and the APK. A non-Gradle export
refuses a Min SDK override, so `tools/patch_minsdk.py` patches the template's binary manifest (24 -> 28) into
`build-apk/android_debug_min28.apk`, used as the custom debug template (the script fills the path into the export
preset for the export and restores it). The preset holds no keystore: the APK is signed with the Godot editor's own
debug keystore (editor settings). Install with
`adb install -r build-apk/emil_editor.apk`.

Headless self-test: `godot --headless --path editor --fixed-fps 60 --quit-after 2400 -- --selftest`. It prints
`EMIL selftest: ...` lines (pick, push, bullet, trip, car, drop, curl, parameter round trip, rebuild, steps, burst,
kill).
