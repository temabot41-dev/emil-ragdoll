#!/bin/sh
# Build the EMIL Editor APK: GDExtension for Linux and Android arm64, minSdk-28 template, export.
# Run from the repo root. Output: build-apk/emil_editor.apk. Install is a separate step.
set -e
# Optional: sh tools/make_apk.sh [tuned.params]  - a tuned parameter file (name=value lines, e.g. a best.params written
# by the opt bench) is packed as res://params/best.params and offered by the "Числа" switch of the Steps tool;
# without it the switch is inactive.
PARAMS_FILE="$1"
mkdir -p editor/params
if [ -n "$PARAMS_FILE" ]; then cp "$PARAMS_FILE" editor/params/best.params; else rm -f editor/params/best.params; fi
DEPS=${EMIL_DEPS:-$HOME/.cache/emil_deps}  # all of these default to the arm64 build host layout
GODOT=${GODOT:-$HOME/godot/Godot_v4.7.2-stable_linux.arm64}
TPL=${TPL:-$HOME/.local/share/godot/export_templates/4.7.2.stable/android_debug.apk}
MJ_ANDROID=$DEPS/mujoco-3.17.0/build-android/lib
MJ_LINUX=$DEPS/mujoco-3.17.0/build-linux-nolto/lib  # no LTO: fast relink of the editor extension
cmake -S . -B build-godot -DEMIL_GODOT=ON -DMUJOCO_LIBDIR=$MJ_LINUX >/dev/null
cmake --build build-godot -j8 --target emil_godot
cmake -S . -B build-godot-android -DEMIL_GODOT=ON -DCMAKE_TOOLCHAIN_FILE=tools/android-arm64.cmake \
  -DMUJOCO_LIBDIR=$MJ_ANDROID >/dev/null
cmake --build build-godot-android -j8 --target emil_godot
mkdir -p build-apk
# Non-Gradle export cannot override Min SDK: patch the template manifest (minSdkVersion 24 -> 28) once.
[ -f build-apk/android_debug_min28.apk ] || python3 -I tools/patch_minsdk.py "$TPL" build-apk/android_debug_min28.apk 28
$GODOT --headless --path editor --import >/dev/null 2>&1 || true
# The export preset keeps no local paths: the patched template path is filled in for the export and restored after.
cp editor/export_presets.cfg build-apk/export_presets.cfg.orig
trap 'cp build-apk/export_presets.cfg.orig editor/export_presets.cfg' EXIT
sed -i "s|^custom_template/debug=.*|custom_template/debug=\"$PWD/build-apk/android_debug_min28.apk\"|" editor/export_presets.cfg
$GODOT --headless --path editor --export-debug "Android" ../build-apk/emil_editor.apk
# Headless self-test of all commands on Linux (prints "EMIL selftest:" lines).
$GODOT --headless --path editor --fixed-fps 60 --quit-after 1800 -- --selftest 2>&1 | grep "EMIL"
