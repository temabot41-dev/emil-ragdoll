#!/bin/bash
# Sanity check: build the bench, selftest (must be 0 errors), short set (push 120 runs + reactions).
# Usage: bash tools/check.sh [label]   (needs tools/setup_deps.sh done first)
# Env: EMIL_DEPS, EMIL_TMP, JOBS as in setup_deps.sh; BUILD_DIR (default build).
# Results of x86_64 and arm64 builds differ in the last digits (float order), so only the summary lines are printed and
# the exit code is non-zero when the build or selftest fails or a bench crashes. CSVs go to $EMIL_TMP/{pk,rx}/<label>/.
set -euo pipefail
cd "$(dirname "$0")/.."
export EMIL_DEPS="${EMIL_DEPS:-$HOME/.cache/emil_deps}"
export EMIL_TMP="${EMIL_TMP:-/tmp/emil}"
JOBS="${JOBS:-$(nproc)}"
BUILD="${BUILD_DIR:-build}"
LABEL="${1:-check}"
mkdir -p "$EMIL_TMP"
t0=$(date +%s)

# Wheel fallback of setup_deps.sh: shared libmujoco and its headers instead of the static build.
EXTRA=""
W="$EMIL_DEPS/mujoco-3.17.0-wheel"
if [ ! -f "$EMIL_DEPS/mujoco-3.17.0/build-linux-nolto/lib/libmujoco.a" ] && [ -f "$W/mujoco/libmujoco.so.3.17.0" ]; then
  EXTRA="-DMUJOCO_ROOT=$W/mujoco -DMUJOCO_SO=$W/mujoco/libmujoco.so.3.17.0"; echo "check: using the MuJoCo wheel (shared)"
fi
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DEMIL_DEPS="$EMIL_DEPS" $EXTRA > "$EMIL_TMP/check_cmake.log" 2>&1 || { tail -20 "$EMIL_TMP/check_cmake.log"; exit 1; }
cmake --build "$BUILD" -j"$JOBS" --target emil_bench > "$EMIL_TMP/check_build.log" 2>&1 || { tail -30 "$EMIL_TMP/check_build.log"; exit 1; }
t1=$(date +%s)

"$BUILD/emil_bench" selftest > "$EMIL_TMP/check_selftest.txt"
tail -1 "$EMIL_TMP/check_selftest.txt"
grep -q "^selftest: 0 errors" "$EMIL_TMP/check_selftest.txt" || { echo "check: SELFTEST FAILED"; exit 1; }

"$BUILD/emil_bench" push "$LABEL" "$JOBS" > "$EMIL_TMP/check_push.txt"
grep -E "^(moment|all) " "$EMIL_TMP/check_push.txt"
"$BUILD/emil_bench" rx "$LABEL" "$JOBS" > "$EMIL_TMP/check_rx.txt"
echo "rx: $(($(wc -l < "$EMIL_TMP/check_rx.txt") - 2)) scenes, table in $EMIL_TMP/check_rx.txt"
t2=$(date +%s)
echo "check: OK. build $((t1-t0)) s, run $((t2-t1)) s (EMIL_DEPS=$EMIL_DEPS, EMIL_TMP=$EMIL_TMP)"
