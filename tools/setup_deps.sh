#!/bin/bash
# Dependency setup (Ubuntu x86_64, ~4 vCPU): toolchain + MuJoCo 3.17.0 static libs without LTO,
# only the targets EMIL links.
# Idempotent: when the libraries exist it returns at once (keep $EMIL_DEPS cached between runs).
# Usage: bash tools/setup_deps.sh        (then: bash tools/check.sh)
# Env: EMIL_DEPS (default $HOME/.cache/emil_deps; CMake reads the same variable), JOBS (default all cores).
set -euo pipefail
V=3.17.0
D="${EMIL_DEPS:-$HOME/.cache/emil_deps}"
S="$D/mujoco-$V"
B="$S/build-linux-nolto"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="${JOBS:-$(nproc)}"
LIBS="libmujoco libtinyxml2 libqhullstatic_r libccd libtinyobjloader libminiz librender_noop"
t0=$(date +%s)

have_libs() { for l in $LIBS; do [ -f "$B/lib/$l.a" ] || return 1; done; }
if have_libs; then echo "setup_deps: MuJoCo libs already in $B/lib (cached)"; echo "export EMIL_DEPS=$D"; exit 0; fi
W="$D/mujoco-$V-wheel"   # fallback: official PyPI wheel (shared libmujoco + headers), used when GitHub is unreachable
if [ -f "$W/mujoco/libmujoco.so.$V" ]; then echo "setup_deps: MuJoCo wheel already in $W (cached; check.sh uses MUJOCO_SO)"; exit 0; fi
wheel_fallback() {
  echo "setup_deps: GitHub source not reachable; using the PyPI wheel mujoco==$V (shared library)"
  rm -rf "$W"; mkdir -p "$W/dl"
  python3 -m pip download -q --no-deps "mujoco==$V" -d "$W/dl"
  (cd "$W" && python3 -m zipfile -e dl/mujoco-$V-*.whl .)
  rm -rf "$W/dl"
  [ -f "$W/mujoco/libmujoco.so.$V" ] || { echo "setup_deps: wheel has no libmujoco.so.$V"; exit 1; }
  echo "setup_deps: done (wheel) in $(( $(date +%s) - t0 )) s"; exit 0
}

# 1. Toolchain (skipped when present; sudo only when not root).
need=""
command -v cmake >/dev/null || need="$need cmake"
command -v g++ >/dev/null || need="$need g++"
command -v make >/dev/null || need="$need make"
command -v patch >/dev/null || need="$need patch"
command -v curl >/dev/null || need="$need curl"
command -v git >/dev/null || need="$need git"  # MuJoCo fetches its dependencies (ccd, qhull, tinyxml2, ...) with FetchContent
if [ -n "$need" ]; then
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq --no-install-recommends $need ca-certificates
fi
# Ninja is optional (faster than make); best effort.
command -v ninja >/dev/null || { SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo; $SUDO apt-get install -y -qq ninja-build >/dev/null 2>&1 || true; }
GEN=""; command -v ninja >/dev/null && GEN="-G Ninja"

# 2. Source: official tag tarball + our build-only patch (static library, no plugins/renderer/install, no -Werror).
mkdir -p "$D"
if [ ! -d "$S" ]; then
  tmp="$D/mj_src_tmp"; rm -rf "$tmp"; mkdir -p "$tmp"
  curl -fsSL --retry 3 -o "$tmp/mujoco.tar.gz" "https://github.com/google-deepmind/mujoco/archive/refs/tags/$V.tar.gz" \
    || { rm -rf "$tmp"; wheel_fallback; }
  tar xzf "$tmp/mujoco.tar.gz" -C "$tmp"
  (cd "$tmp/mujoco-$V" && patch -p1 -s < "$HERE/third_party/mujoco-$V-emil-static.patch")
  mv "$tmp/mujoco-$V" "$S"; rm -rf "$tmp"
fi
t1=$(date +%s)

# 3. Configure (fetches the dependencies) and build only the libraries the bench links; no LTO
# (same as build-linux-nolto on arm64).
cmake -S "$S" -B "$B" $GEN -DCMAKE_BUILD_TYPE=Release -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF > "$D/mj_nolto_cfg.log" 2>&1 || { tail -20 "$D/mj_nolto_cfg.log"; exit 1; }
t2=$(date +%s)
TARGETS=""; for l in $LIBS; do TARGETS="$TARGETS ${l#lib}"; done
cmake --build "$B" -j"$JOBS" --target $TARGETS > "$D/mj_nolto_build.log" 2>&1 || { tail -30 "$D/mj_nolto_build.log"; exit 1; }
t3=$(date +%s)
have_libs || { echo "setup_deps: libraries missing in $B/lib"; ls "$B/lib"; exit 1; }
echo "setup_deps: done. source+tools $((t1-t0)) s, configure $((t2-t1)) s, build $((t3-t2)) s, total $((t3-t0)) s"
echo "export EMIL_DEPS=$D"
