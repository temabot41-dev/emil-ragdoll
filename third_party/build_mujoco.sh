#!/bin/bash
# Stock MuJoCo 3.17.0 (Apache-2.0) as static libraries for Linux arm64 (bench, editor on Linux) and Android arm64 (APK).
# Source is unmodified except four build-only edits (static library, no plugins, no classic renderer/UI, no -Werror),
# listed in third_party/MUJOCO_NOTICE.md. Output: $D/mujoco-3.17.0/build-{linux,android}/lib/*.a
set -e
D=${EMIL_DEPS:-$HOME/.cache/emil_deps}; V=3.17.0; S=$D/mujoco-$V
HERE=$(cd "$(dirname "$0")/.." && pwd)
NDK=${ANDROID_NDK:-$HOME/android/android-ndk-r27c}; export ANDROID_NDK=$NDK
cd $D
[ -d $S ] || { curl -sSL -o mujoco-$V.tar.gz https://github.com/google-deepmind/mujoco/archive/refs/tags/$V.tar.gz; tar xzf mujoco-$V.tar.gz; }
cd $S
sed -i 's/^  add_library(mujoco SHARED \${MUJOCO_RESOURCE_FILES})$/  add_library(mujoco STATIC ${MUJOCO_RESOURCE_FILES})  # emil: static/' CMakeLists.txt
sed -i 's/^if(NOT EMSCRIPTEN AND NOT MUJOCO_USE_FILAMENT_MJR_COMPAT)$/if(FALSE)  # emil: no classic renderer\/UI/' CMakeLists.txt
sed -i 's/^if(NOT EMSCRIPTEN AND NOT (APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS))$/if(FALSE)  # emil: no install\/export rules/' CMakeLists.txt
sed -i '/^ *-Werror$/d' cmake/MujocoOptions.cmake
sed -i '/^if(NOT EMSCRIPTEN)$/{N;s/^if(NOT EMSCRIPTEN)\n  add_subdirectory(plugin\/elasticity)/if(FALSE)  # emil: no plugins\n  add_subdirectory(plugin\/elasticity)/}' CMakeLists.txt
COMMON="-DCMAKE_BUILD_TYPE=Release -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
rm -rf build-linux build-android; cmake -S . -B build-linux $COMMON > $D/mj_linux_cfg.log 2>&1
cmake --build build-linux -j6 > $D/mj_linux_build.log 2>&1
# Dev build without LTO (CMake default for Linux bench/editor): relinking emil_bench takes seconds instead of minutes.
cmake -S . -B build-linux-nolto $COMMON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON $(for d in $S/build-linux/_deps/*-src; do n=$(basename $d -src | tr 'a-z-' 'A-Z_'); printf -- "-DFETCHCONTENT_SOURCE_DIR_$n=$d "; done) > $D/mj_nolto_cfg.log 2>&1
cmake --build build-linux-nolto -j3 > $D/mj_nolto_build.log 2>&1
F=""; for d in $S/build-linux/_deps/*-src; do n=$(basename $d -src | tr 'a-z-' 'A-Z_'); F="$F -DFETCHCONTENT_SOURCE_DIR_$n=$d"; done  # reuse sources, no network
cmake -S . -B build-android $COMMON -DFETCHCONTENT_FULLY_DISCONNECTED=ON $F -DCMAKE_TOOLCHAIN_FILE=$HERE/tools/android-arm64.cmake -DLIBM_LIBRARY=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/28/libm.so > $D/mj_android_cfg.log 2>&1
cmake --build build-android -j6 > $D/mj_android_build.log 2>&1
for b in build-linux build-android; do echo "$b: $(find $S/$b -name '*.a' | wc -l) static libs, mujoco: $(find $S/$b -name 'libmujoco.a' | head -1)"; done
