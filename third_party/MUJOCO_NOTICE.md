# MuJoCo (third-party)

EMIL links MuJoCo 3.17.0 by Google DeepMind, licensed under the Apache License 2.0
(https://github.com/google-deepmind/mujoco, `LICENSE` in the MuJoCo source tree). The MuJoCo license text must ship with every
binary that contains it (bench, Godot extension, editor APK).

Source: the official release tarball `mujoco-3.17.0.tar.gz`, built by `third_party/build_mujoco.sh` into
`$EMIL_DEPS/mujoco-3.17.0/build-{linux,android}` (default `$HOME/.cache/emil_deps`). The same
build-only edits are kept as `third_party/mujoco-3.17.0-emil-static.patch`; `tools/setup_deps.sh` applies it and builds the Linux libraries without LTO.

MuJoCo's physics and code are unmodified. Build files only (changes made by the build script):
- `CMakeLists.txt`: the `mujoco` library is built `STATIC` instead of `SHARED`;
- `CMakeLists.txt`: the plugin subdirectories (elasticity, actuator, sensor, sdf) are not built;
- `CMakeLists.txt`: the classic OpenGL renderer/UI and the install/export rules are not built;
- `cmake/MujocoOptions.cmake`: `-Werror` removed (newer compilers warn in third-party headers).

The Android build uses `tools/android-arm64.cmake` (API 28) and reuses the dependency sources fetched by the Linux build
(ccd, lodepng, marchingcubecpp, miniz, qhull, tinyobjloader, tinyxml2 — each under its own license, see their source trees).
