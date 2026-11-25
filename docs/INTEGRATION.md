# Status and integration

## This is a very raw, early version
EMIL is an early prototype. A lot of it works poorly, and a lot is missing. It has only been run in its own editor app
and on its benches, never inside a real game level. Expect rough motion, hard landings in some falls, and parameters
that change between versions.

Known weak spots (not complete):
- in the hardest forward falls the arms reach the floor too late and the head can hit hard;
- dodging a car in a hurry often fails, the body is caught in the air;
- a few death falls after a burst of bullets still land the head hard;
- recovery from very strong pushes ends in a fall more often than a person would.

## What is not there yet
- **No contacts with the Godot scene.** The core runs its own MuJoCo world. That world contains a floor plane and a few
  built-in test props (a car box, a swaying deck, bridge and stair benches). Godot colliders, level geometry, other
  Godot physics bodies and Godot characters are not seen by the core at all.
- **No partial ragdoll.** The whole body is always simulated. A body that is partly animated and partly physical
  (for example, legs animated while the upper body reacts) does not exist.
- **No transitions between animation and physics.** There is no hand-off from an animation to the active ragdoll and
  no blend back from physics to animation.
- **No getting up** after a fall.
- **No walking or running on command.** The character stands, steps to recover balance, takes commanded single steps
  in the editor and reacts to hits; it does not walk to a goal.
- Several characters in one world (`set_bot_count`) work only for simple tests; interactions between them are barely
  tested.
- Only Linux and Android arm64 builds have been tried.

## How to add contacts with a game scene (a general plan)
This is written for whoever extends EMIL, including AI coding assistants. It is a direction, not a finished design.

1. **Static level geometry.** Before the core builds its world (`World::Build` in `core/emil_world.cpp`), collect the
   Godot static colliders near the play area and turn them into MuJoCo geoms in the world body: boxes, spheres and
   capsules map directly; convex meshes become a `<mesh>` asset with a `type="mesh"` geom; terrain becomes an
   `hfield`. Add a small core API that stores these shapes and writes them into the model XML, next to the existing
   built-in props. Mind the axes: MuJoCo is z-up, Godot is y-up; the extension already converts
   (`godot/src/emil_godot.cpp`, `(x, y, z)_godot = (x, z, -y)_mujoco`), reuse that conversion. Units are metres.
2. **The model is fixed after it is compiled.** MuJoCo cannot add geoms to a running world. Either rebuild the world
   when the level changes (characters must then be re-spawned in their current pose), or keep a fixed pool of
   static geoms around each character and move them as the character travels. Keep the number of geoms near the
   characters small; contacts are the main cost of the step.
3. **Moving Godot objects** (doors, vehicles, props): add each as a MuJoCo mocap body and copy its Godot transform into
   `mjData.mocap_pos` / `mocap_quat` before every core step. The character then collides with it. For two-way
   coupling, read the contact forces on that body (`mj_contactForce`) after the step and apply them to the Godot
   `RigidBody3D`.
4. **Hits and bullets** from Godot: cast the ray in Godot against the rendered character, or use the core's own ray
   (`pick`), then call `shoot` / `push` with the body, point and impulse.
5. **Timing:** step the core from `_physics_process` at a fixed rate (the extension already does 60 Hz), read the bone
   transforms with `get_buffer` for rendering, and never teleport core bodies except through the API.
6. **Filtering:** use MuJoCo `contype` / `conaffinity` so that level geoms do not test against each other; keep the
   character's own self-collision settings from `core/emil_body.cpp`.
7. **Test** every new contact type on the benches first (a box, a step, a slope, a moving door), then in the editor,
   before using it in a game.

## License reminder
EMIL is MIT-licensed, Copyright (c) 2026 TEMIR. Any copy or substantial part of this code, including shipped builds,
must keep that copyright notice and the license text (`LICENSE`), and TEMIR should be credited as the author.
The builds also contain MuJoCo (Apache-2.0) and other components whose license texts must ship with them
(`THIRD_PARTY_NOTICES.md`).
