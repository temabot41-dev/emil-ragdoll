# EMIL architecture

One MuJoCo model (`mjModel`) holds the floor, all bots and the car; it is built once (`World::Build`). Each bot is
31 hinge joints on a de Leva body (`core/emil_body.*`), driven only by joint torques. Dependencies go only downward.

## Layers and files
| layer | file | role |
|---|---|---|
| L0 body | `core/emil_body.h/.cpp` | anthropometry (de Leva), joint table (limits, torque at 80 %), MJCF text, `BodyDims` |
| L0 muscles | `core/l0_muscles.cpp` | per substep: PD to `Bot::target` with `Bot::kp`, gravity compensation, virtual CoM force (`VfTorque`, J^T, ZMP-capped), stance-hip torso torque (SIMBICON), stance ankle ZMP cap, torque limits -> `d->ctrl` |
| L1 balance | `core/l1_balance.cpp`, `core/l1_leg_ik.h` | per frame from the delayed percept: stance (capture point, ankle and hip strategy, virtual force), step decision and leg choice, step target (LIPM capture point at touchdown + sole edge margin, geometric limits), swing (time based, IK or SIMBICON law `step_law`), crossover, landing compliance; writes targets, gains |
| L2 reactions | `core/l2_reactions.cpp` | bullets (wound, flinch), airborne tuck, landing, fall protective pose, lying relax, curl, car; changes only `BalanceInput`, targets and stiffness after L1 |
| L3 perception | `core/l3_perception.cpp` | snapshot `Percept` (CoM, velocities, feet, pelvis frame, joint angles) into a double buffer; controllers read it one control step late; per-substep foot normal force + contact hysteresis |
| world | `core/emil_world.h/.cpp` | `World`: build, reset, `StepFrame`, impulses, `Shoot`, `DropFrom`, `Car`, `Curl`, `Ray`, kinematic car |
| params | `core/emil_params.h` | one table `{name, default, min, max, description}`; `ParamId` enum; used by bench (`-p`) and editor sliders |
| wrapper | `godot/src/emil_godot.cpp` | `EmilWorld : Node` (create N bots, step in `_physics_process`, tools, poses array, params by name, core ms) |

## Frame (60 Hz, dt 1/480 s, 8 substeps)
```
StepFrame:
  for each bot: Perceive (pc[cur]) -> Balance (reads pc[cur^1]) -> Reactions
  8 x substep: CarSubstep -> mj_step1 -> for each bot: Muscles -> mj_step2 -> FootContacts
```
No allocations per frame: all state in flat arrays inside `Bot` / `World`; MuJoCo data preallocated.
Collision: capsules and boxes (feet are boxes; a flat foot gives 4 floor contacts).

## Key conventions (verified by `emil_bench selftest`)
Positive joint angle = flexion, abduction (out), external rotation, knee flex, ankle dorsiflexion (toes up),
inversion (sole in); spine roll + = lean right, yaw + = turn left. Hip chain: flex (outer), abd, rot (inner).
`pelvis_pitch` + = forward lean, `pelvis_roll` + = top leans left. Swing thigh world angle = flex - pitch.
`torso_tau` = torque on the pelvis (applied as the reaction of the stance hip torques).

## Parameters
Add a row to `kParams` and an id to `ParamId` in `core/emil_params.h`; read it as `pr[P_NAME]` in L1/L0. It shows up in
the bench (`-p name=value`) and the editor sliders automatically. Constants that are physics (limits, geometry)
stay `constexpr` next to the code with a one-line reason.

## Adding a reaction
1. Add the trigger (world call or detection) in `core/emil_world.cpp` / `World` API if needed.
2. Implement it in `core/l2_reactions.cpp`: change `Bot::in` (`BalanceInput`: lean, trunk_lean, stiffness, leg_weak,
   curl) and, if needed, targets/stiffness after `Balance`; keep persistent state in `Bot` fields (no allocation).
3. Add a scene to `bench/react.cpp` (table row + CSV), run `emil_bench rx <label> 8`, compare with the previous label.
4. Expose it in `godot/src/emil_godot.cpp` and the editor tool bar if the user needs it.
