# Read-only model sharing audit

The native headless training path can share one fully configured MuJoCo `mjModel` while keeping a separate `mjData` for every world. This is a source review, not an implemented optimization or benchmark. No training process, container, physics definition, or source file was changed for the audit.

## Model ownership and mutations

Each [WalkingSimulation implementation](../src/simulation.cpp) currently owns its model and data through separate `unique_ptr` members. Its constructor loads and compiles the XML anew for each instance. The only direct model writes found are in that constructor: the configured physics timestep, offscreen framebuffer dimensions, rendering quality settings, and the floor's geometry group. These writes must finish before a shared model is made available to workers.

The `reset`, `advance`, `feet_contacts`, and physical-metric methods do not mutate the model. Reset changes each world's `mjData`, command, policy object, action/history state, contacts, and counters. Physics stepping and telemetry write only the world's data and local fields. No custom MuJoCo global callbacks, per-world mass changes, or friction randomization were found in the reviewed training path.

The bundled MuJoCo 3.3.2 primary headers were read directly from [the dependency archive](mujoco-3.3.2-linux-x86_64.tar.gz), member `mujoco-3.3.2/include/mujoco/mujoco.h`. The signatures of `mj_step`, `mj_forward`, `mj_resetData`, and `mj_makeData` accept `const mjModel*`. The metric functions `mj_objectVelocity`, `mj_contactForce`, and `mj_jac` also accept a const model. These interfaces support a shared read-only model with independent mutable simulation data.

## Worker boundaries

The [training pool and world construction](../training/walk_train.cpp) create one simulation object per environment, currently in a sequential loop. The pool distributes environment indices by worker index and stride; a world is processed by exactly one worker in each job. The caller waits for every worker before evaluating post-step critic values and dispatching resets. Pool destruction joins the worker threads before the earlier-declared world collection is destroyed.

Sharing the model does not permit sharing histories, commands, random generators, telemetry, or data arenas. All 64 environments still need independent persistent `mjData` objects even when there are only eight worker threads. The same world must not be stepped, reset, rendered, or read concurrently without coordination.

## Constraints on a future implementation

- Complete model compilation and configuration once, before workers can access it. Retain shared ownership until every associated `mjData` has been destroyed; multiple owning `unique_ptr` instances must never point to one model.
- Share only worlds using the same verified assets and configured model settings. Preserve the existing timestep, control decimation, actuator order, gains, observation scales, and dynamics contract.
- Keep renderer contexts, scenes, cameras, and frame buffers private. Framebuffer dimensions currently reside in `mjModel`, so a different-size viewer must not rewrite a model concurrently used by training worlds.
- Keep loaded policy modules private. External-action training does not enter the reset branch that changes global cuDNN settings for loaded CUDA policies.
- Reassess sharing if later work introduces per-world model mutations such as mass, friction, damping, geometry, or timestep changes. The current audit does not authorize those changes.

## What the audit does and does not establish

For 64 otherwise identical worlds, sharing would remove 63 repeated compiled-model instances and their repeated XML/mesh compilation. Per-world `mjData` buffers and arenas, observations, rollout tensors, and telemetry would remain.

The reported roughly 10 GB process footprint and roughly minute-long initialization were not measured or attributed by this audit. It establishes neither a specific memory saving nor a steady-state transitions-per-second improvement. A later comparison should separate `mjModel::nbuffer` from each world's `mjData::nbuffer` and `mjData::narena`, and measure constructor time independently of rollout throughput. The training timer currently starts after world construction, so its reported transitions per second does not include this initialization cost.

Before using shared-model results as continuation evidence, compare identical seeded action sequences and resulting states/metrics against the existing implementation. This audit only identifies the supported ownership boundary; implementation and runtime equivalence remain outstanding.
