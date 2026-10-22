# Droid Blocks module contract v0

This document defines the smallest module language needed for the first
learning experiments. It is intentionally narrower than the eventual hardware
catalog: one motor SKU, five optional reward-sensor families, actuator feedback,
and passive LEGO-like parts.

The source of truth is the versioned JSON manifest in
`config/module_catalog.json`. Native validation, reward functions, motor
dynamics, and actuator-feedback estimators are implemented in
`include/droid/module_catalog.hpp` and `src/module_catalog.cpp`. Numeric motor
constants are reference-simulator values, not specifications for a selected
physical motor. They must eventually be replaced by bench measurements.

## 1. Assembly model

An assembly is a graph. Nodes are module or passive-part instances and edges are
compatible connector pairs with rigid local transforms. Every instance has an
opaque ID. All physical quantities use SI units and all directions are expressed
in the emitting module's local, right-handed frame.

The policy may receive:

- Opaque instance and type IDs.
- Physical capability descriptors.
- Connector adjacency and local transforms.
- Timestamped sensor observations, validity, and reward rates.
- Timestamped actuator feedback that carries no reward.
- One normalized action slot for every motor instance.

The policy must not receive authored functional roles such as chassis, front,
rear, left, right, leg, wheel role, or task role. A rotary motor remains the
same motor whether its axle is connected to a wheel, a gear train, an arm, or
nothing. Permuting instance IDs must only permute the corresponding policy
inputs and outputs.

### Passive parts

Plastic blocks, wheels, gears, and axles are passive. Their manifests may define
collision geometry, mass, inertia, friction, compliance, connector frames,
joint constraints, and transmission constraints. They emit no observations,
actions, or rewards unless an explicit sensor is attached. Family names may be
used by the assembly compiler, but functional names are not policy features.

This separation matters: attaching a circular plastic part must not silently
install a locomotion controller or tell the learner that it is a wheel.

## 2. Sensor contract

Each sensor family declares:

| Field | Meaning |
| --- | --- |
| `channels` | Typed observations with shapes and SI units. |
| `sample_period_s` | Nominal physical sampling interval. |
| `nominal_latency_s` | Expected measurement latency. |
| `stale_after_s` | Age after which the last sample is invalid. |
| `missing_reward` | Reward rate used after a sample becomes stale. |
| `reward.function_id` | Versioned, deterministic local reward function. |
| `reward.parameters` | Calibration owned by the sensor instance. |
| `hard_safety` | Noncompensable conditions sent to the safety layer. |

A runtime sample has this logical form:

```json
{
  "module_id": "opaque-instance-id",
  "sequence": 42,
  "sample_time_ns": 1234567890,
  "valid": true,
  "observations": {},
  "reward_rate": 0.0,
  "safety_flags": []
}
```

`reward_rate` is dimensionless reward per second and is bounded to `[-1, 1]`
by the sensor. A multi-axis sensor emits one reward scalar for the entire sensor
instance, not one reward per channel.

Only sensor instances physically present and registered in the current assembly
are voters. Merely existing in the catalog does not create a sensor instance or
a reward. For an environment transition lasting `dt_s`, the global objective is
exactly:

```text
transition_reward = dt_s * sum(each discovered sensor instance's reward_rate)
```

There is no central reweighting, family normalization, or final clipping.
Connecting a duplicate sensor under a distinct module ID deliberately adds a
duplicate vote. Algorithms
may use value normalization internally, and evaluation should report both raw
total reward and reward per sensor, but neither changes the environment
objective.

Using reward rate rather than reward per packet prevents a fast sensor from
outvoting a slow sensor merely because it samples more often. After a sensor is
registered, unplugging, silencing, saturating, or corrupting it cannot erase its
preference: the assembly registry retains that instance and applies its declared
`missing_reward` after the timeout. A deliberately rebuilt assembly may establish
a new registry after physical enumeration. Loss of a safety-critical sensor also
asks the safety layer to stop.

Hard safety is evaluated before action execution and is never added to reward.
Ten positive sensors cannot cancel one overtemperature veto.

## 3. Sensor families

The v0 catalog contains exactly five optional reward-sensor families.

| Family | Observation channels | Local preference |
| --- | --- | --- |
| Power | Bus voltage (V), signed current (A), remaining energy fraction, temperature, charging state | Rewards net charging, penalizes discharge and dangerously low stored energy. |
| Six-axis IMU | Specific force (m/s^2), angular velocity (rad/s) | Penalizes severe acceleration deviations, free fall, impacts, and extreme rotation; it defines no privileged "upright" pose. |
| Touch/force | Contact flag and normal force (N) | Neutral for gentle contact; negative as force approaches the structural limit. |
| ToF proximity | Local ray distance (m) and validity | Neutral in clear space; negative inside the configured danger envelope. |
| Ambient light | Illuminance (lux) and saturation flag | Log-scaled positive preference for illumination. |

The first four primarily express viability. Ambient light supplies a simple
appetite, preventing "spend no energy and never move" from being the only good
solution. Later sensor variants may encode other preferences, but a variant
must receive a new versioned family ID instead of silently changing a reward
function.

All five families are optional. With no registered sensors, the reward is zero;
there are no imaginary votes from absent catalog entries.

The v0 set intentionally excludes cameras, microphones, magnetometers, GPS,
and simulator world position. They add calibration, bandwidth, privacy, or
sim-to-real complexity that is unnecessary for the first experiment.

## 4. Motor contract

The sole v0 actuator is a continuous rotary, current-controlled DC gearmotor.
Every motor instance exposes one action:

```text
effort in [-1, 1]
```

`effort` is the signed fraction of thermally available winding current. It is
not a target speed or a request to move a wheel. A local current loop turns the
request into winding current; gearbox ratio and efficiency turn motor torque
into output-shaft torque. The reference model includes back EMF, bus-voltage
limits, current response lag, viscous and Coulomb friction, copper heating,
passive cooling, and linear thermal derating.

In compact form:

```text
I_target = effort * thermally_available_peak_current
tau_electromagnetic = efficiency * gear_ratio * torque_constant * I
tau_output = tau_electromagnetic - viscous_friction - Coulomb_friction
dT/dt = (I^2 * resistance - (T - ambient) / thermal_resistance)
         / thermal_capacity
```

`step_motor_current_thermal` is the dependency-free executable reference. The
simulator remains responsible for integrating output-shaft position and
velocity.

The motor driver owns hard current, voltage, speed, temperature, joint-limit,
and command-timeout enforcement. At a hard fault it requests zero current and
reports a veto and fault flags. Current and temperature still decay according
to physics; a safety veto is not an instantaneous deletion of stored energy.

### Actuator feedback is not reward

The motor electronics report feedback required for control and protection:

- Wrapped shaft position (rad) and velocity (rad/s).
- Winding current (A), bus voltage (V), and temperature (degC).
- Estimated output torque (N*m).
- A load-impedance estimate (N*m*s/rad).
- A smoothed stuck score in `[0, 1]`, a derived stuck flag, and fault flags.

These values are grouped under the motor's `actuator_feedback` contract, not
under `sensor_families`. Although physical transducers produce several values,
they describe the actuator's state; they do not express a preference and never
enter `transition_reward`. A robot with motors and no optional reward sensors
therefore has observations and hard safety, but zero objective.

`actuator_feedback_sample` is the deterministic reference estimator. It defines
load impedance as absolute torque divided by floored absolute shaft velocity,
and low-pass filters evidence that commanded effort is appreciable while shaft
velocity remains near zero. A stuck flag is diagnostic; hard current, voltage,
speed, temperature, joint-limit, timeout, and stale-feedback rules remain in the
separate safety layer.

## 5. Integration boundary

The current hand-authored demo keeps its automatic drive behavior for
visualization. The implemented learning environment exposes a separate,
deterministic synchronous interface:

```text
reset(assembly, seed, record, episode_profile) -> observation, info
step({motor_instance_id: effort}, dt_s)
  -> observation, reward, reward_components, safety,
     terminated, truncated, info
```

The assembly compiler should generate MuJoCo names from opaque instance IDs and
maintain a registry that maps those names back to module manifests. The runtime
must discover action and observation slots from that registry rather than from
hard-coded wheel names or positions.

MuJoCo root transforms, exact world position, global contact count, authored
part roles, and noiseless state may be retained for visualization and benchmark
metrics. They are privileged evaluation data and must not enter policy
observations unless an equivalent physical sensor exists.

The policy callback is narrower still: current policy-safe observation, opaque
motor IDs, control-step index, and a policy-only seed. Environment seed,
episode realization, reward, safety, lifecycle, reset/step `info`, and
visualization state remain evaluator-owned. See
[`AGENT_ENVIRONMENT.md`](AGENT_ENVIRONMENT.md) and
[`LEARNING_EXPERIMENT.md`](LEARNING_EXPERIMENT.md).
