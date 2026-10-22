import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';

// This auditor never opens a simulator or trains a policy. Native replay and
// neural optimization are deliberately distinguished in its exported evidence.
const PROJECT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SEEDS = [101, 202, 303];
const ARMS = ['frozen', 'adapt', 'scratch'];
const PROBES = ['quiet', 'source_quiet', 'negative', 'positive', 'rhythm_2s', 'rhythm_4s'];
const PLAYABLE = new Set([...ARMS, 'source_frozen']);
const REQUESTS = [-.15, 0, .15];
const LOCKED_PROTOCOL_SHA = '82d83fe5004304bba9f8b17ac8710a8212fc9c5ec729908f05e14f98b08fcef1';
const LOCKED_SOURCE_MANIFEST_SHA = '7a27305fbcf6b3826b022b86cabc0a488e3ef2dc082c868bbf09bf49a4a49480';
const REFERENCE_SHA = 'a0421bb5fa1b792af934986a3758fdb517081dceb6fdaaf68f1498116decd309';
const GENERATOR = 'tools/summarize-actuation-transfer.mjs';
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const readJson = filename => JSON.parse(fs.readFileSync(filename, 'utf8').replace(/^\uFEFF/, ''));
const readGzip = filename => JSON.parse(gunzipSync(fs.readFileSync(filename)).toString('utf8'));
const hashFile = filename => sha(fs.readFileSync(filename));
const clone = value => structuredClone(value);
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function finite(value, label) {
  assert.equal(typeof value, 'number', label);
  assert.ok(Number.isFinite(value), label + ' must be finite');
  return value;
}
function finiteTree(value, label = 'JSON') {
  if (typeof value === 'number') finite(value, label);
  else if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) finiteTree(item, `${label}.${key}`);
}
function near(actual, expected, label, tolerance = 1e-10) {
  finite(actual, label); finite(expected, label);
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
}
function hash(value, label) { assert.match(value, /^[a-f0-9]{64}$/, label); return value; }
function keys(value, expected, label) {
  assert.ok(value && typeof value === 'object' && !Array.isArray(value), label);
  assert.deepEqual(Object.keys(value).sort(), [...expected].sort(), label);
}
function inside(root, name) {
  assert.equal(typeof name, 'string');
  assert.ok(name && !path.isAbsolute(name) && !name.includes(':') && !name.split(/[\\/]/).includes('..'), 'Unsafe evidence path');
  const full = path.resolve(root, name);
  for (let current = full; current !== path.resolve(root); current = path.dirname(current)) {
    assert.ok(current.startsWith(path.resolve(root) + path.sep), 'Evidence path escaped root');
    if (fs.existsSync(current)) assert.ok(!fs.lstatSync(current).isSymbolicLink(), 'Symlink evidence is not accepted');
  }
  return full;
}
function allFiles(root, prefix = '') {
  const result = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    assert.ok(!entry.isSymbolicLink(), 'No symbolic links in an evidence archive');
    if (entry.isDirectory()) result.push(...allFiles(path.join(root, entry.name), relative));
    else if (entry.isFile()) result.push(relative);
    else assert.fail('Non-file archive entry: ' + relative);
  }
  return result.sort();
}
function writeNew(filename, value) {
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  fs.writeFileSync(filename, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
}
function canonical(value) {
  if (typeof value === 'number' && Object.is(value, -0)) return '-0';
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
  return JSON.stringify(value);
}
function caseConfig(lengths, weight, side, hinge) {
  return { assembly: { schema: 'construction_kit_v2', name: 'The same parts, rebuilt', segments: lengths,
    powered_hinge: hinge, blocks: weight ? [{ id: 'block-1', segment: 1, slot: 0, side: -1 }] : [],
    sensor: { segment: 1, slot: 2, side: -1 }, light: { segment: 1, slot: 2, side: 1, face: 1 } },
    sun: { position_m: [side * .3, 0, .2], intensity_lux: 1000 } };
}

export function validateProtocol(protocol) {
  finiteTree(protocol);
  assert.equal(protocol.schema, 'actuation_transfer_protocol_v1');
  assert.equal(protocol.status, 'locked_before_native_learning');
  assert.deepEqual(protocol.source_seeds, SEEDS);
  assert.equal(protocol.development_seed, 404);
  assert.deepEqual(protocol.arms, ARMS);
  assert.deepEqual(protocol.controls, ['source_frozen', 'source_quiet', 'quiet', 'negative', 'positive', 'rhythm_2s', 'rhythm_4s']);
  for (const [key, expected] of Object.entries({ source_decisions_per_seed: 98304, development_decisions: 32768,
    target_decisions: 640, decision_dt_s: .1, native_dt_s: .02 })) near(protocol[key], expected, key);
  assert.deepEqual(protocol.early_windows_s, [6.4, 12.8, 32, 64]);
  assert.equal(protocol.policy_input.dimension, 204);
  assert.equal(protocol.policy_input.frames, 12); assert.equal(protocol.policy_input.frame_size, 17);
  assert.equal(protocol.ppo.implementation, 'stable-baselines3'); assert.equal(protocol.ppo.version, '2.9.0');
  for (const [key, expected] of Object.entries({ source_n_steps: 256, source_n_envs: 8, target_n_steps: 64,
    target_n_envs: 1, n_epochs: 4, batch_size: 64, learning_rate: .0003, gamma: .92 ** (.1 / .4),
    gae_lambda: .95, clip_range: .2, ent_coef: .01, vf_coef: .5, max_grad_norm: .5 })) near(protocol.ppo[key], expected, 'PPO ' + key);
  const source = [];
  for (const a of [3, 4]) for (const b of [3, 4]) for (const weight of [false, true]) for (const side of [-1, 1]) source.push(caseConfig([a, b], weight, side, 0));
  assert.deepEqual(protocol.source_cases, source, 'Exact source domain');
  const target = [];
  for (const [id, lengths, weight] of [['long-upper', [4, 3], false], ['long-lower', [3, 4], false], ['weighted', [4, 4], true]]) {
    for (const side of [-1, 1]) target.push({ id: `${id}-${side < 0 ? 'left' : 'right'}`, case: caseConfig(lengths, weight, side, 1) });
  }
  assert.deepEqual(protocol.target_cases, target, 'Exact target case order and actuation');
  assert.deepEqual(protocol.development_cases, [-1, 1].map(side => caseConfig([3, 3], false, side, 1)));
  assert.equal(protocol.reference.report_sha256, REFERENCE_SHA);
  assert.equal(protocol.reference.overall_passed, false);
  return protocol;
}

export function auditArchive(root, protocolSHA) {
  assert.equal(hashFile(inside(root, 'source-manifest.json')), LOCKED_SOURCE_MANIFEST_SHA, 'The pre-learning source manifest changed');
  const manifest = readJson(inside(root, 'source-manifest.json'));
  assert.equal(manifest.schema, 'actuation_transfer_source_v1');
  assert.equal(manifest.protocol_sha256, protocolSHA);
  const names = manifest.files.map(file => file.path);
  assert.equal(new Set(names).size, names.length, 'Duplicate archived source');
  assert.deepEqual(allFiles(inside(root, 'source')), [...names].sort(), 'Exact source archive membership');
  for (const entry of manifest.files) {
    const filename = inside(inside(root, 'source'), entry.path);
    assert.equal(hashFile(filename), hash(entry.sha256), 'Archived file changed: ' + entry.path);
    assert.equal(fs.statSync(filename).size, entry.bytes, 'Archived byte count');
  }
  for (const required of ['training/actuation_transfer.py', 'training/native_transfer.py', 'training/requirements-transfer.txt',
    'src/transfer_env_cli.cpp', 'src/construction.cpp', 'build/native/droid-transfer-env', 'docs/ACTUATION_TRANSFER_EXPERIMENT.md',
    'build/actuation-transfer-qa/adapter-tests.json', 'build/actuation-transfer-qa/bridge-tests.json']) assert.ok(names.includes(required), 'Missing provenance: ' + required);
  for (const name of ['adapter-tests.json', 'bridge-tests.json']) {
    const check = readJson(inside(root, 'source/build/actuation-transfer-qa/' + name));
    assert.equal(name === 'adapter-tests.json' ? check.pass : check.passed, true, 'Archived prerequisite ' + name);
  }
  return { files: names.length, manifest_sha256: hashFile(inside(root, 'source-manifest.json')),
    native_binary_sha256: manifest.files.find(file => file.path === 'build/native/droid-transfer-env').sha256 };
}

export function auditReference(root, protocol) {
  const filename = inside(root, 'reference-report.json');
  assert.equal(hashFile(filename), REFERENCE_SHA, 'Reference report changed');
  const report = readJson(filename);
  assert.equal(report.schema, 'sb3_cartpole_reference_v1'); assert.equal(report.complete, true);
  assert.equal(report.passed, false, 'The failed reference gate must remain failed');
  assert.ok(Object.values(report.checks).length >= 11 && Object.values(report.checks).every(value => value === true));
  for (const key of ['trained', 'untrained', 'reloaded']) {
    assert.equal(report[key].episodes.length, 20);
    assert.deepEqual(report[key].episodes.map(row => row.seed), Array.from({ length: 20 }, (_, i) => 1000 + i));
    near(mean(report[key].episodes.map(row => finite(row.return))), report[key].mean_return, 'Reference mean ' + key);
  }
  near(report.trained.mean_return, 409.2, 'Reference trained return'); near(report.untrained.mean_return, 333.2, 'Reference untrained return');
  near(report.trained.mean_return - report.untrained.mean_return, report.mean_return_gain, 'Reference gain');
  near(report.mean_return_gain, 76, 'Failed gain remains 76');
  assert.deepEqual(report.trained.episodes, report.reloaded.episodes);
  assert.equal(report.gate.mean_gain_at_least_100, false);
  assert.equal(protocol.reference.required_gain, 100);
  return { overall_passed: false, complete: true, integrity_passed: true, trained_mean_return: 409.2,
    untrained_mean_return: 333.2, gain: 76, required_gain: 100, report_sha256: REFERENCE_SHA,
    limitation: 'CartPole missed its prespecified gain threshold (76 < 100), despite 409.2 trained return and passing integrity checks. No rerun or retuning. Native comparisons remain exploratory.' };
}

export function auditTraining(root, name, protocol, protocolSHA) {
  const folder = inside(root, name), report = readJson(inside(folder, 'result.json'));
  finiteTree(report);
  const development = name === 'elbow-development', seed = development ? 404 : Number(name.slice(7));
  const budget = development ? protocol.development_decisions : protocol.source_decisions_per_seed;
  const cases = development ? protocol.development_cases : protocol.source_cases;
  assert.equal(report.seed, seed); assert.equal(report.development_elbow_exposure, development);
  assert.equal(report.protocol_sha256, protocolSHA); assert.equal(report.requested_decisions, budget);
  assert.equal(hashFile(inside(folder, 'policy.zip')), hash(report.policy_sha256), 'Training checkpoint hash');
  assert.equal(hashFile(inside(folder, 'native-tape.json.gz')), hash(report.native_tape_sha256), 'Training native tape hash');
  hash(report.parameter_sha256, 'Source parameter hash');
  const tape = readGzip(inside(folder, 'native-tape.json.gz'));
  assert.equal(tape.native_stream_sha256, hash(report.native_stream_sha256));
  assert.deepEqual(tape.requests[0], { op: 'spec' });
  let decisions = 0, resets = 0, configured = false;
  const allowed = new Set(cases.map(canonical));
  for (const request of tape.requests.slice(1)) {
    if (request.op === 'configure') {
      assert.equal(configured, false, 'Training may not restart its batch'); configured = true;
      assert.equal(request.cases.length, 8); assert.equal(request.record, false);
      request.cases.forEach(config => assert.ok(allowed.has(canonical(config)), 'Training task outside allowed domain'));
    } else if (request.op === 'step') {
      assert.equal(configured, true); assert.equal(request.record, false);
      assert.equal(request.actions.length, 8); request.actions.forEach(action => assert.ok(Number.isInteger(action) && action >= 0 && action <= 2));
      decisions += request.actions.length;
    } else if (request.op === 'reset') {
      assert.equal(configured, true); assert.equal(request.record, false);
      assert.equal(new Set(request.slots.map(slot => slot.index)).size, request.slots.length);
      for (const slot of request.slots) {
        assert.ok(Number.isInteger(slot.index) && slot.index >= 0 && slot.index < 8);
        assert.ok(allowed.has(canonical(slot.case)), 'Reset task outside permitted training domain'); resets++;
      }
    } else assert.fail('Unexpected native training operation ' + request.op);
  }
  // A failed partial native batch has bounded uncertain exposure and cannot be
  // certified complete, even if all previous requests are structurally valid.
  if (report.exposure_exact) near(decisions, report.actual_decisions, 'Tape decision exposure', 0);
  assert.ok(report.actual_decisions <= budget && report.actual_decisions >= 0);
  assert.ok(report.native_steps <= report.actual_decisions * 5 && report.native_steps >= report.actual_decisions);
  near(resets, report.episodes, 'Reset/episode count', 0);
  const replay = readJson(inside(folder, 'native-replay.json'));
  finiteTree(replay);
  assert.equal(replay.tape_sha256, report.native_tape_sha256);
  assert.equal(replay.expected_stream_sha256, report.native_stream_sha256);
  const replayExact = replay.exact === true && replay.error === null && replay.requests_replayed === tape.requests.length &&
    replay.actual_stream_sha256 === replay.expected_stream_sha256;
  assert.equal(replay.exact, replayExact, 'Source replay claim');
  const logs = fs.readFileSync(inside(folder, 'learning.jsonl'), 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
  logs.forEach((row, i) => { finiteTree(row); assert.equal(row.decisions, (i + 1) * 2048); assert.equal(row.preceding_update_epochs, i * 4); });
  if (report.complete) {
    assert.equal(report.actual_decisions, budget); assert.equal(report.error, null); assert.equal(report.native_failure, null);
    assert.equal(report.parameters_finite, true); assert.equal(report.exposure_exact, true); assert.equal(report.stopped_by_clock, false);
    assert.ok(report.learning_wall_s <= protocol.wall_time_caps_s[development ? 'reference_development' : 'source_per_seed']);
    assert.equal(logs.length, budget / 2048); assert.equal(report.optimization_epochs, logs.length * 4);
    assert.equal(report.optimizer_steps, logs.length * 4 * 32);
  }
  return { name, ...report, native_replay: replay, audit_complete: report.complete === true && replayExact,
    audited_request_count: tape.requests.length, audited_reset_count: resets };
}

function localObservation(observation, time) {
  keys(observation, ['schema_version', 'reward_sensors', 'actuator_feedback'], 'Policy observation boundary');
  assert.equal(observation.schema_version, 'construction_observation_v2');
  assert.equal(observation.reward_sensors.length, 2); assert.equal(observation.actuator_feedback.length, 1);
  const sensors = new Map(observation.reward_sensors.map(sensor => [sensor.family_id, sensor]));
  assert.equal(sensors.size, 2);
  for (const [family, period, latency, fields] of [
    ['imu_6axis_v0', .01, .006, ['specific_force_m_s2', 'angular_velocity_rad_s']],
    ['ambient_light_v0', .1, .02, ['illuminance_lux', 'saturated']]]) {
    const sensor = sensors.get(family);
    keys(sensor, ['module_id', 'family_id', 'valid', 'sequence', 'sample_time_s', 'delivered_time_s', 'age_s', 'sample_period_s', 'latency_s', 'observations'], 'Local sensor boundary');
    keys(sensor.observations, fields, 'Local sensor channels');
    near(sensor.sample_period_s, period, 'Sample period'); near(sensor.latency_s, latency, 'Sensor latency');
    const sequence = Math.floor((time - latency + 1e-8) / period) + 1;
    assert.equal(sensor.valid, true); assert.equal(sensor.sequence, sequence, 'Sensor delivery cadence');
    near(sensor.sample_time_s, (sequence - 1) * period, 'Sample clock', 2e-6);
    near(sensor.delivered_time_s, sensor.sample_time_s + latency, 'Delivery clock', 2e-6);
    near(sensor.age_s, time - sensor.sample_time_s, 'Sample age', 2e-6);
  }
  const motor = observation.actuator_feedback[0];
  keys(motor, ['module_id', 'sku_id', 'valid', 'feedback'], 'Motor observation boundary');
  assert.equal(motor.module_id, 'motor-0001'); assert.equal(motor.sku_id, 'rotary_dc_gearmotor_v0'); assert.equal(motor.valid, true);
  keys(motor.feedback, ['position_rad', 'velocity_rad_s', 'current_a', 'bus_voltage_v', 'temperature_c', 'output_torque_nm',
    'load_impedance_nm_s_per_rad', 'stuck_score', 'stuck', 'fault_flags'], 'Motor channel boundary');
  assert.ok(Array.isArray(motor.feedback.fault_flags));
  for (const field of ['specific_force_m_s2', 'angular_velocity_rad_s']) assert.equal(sensors.get('imu_6axis_v0').observations[field].length, 3);
  const lux = sensors.get('ambient_light_v0').observations.illuminance_lux;
  assert.ok(lux >= 0 && lux <= 1000);
  return { motor: motor.feedback, light: sensors.get('ambient_light_v0'), imu: sensors.get('imu_6axis_v0') };
}

function displayPhysics(physics) {
  // Keep native geometry at every retained endpoint; no renderer reconstruction.
  const result = {};
  for (const key of ['schema', 'assembly', 'elapsed_s', 'sim_time', 'tick', 'geometry', 'segments', 'joints',
    'tip_position_m', 'sensor', 'actuator', 'command', 'reward', 'safety', 'light_sensor', 'sun']) result[key] = physics[key];
  return result;
}
function windowMetric(start, end) {
  return { start_s: start, end_s: end, available_duration_s: 0, native_reward: 0, light_reward: 0, imu_reward: 0,
    light_valid_duration_s: 0, lux_seconds: 0, acquisitions: new Set() };
}
function addHeldLight(window, start, end, sample) {
  const duration = Math.max(0, Math.min(end, window.end_s) - Math.max(start, window.start_s));
  if (duration && sample?.valid) {
    window.light_valid_duration_s += duration;
    window.lux_seconds += duration * sample.observations.illuminance_lux;
    window.acquisitions.add(sample.sequence);
  }
}
function finishWindow(window) {
  const { acquisitions, ...rest } = window;
  return { ...rest, held_mean_lux: window.light_valid_duration_s > 0 ? window.lux_seconds / window.light_valid_duration_s : null,
    distinct_held_acquisitions: acquisitions.size,
    complete: Math.abs(window.available_duration_s - (window.end_s - window.start_s)) < 1e-8 };
}

export function auditTrace(trace, record, config) {
  finiteTree(trace);
  assert.equal(trace.schema, 'actuation_transfer_trace_v1'); assert.deepEqual(trace.case, config);
  assert.ok(Array.isArray(trace.steps) && trace.steps.length <= 640);
  const initial = trace.initial_physics;
  if (!initial) {
    assert.equal(trace.steps.length, 0, 'Missing initial physics on nonempty trace');
    assert.ok(record.error, 'No initial state without a recorded failure');
    assert.equal(record.metrics.complete, false); assert.equal(record.metrics.decisions, 0);
    near(record.metrics.native_reward, 0, 'Empty failed opportunity return');
    return { metrics: { complete: false, decisions: 0, native_reward: 0, safety_stops: 0 }, frames: [], prefix: [], initial_physics: null };
  }
  assert.deepEqual(initial.assembly, config.assembly); assert.deepEqual(initial.sun, config.sun);
  near(initial.elapsed_s, 0, 'Initial time'); near(initial.reward.cumulative, 0, 'Initial return');
  near(initial.diagnostics.electrical_energy_j, 0, 'Initial energy');
  assert.equal(initial.safety.veto, false); assert.deepEqual(initial.safety.flags, []);
  let time = 0, cumulative = 0, lightReward = 0, imuReward = 0, previousEffort = 0, previousObservation = null;
  let energy = 0, nativeSteps = 0, safetyStops = 0, maxSpeed = 0, maxCurrent = 0, maxTemperature = 0, maxForce = 0, maxGyro = 0;
  let previousLight = null;
  const sensorWindows = [windowMetric(0, 12.8), windowMetric(51.2, 64)];
  const allFlags = new Set(), endpointReturns = [], prefix = [];
  const frames = [{ time_s: 0, physics: displayPhysics(initial), cumulative_reward: 0, action: null, executed_effort: 0 }];
  for (const [i, entry] of trace.steps.entries()) {
    const row = entry.result;
    assert.equal(entry.slot, 0); assert.equal(entry.decision, i + 1); assert.ok(Number.isInteger(entry.action) && entry.action >= 0 && entry.action <= 2);
    assert.equal(safetyStops, 0, 'A target must stop after its first native safety stop');
    assert.ok(Number.isInteger(row.native_steps) && row.native_steps >= 1 && row.native_steps <= 5);
    assert.equal(row.efforts.length, row.native_steps); assert.equal(row.native_transitions.length, row.native_steps);
    hash(row.transition_sha256, 'Native transition commitment');
    let reward = 0, light = 0, imu = 0, speed = 0, current = 0, temperature = 0, terminal = false;
    for (let j = 0; j < row.native_steps; j++) {
      const transition = row.native_transitions[j], effort = row.efforts[j], request = REQUESTS[entry.action];
      let governed = 0;
      if (previousObservation) {
        const velocity = previousObservation.actuator_feedback[0].feedback.velocity_rad_s, sign = Math.sign(request);
        const target = request ? sign * Math.min(Math.abs(request), .05 * Math.max(0, 3 - sign * velocity)) : 0;
        governed = Math.max(previousEffort - .1, Math.min(previousEffort + .1, target));
      }
      near(effort, governed, 'Governor per native control', 1e-12);
      near(transition.info.elapsed_control_dt_s, .02, 'Native duration'); assert.equal(transition.info.physics_steps_executed, 10);
      assert.equal(transition.truncated, false);
      const before = time; time = (++nativeSteps) * .02;
      const observed = localObservation(transition.observation, time);
      assert.equal(transition.reward_components.length, 2);
      const components = new Map(transition.reward_components.map(component => [component.family_id, component.transition_reward]));
      assert.equal(components.size, 2); assert.ok(components.has('ambient_light_v0') && components.has('imu_6axis_v0'));
      const l = finite(components.get('ambient_light_v0')), m = finite(components.get('imu_6axis_v0'));
      near(transition.reward, l + m, 'Native sensor reward sum', 1e-12);
      assert.ok(Math.abs(l) <= .02 + 1e-12 && Math.abs(m) <= .02 + 1e-12, 'Native reward bound');
      reward += transition.reward; light += l; imu += m;
      speed = Math.max(speed, Math.abs(observed.motor.velocity_rad_s)); current = Math.max(current, Math.abs(observed.motor.current_a));
      temperature = Math.max(temperature, observed.motor.temperature_c);
      maxForce = Math.max(maxForce, Math.hypot(...observed.imu.observations.specific_force_m_s2));
      maxGyro = Math.max(maxGyro, Math.hypot(...observed.imu.observations.angular_velocity_rad_s));
      terminal = transition.terminated || transition.safety.veto || observed.motor.fault_flags.length > 0;
      assert.equal(typeof transition.terminated, 'boolean'); assert.equal(typeof transition.safety.veto, 'boolean');
      for (const flag of [...transition.safety.flags, ...observed.motor.fault_flags]) allFlags.add(flag);
      if (terminal) assert.equal(j, row.native_steps - 1, 'No native controls after stop');
      for (const window of sensorWindows) {
        const overlap = Math.max(0, Math.min(time, window.end_s) - Math.max(before, window.start_s));
        if (overlap > 1e-10) {
          near(overlap, .02, 'Declared observation window aligns with native controls');
          window.available_duration_s += overlap; window.native_reward += transition.reward; window.light_reward += l; window.imu_reward += m;
        }
        // Reward integration acquires/delivers before each .002s physics vote.
        // A newly delivered light value therefore owns that delivery tick's dt.
        if (!previousLight || observed.light.sequence !== previousLight.sequence) {
          assert.ok(!previousLight || observed.light.sequence === previousLight.sequence + 1, 'Light acquisition sequence jump');
          const change = observed.light.delivered_time_s - .002;
          assert.ok(change >= before - 2e-6 && change <= time + 2e-6, 'Light delivery inside native interval');
          addHeldLight(window, before, change, previousLight); addHeldLight(window, change, time, observed.light);
        } else {
          assert.deepEqual(observed.light.observations, previousLight.observations, 'Held light packet changed without acquisition');
          addHeldLight(window, before, time, observed.light);
        }
      }
      previousLight = observed.light; previousObservation = transition.observation; previousEffort = effort;
    }
    assert.equal(row.terminated, terminal); if (terminal) safetyStops++;
    if (!terminal) assert.equal(row.native_steps, 5, 'Short native decision without stop');
    near(row.elapsed_s, row.native_steps * .02, 'Macro duration'); near(row.time_s, time, 'Macro clock');
    for (const [field, value] of Object.entries({ reward, light_reward: light, imu_reward: imu, max_speed_rad_s: speed,
      max_current_a: current, max_temperature_c: temperature, executed_effort: previousEffort, previous_request: REQUESTS[entry.action] })) near(row[field], value, field, 1e-12);
    assert.deepEqual(row.observation, previousObservation, 'Macro observation must equal last native observation');
    assert.deepEqual(row.physics.assembly, config.assembly); assert.deepEqual(row.physics.sun, config.sun);
    near(row.physics.elapsed_s, time, 'Physical clock'); near(row.physics.tick, nativeSteps * 10, 'Physics ticks', 0);
    cumulative += reward; lightReward += light; imuReward += imu; endpointReturns.push(cumulative);
    near(row.physics.reward.cumulative, cumulative, 'Cumulative physical return', 1e-10);
    near(row.energy_j, row.physics.diagnostics.electrical_energy_j, 'Native recorded energy', 1e-12);
    // Native V*I is signed; regeneration can reduce cumulative electrical energy.
    energy = row.energy_j;
    assert.deepEqual(row.safety, row.physics.safety); assert.equal(row.safety.veto, row.native_transitions.at(-1).safety.veto);
    maxSpeed = Math.max(maxSpeed, speed); maxCurrent = Math.max(maxCurrent, current); maxTemperature = Math.max(maxTemperature, temperature);
    if (i < 64) prefix.push(sha(canonical(entry)));
    if ((i + 1) % 2 === 0 || i === 0 || i === trace.steps.length - 1) frames.push({ time_s: row.time_s,
      physics: displayPhysics(row.physics), cumulative_reward: cumulative, action: entry.action, executed_effort: row.executed_effort });
  }
  const physicalComplete = trace.steps.length === 640 && safetyStops === 0;
  const result = { complete: physicalComplete && record.error === null && record.native_failure === null,
    physical_horizon_complete: physicalComplete, decisions: trace.steps.length, native_steps: nativeSteps, physics_steps: nativeSteps * 10,
    duration_s: time, native_reward: cumulative, light_reward: lightReward, imu_reward: imuReward, energy_j: energy,
    safety_stops: safetyStops, safety_flags: [...allFlags].sort(), max_speed_rad_s: maxSpeed, max_current_a: maxCurrent,
    max_temperature_c: maxTemperature, max_imu_force_m_s2: maxForce, max_imu_angular_speed_rad_s: maxGyro,
    windows: [64, 128, 320, 640].map(count => ({ seconds: count * .1,
      native_reward: endpointReturns[Math.min(trace.steps.length, count) - 1] ?? 0, available_decisions: Math.min(trace.steps.length, count) })),
    observation_windows: sensorWindows.map(finishWindow),
    horizon_completed_diagnostic: cumulative - 2 * (64 - time),
    horizon_completed_diagnostic_definition: 'Evaluator-only lower-bound completion at -2 reward per unobserved second; never a native reward or PPO objective.' };
  const recorded = record.metrics;
  for (const key of ['complete', 'physical_horizon_complete']) if (key in recorded) assert.equal(recorded[key], result[key], 'Recorded ' + key);
  for (const key of ['decisions', 'native_steps', 'duration_s', 'native_reward', 'light_reward', 'imu_reward', 'energy_j',
    'safety_stops', 'max_speed_rad_s', 'max_current_a', 'max_temperature_c']) {
    if (trace.steps.length || key in recorded) near(recorded[key], result[key], 'Recomputed ' + key);
  }
  if (trace.steps.length) for (let i = 0; i < 4; i++) {
    near(recorded.windows[i].seconds, result.windows[i].seconds, 'Window horizon');
    near(recorded.windows[i].native_reward, result.windows[i].native_reward, 'Window return');
    near(recorded.windows[i].available_decisions, result.windows[i].available_decisions, 'Window exposure', 0);
  }
  return { metrics: result, frames, prefix, initial_physics: displayPhysics(initial) };
}

function expectedTrials(protocol) {
  const expected = [];
  const append = (item, seed, arm, development = false) => {
    const config = clone(item.case);
    if (arm === 'source_frozen' || arm === 'source_quiet') config.assembly.powered_hinge = 0;
    expected.push({ id: `${item.id}--${seed}--${arm}`, case_id: item.id, source_seed: seed, arm, case: config, development });
  };
  for (const item of protocol.target_cases) {
    for (const seed of SEEDS) for (const arm of [...ARMS, 'source_frozen']) append(item, seed, arm);
    for (const arm of PROBES) append(item, 0, arm);
  }
  protocol.development_cases.forEach((config, i) => {
    for (const arm of ['development_frozen', 'quiet']) append({ id: 'elbow-development-' + i, case: config }, 404, arm, true);
  });
  assert.equal(expected.length, 112);
  return expected;
}

function auditTrial(root, expected, protocolSHA, training) {
  const folder = inside(root, 'trials/' + expected.id), record = readJson(inside(folder, 'result.json'));
  finiteTree(record);
  assert.equal(record.schema, 'actuation_transfer_trial_v1');
  for (const key of ['id', 'case_id', 'source_seed', 'arm', 'case']) assert.deepEqual(record[key], expected[key], 'Trial membership ' + key);
  assert.equal(record.protocol_sha256, protocolSHA); assert.equal(record.new_interaction_budget, 640);
  assert.equal(record.post_final_update_evaluation, false);
  assert.ok(record.error === null || (typeof record.error === 'string' && record.error.length > 0), 'Explicit trial error status required');
  assert.ok('native_failure' in record, 'Explicit native failure status required');
  assert.equal(hashFile(inside(folder, 'trace.json.gz')), hash(record.trace_sha256), 'Compressed native trace hash');
  const source = training.get(expected.development ? 'elbow-development' : 'source-' + expected.source_seed);
  const updating = ['adapt', 'scratch'].includes(expected.arm), frozen = ['frozen', 'source_frozen', 'development_frozen'].includes(expected.arm);
  if (frozen || expected.arm === 'adapt') {
    assert.ok(source, 'Missing source checkpoint provenance'); assert.equal(record.source_policy_sha256, source.policy_sha256);
    assert.equal(record.initial_parameter_sha256, source.parameter_sha256, 'Transferred parameters differ from source');
  } else assert.equal(record.source_policy_sha256, null);
  if (frozen) {
    assert.equal(hash(record.initial_parameter_sha256), hash(record.final_parameter_sha256), 'Frozen policy changed');
    assert.equal(record.optimization_epochs, 0); assert.equal(record.optimizer_steps, 0);
  } else if (updating) {
    hash(record.initial_parameter_sha256); hash(record.final_parameter_sha256);
    const after = inside(folder, 'policy_after.zip');
    if (!record.error) assert.ok(fs.existsSync(after), 'Missing adapted checkpoint');
    if (fs.existsSync(after)) record.policy_after_sha256 = hashFile(after);
    const logsPath = inside(folder, 'learning.jsonl');
    const logs = fs.existsSync(logsPath) ? fs.readFileSync(logsPath, 'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse) : [];
    logs.forEach((row, i) => { finiteTree(row); assert.equal(row.decisions, (i + 1) * 64); assert.equal(row.preceding_update_epochs, i * 4); });
    if (record.metrics.complete) {
      assert.equal(logs.length, 10); assert.equal(record.optimization_epochs, 40); assert.equal(record.optimizer_steps, 40);
    }
    record.update_locations_decisions = logs.map(row => row.decisions);
  } else {
    assert.equal(record.initial_parameter_sha256, null); assert.equal(record.final_parameter_sha256, null);
    assert.equal(record.optimization_epochs, 0); assert.equal(record.optimizer_steps, 0);
  }
  const trace = readGzip(inside(folder, 'trace.json.gz'));
  for (const [i, entry] of trace.steps.entries()) {
    const arm = expected.arm;
    if (arm === 'quiet' || arm === 'source_quiet') assert.equal(entry.action, 1, 'Quiet action schedule');
    else if (arm === 'negative') assert.equal(entry.action, 0, 'Negative action schedule');
    else if (arm === 'positive') assert.equal(entry.action, 2, 'Positive action schedule');
    else if (arm.startsWith('rhythm_')) {
      const period = arm === 'rhythm_2s' ? 20 : 40; assert.equal(entry.action, i % period < period / 2 ? 0 : 2, 'Probe action schedule');
    }
  }
  const audited = auditTrace(trace, record, expected.case);
  const replayExact = record.replay.exact === true && record.replay.error === null && record.replay.decisions === trace.steps.length;
  assert.equal(record.replay.exact, replayExact, 'Recorded replay claim');
  audited.metrics.integrity_verified = true;
  return { ...audited, record: { ...record, metrics: audited.metrics, replay: { ...record.replay, exact: replayExact } } };
}

export function summarize(protocol, trials, training, prefixes, issues) {
  const find = (caseID, seed, arm) => trials.find(row => row.case_id === caseID && row.source_seed === seed && row.arm === arm);
  const safe = row => row?.metrics?.complete === true && row.metrics.integrity_verified === true && row.metrics.safety_stops === 0 &&
    row.metrics.safety_flags?.length === 0 && row.error === null && row.native_failure === null && row.replay?.exact === true;
  const difference = (a, b) => Number.isFinite(a?.metrics?.native_reward) && Number.isFinite(b?.metrics?.native_reward) ? a.metrics.native_reward - b.metrics.native_reward : null;
  const pairs = [];
  for (const seed of SEEDS) for (const item of protocol.target_cases) {
    const adapt = find(item.id, seed, 'adapt'), scratch = find(item.id, seed, 'scratch'), frozen = find(item.id, seed, 'frozen');
    const quiet = find(item.id, 0, 'quiet'), source = find(item.id, seed, 'source_frozen'), sourceQuiet = find(item.id, 0, 'source_quiet');
    const a = prefixes.get(adapt?.id), f = prefixes.get(frozen?.id);
    const prefixCount = Math.min(a?.length ?? 0, f?.length ?? 0, 64);
    const prefixIdentical = prefixCount === 64 && a.slice(0, 64).every((value, i) => value === f[i]);
    const adaptQuiet = difference(adapt, quiet), sourceQuietDiff = difference(source, sourceQuiet);
    const availableProbes = ['quiet', 'negative', 'positive', 'rhythm_2s', 'rhythm_4s'].map(arm => find(item.id, 0, arm)).filter(safe);
    const witness = availableProbes.sort((x, y) => y.metrics.native_reward - x.metrics.native_reward)[0];
    pairs.push({ case_id: item.id, source_seed: seed, adapt_minus_scratch: difference(adapt, scratch),
      frozen_minus_scratch: difference(frozen, scratch), adapt_minus_frozen: difference(adapt, frozen),
      adapt_minus_quiet: adaptQuiet, source_frozen_minus_source_quiet: sourceQuietDiff,
      adapt_minus_safe_probe_witness: difference(adapt, witness), safe_probe_witness: witness?.arm ?? null,
      target_triple_complete_safe: [adapt, scratch, frozen].every(safe),
      prefix_decisions_compared: prefixCount, frozen_adapt_first_64_identical: prefixIdentical,
      useful_behavior: [adapt, quiet, source, sourceQuiet].every(safe) && adaptQuiet > 0 && sourceQuietDiff > 0,
      actual_duration_s: { frozen: frozen?.metrics?.duration_s ?? null, adapt: adapt?.metrics?.duration_s ?? null, scratch: scratch?.metrics?.duration_s ?? null } });
  }
  const seedMeans = SEEDS.map(seed => {
    const selected = pairs.filter(pair => pair.source_seed === seed), values = selected.map(pair => pair.adapt_minus_scratch);
    return { source_seed: seed, cases: selected.length, mean_adapt_minus_scratch: values.every(Number.isFinite) ? mean(values) : null,
      all_target_triples_complete_safe: selected.every(pair => pair.target_triple_complete_safe) };
  });
  const values = seedMeans.map(row => row.mean_adapt_minus_scratch), med = values.every(Number.isFinite) ? median(values) : null;
  const allTargets = trials.filter(row => ARMS.includes(row.arm) && SEEDS.includes(row.source_seed));
  const integrity = issues.length === 0 && trials.length === 112 && trials.every(row => row.metrics?.integrity_verified && row.replay?.exact) &&
    training.size === 4 && [...training.values()].every(row => row.native_replay.exact);
  const prerequisites = training.size === 4 && [...training.values()].every(row => row.audit_complete);
  const allTargetSafe = allTargets.length === 54 && allTargets.every(safe), prefixesPass = pairs.every(pair => pair.frozen_adapt_first_64_identical);
  const positiveSeeds = values.filter(value => Number.isFinite(value) && value > 0).length;
  const passed = integrity && prerequisites && allTargetSafe && prefixesPass && med > 0 && positiveSeeds >= 2;
  return { evidence_status: 'exploratory', integrity_passed: integrity, prerequisite_training_complete: prerequisites,
    transfer_gate_passed: passed, target_arms_complete_safe: allTargetSafe, target_arm_count: allTargets.length,
    expected_target_arm_count: 54, paired_case_count: pairs.length, frozen_adapt_prefixes_passed: prefixesPass,
    median_seed_mean_adapt_minus_scratch: med, positive_seed_means: positiveSeeds, required_positive_seed_means: 2,
    seed_means: seedMeans, pairs, useful_behavior_pairs: pairs.filter(pair => pair.useful_behavior).length,
    useful_behavior_all_pairs: passed && pairs.every(pair => pair.useful_behavior),
    conclusion: passed ? 'The bounded exploratory transfer gate passed. Check the separate quiet/source controls before calling the behavior useful.' :
      'The bounded exploratory transfer gate did not pass. All recorded returns, failures and controls remain part of this result.',
    exposure: { actual_recorded_trial_decisions: trials.reduce((sum, row) => sum + (row.metrics?.decisions ?? 0), 0),
      actual_training_decisions: [...training.values()].reduce((sum, row) => sum + row.actual_decisions, 0),
      trial_replay_decisions: trials.reduce((sum, row) => sum + (row.replay?.decisions ?? 0), 0),
      training_replay_requests: [...training.values()].reduce((sum, row) => sum + row.native_replay.requests_replayed, 0) },
    limitations: ['Three source training seeds and six declared cases; no arbitrary topology, contact or hardware generalization claim.',
      'CartPole reference gain gate remains failed: trained 409.2, untrained 333.2, gain 76, required 100; native results are exploratory.',
      'Native action replay verifies the simulator response to recorded requests, not replay of neural action sampling or PPO optimizer computation.',
      'Compressed traces and archived files are independently hashed. Original C++ transition hashes are checked through recorded exact native replay, not reconstructed from Python-reserialized number spellings.',
      'Net electrical energy is independently cross-checked against native cumulative diagnostics; signed V*I can decrease through regeneration. The .002-second motor power is not separately serialized for reintegration.',
      'Light statistics integrate held packets using native delivery ticks and .002-second physics vote timing; repeated packets are not independent acquisitions.',
      'Saved policies receive their final PPO update after the final measured decision; no post-final-update performance is implied.'] };
}

function exportWeb(webRoot, auditFolder, data, playableIDs) {
  fs.mkdirSync(webRoot, { recursive: true });
  const target = inside(webRoot, 'transfer-data.json'), traceRoot = inside(webRoot, 'transfer-traces');
  if (fs.existsSync(target)) assert.equal(readJson(target).schema, 'actuation_transfer_view_v1', 'Refuse to replace unrelated web data');
  fs.mkdirSync(traceRoot, { recursive: true });
  for (const id of playableIDs) {
    const filename = inside(traceRoot, id + '.json');
    if (fs.existsSync(filename)) {
      const old = readJson(filename);
      assert.equal(old.schema, 'actuation_transfer_playback_v1'); assert.equal(old.id, id);
    }
  }
  for (const id of playableIDs) fs.copyFileSync(inside(auditFolder, 'playback/' + id + '.json'), inside(traceRoot, id + '.json'));
  fs.writeFileSync(target, JSON.stringify(data) + '\n');
}

export function run({ input, webRoot, output }) {
  input = path.resolve(input); webRoot = path.resolve(webRoot); output = path.resolve(output);
  for (const destination of [webRoot, output]) {
    assert.ok(destination !== input && !destination.startsWith(input + path.sep) && !input.startsWith(destination + path.sep), 'Outputs must not overlap immutable source evidence');
  }
  assert.ok(!fs.existsSync(output), 'Audit output must be a new directory; use --output for another audit');
  fs.mkdirSync(output, { recursive: true });
  fs.copyFileSync(fileURLToPath(import.meta.url), path.join(output, 'audit-source.mjs'), fs.constants.COPYFILE_EXCL);
  const issues = [], training = new Map(), prefixes = new Map(), trials = [], playableIDs = [];
  const checked = (label, action) => { try { return action(); } catch (error) { issues.push({ scope: label, error: error.message }); return null; } };
  const protocolSHA = checked('protocol hash', () => {
    const actual = hashFile(inside(input, 'protocol.json'));
    assert.equal(actual, fs.readFileSync(inside(input, 'protocol.sha256'), 'utf8').trim());
    assert.equal(actual, LOCKED_PROTOCOL_SHA, 'The declared pre-learning protocol changed'); return actual;
  });
  const protocol = checked('protocol contract', () => validateProtocol(readJson(inside(input, 'protocol.json'))));
  if (!protocol || !protocolSHA) {
    const report = { schema: 'actuation_transfer_audit_v1', integrity_passed: false, issues };
    writeNew(path.join(output, 'audited-results.json'), report); return { report, exitCode: 1 };
  }
  const archive = checked('source archive', () => auditArchive(input, protocolSHA));
  const reference = checked('CartPole reference', () => auditReference(input, protocol));
  const runtime = checked('pinned CPU runtime', () => {
    const value = readJson(inside(input, 'runtime.json'));
    assert.deepEqual(value, { python: '3.12.14', torch: '2.8.0+cpu', sb3: '2.9.0', numpy: '2.2.6', threads: 1, device: 'cpu' });
    return { ...value, sha256: hashFile(inside(input, 'runtime.json')) };
  });
  for (const name of ['elbow-development', ...SEEDS.map(seed => 'source-' + seed)]) {
    const result = checked(name, () => auditTraining(input, name, protocol, protocolSHA)); if (result) training.set(name, result);
  }
  const expected = expectedTrials(protocol);
  checked('trial membership', () => assert.deepEqual(fs.readdirSync(inside(input, 'trials'), { withFileTypes: true }).map(entry => {
    assert.ok(entry.isDirectory() && !entry.isSymbolicLink()); return entry.name;
  }).sort(), expected.map(row => row.id).sort(), 'Expected exactly 108 target/control and four development trials'));
  const executionErrors = fs.readdirSync(input).filter(name => /^execution-error-.*\.json$/.test(name));
  if (executionErrors.length) issues.push({ scope: 'recorded execution failures', files: executionErrors, error: 'Recorded orchestration errors require explicit review; the audit cannot pass automatically.' });
  for (const item of expected) {
    const result = checked(item.id, () => auditTrial(input, item, protocolSHA, training));
    if (!result) {
      trials.push({ id: item.id, case_id: item.case_id, source_seed: item.source_seed, arm: item.arm,
        metrics: { complete: false, integrity_verified: false }, replay: { exact: false }, error: 'Evidence audit failed; see audit issues.', native_failure: null, trace_url: null });
      continue;
    }
    prefixes.set(item.id, result.prefix);
    const { record } = result;
    const playable = PLAYABLE.has(item.arm) && result.initial_physics !== null;
    record.trace_url = playable ? 'transfer-traces/' + item.id + '.json' : null;
    trials.push(record);
    if (playable) {
      writeNew(inside(output, 'playback/' + item.id + '.json'), { schema: 'actuation_transfer_playback_v1', generated_by: GENERATOR,
        id: item.id, protocol_sha256: protocolSHA, source_trace_sha256: record.trace_sha256,
        sampling: 'Native decision endpoints at 5 Hz, plus the first and last decision; hold each frame, never interpolate physics.',
        initial_physics: result.initial_physics, frames: result.frames });
      playableIDs.push(item.id);
    }
  }
  const summary = summarize(protocol, trials, training, prefixes, issues);
  const provenance = { report_path: path.relative(PROJECT, input).replaceAll('\\', '/'), protocol_sha256: protocolSHA,
    archive, runtime, audit_tool_sha256: hashFile(fileURLToPath(import.meta.url)), audited_at_utc: new Date().toISOString() };
  const development = { training: training.get('elbow-development') ?? null,
    trials: trials.filter(row => row.source_seed === 404), excluded_from_transfer_score: true,
    description: 'Explicitly elbow-exposed [3,3] reference, trained for 32,768 decisions; larger budget and separate weights.' };
  development.comparisons = protocol.development_cases.map((_, i) => {
    const rows = development.trials.filter(row => row.case_id === 'elbow-development-' + i);
    const active = rows.find(row => row.arm === 'development_frozen'), quiet = rows.find(row => row.arm === 'quiet');
    const available = [active, quiet].every(row => Number.isFinite(row?.metrics?.native_reward));
    return { case_id: 'elbow-development-' + i, frozen_minus_quiet: available ? active.metrics.native_reward - quiet.metrics.native_reward : null,
      complete_safe: [active, quiet].every(row => row?.metrics?.complete && row.metrics.safety_stops === 0 && row.replay?.exact) };
  });
  const report = { schema: 'actuation_transfer_audit_v1', provenance, protocol, summary, issues,
    source_training: SEEDS.map(seed => training.get('source-' + seed) ?? null), trials, development, reference };
  writeNew(inside(output, 'audited-results.json'), report);
  const cases = protocol.target_cases.map(item => ({ ...item, title: item.id.split('-').map(word => word[0].toUpperCase() + word.slice(1)).join(' '),
    description: `Links ${item.case.assembly.segments.join(' + ')} studs; ${item.case.assembly.blocks.length ? 'one added block' : 'no added blocks'}; sun ${item.id.endsWith('left') ? 'left' : 'right'}. Target motor at elbow; source motor at base.` }));
  const webData = { schema: 'actuation_transfer_view_v1', generated_by: GENERATOR, provenance, protocol, summary,
    cases, seeds: SEEDS, default_case_id: cases[0].id, default_seed: 101,
    trials: trials.filter(row => row.source_seed !== 404), development, reference,
    source_training: report.source_training };
  // Scientific negative outcomes can be exported. Corrupt or unauditable inputs
  // cannot silently replace the last verified browser record.
  if (!issues.length) exportWeb(webRoot, output, webData, playableIDs);
  return { report, exitCode: issues.length ? 1 : 0, exported: !issues.length, playable_traces: playableIDs.length };
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  try {
    const args = {};
    for (let i = 2; i < process.argv.length; i += 2) {
      const key = process.argv[i]; assert.ok(['--input', '--web-root', '--output'].includes(key) && process.argv[i + 1], 'Usage: node tools/summarize-actuation-transfer.mjs --input DIR --web-root web [--output NEW_DIR]');
      assert.ok(!(key in args), 'Duplicate argument'); args[key] = process.argv[i + 1];
    }
    assert.ok(args['--input'] && args['--web-root'], '--input and --web-root are required');
    const result = run({ input: args['--input'], webRoot: args['--web-root'], output: args['--output'] ?? path.join(PROJECT, 'build/actuation-transfer-audit') });
    console.log(JSON.stringify({ audit_output: args['--output'] ?? 'build/actuation-transfer-audit', exported: result.exported ?? false,
      playable_traces: result.playable_traces ?? 0, integrity_passed: result.report.summary?.integrity_passed ?? false,
      transfer_gate_passed: result.report.summary?.transfer_gate_passed ?? false,
      median_seed_mean_adapt_minus_scratch: result.report.summary?.median_seed_mean_adapt_minus_scratch ?? null,
      issues: result.report.issues }, null, 2));
    process.exitCode = result.exitCode;
  } catch (error) { console.error(error.stack); process.exitCode = 1; }
}
