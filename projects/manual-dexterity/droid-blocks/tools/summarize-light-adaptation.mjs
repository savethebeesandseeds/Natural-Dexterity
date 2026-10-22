import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REFERENCE = 'artifacts/reports/light-learning-v2-development-20260905/report.json';
const REFERENCE_SHA = '931a26b17b399456bb5461293e382ae9880486f1b38157007500eead65998230';
const SHA = /^[a-f0-9]{64}$/;
const DIFFERENCES = {
  after_move_integrated_sensor_reward: ['after_move', 'integrated_sensor_reward'],
  early_after_move_mean_received_lux: ['early_after_move_20s', 'mean_received_lux'],
  late_after_move_mean_received_lux: ['late_after_move_20s', 'mean_received_lux'],
  after_move_electrical_energy_j: ['after_move', 'electrical_energy_j'],
  after_move_native_veto_steps: ['after_move', 'native_veto_steps'],
  after_move_feedback_flag_steps: ['after_move', 'feedback_flag_steps'],
};
const CHECKS = [
  'identical_prefix_commands', 'identical_prefix_observations', 'identical_prefix_full_transitions',
  'identical_prefix_public_physical_snapshot', 'identical_prefix_controller_state_fingerprint',
  'identical_prefix_controller_diagnostics', 'continued_matches_archived_v2_commands',
  'continued_matches_archived_v2_observation_hash', 'continued_matches_archived_v2_return',
  'continued_matches_archived_v2_metrics', 'frozen_parameters_unchanged',
  'continued_full_replay', 'frozen_full_replay',
];
const ADDITIVE = ['steps', 'duration_s', 'integrated_sensor_reward', 'integrated_light_reward',
  'integrated_imu_reward', 'integrated_received_lux_s', 'electrical_energy_j',
  'native_veto_steps', 'feedback_flag_steps', 'unique_light_samples'];
const MAXIMA = ['max_current_a', 'max_temperature_c', 'max_motor_speed_rad_s'];

const readJson = (filename) => JSON.parse(fs.readFileSync(filename, 'utf8').replace(/^\uFEFF/, ''));
const hashFile = (filename) => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
function safeJoin(root, relative) {
  assert.equal(typeof relative, 'string', 'Snapshot path must be a string');
  assert.ok(relative.length && !path.isAbsolute(relative) && !relative.includes(':') &&
    !relative.split(/[\\/]/).includes('..'), `Unsafe snapshot path: ${relative}`);
  const resolved = path.resolve(root, relative);
  const local = path.relative(path.resolve(root), resolved);
  assert.ok(local && !local.startsWith(`..${path.sep}`) && local !== '..' && !path.isAbsolute(local),
    `Snapshot escaped report directory: ${relative}`);
  return resolved;
}
function finite(value, label) {
  assert.equal(typeof value, 'number', `${label} must be numeric`);
  assert.ok(Number.isFinite(value), `${label} must be finite`);
  return value;
}
function near(actual, expected, label) {
  finite(actual, label);
  finite(expected, label);
  assert.ok(Math.abs(actual - expected) <= 1e-8 * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} differs from ${expected}`);
}
function sha(value, label) {
  assert.equal(typeof value, 'string', `${label} must be a string`);
  assert.match(value, SHA, `${label} is not SHA-256`);
}
function median(values) {
  assert.ok(values.length, 'Cannot take an empty median');
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function verifyMetrics(metrics, expectedSteps, label) {
  assert.ok(metrics && typeof metrics === 'object', `${label} is missing`);
  for (const field of [...ADDITIVE, ...MAXIMA, 'mean_sensor_reward_rate', 'mean_received_lux']) finite(metrics[field], `${label}.${field}`);
  assert.equal(metrics.steps, expectedSteps, `${label} has the wrong step budget`);
  near(metrics.duration_s, expectedSteps * .02, `${label} duration`);
  near(metrics.integrated_sensor_reward, metrics.integrated_light_reward + metrics.integrated_imu_reward, `${label} sensor votes`);
  near(metrics.mean_sensor_reward_rate, metrics.integrated_sensor_reward / metrics.duration_s, `${label} mean reward`);
  near(metrics.mean_received_lux, metrics.integrated_received_lux_s / metrics.duration_s, `${label} mean lux`);
  for (const field of ['native_veto_steps', 'feedback_flag_steps', 'unique_light_samples']) {
    assert.ok(Number.isSafeInteger(metrics[field]) && metrics[field] >= 0 && metrics[field] <= expectedSteps,
      `${label}.${field} is an invalid count`);
  }
  assert.equal(metrics.unique_light_samples, expectedSteps / 5, `${label} sample-and-hold count differs`);
  assert.ok(metrics.electrical_energy_j >= -1e-10 && metrics.integrated_received_lux_s >= 0,
    `${label} has negative energy or illuminance exposure`);
}
function verifyPartition(whole, parts, label) {
  for (const field of ADDITIVE) near(whole[field], parts.reduce((sum, part) => sum + part[field], 0), `${label}.${field}`);
  for (const field of MAXIMA) near(whole[field], Math.max(...parts.map((part) => part[field])), `${label}.${field}`);
}
function verifyBranch(branch, name, seed) {
  assert.equal(branch.branch, name);
  assert.equal(branch.mode, 'learner');
  assert.equal(branch.seed, seed);
  assert.equal(branch.completed, true, `${name} is incomplete`);
  assert.equal(branch.completed_steps, 6000);
  assert.equal(branch.error, '');
  assert.equal(branch.exact_controller_and_observation_replay, true, `${name} controller/observation replay failed`);
  assert.equal(branch.exact_full_transition_replay, true, `${name} reward/transition replay failed`);
  assert.equal(branch.replay.completed_steps, 6000);
  assert.equal(branch.replay.error, '');
  for (const field of ['effort_trace_sha256', 'observation_trace_sha256', 'transition_trace_sha256',
    'final_public_physical_snapshot_sha256']) sha(branch[field], `${name}.${field}`);
  assert.equal(branch.efforts.length, 6000);
  let previousEffort = 0;
  for (let i = 0; i < branch.efforts.length; ++i) {
    const effort = finite(branch.efforts[i], `${name} effort ${i}`);
    assert.ok(Math.abs(effort) <= .15 + 1e-12 && Math.abs(effort - previousEffort) <= .1 + 1e-12,
      `${name} command/slew bound exceeded at ${i}`);
    if (i < 5) assert.equal(effort, 0, 'Quiet startup moved');
    previousEffort = effort;
  }
  verifyMetrics(branch.whole_run, 6000, `${name}.whole_run`);
  for (const group of ['before_move', 'after_move']) verifyMetrics(branch[group], 3000, `${name}.${group}`);
  for (const group of ['first_phase_final_20s', 'early_after_move_20s', 'middle_after_move_20s', 'late_after_move_20s'])
    verifyMetrics(branch[group], 1000, `${name}.${group}`);
  verifyPartition(branch.whole_run, [branch.before_move, branch.after_move], `${name} before+after`);
  verifyPartition(branch.after_move, [branch.early_after_move_20s, branch.middle_after_move_20s, branch.late_after_move_20s], `${name} postmove windows`);
  assert.equal(branch.curve.length, 1200, `${name} display curve is incomplete`);
  for (let i = 0; i < branch.curve.length; ++i) {
    const point = branch.curve[i];
    near(point.time_s, (i + 1) * .1, `${name} curve time`);
    for (const key of ['illuminance_lux', 'sensor_reward_rate', 'effort', 'cumulative_sensor_reward']) finite(point[key], `${name} curve ${key}`);
    assert.equal(point.effort, branch.efforts[(i + 1) * 5 - 1], `${name} curve command differs`);
    assert.ok(point.illuminance_lux >= 0, `${name} curve has negative lux`);
  }
  near(branch.curve[599].cumulative_sensor_reward, branch.before_move.integrated_sensor_reward, `${name} intervention cumulative reward`);
  near(branch.curve.at(-1).cumulative_sensor_reward, branch.whole_run.integrated_sensor_reward, `${name} final cumulative reward`);
  const prefix = branch.prefix;
  assert.equal(prefix.steps, 3000);
  for (const field of ['commands_sha256', 'observation_trace_sha256', 'transition_trace_sha256',
    'physical_snapshot_sha256', 'state_fingerprint_sha256', 'parameter_fingerprint_sha256']) sha(prefix[field], `${name}.prefix.${field}`);
  assert.equal(prefix.state_fingerprint_sha256, prefix.controller_diagnostics.state_fingerprint_sha256);
  assert.equal(prefix.parameter_fingerprint_sha256, prefix.controller_diagnostics.parameter_fingerprint_sha256);
  assert.equal(prefix.controller_diagnostics.step, 3000);
  assert.equal(prefix.controller_diagnostics.learning_frozen, false);
  assert.equal(prefix.controller_diagnostics.learning_enabled, true);
  assert.equal(prefix.controller_diagnostics.awaiting_observation, false);
  const final = branch.controller_final;
  assert.equal(final.step, 6000);
  assert.equal(final.decision_count, 300);
  assert.equal(final.update_count, 299);
  assert.equal(final.replay_update_count, 1196);
  assert.equal(final.awaiting_observation, false);
  assert.equal(final.learning_frozen, name === 'frozen');
  assert.equal(final.learning_enabled, name !== 'frozen');
  sha(final.state_fingerprint_sha256, `${name} final state fingerprint`);
  sha(final.parameter_fingerprint_sha256, `${name} final parameter fingerprint`);
  if (name === 'frozen') {
    assert.equal(branch.frozen_parameter_checks, 3000);
    assert.equal(branch.frozen_parameters_unchanged, true);
    assert.equal(branch.frozen_parameter_fingerprint_sha256, prefix.parameter_fingerprint_sha256);
    assert.equal(final.parameter_fingerprint_sha256, prefix.parameter_fingerprint_sha256);
    assert.equal(final.suppressed_update_count, final.update_count - prefix.controller_diagnostics.update_count);
    assert.equal(final.suppressed_replay_update_count, final.replay_update_count - prefix.controller_diagnostics.replay_update_count);
  } else {
    assert.equal(branch.frozen_parameter_checks, 0);
    assert.equal(final.suppressed_update_count, 0);
    assert.equal(final.suppressed_replay_update_count, 0);
  }
}

export function summarizeLightAdaptation(reportPath) {
  const absolute = path.resolve(reportPath);
  const reportDirectory = path.dirname(absolute);
  const report = readJson(absolute);
  assert.equal(report.schema, 'light_adaptation_v1');
  assert.equal(report.complete, true, 'Report did not pass all comparisons');
  assert.equal(report.finished, true, 'Report is unfinished');
  assert.equal(report.failed_pairs, 0);
  assert.equal(report.pairs.length, 6);
  assert.equal(report.protocol.steps, 6000);
  assert.equal(report.protocol.control_dt_s, .02);
  assert.equal(report.protocol.intervention_step, 3000);
  assert.equal(report.protocol.intervention_time_s, 60);
  assert.equal(report.protocol.reset_at_intervention, false);
  assert.equal(report.protocol.body_count, 2);
  assert.deepEqual(report.protocol.seeds, [1, 2, 3]);
  assert.deepEqual(report.protocol.early_window_s, [60, 80]);
  assert.deepEqual(report.protocol.late_window_s, [100, 120]);
  assert.equal(report.protocol.difference_direction, 'continued minus frozen');
  assert.deepEqual(report.protocol.initial_sun, {position_m: [.3, 0, .2], intensity_lux: 1000});
  assert.deepEqual(report.protocol.moved_sun, {position_m: [-.3, 0, .2], intensity_lux: 1000});
  const manifest = report.source_sha256;
  assert.ok(manifest && typeof manifest === 'object' && !Array.isArray(manifest));
  for (const required of ['CMakeLists.txt', 'run.sh', 'Dockerfile', 'setup.sh',
    'src/light_adaptation_cli.cpp', 'tools/summarize-light-adaptation.mjs',
    'src/light_learner.cpp', 'include/droid/light_learner.hpp', 'src/light_session.cpp', 'include/droid/light_session.hpp',
    'src/construction.cpp', 'include/droid/construction.hpp', 'src/module_catalog.cpp', 'include/droid/module_catalog.hpp',
    'src/simulation.cpp', 'include/droid/simulation.hpp', 'src/policy.cpp', 'include/droid/policy.hpp',
    'tests/native_light_learner_tests.cpp', 'tests/native_light_session_tests.cpp', 'tests/native_construction_light_tests.cpp',
    'config/module_catalog.json', 'models/droid.xml', 'config/learning_experiment_v3.json',
    'artifacts/light-search-policy-v3-seed0.json', 'docs/LIGHT_ADAPTATION_EXPERIMENT.md', REFERENCE]) {
    assert.ok(Object.hasOwn(manifest, required), `Required provenance is missing: ${required}`);
  }
  for (const [relative, expected] of Object.entries(manifest)) {
    sha(expected, `Source hash ${relative}`);
    assert.equal(hashFile(safeJoin(reportDirectory, `source/${relative}`)), expected, `Archived source differs: ${relative}`);
  }
  assert.equal(report.reference_report.path, REFERENCE);
  assert.equal(report.reference_report.sha256, REFERENCE_SHA);
  assert.equal(report.reference_report.snapshot, `source/${REFERENCE}`);
  assert.equal(manifest[REFERENCE], REFERENCE_SHA);
  const reference = readJson(safeJoin(reportDirectory, report.reference_report.snapshot));
  assert.equal(reference.complete, true);
  assert.equal(reference.failed_trials, 0);
  assert.equal(reference.control_spec.learner.learner_id, 'history_expected_sarsa_v2');
  assert.equal(report.control_spec.learner.learner_id, 'history_expected_sarsa_v2');
  assert.deepEqual(report.physics_spec, reference.physics_spec, 'Physics specification differs from archived v2');
  assert.deepEqual(report.module_catalog, readJson(safeJoin(reportDirectory, 'source/config/module_catalog.json')));
  const referenceParent = path.posix.dirname(REFERENCE);
  for (const [relative, expected] of Object.entries(reference.source_sha256)) {
    assert.equal(manifest[`${referenceParent}/source/${relative}`], expected, `Archived v2 source is missing: ${relative}`);
  }
  for (const relative of ['config/learning_experiment_v3.json', 'artifacts/light-search-policy-v3-seed0.json']) {
    assert.equal(manifest[relative], reference.source_sha256[relative], `Frozen v3 input changed: ${relative}`);
  }

  const rows = [];
  const seen = new Set();
  for (const pair of report.pairs) {
    assert.ok(Number.isInteger(pair.body_index) && pair.body_index >= 0 && pair.body_index < 2);
    assert.ok([1, 2, 3].includes(pair.seed));
    assert.equal(pair.pair_id, `body-${pair.body_index}-seed-${pair.seed}`);
    assert.ok(!seen.has(pair.pair_id), 'Duplicate pair');
    seen.add(pair.pair_id);
    assert.equal(pair.valid, true, `Invalid pair: ${pair.pair_id}`);
    assert.deepEqual(Object.keys(pair.checks).sort(), [...CHECKS].sort());
    for (const [name, passed] of Object.entries(pair.checks)) assert.equal(passed, true, `${pair.pair_id}: ${name} failed`);
    const construction = reference.constructions[pair.body_index];
    assert.deepEqual(pair.assembly, construction.assembly);
    assert.equal(pair.assembly_sha256, construction.assembly_sha256);
    const originals = construction.trials.filter((trial) => trial.mode === 'learner' && trial.seed === pair.seed);
    assert.equal(originals.length, 1, 'Original learner trial is ambiguous');
    const original = originals[0];
    assert.equal(original.completed, true);
    assert.equal(original.exact_controller_and_observation_replay, true);
    const c = pair.continued, f = pair.frozen;
    verifyBranch(c, 'continued', pair.seed);
    verifyBranch(f, 'frozen', pair.seed);
    assert.deepEqual(c.efforts.slice(0, 3000), f.efforts.slice(0, 3000), 'Actual prefix commands differ');
    assert.deepEqual(c.prefix, f.prefix, 'Recorded prefix observations, physical state or controller state differ');
    assert.deepEqual(c.before_move, f.before_move, 'Pre-intervention metrics differ');
    assert.deepEqual(c.curve.slice(0, 600), f.curve.slice(0, 600), 'Pre-intervention curves differ');
    assert.deepEqual(c.efforts, original.efforts, 'Continued commands do not reproduce original v2');
    assert.equal(c.observation_trace_sha256, original.observation_trace_sha256, 'Continued observations do not reproduce original v2');
    assert.equal(c.whole_run.integrated_sensor_reward, original.whole_run.integrated_sensor_reward, 'Continued raw return differs from original v2');
    for (const [current, old] of [['whole_run', 'whole_run'], ['before_move', 'before_move'], ['after_move', 'after_move'],
      ['first_phase_final_20s', 'first_phase_final_20s'], ['late_after_move_20s', 'second_phase_final_20s']]) {
      for (const [field, expected] of Object.entries(original[old])) assert.equal(c[current][field], expected, `Continued ${current}.${field} differs from original v2`);
    }
    assert.deepEqual(Object.keys(pair.differences).sort(), Object.keys(DIFFERENCES).sort());
    for (const [field, [group, metric]] of Object.entries(DIFFERENCES)) {
      near(pair.differences[field], c[group][metric] - f[group][metric], `${pair.pair_id} paired ${field}`);
    }
    const compact = (branch) => ({
      after_move_integrated_sensor_reward: branch.after_move.integrated_sensor_reward,
      early_after_move_mean_received_lux: branch.early_after_move_20s.mean_received_lux,
      late_after_move_mean_received_lux: branch.late_after_move_20s.mean_received_lux,
      after_move_electrical_energy_j: branch.after_move.electrical_energy_j,
      after_move_native_veto_steps: branch.after_move.native_veto_steps,
      after_move_feedback_flag_steps: branch.after_move.feedback_flag_steps,
    });
    rows.push({pair_id: pair.pair_id, body: pair.assembly.name, body_index: pair.body_index, seed: pair.seed,
      continued: compact(c), frozen: compact(f), differences: pair.differences});
  }
  assert.equal(seen.size, 6);
  const medians = (subset) => Object.fromEntries(Object.keys(DIFFERENCES).map((field) =>
    [field, median(subset.map((row) => row.differences[field]))]));
  return {
    schema: 'light_adaptation_summary_v1', complete: true, pair_count: 6,
    report_sha256: hashFile(absolute), reference_report_sha256: REFERENCE_SHA,
    sources_verified: true, source_file_count: Object.keys(manifest).length,
    invariants_verified: true, original_continued_compatibility_verified: true,
    full_transition_replays_verified: true,
    fingerprint_scope: 'Recorded native bit-exact fingerprints compared; private controller state is not reconstructed by this JavaScript verifier',
    primary_metric: 'postmove integrated SUM sensor reward; continued minus frozen',
    pairs: rows, paired_median_differences: medians(rows),
    by_body: [0, 1].map((body) => {
      const subset = rows.filter((row) => row.body_index === body);
      assert.equal(subset.length, 3);
      return {body: subset[0].body, body_index: body, seeds: subset.map((row) => row.seed),
        positive_primary_difference_count: subset.filter((row) => row.differences.after_move_integrated_sensor_reward > 0).length,
        paired_median_differences: medians(subset)};
    }),
    interpretation_limit: 'Six exposed development prefixes; estimates the total effect of continued learning in this moving-sun scenario, not an interaction against stationary sun or generalization to new bodies',
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  if (process.argv.length !== 3) throw new Error('Usage: node tools/summarize-light-adaptation.mjs REPORT_JSON');
  console.log(JSON.stringify(summarizeLightAdaptation(process.argv[2]), null, 2));
}
