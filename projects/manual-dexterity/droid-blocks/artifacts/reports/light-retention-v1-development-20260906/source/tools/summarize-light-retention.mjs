import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REFERENCE = 'artifacts/reports/light-learning-v2-development-20260905/report.json';
const REFERENCE_SHA = '931a26b17b399456bb5461293e382ae9880486f1b38157007500eead65998230';
const SHA = /^[a-f0-9]{64}$/;
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
const PROTOCOL = 'docs/LIGHT_RETENTION_EXPERIMENT.md';
const PROTOCOL_SHA = 'd3a8daf76154004c1610d0d15e6b60983782ebfa081ef1739c3341977fd44af9';
const BRANCHES = ['continued', 'current_frozen', 'recalled_frozen'];
const DIFFERENCES = {
  primary_early_recalled_minus_current_reward: ['recalled_frozen', 'current_frozen', 'early_after_return_20s', 'integrated_sensor_reward'],
  full_return_recalled_minus_current_reward: ['recalled_frozen', 'current_frozen', 'after_return', 'integrated_sensor_reward'],
  early_lux_recalled_minus_current: ['recalled_frozen', 'current_frozen', 'early_after_return_20s', 'mean_received_lux'],
  late_lux_recalled_minus_current: ['recalled_frozen', 'current_frozen', 'late_after_return_20s', 'mean_received_lux'],
  energy_recalled_minus_current: ['recalled_frozen', 'current_frozen', 'after_return', 'electrical_energy_j'],
  full_return_continued_minus_current_reward: ['continued', 'current_frozen', 'after_return', 'integrated_sensor_reward'],
  full_return_continued_minus_recalled_reward: ['continued', 'recalled_frozen', 'after_return', 'integrated_sensor_reward'],
};
const CHECKS = [
  'identical_prefix_commands', 'identical_prefix_observations', 'identical_prefix_full_transitions',
  'identical_prefix_public_physical_snapshot', 'identical_prefix_controller_state_fingerprint',
  'identical_prefix_controller_diagnostics', 'identical_first_A_checkpoint',
  'all_prefixes_match_archived_v2', 'restoration_changes_only_values',
  'current_frozen_parameters_unchanged', 'recalled_frozen_parameters_unchanged',
  'continued_full_replay', 'current_frozen_full_replay', 'recalled_frozen_full_replay',
];
function verifyBranch(branch, name, seed, original) {
  const frozen = name !== 'continued', recalled = name === 'recalled_frozen';
  assert.equal(branch.branch, name);
  assert.equal(branch.mode, 'learner');
  assert.equal(branch.seed, seed);
  assert.equal(branch.completed, true, name + ' is incomplete');
  assert.equal(branch.completed_steps, 9000);
  assert.equal(branch.recorded_transition_count, 9000);
  assert.equal(branch.recorded_effort_count, 9000);
  assert.equal(branch.error, '');
  assert.deepEqual(branch.snapshot_errors, []);
  assert.equal(branch.final_controller_snapshot_available, true);
  assert.equal(branch.final_public_physical_snapshot_available, true);
  assert.equal(branch.last_attempted_action.step, 9000);
  assert.equal(branch.last_attempted_action.fully_accepted, true);
  assert.equal(branch.exact_controller_and_observation_replay, true);
  assert.equal(branch.exact_full_transition_replay, true);
  assert.equal(branch.replay.completed_steps, 9000);
  assert.equal(branch.replay.error, '');
  for (const field of ['effort_trace_sha256', 'observation_trace_sha256', 'transition_trace_sha256',
    'final_public_physical_snapshot_sha256']) sha(branch[field], name + '.' + field);
  assert.equal(branch.efforts.length, 9000);
  assert.equal(branch.last_attempted_action.effort, branch.efforts.at(-1));
  let previous = 0;
  for (let i = 0; i < branch.efforts.length; ++i) {
    const effort = finite(branch.efforts[i], name + ' effort ' + i);
    assert.ok(Math.abs(effort) <= .15 + 1e-12 && Math.abs(effort - previous) <= .1 + 1e-12,
      name + ' command/slew bound exceeded at ' + i);
    if (i < 5) assert.equal(effort, 0, 'Quiet startup moved');
    previous = effort;
  }
  verifyMetrics(branch.whole_run, 9000, name + '.whole_run');
  verifyMetrics(branch.prefix120_metrics, 6000, name + '.prefix120_metrics');
  for (const group of ['before_first_move', 'middle_condition', 'after_return'])
    verifyMetrics(branch[group], 3000, name + '.' + group);
  for (const group of ['first_phase_final_20s', 'second_phase_final_20s', 'early_after_return_20s',
    'middle_after_return_20s', 'late_after_return_20s']) verifyMetrics(branch[group], 1000, name + '.' + group);
  verifyPartition(branch.whole_run, [branch.prefix120_metrics, branch.after_return], name + ' full horizon');
  verifyPartition(branch.prefix120_metrics, [branch.before_first_move, branch.middle_condition], name + ' first120');
  verifyPartition(branch.after_return, [branch.early_after_return_20s, branch.middle_after_return_20s,
    branch.late_after_return_20s], name + ' return windows');
  assert.equal(branch.curve.length, 1800);
  for (let i = 0; i < branch.curve.length; ++i) {
    const point = branch.curve[i];
    near(point.time_s, (i + 1) * .1, name + ' curve timestamp');
    for (const field of ['illuminance_lux', 'sensor_reward_rate', 'effort', 'cumulative_sensor_reward'])
      finite(point[field], name + ' curve ' + field);
    assert.ok(point.illuminance_lux >= 0);
    assert.equal(point.effort, branch.efforts[(i + 1) * 5 - 1]);
  }
  near(branch.curve[599].cumulative_sensor_reward, branch.before_first_move.integrated_sensor_reward, name + ' first move reward');
  near(branch.curve[1199].cumulative_sensor_reward, branch.prefix120_metrics.integrated_sensor_reward, name + ' return reward');
  near(branch.curve.at(-1).cumulative_sensor_reward, branch.whole_run.integrated_sensor_reward, name + ' total reward');
  const checkpoint = branch.checkpoint60;
  assert.equal(checkpoint.steps, 3000);
  assert.equal(checkpoint.capture_readonly_verified, true);
  assert.equal(checkpoint.controller_diagnostics.step, 3000);
  assert.equal(checkpoint.controller_diagnostics.learning_frozen, false);
  assert.equal(checkpoint.controller_diagnostics.awaiting_observation, false);
  sha(checkpoint.captured_parameter_fingerprint_sha256, name + ' first-A value fingerprint');
  assert.equal(checkpoint.captured_parameter_fingerprint_sha256, checkpoint.controller_diagnostics.parameter_fingerprint_sha256);
  assert.equal(checkpoint.state_fingerprint_before_capture_sha256, checkpoint.controller_diagnostics.state_fingerprint_sha256);
  assert.equal(checkpoint.state_fingerprint_after_capture_sha256, checkpoint.state_fingerprint_before_capture_sha256);
  const prefix = branch.prefix120;
  assert.equal(prefix.steps, 6000);
  for (const field of ['commands_sha256', 'observation_trace_sha256', 'transition_trace_sha256',
    'physical_snapshot_sha256', 'state_fingerprint_sha256', 'parameter_fingerprint_sha256',
    'nonparameter_state_fingerprint_sha256']) sha(prefix[field], name + '.prefix120.' + field);
  for (const field of ['state_fingerprint_sha256', 'parameter_fingerprint_sha256', 'nonparameter_state_fingerprint_sha256'])
    assert.equal(prefix[field], prefix.controller_diagnostics[field]);
  assert.equal(prefix.controller_diagnostics.step, 6000);
  assert.equal(prefix.controller_diagnostics.learning_frozen, false);
  assert.equal(prefix.controller_diagnostics.learning_enabled, true);
  assert.equal(prefix.controller_diagnostics.awaiting_observation, false);
  const final = branch.controller_final;
  assert.equal(final.step, 9000);
  assert.equal(final.decision_count, 450);
  assert.equal(final.update_count, 449);
  assert.equal(final.replay_update_count, 1796);
  assert.equal(final.awaiting_observation, false);
  assert.equal(final.learning_frozen, frozen);
  assert.equal(final.learning_enabled, !frozen);
  for (const field of ['state_fingerprint_sha256', 'parameter_fingerprint_sha256', 'nonparameter_state_fingerprint_sha256'])
    sha(final[field], name + ' final ' + field);
  if (frozen) {
    const expected = recalled ? checkpoint.captured_parameter_fingerprint_sha256 : prefix.parameter_fingerprint_sha256;
    assert.equal(branch.frozen_parameter_checks, 3000);
    assert.equal(branch.frozen_parameters_unchanged, true);
    assert.equal(branch.frozen_parameter_fingerprint_sha256, expected);
    assert.equal(final.parameter_fingerprint_sha256, expected);
    assert.equal(final.suppressed_update_count, final.update_count - prefix.controller_diagnostics.update_count);
    assert.equal(final.suppressed_replay_update_count, final.replay_update_count - prefix.controller_diagnostics.replay_update_count);
  } else {
    assert.equal(branch.frozen_parameter_checks, 0);
    assert.equal(final.suppressed_update_count, 0);
    assert.equal(final.suppressed_replay_update_count, 0);
  }
  if (recalled) {
    const restore = branch.restoration;
    assert.equal(restore.before_parameter_fingerprint_sha256, prefix.parameter_fingerprint_sha256);
    assert.equal(restore.after_parameter_fingerprint_sha256, checkpoint.captured_parameter_fingerprint_sha256);
    assert.equal(restore.before_nonparameter_state_fingerprint_sha256, prefix.nonparameter_state_fingerprint_sha256);
    assert.equal(restore.after_nonparameter_state_fingerprint_sha256, restore.before_nonparameter_state_fingerprint_sha256);
    assert.equal(restore.public_physical_state_unchanged, true);
    const after = restore.after_restore_controller_diagnostics;
    assert.equal(after.nonparameter_state_fingerprint_sha256, prefix.nonparameter_state_fingerprint_sha256);
    assert.equal(after.parameter_fingerprint_sha256, checkpoint.captured_parameter_fingerprint_sha256);
    assert.equal(after.learning_frozen, false, 'Restoration also changed the freeze flag');
    assert.equal(after.step, 6000);
    assert.equal(after.awaiting_observation, false);
  } else assert.equal(branch.restoration, null);
  assert.deepEqual(Object.keys(branch.reference_compatibility).sort(), ['commands', 'metrics', 'observations', 'return']);
  for (const passed of Object.values(branch.reference_compatibility)) assert.equal(passed, true);
  assert.deepEqual(branch.efforts.slice(0, 6000), original.efforts, name + ' prefix commands differ from original v2');
  assert.equal(prefix.observation_trace_sha256, original.observation_trace_sha256);
  assert.equal(branch.prefix120_metrics.integrated_sensor_reward, original.whole_run.integrated_sensor_reward);
  for (const [group, old] of [['prefix120_metrics', 'whole_run'], ['before_first_move', 'before_move'],
    ['middle_condition', 'after_move'], ['first_phase_final_20s', 'first_phase_final_20s'],
    ['second_phase_final_20s', 'second_phase_final_20s']]) {
    for (const [field, expected] of Object.entries(original[old]))
      assert.equal(branch[group][field], expected, name + ' original metric differs: ' + group + '.' + field);
  }
}
export function summarizeLightRetention(reportPath) {
  const absolute = path.resolve(reportPath), directory = path.dirname(absolute);
  const report = readJson(absolute);
  assert.equal(report.schema, 'light_retention_v1');
  assert.equal(report.complete, true, 'Retention evidence is incomplete or invalid');
  assert.equal(report.finished, true);
  assert.equal(report.failed_cases, 0);
  assert.equal(report.cases.length, 6);
  assert.equal(report.protocol.steps, 9000);
  assert.equal(report.protocol.control_dt_s, .02);
  assert.equal(report.protocol.first_move_step, 3000);
  assert.equal(report.protocol.return_step, 6000);
  assert.equal(report.protocol.first_move_time_s, 60);
  assert.equal(report.protocol.return_time_s, 120);
  assert.equal(report.protocol.reset_at_interventions, false);
  assert.equal(report.protocol.body_count, 2);
  assert.deepEqual(report.protocol.seeds, [1, 2, 3]);
  assert.deepEqual(report.protocol.branches, BRANCHES);
  assert.deepEqual(report.protocol.early_window_s, [120, 140]);
  assert.deepEqual(report.protocol.late_window_s, [160, 180]);
  assert.deepEqual(report.protocol.initial_sun, {position_m: [.3, 0, .2], intensity_lux: 1000});
  assert.deepEqual(report.protocol.middle_sun, {position_m: [-.3, 0, .2], intensity_lux: 1000});
  assert.deepEqual(report.protocol.returned_sun, report.protocol.initial_sun);
  assert.equal(report.protocol.document, PROTOCOL);
  assert.equal(report.protocol.document_sha256, PROTOCOL_SHA);
  const manifest = report.source_sha256;
  assert.ok(manifest && typeof manifest === 'object' && !Array.isArray(manifest));
  for (const required of ['CMakeLists.txt', 'run.sh', 'Dockerfile', 'setup.sh',
    'src/light_retention_cli.cpp', 'tools/summarize-light-retention.mjs',
    'src/light_learner.cpp', 'include/droid/light_learner.hpp', 'src/light_session.cpp', 'include/droid/light_session.hpp',
    'src/construction.cpp', 'include/droid/construction.hpp', 'src/module_catalog.cpp', 'include/droid/module_catalog.hpp',
    'src/simulation.cpp', 'include/droid/simulation.hpp', 'src/policy.cpp', 'include/droid/policy.hpp',
    'tests/native_light_learner_tests.cpp', 'tests/native_light_session_tests.cpp', 'tests/native_construction_light_tests.cpp',
    'config/module_catalog.json', 'models/droid.xml', 'config/learning_experiment_v3.json',
    'artifacts/light-search-policy-v3-seed0.json', PROTOCOL, REFERENCE]) {
    assert.ok(Object.hasOwn(manifest, required), 'Required provenance missing: ' + required);
  }
  for (const [relative, expected] of Object.entries(manifest)) {
    sha(expected, 'Source hash ' + relative);
    assert.equal(hashFile(safeJoin(directory, 'source/' + relative)), expected, 'Archived source differs: ' + relative);
  }
  assert.equal(manifest[PROTOCOL], PROTOCOL_SHA);
  assert.equal(report.reference_report.path, REFERENCE);
  assert.equal(report.reference_report.sha256, REFERENCE_SHA);
  assert.equal(report.reference_report.snapshot, 'source/' + REFERENCE);
  assert.equal(manifest[REFERENCE], REFERENCE_SHA);
  const reference = readJson(safeJoin(directory, report.reference_report.snapshot));
  assert.equal(reference.complete, true);
  assert.equal(reference.failed_trials, 0);
  assert.equal(reference.constructions.length, 2);
  assert.equal(reference.control_spec.learner.learner_id, 'history_expected_sarsa_v2');
  assert.equal(report.control_spec.learner.learner_id, 'history_expected_sarsa_v2');
  assert.deepEqual(report.physics_spec, reference.physics_spec, 'Physics specification changed');
  assert.deepEqual(report.module_catalog, readJson(safeJoin(directory, 'source/config/module_catalog.json')));
  const referenceParent = path.posix.dirname(REFERENCE);
  for (const [relative, expected] of Object.entries(reference.source_sha256))
    assert.equal(manifest[referenceParent + '/source/' + relative], expected, 'Archived v2 source missing: ' + relative);
  for (const relative of ['config/learning_experiment_v3.json', 'artifacts/light-search-policy-v3-seed0.json'])
    assert.equal(manifest[relative], reference.source_sha256[relative], 'Frozen v3 input changed: ' + relative);
  const rows = [], seen = new Set();
  for (const item of report.cases) {
    assert.ok(Number.isInteger(item.body_index) && item.body_index >= 0 && item.body_index < 2);
    assert.ok([1, 2, 3].includes(item.seed));
    assert.equal(item.case_id, 'body-' + item.body_index + '-seed-' + item.seed);
    assert.ok(!seen.has(item.case_id), 'Duplicate retention case');
    seen.add(item.case_id);
    assert.equal(item.valid, true, 'Invalid retention case');
    assert.deepEqual(Object.keys(item.checks).sort(), [...CHECKS].sort());
    for (const [name, passed] of Object.entries(item.checks)) assert.equal(passed, true, item.case_id + ': ' + name);
    const construction = reference.constructions[item.body_index];
    assert.deepEqual(item.assembly, construction.assembly);
    assert.equal(item.assembly_sha256, construction.assembly_sha256);
    const originals = construction.trials.filter((trial) => trial.mode === 'learner' && trial.seed === item.seed);
    assert.equal(originals.length, 1);
    const original = originals[0];
    assert.equal(original.completed, true);
    assert.equal(original.exact_controller_and_observation_replay, true);
    for (const name of BRANCHES) verifyBranch(item[name], name, item.seed, original);
    const c = item.continued;
    for (const name of ['current_frozen', 'recalled_frozen']) {
      const branch = item[name];
      assert.deepEqual(branch.checkpoint60, c.checkpoint60, 'First-A checkpoint differs');
      assert.deepEqual(branch.prefix120, c.prefix120, 'Shared return state/history differs');
      assert.deepEqual(branch.prefix120_metrics, c.prefix120_metrics);
      assert.deepEqual(branch.efforts.slice(0, 6000), c.efforts.slice(0, 6000));
      assert.deepEqual(branch.curve.slice(0, 1200), c.curve.slice(0, 1200));
      assert.deepEqual(branch.efforts.slice(6000, 6005), c.efforts.slice(6000, 6005), 'Unfinished option changed at restoration');
      assert.deepEqual(branch.curve[1200], c.curve[1200], 'First post-return option outcome differs');
    }
    assert.deepEqual(Object.keys(item.differences).sort(), Object.keys(DIFFERENCES).sort());
    for (const [field, [a, b, group, metric]] of Object.entries(DIFFERENCES))
      near(item.differences[field], item[a][group][metric] - item[b][group][metric], item.case_id + ' difference ' + field);
    const compact = (branch) => ({
      early_return_integrated_sensor_reward: branch.early_after_return_20s.integrated_sensor_reward,
      full_return_integrated_sensor_reward: branch.after_return.integrated_sensor_reward,
      early_return_mean_received_lux: branch.early_after_return_20s.mean_received_lux,
      late_return_mean_received_lux: branch.late_after_return_20s.mean_received_lux,
      return_electrical_energy_j: branch.after_return.electrical_energy_j,
      return_native_veto_steps: branch.after_return.native_veto_steps,
      return_feedback_flag_steps: branch.after_return.feedback_flag_steps,
      max_current_a: branch.whole_run.max_current_a,
      max_temperature_c: branch.whole_run.max_temperature_c,
      max_motor_speed_rad_s: branch.whole_run.max_motor_speed_rad_s,
    });
    rows.push({case_id: item.case_id, body: item.assembly.name, body_index: item.body_index, seed: item.seed,
      continued: compact(c), current_frozen: compact(item.current_frozen), recalled_frozen: compact(item.recalled_frozen),
      differences: item.differences});
  }
  assert.equal(seen.size, 6);
  const medians = (subset) => Object.fromEntries(Object.keys(DIFFERENCES).map((field) =>
    [field, median(subset.map((row) => row.differences[field]))]));
  return {
    schema: 'light_retention_summary_v1', complete: true, case_count: 6, branch_count: 18,
    report_sha256: hashFile(absolute), reference_report_sha256: REFERENCE_SHA,
    sources_verified: true, source_file_count: Object.keys(manifest).length,
    invariants_verified: true, original_prefix_compatibility_verified: true, full_transition_replays_verified: true,
    fingerprint_scope: 'Recorded native fingerprints compare exact state and parameter bits; this verifier does not reconstruct private controller state',
    primary_metric: 'Early120-140s recalled_frozen minus current_frozen integrated SUM sensor reward',
    cases: rows, paired_median_differences: medians(rows),
    by_body: [0, 1].map((body) => {
      const subset = rows.filter((row) => row.body_index === body);
      assert.equal(subset.length, 3);
      return {body: subset[0].body, body_index: body, seeds: subset.map((row) => row.seed),
        positive_primary_difference_count: subset.filter((row) => row.differences.primary_early_recalled_minus_current_reward > 0).length,
        paired_median_differences: medians(subset)};
    }),
    interpretation_limit: 'Externally selected W60 versus W120 under identical B-derived current state/history; evaluates retained value competence, not autonomous recall, general forgetting or new-body transfer',
  };
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  if (process.argv.length !== 3) throw new Error('Usage: node tools/summarize-light-retention.mjs REPORT_JSON');
  console.log(JSON.stringify(summarizeLightRetention(process.argv[2]), null, 2));
}