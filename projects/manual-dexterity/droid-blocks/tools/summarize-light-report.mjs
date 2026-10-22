import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const reportPath = process.argv[2];
if (!reportPath) throw new Error('Usage: node tools/summarize-light-report.mjs REPORT_JSON');
const report = JSON.parse(fs.readFileSync(reportPath, 'utf8').replace(/^\uFEFF/, ''));
const median = (values) => [...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];
const summary = { complete: report.complete, failed_trials: report.failed_trials, sources_verified: true, rows: [] };
for (const [relative, expected] of Object.entries(report.source_sha256)) {
  const archived = path.join(path.dirname(reportPath), 'source', relative);
  const actual = crypto.createHash('sha256').update(fs.readFileSync(archived)).digest('hex');
  if (actual !== expected) throw new Error(`Archived source hash differs: ${relative}`);
}
for (const construction of report.constructions) {
  for (const trial of construction.trials) {
    if (Math.abs(trial.whole_run.integrated_sensor_reward - trial.before_move.integrated_sensor_reward - trial.after_move.integrated_sensor_reward) > 1e-8) throw new Error('Reward phases do not add up');
    if (Math.abs(trial.whole_run.electrical_energy_j - trial.before_move.electrical_energy_j - trial.after_move.electrical_energy_j) > 1e-8) throw new Error('Energy phases do not add up');
    if (Math.abs(trial.whole_run.integrated_sensor_reward - trial.whole_run.integrated_light_reward - trial.whole_run.integrated_imu_reward) > 1e-8) throw new Error('Sensor votes do not add up');
  }
  for (const mode of ['learner','zero','rhythm','random']) {
    const trials = construction.trials.filter((trial)=>trial.mode === mode);
    if (!trials.length) continue;
    summary.rows.push({ body: construction.assembly.name, mode, seeds: trials.map((trial)=>trial.seed),
      completed: trials.every((trial)=>trial.completed), exact_replay: trials.every((trial)=>trial.exact_controller_and_observation_replay),
      median_integrated_reward: median(trials.map((trial)=>trial.whole_run.integrated_sensor_reward)),
      reward_range: [Math.min(...trials.map((trial)=>trial.whole_run.integrated_sensor_reward)), Math.max(...trials.map((trial)=>trial.whole_run.integrated_sensor_reward))],
      median_final20_lux_before: median(trials.map((trial)=>trial.first_phase_final_20s.mean_received_lux)),
      median_final20_lux_after: median(trials.map((trial)=>trial.second_phase_final_20s.mean_received_lux)),
      median_energy_j: median(trials.map((trial)=>trial.whole_run.electrical_energy_j)),
      flagged_steps: trials.reduce((sum,trial)=>sum+trial.whole_run.feedback_flag_steps,0),
      veto_steps: trials.reduce((sum,trial)=>sum+trial.whole_run.native_veto_steps,0) });
  }
}
console.log(JSON.stringify(summary,null,2));
