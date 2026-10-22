import { readFile, writeFile, realpath, rename, rm, mkdir, lstat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { summarizeLightPrediction } from './summarize-light-prediction.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const within = (root, target) => {
  const relative = path.relative(root, target);
  return relative === '' || (relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative));
};

async function checkedOutputPath(input) {
  const output = path.resolve(input);
  const allowed = [path.join(projectRoot, 'web'), path.join(projectRoot, 'build', 'light-prediction-qa')];
  assert(allowed.some((root) => within(root, output)) && path.extname(output).toLowerCase() === '.json', 'Output must be JSON under web or build/light-prediction-qa');
  let ancestor = path.dirname(output), actualAncestor;
  for (;;) {
    try { actualAncestor = await realpath(ancestor); break; }
    catch (error) {
      if (error.code !== 'ENOENT') throw error;
      const parent = path.dirname(ancestor);
      assert(parent !== ancestor, 'Unable to resolve output directory');
      ancestor = parent;
    }
  }
  const actualRoot = await realpath(projectRoot);
  const resolved = path.resolve(actualAncestor, path.relative(ancestor, output));
  assert([path.join(actualRoot, 'web'), path.join(actualRoot, 'build', 'light-prediction-qa')].some((root) => within(root, resolved)), 'Output resolves outside allowed viewer directories');
  try { assert(!(await lstat(output)).isSymbolicLink(), 'Output must not be a symbolic link'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  return output;
}

export function predictionView(summary) {
  assert(summary.complete && summary.sources_verified && summary.invariants_verified, 'Independent evidence validation has not passed');
  const { report, models } = summary;
  assert(report.complete && report.evaluation.complete && report.trials.length === 72, 'Native evidence is incomplete');
  assert(models.bodies.length === 3 && report.bodies.length === 3, 'Three declared bodies are required');
  return {
    schema: 'light_prediction_view_v1',
    source: {
      report_path: '', report_sha256: summary.report_sha256,
      protocol_sha256: report.protocol.sha256,
      source_count: Object.keys(report.source_sha256).length,
      models_sha256: report.training_models.sha256,
    },
    checks: {
      complete: true, independently_checked: true,
      trials: report.trials.length,
      replays: report.trials.filter((trial) => trial.exact_full_transition_replay === true).length,
      models_sealed: report.training_models.saved_before_evaluation === true && report.training_models.unchanged_after_evaluation === true,
      training_rows_reconstructed: true, regression_refitted: true,
      recursive_forecasts_recomputed: true, request_tapes_reconstructed: true,
      paired_mechanics_verified: true, all_metrics_recomputed: true,
      failed_forecasts: report.evaluation.failed_forecasts,
      readiness_passed: report.evaluation.readiness_passed,
      simpler_candidate_readiness_passed: report.evaluation.simpler_candidate_readiness_passed,
    },
    bodies: report.bodies.map((body) => {
      const fitted = models.bodies.find((entry) => entry.body_index === body.body_index);
      const evaluated = report.evaluation.bodies.find((entry) => entry.body_index === body.body_index);
      assert(fitted && evaluated, 'Body fit or evaluation is missing');
      return {
        id: body.body_index, name: body.assembly.name,
        assembly_sha256: body.assembly_sha256,
        models: fitted.models.map((model) => ({
          id: model.mode, taps: model.taps, state_dimension: model.state_dimension,
          ...model.state_space, fit: model.fit,
        })),
        metrics: evaluated.metrics.map((metric) => {
          assert(typeof metric.readiness?.passed === 'boolean', 'Native readiness is not a checked decision');
          const horizon = metric.horizons.find((entry) => entry.steps === 10);
          return { ...metric, readiness: metric.readiness.passed,
            readiness_checks: metric.readiness.checks ?? {}, changed_request_subset: {
            origin_count: horizon.changed_request_origins,
            rmse: horizon.changed_request_rmse,
            trials: horizon.trials.map((trial) => ({
              trial_id: trial.trial_id, origin_count: trial.changed_request_origins,
              rmse: trial.changed_request_rmse,
            })),
          } };
        }),
        forecasts: evaluated.forecasts,
      };
    }),
  };
}

async function main() {
  const args = process.argv.slice(2);
  assert(args.length >= 1 && args.length <= 2, 'Usage: node tools/export-light-prediction-view.mjs REPORT_JSON [OUTPUT_JSON]');
  const reportPath = await realpath(path.resolve(args[0]));
  assert(within(projectRoot, reportPath), 'Report must be inside the Droid Blocks project for portable provenance');
  const outputPath = await checkedOutputPath(args[1] ?? path.join(projectRoot, 'web', 'prediction-data.json'));
  assert(outputPath !== reportPath, 'Export must never replace its source report');
  const summary = await summarizeLightPrediction(reportPath);
  const digest = sha256(await readFile(reportPath));
  assert(summary.report_sha256 === digest, 'Report changed during independent validation');
  const view = predictionView(summary);
  view.source.report_path = path.relative(projectRoot, reportPath).split(path.sep).join('/');
  assert(view.checks.models_sealed && view.checks.replays === 72, 'Sealed models and 72 full replays are required');
  await mkdir(path.dirname(outputPath), { recursive: true });
  const temporary = outputPath + '.next-' + process.pid;
  let created = false;
  try {
    await writeFile(temporary, JSON.stringify(view) + '\n', { flag: 'wx' }); created = true;
    assert(sha256(await readFile(reportPath)) === digest, 'Report changed during export');
    await rename(temporary, outputPath); created = false;
  } finally { if (created) await rm(temporary, { force: true }); }
  process.stdout.write(`Exported three independently verified bodies to ${path.relative(projectRoot, outputPath)}\nReport SHA-256: ${digest}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch((error) => { process.stderr.write(error.message + '\n'); process.exitCode = 1; });
