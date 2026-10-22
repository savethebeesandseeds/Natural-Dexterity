import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { summarizeLightPrediction } from './summarize-light-prediction.mjs';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const qaRoot = path.join(projectRoot, 'build', 'light-prediction-qa');
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const readJson = filename => JSON.parse(fs.readFileSync(filename, 'utf8').replace(/^\uFEFF/, ''));
const copyJson = value => structuredClone(value);

function contained(root, filename) {
  const relative = path.relative(path.resolve(root), path.resolve(filename));
  return relative !== '' && relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative);
}
function evidencePath(root, relative) {
  assert.equal(typeof relative, 'string');
  assert.ok(relative.length && !relative.includes('\0') && !relative.includes(':') &&
    !path.isAbsolute(relative) && !path.win32.isAbsolute(relative), 'Evidence path must be relative');
  const parts = relative.split(/[\\/]/);
  assert.ok(parts.every(part => part && part !== '.' && part !== '..'), 'Unsafe evidence path component');
  const filename = path.resolve(root, ...parts);
  assert.ok(contained(root, filename), 'Evidence path escaped its root');
  return filename;
}
function regularWithoutSymlinks(root, relative) {
  const filename = evidencePath(root, relative);
  let current = path.resolve(root);
  assert.ok(!fs.lstatSync(current).isSymbolicLink(), 'Symbolic evidence root is forbidden');
  for (const component of relative.split(/[\\/]/)) {
    current = path.join(current, component);
    assert.ok(!fs.lstatSync(current).isSymbolicLink(), 'Symbolic evidence entry is forbidden');
  }
  assert.ok(fs.statSync(filename).isFile(), 'Evidence must be a regular file');
  assert.ok(contained(fs.realpathSync(root), fs.realpathSync(filename)), 'Resolved evidence escaped its root');
  return filename;
}
function assertNewOutput(output) {
  assert.ok(contained(qaRoot, output), 'The test fixture must be below build/light-prediction-qa');
  assert.ok(!fs.existsSync(output), 'Refusing to reuse an existing evidence-test fixture');
  fs.mkdirSync(qaRoot, { recursive: true });
  assert.ok(!fs.lstatSync(qaRoot).isSymbolicLink(), 'QA root must not be a symbolic link');
  const relativeParent = path.relative(qaRoot, path.dirname(output));
  let current = qaRoot;
  for (const component of relativeParent.split(path.sep).filter(Boolean)) {
    current = path.join(current, component);
    if (fs.existsSync(current)) assert.ok(!fs.lstatSync(current).isSymbolicLink(), 'Fixture ancestor must not be a symbolic link');
    else fs.mkdirSync(current);
  }
  fs.mkdirSync(output);
}

// Every linked source/trace stays read-only through this harness's API. It does
// not chmod, truncate, replace, unlink or clean up any linked evidence file.
export async function testLightPredictionEvidence(reportPath, outputPath) {
  const originalReportPath = path.resolve(reportPath);
  const originalRoot = path.dirname(originalReportPath);
  const output = path.resolve(outputPath);
  assert.ok(fs.statSync(originalReportPath).isFile(), 'The completed report must already exist');
  assert.ok(!fs.lstatSync(originalReportPath).isSymbolicLink(), 'The original report must not be a symbolic link');
  assert.ok(!contained(output, originalReportPath), 'Original evidence cannot be inside the new fixture');
  const originalReportBytes = fs.readFileSync(originalReportPath);
  const original = JSON.parse(originalReportBytes.toString('utf8').replace(/^\uFEFF/, ''));
  assert.equal(original.schema, 'light_prediction_v1');
  assert.equal(original.complete, true, 'Run these tests only after the real experiment completes');
  const modelRelative = original.training_models.path;
  assert.equal(modelRelative, 'models.json');
  const originalModelPath = regularWithoutSymlinks(originalRoot, modelRelative);
  const originalModelBytes = fs.readFileSync(originalModelPath);
  assert.equal(sha256(originalModelBytes), original.training_models.sha256, 'Original model bytes do not match the report');
  const originalModels = JSON.parse(originalModelBytes.toString('utf8'));
  assertNewOutput(output);
  const fixture = path.join(output, 'fixture');
  fs.mkdirSync(fixture);
  const mutable = new Set(['report.json', modelRelative]);
  const originals = new Map([[originalReportPath, sha256(originalReportBytes)], [originalModelPath, sha256(originalModelBytes)]]);
  const linked = [];
  const results = {
    schema: 'light_prediction_evidence_tests_v1', complete: false, passed: false,
    original_report: originalReportPath, original_report_sha256: sha256(originalReportBytes),
    original_models_sha256: sha256(originalModelBytes), fixture,
    test_source_sha256: sha256(fs.readFileSync(fileURLToPath(import.meta.url))),
    validator_source_sha256: sha256(fs.readFileSync(new URL('./summarize-light-prediction.mjs', import.meta.url))),
    linked_files: 0, independently_copied_files: 2, cases: [],
    original_evidence_unchanged: false, linked_fixture_evidence_unchanged: false,
  };
  const resultsPath = path.join(output, 'results.json');
  const saveResults = () => fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2) + '\n', { flag: 'w' });
  const writeMutable = (relative, bytes) => {
    assert.ok(mutable.has(relative), 'Only the report and model copies may be changed');
    const filename = evidencePath(fixture, relative);
    if (fs.existsSync(filename)) {
      const info = fs.lstatSync(filename);
      assert.ok(info.isFile() && !info.isSymbolicLink() && info.nlink === 1,
        'Refusing to write a linked or nonregular evidence file');
    }
    fs.writeFileSync(filename, bytes);
  };
  const resetMutable = () => {
    writeMutable('report.json', originalReportBytes);
    writeMutable(modelRelative, originalModelBytes);
  };
  const fixtureReport = path.join(fixture, 'report.json');
  const writeReport = report => writeMutable('report.json', JSON.stringify(report) + '\n');
  const writeModelsAndReport = (models, report) => {
    const bytes = Buffer.from(JSON.stringify(models) + '\n');
    writeMutable(modelRelative, bytes);
    report.training_models.sha256 = sha256(bytes);
    report.training_models.sha256_after_evaluation = sha256(bytes);
    writeReport(report);
  };
  const runCase = async (name, mutation, shouldAccept = false) => {
    resetMutable();
    const report = copyJson(original), models = copyJson(originalModels);
    if (mutation) await mutation(report, models);
    const record = { name, expected: shouldAccept ? 'accept' : 'reject', passed: false };
    results.cases.push(record);results.running_case = name;saveResults();
    const start = performance.now();
    try {
      const summary = await summarizeLightPrediction(fixtureReport);
      record.actual = 'accept';
      if (shouldAccept) {
        assert.equal(summary.complete, true);
        assert.equal(summary.sources_verified, true);
        assert.equal(summary.invariants_verified, true);
        assert.equal(summary.report_sha256, results.original_report_sha256);
      }
      record.passed = shouldAccept;
      if (!shouldAccept) record.error = 'Validator accepted tampered evidence';
    } catch (error) {
      record.actual = 'reject';record.error = String(error?.message ?? error);
      record.passed = !shouldAccept;
    }
    record.elapsed_ms = Math.round(performance.now() - start);
    delete results.running_case;saveResults();
    process.stdout.write(`${record.passed ? 'PASS' : 'FAIL'} ${name}: ${record.actual}\n`);
  };
  let completedCases = false;
  try {
    const references = new Set([
      ...Object.keys(original.source_sha256).map(relative => 'source/' + relative),
      ...original.trials.map(trial => trial.trace.path),
    ]);
    for (const relative of references) {
      assert.ok(!mutable.has(relative), 'A linked artifact overlaps a mutable copy');
      const source = regularWithoutSymlinks(originalRoot, relative);
      const destination = evidencePath(fixture, relative);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      const digest = sha256(fs.readFileSync(source));originals.set(source, digest);
      try { fs.linkSync(source, destination);results.linked_files++; }
      catch (error) {
        if (!['EXDEV', 'EPERM', 'ENOTSUP'].includes(error.code)) throw error;
        fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);results.independently_copied_files++;
      }
      linked.push({ filename: destination, sha256: digest });
    }
    resetMutable();saveResults();
    await runCase('untouched exact-byte fixture', null, true);
    assert.equal(results.cases[0].passed, true, 'Untouched fixture failed; negative-case results would be uninterpretable');
    await runCase('incomplete report', report => { report.complete = false;writeReport(report); });
    await runCase('false full replay', report => { report.trials[0].replay.exact = false;writeReport(report); });
    await runCase('source hash mismatch', report => {
      report.source_sha256[Object.keys(report.source_sha256)[0]] = '0'.repeat(64);writeReport(report);
    });
    await runCase('withheld sequence in training membership with renewed model hash', (report, models) => {
      const body = models.bodies[0];
      const heldOut = original.trials.find(trial => trial.body_index === body.body_index && trial.split === 'test');
      assert.ok(heldOut);
      body.training_membership[0].trial_id = heldOut.trial_id;writeModelsAndReport(models, report);
    });
    await runCase('altered coherent coefficient and A with renewed model hash', (report, models) => {
      const model = models.bodies[0].models.find(value => value.mode === 'history_input');
      model.coefficients[0][1] += .125;model.state_space.A[0][0] += .125;
      writeModelsAndReport(models, report);
    });
    await runCase('altered input B with renewed model hash', (report, models) => {
      const model = models.bodies[0].models.find(value => value.mode === 'history_input');
      model.state_space.B[0][0] += .25;writeModelsAndReport(models, report);
    });
    await runCase('altered delay shift with renewed model hash', (report, models) => {
      const model = models.bodies[0].models.find(value => value.mode === 'history_input');
      assert.equal(model.state_space.A[12][0], 1);
      model.state_space.A[12][0] = .5;writeModelsAndReport(models, report);
    });
    await runCase('future actual output injected into recursive predictions', report => {
      const forecast = report.evaluation.bodies[0].forecasts.find(item =>
        item.predictions.history_input.some((row, index) => row.some((value, channel) => value !== item.actual[index][channel])));
      assert.ok(forecast, 'No differing prediction available to corrupt');
      forecast.predictions.history_input = copyJson(forecast.actual);writeReport(report);
    });
    await runCase('request tape disagrees with declared probe and governed trace', report => {
      const trial = report.trials[0];trial.request_tape[0] = trial.request_tape[0] === 1 ? -1 : 1;
      trial.request_tape_sha256 = sha256('[' + trial.request_tape.map(value => value.toFixed(1)).join(',') + ']');
      writeReport(report);
    });
    await runCase('sampled emitted effort disagrees with native trace', report => {
      report.trials[0].frames[10].output[0] += .125;writeReport(report);
    });
    await runCase('aggregate RMSE corruption', report => {
      const value = report.evaluation.bodies[0].metrics.find(item => item.model_id === 'history_input').horizons.find(item => item.steps === 10);
      assert.equal(typeof value.rmse, 'number');value.rmse += .125;writeReport(report);
    });
    await runCase('readiness gate corruption', report => {
      const value = report.evaluation.bodies[0].metrics.find(item => item.model_id === 'history_input').readiness;
      assert.equal(typeof value.passed, 'boolean');value.passed = !value.passed;writeReport(report);
    });
    completedCases = true;
  } catch (error) {
    results.fatal_error = String(error?.stack ?? error);
  } finally {
    // Restoration writes only independent mutable copies. Keep the fixture and
    // results for inspection; never delete or alter the real archive.
    try {
      resetMutable();
      for (const [filename, expected] of originals)
        assert.equal(sha256(fs.readFileSync(filename)), expected, 'Original evidence changed: ' + filename);
      results.original_evidence_unchanged = true;
      for (const item of linked)
        assert.equal(sha256(fs.readFileSync(item.filename)), item.sha256, 'Linked fixture evidence changed: ' + item.filename);
      results.linked_fixture_evidence_unchanged = true;
      assert.equal(sha256(fs.readFileSync(fixtureReport)), results.original_report_sha256);
      assert.equal(sha256(fs.readFileSync(path.join(fixture, modelRelative))), results.original_models_sha256);
    } catch (error) { results.preservation_error = String(error?.stack ?? error); }
    results.complete = completedCases && results.original_evidence_unchanged && results.linked_fixture_evidence_unchanged;
    results.passed = results.complete && results.cases.every(record => record.passed);
    saveResults();
  }
  return results;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const args = process.argv.slice(2);
    assert.equal(args.length, 4, 'Usage: node tools/test-light-prediction-evidence.mjs --report REPORT --output NEW_QA_DIRECTORY');
    assert.equal(args[0], '--report');assert.equal(args[2], '--output');
    const results = await testLightPredictionEvidence(args[1], args[3]);
    process.stdout.write(`Evidence tests ${results.passed ? 'passed' : 'failed'}; retained at ${path.join(path.resolve(args[3]), 'results.json')}\n`);
    if (!results.passed) process.exitCode = 1;
  } catch (error) { process.stderr.write(String(error?.stack ?? error) + '\n');process.exitCode = 1; }
}
