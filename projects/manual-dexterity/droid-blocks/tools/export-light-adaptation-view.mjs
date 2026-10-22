import { readFile, writeFile, realpath, rename, rm, mkdir, lstat } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { summarizeLightAdaptation } from "./summarize-light-adaptation.mjs";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const finite = (value, label) => { assert(typeof value === "number" && Number.isFinite(value), `${label} must be a finite number`); return value; };
const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const within = (root, target) => {
  const relative = path.relative(root, target);
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
};

async function checkedOutputPath(input) {
  const output = path.resolve(input);
  const allowed = [path.join(projectRoot, "web"), path.join(projectRoot, "build", "light-adaptation-qa")];
  assert(allowed.some((root) => within(root, output)) && path.extname(output).toLowerCase() === ".json", "Output must be a JSON file under web or build/light-adaptation-qa");
  let ancestor = path.dirname(output);
  let actualAncestor;
  for (;;) {
    try { actualAncestor = await realpath(ancestor); break; }
    catch (error) {
      if (error.code !== "ENOENT") throw error;
      const parent = path.dirname(ancestor);
      assert(parent !== ancestor, "Unable to resolve the output directory");
      ancestor = parent;
    }
  }
  const actualRoot = await realpath(projectRoot);
  const resolvedOutput = path.resolve(actualAncestor, path.relative(ancestor, output));
  assert([path.join(actualRoot, "web"), path.join(actualRoot, "build", "light-adaptation-qa")].some((root) => within(root, resolvedOutput)), "Output resolves outside the allowed viewer directories");
  try { assert(!(await lstat(output)).isSymbolicLink(), "Output must not be a symbolic link"); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  return output;
}

function compactBranch(branch, label) {
  assert(record(branch) && branch.completed === true, `${label} is not completed`);
  assert(branch.completed_steps === 6000, `${label} did not complete the declared 6000 steps`);
  assert(branch.exact_controller_and_observation_replay === true, `${label} has no successful exact replay`);
  for (const window of ["after_move", "early_after_move_20s", "late_after_move_20s"]) assert(record(branch[window]), `${label}.${window} is missing`);
  assert(Array.isArray(branch.curve) && branch.curve.length > 1, `${label} has no measured curve`);
  let lastTime = -Infinity;
  const curve = branch.curve.map((sample, index) => {
    const time = finite(sample.time_s, `${label}.curve[${index}].time_s`);
    const lux = finite(sample.illuminance_lux, `${label}.curve[${index}].illuminance_lux`);
    assert(time >= 0 && time <= 120 && time > lastTime, `${label} curve times must increase within the declared duration`);
    assert(lux >= 0, `${label} contains negative illuminance`);
    lastTime = time;
    return [time, lux];
  });
  assert(curve[0][0] <= 1 && curve.at(-1)[0] >= 119, `${label} curve does not cover the declared trial`);
  return {
    reward_after_move: finite(branch.after_move.integrated_sensor_reward, `${label}.after_move.integrated_sensor_reward`),
    energy_after_move_j: finite(branch.after_move.electrical_energy_j, `${label}.after_move.electrical_energy_j`),
    early_after_move_lux: finite(branch.early_after_move_20s.mean_received_lux, `${label}.early_after_move_20s.mean_received_lux`),
    late_after_move_lux: finite(branch.late_after_move_20s.mean_received_lux, `${label}.late_after_move_20s.mean_received_lux`),
    curve,
  };
}

function exportReport(report, source) {
  assert(report.schema === "light_adaptation_v1", "Unsupported native adaptation report schema");
  assert(report.finished === true && report.complete === true, "Only finished, fully valid native reports can be published to the viewer");
  assert(report.protocol?.steps === 6000 && report.protocol?.control_dt_s === 0.02 && report.protocol?.intervention_step === 3000 && report.protocol?.intervention_time_s === 60, "The report does not match this viewer’s 120-second, 60-second-intervention protocol");
  assert(Array.isArray(report.pairs) && report.pairs.length === 6, "The completed comparison must contain six pairs");
  const pairIds = new Set();
  const pairs = report.pairs.map((pair, index) => {
    assert(pair.valid === true, `Pair ${index} is not valid`);
    assert(record(pair.checks) && Object.keys(pair.checks).length > 0 && Object.values(pair.checks).every((passed) => passed === true), `Pair ${index} has incomplete or failed checks`);
    assert(typeof pair.pair_id === "string" && pair.pair_id.length > 0 && !pairIds.has(pair.pair_id), `Pair ${index} needs a unique ID`);
    pairIds.add(pair.pair_id);
    assert(Number.isInteger(pair.body_index), `Pair ${index} has an invalid body index`);
    assert(typeof pair.seed === "string" || Number.isSafeInteger(pair.seed), `Pair ${index} seed must be a string or a safe integer`);
    const continued = compactBranch(pair.continued, `${pair.pair_id}.continued`);
    const frozen = compactBranch(pair.frozen, `${pair.pair_id}.frozen`);
    const delta = finite(pair.differences?.after_move_integrated_sensor_reward, `${pair.pair_id}.differences.after_move_integrated_sensor_reward`);
    const computed = continued.reward_after_move - frozen.reward_after_move;
    assert(Math.abs(delta - computed) <= 1e-8 * Math.max(1, Math.abs(computed)), `${pair.pair_id} reward difference disagrees with its branch metrics`);
    assert(typeof pair.assembly_sha256 === "string" && /^[a-f0-9]{64}$/i.test(pair.assembly_sha256), `${pair.pair_id} has no assembly fingerprint`);
    return {
      id: pair.pair_id,
      body_index: pair.body_index,
      body_name: typeof pair.assembly?.name === "string" && pair.assembly.name.trim() ? pair.assembly.name : `Body ${pair.body_index + 1}`,
      assembly_sha256: pair.assembly_sha256,
      seed: String(pair.seed),
      checks: pair.checks,
      exact_replays: true,
      continued,
      frozen,
      reward_delta: delta,
    };
  });
  const bodies = [...new Set(pairs.map((pair) => pair.body_index))];
  assert(bodies.length === 2, "The report must compare exactly two bodies");
  for (const body of bodies) {
    const selected = pairs.filter((pair) => pair.body_index === body);
    assert(selected.length === 3 && new Set(selected.map((pair) => pair.seed)).size === 3, `Body ${body} must have three distinct seeds`);
    assert(new Set(selected.map((pair) => pair.assembly_sha256)).size === 1, `Body ${body} changes assembly between seeds`);
  }
  assert(pairs.filter((pair) => pair.body_index === bodies[0]).map((pair) => pair.seed).sort().join("|") === pairs.filter((pair) => pair.body_index === bodies[1]).map((pair) => pair.seed).sort().join("|"), "Both bodies must use the same three seeds");
  return {
    schema: "light_adaptation_view_v1",
    source,
    protocol: {
      duration_s: 120,
      intervention_s: 60,
      initial_sun: report.protocol.initial_sun,
      moved_sun: report.protocol.moved_sun,
      scope: "Known development bodies and seeds; no held-out evaluation or generalization claim.",
    },
    validation: { complete: true, independently_checked: true, pair_count: pairs.length, body_count: bodies.length, seeds_per_body: 3, exact_replay_count: pairs.length * 2, all_pair_checks_passed: true },
    pairs,
  };
}

async function main() {
  const args = process.argv.slice(2);
  assert(args.length >= 1 && args.length <= 2, "Usage: node tools/export-light-adaptation-view.mjs REPORT_JSON [OUTPUT_JSON]");
  const reportPath = await realpath(path.resolve(args[0]));
  const relative = path.relative(projectRoot, reportPath);
  assert(relative !== "" && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), "Source report must be inside the Droid Blocks project for portable provenance");
  const outputPath = await checkedOutputPath(args[1] ?? path.join(projectRoot, "web", "adaptation-data.json"));
  assert(path.relative(outputPath, reportPath) !== "", "The export must never replace its source report");
  const independentlyChecked = summarizeLightAdaptation(reportPath);
  assert(independentlyChecked.complete === true && independentlyChecked.invariants_verified === true && independentlyChecked.original_continued_compatibility_verified === true && independentlyChecked.full_transition_replays_verified === true, "Independent report validation did not confirm all required evidence");
  const bytes = await readFile(reportPath);
  const report = JSON.parse(bytes.toString("utf8"));
  const digest = sha256(bytes);
  assert(independentlyChecked.report_sha256 === digest, "The independently checked report does not match the bytes to be exported");
  const source = { report_path: relative.split(path.sep).join("/"), report_sha256: digest, report_schema: report.schema };
  const view = exportReport(report, source);
  assert(sha256(await readFile(reportPath)) === digest, "The source report changed during export; no viewer data was written");
  await mkdir(path.dirname(outputPath), { recursive: true });
  const temporary = `${outputPath}.next-${process.pid}`;
  let created = false;
  try {
    await writeFile(temporary, `${JSON.stringify(view)}\n`, { flag: "wx" }); created = true;
    await rename(temporary, outputPath); created = false;
  } finally { if (created) await rm(temporary, { force: true }); }
  process.stdout.write(`Exported ${view.pairs.length} verified pairs to ${path.relative(projectRoot, outputPath)}\nSource SHA-256: ${digest}\n`);
}

main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
