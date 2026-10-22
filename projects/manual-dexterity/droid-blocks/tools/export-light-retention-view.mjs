import { readFile, writeFile, realpath, rename, rm, mkdir, lstat } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { summarizeLightRetention } from "./summarize-light-retention.mjs";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const finite = (value, label) => { assert(typeof value === "number" && Number.isFinite(value), label + " must be a finite number"); return value; };
const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const within = (root, target) => {
  const relative = path.relative(root, target);
  return relative === "" || (relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative));
};
const sameNumber = (first, second) => Math.abs(first - second) <= 1e-8 * Math.max(1, Math.abs(second));

async function checkedOutputPath(input) {
  const output = path.resolve(input);
  const allowed = [path.join(projectRoot, "web"), path.join(projectRoot, "build", "light-retention-qa")];
  assert(allowed.some((root) => within(root, output)) && path.extname(output).toLowerCase() === ".json", "Output must be a JSON file under web or build/light-retention-qa");
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
  assert([path.join(actualRoot, "web"), path.join(actualRoot, "build", "light-retention-qa")].some((root) => within(root, resolvedOutput)), "Output resolves outside the allowed viewer directories");
  try { assert(!(await lstat(output)).isSymbolicLink(), "Output must not be a symbolic link"); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  return output;
}

function compactBranch(branch, label) {
  assert(record(branch) && branch.completed === true, label + " is not completed");
  assert(branch.completed_steps === 9000, label + " did not complete the declared 9000 steps");
  assert(branch.exact_controller_and_observation_replay === true && branch.exact_full_transition_replay === true, label + " has no successful exact full-transition replay");
  for (const window of ["after_return", "early_after_return_20s", "late_after_return_20s"]) assert(record(branch[window]), label + "." + window + " is missing");
  assert(Array.isArray(branch.curve) && branch.curve.length > 1, label + " has no measured curve");
  let lastTime = -Infinity;
  const curve = branch.curve.map((sample, index) => {
    const time = finite(sample.time_s, label + ".curve[" + index + "].time_s");
    const lux = finite(sample.illuminance_lux, label + ".curve[" + index + "].illuminance_lux");
    assert(time >= 0 && time <= 180 && time > lastTime, label + " curve times must increase within the declared duration");
    assert(lux >= 0, label + " contains negative illuminance");
    lastTime = time;
    return [time, lux];
  });
  assert(curve[0][0] <= 1 && curve.at(-1)[0] >= 179, label + " curve does not cover the declared trial");
  const compact = {
    early_reward: finite(branch.early_after_return_20s.integrated_sensor_reward, label + ".early_after_return_20s.integrated_sensor_reward"),
    return_reward: finite(branch.after_return.integrated_sensor_reward, label + ".after_return.integrated_sensor_reward"),
    early_lux: finite(branch.early_after_return_20s.mean_received_lux, label + ".early_after_return_20s.mean_received_lux"),
    return_lux: finite(branch.after_return.mean_received_lux, label + ".after_return.mean_received_lux"),
    late_lux: finite(branch.late_after_return_20s.mean_received_lux, label + ".late_after_return_20s.mean_received_lux"),
    return_energy_j: finite(branch.after_return.electrical_energy_j, label + ".after_return.electrical_energy_j"),
    late_energy_j: finite(branch.late_after_return_20s.electrical_energy_j, label + ".late_after_return_20s.electrical_energy_j"),
    curve,
  };
  assert([compact.early_lux, compact.return_lux, compact.late_lux, compact.return_energy_j, compact.late_energy_j].every((value) => value >= 0), label + " contains negative received light or energy");
  return compact;
}

function exportReport(report, source) {
  assert(report.schema === "light_retention_v1", "Unsupported native retention report schema");
  assert(report.finished === true && report.complete === true, "Only finished, fully valid native reports can be published to the viewer");
  assert(report.protocol?.steps === 9000 && report.protocol?.control_dt_s === 0.02 && report.protocol?.first_move_step === 3000 && report.protocol?.return_step === 6000 && report.protocol?.first_move_time_s === 60 && report.protocol?.return_time_s === 120, "The report does not match this viewer's 180-second return protocol");
  assert(JSON.stringify(report.protocol.early_window_s) === "[120,140]" && JSON.stringify(report.protocol.late_window_s) === "[160,180]", "The report does not match the declared early and late return windows");
  assert(Array.isArray(report.cases) && report.cases.length === 6, "The completed comparison must contain six cases");
  const caseIds = new Set();
  const cases = report.cases.map((item, index) => {
    assert(item.valid === true, "Case " + index + " is not valid");
    assert(record(item.checks) && Object.keys(item.checks).length > 0 && Object.values(item.checks).every((passed) => passed === true), "Case " + index + " has incomplete or failed checks");
    assert(typeof item.case_id === "string" && item.case_id.length > 0 && !caseIds.has(item.case_id), "Case " + index + " needs a unique ID");
    caseIds.add(item.case_id);
    assert(Number.isInteger(item.body_index) && item.body_index >= 0, "Case " + index + " has an invalid body index");
    assert(typeof item.seed === "string" || Number.isSafeInteger(item.seed), "Case " + index + " seed must be a string or a safe integer");
    const continued = compactBranch(item.continued, item.case_id + ".continued");
    const current = compactBranch(item.current_frozen, item.case_id + ".current_frozen");
    const recalled = compactBranch(item.recalled_frozen, item.case_id + ".recalled_frozen");
    const earlyDelta = finite(item.differences?.primary_early_recalled_minus_current_reward, item.case_id + ".differences.primary_early_recalled_minus_current_reward");
    const returnDelta = finite(item.differences?.full_return_recalled_minus_current_reward, item.case_id + ".differences.full_return_recalled_minus_current_reward");
    assert(sameNumber(earlyDelta, recalled.early_reward - current.early_reward), item.case_id + " early reward difference disagrees with its branch metrics");
    assert(sameNumber(returnDelta, recalled.return_reward - current.return_reward), item.case_id + " full-return reward difference disagrees with its branch metrics");
    assert(typeof item.assembly_sha256 === "string" && /^[a-f0-9]{64}$/i.test(item.assembly_sha256), item.case_id + " has no assembly fingerprint");
    return {
      id: item.case_id,
      body_index: item.body_index,
      body_name: typeof item.assembly?.name === "string" && item.assembly.name.trim() ? item.assembly.name : "Body " + (item.body_index + 1),
      assembly_sha256: item.assembly_sha256,
      seed: String(item.seed),
      checks: item.checks,
      exact_replays: true,
      continued,
      current_frozen: current,
      recalled_frozen: recalled,
      early_reward_delta: earlyDelta,
      return_reward_delta: returnDelta,
    };
  });
  const bodies = [...new Set(cases.map((item) => item.body_index))];
  assert(bodies.length === 2, "The report must compare exactly two bodies");
  for (const body of bodies) {
    const selected = cases.filter((item) => item.body_index === body);
    assert(selected.length === 3 && new Set(selected.map((item) => item.seed)).size === 3, "Body " + body + " must have three distinct seeds");
    assert(new Set(selected.map((item) => item.assembly_sha256)).size === 1, "Body " + body + " changes assembly between seeds");
  }
  assert(cases.filter((item) => item.body_index === bodies[0]).map((item) => item.seed).sort().join("|") === cases.filter((item) => item.body_index === bodies[1]).map((item) => item.seed).sort().join("|"), "Both bodies must use the same three seeds");
  return {
    schema: "light_retention_view_v1",
    source,
    protocol: {
      duration_s: 180,
      first_move_s: 60,
      return_s: 120,
      early_window_s: report.protocol.early_window_s,
      late_window_s: report.protocol.late_window_s,
      initial_sun: report.protocol.initial_sun,
      moved_sun: report.protocol.moved_sun,
      scope: "Known development bodies and seeds. The experiment selects the first-visit checkpoint; no automatic memory retrieval or generalization claim.",
    },
    validation: { complete: true, independently_checked: true, case_count: cases.length, body_count: bodies.length, seeds_per_body: 3, exact_replay_count: cases.length * 3, all_case_checks_passed: true },
    cases,
  };
}

async function main() {
  const args = process.argv.slice(2);
  assert(args.length >= 1 && args.length <= 2, "Usage: node tools/export-light-retention-view.mjs REPORT_JSON [OUTPUT_JSON]");
  const reportPath = await realpath(path.resolve(args[0]));
  const relative = path.relative(projectRoot, reportPath);
  assert(relative !== "" && within(projectRoot, reportPath), "Source report must be inside the Droid Blocks project for portable provenance");
  const outputPath = await checkedOutputPath(args[1] ?? path.join(projectRoot, "web", "retention-data.json"));
  assert(path.relative(outputPath, reportPath) !== "", "The export must never replace its source report");
  const independentlyChecked = summarizeLightRetention(reportPath);
  assert(independentlyChecked.complete === true && independentlyChecked.invariants_verified === true && independentlyChecked.original_prefix_compatibility_verified === true && independentlyChecked.full_transition_replays_verified === true, "Independent report validation did not confirm all required evidence");
  const bytes = await readFile(reportPath);
  const report = JSON.parse(bytes.toString("utf8"));
  const digest = sha256(bytes);
  assert(independentlyChecked.report_sha256 === digest, "The independently checked report does not match the bytes to be exported");
  const source = { report_path: relative.split(path.sep).join("/"), report_sha256: digest, report_schema: report.schema };
  const view = exportReport(report, source);
  assert(sha256(await readFile(reportPath)) === digest, "The source report changed during export; no viewer data was written");
  await mkdir(path.dirname(outputPath), { recursive: true });
  const temporary = outputPath + ".next-" + process.pid;
  let created = false;
  try {
    await writeFile(temporary, JSON.stringify(view) + "\n", { flag: "wx" }); created = true;
    await rename(temporary, outputPath); created = false;
  } finally { if (created) await rm(temporary, { force: true }); }
  process.stdout.write("Exported " + view.cases.length + " independently verified cases to " + path.relative(projectRoot, outputPath) + "\nSource SHA-256: " + digest + "\n");
}

main().catch((error) => { process.stderr.write(error.message + "\n"); process.exitCode = 1; });

