import { readFile, writeFile, realpath, rename, rm, mkdir, lstat } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { summarizeLightContext } from "./summarize-light-context.mjs";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const unitInterval = (value) => finite(value) && value >= 0 && value <= 1;
const modelIds = ["history", "current", "brightness_only", "no_light"];
const trainingIds = [101, 102, 103, 104, 105, 106, 107, 108];
const testIds = [109, 110, 111, 112];
const matchingIds = (values, expected) => Array.isArray(values) && values.length === expected.length && [...values].sort((first, second) => first - second).every((value, index) => value === expected[index]);
const within = (root, target) => {
  const relative = path.relative(root, target);
  return relative === "" || (relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative));
};

async function checkedOutputPath(input) {
  const output = path.resolve(input);
  const allowed = [path.join(projectRoot, "web"), path.join(projectRoot, "build", "light-context-qa")];
  assert(allowed.some((root) => within(root, output)) && path.extname(output).toLowerCase() === ".json", "Output must be a JSON file under web or build/light-context-qa");
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
  assert([path.join(actualRoot, "web"), path.join(actualRoot, "build", "light-context-qa")].some((root) => within(root, resolvedOutput)), "Output resolves outside the allowed viewer directories");
  try { assert(!(await lstat(output)).isSymbolicLink(), "Output must not be a symbolic link"); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  return output;
}

function checkedViewer(view) {
  assert(view?.schema === "light_context_view_v1", "Independent validation did not return the declared viewer schema");
  assert(view.validation?.complete === true && view.validation?.independently_checked === true, "The compact record must retain successful independent validation");
  assert(view.validation.native_trial_count === 48 && view.validation.exact_replay_count === 48, "The compact record must contain 48 native trials and 48 exact replays");
  assert(view.protocol?.duration_s === 12 && view.protocol?.history_s === 1 && matchingIds(view.protocol.training_sequence_ids, trainingIds) && matchingIds(view.protocol.test_sequence_ids, testIds), "The compact record does not match the declared complete-sequence holdout protocol");
  assert(typeof view.source?.report_path === "string" && view.source.report_path.length > 0 && /^[a-f0-9]{64}$/i.test(view.source?.report_sha256 ?? ""), "The compact record is missing source provenance");
  assert(Array.isArray(view.bodies) && view.bodies.length === 2, "The compact record must contain two known bodies");
  const bodyIds = new Set();
  for (const body of view.bodies) {
    assert(Number.isInteger(body.body_index) && body.body_index >= 0 && !bodyIds.has(body.body_index), "A compact body ID is invalid or repeated");
    bodyIds.add(body.body_index);
    assert(typeof body.body_name === "string" && body.body_name.trim().length > 0 && /^[a-f0-9]{64}$/i.test(body.assembly_sha256 ?? ""), "A compact body name or fingerprint is missing");
    assert(Array.isArray(body.models) && body.models.length === modelIds.length && new Set(body.models.map((model) => model.id)).size === modelIds.length, "The compact body must retain four distinct sensor views");
    for (const id of modelIds) {
      const model = body.models.find((candidate) => candidate.id === id);
      assert(model && unitInterval(model.accuracy), "A compact sensor-view accuracy is missing or invalid");
      assert(Array.isArray(model.sequence_accuracies) && matchingIds(model.sequence_accuracies.map((sequence) => sequence.sequence_id), testIds) && model.sequence_accuracies.every((sequence) => unitInterval(sequence.accuracy)), "The compact view must retain four withheld-sequence accuracies");
      const average = model.sequence_accuracies.reduce((sum, sequence) => sum + sequence.accuracy, 0) / testIds.length;
      assert(Math.abs(average - model.accuracy) < 1e-8, "A compact accuracy disagrees with the equally weighted sequence average");
    }
    assert(Array.isArray(body.probes) && matchingIds(body.probes.map((probe) => probe.sequence_id), testIds), "The compact body must retain all four withheld sequences");
    for (const probe of body.probes) {
      for (const id of modelIds) {
        const conditions = probe.curves?.[id];
        assert(conditions, "A compact sensor view has no recorded curves");
        for (const condition of ["A", "B"]) {
          const curve = conditions[condition];
          assert(Array.isArray(curve) && curve.length > 1, "A compact vote curve is missing");
          let previous = -Infinity;
          for (const point of curve) {
            assert(Array.isArray(point) && point.length === 2 && finite(point[0]) && point[0] >= 0 && point[0] <= 12 && point[0] > previous && unitInterval(point[1]), "A compact vote curve has invalid times or votes");
            previous = point[0];
          }
        }
        assert(conditions.A.length === conditions.B.length && conditions.A.every((point, index) => point[0] === conditions.B[index][0]), "Paired compact curves have different evaluation times");
      }
    }
  }
  return view;
}

async function main() {
  const args = process.argv.slice(2);
  assert(args.length >= 1 && args.length <= 2, "Usage: node tools/export-light-context-view.mjs REPORT_JSON [OUTPUT_JSON]");
  const reportPath = await realpath(path.resolve(args[0]));
  const relative = path.relative(projectRoot, reportPath);
  assert(relative !== "" && within(projectRoot, reportPath), "Source report must be inside the Droid Blocks project for portable provenance");
  const outputPath = await checkedOutputPath(args[1] ?? path.join(projectRoot, "web", "context-data.json"));
  assert(path.relative(outputPath, reportPath) !== "", "The export must never replace its source report");
  const summary = summarizeLightContext(reportPath);
  assert(summary.complete === true && summary.sources_verified === true && summary.invariants_verified === true, "Independent validation did not confirm all required evidence");
  const bytes = await readFile(reportPath);
  const digest = sha256(bytes);
  assert(summary.report_sha256 === digest, "The independently checked report does not match the bytes to be exported");
  const view = checkedViewer(summary.viewer);
  assert(view.source.report_sha256 === digest, "The compact viewer fingerprint differs from the independently checked report");
  view.source = { ...view.source, report_path: relative.split(path.sep).join("/"), report_sha256: digest };
  assert(sha256(await readFile(reportPath)) === digest, "The source report changed during export; no viewer data was written");
  await mkdir(path.dirname(outputPath), { recursive: true });
  const temporary = outputPath + ".next-" + process.pid;
  let created = false;
  try {
    await writeFile(temporary, JSON.stringify(view) + "\n", { flag: "wx" }); created = true;
    await rename(temporary, outputPath); created = false;
  } finally { if (created) await rm(temporary, { force: true }); }
  process.stdout.write("Exported " + view.bodies.length + " independently verified bodies to " + path.relative(projectRoot, outputPath) + "\nSource SHA-256: " + digest + "\n");
}

main().catch((error) => { process.stderr.write(error.message + "\n"); process.exitCode = 1; });

