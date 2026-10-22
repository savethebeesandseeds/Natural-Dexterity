"use strict";

const $ = (selector) => document.querySelector(selector);
const ui = {
  loading: $("[data-loading]"), error: $("[data-error]"), errorDetail: $("[data-error-detail]"),
  retry: $("[data-retry]"), results: $("[data-results]"), body: $("[data-body]"), bodyDetail: $("[data-body-detail]"),
  models: $("[data-models]"), historyAccuracy: $("[data-history-accuracy]"), historyDelta: $("[data-history-delta]"),
  probe: $("[data-probe]"), plotTitle: $("[data-plot-title]"), chart: $("[data-chart]"),
  sequenceResults: $("[data-sequence-results]"), validationTitle: $("[data-validation-title]"),
  validationDetail: $("[data-validation-detail]"), checks: $("[data-checks]"),
  sourcePath: $("[data-source-path]"), sourceHash: $("[data-source-hash]"),
  assemblyHash: $("[data-assembly-hash]"), live: $("[data-live]"),
};
const SVG_NS = "http://www.w3.org/2000/svg";
const MODELS = [
  { id: "history", name: "History · 1 second", description: "Recent local sensor readings" },
  { id: "current", name: "Current senses", description: "Only the present readings" },
  { id: "brightness_only", name: "Brightness only", description: "Light without movement readings" },
  { id: "no_light", name: "Without light", description: "Movement history · a control" },
];
const TRAIN_SEQUENCES = [101, 102, 103, 104, 105, 106, 107, 108];
const TEST_SEQUENCES = [109, 110, 111, 112];
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const unitInterval = (value) => finite(value) && value >= 0 && value <= 1;
const fmt = (value, digits = 1) => new Intl.NumberFormat("en", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
const percentage = (value) => fmt(value * 100) + "%";
const percentageDifference = (value) => {
  const points = Math.abs(value * 100) < 0.05 ? 0 : value * 100;
  return (points > 0 ? "+" : points < 0 ? "−" : "") + fmt(Math.abs(points)) + " pp";
};
const matchingIds = (values, expected) => Array.isArray(values) && values.length === expected.length && [...values].sort((first, second) => first - second).every((value, index) => value === expected[index]);
let data = null;
let selectedBody = null;
let selectedModelId = "history";
let loading = false;
let announcement = null;

function assert(condition, message) { if (!condition) throw new Error(message); }

function validate(record) {
  assert(record && record.schema === "light_context_view_v1", "This record uses an unsupported viewer format.");
  assert(record.validation?.complete === true && record.validation?.independently_checked === true, "The native report has not completed independent evidence checks.");
  assert(record.validation.native_trial_count === 48 && record.validation.exact_replay_count === 48, "The record does not contain the complete declared native experiment.");
  assert(record.protocol?.duration_s === 12 && record.protocol?.history_s === 1 && matchingIds(record.protocol.training_sequence_ids, TRAIN_SEQUENCES) && matchingIds(record.protocol.test_sequence_ids, TEST_SEQUENCES), "The training and withheld-sequence protocol is missing or different.");
  assert(typeof record.source?.report_path === "string" && record.source.report_path.length > 0 && /^[a-f0-9]{64}$/i.test(record.source?.report_sha256 ?? ""), "The source report fingerprint is missing.");
  assert(Array.isArray(record.bodies) && record.bodies.length === 2, "The two known bodies are missing.");
  const bodyIds = new Set();
  for (const body of record.bodies) {
    assert(Number.isInteger(body.body_index) && body.body_index >= 0 && !bodyIds.has(body.body_index), "A body ID is invalid or repeated.");
    bodyIds.add(body.body_index);
    assert(typeof body.body_name === "string" && body.body_name.trim().length > 0 && /^[a-f0-9]{64}$/i.test(body.assembly_sha256 ?? ""), "A body label or fingerprint is missing.");
    assert(Array.isArray(body.models) && body.models.length === MODELS.length && new Set(body.models.map((model) => model.id)).size === MODELS.length, "The four sensor views are missing or repeated.");
    for (const definition of MODELS) {
      const model = body.models.find((candidate) => candidate.id === definition.id);
      assert(model && unitInterval(model.accuracy), "A completed accuracy result is missing.");
      assert(Array.isArray(model.sequence_accuracies) && matchingIds(model.sequence_accuracies.map((sequence) => sequence.sequence_id), TEST_SEQUENCES), "A view is missing a withheld-sequence result.");
      assert(model.sequence_accuracies.every((sequence) => unitInterval(sequence.accuracy)), "A sequence accuracy is invalid.");
      const mean = model.sequence_accuracies.reduce((sum, sequence) => sum + sequence.accuracy, 0) / TEST_SEQUENCES.length;
      assert(Math.abs(mean - model.accuracy) < 1e-8, "A reported accuracy does not match the equally weighted sequence average.");
    }
    assert(Array.isArray(body.probes) && matchingIds(body.probes.map((probe) => probe.sequence_id), TEST_SEQUENCES), "The four withheld action sequences are missing.");
    for (const probe of body.probes) {
      for (const { id } of MODELS) {
        const conditions = probe.curves?.[id];
        assert(conditions, "A sensor view has no recorded votes.");
        for (const condition of ["A", "B"]) {
          const curve = conditions[condition];
          assert(Array.isArray(curve) && curve.length > 1, "A recorded vote curve is missing.");
          let previous = -Infinity;
          for (const point of curve) {
            assert(Array.isArray(point) && point.length === 2 && finite(point[0]) && point[0] >= 0 && point[0] <= 12 && point[0] > previous && unitInterval(point[1]), "A recorded vote curve is invalid.");
            previous = point[0];
          }
        }
        assert(conditions.A.length === conditions.B.length && conditions.A.every((point, index) => point[0] === conditions.B[index][0]), "The paired vote curves do not use matching evaluation times.");
      }
    }
  }
  return record;
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = String(value); node.textContent = label;
  return node;
}

function announce(message) {
  window.clearTimeout(announcement); ui.live.textContent = "";
  announcement = window.setTimeout(() => { ui.live.textContent = message; }, 20);
}

function chooseBody(announceChange = true) {
  const previousProbe = ui.probe.value;
  selectedBody = data.bodies.find((body) => String(body.body_index) === ui.body.value);
  if (!selectedBody) return;
  const history = selectedBody.models.find((model) => model.id === "history");
  const current = selectedBody.models.find((model) => model.id === "current");
  ui.bodyDetail.textContent = "Body " + (data.bodies.indexOf(selectedBody) + 1) + " of 2 · 8 training sequences · 4 withheld sequences · 12-second trials";
  ui.historyAccuracy.textContent = percentage(history.accuracy);
  ui.historyDelta.textContent = percentageDifference(history.accuracy - current.accuracy);
  ui.models.replaceChildren(...MODELS.map((definition) => {
    const model = selectedBody.models.find((candidate) => candidate.id === definition.id);
    const accuracies = model.sequence_accuracies.map((sequence) => sequence.accuracy);
    const button = document.createElement("button");
    button.type = "button"; button.className = "model-card"; button.dataset.model = definition.id;
    button.setAttribute("aria-pressed", String(definition.id === selectedModelId));
    for (const [className, text] of [
      ["model-name", definition.name],
      ["model-description", definition.description],
      ["model-accuracy", percentage(model.accuracy)],
      ["model-range", "Sequence range " + percentage(Math.min(...accuracies)) + "–" + percentage(Math.max(...accuracies))],
    ]) {
      const label = document.createElement("span"); label.className = className; label.textContent = text; button.append(label);
    }
    button.addEventListener("click", () => chooseModel(definition.id));
    return button;
  }));
  ui.probe.replaceChildren(...selectedBody.probes.map((probe) => option(probe.sequence_id, "Sequence " + probe.sequence_id)));
  if (selectedBody.probes.some((probe) => String(probe.sequence_id) === previousProbe)) ui.probe.value = previousProbe;
  ui.sourcePath.textContent = data.source.report_path; ui.sourceHash.textContent = data.source.report_sha256;
  ui.assemblyHash.textContent = selectedBody.assembly_sha256;
  renderProbe(false);
  if (announceChange) announce("Showing " + selectedBody.body_name + ". History accuracy " + percentage(history.accuracy) + ", compared with current-senses accuracy " + percentage(current.accuracy) + ". Chance reference is 50 percent.");
}

function chooseModel(id) {
  selectedModelId = id;
  for (const button of ui.models.querySelectorAll("[data-model]")) button.setAttribute("aria-pressed", String(button.dataset.model === id));
  renderProbe();
}

function renderProbe(announceChange = true) {
  if (!selectedBody) return;
  const definition = MODELS.find((model) => model.id === selectedModelId);
  ui.plotTitle.textContent = definition.name + " · " + selectedBody.body_name;
  ui.sequenceResults.replaceChildren(...TEST_SEQUENCES.map((sequenceId) => {
    const row = document.createElement("tr"); row.dataset.selected = String(sequenceId === Number(ui.probe.value));
    const heading = document.createElement("th"); heading.scope = "row"; heading.textContent = String(sequenceId); row.append(heading);
    for (const { id } of MODELS) {
      const result = selectedBody.models.find((model) => model.id === id).sequence_accuracies.find((sequence) => sequence.sequence_id === sequenceId);
      const cell = document.createElement("td"); cell.textContent = percentage(result.accuracy); row.append(cell);
    }
    return row;
  }));
  drawChart();
  if (announceChange) {
    const sequenceAccuracy = selectedBody.models.find((model) => model.id === selectedModelId).sequence_accuracies.find((sequence) => String(sequence.sequence_id) === ui.probe.value).accuracy;
    announce("Showing " + definition.name + " for " + selectedBody.body_name + ", withheld sequence " + ui.probe.value + ". Sequence accuracy " + percentage(sequenceAccuracy) + ". Lines show votes from the separate A and B trials.");
  }
}

function renderEvidence() {
  ui.validationTitle.textContent = "Native records and evaluation passed integrity checks";
  ui.validationDetail.textContent = "Two known bodies · 48 native trials · 48 exact replays · whole action sequences withheld.";
  ui.checks.replaceChildren(...[
    "Verified: native report and source fingerprints.",
    "Verified: 48 native trials and 48 exact replays.",
    "Verified: matching action sequences under conditions A and B.",
    "Verified: complete test sequences withheld from training.",
    "Accuracy and vote curves are evaluation results; passing integrity checks does not require a positive recognition result.",
  ].map((text) => { const item = document.createElement("li"); item.textContent = text; return item; }));
}

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function drawChart() {
  if (!selectedBody) return;
  const probe = selectedBody.probes.find((candidate) => String(candidate.sequence_id) === ui.probe.value);
  if (!probe) return;
  const curves = probe.curves[selectedModelId];
  const definition = MODELS.find((model) => model.id === selectedModelId);
  const width = Math.max(288, ui.chart.clientWidth || 1000);
  const height = Math.max(240, ui.chart.clientHeight || 335);
  const left = width < 500 ? 43 : 62;
  const right = width - 24;
  const top = 28;
  const bottom = height - 40;
  const x = (seconds) => left + seconds / 12 * (right - left);
  const y = (vote) => bottom - vote * (bottom - top);
  const forcedColors = window.matchMedia("(forced-colors: active)").matches;
  const textColor = forcedColors ? "CanvasText" : "#a3b196";
  const gridColor = forcedColors ? "CanvasText" : "#70826b";
  ui.chart.setAttribute("viewBox", "0 0 " + width + " " + height);
  const nodes = [
    svg("title", { id: "chart-title" }, selectedBody.body_name + ", sequence " + probe.sequence_id + ": " + definition.name + " votes for A"),
    svg("desc", { id: "chart-description" }, "The chart spans 0 to 12 seconds and a fixed vote scale from 0 to 1. Solid green is the evaluator's A trial; dashed amber is the B trial. Both use the same action sequence. Values above 0.5 favor A, below 0.5 favor B. p(A) is a smoothed neighbor vote, not calibrated confidence. Only evaluated windows are plotted. First evaluated time " + fmt(curves.A[0][0]) + " seconds; last " + fmt(curves.A.at(-1)[0]) + " seconds."),
  ];
  for (const vote of [0, 0.25, 0.5, 0.75, 1]) {
    nodes.push(svg("line", { x1: left, x2: right, y1: y(vote), y2: y(vote), stroke: gridColor, "stroke-opacity": forcedColors ? 0.6 : 0.24, "stroke-width": 1 }));
    nodes.push(svg("text", { x: left - 9, y: y(vote) + 3, fill: textColor, "font-family": "Consolas, monospace", "font-size": 9, "text-anchor": "end" }, fmt(vote, vote === 0 || vote === 1 ? 0 : vote === 0.5 ? 1 : 2)));
  }
  for (const seconds of width < 500 ? [0, 4, 8, 12] : [0, 2, 4, 6, 8, 10, 12]) {
    nodes.push(svg("line", { x1: x(seconds), x2: x(seconds), y1: bottom, y2: bottom + 5, stroke: textColor, "stroke-width": 1 }));
    nodes.push(svg("text", { x: x(seconds), y: bottom + 21, fill: textColor, "font-family": "Consolas, monospace", "font-size": 10, "text-anchor": "middle" }, String(seconds) + " s"));
  }
  nodes.push(svg("text", { x: left - 9, y: top - 12, fill: textColor, "font-family": "Consolas, monospace", "font-size": 10, "text-anchor": "end" }, "p(A)"));
  nodes.push(svg("line", { x1: left, x2: right, y1: y(0.5), y2: y(0.5), stroke: forcedColors ? "CanvasText" : "#b0b49b", "stroke-opacity": 0.8, "stroke-width": 1, "stroke-dasharray": "2 5" }));
  for (const [condition, color, dash] of [["A", "#8bc3a0", null], ["B", "#e4bc68", "7 5"]]) {
    const stroke = forcedColors ? "CanvasText" : color;
    nodes.push(svg("polyline", { points: curves[condition].map((point) => x(point[0]) + "," + y(point[1])).join(" "), fill: "none", stroke, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", ...(dash ? { "stroke-dasharray": dash } : {}) }));
    for (const point of curves[condition]) {
      nodes.push(svg(condition === "A" ? "circle" : "rect", condition === "A"
        ? { cx: x(point[0]), cy: y(point[1]), r: 2, fill: stroke }
        : { x: x(point[0]) - 2, y: y(point[1]) - 2, width: 4, height: 4, fill: stroke }));
    }
  }
  ui.chart.replaceChildren(...nodes);
}

async function load() {
  if (loading) return;
  loading = true; ui.loading.hidden = false; ui.error.hidden = true; ui.results.hidden = true; ui.retry.disabled = true;
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), 8000);
  try {
    const response = await fetch("/context-data.json", { cache: "no-store", signal: abort.signal });
    if (!response.ok) throw new Error(response.status === 404 ? "The completed native report has not been exported to this page yet." : "The record could not be read (HTTP " + response.status + ").");
    data = validate(await response.json());
    ui.body.replaceChildren(...data.bodies.map((body) => option(body.body_index, (body.body_index + 1) + " · " + body.body_name)));
    ui.results.hidden = false; renderEvidence(); chooseBody(false);
  } catch (error) {
    data = null; selectedBody = null; ui.error.hidden = false;
    ui.errorDetail.textContent = (error.name === "AbortError" ? "Reading the record timed out." : error.message) + " No provisional result is shown.";
  } finally {
    window.clearTimeout(timeout); loading = false; ui.loading.hidden = true; ui.retry.disabled = false;
  }
}

ui.body.addEventListener("change", () => chooseBody());
ui.probe.addEventListener("change", () => renderProbe());
ui.retry.addEventListener("click", load);
new ResizeObserver(drawChart).observe(ui.chart.parentElement);
window.matchMedia("(forced-colors: active)").addEventListener("change", drawChart);
load();

