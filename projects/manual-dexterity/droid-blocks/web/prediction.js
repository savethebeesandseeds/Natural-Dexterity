"use strict";

const $ = (selector) => document.querySelector(selector);
const ui = Object.fromEntries([
  "loading", "error", "error-detail", "retry", "results", "body", "body-detail", "models", "gate", "metrics", "channel-errors",
  "trial", "origin", "channel", "chart", "request-chart", "forecast-label", "forecast-note", "matrix", "row", "column",
  "matrix-content", "no-matrix", "matrix-description", "matrix-dimension", "matrix-canvas", "matrix-entry", "entry-description",
  "matrix-head", "matrix-rows", "download-matrices", "check-title", "check-summary", "checks", "report-path", "report-hash",
  "protocol-hash", "body-hash", "live", "forecast-failure", "trial-errors", "subset-summary",
].map((key) => [key, $(`[data-${key}]`)]));
const SVG_NS = "http://www.w3.org/2000/svg";
const MODELS = [
  { id: "history_input", name: "History + request", description: "One second of outputs, past requests and a future request tape", dimension: 142 },
  { id: "current_input", name: "Current + request", description: "Present outputs and a future request tape", dimension: 12 },
  { id: "history_no_input", name: "History only", description: "Output history; no explicit past or future requests", dimension: 132 },
  { id: "persistence", name: "Persistence", description: "Hold each output at its last measured value", dimension: null },
];
const CHANNELS = [
  { name: "Executed motor effort", short: "effort", scale: .15, unit: "effort", primary: false },
  { name: "Encoder sine", short: "sin(q₁)", scale: 1, unit: "dimensionless", primary: true },
  { name: "Encoder cosine", short: "cos(q₁)", scale: 1, unit: "dimensionless", primary: true },
  { name: "Motor velocity", short: "motor velocity", scale: 10, unit: "rad/s", primary: true },
  { name: "Motor current", short: "current", scale: .5, unit: "A", primary: false },
  { name: "IMU specific force x", short: "force x", scale: 50, unit: "m/s²", primary: true },
  { name: "IMU specific force y", short: "force y", scale: 50, unit: "m/s²", primary: false },
  { name: "IMU specific force z", short: "force z", scale: 50, unit: "m/s²", primary: true },
  { name: "IMU angular velocity x", short: "gyro x", scale: 10, unit: "rad/s", primary: false },
  { name: "IMU angular velocity y", short: "gyro y", scale: 10, unit: "rad/s", primary: true },
  { name: "IMU angular velocity z", short: "gyro z", scale: 10, unit: "rad/s", primary: false },
  { name: "Delivered light", short: "light", scale: 1000, unit: "lux", primary: true },
];
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const hash = (value) => typeof value === "string" && /^[a-f0-9]{64}$/i.test(value);
const vector = (value, length) => Array.isArray(value) && value.length === length && value.every(finite);
const matrix = (value, rows, columns) => Array.isArray(value) && value.length === rows && value.every((row) => vector(row, columns));
const fmt = (value, digits = 3) => Number.isFinite(value) ? new Intl.NumberFormat("en", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value) : "Unavailable";
const compact = (value) => !finite(value) ? "Unavailable" : Math.abs(value) >= 10000 || (Math.abs(value) > 0 && Math.abs(value) < .001) ? value.toExponential(2) : new Intl.NumberFormat("en", { maximumFractionDigits: 3 }).format(value);
const failureFor = (view, id) => { const failure = view.failures?.[id]; return typeof failure === "string" ? failure : failure?.error ?? failure?.reason ?? null; };
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const el = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; };
const svg = (tag, attributes, text) => { const node = document.createElementNS(SVG_NS, tag); for (const [key, value] of Object.entries(attributes ?? {})) node.setAttribute(key, String(value)); if (text !== undefined) node.textContent = text; return node; };
let record = null;
let body = null;
let modelId = "history_input";
let forecast = null;
let channel = 11;
let activeMatrix = null;
let cellRow = 0;
let cellColumn = 0;
let matrixImage = null;
let matrixDownloadUrl = null;
let loading = false;
let resizeFrame = null;

function validate(data) {
  assert(data?.schema === "light_prediction_view_v1", "This record has an unsupported viewer format.");
  assert(data.checks?.complete === true && data.checks?.independently_checked === true, "The record has not completed independent validation.");
  assert(Number.isInteger(data.checks.trials) && data.checks.trials > 0 && data.checks.replays === data.checks.trials && data.checks.models_sealed === true, "Complete native replay and sealed-model checks are missing.");
  assert(typeof data.source?.report_path === "string" && hash(data.source.report_sha256) && hash(data.source.protocol_sha256) && Number.isInteger(data.source.source_count), "The source report or protocol fingerprint is missing.");
  assert(Array.isArray(data.bodies) && data.bodies.length === 3, "The three declared body records are missing.");
  const bodyIds = new Set();
  for (const item of data.bodies) {
    assert((typeof item.id === "string" || Number.isInteger(item.id)) && !bodyIds.has(String(item.id)), "A body identifier is invalid or repeated.");
    bodyIds.add(String(item.id));
    assert(typeof item.name === "string" && hash(item.assembly_sha256), "A body name or assembly fingerprint is missing.");
    assert(Array.isArray(item.models) && item.models.length === 3 && Array.isArray(item.metrics) && item.metrics.length === 4, "A predictor or comparison is missing.");
    for (const spec of MODELS) {
      const measures = item.metrics.filter((entry) => entry.model_id === spec.id);
      assert(measures.length === 1, `Missing or repeated metrics for ${spec.id}.`);
      assert(Array.isArray(measures[0].horizons) && measures[0].horizons.length === 3, "The three forecast horizons are missing.");
      assert(typeof measures[0].readiness === "boolean" || measures[0].readiness === null, "A readiness decision is missing.");
      const hasFailures = Number.isInteger(measures[0].failed_forecasts) && measures[0].failed_forecasts > 0 || item.forecasts?.some((view) => Boolean(failureFor(view, spec.id)));
      if (hasFailures) assert(measures[0].readiness !== true, "A model with failed forecasts cannot pass readiness.");
      for (const steps of [1, 4, 10]) {
        const horizon = measures[0].horizons.find((entry) => entry.steps === steps);
        const unavailableAllowed = hasFailures || horizon?.metric_overflow === true;
        const validError = (value) => finite(value) && value >= 0 || unavailableAllowed && value === null;
        assert(horizon && Math.abs(horizon.seconds - steps / 10) < 1e-10 && validError(horizon.rmse) && (unavailableAllowed && horizon.channels_rmse === null || Array.isArray(horizon.channels_rmse) && horizon.channels_rmse.length === 12 && horizon.channels_rmse.every(validError)), "A forecast error metric is missing without a recorded model failure.");
        assert(Array.isArray(horizon.trials) && horizon.trials.length > 0 && horizon.trials.every((trial) => typeof trial.trial_id === "string" && validError(trial.rmse)), "Trial error ranges cannot be reconstructed.");
      }
      if (spec.id === "persistence") continue;
      const candidates = item.models.filter((entry) => entry.id === spec.id);
      assert(candidates.length === 1, `Missing or repeated model ${spec.id}.`);
      const candidate = candidates[0];
      const n = spec.dimension;
      assert(candidate.state_dimension === n && candidate.taps === (spec.id === "current_input" ? 1 : 11), "The predictive state dimensions differ from the declared model.");
      assert(matrix(candidate.A, n, n) && matrix(candidate.B, n, 1) && matrix(candidate.C, 12, n) && matrix(candidate.D, 12, 1) && vector(candidate.c, n), "A fitted state-space matrix is missing, nonfinite or dimensionally inconsistent.");
      assert(candidate.D.every((row) => row[0] === 0), "The declared zero direct-feedthrough matrix differs.");
    }
    assert(Array.isArray(item.forecasts) && item.forecasts.length > 0, "The withheld recursive forecasts are missing.");
    const origins = new Set();
    for (const view of item.forecasts) {
      const key = `${view.trial_id}:${view.origin_index}`;
      assert(typeof view.trial_id === "string" && Number.isInteger(view.origin_index) && finite(view.origin_time_s) && !origins.has(key), "A forecast origin is missing or repeated.");
      origins.add(key);
      assert(vector(view.initial_output, 12) && matrix(view.actual, 10, 12) && vector(view.requests, 10), "A measured response or request tape is incomplete.");
      for (const spec of MODELS) {
        const predictions = view.predictions?.[spec.id];
        const failure = failureFor(view, spec.id);
        assert(Array.isArray(predictions) && predictions.length <= 10 && predictions.every((row) => vector(row, 12)) && (predictions.length === 10 || typeof failure === "string" && failure.length > 0), "A recursive forecast is incomplete without an explicit failure and retained finite prefix.");
      }
    }
  }
  return data;
}

function fillSelect(select, options, value) {
  select.replaceChildren(...options.map((item) => { const option = el("option", item.label); option.value = String(item.value); return option; }));
  if (value !== undefined && options.some((item) => String(item.value) === String(value))) select.value = String(value);
}

function selectedMetric() { return body.metrics.find((entry) => entry.model_id === modelId); }
function selectedModel() { return body.models.find((entry) => entry.id === modelId); }
function modelName() { return MODELS.find((entry) => entry.id === modelId).name; }

function renderBody() {
  ui["body-detail"].textContent = "Separate fitting for this assembly. A changed body is a new identification case, not a zero-shot transfer claim.";
  ui["body-hash"].textContent = body.assembly_sha256;
  const trials = [...new Map(body.forecasts.map((item) => [item.trial_id, item])).values()];
  fillSelect(ui.trial, trials.map((item) => ({ value: item.trial_id, label: `Sequence ${item.seed} · condition ${item.condition}` })));
  renderOrigins();
  renderModel();
}

function renderOrigins() {
  const previousOrigin = ui.origin.value;
  const forecasts = body.forecasts.filter((item) => item.trial_id === ui.trial.value).sort((a, b) => a.origin_index - b.origin_index);
  fillSelect(ui.origin, forecasts.map((item) => ({ value: item.origin_index, label: `${fmt(item.origin_time_s, 2)} s · frame ${item.origin_index}` })), previousOrigin);
  selectForecast();
}

function selectForecast() {
  forecast = body.forecasts.find((item) => item.trial_id === ui.trial.value && String(item.origin_index) === ui.origin.value);
  drawForecast();
}

function renderModel() {
  ui.models.replaceChildren(...MODELS.map((spec) => {
    const measure = body.metrics.find((item) => item.model_id === spec.id);
    const horizon = measure.horizons.find((item) => item.steps === 10);
    const button = el("button", undefined, "model-card");
    button.type = "button";
    button.setAttribute("aria-pressed", String(spec.id === modelId));
    const score = el("span", fmt(horizon.rmse, 3), "model-score");
    score.append(el("small", " · 1 s RMSE"));
    button.append(el("span", spec.name, "model-name"), el("span", spec.description, "model-description"), score, el("span", spec.dimension === null ? "No fitted coefficients" : `z ∈ R${superscript(spec.dimension)} · ${spec.dimension} state coordinates`, "model-state"));
    if (!finite(horizon.rmse)) score.classList.add("model-score-unavailable");
    button.addEventListener("click", () => { modelId = spec.id; renderModel(); ui.models.querySelector('[aria-pressed="true"]').focus({ preventScroll: true }); announce(`${spec.name}. One-second normalized RMSE ${fmt(horizon.rmse, 3)}.`); });
    return button;
  }));
  renderMetrics();
  renderMatrices();
  drawForecast();
}

function superscript(value) { return String(value).split("").map((character) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(character)]).join(""); }

function renderMetrics() {
  const measure = selectedMetric();
  const candidate = modelId === "history_input" || modelId === "current_input";
  const gate = el("span", !candidate ? "Reference predictor · gate not applicable" : measure.readiness ? "Declared readiness gate: passed" : "Declared readiness gate: not passed", !candidate ? "gate-none" : measure.readiness ? "gate-pass" : "gate-fail");
  ui.gate.replaceChildren(gate);
  ui.metrics.replaceChildren(...[1, 4, 10].map((step) => {
    const horizon = measure.horizons.find((entry) => entry.steps === step);
    const errors = horizon.trials.map((item) => item.rmse);
    const row = el("tr");
    const header = el("th", `${fmt(horizon.seconds, 1)} s · ${step} step${step > 1 ? "s" : ""}`); header.scope = "row";
    row.append(header, el("td", fmt(horizon.rmse, 4)), el("td", errors.every(finite) ? `${fmt(Math.min(...errors), 4)} – ${fmt(Math.max(...errors), 4)}` : "Unavailable · failed forecast"));
    return row;
  }));
  ui["channel-errors"].replaceChildren(...CHANNELS.map((item, index) => {
    const row = el("tr"); row.dataset.selected = String(index === channel);
    const header = el("th", `${item.name} · ${item.unit}`); header.scope = "row";
    header.append(el("small", item.primary ? "Included in primary normalized score" : "Recorded separately from primary score"));
    row.append(header, ...[1, 4, 10].map((step) => { const value = measure.horizons.find((entry) => entry.steps === step).channels_rmse?.[index]; return el("td", finite(value) ? compact(value * item.scale) : "Unavailable"); }));
    return row;
  }));
  const trials = measure.horizons.find((entry) => entry.steps === 10).trials;
  ui["trial-errors"].replaceChildren(...trials.map((trial) => {
    const row = el("tr"), header = el("th", `Sequence ${trial.seed} · ${trial.condition}`); header.scope = "row";
    row.append(header, ...[1, 4, 10].map((step) => el("td", fmt(measure.horizons.find((entry) => entry.steps === step).trials.find((entry) => entry.trial_id === trial.trial_id)?.rmse, 4))));
    return row;
  }));
  const subset = measure.changed_request_subset;
  ui["subset-summary"].textContent = subset ? `Changed-request subset at 1 s: ${subset.origin_count ?? subset.origins ?? "recorded"} origins · primary RMSE ${fmt(subset.rmse, 4)}. This subset includes origins whose future request changes within the next second.` : "The prospective gate also compares the subset where the future request changes within the next second; consult the exported record for its complete counts and errors.";
}

function chartBase(target, min, max, title, description, request = false) {
  const width = Math.max(340, Math.round(target.getBoundingClientRect().width));
  const height = Math.max(request ? 100 : 250, Math.round(target.getBoundingClientRect().height));
  const left = 65, right = 24, top = request ? 12 : 30, bottom = request ? 30 : 45;
  const x = (time) => left + time * (width - left - right);
  const y = (value) => height - bottom - (value - min) / (max - min) * (height - top - bottom);
  const prefix = request ? "request" : "forecast";
  target.setAttribute("viewBox", `0 0 ${width} ${height}`);
  target.replaceChildren(svg("title", { id: `${prefix}-svg-title` }, title), svg("desc", { id: `${prefix}-svg-description` }, description));
  const grid = svg("g", { "font-family": "Consolas, monospace", "font-size": request ? 10 : 11, fill: "#a9b99d" });
  const ticks = request ? [-1, 0, 1] : Array.from({ length: 5 }, (_, index) => min + (max - min) * index / 4);
  for (const value of ticks) {
    grid.append(svg("line", { x1: left, y1: y(value), x2: width - right, y2: y(value), stroke: "#38513f", "stroke-width": 1 }), svg("text", { x: left - 9, y: y(value) + 4, "text-anchor": "end" }, compact(value)));
  }
  for (const value of [0, .2, .4, .6, .8, 1]) {
    grid.append(svg("text", { x: x(value), y: height - bottom + 19, "text-anchor": "middle" }, fmt(value, 1)));
  }
  if (!request) grid.append(svg("text", { x: (left + width - right) / 2, y: height - 7, "text-anchor": "middle", fill: "#93a98e", "font-size": 10 }, "Seconds after forecast origin"));
  target.append(grid);
  return { x, y, width, left, right };
}

function drawForecast() {
  if (!forecast || !body || ui.results.hidden) return;
  const item = CHANNELS[channel];
  const measured = [forecast.initial_output[channel], ...forecast.actual.map((row) => row[channel])].map((value) => value * item.scale);
  const predicted = [forecast.initial_output[channel], ...forecast.predictions[modelId].map((row) => row[channel])].map((value) => value * item.scale);
  const failure = failureFor(forecast, modelId);
  const all = [...measured, ...predicted];
  let min = Math.min(...all), max = Math.max(...all);
  const padding = Math.max((max - min) * .13, item.scale * .001, Math.abs(max) * .005);
  min -= padding; max += padding;
  const description = `${item.name} in ${item.unit}. ${modelName()} starts from the actual measurement at ${fmt(forecast.origin_time_s, 2)} seconds and retains ${predicted.length - 1} predicted steps without future measurements. Measured one-second endpoint ${compact(measured[10])}; predicted ${compact(predicted[10])}.${failure ? ` Recorded forecast failure: ${failure}` : ""}`;
  const axes = chartBase(ui.chart, min, max, `${item.name}: measured and recursive prediction`, description);
  ui.chart.append(svg("text", { x: axes.left, y: 17, fill: "#c2c9aa", "font-size": 11, "font-family": "Consolas, monospace" }, `${item.name} · ${item.unit}`));
  const path = (values) => values.map((value, index) => `${index === 0 ? "M" : "L"}${axes.x(index / 10).toFixed(2)},${axes.y(value).toFixed(2)}`).join(" ");
  if (predicted.length < 11) {
    const end = (predicted.length - 1) / 10;
    ui.chart.append(svg("rect", { x: axes.x(end), y: 25, width: axes.x(1) - axes.x(end), height: Math.max(100, ui.chart.getBoundingClientRect().height - 70), fill: "#bc8759", "fill-opacity": .09 }), svg("line", { x1: axes.x(end), x2: axes.x(end), y1: 25, y2: ui.chart.getBoundingClientRect().height - 45, stroke: "#bc946a", "stroke-dasharray": "3 4" }));
  }
  ui.chart.append(svg("path", { d: path(measured), fill: "none", stroke: "#96c7aa", "stroke-width": 2.5, "stroke-linejoin": "round" }), svg("path", { d: path(predicted), fill: "none", stroke: "#efc375", "stroke-width": 2.5, "stroke-dasharray": "7 5", "stroke-linejoin": "round" }));
  for (let index = 1; index < measured.length; index++) {
    const dot = svg("circle", { cx: axes.x(index / 10), cy: axes.y(measured[index]), r: 2.6, fill: "#96c7aa" });
    dot.append(svg("title", {}, `${fmt(index / 10, 1)} s: measured ${compact(measured[index])}, predicted ${compact(predicted[index])} ${item.unit}`));
    ui.chart.append(dot);
  }
  const initial = svg("circle", { cx: axes.x(0), cy: axes.y(measured[0]), r: 4, fill: "#f0e4ba", stroke: "#173c2c", "stroke-width": 1.5 });
  initial.append(svg("title", {}, `Actual initial measurement: ${compact(measured[0])} ${item.unit}`));
  ui.chart.append(initial);
  ui["forecast-label"].textContent = `Sequence ${forecast.seed} · ${forecast.condition} · origin ${fmt(forecast.origin_time_s, 2)} s`;
  ui["forecast-note"].textContent = `${modelName()}. The shared point at 0 s is measured, not a forecast. End at 1 s: measured ${compact(measured[10])}, predicted ${compact(predicted[10])} ${item.unit}. ${modelId === "history_no_input" || modelId === "persistence" ? "This reference predictor ignores the future request tape shown below." : "Only the planned requests, not future executed effort or sensor readings, are supplied after the origin."}`;
  ui["forecast-failure"].hidden = !failure;
  ui["forecast-failure"].textContent = failure ? `Recorded forecast failure after ${predicted.length - 1} retained finite steps: ${failure}. The predicted line stops at the retained prefix; no value is filled in after it.` : "";
  const requestAxes = chartBase(ui["request-chart"], -1.25, 1.25, "Known future requested motor input", `Ten requested-input intervals, normalized by 0.15: ${forecast.requests.map(compact).join(", ")}. These are not future executed efforts.`, true);
  const requestPath = forecast.requests.map((value, index) => `${index === 0 ? "M" : "L"}${requestAxes.x(index / 10)},${requestAxes.y(value)} L${requestAxes.x((index + 1) / 10)},${requestAxes.y(value)}`).join(" ");
  ui["request-chart"].append(svg("path", { d: requestPath, fill: "none", stroke: "#b4ba8c", "stroke-width": 2 }));
}

function coordinate(index, model = selectedModel()) {
  const outputCount = model.taps * 12;
  if (index < outputCount) return `y[k−${Math.floor(index / 12)}] · ${CHANNELS[index % 12].short}`;
  return `r[k−${index - outputCount + 1}] · previous request`;
}

function matrixLabels(name, values) {
  const model = selectedModel();
  const rowLabel = (index) => name === "C" || name === "D" ? `y[${index}] · ${CHANNELS[index].short}` : `z+[${index}] · ${coordinate(index, model)}`;
  const columnLabel = (index) => name === "B" || name === "D" ? "r[k] · known request" : name === "c" ? "affine offset" : `z[${index}] · ${coordinate(index, model)}`;
  return { rows: values.map((_, index) => rowLabel(index)), columns: values[0].map((_, index) => columnLabel(index)) };
}

function renderMatrices() {
  const model = selectedModel();
  const exists = Boolean(model);
  ui["matrix-content"].hidden = !exists;
  ui["no-matrix"].hidden = exists;
  ui["download-matrices"].hidden = !exists;
  if (matrixDownloadUrl) { URL.revokeObjectURL(matrixDownloadUrl); matrixDownloadUrl = null; }
  if (!exists) {
    ui["matrix-description"].textContent = "Persistence is the measurement-holding reference, with no training step.";
    activeMatrix = null;
    return;
  }
  ui["matrix-description"].textContent = model.id === "current_input"
    ? "12 current-output coordinates. A and c are fitted; B maps the known future request, C selects all outputs, and D is zero."
    : `${model.state_dimension} coordinates: 11 × 12 output values${model.id === "history_input" ? " followed by ten previous requests" : "; past executed effort remains in those outputs"}. A has learned output rows and exact delay shifts. ${model.id === "history_no_input" ? "B is zero because explicit request information is omitted." : "B both affects predicted outputs and inserts the latest request into the delay state."} C selects the first 12 coordinates; D is zero.`;
  matrixDownloadUrl = URL.createObjectURL(new Blob([JSON.stringify({ schema: "light_prediction_matrices_view_v1", source: record.source, body: { id: body.id, name: body.name, assembly_sha256: body.assembly_sha256 }, model }, null, 2)], { type: "application/json" }));
  ui["download-matrices"].href = matrixDownloadUrl;
  ui["download-matrices"].download = `prediction-${String(body.id).replace(/[^a-z0-9_-]/gi, "_")}-${model.id}-matrices.json`;
  renderMatrix();
}

function renderMatrix() {
  const model = selectedModel();
  if (!model) return;
  const name = ui.matrix.value;
  const values = name === "c" ? model.c.map((value) => [value]) : model[name];
  activeMatrix = { name, values, labels: matrixLabels(name, values) };
  cellRow = Math.min(cellRow, values.length - 1); cellColumn = Math.min(cellColumn, values[0].length - 1);
  fillSelect(ui.row, activeMatrix.labels.rows.map((label, index) => ({ value: index, label: `${index} · ${label}` })), cellRow);
  fillSelect(ui.column, activeMatrix.labels.columns.map((label, index) => ({ value: index, label: `${index} · ${label}` })), cellColumn);
  ui["matrix-dimension"].textContent = `${name} · ${values.length} rows × ${values[0].length} column${values[0].length === 1 ? "" : "s"}`;
  const limit = Math.max(...values.map((row) => Math.max(...row.map(Math.abs))));
  matrixImage = document.createElement("canvas"); matrixImage.width = 660; matrixImage.height = 660;
  const context = matrixImage.getContext("2d");
  const rows = values.length, columns = values[0].length;
  for (let row = 0; row < rows; row++) for (let columnIndex = 0; columnIndex < columns; columnIndex++) {
    const ratio = limit === 0 ? 0 : values[row][columnIndex] / limit;
    const neutral = [242, 232, 206], end = ratio < 0 ? [186, 109, 74] : [55, 120, 88];
    const amount = Math.sqrt(Math.abs(ratio));
    context.fillStyle = `rgb(${neutral.map((value, index) => Math.round(value + (end[index] - value) * amount)).join(",")})`;
    context.fillRect(Math.floor(columnIndex * 660 / columns), Math.floor(row * 660 / rows), Math.ceil(660 / columns), Math.ceil(660 / rows));
  }
  ui["matrix-canvas"].setAttribute("aria-label", `${name}, ${rows} by ${columns}. Color uses a square-root magnitude scale from ${compact(-limit)} through zero to ${compact(limit)}. Use arrow keys or row and column menus to inspect exact coefficients.`);
  showCell();
}

function showCell() {
  if (!activeMatrix) return;
  const { values, labels, name } = activeMatrix;
  ui.row.value = String(cellRow); ui.column.value = String(cellColumn);
  const value = values[cellRow][cellColumn];
  ui["matrix-entry"].textContent = `${name}[${cellRow}${name === "c" ? "" : `, ${cellColumn}`}] = ${value.toPrecision(9)}`;
  ui["entry-description"].textContent = `${labels.rows[cellRow]} ← ${labels.columns[cellColumn]}. Values refer to normalized coordinates.`;
  const firstRow = Math.max(0, Math.min(cellRow - 2, values.length - 5));
  const firstColumn = Math.max(0, Math.min(cellColumn - 2, values[0].length - 5));
  const rows = Array.from({ length: Math.min(5, values.length) }, (_, index) => firstRow + index);
  const columns = Array.from({ length: Math.min(5, values[0].length) }, (_, index) => firstColumn + index);
  const heading = el("tr"); heading.append(el("th", "row / col"), ...columns.map((index) => { const header = el("th", index); header.scope = "col"; return header; }));
  ui["matrix-head"].replaceChildren(heading);
  ui["matrix-rows"].replaceChildren(...rows.map((row) => {
    const tr = el("tr"); const header = el("th", row); header.scope = "row"; tr.append(header);
    tr.append(...columns.map((columnIndex) => { const td = el("td", compact(values[row][columnIndex])); td.dataset.selected = String(row === cellRow && columnIndex === cellColumn); td.title = `${name}[${row}, ${columnIndex}] = ${values[row][columnIndex]}`; return td; }));
    return tr;
  }));
  const canvas = ui["matrix-canvas"], context = canvas.getContext("2d");
  context.clearRect(0, 0, 660, 660); context.drawImage(matrixImage, 0, 0);
  const left = cellColumn * 660 / values[0].length, top = cellRow * 660 / values.length;
  const width = 660 / values[0].length, height = 660 / values.length;
  context.strokeStyle = "#102d24"; context.lineWidth = 2.5;
  context.strokeRect(Math.max(1.5, left + .5), Math.max(1.5, top + .5), Math.min(width - 1, 657), Math.min(height - 1, 657));
}

function renderEvidence() {
  ui["check-title"].textContent = `${record.checks.trials} native trials · ${record.checks.replays} exact replays`;
  ui["check-summary"].textContent = `Models sealed before withheld evaluation. Independent evidence checks completed. ${record.source.source_count} source and provenance files recorded.`;
  ui["report-path"].textContent = record.source.report_path;
  ui["report-hash"].textContent = record.source.report_sha256;
  ui["protocol-hash"].textContent = record.source.protocol_sha256;
  ui.checks.replaceChildren(...Object.entries(record.checks).filter(([, value]) => typeof value === "boolean" || typeof value === "number").map(([key, value]) => {
    const outcome = key.endsWith("readiness_passed");
    const label = typeof value === "boolean" ? outcome ? value ? "passed" : "not passed" : value ? "verified" : "not verified" : value;
    return el("li", `${key.replaceAll("_", " ")}: ${label}`);
  }));
}

function announce(message) { ui.live.textContent = message; }

async function load() {
  if (loading) return;
  loading = true; ui.loading.hidden = false; ui.error.hidden = true; ui.results.hidden = true;
  try {
    const response = await fetch("/prediction-data.json", { cache: "no-store", credentials: "same-origin" });
    if (!response.ok) throw new Error(`The experiment record could not be read (HTTP ${response.status}).`);
    record = validate(await response.json());
    fillSelect(ui.body, record.bodies.map((item, index) => ({ value: item.id, label: `${index + 1} · ${item.name}` })));
    fillSelect(ui.channel, CHANNELS.map((item, index) => ({ value: index, label: `${item.name} · ${item.unit}` })), channel);
    body = record.bodies[0];
    ui.results.hidden = false;
    renderEvidence(); renderBody();
    announce("Validated prediction evidence loaded. Choose a body, predictor, withheld trial, forecast origin and measured channel.");
  } catch (error) {
    record = null; body = null; forecast = null;
    ui["error-detail"].textContent = `${error.message} No provisional, missing or nonfinite result is replaced with an invented curve.`;
    ui.error.hidden = false;
    announce("The prediction evidence could not be loaded.");
  } finally { loading = false; ui.loading.hidden = true; }
}

ui.retry.addEventListener("click", load);
ui.body.addEventListener("change", () => { body = record.bodies.find((item) => String(item.id) === ui.body.value); renderBody(); announce(`Selected ${body.name}.`); });
ui.trial.addEventListener("change", renderOrigins);
ui.origin.addEventListener("change", selectForecast);
ui.channel.addEventListener("change", () => { channel = Number(ui.channel.value); renderMetrics(); drawForecast(); });
ui.matrix.addEventListener("change", () => { cellRow = 0; cellColumn = 0; renderMatrix(); });
ui.row.addEventListener("change", () => { cellRow = Number(ui.row.value); showCell(); });
ui.column.addEventListener("change", () => { cellColumn = Number(ui.column.value); showCell(); });
ui["matrix-canvas"].addEventListener("pointermove", (event) => {
  if (!activeMatrix || event.pointerType === "touch") return;
  const bounds = ui["matrix-canvas"].getBoundingClientRect();
  const nextRow = Math.max(0, Math.min(activeMatrix.values.length - 1, Math.floor((event.clientY - bounds.top) / bounds.height * activeMatrix.values.length)));
  const nextColumn = Math.max(0, Math.min(activeMatrix.values[0].length - 1, Math.floor((event.clientX - bounds.left) / bounds.width * activeMatrix.values[0].length)));
  if (nextRow !== cellRow || nextColumn !== cellColumn) { cellRow = nextRow; cellColumn = nextColumn; showCell(); }
});
ui["matrix-canvas"].addEventListener("keydown", (event) => {
  if (!activeMatrix || !["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  if (event.key === "ArrowDown") cellRow = Math.min(activeMatrix.values.length - 1, cellRow + 1);
  if (event.key === "ArrowUp") cellRow = Math.max(0, cellRow - 1);
  if (event.key === "ArrowRight") cellColumn = Math.min(activeMatrix.values[0].length - 1, cellColumn + 1);
  if (event.key === "ArrowLeft") cellColumn = Math.max(0, cellColumn - 1);
  showCell();
});
window.addEventListener("resize", () => { if (resizeFrame !== null) cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(() => { resizeFrame = null; drawForecast(); }); });
window.addEventListener("pagehide", () => { if (matrixDownloadUrl) URL.revokeObjectURL(matrixDownloadUrl); });
load();
