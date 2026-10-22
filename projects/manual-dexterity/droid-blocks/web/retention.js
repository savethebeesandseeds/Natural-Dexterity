"use strict";

const $ = (selector) => document.querySelector(selector);
const ui = {
  loading: $("[data-loading]"), error: $("[data-error]"), errorDetail: $("[data-error-detail]"),
  retry: $("[data-retry]"), results: $("[data-results]"), body: $("[data-body]"), seed: $("[data-seed]"),
  index: $("[data-case-index]"), title: $("[data-case-title]"), chart: $("[data-chart]"),
  chartNote: $("[data-chart-note]"), ranges: [...document.querySelectorAll("[data-range]")],
  recalledReward: $("[data-recalled-reward]"), currentReward: $("[data-current-reward]"),
  delta: $("[data-reward-delta]"), allCases: $("[data-all-case-note]"), measures: $("[data-measures]"),
  fullDelta: $("[data-full-return-difference]"), validationTitle: $("[data-validation-title]"),
  validationDetail: $("[data-validation-detail]"), checks: $("[data-checks]"),
  sourcePath: $("[data-source-path]"), sourceHash: $("[data-source-hash]"),
  assemblyHash: $("[data-assembly-hash]"), caseId: $("[data-case-id]"), live: $("[data-live]"),
};
const SVG_NS = "http://www.w3.org/2000/svg";
const BRANCHES = [
  { key: "continued", label: "Keep learning", color: "#8bc3a0", dash: null },
  { key: "current_frozen", label: "Current values", color: "#e4bc68", dash: "7 5" },
  { key: "recalled_frozen", label: "First-visit values", color: "#a9bbef", dash: "1 5" },
];
const MEASURES = [
  { key: "early_reward", label: "Early sensor reward", window: "120–140 s", digits: 3, unit: "" },
  { key: "return_reward", label: "Full-return sensor reward", window: "120–180 s", digits: 3, unit: "" },
  { key: "early_lux", label: "Early received light", window: "Mean · 120–140 s", digits: 1, unit: " lx" },
  { key: "return_lux", label: "Full-return received light", window: "Mean · 120–180 s", digits: 1, unit: " lx" },
  { key: "late_lux", label: "Late received light", window: "Mean · 160–180 s", digits: 1, unit: " lx" },
  { key: "return_energy_j", label: "Full-return electrical energy", window: "120–180 s", digits: 2, unit: " J" },
  { key: "late_energy_j", label: "Late electrical energy", window: "160–180 s", digits: 2, unit: " J" },
];
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const fmt = (value, digits = 3) => new Intl.NumberFormat("en", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
const signed = (value, digits = 3) => {
  const rounded = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value;
  return (rounded > 0 ? "+" : rounded < 0 ? "−" : "") + fmt(Math.abs(rounded), digits);
};
const sameNumber = (first, second) => Math.abs(first - second) <= 1e-8 * Math.max(1, Math.abs(second));
let data = null;
let selected = null;
let range = "all";
let loading = false;
let announcement = null;

function assert(condition, message) { if (!condition) throw new Error(message); }

function validate(record) {
  assert(record && record.schema === "light_retention_view_v1", "This record uses an unsupported viewer format.");
  assert(record.validation?.complete === true && record.validation?.independently_checked === true && record.validation?.all_case_checks_passed === true, "The native report has not completed all required checks.");
  assert(record.validation.case_count === 6 && record.validation.body_count === 2 && record.validation.seeds_per_body === 3 && record.validation.exact_replay_count === 18, "The record does not contain the complete declared comparison.");
  assert(record.protocol?.duration_s === 180 && record.protocol?.first_move_s === 60 && record.protocol?.return_s === 120, "The record does not use the declared three-minute return protocol.");
  assert(typeof record.source?.report_path === "string" && record.source.report_path.length > 0 && /^[a-f0-9]{64}$/i.test(record.source?.report_sha256 ?? ""), "The source report fingerprint is missing.");
  assert(Array.isArray(record.cases) && record.cases.length === 6, "The six comparisons are missing.");
  const ids = new Set();
  const bodies = new Map();
  for (const item of record.cases) {
    assert(typeof item.id === "string" && item.id.length > 0 && !ids.has(item.id), "A case ID is missing or repeated.");
    ids.add(item.id);
    assert(Number.isInteger(item.body_index) && item.body_index >= 0 && typeof item.body_name === "string" && typeof item.seed === "string", "A body or seed label is invalid.");
    assert(/^[a-f0-9]{64}$/i.test(item.assembly_sha256 ?? ""), "A body fingerprint is missing.");
    assert(item.exact_replays === true && item.checks && Object.keys(item.checks).length > 0 && Object.values(item.checks).every((value) => value === true), "A case has incomplete native checks.");
    const matching = bodies.get(item.body_index) ?? [];
    assert(!matching.some((other) => other.seed === item.seed), "A body and seed combination is repeated.");
    matching.push(item); bodies.set(item.body_index, matching);
    for (const { key } of BRANCHES) {
      const branch = item[key];
      assert(branch && MEASURES.every((measurement) => finite(branch[measurement.key])), "A completed measurement is missing.");
      assert([branch.early_lux, branch.return_lux, branch.late_lux, branch.return_energy_j, branch.late_energy_j].every((value) => value >= 0), "A received-light or energy measurement is invalid.");
      assert(Array.isArray(branch.curve) && branch.curve.length > 1, "A measured light curve is missing.");
      let previous = -Infinity;
      for (const point of branch.curve) {
        assert(Array.isArray(point) && point.length === 2 && finite(point[0]) && finite(point[1]) && point[0] >= 0 && point[0] <= 180 && point[0] > previous && point[1] >= 0, "A measured light curve is invalid.");
        previous = point[0];
      }
      assert(branch.curve[0][0] <= 1 && branch.curve.at(-1)[0] >= 179, "A curve does not cover the full return experiment.");
    }
    assert(finite(item.early_reward_delta) && finite(item.return_reward_delta), "A paired reward difference is missing.");
    assert(sameNumber(item.early_reward_delta, item.recalled_frozen.early_reward - item.current_frozen.early_reward) && sameNumber(item.return_reward_delta, item.recalled_frozen.return_reward - item.current_frozen.return_reward), "A reward difference disagrees with its branch measurements.");
  }
  assert(bodies.size === 2, "The record must contain two development bodies.");
  const seedSets = [];
  for (const cases of bodies.values()) {
    assert(cases.length === 3 && new Set(cases.map((item) => item.assembly_sha256)).size === 1, "Each body must have three seeds and a matching assembly.");
    seedSets.push(cases.map((item) => item.seed).sort().join("|"));
  }
  assert(seedSets[0] === seedSets[1], "The two bodies must use the same three seeds.");
  return record;
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = String(value); node.textContent = label;
  return node;
}

function fillSeeds(preferred) {
  const cases = data.cases.filter((item) => String(item.body_index) === ui.body.value);
  ui.seed.replaceChildren(...cases.map((item) => option(item.seed, "Seed " + item.seed)));
  if (cases.some((item) => item.seed === preferred)) ui.seed.value = preferred;
}

function announce(message) {
  window.clearTimeout(announcement); ui.live.textContent = "";
  announcement = window.setTimeout(() => { ui.live.textContent = message; }, 20);
}

function choose(announceChange = true) {
  selected = data.cases.find((item) => String(item.body_index) === ui.body.value && item.seed === ui.seed.value);
  if (!selected) return;
  ui.title.textContent = selected.body_name + " · seed " + selected.seed;
  ui.index.textContent = "Case " + (data.cases.indexOf(selected) + 1) + " of " + data.cases.length;
  ui.recalledReward.textContent = fmt(selected.recalled_frozen.early_reward);
  ui.currentReward.textContent = fmt(selected.current_frozen.early_reward);
  ui.delta.textContent = signed(selected.early_reward_delta);
  ui.delta.parentElement.dataset.direction = selected.early_reward_delta < -0.0005 ? "negative" : selected.early_reward_delta > 0.0005 ? "positive" : "equal";
  const bodyMedians = [...new Set(data.cases.map((item) => item.body_index))].map((bodyIndex) => {
    const cases = data.cases.filter((item) => item.body_index === bodyIndex);
    const differences = cases.map((item) => item.early_reward_delta).sort((first, second) => first - second);
    return cases[0].body_name + ": " + signed(differences[Math.floor(differences.length / 2)]);
  });
  const positive = data.cases.filter((item) => item.early_reward_delta > 0).length;
  const negative = data.cases.filter((item) => item.early_reward_delta < 0).length;
  const equal = data.cases.length - positive - negative;
  ui.allCases.textContent = "Predeclared summary · median paired early reward difference within each body (three seeds): " + bodyMedians.join("; ") + ". Across all six cases: " + positive + " positive, " + negative + " negative" + (equal ? ", " + equal + " equal" : "") + ". First-visit minus current values throughout.";
  ui.measures.replaceChildren(...MEASURES.map((measurement) => {
    const row = document.createElement("tr");
    const heading = document.createElement("th"); heading.scope = "row"; heading.textContent = measurement.label;
    const windowLabel = document.createElement("small"); windowLabel.textContent = measurement.window; heading.append(windowLabel); row.append(heading);
    for (const { key } of BRANCHES) {
      const cell = document.createElement("td"); cell.textContent = fmt(selected[key][measurement.key], measurement.digits) + measurement.unit; row.append(cell);
    }
    return row;
  }));
  ui.fullDelta.textContent = "Across the full return (120–180 s), first-visit minus current values earned " + signed(selected.return_reward_delta) + " sensor reward.";
  ui.validationTitle.textContent = "All six cases passed independent evidence validation";
  ui.validationDetail.textContent = "Two bodies · three seeds each · 18 exact full-transition replays checked.";
  ui.checks.replaceChildren(...Object.entries(selected.checks).map(([key]) => {
    const item = document.createElement("li"); item.textContent = "Passed: " + key.replace(/_/g, " "); return item;
  }));
  const replay = document.createElement("li");
  replay.textContent = "Passed: exact controller, observation and full-transition replay for all three paths.";
  ui.checks.append(replay);
  ui.sourcePath.textContent = data.source.report_path; ui.sourceHash.textContent = data.source.report_sha256;
  ui.assemblyHash.textContent = selected.assembly_sha256; ui.caseId.textContent = selected.id;
  drawChart();
  if (announceChange) announce("Showing " + selected.body_name + ", seed " + selected.seed + ". Early reward after the return: first-visit values " + fmt(selected.recalled_frozen.early_reward) + "; current values " + fmt(selected.current_frozen.early_reward) + ". First-visit minus current: " + signed(selected.early_reward_delta) + ".");
}

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function ceiling(value) {
  const power = 10 ** Math.floor(Math.log10(Math.max(value, 1)));
  return Math.ceil(value / power / 0.5) * power * 0.5;
}

function drawChart() {
  if (!selected) return;
  const width = Math.max(300, ui.chart.clientWidth || 1000);
  const height = Math.max(240, ui.chart.clientHeight || 350);
  const left = width < 500 ? 44 : 62;
  const right = width - 23;
  const top = 41;
  const bottom = height - 42;
  const start = range === "return" ? 120 : 0;
  const curves = BRANCHES.map((branch) => ({ ...branch, points: selected[branch.key].curve.filter((point) => point[0] >= start) }));
  const high = curves.reduce((largest, branch) => branch.points.reduce((value, point) => Math.max(value, point[1]), largest), 1);
  const maxLux = ceiling(high * 1.06);
  const x = (seconds) => left + (seconds - start) / (180 - start) * (right - left);
  const y = (lux) => bottom - lux / maxLux * (bottom - top);
  const forcedColors = window.matchMedia("(forced-colors: active)").matches;
  const textColor = forcedColors ? "CanvasText" : "#a3b196";
  const gridColor = forcedColors ? "CanvasText" : "#70826b";
  ui.chart.setAttribute("viewBox", "0 0 " + width + " " + height);
  const description = "Solid green: keep learning. Dashed amber: current values. Dotted blue: first-visit values. All share the first 120 seconds. The sun moves to B at 60 seconds and returns to A at 120 seconds. The plot shows " + start + " to 180 seconds and 0 to " + fmt(maxLux, 0) + " lux. The primary reward window is 120 to 140 seconds. Mean light in the last 20 seconds: " + BRANCHES.map((branch) => branch.label.toLowerCase() + " " + fmt(selected[branch.key].late_lux, 1) + " lux").join("; ") + ".";
  const nodes = [
    svg("title", { id: "chart-title" }, selected.body_name + ", seed " + selected.seed + ": received light on the return"),
    svg("desc", { id: "chart-description" }, description),
    svg("rect", { x: x(120), y: top, width: x(140) - x(120), height: bottom - top, fill: forcedColors ? "Highlight" : "#a9bbef", "fill-opacity": 0.1 }),
  ];
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const value = maxLux * fraction;
    nodes.push(svg("line", { x1: left, x2: right, y1: y(value), y2: y(value), stroke: gridColor, "stroke-opacity": forcedColors ? 0.6 : 0.24, "stroke-width": 1 }));
    nodes.push(svg("text", { x: left - 9, y: y(value) + 3, fill: textColor, "font-family": "Consolas, monospace", "font-size": 9, "text-anchor": "end" }, fmt(value, value < 10 && value % 1 !== 0 ? 1 : 0)));
  }
  const ticks = range === "return" ? [120, 140, 160, 180] : width < 500 ? [0, 60, 120, 180] : [0, 30, 60, 90, 120, 150, 180];
  for (const seconds of ticks) {
    nodes.push(svg("line", { x1: x(seconds), x2: x(seconds), y1: bottom, y2: bottom + 5, stroke: textColor, "stroke-width": 1 }));
    nodes.push(svg("text", { x: x(seconds), y: bottom + 21, fill: textColor, "font-family": "Consolas, monospace", "font-size": 10, "text-anchor": "middle" }, String(seconds) + " s"));
  }
  nodes.push(svg("text", { x: left - 9, y: top - 15, fill: textColor, "font-family": "Consolas, monospace", "font-size": 10, "text-anchor": "end" }, "lx"));
  for (const branch of curves) {
    nodes.push(svg("polyline", {
      points: branch.points.map((point) => x(point[0]) + "," + y(point[1])).join(" "),
      fill: "none", stroke: forcedColors ? "CanvasText" : branch.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(branch.dash ? { "stroke-dasharray": branch.dash } : {}),
    }));
  }
  for (const [time, label] of [[60, "Sun to B"], [120, range === "return" ? "Sun returns to A" : "Sun returns"]]) {
    if (time < start) continue;
    const markerColor = forcedColors ? "CanvasText" : "#ddca91";
    nodes.push(svg("line", { x1: x(time), x2: x(time), y1: top - 6, y2: bottom, stroke: markerColor, "stroke-width": 1, "stroke-dasharray": "3 5" }));
    nodes.push(svg("circle", { cx: x(time), cy: top - 8, r: 3, fill: markerColor }));
    nodes.push(svg("text", { x: x(time) + (range === "return" ? 8 : 0), y: top - 18, fill: markerColor, "font-size": width < 500 ? 10 : 11, "text-anchor": range === "return" ? "start" : "middle" }, label));
  }
  ui.chart.replaceChildren(...nodes);
  ui.chartNote.textContent = range === "return"
    ? "A closer look at the return. The shaded first 20 seconds are the primary comparison; the brightness scale fits this view."
    : "The sun moves at 60 seconds and returns at 120. All three paths share the first two minutes. Shading marks the first 20 seconds back.";
}

async function load() {
  if (loading) return;
  loading = true; ui.loading.hidden = false; ui.error.hidden = true; ui.results.hidden = true; ui.retry.disabled = true;
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), 8000);
  try {
    const response = await fetch("/retention-data.json", { cache: "no-store", signal: abort.signal });
    if (!response.ok) throw new Error(response.status === 404 ? "The completed native report has not been exported to this page yet." : "The record could not be read (HTTP " + response.status + ").");
    data = validate(await response.json());
    const bodies = [...new Map(data.cases.map((item) => [item.body_index, item])).values()];
    ui.body.replaceChildren(...bodies.map((item) => option(item.body_index, (item.body_index + 1) + " · " + item.body_name)));
    fillSeeds(); ui.results.hidden = false; choose(false);
  } catch (error) {
    data = null; selected = null; ui.error.hidden = false;
    ui.errorDetail.textContent = (error.name === "AbortError" ? "Reading the record timed out." : error.message) + " No provisional result is shown.";
  } finally {
    window.clearTimeout(timeout); loading = false; ui.loading.hidden = true; ui.retry.disabled = false;
  }
}

ui.body.addEventListener("change", () => { fillSeeds(ui.seed.value); choose(); });
ui.seed.addEventListener("change", () => choose());
ui.retry.addEventListener("click", load);
for (const button of ui.ranges) {
  button.addEventListener("click", () => {
    range = button.dataset.range;
    for (const other of ui.ranges) other.setAttribute("aria-pressed", String(other === button));
    drawChart(); announce(range === "return" ? "Chart shows the return, from 120 to 180 seconds. The brightness scale fits this view." : "Chart shows the whole journey, from 0 to 180 seconds.");
  });
}
new ResizeObserver(drawChart).observe(ui.chart.parentElement);
window.matchMedia("(forced-colors: active)").addEventListener("change", drawChart);
load();

