"use strict";

const $ = (selector) => document.querySelector(selector);
const ui = {
  loading: $("[data-loading]"), error: $("[data-error]"), errorDetail: $("[data-error-detail]"),
  retry: $("[data-retry]"), results: $("[data-results]"), body: $("[data-body]"), seed: $("[data-seed]"),
  index: $("[data-pair-index]"), title: $("[data-pair-title]"), chart: $("[data-chart]"),
  continuedReward: $("[data-continued-reward]"), frozenReward: $("[data-frozen-reward]"), delta: $("[data-reward-delta]"),
  continuedEnergy: $("[data-continued-energy]"), frozenEnergy: $("[data-frozen-energy]"),
  continuedEarly: $("[data-continued-early]"), frozenEarly: $("[data-frozen-early]"),
  continuedLate: $("[data-continued-late]"), frozenLate: $("[data-frozen-late]"),
  validationTitle: $("[data-validation-title]"), validationDetail: $("[data-validation-detail]"),
  checks: $("[data-checks]"), sourcePath: $("[data-source-path]"), sourceHash: $("[data-source-hash]"),
  assemblyHash: $("[data-assembly-hash]"), pairId: $("[data-pair-id]"), live: $("[data-live]"),
};
const SVG_NS = "http://www.w3.org/2000/svg";
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const fmt = (value, digits = 3) => new Intl.NumberFormat("en", { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
let data = null;
let selected = null;
let loading = false;
let announcement = null;

function assert(condition, text) { if (!condition) throw new Error(text); }
function validate(record) {
  assert(record && record.schema === "light_adaptation_view_v1", "This record uses an unsupported viewer format.");
  assert(record.validation?.complete === true && record.validation?.independently_checked === true && record.validation?.all_pair_checks_passed === true, "The native report has not completed all required checks.");
  assert(record.validation.pair_count === 6 && record.validation.body_count === 2 && record.validation.seeds_per_body === 3 && record.validation.exact_replay_count === 12, "The record does not contain the complete declared comparison.");
  assert(record.protocol?.duration_s === 120 && record.protocol?.intervention_s === 60, "The record does not use the declared two-minute comparison.");
  assert(typeof record.source?.report_path === "string" && /^[a-f0-9]{64}$/i.test(record.source?.report_sha256 ?? ""), "The source report fingerprint is missing.");
  assert(Array.isArray(record.pairs) && record.pairs.length === 6, "The six paired comparisons are missing.");
  const ids = new Set();
  for (const pair of record.pairs) {
    assert(typeof pair.id === "string" && !ids.has(pair.id), "A pair ID is missing or repeated.");
    ids.add(pair.id);
    assert(Number.isInteger(pair.body_index) && typeof pair.body_name === "string" && typeof pair.seed === "string", "A body or seed label is invalid.");
    assert(pair.exact_replays === true && pair.checks && Object.keys(pair.checks).length > 0 && Object.values(pair.checks).every((value) => value === true), "A pair has incomplete native checks.");
    assert(finite(pair.reward_delta), "A paired reward difference is missing.");
    for (const branch of [pair.continued, pair.frozen]) {
      assert(branch && ["reward_after_move", "energy_after_move_j", "early_after_move_lux", "late_after_move_lux"].every((key) => finite(branch[key])), "A completed measurement is missing.");
      assert(Array.isArray(branch.curve) && branch.curve.length > 1, "A measured light curve is missing.");
      let previous = -Infinity;
      for (const point of branch.curve) {
        assert(Array.isArray(point) && point.length === 2 && finite(point[0]) && finite(point[1]) && point[0] >= 0 && point[0] <= 120 && point[0] > previous && point[1] >= 0, "A measured light curve is invalid.");
        previous = point[0];
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

function fillSeeds(preferred) {
  const pairs = data.pairs.filter((pair) => String(pair.body_index) === ui.body.value);
  ui.seed.replaceChildren(...pairs.map((pair) => option(pair.seed, "Seed " + pair.seed)));
  if (pairs.some((pair) => pair.seed === preferred)) ui.seed.value = preferred;
}

function choose(announceChange = true) {
  selected = data.pairs.find((pair) => String(pair.body_index) === ui.body.value && pair.seed === ui.seed.value);
  if (!selected) return;
  ui.title.textContent = selected.body_name + " · seed " + selected.seed;
  ui.index.textContent = "Pair " + (data.pairs.indexOf(selected) + 1) + " of " + data.pairs.length;
  ui.continuedReward.textContent = fmt(selected.continued.reward_after_move);
  ui.frozenReward.textContent = fmt(selected.frozen.reward_after_move);
  const displayedDelta = Math.abs(selected.reward_delta) < 0.0005 ? 0 : selected.reward_delta;
  ui.delta.textContent = (displayedDelta > 0 ? "+" : displayedDelta < 0 ? "−" : "") + fmt(Math.abs(displayedDelta));
  ui.delta.parentElement.dataset.direction = displayedDelta < 0 ? "negative" : displayedDelta > 0 ? "positive" : "equal";
  ui.continuedEnergy.textContent = fmt(selected.continued.energy_after_move_j, 2) + " J";
  ui.frozenEnergy.textContent = fmt(selected.frozen.energy_after_move_j, 2) + " J";
  ui.continuedEarly.textContent = fmt(selected.continued.early_after_move_lux, 1) + " lx";
  ui.frozenEarly.textContent = fmt(selected.frozen.early_after_move_lux, 1) + " lx";
  ui.continuedLate.textContent = fmt(selected.continued.late_after_move_lux, 1) + " lx";
  ui.frozenLate.textContent = fmt(selected.frozen.late_after_move_lux, 1) + " lx";
  ui.validationTitle.textContent = "All six pairs passed the native report’s checks";
  ui.validationDetail.textContent = "Two bodies · three seeds each · 12 exact controller and observation replays recorded.";
  ui.checks.replaceChildren(...Object.entries(selected.checks).map(([key, passed]) => {
    const item = document.createElement("li");
    item.textContent = (passed ? "Passed: " : "Failed: ") + key.replace(/_/g, " ");
    return item;
  }));
  const replay = document.createElement("li");
  replay.textContent = "Passed: exact controller and observation replay for both paths.";
  ui.checks.append(replay);
  ui.sourcePath.textContent = data.source.report_path;
  ui.sourceHash.textContent = data.source.report_sha256;
  ui.assemblyHash.textContent = selected.assembly_sha256;
  ui.pairId.textContent = selected.id;
  drawChart();
  if (announceChange) {
    window.clearTimeout(announcement); ui.live.textContent = "";
    announcement = window.setTimeout(() => {
      ui.live.textContent = "Showing " + selected.body_name + ", seed " + selected.seed + ". Reward after the move: keep learning " + fmt(selected.continued.reward_after_move) + "; use learned values " + fmt(selected.frozen.reward_after_move) + ".";
    }, 20);
  }
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
  const right = width - 19;
  const top = 36;
  const bottom = height - 42;
  const high = Math.max(1, ...selected.continued.curve.map((point) => point[1]), ...selected.frozen.curve.map((point) => point[1]));
  const maxLux = ceiling(high * 1.06);
  const x = (seconds) => left + seconds / 120 * (right - left);
  const y = (lux) => bottom - lux / maxLux * (bottom - top);
  ui.chart.setAttribute("viewBox", "0 0 " + width + " " + height);
  const nodes = [
    svg("title", { id: "chart-title" }, selected.body_name + ", seed " + selected.seed + ": received light"),
    svg("desc", { id: "chart-description" }, "Solid green: keep learning. Dashed amber: use learned values. Both share the first 60 seconds, then the sunshine moves. The plot spans 0 to 120 seconds and 0 to " + fmt(maxLux, 0) + " lux. Mean light in the last 20 seconds: keep learning " + fmt(selected.continued.late_after_move_lux, 1) + " lux; use learned values " + fmt(selected.frozen.late_after_move_lux, 1) + " lux."),
    svg("rect", { x: left, y: top, width: x(60) - left, height: bottom - top, fill: "#8299740b" }),
  ];
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const value = maxLux * fraction;
    nodes.push(svg("line", { x1: left, x2: right, y1: y(value), y2: y(value), stroke: "#70826b", "stroke-opacity": 0.24, "stroke-width": 1 }));
    nodes.push(svg("text", { x: left - 9, y: y(value) + 3, fill: "#9bab8e", "font-family": "Consolas, monospace", "font-size": 9, "text-anchor": "end" }, fmt(value, value < 10 && value % 1 !== 0 ? 1 : 0)));
  }
  for (const seconds of width < 500 ? [0, 60, 120] : [0, 30, 60, 90, 120]) {
    nodes.push(svg("line", { x1: x(seconds), x2: x(seconds), y1: bottom, y2: bottom + 5, stroke: "#a3ad8b", "stroke-width": 1 }));
    nodes.push(svg("text", { x: x(seconds), y: bottom + 21, fill: "#a3b196", "font-family": "Consolas, monospace", "font-size": 10, "text-anchor": "middle" }, String(seconds) + " s"));
  }
  nodes.push(svg("text", { x: left - 9, y: top - 13, fill: "#99aa8e", "font-family": "Consolas, monospace", "font-size": 10, "text-anchor": "end" }, "lx"));
  for (const [branch, color, dashed] of [[selected.continued, "#8bc3a0", false], [selected.frozen, "#e4bc68", true]]) {
    nodes.push(svg("polyline", {
      points: branch.curve.map((point) => x(point[0]) + "," + y(point[1])).join(" "),
      fill: "none", stroke: color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(dashed ? { "stroke-dasharray": "5 4" } : {}),
    }));
  }
  nodes.push(svg("line", { x1: x(60), x2: x(60), y1: top - 6, y2: bottom, stroke: "#d6c382", "stroke-width": 1, "stroke-dasharray": "3 5" }));
  nodes.push(svg("circle", { cx: x(60), cy: top - 8, r: 3, fill: "#dac688" }));
  nodes.push(svg("text", { x: x(60) + 9, y: top - 9, fill: "#ddca91", "font-size": 11 }, "Sun moves"));
  ui.chart.replaceChildren(...nodes);
}

async function load() {
  if (loading) return;
  loading = true; ui.loading.hidden = false; ui.error.hidden = true; ui.results.hidden = true; ui.retry.disabled = true;
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), 8000);
  try {
    const response = await fetch("/adaptation-data.json", { cache: "no-store", signal: abort.signal });
    if (!response.ok) throw new Error(response.status === 404 ? "The completed native report has not been exported to this page yet." : "The record could not be read (HTTP " + response.status + ").");
    data = validate(await response.json());
    const bodies = [...new Map(data.pairs.map((pair) => [pair.body_index, pair])).values()];
    ui.body.replaceChildren(...bodies.map((pair) => option(pair.body_index, (pair.body_index + 1) + " · " + pair.body_name)));
    fillSeeds();
    ui.results.hidden = false;
    choose(false);
  } catch (error) {
    data = null; selected = null;
    ui.error.hidden = false;
    ui.errorDetail.textContent = (error.name === "AbortError" ? "Reading the record timed out." : error.message) + " No provisional result is shown.";
  } finally {
    window.clearTimeout(timeout); loading = false; ui.loading.hidden = true; ui.retry.disabled = false;
  }
}

ui.body.addEventListener("change", () => { fillSeeds(ui.seed.value); choose(); });
ui.seed.addEventListener("change", () => choose());
ui.retry.addEventListener("click", load);
new ResizeObserver(drawChart).observe(ui.chart.parentElement);
load();

