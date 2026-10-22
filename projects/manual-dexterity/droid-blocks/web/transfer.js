"use strict";

// This page reads exported evidence only. It never connects to an engine session.
const $ = (selector) => document.querySelector(selector);
const SVG_NS = "http://www.w3.org/2000/svg";
const ARMS = ["frozen", "adapt", "scratch", "source_frozen"];
const LABELS = {
  frozen: "Earlier learning", adapt: "Keep learning", scratch: "From scratch",
  source_frozen: "Original motor position",
};
const COLORS = { frozen: "#a48240", adapt: "#527e5f", scratch: "#5c76a1", source_frozen: "#946eab" };
const DASHES = { frozen: "8 5", adapt: "", scratch: "2 5", source_frozen: "10 4 2 4" };
const ui = {
  status: $("[data-load-state]"), title: $("[data-load-title]"), detail: $("[data-load-detail]"), retry: $("[data-retry]"),
  studyStatus: $("[data-study-status]"),
  cases: $("[data-case]"), seeds: $("[data-seed]"), reference: $("[data-reference]"), description: $("[data-case-description]"),
  panels: $("[data-panels]"), playback: $(".shared-playback"), play: $("[data-play]"), restart: $("[data-restart]"), speed: $("[data-speed]"),
  time: $("[data-time]"), timeline: $("[data-timeline]"), duration: $("[data-duration-label]"), playbackNote: $("[data-playback-note]"),
  chart: $("[data-reward-chart]"), sourceLegend: $("[data-source-legend]"), delta: $("[data-delta]"), deltaNote: $("[data-delta-note]"),
  sourceBudget: $("[data-source-budget]"), targetBudget: $("[data-target-budget]"), scope: $("[data-scope]"),
  evidence: $("[data-evidence-note]"), measures: $("[data-measures]"), sampleNote: $("[data-sample-note]"),
  protocol: $("[data-protocol]"), live: $("[data-live]"),
  outcome: $("[data-outcome]"), verdict: $("[data-overall-verdict]"), conclusion: $("[data-overall-conclusion]"),
  seedMeans: $("[data-seed-means]"), criterion: $("[data-overall-criterion]"), integrity: $("[data-overall-integrity]"),
  controlSummary: $("[data-control-summary]"), usefulness: $("[data-usefulness]"),
};
const panels = Object.fromEntries(ARMS.map((arm) => {
  const node = $(`[data-panel="${arm}"]`);
  return [arm, {
    node, scene: node.querySelector("[data-motion]"), status: node.querySelector("[data-stage-status]"),
    reward: node.querySelector("[data-reward]"), light: node.querySelector("[data-light]"),
    note: node.querySelector("[data-panel-note]"), trial: null, trace: null, error: null, rendered: "",
  }];
}));
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const vector = (value, length = 3) => Array.isArray(value) && value.length === length && value.every(finite);
const number = (value, digits = 2) => finite(value) ? value.toLocaleString("en", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const signed = (value, digits = 2) => `${value > 0 ? "+" : ""}${number(value, digits)}`;
const visibleArms = () => ui.reference.checked ? ARMS : ARMS.slice(0, 3);
const traceCache = new Map();
let record = null;
let selectedCase = null;
let generation = 0;
let indexGeneration = 0;
let duration = 64;
let time = 0;
let playing = false;
let raf = 0;
let previousAnimationTime = null;
let chartMapping = null;
let indexLoading = false;

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function announce(message) { ui.live.textContent = message; }

function setStatus(state, title, detail) {
  ui.status.dataset.state = state;
  ui.title.textContent = title;
  ui.detail.textContent = detail;
}

function assetURL(path) {
  if (typeof path !== "string" || !path.trim()) throw new Error("No playback file was declared.");
  const url = new URL(path, window.location.href);
  const pathname = decodeURIComponent(url.pathname);
  if (!["http:", "https:"].includes(url.protocol) || url.origin !== window.location.origin ||
      url.username || url.password || url.search || url.hash || !pathname.endsWith(".json") ||
      /(^|\/)api(\/|$)/i.test(pathname) || pathname.includes("\\")) {
    throw new Error("The playback file must be a local JSON evidence asset.");
  }
  return url.href;
}

async function readJSON(path) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(assetURL(path), { method: "GET", cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(response.status === 404 ? "This evidence file is still preparing." : `The evidence file returned HTTP ${response.status}.`);
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The evidence file did not arrive in time. Read again to retry.");
    if (error instanceof SyntaxError) throw new Error("The evidence file is not a readable JSON record yet.");
    throw error;
  } finally { window.clearTimeout(timeout); }
}

function checkIndex(value) {
  if (value?.schema !== "actuation_transfer_view_v1" || !Array.isArray(value.cases) ||
      !Array.isArray(value.seeds) || !Array.isArray(value.trials) || !value.protocol) {
    throw new Error("The comparison index is not in the expected playback format.");
  }
  if (value.cases.some((item) => !item || typeof item.id !== "string" || !item.id) ||
      new Set(value.cases.map((item) => item.id)).size !== value.cases.length ||
      value.seeds.some((seed) => !Number.isSafeInteger(seed)) || new Set(value.seeds).size !== value.seeds.length) {
    throw new Error("The comparison index has ambiguous case or seed labels.");
  }
  const ids = new Set();
  const matches = new Set();
  for (const trial of value.trials) {
    if (!trial || typeof trial.id !== "string" || !trial.id || ids.has(trial.id)) throw new Error("The index has an invalid or repeated trial ID.");
    ids.add(trial.id);
    if (!ARMS.includes(trial.arm)) continue;
    const key = JSON.stringify([trial.case_id, trial.source_seed, trial.arm]);
    if (matches.has(key)) throw new Error("The index repeats a comparison arm for the same case and seed.");
    matches.add(key);
  }
  return value;
}

function checkTrace(value, trial) {
  if (value?.schema !== "actuation_transfer_playback_v1" || value.id !== trial.id || !Array.isArray(value.frames)) {
    throw new Error("This playback file does not match its declared trial.");
  }
  const samples = [];
  if (value.initial_physics && typeof value.initial_physics === "object") {
    samples.push({ time_s: 0, physics: value.initial_physics, cumulative_reward: value.initial_physics.reward?.cumulative ?? 0 });
  }
  let previous = -1;
  for (const frame of value.frames) {
    if (!frame || !finite(frame.time_s) || frame.time_s < 0 || frame.time_s <= previous ||
        !frame.physics || typeof frame.physics !== "object" ||
        (frame.cumulative_reward !== null && frame.cumulative_reward !== undefined && !finite(frame.cumulative_reward))) {
      throw new Error("The saved frames have an invalid time, state or reward.");
    }
    previous = frame.time_s;
    if (frame.time_s > duration + 1e-6) throw new Error("A saved frame extends beyond the declared comparison time.");
    if (frame.time_s === 0 && samples.length) samples[0] = frame;
    else samples.push(frame);
  }
  if (!samples.length) throw new Error("This trial has no saved physical snapshots yet.");
  const last = samples[samples.length - 1].time_s;
  if (finite(trial.metrics?.duration_s) && Math.abs(last - trial.metrics.duration_s) > 1e-5) {
    throw new Error("The last physical snapshot and recorded trial duration disagree.");
  }
  return { samples, last, value };
}

async function traceFor(trial) {
  const url = assetURL(trial.trace_url);
  let pending = traceCache.get(url);
  if (!pending) {
    pending = readJSON(url);
    traceCache.set(url, pending);
    pending.catch(() => { if (traceCache.get(url) === pending) traceCache.delete(url); });
    // Retain a few recently requested comparisons, not the entire experiment.
    while (traceCache.size > 8) traceCache.delete(traceCache.keys().next().value);
  } else {
    traceCache.delete(url);
    traceCache.set(url, pending);
  }
  return checkTrace(await pending, trial);
}

function option(value, label) {
  const node = document.createElement("option");
  node.value = String(value); node.textContent = label;
  return node;
}

function describeOutcome() {
  const summary = record.summary ?? {};
  const passed = summary.transfer_gate_passed;
  ui.outcome.dataset.result = passed === true ? "passed" : passed === false ? "failed" : "unreported";
  ui.verdict.textContent = passed === true ? "Declared criterion met" : passed === false ? "Declared criterion not met" : "Awaiting the audited result";
  ui.conclusion.textContent = typeof summary.conclusion === "string" ? summary.conclusion : "The overall conclusion has not been reported yet.";
  const means = Array.isArray(summary.seed_means) ? summary.seed_means : [];
  // Keep the declared seed order; changing the selected replay never changes this result.
  ui.seedMeans.replaceChildren(...record.seeds.map((seed) => {
    const row = means.find((item) => item?.source_seed === seed);
    const value = row?.mean_adapt_minus_scratch;
    const entry = document.createElement("div");
    entry.dataset.sign = !finite(value) ? "unknown" : value > 0 ? "positive" : value < 0 ? "negative" : "zero";
    const term = document.createElement("dt"); term.textContent = `Learning seed ${seed}`;
    const measurement = document.createElement("dd");
    const amount = document.createElement("strong"); amount.textContent = finite(value) ? signed(value, 3) : "Not reported";
    const scope = document.createElement("small");
    scope.textContent = finite(row?.cases) ? `mean across ${number(row.cases, 0)} cases${row.all_target_triples_complete_safe === false ? " · incomplete or unsafe comparison" : ""}` : "case average not available";
    measurement.append(amount, scope); entry.append(term, measurement);
    return entry;
  }));
  const median = summary.median_seed_mean_adapt_minus_scratch;
  const positive = summary.positive_seed_means;
  const required = summary.required_positive_seed_means;
  ui.criterion.textContent = `Median seed average: ${finite(median) ? signed(median, 3) : "not reported"}. ${finite(positive) ? `${number(positive, 0)} of ${record.seeds.length} seed averages are positive.` : "The positive-seed count is not reported."} ${finite(required) ? `The declared criterion requires a positive median and at least ${number(required, 0)} positive seed averages, together with the evidence and safety checks.` : "See the declared protocol for the overall criterion."}`;
  const checks = [];
  if (typeof summary.integrity_passed === "boolean") checks.push(summary.integrity_passed ? "Evidence checks passed" : "Evidence checks did not pass");
  if (typeof summary.target_arms_complete_safe === "boolean") checks.push(summary.target_arms_complete_safe ? "target runs completed with their safety checks satisfied" : "some target runs were incomplete or did not satisfy safety checks");
  ui.integrity.textContent = checks.length ? `${checks.join("; ")}. Reproducible evidence and a control benefit are separate questions.` : "Evidence and safety check status has not been reported.";
  const pairs = Array.isArray(summary.pairs) ? summary.pairs : [];
  if (pairs.length) {
    const comparedWithQuiet = (key, introduction) => {
      const available = pairs.filter((pair) => finite(pair?.[key]));
      const higher = available.filter((pair) => pair[key] > 0).length;
      const absent = pairs.length - available.length;
      return `${introduction} earned more than a quiet motor in ${higher} of ${pairs.length} declared comparisons.${absent ? ` ${absent} comparison(s) lack a recorded difference.` : ""}`;
    };
    ui.controlSummary.textContent = `${comparedWithQuiet("source_frozen_minus_source_quiet", "Before the rebuild, saved learning")} ${comparedWithQuiet("adapt_minus_quiet", "After the rebuild, continued learning")}`;
  } else ui.controlSummary.textContent = "The source-versus-quiet and adapting-versus-quiet comparisons have not been reported yet.";
  const useful = summary.useful_behavior_pairs;
  const pairCount = summary.paired_case_count;
  ui.usefulness.textContent = `${finite(useful) && finite(pairCount) ? `Both quiet comparisons and their completeness/safety checks passed in ${number(useful, 0)} of ${number(pairCount, 0)} matched pairs. ` : ""}Beating scratch alone does not establish useful behavior. A quiet motor still allows gravity and passive movement.`;
}

function describeProtocol() {
  const protocol = record.protocol;
  const decisions = protocol.target_decisions;
  const dt = protocol.decision_dt_s;
  const declaredDuration = finite(decisions) && finite(dt) ? decisions * dt : null;
  duration = finite(declaredDuration) && declaredDuration > 0 ? declaredDuration : 64;
  ui.timeline.max = String(duration);
  ui.duration.textContent = `${number(duration, 0)} seconds of new experience`;
  ui.sourceBudget.textContent = finite(protocol.source_decisions_per_seed) ? `${number(protocol.source_decisions_per_seed, 0)} decisions per learning seed` : "See the declared protocol below";
  ui.targetBudget.textContent = finite(decisions) ? `${number(decisions, 0)} decisions · ${number(duration, 0)} simulated seconds` : "See the declared protocol below";
  if (typeof protocol.scope === "string") ui.scope.textContent = protocol.scope;
  const evidenceStatus = record.summary?.evidence_status ?? record.evidence_status;
  ui.studyStatus.hidden = typeof evidenceStatus !== "string";
  if (!ui.studyStatus.hidden) {
    const reason = record.summary?.evidence_reason ?? record.evidence_reason ?? record.reference?.limitation ?? record.summary?.conclusion;
    ui.studyStatus.textContent = `Declared experiment status: ${evidenceStatus.replaceAll("_", " ")}. ${typeof reason === "string" ? reason : "The protocol, reference checks and complete experiment summary are available below."}`;
  }
  ui.protocol.textContent = JSON.stringify({ protocol, summary: record.summary ?? null, development: record.development ?? null, reference: record.reference ?? null }, null, 2);
  describeOutcome();
  ui.sampleNote.textContent = "Animation holds each saved native snapshot until the next one arrives on the replay clock. Playback is sampled for display; numerical results retain the native record. There is no interpolation or live controller in this page.";
}

async function loadIndex() {
  if (indexLoading) return;
  indexLoading = true;
  const request = ++indexGeneration;
  ++generation;
  setPlaying(false);
  ui.retry.disabled = true;
  ui.cases.disabled = true; ui.seeds.disabled = true; ui.reference.disabled = true;
  const oldCase = ui.cases.value;
  const oldSeed = ui.seeds.value;
  setStatus("loading", "Reading the experiment record…", "The comparison reads saved native simulations. It does not start a new experiment.");
  try {
    const next = checkIndex(await readJSON("/transfer-data.json"));
    if (request !== indexGeneration) return;
    record = next;
    traceCache.clear();
    describeProtocol();
    if (!record.cases.length || !record.seeds.length) throw new Error("The index is present; its declared comparison cases are still preparing.");
    ui.cases.replaceChildren(...record.cases.map((item) => option(item.id, item.title || item.id)));
    ui.seeds.replaceChildren(...record.seeds.map((seed) => option(seed, String(seed))));
    ui.cases.value = record.cases.some((item) => item.id === oldCase) ? oldCase : record.cases[0].id;
    ui.seeds.value = record.seeds.some((seed) => String(seed) === oldSeed) ? oldSeed : String(record.seeds.includes(101) ? 101 : record.seeds[0]);
    ui.cases.disabled = false; ui.seeds.disabled = false; ui.reference.disabled = false;
    await selectComparison();
  } catch (error) {
    if (request !== indexGeneration) return;
    if (!record || !Object.values(panels).some((panel) => panel.trace)) {
      setStatus("preparing", "The movement evidence is preparing", `${error.message} No comparison result is implied by these empty panels.`);
      for (const arm of ARMS) resetPanel(arm, "Evidence preparing");
      chartMapping = null; drawChart(); updatePlayback();
    } else {
      setStatus("partial", "The record could not be refreshed", `${error.message} The previously loaded snapshots remain on screen.`);
      ui.cases.disabled = false; ui.seeds.disabled = false; ui.reference.disabled = false;
    }
  } finally {
    if (request === indexGeneration) { indexLoading = false; ui.retry.disabled = false; }
  }
}

function resetPanel(arm, message) {
  const panel = panels[arm];
  panel.trial = null; panel.trace = null; panel.error = null; panel.rendered = "";
  panel.node.dataset.state = "waiting";
  panel.status.textContent = message;
  panel.reward.textContent = "—"; panel.light.textContent = "— lx";
  panel.note.textContent = message === "Evidence preparing" ? "A saved native trace will provide this creature’s movements." : "Reading this comparison’s physical snapshots.";
  drawWaiting(panel, message);
}

async function selectComparison({ keepTime = false } = {}) {
  if (!record) return;
  const request = ++generation;
  setPlaying(false);
  if (!keepTime) time = 0;
  selectedCase = record.cases.find((item) => item.id === ui.cases.value);
  const seed = Number(ui.seeds.value);
  ui.description.textContent = selectedCase?.description || "The same parts and sunshine are shared by all three rebuilt-body comparisons.";
  ui.panels.classList.toggle("has-reference", ui.reference.checked);
  panels.source_frozen.node.hidden = !ui.reference.checked;
  ui.sourceLegend.hidden = !ui.reference.checked;
  setStatus("loading", "Reading the saved movements…", "The three arms stay in a fixed order. Case and learning seed are chosen independently of the result.");
  for (const arm of visibleArms()) {
    resetPanel(arm, "Reading saved movement…");
    const panel = panels[arm];
    panel.trial = record.trials.find((trial) => trial.case_id === selectedCase?.id && trial.source_seed === seed && trial.arm === arm) ?? null;
    if (!panel.trial) {
      panel.error = "No trial record is available for this arm yet.";
      panel.status.textContent = "Evidence preparing";
      panel.note.textContent = panel.error;
      drawWaiting(panel, "Evidence preparing");
    }
  }
  updateMeasures(); drawChart(); updatePlayback();
  await Promise.all(visibleArms().map(async (arm) => {
    const panel = panels[arm];
    const trial = panel.trial;
    if (!trial) return;
    try {
      const trace = await traceFor(trial);
      if (request !== generation) return;
      panel.trace = trace; panel.node.dataset.state = "ready"; panel.rendered = "";
    } catch (error) {
      if (request !== generation) return;
      panel.error = error.message; panel.node.dataset.state = "error";
      panel.status.textContent = "Playback unavailable"; panel.note.textContent = error.message;
      drawWaiting(panel, "Playback unavailable");
    }
    if (request === generation) { drawChart(); updatePlayback(); }
  }));
  if (request !== generation) return;
  const present = visibleArms().filter((arm) => panels[arm].trace).length;
  const total = visibleArms().length;
  const complete = visibleArms().every((arm) => panels[arm].trace && panels[arm].trial.metrics?.complete === true);
  setStatus(present === total && complete ? "ready" : "partial",
    present === total ? complete ? "Saved movements are ready to compare" : "The record includes an incomplete run" : `${present} of ${total} saved movements are available`,
    `${selectedCase?.title || selectedCase?.id || "Selected creature"} · learning seed ${seed}. ${present ? "Press Play together, or scrub to any recorded moment." : "Read again when the evidence export is ready."} This is one selected comparison; all declared outcomes remain in the experiment record.`);
  updateMeasures(); updatePlayback();
  announce(`${present} recorded movements loaded for ${selectedCase?.title || selectedCase?.id}, learning seed ${seed}. Playback is paused.`);
}

function sampleAt(trace, at) {
  let low = 0; let high = trace.samples.length - 1;
  if (at + 1e-8 < trace.samples[0].time_s) return -1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (trace.samples[middle].time_s <= at + 1e-8) low = middle;
    else high = middle - 1;
  }
  return low;
}

function drawWaiting(panel, message) {
  panel.scene.setAttribute("viewBox", "0 0 360 340");
  panel.scene.setAttribute("aria-label", message);
  panel.scene.replaceChildren(
    svg("circle", { cx: 180, cy: 152, r: 44, fill: "none", stroke: "#45604a", "stroke-dasharray": "3 8" }),
    svg("text", { x: 180, y: 156, fill: "#8ba68e", "text-anchor": "middle", "font-size": 25 }, "· · ·"),
    svg("text", { x: 180, y: 225, fill: "#a2b293", "text-anchor": "middle", "font-size": 12, "font-family": "Georgia, serif" }, message),
  );
}

function projection(physics, width, height) {
  const assembly = physics.assembly ?? selectedCase?.case?.assembly;
  let reach = Array.isArray(assembly?.segments) && assembly.segments.every(finite) ? assembly.segments.reduce((sum, length) => sum + length, 0) * 0.04 + 0.07 : null;
  if (!finite(reach) && Array.isArray(physics.segments)) {
    const lengths = physics.segments.map((segment) => segment.length_studs);
    if (lengths.length && lengths.every(finite)) reach = lengths.reduce((sum, length) => sum + length, 0) * 0.04 + 0.07;
  }
  // A fixed view bound is display framing only; every rendered part uses native coordinates.
  if (!finite(reach) || reach <= 0) reach = 0.45;
  const source = vector(physics.sun?.position_m) ? physics.sun.position_m : null;
  const minX = Math.min(-reach, source ? source[0] - 0.08 : -reach);
  const maxX = Math.max(reach, source ? source[0] + 0.08 : reach);
  const minZ = Math.min(-reach, source ? source[2] - 0.08 : -reach);
  const maxZ = Math.max(reach, source ? source[2] + 0.08 : reach);
  const scale = Math.min((width - 46) / (maxX - minX), (height - 83) / (maxZ - minZ));
  const origin = [width / 2 - (maxX + minX) * scale / 2, (height - 28) / 2 + (maxZ + minZ) * scale / 2];
  return { scale, source, project: (point) => [origin[0] + point[0] * scale, origin[1] - point[2] * scale] };
}

function drawScene(arm, sample, index) {
  const panel = panels[arm];
  const width = Math.max(130, Math.round(panel.scene.clientWidth || 360));
  const height = Math.max(180, Math.round(panel.scene.clientHeight || 340));
  const key = `${index}:${width}:${height}`;
  if (panel.rendered === key) return;
  panel.rendered = key;
  const physics = sample.physics;
  if (!Array.isArray(physics.geometry) || !physics.geometry.length) {
    drawWaiting(panel, "Geometry not present in this sample");
    return;
  }
  const { scale, source, project } = projection(physics, width, height);
  const nodes = [svg("title", {}, `${LABELS[arm]} at ${number(sample.time_s, 1)} seconds`),
    svg("desc", {}, "An anchored two-link construction, drawn from recorded native physical coordinates. Gold marks the powered hinge; the small pale joint is passive. The sun and light sensor are shown at their recorded positions.")];
  for (let x = 18; x < width; x += 26) for (let y = 15; y < height - 34; y += 26) {
    nodes.push(svg("circle", { cx: x, cy: y, r: 0.65, fill: "#355543", opacity: 0.6 }));
  }
  const [anchorX, anchorY] = project([0, 0, 0]);
  nodes.push(svg("line", { x1: anchorX - 17, x2: anchorX + 17, y1: anchorY - 15, y2: anchorY - 15, stroke: "#7b9277", "stroke-width": 4, "stroke-linecap": "round" }),
    svg("line", { x1: anchorX, x2: anchorX, y1: anchorY - 15, y2: anchorY, stroke: "#7b9277", "stroke-width": 3 }));
  if (source) {
    const [x, y] = project(source);
    const group = svg("g", { "aria-label": "Recorded light source" });
    group.append(svg("circle", { cx: x, cy: y, r: 26, fill: "#e5bd65", opacity: 0.05 }));
    for (let ray = 0; ray < 12; ++ray) {
      const angle = ray * Math.PI / 6;
      group.append(svg("line", { x1: x + Math.cos(angle) * 17, y1: y + Math.sin(angle) * 17, x2: x + Math.cos(angle) * 22, y2: y + Math.sin(angle) * 22, stroke: "#ceb566", "stroke-width": 1.2, "stroke-linecap": "round" }));
    }
    group.append(svg("circle", { cx: x, cy: y, r: 11, fill: "#e2c16e", stroke: "#f2dc92", "stroke-width": 1.5 }));
    nodes.push(group);
  }
  for (const part of physics.geometry) {
    if (!vector(part.position_m) || !vector(part.size_m) || !vector(part.quaternion, 4) || part.size_m.some((size) => size < 0)) continue;
    const [x, y] = project(part.position_m);
    const [qw, qx, qy, qz] = part.quaternion;
    const angle = Math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy * qy + qz * qz)) * 180 / Math.PI;
    const partWidth = Math.max(2, part.size_m[0] * scale);
    const partHeight = Math.max(2, part.size_m[2] * scale);
    const color = part.kind === "light" ? "#dcc27b" : part.kind === "sensor" ? "#8eaa6a" : part.kind === "block" ? "#bf926d" : part.segment === 0 ? "#669885" : "#8999af";
    const group = svg("g", { transform: `translate(${x} ${y}) rotate(${angle})` });
    group.append(svg("rect", { x: -partWidth / 2, y: -partHeight / 2, width: partWidth, height: partHeight, rx: Math.min(3, partWidth / 4), fill: color, stroke: "#d5dfbe", "stroke-opacity": 0.5, "stroke-width": 0.8 }));
    if (part.kind === "block" || part.kind === "light" || part.kind === "sensor") {
      group.append(svg("circle", { cx: 0, cy: 0, r: Math.min(2.2, partWidth * 0.18), fill: part.kind === "light" ? "#fff1af" : "#e4e4c3", opacity: 0.75 }));
    }
    nodes.push(group);
  }
  for (const [name, radius, fill] of [["passive_m", 4.5, "#dbd8b3"], ["motor_m", 8.5, "#e4b963"]]) {
    if (!vector(physics.joints?.[name])) continue;
    const [x, y] = project(physics.joints[name]);
    nodes.push(svg("circle", { cx: x, cy: y, r: radius, fill, stroke: "#263f2c", "stroke-width": 2 }),
      svg("circle", { cx: x, cy: y, r: name === "motor_m" ? 2.6 : 1.3, fill: "#536347" }));
  }
  const eye = physics.light_sensor;
  if (vector(eye?.position_m) && vector(eye?.direction_m)) {
    const [x, y] = project(eye.position_m);
    const end = eye.position_m.map((coordinate, axis) => coordinate + eye.direction_m[axis] * 0.06);
    const [endX, endY] = project(end);
    nodes.push(svg("line", { x1: x, y1: y, x2: endX, y2: endY, stroke: "#f2d891", "stroke-width": 1.6, "stroke-linecap": "round", opacity: 0.8 }),
      svg("circle", { cx: x, cy: y, r: 3, fill: "#f2d891" }),
      svg("circle", { cx: endX, cy: endY, r: 1.6, fill: "#f2d891" }));
  }
  nodes.push(svg("circle", { cx: 15, cy: 17, r: 3.5, fill: "#e4b963" }),
    svg("text", { x: 24, y: 20, fill: "#adbb9d", "font-size": 9, "font-family": "Consolas, monospace" }, "powered hinge"));
  panel.scene.setAttribute("viewBox", `0 0 ${width} ${height}`);
  panel.scene.setAttribute("aria-label", `${LABELS[arm]}: recorded body at ${number(sample.time_s, 1)} seconds. Gold marks its powered hinge.`);
  panel.scene.replaceChildren(...nodes);
}

function renderPanel(arm) {
  const panel = panels[arm];
  if (!panel.trace) return;
  const index = sampleAt(panel.trace, time);
  if (index < 0) {
    if (panel.rendered !== "before-first") { drawWaiting(panel, "First recorded snapshot is still ahead"); panel.rendered = "before-first"; }
    panel.status.textContent = "Before the first saved snapshot";
    panel.reward.textContent = "—"; panel.light.textContent = "— lx";
    return;
  }
  const sample = panel.trace.samples[index];
  drawScene(arm, sample, index);
  panel.reward.textContent = number(sample.cumulative_reward);
  const sensor = sample.physics.light_sensor;
  const hasLightSample = sensor?.valid === true && finite(sensor.observations?.illuminance_lux);
  panel.light.textContent = hasLightSample ? `${number(sensor.observations.illuminance_lux, 0)} lx` : "— lx";
  panel.light.title = hasLightSample ? "Delivered local light-sensor measurement" : "No valid delivered light sample at this moment";
  const atEnd = time >= panel.trace.last - 1e-8;
  const complete = panel.trial.metrics?.complete === true;
  panel.status.textContent = atEnd ? `${complete ? "Run complete" : "Run ended"} · ${number(panel.trace.last, 1)} s` : `Saved moment · ${number(sample.time_s, 1)} s`;
  const stops = panel.trial.metrics?.safety_stops;
  panel.note.textContent = atEnd && !complete ? `This run ended at ${number(panel.trace.last, 1)} s; its last native position is held.${finite(stops) && stops > 0 ? ` ${number(stops, 0)} recorded safety stop(s).` : ""}` :
    arm === "source_frozen" ? "Original motor at the anchor. This reference has a different actuation layout." :
      `${complete ? "Complete recorded run" : "Incomplete recorded run"} · gold marks the motor at the elbow.${finite(stops) && stops > 0 ? ` ${number(stops, 0)} safety stop(s).` : ""}`;
}

function drawChart() {
  const width = 1000; const height = 220;
  const bounds = { left: 59, right: 979, top: 16, bottom: 181 };
  const series = visibleArms().map((arm) => ({ arm, samples: panels[arm].trace?.samples.filter((sample) => finite(sample.cumulative_reward)) ?? [] }));
  const rewards = series.flatMap((item) => item.samples.map((sample) => sample.cumulative_reward));
  const nodes = [];
  if (!rewards.length) {
    nodes.push(svg("line", { x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom, stroke: "#cbbd9f" }),
      svg("text", { x: width / 2, y: height / 2, "text-anchor": "middle", fill: "#938266", "font-size": 13, "font-family": "Georgia, serif" }, "The recorded reward curves will appear with the movements."));
    ui.chart.replaceChildren(...nodes); chartMapping = null;
    return;
  }
  let minimum = Math.min(0, ...rewards);
  let maximum = Math.max(0, ...rewards);
  const margin = Math.max((maximum - minimum) * 0.08, 0.01);
  minimum -= margin; maximum += margin;
  const x = (value) => bounds.left + value / duration * (bounds.right - bounds.left);
  const y = (value) => bounds.bottom - (value - minimum) / (maximum - minimum) * (bounds.bottom - bounds.top);
  for (let tick = 0; tick <= 4; ++tick) {
    const value = minimum + (maximum - minimum) * tick / 4;
    const py = y(value);
    nodes.push(svg("line", { x1: bounds.left, x2: bounds.right, y1: py, y2: py, stroke: "#dbceb1", "stroke-dasharray": "2 5" }),
      svg("text", { x: bounds.left - 10, y: py + 4, "text-anchor": "end", fill: "#928164", "font-size": 10, "font-family": "Consolas, monospace" }, number(value, Math.abs(value) >= 10 ? 1 : 2)));
    const seconds = duration * tick / 4;
    nodes.push(svg("text", { x: x(seconds), y: bounds.bottom + 25, "text-anchor": "middle", fill: "#928164", "font-size": 10, "font-family": "Consolas, monospace" }, `${number(seconds, 0)} s`));
  }
  nodes.push(svg("line", { x1: bounds.left, x2: bounds.right, y1: y(0), y2: y(0), stroke: "#bcb190", "stroke-width": 1 }));
  for (const item of series) {
    if (!item.samples.length) continue;
    const path = item.samples.map((sample, index) => `${index ? "L" : "M"}${x(sample.time_s).toFixed(3)},${y(sample.cumulative_reward).toFixed(3)}`).join(" ");
    const curve = svg("path", { d: path, fill: "none", stroke: COLORS[item.arm], "stroke-width": 2.4, "stroke-dasharray": DASHES[item.arm], "stroke-linecap": "round", "stroke-linejoin": "round" });
    curve.append(svg("title", {}, `${LABELS[item.arm]} — cumulative native reward`));
    nodes.push(curve);
    const last = item.samples[item.samples.length - 1];
    nodes.push(svg("circle", { cx: x(last.time_s), cy: y(last.cumulative_reward), r: 2.5, fill: COLORS[item.arm] }));
  }
  const cursor = svg("line", { x1: x(time), x2: x(time), y1: bounds.top, y2: bounds.bottom, stroke: "#4b664b", "stroke-width": 1.2, "stroke-dasharray": "3 4" });
  const dots = svg("g");
  nodes.push(cursor, dots);
  ui.chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  ui.chart.setAttribute("aria-label", "Cumulative native reward for the selected saved runs. The vertical cursor follows the shared replay time. Curves end at each run’s last saved sample.");
  ui.chart.replaceChildren(...nodes);
  chartMapping = { x, y, cursor, dots };
  updateChartCursor();
}

function updateChartCursor() {
  if (!chartMapping) return;
  const { x, y, cursor, dots } = chartMapping;
  cursor.setAttribute("x1", String(x(time))); cursor.setAttribute("x2", String(x(time)));
  const markers = [];
  for (const arm of visibleArms()) {
    const trace = panels[arm].trace;
    if (!trace) continue;
    const index = sampleAt(trace, time);
    if (index < 0) continue;
    const sample = trace.samples[index];
    if (!finite(sample.cumulative_reward)) continue;
    markers.push(svg("circle", { cx: x(sample.time_s), cy: y(sample.cumulative_reward), r: 4, fill: COLORS[arm], stroke: "#fbf1dc", "stroke-width": 1.5 }));
  }
  dots.replaceChildren(...markers);
}

function updateMeasures() {
  const rows = [];
  for (const arm of visibleArms()) {
    const trial = panels[arm].trial;
    const metrics = trial?.metrics;
    const row = document.createElement("tr");
    const name = document.createElement("th"); name.scope = "row";
    if (trial?.trace_url) {
      try {
        const link = document.createElement("a"); link.href = assetURL(trial.trace_url); link.textContent = LABELS[arm]; link.download = "";
        name.append(link);
      } catch { name.textContent = LABELS[arm]; }
    } else name.textContent = LABELS[arm];
    row.append(name);
    const replay = trial?.replay?.exact === true ? "Exact · recorded check" : trial?.replay?.exact === false ? "Replay mismatch" : "Not reported";
    for (const text of [finite(metrics?.duration_s) ? `${number(metrics.duration_s, 1)} s${metrics.complete === true ? "" : " · incomplete"}` : "Not reported", number(metrics?.native_reward), finite(metrics?.energy_j) ? `${number(metrics.energy_j, 3)} J` : "Not reported", replay]) {
      const cell = document.createElement("td"); cell.textContent = text; row.append(cell);
    }
    rows.push(row);
  }
  ui.measures.replaceChildren(...rows);
  const adapt = panels.adapt.trial?.metrics;
  const scratch = panels.scratch.trial?.metrics;
  const comparable = adapt?.complete === true && scratch?.complete === true && finite(adapt.native_reward) && finite(scratch.native_reward);
  ui.delta.textContent = comparable ? signed(adapt.native_reward - scratch.native_reward) : "—";
  ui.deltaNote.textContent = comparable ? "This creature and seed only. The whole-experiment result is shown above." : "A full-run comparison requires both adapting and scratch runs to be complete.";
  const matched = visibleArms().map((arm) => panels[arm].trial).filter(Boolean);
  const exact = matched.filter((trial) => trial.replay?.exact === true).length;
  ui.evidence.textContent = matched.length ? `${exact} of ${matched.length} selected trial records report exact native replay. These checks come from the exported evidence; playback does not rerun physics. A missing or failed run remains part of the experiment.` : "No selected trial records are available yet.";
}

function updatePlayback() {
  const available = visibleArms().some((arm) => panels[arm].trace);
  ui.playback.classList.toggle("has-recordings", available);
  ui.play.disabled = !available; ui.restart.disabled = !available; ui.timeline.disabled = !available;
  ui.time.textContent = `${number(time, 1)} / ${number(duration, 1)} s`;
  ui.timeline.value = String(time);
  ui.timeline.setAttribute("aria-valuetext", `${number(time, 1)} seconds of ${number(duration, 1)} seconds`);
  ui.play.textContent = playing ? "Pause together Ⅱ" : "Play together ▶";
  ui.play.setAttribute("aria-label", playing ? "Pause all recorded simulations" : "Play all recorded simulations");
  ui.play.setAttribute("aria-pressed", String(playing));
  for (const arm of visibleArms()) renderPanel(arm);
  updateChartCursor();
}

function setPlaying(next) {
  playing = next;
  if (raf) window.cancelAnimationFrame(raf);
  raf = 0; previousAnimationTime = null;
  if (playing) raf = window.requestAnimationFrame(animate);
  updatePlayback();
}

function animate(timestamp) {
  if (!playing) return;
  if (previousAnimationTime !== null) {
    time = Math.min(duration, time + Math.min((timestamp - previousAnimationTime) / 1000, 0.25) * Number(ui.speed.value));
  }
  previousAnimationTime = timestamp;
  if (time >= duration) {
    setPlaying(false); announce("The shared replay has reached its end. Restart to watch again."); return;
  }
  updatePlayback();
  raf = window.requestAnimationFrame(animate);
}

ui.cases.addEventListener("change", () => { void selectComparison(); });
ui.seeds.addEventListener("change", () => { void selectComparison(); });
ui.reference.addEventListener("change", () => { void selectComparison({ keepTime: true }); });
ui.retry.addEventListener("click", () => { void loadIndex(); });
ui.play.addEventListener("click", () => {
  if (time >= duration) time = 0;
  setPlaying(!playing);
  announce(playing ? "Playing saved simulations together." : `Replay paused at ${number(time, 1)} seconds.`);
});
ui.restart.addEventListener("click", () => {
  time = 0; setPlaying(false); announce("All saved simulations are back at their first moment. Playback is paused.");
});
ui.timeline.addEventListener("input", () => {
  time = Math.min(duration, Math.max(0, Number(ui.timeline.value)));
  setPlaying(false);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && playing) { setPlaying(false); announce("Replay paused while the page was hidden."); }
});
window.addEventListener("resize", () => { for (const panel of Object.values(panels)) panel.rendered = ""; updatePlayback(); });
for (const arm of ARMS) drawWaiting(panels[arm], "Evidence preparing");
drawChart();
void loadIndex();
