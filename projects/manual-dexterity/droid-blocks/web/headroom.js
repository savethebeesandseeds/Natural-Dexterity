"use strict";

// Read exported evidence only. This page never connects to an engine session.
const $ = (selector) => document.querySelector(selector);
const SVG_NS = "http://www.w3.org/2000/svg";
const ARMS = ["quiet", "chosen"];
const COLORS = { quiet: "#a48240", chosen: "#527e5f" };
const ui = {
  status: $("[data-load-state]"), title: $("[data-load-title]"), detail: $("[data-load-detail]"), retry: $("[data-retry]"),
  body: $("[data-body]"), hinge: $("[data-hinge]"), lamp: $("[data-lamp]"), controller: $("[data-controller]"), description: $("[data-case-description]"),
  chosenTitle: $("[data-chosen-title]"), chosenSubtitle: $("[data-chosen-subtitle]"), chosenLegend: $("[data-chosen-legend]"),
  playback: $(".shared-playback"), play: $("[data-play]"), restart: $("[data-restart]"), speed: $("[data-speed]"),
  time: $("[data-time]"), timeline: $("[data-timeline]"), duration: $("[data-duration-label]"), chart: $("[data-reward-chart]"),
  delta: $("[data-delta]"), deltaNote: $("[data-delta-note]"), conclusion: $("[data-conclusion]"), integrity: $("[data-integrity]"),
  studyHead: $("[data-study-head]"), studyBody: $("[data-study-body]"), studyCaption: $("[data-study-caption]"), studyNote: $("[data-study-note]"),
  measures: $("[data-measures]"), sampleNote: $("[data-sample-note]"), protocol: $("[data-protocol]"), live: $("[data-live]"),
};
const panels = Object.fromEntries(ARMS.map((arm) => {
  const node = $(`[data-panel="${arm}"]`);
  return [arm, { node, scene: node.querySelector("[data-motion]"), status: node.querySelector("[data-stage-status]"),
    reward: node.querySelector("[data-reward]"), light: node.querySelector("[data-light]"), note: node.querySelector("[data-panel-note]"),
    trial: null, trace: null, rendered: "" }];
}));
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const vector = (value, length = 3) => Array.isArray(value) && value.length === length && value.every(finite);
const number = (value, digits = 2) => finite(value) ? value.toLocaleString("en", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
const signed = (value, digits = 2) => `${value > 0 ? "+" : ""}${number(value, digits)}`;
const element = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
const option = (value, text) => { const node = element("option", text); node.value = String(value); return node; };
const bodyKey = (item) => JSON.stringify([item.case.assembly.segments, item.case.assembly.blocks?.length ?? 0]);
const hingeKey = (item) => String(item.case.assembly.powered_hinge);
const lampKey = (item) => JSON.stringify(item.case.sun.position_m);
const lowerLamp = (item) => item.case.sun.position_m[2] < 0;
const bodyLabel = (item) => `${item.case.assembly.segments.join(" + ")} studs${item.case.assembly.blocks?.length ? " · weighted" : ""}`;
const hingeLabel = (item) => item.case.assembly.powered_hinge === 0 ? "At the anchor" : "At the elbow";
const lampLabel = (item) => `${lowerLamp(item) ? "Below" : "Above"} · ${item.case.sun.position_m[0] < 0 ? "left" : "right"}`;
const unique = (items, key) => [...new Map(items.map((item) => [key(item), item])).values()];
const selections = [ui.body, ui.hinge, ui.lamp, ui.controller];
const traceCache = new Map();
let record = null;
let selectedCase = null;
let trialIndex = new Map();
let quietController = null;
let duration = 64;
let threshold = 3.2;
let time = 0;
let playing = false;
let raf = 0;
let previousAnimationTime = null;
let chartMapping = null;
let generation = 0;
let indexLoading = false;

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function announce(message) { ui.live.textContent = message; }
function setStatus(state, title, detail) {
  ui.status.dataset.state = state; ui.title.textContent = title; ui.detail.textContent = detail;
}
function assetURL(path) {
  if (typeof path !== "string" || !path.trim()) throw new Error("No playback file was declared.");
  const url = new URL(path, window.location.href);
  const pathname = decodeURIComponent(url.pathname);
  if (!["http:", "https:"].includes(url.protocol) || url.origin !== window.location.origin || url.username || url.password ||
      url.search || url.hash || !pathname.endsWith(".json") || /(^|\/)api(\/|$)/i.test(pathname) || pathname.includes("\\")) {
    throw new Error("Playback must use a local JSON evidence asset.");
  }
  return url.href;
}
async function readJSON(path) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(assetURL(path), { method: "GET", cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(response.status === 404 ? "This evidence file is still preparing." : `Evidence returned HTTP ${response.status}.`);
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The evidence file did not arrive in time. Read again to retry.");
    if (error instanceof SyntaxError) throw new Error("The evidence file is not a readable JSON record yet.");
    throw error;
  } finally { window.clearTimeout(timeout); }
}
function checkIndex(value) {
  if (value?.schema !== "light_headroom_view_v1" || !Array.isArray(value.cases) || !Array.isArray(value.controllers) ||
      !Array.isArray(value.trials) || !value.protocol) throw new Error("The index is not in the expected headroom playback format.");
  const caseIds = new Set(); const dimensions = new Set(); const controllerIds = new Set(); const trialIds = new Set(); const pairs = new Set();
  for (const item of value.cases) {
    if (!item || typeof item.id !== "string" || !item.id || caseIds.has(item.id) || !vector(item.case?.assembly?.segments, 2) ||
        ![0, 1].includes(item.case.assembly.powered_hinge) || !vector(item.case?.sun?.position_m)) {
      throw new Error("The index has invalid or repeated body, hinge or lamp declarations.");
    }
    const dimension = JSON.stringify([bodyKey(item), hingeKey(item), lampKey(item)]);
    if (dimensions.has(dimension)) throw new Error("Two cases declare the same body, hinge and lamp.");
    caseIds.add(item.id); dimensions.add(dimension);
  }
  for (const controller of value.controllers) {
    if (!controller || typeof controller.id !== "string" || !controller.id || controllerIds.has(controller.id) ||
        !["quiet", "handwritten", "learner"].includes(controller.kind) ||
        (controller.kind === "learner" && !Number.isSafeInteger(controller.seed))) {
      throw new Error("The index has invalid or ambiguous controller declarations.");
    }
    controllerIds.add(controller.id);
  }
  if (value.controllers.length && value.controllers.filter((item) => item.kind === "quiet").length !== 1) {
    throw new Error("The index must identify exactly one quiet control.");
  }
  for (const trial of value.trials) {
    const key = JSON.stringify([trial?.case_id, trial?.controller_id]);
    if (!trial || typeof trial.id !== "string" || !trial.id || trialIds.has(trial.id) || pairs.has(key) ||
        !caseIds.has(trial.case_id) || !controllerIds.has(trial.controller_id)) {
      throw new Error("The index has an invalid or repeated case/controller trial.");
    }
    trialIds.add(trial.id); pairs.add(key);
  }
  return value;
}
function checkTrace(value, trial) {
  if (value?.schema !== "light_headroom_playback_v1" || value.id !== trial.id || !Array.isArray(value.frames)) {
    throw new Error("This playback file does not match its declared trial.");
  }
  const samples = [];
  if (value.initial_physics && typeof value.initial_physics === "object") {
    samples.push({ time_s: 0, physics: value.initial_physics, cumulative_reward: value.initial_physics.reward?.cumulative ?? 0 });
  }
  let previous = -1;
  for (const frame of value.frames) {
    if (!frame || !finite(frame.time_s) || frame.time_s < 0 || frame.time_s <= previous || frame.time_s > duration + 1e-5 ||
        !frame.physics || typeof frame.physics !== "object" || !finite(frame.cumulative_reward)) {
      throw new Error("Saved snapshots have an invalid time, physical state or reward.");
    }
    previous = frame.time_s;
    if (frame.time_s === 0 && samples.length) samples[0] = frame;
    else samples.push(frame);
  }
  if (!samples.length) throw new Error("This trial has no saved physical snapshots yet.");
  const last = samples[samples.length - 1].time_s;
  if (finite(trial.metrics?.duration_s) && (last > trial.metrics.duration_s + 1e-5 ||
      trial.metrics.duration_s - last >= 0.2 + 1e-5 ||
      (trial.metrics.complete === true && Math.abs(last - trial.metrics.duration_s) > 1e-5))) {
    throw new Error("The saved snapshot schedule and recorded trial duration disagree.");
  }
  return { samples, last };
}
async function traceFor(trial) {
  const url = assetURL(trial.trace_url);
  let pending = traceCache.get(url);
  if (!pending) {
    pending = readJSON(url); traceCache.set(url, pending);
    pending.catch(() => { if (traceCache.get(url) === pending) traceCache.delete(url); });
    while (traceCache.size > 12) traceCache.delete(traceCache.keys().next().value);
  } else { traceCache.delete(url); traceCache.set(url, pending); }
  return checkTrace(await pending, trial);
}
const trialFor = (item, controller) => trialIndex.get(JSON.stringify([item.id, controller.id])) ?? null;
const fullRun = (trial) => trial?.metrics?.complete === true && finite(trial.metrics.duration_s) && Math.abs(trial.metrics.duration_s - duration) < 1e-5;
const safeRun = (trial) => fullRun(trial) && trial.metrics.safe === true && trial.metrics.safety_stops === 0;
const gainFor = (trial, quiet) => finite(trial?.metrics?.native_reward) && finite(quiet?.metrics?.native_reward) ? trial.metrics.native_reward - quiet.metrics.native_reward : null;
const metCriterion = (trial, quiet) => safeRun(trial) && safeRun(quiet) && finite(gainFor(trial, quiet)) && gainFor(trial, quiet) > threshold;
const pairStatus = (trial, quiet) => !fullRun(trial) || !fullRun(quiet) ? "Incomplete or missing full-run evidence" :
  !safeRun(trial) || !safeRun(quiet) ? "Safety criterion not met" : metCriterion(trial, quiet) ? `Gain exceeds ${number(threshold, 1)}` : `Gain does not exceed ${number(threshold, 1)}`;

function rebuildSelectors(preferredCaseId, preferredControllerId) {
  const preferred = record.cases.find((item) => item.id === preferredCaseId) ?? record.cases[0];
  ui.body.replaceChildren(...unique(record.cases, bodyKey).map((item) => option(bodyKey(item), bodyLabel(item))));
  ui.body.value = bodyKey(preferred);
  updateDimensions(preferred);
  ui.controller.replaceChildren(...record.controllers.map((item) => option(item.id, item.title || item.id)));
  const fallback = record.controllers.find((item) => item.kind !== "quiet") ?? quietController;
  ui.controller.value = record.controllers.some((item) => item.id === preferredControllerId) ? preferredControllerId : fallback.id;
}
function updateDimensions(preferred = selectedCase) {
  const bodyCases = record.cases.filter((item) => bodyKey(item) === ui.body.value);
  const oldHinge = preferred ? hingeKey(preferred) : ui.hinge.value;
  ui.hinge.replaceChildren(...unique(bodyCases, hingeKey).map((item) => option(hingeKey(item), hingeLabel(item))));
  ui.hinge.value = bodyCases.some((item) => hingeKey(item) === oldHinge) ? oldHinge : hingeKey(bodyCases[0]);
  const hingeCases = bodyCases.filter((item) => hingeKey(item) === ui.hinge.value);
  const oldLamp = preferred ? lampKey(preferred) : ui.lamp.value;
  ui.lamp.replaceChildren(...unique(hingeCases, lampKey).map((item) => option(lampKey(item), lampLabel(item))));
  ui.lamp.value = hingeCases.some((item) => lampKey(item) === oldLamp) ? oldLamp : lampKey(hingeCases[0]);
}
function describeGates() {
  const lower = record.cases.filter(lowerLamp);
  const handwritten = record.controllers.filter((item) => item.kind === "handwritten");
  const learners = record.controllers.filter((item) => item.kind === "learner");
  const everyTrialDeclared = record.cases.every((item) => record.controllers.every((controller) => trialFor(item, controller)));
  const completeGrid = record.cases.length === 24 && lower.length === 12 && record.controllers.length === 9 &&
    learners.length === 3 && [101, 202, 303].every((seed) => learners.some((item) => item.seed === seed)) && handwritten.length === 5 && everyTrialDeclared;
  const headroomCases = lower.filter((item) => handwritten.some((controller) => metCriterion(trialFor(item, controller), trialFor(item, quietController)))).length;
  const learnerCases = lower.filter((item) => learners.length === 3 && learners.every((controller) => metCriterion(trialFor(item, controller), trialFor(item, quietController)))).length;
  const learnerCounts = learners.map((controller) => ({ seed: controller.seed, count: lower.filter((item) => metCriterion(trialFor(item, controller), trialFor(item, quietController))).length }));
  const declared = record.summary ?? {};
  const gate = (name, count, declaredResult, detail) => {
    const node = $(`[data-gate="${name}"]`);
    const measured = count === 12;
    const disagrees = completeGrid && typeof declaredResult === "boolean" && measured !== declaredResult;
    node.dataset.result = !completeGrid || disagrees ? "unreported" : measured ? "passed" : "failed";
    node.querySelector("[data-gate-result]").textContent = !completeGrid ? "Incomplete evidence" : disagrees ? "Summary disagreement" : measured ? "Criterion met" : "Criterion not met";
    node.querySelector("[data-gate-detail]").textContent = `${detail}${disagrees ? " The declared summary and recorded trial metrics disagree; inspect the raw record below." : ""}`;
    return { passed: completeGrid && measured && !disagrees, ready: completeGrid && !disagrees };
  };
  const h = gate("headroom", headroomCases, declared.headroom_gate_passed,
    `${headroomCases} of 12 lower-lamp cases have at least one complete, safe handwritten witness with gain above ${number(threshold, 1)} versus quiet. Different cases may use different witnesses.`);
  const l = gate("learner", learnerCases, declared.learner_gate_passed,
    `${learnerCases} of 12 lower-lamp cases meet the gain and safety criterion for all three seeds. ${learnerCounts.map((item) => `Seed ${item.seed}: ${item.count}/12`).join(" · ")}.`);
  ui.conclusion.textContent = !h.ready || !l.ready ? "The complete declared evidence is needed to assess both gates. Missing, incomplete and unsafe trials remain visible below." :
    h.passed && l.passed ? "Useful handwritten behavior and the unchanged learner both meet the declared lower-lamp criterion in this development comparison." :
      h.passed ? "Useful behavior is available in every lower-lamp case, but the unchanged learner does not meet the criterion across every case and seed." :
        l.passed ? "The unchanged learner meets the lower-lamp criterion, but handwritten witnesses do not establish headroom in every case." :
          "Neither gate meets the declared criterion. Handwritten witnesses do not establish useful behavior in every lower-lamp case, and the unchanged learner does not achieve it across every case and seed.";
  const completeSafe = record.trials.filter(safeRun).length;
  const replayed = record.trials.filter((trial) => trial.replay?.exact === true).length;
  ui.integrity.textContent = `${completeSafe} of ${record.trials.length} recorded trials completed safely; ${replayed} report exact native replay. Upper-lamp cases remain visible as context. These gates establish neither transfer after rebuilding nor hardware performance.`;
}
function drawStudy() {
  const header = element("tr");
  const first = element("th", "Body · motor · lamp"); first.scope = "col"; header.append(first);
  for (const controller of record.controllers) {
    const cell = element("th", controller.title || controller.id); cell.scope = "col"; header.append(cell);
  }
  ui.studyHead.replaceChildren(header);
  const rows = record.cases.map((item) => {
    const row = element("tr"); row.dataset.caseId = item.id; row.dataset.lampLevel = lowerLamp(item) ? "lower" : "upper";
    const name = element("th"); name.scope = "row";
    const chooseCase = element("button", bodyLabel(item)); chooseCase.type = "button"; chooseCase.dataset.recordCase = item.id;
    chooseCase.append(element("small", `${hingeLabel(item)} · ${lampLabel(item)}${lowerLamp(item) ? " · gate case" : " · context"}`)); name.append(chooseCase); row.append(name);
    const quiet = trialFor(item, quietController);
    for (const controller of record.controllers) {
      const trial = trialFor(item, controller); const gain = gainFor(trial, quiet);
      const cell = element("td"); const button = element("button"); button.type = "button";
      button.dataset.recordCase = item.id; button.dataset.recordController = controller.id;
      if (controller.kind === "quiet") {
        button.textContent = `${number(trial?.metrics?.native_reward)}${safeRun(trial) ? "" : " ?"}`;
        button.title = `Absolute quiet reward${safeRun(trial) ? " · complete safe run" : " · incomplete, unsafe or missing evidence"}`;
      } else {
        const status = pairStatus(trial, quiet);
        const mark = !fullRun(trial) || !fullRun(quiet) ? "?" : metCriterion(trial, quiet) ? "●" : "○";
        button.textContent = `${finite(gain) ? signed(gain) : "—"} ${mark}`;
        button.title = `${status}${!fullRun(trial) || !fullRun(quiet) ? "; any shown gain uses recorded prefixes" : ""}`;
        button.dataset.sign = !finite(gain) ? "unknown" : gain > 0 ? "positive" : gain < 0 ? "negative" : "zero";
      }
      button.setAttribute("aria-label", `${bodyLabel(item)}, ${hingeLabel(item)}, lamp ${lampLabel(item)}. ${controller.title || controller.id}: ${button.textContent}. ${button.title}. Watch recorded trial.`);
      cell.append(button); row.append(cell);
    }
    return row;
  });
  ui.studyBody.replaceChildren(...rows);
  ui.studyCaption.textContent = `${record.cases.length} declared cases × ${record.controllers.length} controllers · quiet reward and gains versus quiet`;
  ui.studyNote.textContent = `${record.trials.length} trial records. All ${record.cases.filter(lowerLamp).length} lower-lamp gate cases and ${record.cases.filter((item) => !lowerLamp(item)).length} upper-lamp context cases are shown. Case order and the initial selection do not depend on performance. ? retains incomplete or missing evidence.`;
}
function markSelection() {
  for (const row of ui.studyBody.querySelectorAll("tr[data-case-id]")) row.dataset.selected = String(row.dataset.caseId === selectedCase?.id);
  for (const button of ui.studyBody.querySelectorAll("button[data-record-controller]")) {
    button.setAttribute("aria-pressed", String(button.dataset.recordCase === selectedCase?.id && button.dataset.recordController === ui.controller.value));
  }
}
function updateMeasures() {
  if (!record || !selectedCase) return;
  const quiet = trialFor(selectedCase, quietController);
  ui.measures.replaceChildren(...record.controllers.map((controller) => {
    const trial = trialFor(selectedCase, controller); const metrics = trial?.metrics; const row = element("tr");
    const name = element("th"); name.scope = "row";
    if (trial?.trace_url) {
      try { const link = element("a", controller.title || controller.id); link.href = assetURL(trial.trace_url); link.download = ""; name.append(link); }
      catch { name.textContent = controller.title || controller.id; }
    } else name.textContent = controller.title || controller.id;
    row.append(name);
    const safety = safeRun(trial) ? "Complete · safe" : !trial ? "Missing record" :
      `${fullRun(trial) ? "Complete" : "Incomplete"} · ${metrics?.safe === true ? "safe" : metrics?.safe === false ? "unsafe" : "safety unreported"}${finite(metrics?.safety_stops) && metrics.safety_stops ? ` · ${metrics.safety_stops} flagged/veto steps` : ""}`;
    for (const text of [finite(metrics?.duration_s) ? `${number(metrics.duration_s, 1)} s` : "—", number(metrics?.native_reward),
      controller.kind === "quiet" ? "Control" : finite(gainFor(trial, quiet)) ? `${signed(gainFor(trial, quiet))}${fullRun(trial) && fullRun(quiet) ? "" : " · prefix"}` : "—",
      finite(metrics?.energy_j) ? `${number(metrics.energy_j, 3)} J` : "—", safety]) row.append(element("td", text));
    return row;
  }));
  const chosen = panels.chosen.trial; const gain = gainFor(chosen, quiet);
  const full = fullRun(chosen) && fullRun(quiet) && finite(gain);
  ui.delta.textContent = full ? signed(gain) : "—";
  ui.deltaNote.textContent = full ? `${pairStatus(chosen, quiet)}. This case only; both study gates include every lower-lamp case.` :
    "A full-run gain requires both trials to complete. Recorded prefix rewards remain in the table.";
}
function resetPanel(arm, message) {
  const panel = panels[arm]; panel.trial = null; panel.trace = null; panel.rendered = ""; panel.node.dataset.state = "waiting";
  panel.status.textContent = message; panel.reward.textContent = "—"; panel.light.textContent = "— lx";
  panel.note.textContent = "Saved native snapshots provide this creature’s movements."; drawWaiting(panel, message);
}
async function loadIndex() {
  if (indexLoading) return;
  indexLoading = true; ++generation; setPlaying(false); ui.retry.disabled = true;
  selections.forEach((node) => { node.disabled = true; });
  const oldCase = selectedCase?.id; const oldController = ui.controller.value;
  setStatus("loading", "Reading the experiment record…", "This page reads saved native evidence. It does not start an experiment.");
  try {
    const next = checkIndex(await readJSON("/headroom-data.json"));
    if (!next.cases.length || !next.controllers.length) throw new Error("The index is present; its cases and controllers are still preparing.");
    record = next; quietController = record.controllers.find((item) => item.kind === "quiet");
    trialIndex = new Map(record.trials.map((trial) => [JSON.stringify([trial.case_id, trial.controller_id]), trial]));
    duration = finite(record.protocol.duration_s) && record.protocol.duration_s > 0 ? record.protocol.duration_s : 64;
    threshold = finite(record.protocol.gain_threshold) ? record.protocol.gain_threshold : 3.2;
    ui.timeline.max = String(duration); ui.duration.textContent = `${number(duration, 0)} seconds of opportunity`;
    traceCache.clear(); rebuildSelectors(oldCase, oldController); drawStudy(); describeGates();
    ui.protocol.textContent = JSON.stringify({ protocol: record.protocol, summary: record.summary ?? null, provenance: record.provenance ?? null }, null, 2);
    ui.sampleNote.textContent = "Playback holds sampled native snapshots without interpolation. The experiment’s numerical reward, energy, safety and replay checks come from its full native records.";
    selections.forEach((node) => { node.disabled = false; });
    await selectComparison();
  } catch (error) {
    if (record && selectedCase) {
      selections.forEach((node) => { node.disabled = false; });
      setStatus("error", "The evidence could not be refreshed", `${error.message} Previously loaded evidence remains visible.`);
    } else {
      for (const arm of ARMS) resetPanel(arm, "Evidence preparing");
      setStatus("error", "The movement evidence is preparing", `${error.message} Empty panels imply no experiment result.`);
      drawChart(); updatePlayback();
    }
    announce(ui.detail.textContent);
  } finally { indexLoading = false; ui.retry.disabled = false; }
}
async function selectComparison() {
  if (!record) return;
  const request = ++generation; setPlaying(false); time = 0;
  selectedCase = record.cases.find((item) => bodyKey(item) === ui.body.value && hingeKey(item) === ui.hinge.value && lampKey(item) === ui.lamp.value);
  const controller = record.controllers.find((item) => item.id === ui.controller.value);
  if (!selectedCase || !controller) return;
  const position = selectedCase.case.sun.position_m;
  ui.description.textContent = `${bodyLabel(selectedCase)} · powered hinge ${selectedCase.case.assembly.powered_hinge === 0 ? "at the anchor" : "at the elbow"} · lamp ${lampLabel(selectedCase).toLowerCase()} (x ${signed(position[0], 1)} m, z ${signed(position[2], 1)} m). ${lowerLamp(selectedCase) ? "This is one of the lower-lamp gate cases." : "This upper-lamp case supplies context for the lower-lamp gates."}`;
  ui.chosenTitle.textContent = controller.title || controller.id; ui.chosenLegend.textContent = controller.title || controller.id;
  ui.chosenSubtitle.textContent = controller.kind === "learner" ? `Native learner · seed ${controller.seed}` : controller.kind === "quiet" ? "The same quiet recording, for reference" : "Handwritten reference controller";
  setStatus("loading", "Reading the matched saved movements…", "Quiet and the selected controller share the same body, lamp and opportunity. Playback starts paused.");
  for (const arm of ARMS) {
    resetPanel(arm, "Reading saved movement…");
    panels[arm].trial = trialFor(selectedCase, arm === "quiet" ? quietController : controller);
    if (!panels[arm].trial) { panels[arm].status.textContent = "Missing trial record"; drawWaiting(panels[arm], "Missing trial record"); }
  }
  markSelection(); updateMeasures(); drawChart(); updatePlayback();
  await Promise.all(ARMS.map(async (arm) => {
    const panel = panels[arm]; const trial = panel.trial;
    if (!trial) return;
    try {
      const trace = await traceFor(trial);
      if (request !== generation) return;
      panel.trace = trace; panel.node.dataset.state = "ready"; panel.rendered = "";
    } catch (error) {
      if (request !== generation) return;
      panel.node.dataset.state = "error"; panel.status.textContent = "Playback unavailable";
      panel.note.textContent = error.message; drawWaiting(panel, "Playback unavailable");
    }
    if (request === generation) { drawChart(); updatePlayback(); }
  }));
  if (request !== generation) return;
  const available = ARMS.filter((arm) => panels[arm].trace).length;
  setStatus(available === 2 ? "ready" : "partial", available === 2 ? "Saved movements are ready to compare" : `${available} of 2 saved movements are available`,
    `${bodyLabel(selectedCase)} · ${hingeLabel(selectedCase)} · lamp ${lampLabel(selectedCase)}. ${available ? "Press Play together or scrub to a saved moment." : "Read again to retry the evidence export."} A selected replay never changes the whole-study result.`);
  updateMeasures(); updatePlayback(); announce(`${available} saved movements loaded. Playback is paused.`);
}
function sampleAt(trace, at) {
  if (at + 1e-8 < trace.samples[0].time_s) return -1;
  let low = 0; let high = trace.samples.length - 1;
  while (low < high) { const middle = Math.ceil((low + high) / 2); if (trace.samples[middle].time_s <= at + 1e-8) low = middle; else high = middle - 1; }
  return low;
}
function drawWaiting(panel, message) {
  panel.scene.setAttribute("viewBox", "0 0 360 340"); panel.scene.setAttribute("aria-label", message);
  panel.scene.replaceChildren(svg("circle", { cx: 180, cy: 145, r: 44, fill: "none", stroke: "#45604a", "stroke-dasharray": "3 8" }),
    svg("text", { x: 180, y: 149, fill: "#8ba68e", "text-anchor": "middle", "font-size": 25 }, "· · ·"),
    svg("text", { x: 180, y: 215, fill: "#a2b293", "text-anchor": "middle", "font-size": 12, "font-family": "Georgia, serif" }, message));
}
function projection(physics, width, height) {
  const assembly = physics.assembly ?? selectedCase?.case.assembly;
  let reach = vector(assembly?.segments, 2) ? assembly.segments.reduce((sum, length) => sum + length, 0) * 0.04 + 0.07 : 0.45;
  if (!finite(reach) || reach <= 0) reach = 0.45;
  const source = vector(physics.sun?.position_m) ? physics.sun.position_m : null;
  const minX = Math.min(-reach, source ? source[0] - 0.08 : -reach); const maxX = Math.max(reach, source ? source[0] + 0.08 : reach);
  const minZ = Math.min(-reach, source ? source[2] - 0.08 : -reach); const maxZ = Math.max(reach, source ? source[2] + 0.08 : reach);
  const scale = Math.min((width - 46) / (maxX - minX), (height - 83) / (maxZ - minZ));
  const origin = [width / 2 - (maxX + minX) * scale / 2, (height - 28) / 2 + (maxZ + minZ) * scale / 2];
  return { scale, source, project: (point) => [origin[0] + point[0] * scale, origin[1] - point[2] * scale] };
}
function drawScene(arm, sample, index) {
  const panel = panels[arm]; const width = Math.max(130, Math.round(panel.scene.clientWidth || 360));
  const height = Math.max(180, Math.round(panel.scene.clientHeight || 340)); const key = `${index}:${width}:${height}`;
  if (panel.rendered === key) return;
  panel.rendered = key; const physics = sample.physics;
  if (!Array.isArray(physics.geometry) || !physics.geometry.length) { drawWaiting(panel, "Geometry not present in this sample"); return; }
  const { scale, source, project } = projection(physics, width, height);
  const label = arm === "quiet" ? "Quiet motor" : ui.chosenTitle.textContent;
  const nodes = [svg("title", {}, `${label} at ${number(sample.time_s, 1)} seconds`), svg("desc", {}, "An anchored two-link construction drawn from saved native coordinates. Gold marks the powered hinge; the pale joint is passive. The lamp and the light sensor are shown at their recorded positions.")];
  for (let x = 18; x < width; x += 26) for (let y = 15; y < height - 34; y += 26) nodes.push(svg("circle", { cx: x, cy: y, r: 0.65, fill: "#355543", opacity: 0.6 }));
  const [anchorX, anchorY] = project([0, 0, 0]);
  nodes.push(svg("line", { x1: anchorX - 17, x2: anchorX + 17, y1: anchorY - 15, y2: anchorY - 15, stroke: "#7b9277", "stroke-width": 4, "stroke-linecap": "round" }),
    svg("line", { x1: anchorX, x2: anchorX, y1: anchorY - 15, y2: anchorY, stroke: "#7b9277", "stroke-width": 3 }));
  if (source) {
    const [x, y] = project(source); const group = svg("g", { "aria-label": "Recorded light source" });
    group.append(svg("circle", { cx: x, cy: y, r: 26, fill: "#e5bd65", opacity: 0.05 }));
    for (let ray = 0; ray < 12; ++ray) {
      const angle = ray * Math.PI / 6;
      group.append(svg("line", { x1: x + Math.cos(angle) * 17, y1: y + Math.sin(angle) * 17, x2: x + Math.cos(angle) * 22, y2: y + Math.sin(angle) * 22, stroke: "#ceb566", "stroke-width": 1.2, "stroke-linecap": "round" }));
    }
    group.append(svg("circle", { cx: x, cy: y, r: 11, fill: "#e2c16e", stroke: "#f2dc92", "stroke-width": 1.5 })); nodes.push(group);
  }
  for (const part of physics.geometry) {
    if (!vector(part.position_m) || !vector(part.size_m) || !vector(part.quaternion, 4) || part.size_m.some((size) => size < 0)) continue;
    const [x, y] = project(part.position_m); const [qw, qx, qy, qz] = part.quaternion;
    const angle = Math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy * qy + qz * qz)) * 180 / Math.PI;
    const partWidth = Math.max(2, part.size_m[0] * scale); const partHeight = Math.max(2, part.size_m[2] * scale);
    const color = part.kind === "light" ? "#dcc27b" : part.kind === "sensor" ? "#8eaa6a" : part.kind === "block" ? "#bf926d" : part.segment === 0 ? "#669885" : "#8999af";
    const group = svg("g", { transform: `translate(${x} ${y}) rotate(${angle})` });
    group.append(svg("rect", { x: -partWidth / 2, y: -partHeight / 2, width: partWidth, height: partHeight, rx: Math.min(3, partWidth / 4), fill: color, stroke: "#d5dfbe", "stroke-opacity": 0.5, "stroke-width": 0.8 }));
    if (["block", "light", "sensor"].includes(part.kind)) group.append(svg("circle", { cx: 0, cy: 0, r: Math.min(2.2, partWidth * 0.18), fill: part.kind === "light" ? "#fff1af" : "#e4e4c3", opacity: 0.75 }));
    nodes.push(group);
  }
  for (const [name, radius, fill] of [["passive_m", 4.5, "#dbd8b3"], ["motor_m", 8.5, "#e4b963"]]) {
    if (!vector(physics.joints?.[name])) continue;
    const [x, y] = project(physics.joints[name]);
    nodes.push(svg("circle", { cx: x, cy: y, r: radius, fill, stroke: "#263f2c", "stroke-width": 2 }), svg("circle", { cx: x, cy: y, r: name === "motor_m" ? 2.6 : 1.3, fill: "#536347" }));
  }
  const eye = physics.light_sensor;
  if (vector(eye?.position_m) && vector(eye?.direction_m)) {
    const [x, y] = project(eye.position_m); const end = eye.position_m.map((coordinate, axis) => coordinate + eye.direction_m[axis] * 0.06); const [endX, endY] = project(end);
    nodes.push(svg("line", { x1: x, y1: y, x2: endX, y2: endY, stroke: "#f2d891", "stroke-width": 1.6, "stroke-linecap": "round", opacity: 0.8 }),
      svg("circle", { cx: x, cy: y, r: 3, fill: "#f2d891" }), svg("circle", { cx: endX, cy: endY, r: 1.6, fill: "#f2d891" }));
  }
  nodes.push(svg("circle", { cx: 15, cy: 17, r: 3.5, fill: "#e4b963" }), svg("text", { x: 24, y: 20, fill: "#adbb9d", "font-size": 9, "font-family": "Consolas, monospace" }, "powered hinge"));
  panel.scene.setAttribute("viewBox", `0 0 ${width} ${height}`); panel.scene.setAttribute("aria-label", `${label}: recorded body at ${number(sample.time_s, 1)} seconds. Gold marks its powered hinge.`);
  panel.scene.replaceChildren(...nodes);
}
function renderPanel(arm) {
  const panel = panels[arm]; if (!panel.trace) return;
  const index = sampleAt(panel.trace, time);
  if (index < 0) { drawWaiting(panel, "Before the first saved snapshot"); panel.rendered = "before-first"; panel.status.textContent = "Before the first saved snapshot"; panel.reward.textContent = "—"; panel.light.textContent = "— lx"; return; }
  const sample = panel.trace.samples[index]; drawScene(arm, sample, index); panel.reward.textContent = number(sample.cumulative_reward);
  const sensor = sample.physics.light_sensor; const hasLight = sensor?.valid === true && finite(sensor.observations?.illuminance_lux);
  panel.light.textContent = hasLight ? `${number(sensor.observations.illuminance_lux, 0)} lx` : "— lx";
  panel.light.title = hasLight ? "Delivered local light measurement" : "No valid delivered light sample at this moment";
  const atEnd = time >= panel.trace.last - 1e-8;
  panel.status.textContent = atEnd ? `${fullRun(panel.trial) ? "Run complete" : "Last saved moment"} · ${number(panel.trace.last, 1)} s` : `Saved moment · ${number(sample.time_s, 1)} s`;
  panel.note.textContent = atEnd && !fullRun(panel.trial) ? `The run ended at ${number(panel.trial.metrics?.duration_s, 2)} s; its last sampled position (${number(panel.trace.last, 1)} s) is held. ${panel.trial.error || "This prefix does not meet the full-run criterion."}` :
    `${safeRun(panel.trial) ? "Complete safe recorded run" : fullRun(panel.trial) ? "Complete run; safety criterion not met" : "Incomplete recorded run"} · gold marks the powered hinge.`;
}
function drawChart() {
  const bounds = { left: 59, right: 979, top: 16, bottom: 181 };
  const series = ARMS.map((arm) => ({ arm, samples: panels[arm].trace?.samples.filter((sample) => finite(sample.cumulative_reward)) ?? [] }));
  const rewards = series.flatMap((item) => item.samples.map((sample) => sample.cumulative_reward)); const nodes = [];
  if (!rewards.length) {
    ui.chart.replaceChildren(svg("line", { x1: bounds.left, x2: bounds.right, y1: bounds.bottom, y2: bounds.bottom, stroke: "#cbbd9f" }),
      svg("text", { x: 500, y: 110, "text-anchor": "middle", fill: "#938266", "font-size": 13, "font-family": "Georgia, serif" }, "Recorded reward curves will appear with the movements."));
    chartMapping = null; return;
  }
  let minimum = Math.min(0, ...rewards); let maximum = Math.max(0, ...rewards);
  const margin = Math.max((maximum - minimum) * 0.08, 0.01); minimum -= margin; maximum += margin;
  const x = (value) => bounds.left + value / duration * (bounds.right - bounds.left);
  const y = (value) => bounds.bottom - (value - minimum) / (maximum - minimum) * (bounds.bottom - bounds.top);
  for (let tick = 0; tick <= 4; ++tick) {
    const value = minimum + (maximum - minimum) * tick / 4; const py = y(value); const seconds = duration * tick / 4;
    nodes.push(svg("line", { x1: bounds.left, x2: bounds.right, y1: py, y2: py, stroke: "#dbceb1", "stroke-dasharray": "2 5" }),
      svg("text", { x: bounds.left - 10, y: py + 4, "text-anchor": "end", fill: "#928164", "font-size": 10, "font-family": "Consolas, monospace" }, number(value, Math.abs(value) >= 10 ? 1 : 2)),
      svg("text", { x: x(seconds), y: bounds.bottom + 25, "text-anchor": "middle", fill: "#928164", "font-size": 10, "font-family": "Consolas, monospace" }, `${number(seconds, 0)} s`));
  }
  nodes.push(svg("line", { x1: bounds.left, x2: bounds.right, y1: y(0), y2: y(0), stroke: "#bcb190" }));
  for (const item of series) {
    if (!item.samples.length) continue;
    const path = item.samples.map((sample, index) => `${index ? "L" : "M"}${x(sample.time_s).toFixed(3)},${y(sample.cumulative_reward).toFixed(3)}`).join(" ");
    const curve = svg("path", { d: path, fill: "none", stroke: COLORS[item.arm], "stroke-width": 2.4, "stroke-dasharray": item.arm === "quiet" ? "8 5" : "", "stroke-linejoin": "round" });
    curve.append(svg("title", {}, `${item.arm === "quiet" ? "Quiet motor" : ui.chosenTitle.textContent} — cumulative native reward`)); nodes.push(curve);
  }
  const cursor = svg("line", { x1: x(time), x2: x(time), y1: bounds.top, y2: bounds.bottom, stroke: "#4b664b", "stroke-width": 1.2, "stroke-dasharray": "3 4" }); const dots = svg("g");
  nodes.push(cursor, dots); ui.chart.replaceChildren(...nodes);
  ui.chart.setAttribute("aria-label", "Cumulative native reward for quiet and the selected controller. The vertical cursor follows shared replay time; each curve ends at its last saved sample.");
  chartMapping = { x, y, cursor, dots }; updateChartCursor();
}
function updateChartCursor() {
  if (!chartMapping) return;
  const { x, y, cursor, dots } = chartMapping; cursor.setAttribute("x1", String(x(time))); cursor.setAttribute("x2", String(x(time)));
  const markers = [];
  for (const arm of ARMS) {
    const trace = panels[arm].trace; if (!trace) continue;
    const index = sampleAt(trace, time); if (index < 0) continue; const sample = trace.samples[index];
    markers.push(svg("circle", { cx: x(sample.time_s), cy: y(sample.cumulative_reward), r: 4, fill: COLORS[arm], stroke: "#fbf1dc", "stroke-width": 1.5 }));
  }
  dots.replaceChildren(...markers);
}
function updatePlayback() {
  const available = ARMS.some((arm) => panels[arm].trace); ui.playback.classList.toggle("has-recordings", available);
  ui.play.disabled = !available; ui.restart.disabled = !available; ui.timeline.disabled = !available;
  ui.time.textContent = `${number(time, 1)} / ${number(duration, 1)} s`; ui.timeline.value = String(time);
  ui.timeline.setAttribute("aria-valuetext", `${number(time, 1)} seconds of ${number(duration, 1)} seconds`);
  ui.play.textContent = playing ? "Pause together Ⅱ" : "Play together ▶"; ui.play.setAttribute("aria-pressed", String(playing));
  ui.play.setAttribute("aria-label", playing ? "Pause both saved simulations" : "Play both saved simulations");
  for (const arm of ARMS) renderPanel(arm); updateChartCursor();
}
function setPlaying(next) {
  playing = next; if (raf) window.cancelAnimationFrame(raf); raf = 0; previousAnimationTime = null;
  if (playing) raf = window.requestAnimationFrame(animate); updatePlayback();
}
function animate(timestamp) {
  if (!playing) return;
  if (previousAnimationTime !== null) time = Math.min(duration, time + Math.min((timestamp - previousAnimationTime) / 1000, 0.25) * Number(ui.speed.value));
  previousAnimationTime = timestamp;
  if (time >= duration) { setPlaying(false); announce("The shared replay reached its end. Restart to watch again."); return; }
  updatePlayback(); raf = window.requestAnimationFrame(animate);
}
ui.body.addEventListener("change", () => { updateDimensions(); void selectComparison(); });
ui.hinge.addEventListener("change", () => { const previousLamp = ui.lamp.value; updateDimensions(null); if ([...ui.lamp.options].some((item) => item.value === previousLamp)) ui.lamp.value = previousLamp; void selectComparison(); });
ui.lamp.addEventListener("change", () => { void selectComparison(); });
ui.controller.addEventListener("change", () => { void selectComparison(); });
ui.studyBody.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-record-case]"); if (!button || !record) return;
  const item = record.cases.find((candidate) => candidate.id === button.dataset.recordCase); if (!item) return;
  ui.body.value = bodyKey(item); updateDimensions(item);
  if (button.dataset.recordController) ui.controller.value = button.dataset.recordController;
  void selectComparison();
});
ui.retry.addEventListener("click", () => { void loadIndex(); });
ui.play.addEventListener("click", () => { if (time >= duration) time = 0; setPlaying(!playing); announce(playing ? "Playing saved movements together." : `Replay paused at ${number(time, 1)} seconds.`); });
ui.restart.addEventListener("click", () => { time = 0; setPlaying(false); announce("Both recordings are back at their first saved moment. Playback is paused."); });
ui.timeline.addEventListener("input", () => { time = Math.min(duration, Math.max(0, Number(ui.timeline.value))); setPlaying(false); });
document.addEventListener("visibilitychange", () => { if (document.hidden && playing) { setPlaying(false); announce("Replay paused while the page was hidden."); } });
window.addEventListener("resize", () => { for (const panel of Object.values(panels)) panel.rendered = ""; updatePlayback(); });
for (const arm of ARMS) drawWaiting(panels[arm], "Evidence preparing");
drawChart(); void loadIndex();
