"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const SVG_NS = "http://www.w3.org/2000/svg";
const ui = {
  scene: $("[data-scene]"), connection: $("[data-connection]"), connectionText: $("[data-connection-text]"), reconnect: $("[data-reconnect]"),
  bodyName: $("[data-body-name]"), status: $("[data-run-status]"), sceneNote: $("[data-scene-note]"), sceneState: $("[data-scene-state]"),
  time: $("[data-time]"), wake: $("[data-wake]"), pause: $("[data-pause]"), rebuild: $("[data-rebuild]"), runNote: $("[data-run-note]"),
  draft: $("[data-draft-status]"), brainStatus: $("[data-brain-status]"), policyChip: $("[data-policy-chip]"),
  message: $("[data-message]"), live: $("[data-live]"), light: $("[data-light]"), lightMeter: $("[data-light-meter]"), lightNote: $("[data-light-note]"),
  motion: $("[data-motion]"), effort: $("[data-effort]"), effortNote: $("[data-effort-note]"), sampleNote: $("[data-sample-note]"),
  chart: $("[data-light-chart]"), historyTime: $("[data-history-time]"), policyId: $("[data-policy-id]"), frozen: $("[data-frozen]"),
  counts: $("[data-counts]"), frames: $("[data-frames]"), checkpoint: $("[data-checkpoint]"), actor: $("[data-actor]"), spec: $("[data-spec]"),
};
const BODIES = { long_upper: "Long upper beam", long_lower: "Long lower beam", weighted: "A little weighted creature" };
const STATUSES = { ready: "Ready to wake", running: "Awake · moving", paused: "Paused", completed: "This run is complete", stopped: "Safety stop", error: "Movement stopped" };
const choiceButtons = $$("[data-body], [data-motor], [data-lamp]");
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const vector = (value, length = 3) => Array.isArray(value) && value.length === length && value.every(finite);
const number = (value, digits = 1) => finite(value) ? value.toFixed(digits) : "—";
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const validCase = (value) => value && Object.hasOwn(BODIES, value.body) && ["base", "elbow"].includes(value.motor) && ["left", "right"].includes(value.lamp);
const sameCase = (a, b) => validCase(a) && validCase(b) && ["body", "motor", "lamp"].every((field) => a[field] === b[field]);
const sha = (value) => typeof value === "string" && /^[a-f0-9]{64}$/i.test(value);
let state = null;
let draft = { body: "long_upper", motor: "base", lamp: "left" };
let connected = false;
let busy = false;
let polling = false;
let generation = 0;
let brainIdentity = null;
let sceneSignature = "";
let chartSignature = "";
let commandWarning = "";
let connectionProblem = "";
let announcementTimer;

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}
function announce(text) {
  window.clearTimeout(announcementTimer);
  ui.live.textContent = "";
  announcementTimer = window.setTimeout(() => { ui.live.textContent = text; }, 30);
}
function message(text = "") {
  if (ui.message.textContent !== text) ui.message.textContent = text;
  ui.message.hidden = !text;
}
function dirty() { return !!state && !sameCase(draft, state.case); }
function caseDescription(value) {
  return `${BODIES[value.body]} · motor ${value.motor === "base" ? "at the anchor" : "at the elbow"} · sunshine ${value.lamp}`;
}
function validateState(next) {
  if (next?.schema !== "pretrained_session_v1" || !Number.isSafeInteger(next.session_id) ||
      !Number.isSafeInteger(next.assembly_revision) || !validCase(next.case) || !Object.hasOwn(STATUSES, next.status) ||
      typeof next.running !== "boolean" || next.running !== (next.status === "running") ||
      !finite(next.elapsed_s) || next.elapsed_s < 0 || next.duration_s !== 64 ||
      !Array.isArray(next.physics?.geometry) || !next.physics.geometry.length ||
      !vector(next.physics.joints?.motor_m) || !vector(next.physics.joints?.passive_m) ||
      !vector(next.physics.sun?.position_m) || !Array.isArray(next.history) ||
      !Number.isSafeInteger(next.macro_decisions) || next.macro_decisions < 0 ||
      !Number.isSafeInteger(next.native_steps) || next.native_steps < 0) {
    throw new Error("The pretrained engine returned an incompatible state.");
  }
  const policy = next.policy;
  if (!policy || policy.source_seed !== 101 || policy.training !== "pooled" || policy.frozen !== true ||
      policy.parameters !== 17475 || !sha(policy.source_checkpoint_sha256) || !sha(policy.actor_sha256)) {
    throw new Error("The engine did not confirm the declared frozen pretrained policy.");
  }
  const identity = `${policy.source_checkpoint_sha256}:${policy.actor_sha256}`;
  if (brainIdentity && brainIdentity !== identity) throw new Error("The saved brain changed. Reconnect after checking the engine's policy.");
  return identity;
}
function acceptState(next) {
  const identity = validateState(next);
  const hadDraft = dirty();
  if (!state || (state.session_id !== next.session_id && !hadDraft)) draft = { ...next.case };
  state = next;
  brainIdentity = identity;
  connected = true;
  connectionProblem = "";
  update();
}
async function fetchJSON(path, options = {}, timeoutMs = 5000) {
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), timeoutMs);
  try {
    const response = await fetch(path, { cache: "no-store", ...options, signal: abort.signal });
    let next;
    try { next = await response.json(); } catch { throw new Error(`The engine returned an unreadable response (HTTP ${response.status}).`); }
    if (!response.ok) throw new Error(typeof next?.error === "string" ? next.error : `HTTP ${response.status}`);
    return next;
  } finally { window.clearTimeout(timeout); }
}
async function poll() {
  if (polling || busy || document.hidden) return;
  polling = true;
  const version = generation;
  try {
    const next = await fetchJSON("/api/pretrained/state");
    if (generation === version) acceptState(next);
  } catch (error) {
    if (generation === version) {
      connected = false;
      connectionProblem = `The workshop connection could not be confirmed: ${error.message}. It will keep checking; you can also try again.`;
      ui.connectionText.title = error.message;
      update();
    }
  } finally { polling = false; }
}
async function control(action) {
  if (busy || !connected || !state) return;
  if (action === "wake" && (dirty() || !["ready", "paused"].includes(state.status))) return;
  if (action === "pause" && !state.running) return;
  if (action === "rebuild" && state.running) return;
  const requestedCase = { ...draft };
  const expectedId = state.session_id;
  busy = true;
  generation += 1;
  commandWarning = "";
  update();
  try {
    const next = await fetchJSON("/api/pretrained/control", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, expected_session_id: expectedId, ...(action === "rebuild" ? requestedCase : {}) }),
    }, 10000);
    if (action === "rebuild" && (!sameCase(next.case, requestedCase) || next.status !== "ready" || next.session_id === expectedId)) {
      throw new Error("The engine did not confirm the requested fresh body.");
    }
    acceptState(next);
    announce(action === "rebuild" ? "Your rebuilt creature is ready. The same pretrained brain comes with it; recent sensor memory starts fresh." :
      action === "pause" ? next.status === "paused" ? "Paused. This body and its recent sensor history are preserved." : "The run has already ended. Its final body and readings remain visible." : "Your pretrained creature is awake.");
  } catch (error) {
    connected = false;
    commandWarning = `The command could not be confirmed: ${error.message}. Checking the current engine state; the command will not be retried automatically.`;
    announce("The command could not be confirmed. Waiting to check the current engine state.");
  } finally {
    busy = false;
    update();
    void poll();
  }
}
function reading(node, value, unit, digits = 1) {
  const nodes = [document.createTextNode(finite(value) ? number(value, digits) : "—")];
  if (unit) { const small = document.createElement("small"); small.textContent = ` ${unit}`; nodes.push(small); }
  node.replaceChildren(...nodes);
}
function update() {
  const changed = dirty();
  const active = state?.running === true;
  const ended = state && ["completed", "stopped", "error"].includes(state.status);
  ui.connection.dataset.state = connected ? "connected" : state ? "disconnected" : "waiting";
  ui.connectionText.textContent = connected ? "Native engine connected" : state ? "Engine link interrupted" : "Connecting to the engine";
  ui.reconnect.hidden = !connectionProblem || connected;
  ui.reconnect.disabled = busy;
  if (connected) ui.connectionText.removeAttribute("title");
  ui.bodyName.textContent = state ? BODIES[state.case.body] : "Your creature";
  ui.status.textContent = busy ? "Applying your choice…" : state ? STATUSES[state.status] : "Waiting for the engine";
  ui.sceneNote.textContent = state ? `${state.case.motor === "base" ? "Motor at the anchor" : "Motor at the elbow"} · sunshine ${state.case.lamp}` : "A little creature, ready to wake";
  ui.time.textContent = `${number(state?.elapsed_s ?? 0)} / 64 s`;
  ui.wake.disabled = !connected || busy || changed || !state || !["ready", "paused"].includes(state.status);
  ui.wake.textContent = state?.status === "paused" ? "Keep going ↗" : "Wake this creature ↗";
  ui.pause.disabled = !connected || busy || !active;
  ui.rebuild.disabled = !connected || busy || active;
  choiceButtons.forEach((button) => {
    const field = button.hasAttribute("data-body") ? "body" : button.hasAttribute("data-motor") ? "motor" : "lamp";
    button.setAttribute("aria-pressed", String(button.dataset[field] === draft[field]));
    button.disabled = !connected || busy;
  });
  ui.draft.dataset.dirty = String(changed);
  ui.draft.textContent = !state ? "Waiting to confirm this body with the engine." : changed ? active ? "A new body is drafted. Pause, then press Rebuild to apply your choices." : "A new body is drafted. Press Rebuild to apply these choices, then wake it." : "These choices match the creature on the bench.";
  ui.runNote.textContent = !connected ? state ? "The last received body stays visible. Controls return when the engine reconnects." : "Waiting for native physics and the saved brain." :
    busy ? "Waiting for the engine to confirm your command." : changed ? `Your draft: ${caseDescription(draft)}. ${active ? "Pause, then rebuild to start fresh." : "Rebuild starts a fresh run."}` :
    ended ? "This run has ended. Rebuild to start fresh with the same saved brain." : active ? "Its saved brain chooses from local feedback. Pause to inspect; rebuild to start fresh." :
    state?.status === "paused" ? "Paused at this moment. Keep going preserves its body and recent sensor memory." : "Ready from the start. Wake it to use its pretrained brain.";
  ui.sceneState.hidden = !!state && connected;
  ui.sceneState.textContent = state ? "Engine link interrupted · last received body" : "Connecting to native physics…";
  ui.brainStatus.textContent = state ? "Pretrained brain loaded · no training here" : "Preparing its pretrained brain";
  ui.policyChip.textContent = state ? "Same frozen brain" : "Simulation pretrained";
  const faults = state?.status === "stopped" && Array.isArray(state.physics.actuator?.fault_flags) ? state.physics.actuator.fault_flags : [];
  message(commandWarning || connectionProblem || (typeof state?.error === "string" && state.error ? state.error : faults.length ? `The native engine stopped this run after motor feedback: ${faults.map(String).join(", ")}. Its last body and sensor readings remain visible.` : ""));
  updateSensors();
  if (state) {
    ui.policyId.textContent = String(state.policy.id ?? "Pooled pretrained policy · seed 101");
    ui.frozen.textContent = "Fixed · 17,475 parameters · zero learning updates";
    ui.counts.textContent = `${state.macro_decisions} / ${state.native_steps}`;
    ui.frames.textContent = Number.isSafeInteger(state.history_frames) ? String(state.history_frames) : "—";
    ui.checkpoint.textContent = state.policy.source_checkpoint_sha256;
    ui.actor.textContent = state.policy.actor_sha256;
  }
  drawScene();
  drawHistory();
}
function updateSensors() {
  const light = state?.physics.light_sensor;
  const hasLight = light?.valid === true && finite(light.observations?.illuminance_lux);
  const lux = hasLight ? light.observations.illuminance_lux : null;
  const sensor = state?.physics.sensor;
  const gyro = sensor?.valid === true && vector(sensor.observations?.angular_velocity_rad_s) ? sensor.observations.angular_velocity_rad_s : null;
  const history = state?.history ?? [];
  const last = history.at(-1);
  const effort = finite(state?.previous_effort) ? state.previous_effort : finite(state?.physics.command) ? state.physics.command : finite(last?.effort) ? last.effort : null;
  reading(ui.light, lux, "lx", 0);
  reading(ui.motion, gyro ? Math.hypot(...gyro) : null, "rad/s", 2);
  ui.effort.textContent = finite(effort) ? `${effort > 0 ? "+" : ""}${number(effort, 2)}` : "—";
  ui.effortNote.textContent = state?.macro_decisions ? effort > 0 ? "Positive effort at this moment." : effort < 0 ? "Negative effort at this moment." : "The motor is quiet at this moment." : "Its first choice is still ahead.";
  ui.lightMeter.hidden = !hasLight;
  if (hasLight) { ui.lightMeter.value = clamp(lux, 0, 1000); ui.lightMeter.textContent = `${number(lux, 0)} lx`; }
  const first = history.find((sample) => sample.light_valid === true && finite(sample.illuminance_lux));
  if (!hasLight) ui.lightNote.textContent = "Its first delivered sample is still ahead.";
  else if (first) {
    const change = lux - first.illuminance_lux;
    ui.lightNote.textContent = Math.abs(change) < 5 ? "Near this run's first measured brightness." : `${number(Math.abs(change), 0)} lx ${change > 0 ? "brighter" : "dimmer"} than its first measurement.`;
  } else ui.lightNote.textContent = "Brightness measured on its receiving face.";
  ui.sampleNote.textContent = !state ? "Waiting for the first delivered sensor readings." : !hasLight || !gyro ? "Waiting for delivered light and motion samples. Missing samples are shown as —." :
    `Delivered local readings · ${finite(light.age_s) ? `${number(light.age_s, 2)} s light-sample age` : "sample timing from native physics"}.`;
}
function drawScene() {
  const width = Math.max(240, Math.round(ui.scene.clientWidth || 800));
  const height = Math.max(240, Math.round(ui.scene.clientHeight || 490));
  const physics = state?.physics;
  const key = JSON.stringify([physics?.geometry, physics?.joints, physics?.sun, physics?.light_sensor, width, height]);
  if (key === sceneSignature) return;
  sceneSignature = key;
  ui.scene.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const nodes = [svg("title", { id: "pretrained-scene-title" }, state ? `${BODIES[state.case.body]} at ${number(state.elapsed_s)} simulated seconds` : "Your pretrained creature"),
    svg("desc", { id: "pretrained-scene-description" }, "An anchored two-link creature drawn from native physical coordinates. Gold marks its powered hinge. The lamp and the receiving direction of its light-sensitive face are visible.")];
  for (let x = 18; x < width; x += 27) for (let y = 54; y < height - 48; y += 27) nodes.push(svg("circle", { cx: x, cy: y, r: 0.65, fill: "#567956", opacity: .32 }));
  if (!physics) { ui.scene.replaceChildren(...nodes); return; }
  const lengths = physics.assembly?.segments;
  const reach = vector(lengths, 2) ? lengths.reduce((sum, length) => sum + length, 0) * .04 + .07 : .47;
  const source = physics.sun.position_m;
  const minX = Math.min(-reach, source[0] - .08), maxX = Math.max(reach, source[0] + .08);
  const minZ = Math.min(-reach, source[2] - .08), maxZ = Math.max(reach, source[2] + .08);
  const scale = Math.min((width - 52) / (maxX - minX), (height - 112) / (maxZ - minZ));
  const origin = [(width - (maxX + minX) * scale) / 2, 56 + maxZ * scale];
  const project = (point) => [origin[0] + point[0] * scale, origin[1] - point[2] * scale];
  const [anchorX, anchorY] = project([0, 0, 0]);
  nodes.push(svg("line", { x1: anchorX - 21, x2: anchorX + 21, y1: anchorY - 20, y2: anchorY - 20, stroke: "#9cae8c", "stroke-width": 4, "stroke-linecap": "round" }),
    svg("line", { x1: anchorX, x2: anchorX, y1: anchorY - 20, y2: anchorY, stroke: "#819978", "stroke-width": 4 }));
  const eye = physics.light_sensor;
  if (vector(eye?.position_m)) {
    const [x, y] = project(eye.position_m), [sunX, sunY] = project(source);
    nodes.push(svg("line", { x1: x, y1: y, x2: sunX, y2: sunY, stroke: "#d6bc66", opacity: .2, "stroke-width": 1, "stroke-dasharray": "3 7" }));
  }
  for (const part of physics.geometry) {
    if (!vector(part.position_m) || !vector(part.size_m) || !vector(part.quaternion, 4) || part.size_m.some((size) => size < 0)) continue;
    const [x, y] = project(part.position_m), [qw, qx, qy, qz] = part.quaternion;
    const angle = Math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy * qy + qz * qz)) * 180 / Math.PI;
    const w = Math.max(2, part.size_m[0] * scale), h = Math.max(2, part.size_m[2] * scale);
    const color = part.kind === "light" ? "#d9c075" : part.kind === "sensor" ? "#96af71" : part.kind === "block" ? "#bd916e" : part.segment === 0 ? "#70a38e" : "#8fa3c0";
    const group = svg("g", { transform: `translate(${x} ${y}) rotate(${angle})` });
    group.append(svg("rect", { x: -w / 2 + .6, y: -h / 2 + .6, width: Math.max(1, w - 1.2), height: Math.max(1, h - 1.2), rx: Math.min(3, w / 6), fill: color, stroke: "#d9e0bd", "stroke-opacity": .65, "stroke-width": .85 }),
      svg("circle", { cx: 0, cy: 0, r: Math.min(3, w * .14), fill: part.kind === "light" ? "#526547" : "#41624a", opacity: .6 }));
    nodes.push(group);
  }
  for (const [name, radius, fill] of [["passive_m", 5, "#dddcbb"], ["motor_m", 9, "#e4be70"]]) {
    const [x, y] = project(physics.joints[name]);
    nodes.push(svg("circle", { cx: x, cy: y, r: radius, fill, stroke: "#213d2c", "stroke-width": 2 }),
      svg("circle", { cx: x, cy: y, r: name === "motor_m" ? 2.8 : 1.6, fill: "#627250" }));
  }
  if (vector(eye?.position_m) && vector(eye?.direction_m)) {
    const [x, y] = project(eye.position_m), end = eye.position_m.map((value, axis) => value + eye.direction_m[axis] * .055), [endX, endY] = project(end);
    const brightness = eye.valid === true && finite(eye.observations?.illuminance_lux) ? clamp(eye.observations.illuminance_lux / 1000, 0, 1) : 0;
    nodes.push(svg("circle", { cx: x, cy: y, r: 11 + brightness * 17, fill: "#e9d47a", opacity: .08 + brightness * .18 }),
      svg("line", { x1: x, y1: y, x2: endX, y2: endY, stroke: "#edda91", "stroke-width": 2, "stroke-linecap": "round" }),
      svg("circle", { cx: x, cy: y, r: 2.6, fill: "#f0dd95" }), svg("circle", { cx: endX, cy: endY, r: 1.8, fill: "#f0dd95" }));
  }
  const [sunX, sunY] = project(source);
  const sun = svg("g", { "aria-label": "The sunshine" });
  sun.append(svg("circle", { cx: sunX, cy: sunY, r: 31, fill: "#f0d077", opacity: .045 }), svg("circle", { cx: sunX, cy: sunY, r: 23, fill: "#f0d077", opacity: .07 }));
  for (let ray = 0; ray < 12; ray++) { const angle = ray * Math.PI / 6; sun.append(svg("line", { x1: sunX + Math.cos(angle) * 17, y1: sunY + Math.sin(angle) * 17, x2: sunX + Math.cos(angle) * 24, y2: sunY + Math.sin(angle) * 24, stroke: "#d6bd70", "stroke-width": 1.4, "stroke-linecap": "round" })); }
  sun.append(svg("circle", { cx: sunX, cy: sunY, r: 11, fill: "#dfc077", stroke: "#f5dda0", "stroke-width": 1.4 }));
  nodes.push(sun);
  ui.scene.replaceChildren(...nodes);
}
function drawHistory() {
  ui.historyTime.textContent = `${number(state?.elapsed_s ?? 0)} / 64 s`;
  const history = state?.history ?? [];
  const key = JSON.stringify([state?.session_id, history]);
  if (key === chartSignature) return;
  chartSignature = key;
  const x = (time) => 38 + clamp(time, 0, 64) / 64 * 849;
  const y = (lux) => 87 - clamp(lux, 0, 1000) / 1000 * 73;
  const nodes = [svg("title", {}, "Delivered light over this run"), svg("desc", {}, "Illuminance on a fixed zero to one thousand lux scale, across zero to sixty-four simulated seconds. Gaps indicate missing samples.")];
  for (const lux of [0, 500, 1000]) nodes.push(svg("line", { x1: 38, x2: 887, y1: y(lux), y2: y(lux), stroke: "#bbb991", "stroke-width": .7, "stroke-dasharray": lux ? "2 6" : "" }),
    svg("text", { x: 28, y: y(lux) + 3, "text-anchor": "end", fill: "#98906e", "font-size": 8, "font-family": "Consolas, monospace" }, String(lux)));
  let path = "", drawing = false, valid = 0;
  for (const sample of history) {
    if (sample.light_valid !== true || !finite(sample.time_s) || !finite(sample.illuminance_lux)) { drawing = false; continue; }
    path += `${drawing ? "L" : "M"}${x(sample.time_s).toFixed(2)},${y(sample.illuminance_lux).toFixed(2)} `;
    drawing = true; valid += 1;
  }
  if (valid) {
    nodes.push(svg("path", { d: path, fill: "none", stroke: "#8c9557", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    const last = history.findLast((sample) => sample.light_valid === true && finite(sample.time_s) && finite(sample.illuminance_lux));
    nodes.push(svg("circle", { cx: x(last.time_s), cy: y(last.illuminance_lux), r: 2.7, fill: "#7a8950" }));
  } else nodes.push(svg("text", { x: 460, y: 49, "text-anchor": "middle", fill: "#959071", "font-size": 12, "font-family": "Georgia, serif" }, "Its first delivered light sample is still ahead."));
  for (const seconds of [0, 16, 32, 48, 64]) nodes.push(svg("text", { x: x(seconds), y: 105, "text-anchor": "middle", fill: "#96906d", "font-size": 8, "font-family": "Consolas, monospace" }, `${seconds} s`));
  ui.chart.replaceChildren(...nodes);
  ui.chart.setAttribute("aria-label", valid ? `Delivered light in this run, ${valid} valid samples on a fixed zero to one thousand lux scale.` : "Delivered light over this run; no valid samples yet.");
}
choiceButtons.forEach((button) => button.addEventListener("click", () => {
  if (!connected || busy) return;
  const field = button.hasAttribute("data-body") ? "body" : button.hasAttribute("data-motor") ? "motor" : "lamp";
  if (draft[field] === button.dataset[field]) return;
  draft = { ...draft, [field]: button.dataset[field] };
  update();
  announce(dirty() ? state.running ? "Your new choices are drafted. Pause, then press Rebuild to apply them." : "Your new choices are drafted. Press Rebuild to apply them." : "Your choices match the creature on the bench.");
}));
ui.wake.addEventListener("click", () => { void control("wake"); });
ui.pause.addEventListener("click", () => { void control("pause"); });
ui.rebuild.addEventListener("click", () => { void control("rebuild"); });
ui.reconnect.addEventListener("click", () => { void poll(); });
new ResizeObserver(() => { sceneSignature = ""; drawScene(); }).observe(ui.scene.parentElement);
document.addEventListener("visibilitychange", () => { if (!document.hidden) void poll(); });
update();
void poll();
window.setInterval(() => { void poll(); }, 150);
fetchJSON("/api/pretrained/spec").then((value) => { ui.spec.textContent = JSON.stringify(value, null, 2); }).catch((error) => {
  ui.spec.textContent = `The optional simulation specification could not be read: ${error.message}`;
});
