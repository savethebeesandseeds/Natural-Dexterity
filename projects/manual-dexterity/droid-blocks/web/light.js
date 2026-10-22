"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const ui = {
  scene: $("#light-scene"), name: $("[data-creature-name]"), status: $("[data-run-status]"), connection: $("[data-connection]"),
  focus: $("[data-focus]"), motionLux: $("[data-motion-lux]"),
  time: $("[data-time]"), sceneStatus: $("[data-scene-status]"), runNote: $("[data-run-note]"), start: $("[data-start]"), pause: $("[data-pause]"), reset: $("[data-reset]"),
  import: $("[data-import]"), segment: $("[data-segment]"), slot: $("[data-slot]"), side: $("[data-side]"), face: $("[data-face]"), attach: $("[data-attach]"),
  message: $("[data-message]"), live: $("[data-live]"), lux: $("[data-lux]"), sampleNote: $("[data-sample-note]"), chart: $("[data-chart]"), chartStart: $("[data-chart-start]"), chartEnd: $("[data-chart-end]"),
  decision: $("[data-decision]"), decisions: $("[data-decisions]"), memory: $("[data-memory]"), memoryNote: $("[data-memory-note]"), modeNote: $("[data-mode-note]"),
  algorithm: $("[data-algorithm]"), timing: $("[data-light-timing]"), controller: $("[data-controller-id]"), effort: $("[data-effort]"), sunPosition: $("[data-sun-position]"), direction: $("[data-face-direction]"), updates: $("[data-updates]"),
};
const SVG_NS = "http://www.w3.org/2000/svg";
const MODES = { learner: "Learn from light", zero: "Quiet motor", rhythm: "Following a rhythm", random: "Random moves" };
const SUN_PRESETS = { left: [-0.3, 0, 0.2], right: [0.3, 0, 0.2], above: [0, 0, 0.4], below: [0, 0, -0.55] };
const num = (value, fallback = 0) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
const fixed = (value, digits = 2) => num(value).toFixed(digits);
const copy = (value) => JSON.parse(JSON.stringify(value));
const vec = (value) => Array.isArray(value) && value.length === 3 && value.every((item) => typeof item === "number" && Number.isFinite(item));
const canonical = (value) => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value;
const identity = (value) => JSON.stringify(canonical(value));
let state = null;
let spec = null;
let connected = false;
let busy = false;
let polling = false;
let generation = 0;
let chosenMode = "learner";
let modeChosen = false;
let mountSignature = "";
let sceneSignature = "";
let chartSignature = "";
let plotMapping = null;
let dragging = null;
let dragPoint = null;
let announceTimer = null;
let focusView = false;
let workshopScroll = 0;

function setFocusView(expanded) {
  if (focusView === expanded) return;
  if (expanded) workshopScroll = window.scrollY;
  focusView = expanded;
  document.body.classList.toggle("motion-focus", expanded);
  ui.focus.setAttribute("aria-pressed", String(expanded));
  ui.focus.textContent = expanded ? "Back to workshop" : "Expand view";
  sceneSignature = "";
  drawScene();
  window.scrollTo({ top: expanded ? 0 : workshopScroll, behavior: "auto" });
  ui.focus.focus({ preventScroll: true });
  announce(expanded ? "Expanded movement view. Press Escape or Back to workshop to return." : "Workshop view restored.");
}

function announce(text) {
  window.clearTimeout(announceTimer); ui.live.textContent = "";
  announceTimer = window.setTimeout(() => { ui.live.textContent = text; }, 30);
}
function message(text = "") {
  if (ui.message.textContent !== text) ui.message.textContent = text;
  ui.message.hidden = !text;
}
function running() { return state?.status === "running"; }
function canChangeBody() { return connected && !busy && !running(); }
function bodyName() { return state?.assembly?.name || "Its place in the sunshine"; }

function fillSlots(segment, selected = 0) {
  const length = state?.assembly?.segments?.[segment] ?? 3;
  ui.slot.replaceChildren(...Array.from({ length }, (_, index) => {
    const option = document.createElement("option"); option.value = String(index); option.textContent = String(index + 1); return option;
  }));
  ui.slot.value = String(Math.min(Math.max(0, selected), length - 1));
}

function update() {
  const status = state?.status;
  const active = running();
  const learner = state?.mode === "learner";
  const diagnostics = learner ? state?.learner : null;
  const statusText = { ready: "Ready to try", running: "Trying", paused: "Paused", completed: "This try is complete", error: "Movement stopped" };
  ui.name.textContent = bodyName();
  ui.status.textContent = state ? `${MODES[state.mode] ?? state.mode} · ${statusText[status] ?? status}` : "Waiting for a body";
  ui.time.textContent = `${fixed(state?.elapsed_s, 1)} / ${fixed(state?.duration_s ?? 120, 0)} s`;
  ui.sceneStatus.textContent = dragging ? "Move the sunshine · release to place" : !state ? "Waiting for the workshop" : active ? "Body in motion" : status === "error" ? "Stopped · last received body" : "Paused body";
  ui.start.disabled = !connected || busy || active;
  ui.start.textContent = status === "paused" && state?.mode === chosenMode ? "Keep going ↗" : "Let it try ↗";
  ui.pause.disabled = !connected || busy || !active;
  ui.reset.disabled = !connected || busy;
  ui.import.disabled = !canChangeBody();
  [ui.segment, ui.slot, ui.side, ui.face, ui.attach].forEach((control) => { control.disabled = !canChangeBody(); });
  $$("[data-sun]").forEach((button) => { button.disabled = !connected || busy; });
  $$("[data-mode]").forEach((button) => {
    button.disabled = busy;
    button.setAttribute("aria-pressed", String(button.dataset.mode === chosenMode));
  });
  ui.modeNote.textContent = chosenMode !== state?.mode && state ? `${MODES[chosenMode]} is selected for a fresh try. ${active ? "Pause, then wake it to begin." : "Wake it to begin."}` : "Choose a kind of try, then wake it. Changing the kind starts again.";
  ui.runNote.textContent = status === "error" ? "This try stopped. Its readings and memory remain visible. Start again for a fresh try." : status === "paused" ? "Paused at this moment. Keep going to continue with the same memory, or start again." : status === "completed" ? "Two minutes of trying and noticing. Move the face or sunshine, then begin another try." : "A new try begins with fresh memory. Move the sun to give it something new to notice.";
  if (!connected) ui.runNote.textContent = state ? "The connection paused. This is the last state the workshop received." : "The workshop is waiting for the local engine.";
  const assemblyKey = identity(state?.assembly);
  if (state?.assembly?.light && assemblyKey !== mountSignature) {
    mountSignature = assemblyKey;
    const light = state.assembly.light;
    ui.segment.value = String(light.segment); fillSlots(light.segment, light.slot);
    ui.side.value = String(light.side); ui.face.value = String(light.face);
  }
  const light = state?.physics?.light_sensor;
  const value = light?.valid === true && typeof light?.observations?.illuminance_lux === "number" ? light.observations.illuminance_lux : null;
  ui.lux.textContent = value === null ? "— lx" : `${fixed(value, 0)} lx`;
  ui.motionLux.textContent = `Light · ${ui.lux.textContent}`;
  ui.sampleNote.textContent = value === null ? "Waiting for its first delivered sample" : light.observations?.saturated ? "Its reading has reached the sensor’s limit" : active ? "Measured at the attached face" : "Last sample · body paused";
  const action = diagnostics?.action_id ?? diagnostics?.action_label;
  const actions = { negative: "Trying a little − effort.", coast: "Letting the body coast.", positive: "Trying a little + effort." };
  ui.decision.textContent = !state ? "Its first move is still ahead." : !learner ? state.mode === "zero" ? "The motor is quiet. Gravity can still move the body." : state.mode === "rhythm" ? "It follows the same set rhythm, without changing it from experience." : "It samples moves without learning from their results." : status === "ready" ? "Its first move is still ahead." : status === "error" ? "The trial stopped. These are its last recorded choices." : status === "paused" ? `Paused. ${actions[action]?.replace("Trying", "Last choice:").replace("Letting", "Last choice: letting") ?? "Its last choice is kept."}` : status === "completed" ? "This try has ended. Its last choices and measurements are kept here." : diagnostics?.phase === "warming" ? "Waiting for its first sensor readings." : actions[action] ?? "Choosing a short move from what it has experienced.";
  ui.decisions.textContent = learner && diagnostics ? String(num(diagnostics.decision_count)) : "—";
  ui.memory.textContent = learner && diagnostics ? String(num(diagnostics.history_size)) : "—";
  ui.memoryNote.textContent = learner ? "It remembers recent moments and uses them to choose what to try next. Moving the sunshine keeps this memory." : "This comparison does not learn. Try it on the same body and notice how its movements feel different.";
  ui.controller.textContent = learner ? diagnostics?.learner_id ?? diagnostics?.algorithm_id ?? "Waiting for learner details" : state ? MODES[state.mode] ?? state.mode : "—";
  ui.effort.textContent = state ? fixed(learner ? diagnostics?.effort : state.physics?.command, 3) : "—";
  ui.updates.textContent = learner && diagnostics ? String(num(diagnostics.update_count ?? diagnostics.updates)) : "Not learning";
  ui.sunPosition.textContent = vec(state?.physics?.sun?.position_m) ? [state.physics.sun.position_m[0], state.physics.sun.position_m[2]].map((value) => fixed(value, 3)).join(", ") : "—";
  ui.direction.textContent = vec(light?.direction_m) ? light.direction_m.map((value) => fixed(value, 3)).join(", ") : "—";
  ui.algorithm.textContent = learner ? "A linear Expected SARSA learner updates estimates for negative effort, coasting and positive effort from sensor experience. Its learned values estimate future summed sensor reward, not a sun angle or a known best pose. Learning belongs to this try; reset, a fresh start or a rebuilt body clears it." : "This mode uses a fixed comparison controller. It does not update learned action values.";
  if (light) ui.timing.textContent = `Light samples arrive every ${fixed(num(light.sample_period_s) * 1000, 0)} ms with ${fixed(num(light.latency_s) * 1000, 0)} ms delivery delay. A source move keeps already acquired samples in flight. Distance and the receiving face’s angle both affect the reading.`;
  drawScene(); drawChart();
}

function accept(next) {
  if (next?.schema !== "light_session_v1" || !next.assembly || !next.physics) throw new Error("The light workshop returned an incompatible state.");
  state = next; connected = true;
  if (!modeChosen) chosenMode = next.mode ?? "learner";
  ui.connection.dataset.state = "connected";
  ui.connection.replaceChildren(Object.assign(document.createElement("i"), { ariaHidden: "true" }), document.createTextNode("Workshop connected"));
  if (next.error) message(next.error);
  update();
}

async function fetchJSON(url, options = {}) {
  const abort = new AbortController(); const timeout = window.setTimeout(() => abort.abort(), 8000);
  try {
    const response = await fetch(url, { cache: "no-store", ...options, signal: abort.signal });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
  } finally { window.clearTimeout(timeout); }
}

async function control(action, fields = {}, success = "") {
  if (!connected || busy) return;
  busy = true; generation += 1; message(); update();
  try {
    const next = await fetchJSON("/api/light/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, ...fields }) });
    accept(next);
    if (success && next.status !== "error") announce(success);
  } catch (error) { message(`The command could not be confirmed: ${error.message}. Checking the workshop state again.`); }
  finally { busy = false; dragPoint = null; update(); }
}

async function poll() {
  if (polling || busy || document.hidden) return;
  polling = true; const current = generation;
  try {
    const next = await fetchJSON("/api/light/state");
    if (current === generation) accept(next);
  } catch {
    if (current === generation) {
      connected = false; ui.connection.dataset.state = "disconnected";
      ui.connection.replaceChildren(Object.assign(document.createElement("i"), { ariaHidden: "true" }), document.createTextNode("Workshop disconnected"));
      update();
    }
  } finally { polling = false; }
}

function socketOccupied(assembly, mount) {
  return [...(assembly.blocks ?? []), assembly.sensor].some((part) => part && part.segment === mount.segment && part.slot === mount.slot && part.side === mount.side);
}

async function importWorkshopBody() {
  if (!canChangeBody()) return;
  busy = true; generation += 1; message(); update();
  try {
    const workshop = await fetchJSON("/api/construction/state");
    const original = workshop.assembly;
    if (!original || !["construction_kit_v1", "construction_kit_v2"].includes(original.schema) || !Array.isArray(original.segments)) throw new Error("The building bench has no compatible body.");
    const assembly = copy(original); assembly.schema = "construction_kit_v2";
    if (!assembly.light) {
      let mount = null;
      const imu = assembly.sensor;
      for (const segment of [imu.segment, 1 - imu.segment]) {
        const center = Math.min(imu.slot, assembly.segments[segment] - 1);
        for (const offset of [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]) {
          const slot = center + offset;
          if (slot < 0 || slot >= assembly.segments[segment]) continue;
          for (const side of [-imu.side, imu.side]) {
            const candidate = { segment, slot, side, face: 1 };
            if (!mount && !socketOccupied(assembly, candidate)) mount = candidate;
          }
        }
      }
      if (!mount) throw new Error("There is no empty socket for a light sensor. Free one on the building bench first.");
      assembly.light = mount;
    }
    const next = await fetchJSON("/api/light/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "build", assembly }) });
    accept(next);
    if (next.status !== "error") announce("A copy of your workshop body is ready here with its light sensor. Your original workshop body is unchanged.");
  } catch (error) { message(`The workshop body could not be brought here: ${error.message}`); }
  finally { busy = false; update(); }
}

function moveSun(position) {
  if (!vec(position)) return;
  const bounded = [Math.max(-2, Math.min(2, position[0])), 0, Math.max(-2, Math.min(2, position[2]))];
  return control("sun", { sun: { position_m: bounded, intensity_lux: num(state?.physics?.sun?.intensity_lux, 1000) } }, "Sunshine moved. The same body and memory continue.");
}

function svg(tag, attributes = {}, text) {
  const element = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  if (text !== undefined) element.textContent = text;
  return element;
}

function drawScene() {
  const width = Math.max(300, ui.scene.clientWidth || 750);
  const height = Math.max(240, ui.scene.clientHeight || 470);
  const physics = state?.physics;
  const key = JSON.stringify([physics?.geometry, physics?.joints, physics?.sun, physics?.light_sensor, dragPoint, width, height, running(), connected, busy]);
  if (key === sceneSignature) return;
  sceneSignature = key; ui.scene.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const source = physics?.sun?.position_m ?? [0.3, 0, 0.2];
  const reach = (state?.assembly?.segments ?? [3, 3]).reduce((sum, value) => sum + num(value), 0) * 0.04 + 0.07;
  const minX = Math.min(-reach, num(source[0]) - 0.07); const maxX = Math.max(reach, num(source[0]) + 0.07);
  const minZ = Math.min(-reach, num(source[2]) - 0.06); const maxZ = Math.max(reach * 0.45, num(source[2]) + 0.07);
  const scale = Math.min((width - 70) / (maxX - minX), (height - 100) / (maxZ - minZ));
  const origin = [(width - (maxX + minX) * scale) / 2, 55 + maxZ * scale];
  plotMapping = { width, height, scale, origin };
  const project = (point) => [origin[0] + num(point?.[0]) * scale, origin[1] - num(point?.[2]) * scale];
  const nodes = [svg("title", { id: "light-scene-title" }, `${bodyName()} and the sunshine`), svg("desc", { id: "light-scene-desc" }, "Native body positions and the direction of the attached receiving face. Focus the sun and use arrow keys to move it. Moving sunshine preserves this try’s body and memory.")];
  for (let x = 18; x < width; x += 28) for (let y = 47; y < height - 25; y += 28) nodes.push(svg("circle", { cx: x, cy: y, r: 0.6, fill: "#69856a40" }));
  if (!physics) {
    nodes.push(svg("text", { x: width / 2, y: height / 2, fill: "#97ad91", "text-anchor": "middle", "font-size": 14 }, "Waiting for your creature…"));
    ui.scene.replaceChildren(...nodes); return;
  }
  const sourcePoint = project(source);
  const light = physics.light_sensor;
  if (vec(light?.position_m)) {
    const lightPoint = project(light.position_m);
    nodes.push(svg("line", { x1: sourcePoint[0], y1: sourcePoint[1], x2: lightPoint[0], y2: lightPoint[1], stroke: "#d7be643d", "stroke-width": 1, "stroke-dasharray": "3 7" }));
  }
  const motor = project(physics.joints?.motor_m);
  nodes.push(svg("path", { d: `M ${motor[0] - 40} ${motor[1] - 20} h 80`, stroke: "#8d9a78", "stroke-width": 3, "stroke-linecap": "round" }));
  nodes.push(svg("line", { x1: motor[0], x2: motor[0], y1: motor[1] - 20, y2: motor[1], stroke: "#819174", "stroke-width": 5 }));
  for (const part of physics.geometry ?? []) {
    const point = project(part.position_m); const q = part.quaternion ?? [1, 0, 0, 0];
    const angle = Math.atan2(2 * (num(q[0], 1) * num(q[2]) + num(q[1]) * num(q[3])), 1 - 2 * (num(q[2]) ** 2 + num(q[3]) ** 2)) * 180 / Math.PI;
    const w = num(part.size_m?.[0], 0.04) * scale; const h = num(part.size_m?.[2], 0.04) * scale;
    const group = svg("g", { transform: `translate(${point[0]},${point[1]}) rotate(${angle})` });
    const color = part.kind === "light" ? "#dcc27b" : part.kind === "sensor" ? "#8eaa6a" : part.kind === "block" ? "#bf926d" : part.segment === 0 ? "#669885" : "#8999af";
    group.append(svg("rect", { x: -w / 2 + 0.7, y: -h / 2 + 0.7, width: Math.max(1, w - 1.4), height: Math.max(1, h - 1.4), rx: Math.min(4, w / 7), fill: color, stroke: "#bdc8a6", "stroke-width": 0.8 }));
    group.append(svg("circle", { cx: 0, cy: 0, r: w * (part.kind === "light" ? 0.23 : 0.12), fill: part.kind === "light" ? "#556542" : "#36504455", stroke: "#dfdfb67a", "stroke-width": 0.7 }));
    nodes.push(group);
  }
  const elbow = project(physics.joints?.passive_m);
  nodes.push(svg("circle", { cx: elbow[0], cy: elbow[1], r: 5.5, fill: "#244b3b", stroke: "#c6cea2", "stroke-width": 1.5 }));
  nodes.push(svg("circle", { cx: motor[0], cy: motor[1], r: 10, fill: "#c7ab68", stroke: "#e7d197", "stroke-width": 2 }));
  if (vec(light?.position_m) && vec(light?.direction_m)) {
    const start = project(light.position_m); const end = project(light.position_m.map((value, index) => value + light.direction_m[index] * 0.055));
    const dx = end[0] - start[0]; const dy = end[1] - start[1]; const len = Math.hypot(dx, dy) || 1;
    const brightness = light.valid ? Math.max(0, Math.min(1, num(light.observations?.illuminance_lux) / 1000)) : 0;
    nodes.push(svg("circle", { cx: start[0], cy: start[1], r: 10 + brightness * 13, fill: "#e8d379", opacity: 0.08 + brightness * 0.2 }));
    nodes.push(svg("line", { x1: start[0], y1: start[1], x2: end[0], y2: end[1], stroke: "#ecd38a", "stroke-width": 2 }));
    nodes.push(svg("path", { d: `M ${end[0] - dx / len * 5 - dy / len * 3} ${end[1] - dy / len * 5 + dx / len * 3} L ${end[0]} ${end[1]} L ${end[0] - dx / len * 5 + dy / len * 3} ${end[1] - dy / len * 5 - dx / len * 3}`, fill: "none", stroke: "#ecd38a", "stroke-width": 1.5 }));
    if (!running()) nodes.push(svg("text", { x: Math.max(55, Math.min(width - 70, start[0] + 12)), y: Math.min(height - 35, start[1] + 28), fill: "#c3c395", "font-size": 10 }, "receiving face"));
  }
  nodes.push(drawSun(sourcePoint, false));
  if (dragPoint) nodes.push(drawSun(project(dragPoint), true));
  const focusedSun = document.activeElement?.getAttribute("data-sun-drag") === "true";
  ui.scene.replaceChildren(...nodes);
  if (focusedSun) ui.scene.querySelector("[data-sun-drag]")?.focus();
}

function drawSun(point, ghost) {
  const group = svg("g", { transform: `translate(${point[0]},${point[1]})`, opacity: ghost ? 0.65 : dragging ? 0.45 : 1 });
  if (!ghost) {
    group.setAttribute("class", "sun-handle"); group.setAttribute("data-sun-drag", "true");
    group.setAttribute("role", "button"); group.setAttribute("tabindex", connected && !busy ? "0" : "-1");
    group.setAttribute("aria-disabled", String(!connected || busy));
    group.setAttribute("aria-label", `Sunshine at x ${fixed(state?.physics?.sun?.position_m?.[0])}, height ${fixed(state?.physics?.sun?.position_m?.[2])} metres. Arrow keys move the sunshine; its movement does not reset the try.`);
    group.addEventListener("keydown", (event) => {
      const moves = { ArrowLeft: [-0.04, 0, 0], ArrowRight: [0.04, 0, 0], ArrowUp: [0, 0, 0.04], ArrowDown: [0, 0, -0.04] };
      if (moves[event.key] && connected && !busy) { event.preventDefault(); moveSun(state.physics.sun.position_m.map((value, index) => value + moves[event.key][index])); }
    });
  }
  group.append(svg("circle", { cx: 0, cy: 0, r: 31, fill: "#e1c66e13", stroke: "transparent", class: "sun-focus" }));
  for (let index = 0; index < 10; index += 1) {
    const a = index * Math.PI / 5;
    group.append(svg("line", { x1: Math.cos(a) * 21, y1: Math.sin(a) * 21, x2: Math.cos(a) * 27, y2: Math.sin(a) * 27, stroke: "#d9bf69", "stroke-width": 1.5 }));
  }
  group.append(svg("circle", { cx: 0, cy: 0, r: 14, fill: "#e2c878", stroke: "#f0dc9b", "stroke-width": 1 }));
  group.append(svg("text", { x: 0, y: 43, fill: "#ddca90", "font-size": 11, "text-anchor": "middle" }, ghost ? "release to place" : "sunshine"));
  return group;
}

function drawChart() {
  const history = (state?.history ?? []).filter((item) => typeof item.time_s === "number" && typeof item.illuminance_lux === "number" && Number.isFinite(item.illuminance_lux));
  const key = JSON.stringify(history);
  if (key === chartSignature) return;
  chartSignature = key;
  const width = 750; const height = 115; const minTime = history[0]?.time_s ?? 0; const maxTime = Math.max(minTime + 1, history.at(-1)?.time_s ?? 1);
  const maxLight = Math.max(100, ...history.map((item) => num(item.illuminance_lux)));
  const px = (time) => 3 + (time - minTime) / (maxTime - minTime) * (width - 6);
  const py = (value) => height - 7 - Math.max(0, value) / maxLight * (height - 19);
  const nodes = [];
  for (const fraction of [0, 0.5, 1]) nodes.push(svg("line", { x1: 0, x2: width, y1: py(maxLight * fraction), y2: py(maxLight * fraction), stroke: "#c8bd9b", "stroke-width": 0.7, "stroke-dasharray": "3 5" }));
  if (history.length > 1) {
    const points = history.map((item) => [px(item.time_s), py(item.illuminance_lux)]);
    nodes.push(svg("path", { d: `M ${points[0][0]} ${height - 7} L ${points.map((point) => point.join(" ")).join(" L ")} L ${points.at(-1)[0]} ${height - 7} Z`, fill: "#b9c89133" }));
    nodes.push(svg("polyline", { points: points.map((point) => point.join(",")).join(" "), fill: "none", stroke: "#849a58", "stroke-width": 2 }));
  } else nodes.push(svg("text", { x: width / 2, y: height / 2, "text-anchor": "middle", fill: "#9b8c6d", "font-size": 12 }, "Its light history will grow here."));
  ui.chart.replaceChildren(...nodes);
  ui.chart.setAttribute("aria-label", history.length ? `${history.length} measured light samples from ${fixed(minTime, 1)} to ${fixed(history.at(-1).time_s, 1)} seconds. Latest ${fixed(history.at(-1).illuminance_lux, 0)} lux. Chart scale 0 to ${fixed(maxLight, 0)} lux.` : "Waiting for measured light history.");
  ui.chartStart.textContent = `${fixed(minTime, 0)} s · 0–${fixed(maxLight, 0)} lx`;
  ui.chartEnd.textContent = `${fixed(history.at(-1)?.time_s, 1)} s`;
}

ui.focus.addEventListener("click", () => setFocusView(!focusView));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && focusView) {
    event.preventDefault();
    setFocusView(false);
  }
});
ui.start.addEventListener("click", () => control(state?.status === "paused" && state?.mode === chosenMode ? "play" : "start", state?.status === "paused" && state?.mode === chosenMode ? {} : { mode: chosenMode }, "The creature is trying. You can move the sunshine while it continues."));
ui.pause.addEventListener("click", () => control("pause", {}, "Paused. Its body and memory are kept at this moment."));
ui.reset.addEventListener("click", () => control("reset", {}, "The body is ready for a fresh try. Previous memory has been cleared."));
ui.import.addEventListener("click", importWorkshopBody);
$$("[data-mode]").forEach((button) => button.addEventListener("click", () => { chosenMode = button.dataset.mode; modeChosen = true; update(); }));
$$("[data-sun]").forEach((button) => button.addEventListener("click", () => moveSun(SUN_PRESETS[button.dataset.sun])));
ui.segment.addEventListener("change", () => fillSlots(Number(ui.segment.value), Number(ui.slot.value)));
ui.attach.addEventListener("click", () => {
  if (!canChangeBody()) return;
  const assembly = copy(state.assembly);
  const light = { segment: Number(ui.segment.value), slot: Number(ui.slot.value), side: Number(ui.side.value), face: Number(ui.face.value) };
  if (socketOccupied(assembly, light)) { message("That socket already holds a part. Choose an empty socket for the light sensor."); return; }
  assembly.light = light;
  control("build", { assembly }, "The light-sensitive face is attached. Wake this body for a fresh try.");
});
ui.scene.addEventListener("pointerdown", (event) => {
  if (!event.target.closest("[data-sun-drag]") || !connected || busy || !plotMapping) return;
  event.preventDefault(); dragging = { pointerId: event.pointerId, mapping: copy(plotMapping) };
  dragPoint = copy(state.physics.sun.position_m); ui.scene.setPointerCapture(event.pointerId); update();
});
ui.scene.addEventListener("pointermove", (event) => {
  if (!dragging || event.pointerId !== dragging.pointerId) return;
  const rect = ui.scene.getBoundingClientRect(); const map = dragging.mapping;
  const ratio = Math.min(rect.width / map.width, rect.height / map.height);
  const x = (event.clientX - rect.left - (rect.width - map.width * ratio) / 2) / ratio;
  const y = (event.clientY - rect.top - (rect.height - map.height * ratio) / 2) / ratio;
  dragPoint = [Math.max(-2, Math.min(2, (x - map.origin[0]) / map.scale)), 0, Math.max(-2, Math.min(2, (map.origin[1] - y) / map.scale))];
  drawScene();
});
ui.scene.addEventListener("pointerup", (event) => {
  if (!dragging || event.pointerId !== dragging.pointerId) return;
  const position = dragPoint; dragging = null;
  ui.scene.releasePointerCapture(event.pointerId); moveSun(position);
});
ui.scene.addEventListener("pointercancel", () => { dragging = null; dragPoint = null; update(); });
ui.scene.addEventListener("lostpointercapture", (event) => {
  if (dragging?.pointerId === event.pointerId) { dragging = null; dragPoint = null; update(); }
});
new ResizeObserver(() => { sceneSignature = ""; drawScene(); }).observe(ui.scene.parentElement);
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
update(); poll(); window.setInterval(poll, 100);
fetchJSON("/api/light/spec").then((value) => { spec = value; }).catch(() => { spec = null; });
