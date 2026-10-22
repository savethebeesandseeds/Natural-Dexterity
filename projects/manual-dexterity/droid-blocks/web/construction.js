"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const ui = {
  scene: $("[data-scene]"), connection: $("[data-connection]"), name: $("[data-name]"),
  benchTitle: $("[data-bench-title]"), bodyStatus: $("[data-body-status]"), instruction: $("[data-scene-instruction]"),
  preview: $("[data-preview-chip]"), time: $("[data-time]"), wake: $("[data-wake]"), pause: $("[data-pause]"),
  reset: $("[data-reset]"), runNote: $("[data-run-note]"), undo: $("[data-undo]"), count: $("[data-block-count]"),
  selection: $("[data-selection-panel]"), selectionTitle: $("[data-selection-title]"), selectionNote: $("[data-selection-note]"),
  segment: $("[data-part-segment]"), slot: $("[data-part-slot]"), side: $("[data-part-side]"),
  move: $("[data-move-part]"), remove: $("[data-remove-part]"), deselect: $("[data-deselect]"),
  save: $("[data-save]"), load: $("[data-load]"), restore: $("[data-restore]"), saveNote: $("[data-save-note]"), file: $("[data-file]"), message: $("[data-message]"),
  cards: $("[data-motion-cards]"), patternNote: $("[data-pattern-note]"), gyro: $("[data-gyro]"), accel: $("[data-accel]"),
  sensorStatus: $("[data-sensor-status]"), gyroVector: $("[data-gyro-vector]"), accelVector: $("[data-accel-vector]"),
  technical: $("[data-technical-state]"), live: $("[data-live]"),
};
const SVG_NS = "http://www.w3.org/2000/svg";
const STUD = 0.04;
const LOCAL_SAVE_KEY = "droid-blocks.construction.saved-body.v1";
const copy = (value) => JSON.parse(JSON.stringify(value));
const num = (value, fallback = 0) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
const fixed = (value, digits = 2) => num(value).toFixed(digits);
const magnitude = (values) => Array.isArray(values) ? Math.hypot(...values.map((value) => num(value))) : null;
const canonical = (value) => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value;
const same = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const starters = {
  balanced: { schema: "construction_kit_v1", name: "Little pendulum", segments: [3, 3], blocks: [{ id: "block-1", segment: 1, slot: 2, side: 1 }], sensor: { segment: 1, slot: 2, side: -1 } },
  long: { schema: "construction_kit_v1", name: "Long tail", segments: [3, 6], blocks: [{ id: "block-1", segment: 1, slot: 5, side: 1 }], sensor: { segment: 1, slot: 4, side: -1 } },
  wide: { schema: "construction_kit_v1", name: "Side pockets", segments: [4, 2], blocks: [{ id: "block-1", segment: 0, slot: 2, side: -1 }, { id: "block-2", segment: 0, slot: 2, side: 1 }, { id: "block-3", segment: 1, slot: 1, side: 1 }], sensor: { segment: 1, slot: 1, side: -1 } },
};
let draft = copy(starters.balanced);
let state = null;
let connected = false;
let busy = false;
let polling = false;
let generation = 0;
let initialized = false;
let selection = null;
let tool = "block";
let undoStack = [];
let sceneSignature = "";
let cardsSignature = "";
let lastRevision = null;
let announcementTimer = null;
let selectionFormSignature = "";
let savedBody = null;

function announce(text) {
  window.clearTimeout(announcementTimer);
  ui.live.textContent = "";
  announcementTimer = window.setTimeout(() => { ui.live.textContent = text; }, 30);
}

function showMessage(text = "") {
  if (ui.message.textContent !== text) ui.message.textContent = text;
  ui.message.hidden = !text;
}

function dirty() { return !state || !same(draft, state.assembly); }
function running() { return state?.status === "running"; }
function editable() { return !busy && !running(); }

function validateAssembly(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("A body file must contain one assembly object.");
  const allowed = ["schema", "name", "segments", "blocks", "sensor"];
  if (Object.keys(raw).some((key) => !allowed.includes(key))) throw new Error("This file has fields outside the construction kit.");
  if (raw.schema !== "construction_kit_v1") throw new Error("This file uses a different construction kit.");
  if (typeof raw.name !== "string" || [...raw.name].length > 60) throw new Error("The creature’s name can have up to 60 characters.");
  if (!Array.isArray(raw.segments) || raw.segments.length !== 2 || raw.segments.some((length) => !Number.isInteger(length) || length < 2 || length > 6)) throw new Error("Each of the two beams needs 2–6 studs.");
  if (!Array.isArray(raw.blocks) || raw.blocks.length > 12) throw new Error("This kit holds up to 12 blocks.");
  const occupied = new Set();
  const ids = new Set();
  const checkPosition = (part, sensor) => {
    if (!part || typeof part !== "object" || Array.isArray(part)) throw new Error("Every part needs a socket position.");
    const keys = sensor ? ["segment", "slot", "side"] : ["id", "segment", "slot", "side"];
    if (Object.keys(part).some((key) => !keys.includes(key))) throw new Error("A part has fields outside this kit.");
    if (![0, 1].includes(part.segment) || !Number.isInteger(part.slot) || part.slot < 0 || part.slot >= raw.segments[part.segment] || ![-1, 1].includes(part.side)) throw new Error("A part is outside its beam’s available sockets.");
    const key = `${part.segment}:${part.slot}:${part.side}`;
    if (occupied.has(key)) throw new Error("Two parts cannot occupy the same socket.");
    occupied.add(key);
    if (!sensor) {
      if (typeof part.id !== "string" || [...part.id].length < 1 || [...part.id].length > 60 || ids.has(part.id)) throw new Error("Blocks need unique IDs of 1–60 characters.");
      ids.add(part.id);
    }
  };
  raw.blocks.forEach((part) => checkPosition(part, false));
  checkPosition(raw.sensor, true);
  return copy(raw);
}

function selectedPart() { return selection === "sensor" ? draft.sensor : draft.blocks.find((block) => block.id === selection?.slice(6)); }
function occupiedAt(position, except = null) {
  const matches = (part) => part.segment === position.segment && part.slot === position.slot && part.side === position.side;
  if (except !== "sensor" && matches(draft.sensor)) return true;
  return draft.blocks.some((block) => block.id !== except?.slice(6) && matches(block));
}

function changeDraft(change, message) {
  if (!editable()) return;
  const previous = copy(draft);
  const next = copy(draft);
  change(next);
  try { draft = validateAssembly(next); } catch (error) { showMessage(error.message); return; }
  if (same(previous, draft)) return;
  undoStack.push(previous);
  if (undoStack.length > 50) undoStack.shift();
  showMessage();
  sceneSignature = "";
  updateUI();
  if (message) announce(message);
}

function selectPart(id) {
  if (!editable()) return;
  selection = id;
  tool = id === "sensor" ? "sensor" : "move";
  updateUI();
  announce(id === "sensor" ? "Eye selected. Choose an empty socket to move it." : "Block selected. Choose an empty socket to move it, or use the part controls.");
}

function useSocket(segment, slot, side) {
  if (!editable()) return;
  const position = { segment, slot, side };
  const moving = tool === "sensor" ? "sensor" : tool === "move" ? selection : null;
  if (occupiedAt(position, moving)) { showMessage("That socket already holds a part. Choose an empty one."); return; }
  if (moving) {
    changeDraft((next) => Object.assign(moving === "sensor" ? next.sensor : next.blocks.find((part) => part.id === moving.slice(6)), position), "Part moved. Your draft is ready to wake.");
  } else {
    if (draft.blocks.length >= 12) { showMessage("All 12 blocks are in use. Move or remove one to make room."); return; }
    let nextId = 1;
    while (draft.blocks.some((block) => block.id === `block-${nextId}`)) nextId += 1;
    changeDraft((next) => next.blocks.push({ id: `block-${nextId}`, ...position }), "A block snapped into place.");
  }
}

function fillSlotOptions(segment, selectedSlot = 0) {
  ui.slot.replaceChildren(...Array.from({ length: draft.segments[segment] }, (_, index) => {
    const option = document.createElement("option");
    option.value = String(index); option.textContent = String(index + 1);
    return option;
  }));
  ui.slot.value = String(Math.min(selectedSlot, draft.segments[segment] - 1));
}

function updateEditor() {
  const enabled = editable();
  ui.name.disabled = !enabled;
  if (document.activeElement !== ui.name) ui.name.value = draft.name;
  $$("[data-starter], [data-tool]").forEach((button) => { button.disabled = !enabled; });
  $$("[data-length]").forEach((button) => {
    const [segment, delta] = button.dataset.length.split(":").map(Number);
    const length = draft.segments[segment];
    button.disabled = !enabled || (delta < 0 ? length <= 2 : length >= 6);
  });
  $$("[data-length-value]").forEach((output) => { output.textContent = String(draft.segments[Number(output.dataset.lengthValue)]); });
  $$("[data-tool]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.tool === tool)));
  ui.undo.disabled = !enabled || undoStack.length === 0;
  ui.load.disabled = !enabled;
  ui.restore.disabled = !enabled || !savedBody;
  ui.save.disabled = busy;
  ui.count.textContent = `${draft.blocks.length} / 12`;
  const part = selectedPart();
  ui.selection.hidden = !part;
  if (part) {
    ui.selectionTitle.textContent = selection === "sensor" ? "Its attached eye" : "Your selected block";
    ui.selectionNote.textContent = enabled ? "Choose an empty socket, or choose its position below." : "Pause the movement before moving this part.";
    const formSignature = JSON.stringify([selection, part, draft.segments]);
    if (formSignature !== selectionFormSignature) {
      selectionFormSignature = formSignature;
      ui.segment.value = String(part.segment);
      fillSlotOptions(part.segment, part.slot);
      ui.side.value = String(part.side);
    }
    [ui.segment, ui.slot, ui.side, ui.move, ui.deselect].forEach((control) => { control.disabled = !enabled; });
    ui.remove.disabled = !enabled || selection === "sensor";
    ui.remove.hidden = selection === "sensor";
  } else selectionFormSignature = "";
}

function updateUI() {
  updateEditor();
  const changed = dirty();
  const status = state?.status ?? "waiting";
  const active = running();
  const statusText = { ready: "Ready to wake", running: state?.mode === "replay" ? "Replaying a movement" : "Trying movements", paused: "Paused", completed: state?.mode === "replay" ? "Replay complete" : "Exploration complete", error: "Movement stopped", waiting: "Waiting for native physics" };
  const currentPattern = (state?.patterns ?? []).find((pattern) => (pattern.id ?? pattern.pattern_id) === state?.explorer?.pattern_id);
  const patternName = state?.explorer?.pattern_name ?? currentPattern?.name ?? currentPattern?.label ?? "Movement";
  if (["running", "paused"].includes(status)) {
    const trialLabel = state?.mode === "replay" ? "replay" : `${num(state?.trial_index) + 1} of ${num(state?.trial_count, 5)}`;
    statusText[status] = `${status === "paused" ? "Paused · " : ""}${patternName} · ${trialLabel}`;
  }
  ui.benchTitle.textContent = draft.name || "My swinging creature";
  ui.bodyStatus.textContent = changed && state ? "Draft · not applied yet" : statusText[status] ?? status;
  ui.time.textContent = `00:${fixed(state?.physics?.sim_time ?? state?.elapsed_s, 1).padStart(4, "0")}`;
  ui.preview.textContent = changed ? "Paused draft · wake to apply" : active ? "Body in motion" : status === "error" ? "Stopped · last native state" : "Paused body";
  ui.instruction.textContent = active ? "Watch the freely moving elbow and the eye’s readings." : !connected ? "Reconnect to the engine to wake this body." : tool === "sensor" ? "Choose an empty socket for its eye." : tool === "move" ? "Choose an empty socket to move the selected block." : "Click a little socket to snap on a block.";
  ui.wake.disabled = !connected || busy || active;
  ui.wake.textContent = !changed && status === "paused" ? "Keep going ↗" : "Wake this body ↗";
  ui.pause.disabled = !connected || busy || !active;
  ui.reset.disabled = !connected || busy || active;
  ui.runNote.textContent = changed ? "Your changes are ready to try. Wake this body to start a fresh set of movements." : status === "error" ? "The engine stopped this trial. Its last state and completed movement cards remain visible. Start again to reset the body." : state?.mode === "replay" ? "Replaying a recorded effort pattern on this body. Pause to inspect it." : "Each new exploration tries the same movement patterns. Completed cards keep what the attached eye observed.";
  if (!connected && state) ui.runNote.textContent = "Engine link interrupted. The last received state stays visible; pause and edits will be available when the link returns.";
  updatePatterns();
  updateSensor();
  renderScene();
}

function acceptState(next) {
  if (next?.schema !== "construction_session_v1" || !next.assembly || !next.physics) throw new Error("The construction service returned an incompatible state.");
  const assembly = validateAssembly(next.assembly);
  const previousChanged = dirty();
  const revisionChanged = lastRevision !== null && next.assembly_revision !== lastRevision;
  state = next;
  if ((!initialized && undoStack.length === 0) || (revisionChanged && !previousChanged)) { draft = assembly; selection = null; undoStack = []; }
  initialized = true;
  if (revisionChanged && previousChanged && !same(draft, assembly)) showMessage("The engine’s body changed elsewhere. Your draft is still here; waking will apply it as a new body.");
  lastRevision = next.assembly_revision;
  connected = true;
  ui.connection.dataset.state = "connected";
  ui.connection.replaceChildren(Object.assign(document.createElement("i"), { ariaHidden: "true" }), document.createTextNode("Native engine connected"));
  if (next.error) showMessage(next.error);
  updateUI();
}

async function request(action, fields = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch("/api/construction/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, ...fields }), signal: controller.signal });
    const next = await response.json();
    if (!response.ok) throw new Error(next.error || `HTTP ${response.status}`);
    acceptState(next);
    return next;
  } finally { window.clearTimeout(timeout); }
}

async function control(action, fields = {}) {
  if (busy || !connected) return;
  busy = true; generation += 1; showMessage(); updateUI();
  try {
    if (action === "wake") {
      if (!dirty() && state?.status === "paused") await request("play");
      else {
        if (dirty()) {
          const requested = validateAssembly(draft);
          const built = await request("build", { assembly: requested });
          if (built.status === "error" || !same(built.assembly, requested)) throw new Error("The engine did not confirm this draft as its applied body. Exploration was not started.");
        }
        await request("explore");
      }
      announce("Your creature is moving. Pause whenever you want to change its body.");
    } else {
      await request(action, fields);
      announce(action === "pause" ? "Movement paused. The body is ready to inspect or change." : action === "replay" ? "Replaying this movement on the same body." : "The native body is back at rest. Completed movement cards remain.");
    }
  } catch (error) {
    showMessage(`The command could not be confirmed: ${error.message}. The page will check the engine again.`);
  } finally { busy = false; updateUI(); }
}

async function poll() {
  if (polling || busy || document.hidden) return;
  polling = true;
  const version = generation;
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), 5000);
  try {
    const response = await fetch("/api/construction/state", { cache: "no-store", signal: abort.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const next = await response.json();
    if (generation === version) acceptState(next);
  } catch (error) {
    if (generation === version) {
      connected = false;
      ui.connection.dataset.state = "disconnected";
      ui.connection.replaceChildren(Object.assign(document.createElement("i"), { ariaHidden: "true" }), document.createTextNode("Engine link interrupted"));
      updateUI();
    }
  } finally { window.clearTimeout(timeout); polling = false; }
}

function textNode(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function updatePatterns() {
  const patterns = Array.isArray(state?.patterns) ? state.patterns : [];
  const repertoire = Array.isArray(state?.repertoire) ? state.repertoire : [];
  const changed = dirty();
  const key = JSON.stringify([patterns, repertoire.map((result) => [result.pattern_id, result.gyro_rms_rad_s, result.specific_force_variation_m_s2, result.duration_s]), changed, state?.status, state?.mode, state?.explorer?.pattern_id, busy, connected]);
  if (key === cardsSignature) return;
  cardsSignature = key;
  if (!patterns.length) return;
  ui.cards.replaceChildren(...patterns.map((pattern, index) => {
    const id = pattern.id ?? pattern.pattern_id;
    const result = repertoire.find((item) => item.pattern_id === id);
    const card = textNode("button", "motion-card");
    card.type = "button";
    card.disabled = !result || changed || running() || busy || !connected;
    card.setAttribute("aria-pressed", String(state?.mode === "replay" && state?.explorer?.pattern_id === id));
    card.append(textNode("span", "", ["∿", "≈", "≋", "··", "↝"][index % 5]));
    card.append(textNode("strong", "", pattern.name ?? pattern.label ?? id));
    card.append(textNode("small", "", result ? `Turning: ${fixed(result.gyro_rms_rad_s)} rad/s RMS` : pattern.description ?? "Waiting for its first try"));
    card.append(textNode("em", "", result ? `${fixed(result.duration_s, 0)} s · ${changed ? "previous body" : "replay this movement ↗"}` : `${fixed(pattern.duration_s, 0)} s · not tried yet`));
    card.addEventListener("click", () => control("replay", { pattern_id: id }));
    return card;
  }));
  ui.patternNote.textContent = changed ? "These cards belong to the applied body. Waking your new draft starts a fresh collection." : running() ? `Trying ${Math.min(num(state.trial_index) + 1, num(state.trial_count, patterns.length))} of ${num(state.trial_count, patterns.length)} movements. Cards unlock when exploration pauses or ends.` : "Replay the same effort pattern and compare what you notice. No best-movement score.";
}

function updateSensor() {
  const sensor = state?.physics?.sensor;
  const sample = sensor;
  const valid = sample?.valid === true;
  const gyro = sample?.observations?.angular_velocity_rad_s;
  const accel = sample?.observations?.specific_force_m_s2;
  const gyroMagnitude = valid ? magnitude(gyro) : null;
  const accelMagnitude = valid ? magnitude(accel) : null;
  ui.gyro.textContent = gyroMagnitude === null ? "— rad/s" : `${fixed(gyroMagnitude)} rad/s`;
  ui.accel.textContent = accelMagnitude === null ? "— m/s²" : `${fixed(accelMagnitude)} m/s²`;
  ui.gyroVector.textContent = gyroMagnitude === null ? "—" : gyro.map((value) => fixed(value, 3)).join(", ");
  ui.accelVector.textContent = accelMagnitude === null ? "—" : accel.map((value) => fixed(value, 3)).join(", ");
  ui.sensorStatus.textContent = dirty() ? "Readings belong to the applied body" : !valid ? "Waiting for a delivered sensor sample" : running() ? "Live readings at the attached eye" : "Last reading · body paused";
  ui.technical.textContent = state ? `Native assembly revision ${state.assembly_revision}. ${state.mode} · ${state.status}. ${num(state.physics?.tick)} physics ticks. ${dirty() ? "An unapplied draft is shown on the bench." : "The displayed body matches the native assembly."}` : "No active native assembly yet.";
}

function svgElement(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function draftPhysics(assembly) {
  const length0 = assembly.segments[0] * STUD;
  const lengths = assembly.segments.map((length) => length * STUD);
  const position = (part, sensor = false) => [(sensor ? 0.038 : STUD) * part.side, 0, -(part.segment === 1 ? length0 : 0) - (part.slot + 0.5) * STUD];
  const segments = assembly.segments.map((length, index) => ({ index, length_studs: length, start_m: [0, 0, index ? -length0 : 0], end_m: [0, 0, -(index ? length0 + lengths[1] : length0)] }));
  const geometry = [];
  for (let segment = 0; segment < 2; segment += 1) for (let slot = 0; slot < assembly.segments[segment]; slot += 1) geometry.push({ id: `beam-${segment}-${slot}`, kind: "beam", segment, slot, side: 0, position_m: [0, 0, -(segment ? length0 : 0) - (slot + 0.5) * STUD], size_m: [STUD, 0.022, STUD], quaternion: [1, 0, 0, 0] });
  assembly.blocks.forEach((block) => geometry.push({ ...block, kind: "block", position_m: position(block), size_m: [STUD, STUD, STUD], quaternion: [1, 0, 0, 0] }));
  geometry.push({ id: "sensor", ...assembly.sensor, kind: "sensor", position_m: position(assembly.sensor, true), size_m: [0.036, 0.036, 0.036], quaternion: [1, 0, 0, 0] });
  return { geometry, segments, joints: { motor_m: [0, 0, 0], passive_m: [0, 0, -length0] } };
}

function renderScene() {
  const changed = dirty();
  const physical = changed || !state ? draftPhysics(draft) : state.physics;
  const width = Math.max(300, ui.scene.clientWidth || 700);
  const height = width < 600 ? 400 : 490;
  const key = JSON.stringify([changed, physical.geometry, physical.joints, physical.segments, state?.trail, selection, tool, editable(), width, draft]);
  if (key === sceneSignature) return;
  sceneSignature = key;
  ui.scene.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const reach = draft.segments.reduce((sum, value) => sum + value, 0) * STUD + STUD * 1.6;
  const scale = Math.min((width - 70) / (reach * 2), (height - 135) / reach);
  const origin = [width / 2, 88];
  const project = (point) => [origin[0] + num(point?.[0]) * scale, origin[1] - num(point?.[2]) * scale];
  const nodes = [svgElement("title", { id: "scene-title" }, draft.name), svgElement("desc", { id: "scene-description" }, changed ? "Paused editable draft. Choose an empty socket to add or move a part. Waking builds this body in native physics." : "Native two-beam creature attached at a powered base hinge, with a freely moving elbow and an attached motion sensor.")];
  for (let x = 20; x < width; x += 28) for (let y = 50; y < height - 30; y += 28) nodes.push(svgElement("circle", { cx: x, cy: y, r: 0.7, fill: "#5e7b653e" }));
  if (!changed && Array.isArray(state?.trail) && state.trail.length > 1) nodes.push(svgElement("polyline", { points: state.trail.map((item) => project(item.position).join(",")).join(" "), fill: "none", stroke: "#bfb777", "stroke-opacity": 0.4, "stroke-width": 1.5, "stroke-dasharray": "2 5" }));
  const motor = project(physical.joints?.motor_m ?? [0, 0, 0]);
  nodes.push(svgElement("path", { d: `M ${motor[0] - 54} ${motor[1] - 26} h 108`, stroke: "#80927a", "stroke-width": 4, "stroke-linecap": "round" }));
  for (let n = -4; n <= 4; n += 1) nodes.push(svgElement("path", { d: `M ${motor[0] + n * 11} ${motor[1] - 27} l 7 -8`, stroke: "#657e68", "stroke-width": 1 }));
  nodes.push(svgElement("line", { x1: motor[0], x2: motor[0], y1: motor[1] - 25, y2: motor[1], stroke: "#899c82", "stroke-width": 6 }));
  const geometry = Array.isArray(physical.geometry) ? physical.geometry : [];
  const beams = geometry.filter((part) => part.kind === "beam");
  for (const part of beams) nodes.push(drawPart(part, project, scale));
  if (editable()) {
    const segments = Array.isArray(physical.segments) ? physical.segments : [];
    for (const segment of segments) {
      const start = segment.start_m;
      const end = segment.end_m;
      const length = draft.segments[segment.index];
      const dx = num(end?.[0]) - num(start?.[0]);
      const dz = num(end?.[2]) - num(start?.[2]);
      const distance = Math.hypot(dx, dz) || 1;
      for (let slot = 0; slot < length; slot += 1) for (const side of [-1, 1]) {
        const position = { segment: segment.index, slot, side };
        if (occupiedAt(position)) continue;
        const fraction = (slot + 0.5) / length;
        const point = project([num(start?.[0]) + dx * fraction - dz / distance * STUD * side, 0, num(start?.[2]) + dz * fraction + dx / distance * STUD * side]);
        const socket = svgElement("g", { class: "socket", role: "button", tabindex: 0, "data-socket": `${segment.index}:${slot}:${side}`, "aria-label": `${tool === "block" ? "Add block" : "Move selected part"}, ${segment.index === 0 ? "upper" : "lower"} beam, stud ${slot + 1}, ${side < 0 ? "left" : "right"} side` });
        socket.append(svgElement("circle", { cx: point[0], cy: point[1], r: Math.max(10, STUD * scale * 0.48), fill: "transparent", stroke: "none" }));
        socket.append(svgElement("circle", { cx: point[0], cy: point[1], r: Math.min(7, STUD * scale * 0.28), fill: "#173d2c", stroke: "#668363", "stroke-width": 1, "stroke-dasharray": "2 2" }));
        const activate = () => useSocket(segment.index, slot, side);
        socket.addEventListener("click", activate);
        socket.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); } });
        nodes.push(socket);
      }
    }
  }
  for (const part of geometry.filter((item) => item.kind !== "beam")) nodes.push(drawPart(part, project, scale));
  const elbow = project(physical.joints?.passive_m);
  nodes.push(svgElement("circle", { cx: elbow[0], cy: elbow[1], r: 8, fill: "#163c31", stroke: "#b8b594", "stroke-width": 2 }));
  nodes.push(svgElement("circle", { cx: elbow[0], cy: elbow[1], r: 2.5, fill: "#d7ce9d" }));
  nodes.push(svgElement("circle", { cx: motor[0], cy: motor[1], r: 13, fill: "#cca45c", stroke: "#e0c786", "stroke-width": 2 }));
  nodes.push(svgElement("path", { d: `M ${motor[0]-4} ${motor[1]} l 4 -5 l 4 5 l -4 5 z`, fill: "#324c35" }));
  nodes.push(svgElement("text", { x: motor[0] + 23, y: motor[1] - 5, class: "scene-structure-label" }, "powered hinge"));
  if (!running()) nodes.push(svgElement("text", { x: elbow[0] + 16, y: elbow[1] + 4, class: "scene-structure-label" }, "free elbow"));
  const focused = document.activeElement?.getAttribute("data-part-id");
  const focusedSocket = document.activeElement?.getAttribute("data-socket");
  ui.scene.replaceChildren(...nodes);
  if (focused) [...ui.scene.querySelectorAll("[data-part-id]")].find((part) => part.getAttribute("data-part-id") === focused)?.focus();
  else if (focusedSocket) [...ui.scene.querySelectorAll("[data-socket]")].find((part) => part.getAttribute("data-socket") === focusedSocket)?.focus();
}

function drawPart(part, project, scale) {
  const point = project(part.position_m);
  const quaternion = part.quaternion ?? [1, 0, 0, 0];
  const angle = Math.atan2(2 * (num(quaternion[0], 1) * num(quaternion[2]) + num(quaternion[1]) * num(quaternion[3])), 1 - 2 * (num(quaternion[2]) ** 2 + num(quaternion[3]) ** 2)) * 180 / Math.PI;
  const width = num(part.size_m?.[0], STUD) * scale;
  const height = num(part.size_m?.[2], STUD) * scale;
  const isSensor = part.kind === "sensor";
  const identity = isSensor ? "sensor" : `block:${part.id}`;
  const mounted = part.kind !== "beam";
  const group = svgElement("g", { transform: `translate(${point[0]} ${point[1]}) rotate(${angle})`, class: mounted ? "mounted-part" : "beam-group" });
  const fill = isSensor ? "#d1b55f" : part.kind === "block" ? "#bc8c6e" : part.segment === 0 ? "#5b9581" : "#7f91ac";
  group.append(svgElement("rect", { x: -width / 2 + 0.7, y: -height / 2 + 0.7, width: Math.max(1, width - 1.4), height: Math.max(1, height - 1.4), rx: Math.min(4, width / 7), fill, stroke: isSensor ? "#f0d48a" : "#b6c2a8", "stroke-width": 0.8, class: "part-shape" }));
  if (isSensor) {
    group.append(svgElement("circle", { cx: 0, cy: 0, r: width * 0.28, fill: "#546039", stroke: "#ece0a0", "stroke-width": 1 }));
    group.append(svgElement("circle", { cx: 0, cy: 0, r: width * 0.1, fill: "#eed590" }));
  } else group.append(svgElement("circle", { cx: 0, cy: 0, r: width * 0.13, fill: "#263f3538", stroke: "#e4dfbb50", "stroke-width": 0.7 }));
  if (mounted) {
    group.setAttribute("role", "button");
    group.setAttribute("tabindex", editable() ? "0" : "-1");
    group.setAttribute("aria-disabled", String(!editable()));
    group.setAttribute("aria-pressed", String(selection === identity));
    group.setAttribute("data-part-id", identity);
    group.setAttribute("data-socket", `${part.segment}:${part.slot}:${part.side}`);
    group.setAttribute("aria-label", `${isSensor ? "Motion sensor eye" : "Block"}, ${part.segment === 0 ? "upper" : "lower"} beam, stud ${num(part.slot) + 1}, ${part.side < 0 ? "left" : "right"} side. Select to move${isSensor ? "" : " or remove"}.`);
    group.addEventListener("click", () => selectPart(identity));
    group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectPart(identity); } });
  }
  return group;
}

ui.wake.addEventListener("click", () => control("wake"));
ui.pause.addEventListener("click", () => control("pause"));
ui.reset.addEventListener("click", () => control("reset"));
ui.name.addEventListener("change", () => changeDraft((next) => { next.name = ui.name.value.trim() || "My swinging creature"; }, "Creature renamed."));
$$("[data-starter]").forEach((button) => button.addEventListener("click", () => {
  if (!editable()) return;
  undoStack.push(copy(draft)); draft = copy(starters[button.dataset.starter]); selection = null; tool = "block";
  showMessage(); updateUI(); announce("Starter shape ready. Every part can still be changed.");
}));
$$("[data-length]").forEach((button) => button.addEventListener("click", () => {
  const [segment, delta] = button.dataset.length.split(":").map(Number);
  const length = draft.segments[segment] + delta;
  if (delta < 0 && [...draft.blocks, draft.sensor].some((part) => part.segment === segment && part.slot >= length)) { showMessage("There is a part on that end stud. Move or remove it before shortening the beam."); return; }
  changeDraft((next) => { next.segments[segment] = length; }, `${segment === 0 ? "Upper" : "Lower"} beam now has ${length} studs.`);
}));
$$("[data-tool]").forEach((button) => button.addEventListener("click", () => {
  if (!editable()) return;
  tool = button.dataset.tool; selection = tool === "sensor" ? "sensor" : null;
  updateUI(); announce(tool === "sensor" ? "Choose an empty socket to move its eye." : "Choose an empty socket to add a block.");
}));
ui.undo.addEventListener("click", () => { if (!editable() || !undoStack.length) return; draft = undoStack.pop(); selection = null; tool = "block"; showMessage(); updateUI(); announce("Last body change undone."); });
ui.deselect.addEventListener("click", () => { selection = null; tool = "block"; updateUI(); });
ui.segment.addEventListener("change", () => fillSlotOptions(Number(ui.segment.value), Number(ui.slot.value)));
ui.move.addEventListener("click", () => useSocket(Number(ui.segment.value), Number(ui.slot.value), Number(ui.side.value)));
ui.remove.addEventListener("click", () => { if (!selection || selection === "sensor") return; const id = selection.slice(6); selection = null; tool = "block"; changeDraft((next) => { next.blocks = next.blocks.filter((part) => part.id !== id); }, "Block removed."); });
ui.save.addEventListener("click", () => {
  let storageError = "";
  try {
    localStorage.setItem(LOCAL_SAVE_KEY, JSON.stringify({ schema: "construction_local_save_v1", saved_at: new Date().toISOString(), assembly: draft }));
    savedBody = copy(draft);
  } catch (error) { storageError = error.message; }
  const file = new Blob([JSON.stringify(draft, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a"); link.href = url; link.download = `${draft.name.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-|-$/g, "") || "my-creature"}.json`;
  document.body.append(link); link.click(); link.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  ui.saveNote.textContent = storageError ? "Browser storage could not save this draft. A JSON download was prepared; its completion depends on the browser." : "Saved in this browser. Use Restore saved body to bring it back.";
  if (storageError) showMessage(`The browser copy could not be saved: ${storageError}. The JSON download was still prepared.`);
  else showMessage();
  updateUI();
  announce(storageError ? "Browser storage could not save this draft. A JSON download was prepared." : "Body saved in this browser. A JSON download was prepared too.");
});
ui.restore.addEventListener("click", () => {
  if (!editable()) return;
  try {
    const raw = localStorage.getItem(LOCAL_SAVE_KEY);
    if (!raw) throw new Error("There is no saved construction body in this browser.");
    const stored = JSON.parse(raw);
    if (stored.schema !== "construction_local_save_v1") throw new Error("The saved browser copy uses a different version.");
    const restored = validateAssembly(stored.assembly);
    undoStack.push(copy(draft)); draft = restored; selection = null; tool = "block"; savedBody = copy(restored);
    showMessage(); updateUI(); announce("Saved body restored as an editable draft. Wake it to apply it to the engine.");
  } catch (error) { showMessage(`The browser copy could not be restored: ${error.message}`); }
});
ui.load.addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", async () => {
  const file = ui.file.files?.[0]; ui.file.value = "";
  if (!file || !editable()) return;
  if (file.size > 100000) { showMessage("That file is too large for a 12-block body design."); return; }
  try {
    const loaded = validateAssembly(JSON.parse(await file.text()));
    if (!editable()) throw new Error("Pause the creature before loading a new body.");
    undoStack.push(copy(draft)); draft = loaded; selection = null; tool = "block";
    showMessage(); updateUI(); announce("Body loaded as a draft. Wake it to apply it to the engine.");
  } catch (error) { showMessage(`That body could not be loaded: ${error.message}`); }
});
new ResizeObserver(() => { sceneSignature = ""; renderScene(); }).observe(ui.scene.parentElement);
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
try {
  const raw = localStorage.getItem(LOCAL_SAVE_KEY);
  if (raw) {
    const stored = JSON.parse(raw);
    if (stored.schema === "construction_local_save_v1") savedBody = validateAssembly(stored.assembly);
  }
} catch { /* Save and restore report actionable storage errors when requested. */ }
updateUI(); poll(); window.setInterval(poll, 100);
