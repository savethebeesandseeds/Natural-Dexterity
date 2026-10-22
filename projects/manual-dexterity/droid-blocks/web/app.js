"use strict";

const select = (selector) => document.querySelector(selector);
const ui = {
  engine: select("[data-engine]"),
  connectionLabel: select("[data-connection-label]"),
  simTime: select("[data-sim-time]"),
  tick: select("[data-tick]"),
  runState: select("[data-run-state]"),
  moduleCount: select("[data-module-count]"),
  trackLength: select("[data-track-length]"),
  canvasSpeed: select("[data-canvas-speed]"),
  offlineTitle: select("[data-offline-title]"),
  offlineDetail: select("[data-offline-detail]"),
  playPause: select("[data-play-pause]"),
  playPauseLabel: select("[data-play-pause-label]"),
  reset: select("[data-reset]"),
  resetLabel: select("[data-reset-label]"),
  sunSides: [...document.querySelectorAll("[data-sun-side]")],
  sunseekerEntry: select("[data-sunseeker-entry]"),
  bodyEntry: select("[data-body-entry]"),
  bodyPanel: select("[data-body-panel]"),
  bodyVariants: [...document.querySelectorAll("[data-body-variant]")],
  bodyStrategies: [...document.querySelectorAll("[data-body-strategy]")],
  bodySetup: select("[data-body-setup]"),
  motorDiscovery: select("[data-motor-discovery]"),
  discoveryMotors: select("[data-discovery-motors]"),
  discoveryNote: select("[data-discovery-note]"),
  bodyPhase: select("[data-body-phase]"),
  experimentNumber: select("[data-experiment-number]"),
  experimentTitle: select("[data-experiment-title]"),
  experimentIntro: select("[data-experiment-intro]"),
  controllerLabel: select("[data-controller-label]"),
  sessionState: select("[data-session-state]"),
  eyeLight: select("[data-eye-light]"),
  eyeReading: select("[data-eye-reading]"),
  lightChange: select("[data-light-change]"),
  caption: select("[data-experiment-caption]"),
  controlError: select("[data-control-error]"),
  progressLabel: select("[data-progress-label]"),
  progressUnit: select("[data-progress-unit]"),
  controllerDetail: select("[data-controller-detail]"),
  controllerDescription: select("[data-controller-description]"),
  exposure: select("[data-exposure]"),
  policyId: select("[data-policy-id]"),
  policyHash: select("[data-policy-hash]"),
  learnedMode: select("[data-learned-mode]"),
  demoMode: select("[data-demo-mode]"),
  viewToggle: select("[data-view-toggle]"),
  detailsToggle: select("[data-details-toggle]"),
  detailsDialog: select("[data-details-dialog]"),
  detailsClose: select("[data-details-close]"),
  trackProgress: select("[data-track-progress]"),
  trackTotal: select("[data-track-total]"),
  progressBar: select("[data-progress-bar]"),
  bodyState: select("[data-body-state]"),
  speed: select("[data-speed]"),
  distance: select("[data-distance]"),
  rewardRate: select("[data-reward-rate]"),
  batterySoc: select("[data-battery-soc]"),
  gaugeMotion: select("[data-gauge-motion]"),
  gaugeBalance: select("[data-gauge-balance]"),
  gaugeEnergy: select("[data-gauge-energy]"),
  rewardSensors: {
    ambient: {
      value: select("[data-sensor-ambient]"),
      reward: select("[data-sensor-ambient-reward]"),
      id: select("[data-sensor-ambient-id]"),
    },
    front: {
      value: select("[data-sensor-front]"),
      reward: select("[data-sensor-front-reward]"),
      id: select("[data-sensor-front-id]"),
    },
    rear: {
      value: select("[data-sensor-rear]"),
      reward: select("[data-sensor-rear-reward]"),
      id: select("[data-sensor-rear-id]"),
    },
  },
  catalogSummary: select("[data-catalog-summary]"),
  motorCatalog: select("[data-motor-catalog]"),
  sensorCatalog: select("[data-sensor-catalog]"),
  liveStatus: select("[data-live-status]"),
  canvas: select("#sim-canvas"),
  canvasShell: select("[data-canvas-shell]"),
  actuators: [...document.querySelectorAll("[data-actuator]")],
};

const ctx = ui.canvas.getContext("2d", { alpha: false });
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const colors = {
  page: "#07110f",
  ground: "#0b1714",
  groundFar: "#0d1d19",
  line: "rgba(183,222,205,.13)",
  ink: "#eff5eb",
  muted: "#738a80",
  signal: "#c9ff67",
  cyan: "#65e8cf",
  chassis: "#357dbc",
  chassisLight: "#58a4df",
  power: "#f29b34",
  wheel: "#131c1a",
  wheelEdge: "#65756e",
};

let latest = null;
let connection = "connecting";
let polling = false;
let controlling = false;
let controlGeneration = 0;
let controlFailure = "";
let lastSession = null;
let previousTrails = new Map();
let lastAnnouncedStatus = "";
let visualPosition = [0, 0, 0.24];
let trail = [];
let lastTrail = null;
let lastFrame = performance.now();
let bodySelection = { variant: "normal", strategy: "discover" };
let discoveryCardIds = "";

const number = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const vector = (value, size, fallback = 0) =>
  Array.from({ length: size }, (_, index) => number(value?.[index], fallback));
const fixed = (value, digits = 2) => number(value).toFixed(digits);
const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : null;

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function waitingForFirstLight(state) {
  return state.body_experiment_physics?.light_valid === false ||
    (state.playground?.mode === "body_lab" && state.playground.body_experiment?.light_sample_valid === false);
}

function rewardSensorComponents(state) {
  const candidates = firstDefined(
    state.reward?.components,
    state.sensors?.reward_instances,
    state.sensors?.instances,
    state.reward_sensors,
    [],
  );
  if (Array.isArray(candidates)) return candidates;
  if (Array.isArray(candidates?.instances)) return candidates.instances;
  if (candidates && typeof candidates === "object") return Object.values(candidates);
  return [];
}

function componentText(component) {
  return [
    component?.family_id,
    component?.type,
    component?.kind,
    component?.name,
    component?.label,
    component?.mount,
    component?.location,
    component?.module_id,
  ].filter(Boolean).join(" ").toLowerCase();
}

function splitMountedRewardSensors(state) {
  const components = rewardSensorComponents(state);
  const ambient = components.find((component) => /ambient|light/.test(componentText(component))) ?? null;
  const touches = components.filter((component) => /touch|contact/.test(componentText(component)));
  const front = touches.find((component) => /front/.test(componentText(component))) ?? touches[0] ?? null;
  const rear = touches.find((component) => /rear|back/.test(componentText(component)))
    ?? touches.find((component) => component !== front)
    ?? null;
  return { ambient, front, rear };
}

function componentObservation(component, ...names) {
  for (const name of names) {
    const value = firstDefined(component?.observations?.[name], component?.observation?.[name], component?.[name]);
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function setRewardSensor(readout, component, value, unitFallback, fallbackId) {
  const numericValue = finite(value);
  const reward = finite(firstDefined(component?.reward_rate, component?.reward, component?.rate));
  readout.value.textContent = numericValue === null ? "—" : fixed(numericValue, unitFallback === "lx" ? 0 : 2);
  readout.reward.textContent = reward === null ? "— r/s" : `${fixed(reward)} r/s`;
  readout.id.textContent = component?.module_id || component?.id || fallbackId;
}

function feedbackRows(state, actuators) {
  const candidate = firstDefined(
    actuators.feedback,
    actuators.feedback_sets,
    state.actuator_feedback?.instances,
    state.actuator_feedback,
    [],
  );
  if (Array.isArray(candidate)) return candidate;
  if (Array.isArray(candidate?.instances)) return candidate.instances;
  if (candidate && typeof candidate === "object") return Object.values(candidate);
  return [];
}

function feedbackValue(actuators, rows, index, names) {
  for (const name of names) {
    const direct = actuators?.[name];
    if (Array.isArray(direct) && direct[index] !== undefined) return direct[index];
    if (rows[index]?.[name] !== undefined) return rows[index][name];
    if (rows[index]?.state?.[name] !== undefined) return rows[index].state[name];
    if (rows[index]?.feedback?.[name] !== undefined) return rows[index].feedback[name];
  }
  return undefined;
}

function formatTime(seconds) {
  const value = Math.max(0, number(seconds));
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function announce(message) {
  ui.liveStatus.textContent = "";
  window.setTimeout(() => { ui.liveStatus.textContent = message; }, 20);
}

function setConnection(next, detail = "") {
  const changed = connection !== next;
  connection = next;
  document.body.classList.toggle("is-connected", next === "connected");
  document.body.classList.toggle("is-disconnected", next === "disconnected");
  ui.connectionLabel.textContent = next === "connected" ? "Live link" : next === "disconnected" ? "Offline" : "Connecting";
  ui.offlineTitle.textContent = next === "disconnected" ? "Engine link interrupted" : "Waking the engine";
  ui.offlineDetail.textContent = detail || (next === "disconnected"
    ? "The page will reconnect automatically when the local service returns."
    : "Waiting for the simulation service on this port.");
  updateControlAvailability();
  if (changed && next === "disconnected") announce("Simulation service disconnected.");
}

function updateControlAvailability() {
  const unavailable = connection !== "connected" || controlling;
  const session = latest?.playground;
  // A reachable old service is not a learned controller.
  ui.playPause.disabled = unavailable || !session || session.mode === "external" || session.status === "error" || session.status === "stopped";
  ui.reset.disabled = unavailable || !session;
  ui.sunSides.forEach((button) => { button.disabled = unavailable || !session; });
  ui.learnedMode.disabled = unavailable || !session;
  ui.demoMode.disabled = unavailable || !session || session.mode === "demo";
  ui.sunseekerEntry.disabled = unavailable || !session;
  ui.bodyEntry.disabled = unavailable || !session;
  [...ui.bodyVariants, ...ui.bodyStrategies].forEach((button) => { button.disabled = unavailable || !session; });
}

function updateBodyExperiment(session) {
  const bodyLab = session.mode === "body_lab";
  const experiment = session.body_experiment;
  document.body.classList.toggle("body-lab-view", bodyLab);
  ui.bodyPanel.hidden = !bodyLab;
  ui.motorDiscovery.hidden = !bodyLab;
  ui.sunseekerEntry.setAttribute("aria-pressed", String(session.mode === "learned"));
  ui.bodyEntry.setAttribute("aria-pressed", String(bodyLab));
  ui.experimentNumber.textContent = `The airship workshop · Experiment ${bodyLab ? "02" : "01"}`;
  ui.experimentTitle.textContent = bodyLab ? "I changed its body." : "Meet the sun-seeker.";
  ui.experimentIntro.textContent = bodyLab
    ? "Change a transmission. Let it try its motors. Watch how a little experience changes the way it moves."
    : "Choose a patch of sunshine. Wake your little rover. Watch what happens when its world gets brighter.";
  if (!bodyLab) return;
  if (!experiment) {
    ui.bodyPhase.textContent = "Waiting for experiment feedback";
    ui.discoveryMotors.replaceChildren();
    discoveryCardIds = "";
    return;
  }
  bodySelection = { variant: experiment.variant, strategy: experiment.strategy };
  ui.bodyVariants.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.bodyVariant === experiment.variant)));
  ui.bodyStrategies.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.bodyStrategy === experiment.strategy)));
  const polarity = number(experiment.probe_polarity);
  const stopped = session.status === "error" || session.status === "stopped";
  const phaseNames = { ready: "Ready to try", settling: "Letting the motion settle", probing: `Trying ${polarity > 0 ? "+" : polarity < 0 ? "−" : "both"} effort${polarity === 0 ? " directions" : ""}`, seeking: "Using what it noticed", completed: "This try is complete" };
  const phase = session.status === "ready" || session.status === "completed" ? session.status : experiment.phase;
  ui.bodyPhase.textContent = stopped ? "Stopped · past evidence retained" : session.status === "paused" ? `Paused · ${phaseNames[phase] ?? phase}` : phaseNames[phase] ?? phase;
  ui.bodySetup.textContent = "You chose the front-left transmission. The rover receives sensor readings, not your choice.";
  const motors = Array.isArray(experiment.motors) ? experiment.motors : [];
  const cardIds = motors.map((motor) => motor.module_id).join("|");
  if (cardIds !== discoveryCardIds) {
    discoveryCardIds = cardIds;
    ui.discoveryMotors.replaceChildren(...motors.map((motor, index) => {
      const card = element("article", "discovery-motor");
      const heading = element("header");
      heading.append(element("strong", "", `M${String(index + 1).padStart(2, "0")}`), element("span", "motor-trial-status", "Not tried"));
      const preference = element("p", "motor-preference", "Waiting to try");
      const evidence = element("small", "motor-evidence", "No light comparison yet");
      const effort = element("small", "motor-effort", "Effort 0.00");
      const identity = element("small", "motor-identity", motor.module_id);
      card.append(heading, preference, evidence, effort, identity);
      return card;
    }));
  }
  motors.forEach((motor, index) => {
    const card = ui.discoveryMotors.children[index];
    const probing = ["running", "paused"].includes(session.status) && experiment.phase === "probing" && (experiment.active_motor_id === motor.module_id || experiment.strategy === "together");
    const confidence = motor.confidence;
    const direction = number(motor.preference);
    card.classList.toggle("is-probing", probing);
    card.dataset.confidence = confidence;
    card.querySelector(".motor-trial-status").textContent = stopped ? "Trial stopped" : probing ? session.status === "paused" ? "Trial paused" : "Trying now" : confidence === "measured" ? "Compared" : confidence === "uncertain" ? "Unsure" : "Not tried";
    card.querySelector(".motor-preference").textContent = confidence === "untried" ? stopped ? "No preference recorded" : "Waiting to try" : confidence === "measured" && direction !== 0 ? `${direction > 0 ? "+" : "−"} effort preferred` : "No clear preference";
    card.querySelector(".motor-evidence").textContent = confidence === "untried" ? "No light comparison yet" : `Light trend: ${fixed(motor.evidence_lux, 2)} ${experiment.evidence_units ?? "lx/s"}`;
    card.querySelector(".motor-effort").textContent = `${stopped ? "Last request" : "Effort now"}: ${number(motor.effort) > 0 ? "+" : ""}${fixed(motor.effort, 2)}`;
  });
  const comparisonNote = experiment.strategy === "together"
    ? "All motors try together, so the light comparison describes their combined effect. It cannot identify one motor’s contribution."
    : "Each motor tries + and − effort. A light difference suggests a preference; a weak response leaves it unsure. New tries clear this memory.";
  ui.discoveryNote.textContent = stopped ? `The trial stopped. These cards retain past evidence; no motor trial is active. Start again for fresh memory. ${comparisonNote}` : comparisonNote;
}

function updatePlayground(state) {
  const session = state.playground;
  if (!session) {
    ui.controllerLabel.textContent = "Cruise demo · workshop update needed";
    ui.sessionState.textContent = "Learned play is unavailable";
    ui.caption.textContent = "This service has not loaded the learned playground yet. The visible motion is the cruise demo.";
    return;
  }
  const learned = session.mode === "learned";
  const bodyLab = session.mode === "body_lab";
  const lightExperiment = learned || bodyLab;
  const running = session.status === "running";
  const completed = session.status === "completed";
  const states = { ready: "Ready to wake", running: "Exploring", paused: "Paused", completed: "Run complete", stopped: session.mode === "external" ? "External control" : "Stopped", error: "Controller stopped" };
  const controllerName = learned ? "Learned sun-seeker" : bodyLab ? "Body discovery · trying & remembering" : session.mode === "demo" ? "Cruise demo" : "External controller";
  ui.controllerLabel.textContent = controllerName;
  ui.controllerDetail.textContent = controllerName;
  ui.sessionState.textContent = states[session.status] ?? session.status;
  ui.runState.textContent = states[session.status] ?? session.status;
  ui.bodyState.textContent = `${session.mode} · ${session.status}`;
  document.body.classList.toggle("is-running", running);
  ui.sunSides.forEach((button) => button.setAttribute("aria-pressed", String(lightExperiment && button.dataset.sunSide === session.side)));
  const playText = completed ? "Try again" : running ? "Pause" : session.status === "paused" ? "Keep going" : bodyLab ? "Let it discover" : learned ? "Wake the rover" : "Play demo";
  ui.playPauseLabel.textContent = playText;
  ui.playPause.setAttribute("aria-label", playText);
  ui.playPause.dataset.actionLabel = playText;
  ui.playPause.setAttribute("aria-pressed", String(running));
  ui.resetLabel.textContent = session.mode === "external" ? "Prepare sun-seeker" : "Start again";
  ui.progressLabel.textContent = lightExperiment ? "Exploration" : "Course";
  ui.progressUnit.textContent = lightExperiment ? "s" : "m";
  if (lightExperiment) {
    ui.trackProgress.textContent = fixed(session.elapsed_s, 1);
    ui.trackTotal.textContent = fixed(session.duration_s, 0);
    ui.progressBar.max = number(session.duration_s, 20);
    ui.progressBar.value = number(session.elapsed_s);
    ui.progressBar.textContent = `${fixed(session.elapsed_s, 1)} seconds`;
  }
  const waitingForLight = waitingForFirstLight(state);
  const lux = waitingForLight ? null : finite(session.illuminance_lux);
  const initial = finite(session.initial_illuminance_lux);
  ui.eyeLight.value = lux ?? 0;
  ui.eyeLight.textContent = waitingForLight ? "Waiting for first sample" : lux === null ? "Unavailable" : `${fixed(lux, 0)} lux`;
  ui.eyeLight.setAttribute("aria-valuetext", ui.eyeLight.textContent);
  ui.eyeReading.textContent = lux === null ? "— lx" : `${fixed(lux, 0)} lx`;
  const gain = initial > 0 && lux !== null ? (lux / initial - 1) * 100 : null;
  ui.lightChange.textContent = waitingForLight ? "Waiting for its first light sample." : !lightExperiment ? "Watching the current sensor reading." : session.status === "ready"
    ? "What will it do when you wake it?"
    : gain === null ? "Its light reading is unavailable."
    : Math.abs(gain) < 1 ? "About the same light as at the start."
    : `${fixed(Math.abs(gain), 0)}% ${gain > 0 ? "brighter" : "dimmer"} at its eye than at the start.`;
  ui.caption.textContent = bodyLab
    ? `A ${fixed(session.duration_s, 0)}-second body experiment. It tries, compares the light, then moves using this run’s memory. Changing a choice starts again.`
    : learned
    ? completed ? "That was 20 seconds of sensing and moving. Try again, or put the sunshine on the other side."
      : "A 20-second experiment using a saved learned controller. Changing sides starts a fresh run."
    : session.mode === "demo" ? "Cruise demo: a hand-written controller drives using world position. Prepare the sun-seeker in Peek inside."
    : "An external controller owns this simulation. Choose a sunshine side to prepare the sun-seeker.";
  ui.controllerDescription.textContent = bodyLab
    ? `A new hand-written probing baseline tries ${session.body_experiment?.strategy === "together" ? "all motors together" : "each motor separately"}, compares delayed light samples and remembers an effort preference for this run. Probe effort eases as a motor spins faster. It is not the pretrained v3 policy. It receives no world position or transmission label. Sensor sample period: ${fixed(number(session.body_experiment?.sensor_period_s) * 1000, 0)} ms; latency: ${fixed(number(session.body_experiment?.sensor_latency_s) * 1000, 0)} ms.`
    : learned
    ? "Frozen v3 controls the motors using sensor readings and motor feedback. It does not train during this experiment."
    : session.mode === "demo" ? "The cruise demo uses privileged world position. This motion is not produced by the learned policy."
    : "Actions are supplied through the external agent API.";
  ui.exposure.textContent = bodyLab ? "Development body experiment · no held-out test" : learned ? "Familiar training world · not a held-out test" : "Outside the learned play session";
  ui.policyId.textContent = bodyLab || learned ? session.policy_id ?? "—" : "—";
  ui.policyHash.textContent = learned ? session.artifact_sha256 ?? "—" : "—";
  const errorMessage = session.error || controlFailure;
  ui.controlError.textContent = errorMessage;
  ui.controlError.hidden = !errorMessage;
  updateBodyExperiment(session);
  const statusKey = `${session.mode}:${session.status}:${session.body_experiment?.phase ?? ""}:${session.body_experiment?.active_motor_id ?? ""}`;
  if (statusKey !== lastAnnouncedStatus) {
    lastAnnouncedStatus = statusKey;
    announce(`${controllerName}. ${states[session.status] ?? session.status}.${bodyLab ? ` ${ui.bodyPhase.textContent}.` : ""}`);
  }
  ui.canvas.setAttribute("aria-label", `${controllerName}. ${states[session.status] ?? session.status}. ${lightExperiment ? `Sunshine on the ${session.side}. ` : ""}${bodyLab ? `Front-left transmission ${session.body_experiment?.variant ?? "unknown"}. ${ui.bodyPhase.textContent}. ` : ""}Light at its eye: ${lux === null ? "unavailable" : `${fixed(lux, 0)} lux`}.`);
}

function updateTelemetry(state) {
  const robot = state.robot ?? {};
  const sensors = state.sensors ?? {};
  const actuators = state.actuators ?? {};
  const agentEvaluation = state.controller?.kind === "policy_evaluation";
  const modules = Array.isArray(robot.modules) ? robot.modules : [];
  const trackLength = Math.max(0.001, number(state.world?.track_length, 40));
  const distance = Math.max(0, number(robot.distance));
  const progress = distance % trackLength;
  const mounted = splitMountedRewardSensors(state);
  const touchSource = firstDefined(sensors.touch_force_n, sensors.touch_forces_n, sensors.touch);
  const frontTouchFallback = Array.isArray(touchSource) ? touchSource[0] : firstDefined(touchSource?.front, touchSource?.front_n);
  const rearTouchFallback = Array.isArray(touchSource) ? touchSource[1] : firstDefined(touchSource?.rear, touchSource?.rear_n);

  document.body.classList.toggle("is-running", Boolean(state.running));
  document.body.classList.toggle("is-agent-evaluation", agentEvaluation);
  ui.engine.textContent = state.engine || "MuJoCo";
  ui.simTime.textContent = formatTime(state.sim_time);
  ui.tick.textContent = String(Math.max(0, Math.round(number(state.tick))));
  ui.runState.textContent = agentEvaluation
    ? "Agent evaluation"
    : state.running ? "Running" : "Paused";
  ui.moduleCount.textContent = String(modules.length);
  ui.trackLength.textContent = fixed(trackLength, 0);
  ui.trackTotal.textContent = fixed(trackLength, 0);
  ui.trackProgress.textContent = fixed(progress, 1);
  ui.progressBar.max = trackLength;
  ui.progressBar.value = progress;
  ui.progressBar.textContent = `${Math.round(progress / trackLength * 100)}%`;
  ui.canvasSpeed.textContent = `${fixed(robot.speed)} m/s`;
  ui.gaugeMotion.textContent = fixed(robot.speed);
  ui.gaugeBalance.textContent = fixed(firstDefined(sensors.tilt_deg, robot.tilt_deg), 1);
  ui.bodyState.textContent = agentEvaluation
    ? "policy control"
    : state.running ? "mobile" : "holding";
  ui.speed.textContent = fixed(robot.speed);
  ui.distance.textContent = fixed(distance);
  ui.rewardRate.textContent = fixed(state.reward?.total_rate);
  const energyPercent = number(firstDefined(
    state.power_system?.energy_fraction,
    sensors.energy_fraction,
  ), 1) * 100;
  ui.batterySoc.textContent = fixed(energyPercent, 1);
  ui.gaugeEnergy.textContent = fixed(energyPercent, 0);
  setRewardSensor(
    ui.rewardSensors.ambient,
    mounted.ambient,
    firstDefined(
      componentObservation(mounted.ambient, "illuminance_lux", "ambient_light_lux", "value"),
      sensors.ambient_light_lux,
      sensors.light_lux,
    ),
    "lx",
    "sensor · light",
  );
  if (waitingForFirstLight(state)) {
    ui.rewardSensors.ambient.value.textContent = "—";
    ui.rewardSensors.ambient.reward.textContent = "Waiting for first sample";
  }
  setRewardSensor(
    ui.rewardSensors.front,
    mounted.front,
    firstDefined(
      componentObservation(mounted.front, "normal_force_n", "force_n", "value"),
      frontTouchFallback,
    ),
    "N",
    "sensor · front",
  );
  setRewardSensor(
    ui.rewardSensors.rear,
    mounted.rear,
    firstDefined(
      componentObservation(mounted.rear, "normal_force_n", "force_n", "value"),
      rearTouchFallback,
    ),
    "N",
    "sensor · rear",
  );
  ui.playPauseLabel.textContent = agentEvaluation
    ? "Return to demo"
    : state.running ? "Pause simulation" : "Start simulation";
  ui.playPause.setAttribute("aria-pressed", String(Boolean(state.running)));

  const commands = vector(actuators.commands, ui.actuators.length);
  const torques = vector(actuators.torque_nm ?? actuators.effort, ui.actuators.length);
  const feedback = feedbackRows(state, actuators);
  ui.actuators.forEach((row, index) => {
    const current = finite(feedbackValue(actuators, feedback, index, ["current_a", "motor_current_a"]));
    const temperature = finite(feedbackValue(actuators, feedback, index, ["temperature_c", "motor_temperature_c"]));
    const impedance = finite(feedbackValue(actuators, feedback, index, [
      "load_impedance_nm_s_per_rad",
      "impedance_nm_s_per_rad",
      "load_impedance",
      "impedance",
    ]));
    const stuckScore = finite(feedbackValue(actuators, feedback, index, ["stuck_score", "stall_score"]));
    const stuckRaw = feedbackValue(actuators, feedback, index, ["stuck", "is_stuck", "stalled"]);
    const stuck = typeof stuckRaw === "boolean" ? stuckRaw : stuckScore === null ? null : stuckScore >= 0.8;
    row.querySelector("[data-target]").textContent = `u ${fixed(commands[index])}`;
    row.querySelector("[data-current]").textContent = current === null ? "—" : `${fixed(current)} A`;
    row.querySelector("[data-temperature]").textContent = temperature === null ? "—" : `${fixed(temperature, 1)} °C`;
    row.querySelector("[data-impedance]").textContent = impedance === null ? "—" : fixed(impedance, 2);
    row.querySelector("[data-stuck]").textContent = stuckScore === null
      ? "— · —"
      : `${stuck ? "yes" : "no"} · ${fixed(stuckScore, 2)}`;
    row.querySelector("[data-stuck-state]").dataset.stuckState = String(stuck === true);
    const bar = row.querySelector("[data-effort]");
    bar.max = 1.7;
    bar.value = Math.min(bar.max, Math.abs(torques[index]));
    bar.textContent = `${Math.round(bar.value / bar.max * 100)}%`;
  });
  updatePlayground(state);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const SENSOR_NAMES = {
  power_v0: "Power monitor",
  imu_6axis_v0: "Six-axis IMU",
  touch_force_v0: "Touch & force",
  proximity_tof_v0: "Time-of-flight proximity",
  ambient_light_v0: "Ambient light",
};

function familyName(familyId) {
  return SENSOR_NAMES[familyId] ?? familyId
    .replace(/_v\d+$/i, "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function listFrom(...candidates) {
  const candidate = firstDefined(...candidates, []);
  if (Array.isArray(candidate)) return candidate;
  if (candidate && typeof candidate === "object") return Object.values(candidate);
  return [];
}

function renderCatalog(catalog) {
  const motorEntries = Object.entries(catalog.motor_skus ?? {});
  const sensorEntries = Object.entries(catalog.reward_sensor_families ?? catalog.sensor_families ?? {})
    .filter(([, sensor]) => sensor.reward !== false)
    .slice(0, 5);
  const instances = listFrom(
    catalog.runtime_demo?.reward_sensor_instances,
    catalog.runtime_demo?.mounted_reward_sensors,
    catalog.runtime_demo?.sensor_instances,
  ).filter((instance) => /ambient|light|touch|contact/.test(componentText(instance))).slice(0, 3);
  const actuatorFeedbackInstances = listFrom(
    catalog.runtime_demo?.actuator_feedback_instances,
    catalog.runtime_demo?.motor_instances,
  ).slice(0, 4);
  const familyCount = sensorEntries.length || 5;
  const mountedCount = instances.length || 3;
  const feedbackCount = actuatorFeedbackInstances.length || 4;
  const installedCounts = new Map();
  instances.forEach((instance) => {
    const familyId = instance.family_id ?? instance.family ?? instance.type;
    installedCounts.set(familyId, (installedCounts.get(familyId) ?? 0) + 1);
  });

  ui.catalogSummary.textContent = `${familyCount} optional reward sensor families · ${mountedCount} mounted reward sensor instances · ${feedbackCount} actuator-feedback sets`;
  ui.motorCatalog.replaceChildren();
  ui.sensorCatalog.replaceChildren();

  for (const [skuId, motor] of motorEntries) {
    const card = element("article", "catalog-contract");
    const header = element("header");
    header.append(element("h3", "", motor.description ?? skuId));
    header.append(element("span", "catalog-badge", `${feedbackCount} mounted sets`));
    card.append(header);
    card.append(element("p", "", "Each motor reports its own policy and safety feedback. It is deliberately outside the reward-voting bus."));
    const facts = element("div", "catalog-facts");
    const action = motor.action ?? {};
    const physics = motor.physics ?? {};
    const feedback = motor.actuator_feedback ?? catalog.actuator_feedback ?? {};
    const fieldNames = (feedback.fields ?? []).map((field) => field.name ?? field).filter(Boolean);
    [
      `action ${action.name ?? "effort"} [${action.minimum ?? -1}, ${action.maximum ?? 1}]`,
      `current + temperature`,
      fieldNames.some((name) => String(name).includes("impedance")) ? "load impedance" : "impedance when available",
      fieldNames.some((name) => String(name).includes("stuck_score")) ? "stuck flag + score" : "stuck estimate when available",
      `peak current ${fixed(physics.peak_current_a, 1)} A`,
      `reward vote false`,
    ].forEach((fact) => facts.append(element("span", "", fact)));
    card.append(facts);
    ui.motorCatalog.append(card);
  }

  if (!motorEntries.length) {
    const card = element("article", "catalog-contract");
    const header = element("header");
    header.append(element("h3", "", "Four mounted motor feedback sets"));
    header.append(element("span", "catalog-badge", "no reward vote"));
    card.append(header, element("p", "", "Current, temperature, load impedance, and stuck score remain observable even while the catalog service is warming up."));
    ui.motorCatalog.append(card);
  }

  for (const [familyId, sensor] of sensorEntries) {
    const installed = installedCounts.get(familyId) ?? 0;
    const card = element("article", "sensor-contract");
    card.dataset.installed = String(installed > 0);
    const header = element("header");
    header.append(element("h3", "", familyName(familyId)));
    header.append(element("span", "catalog-badge", installed ? `${installed} mounted` : "catalogued"));
    card.append(header);
    const channels = (sensor.channels ?? []).map((channel) => channel.name).join(" · ");
    card.append(element("p", "", `${sensor.mounting ?? "module-local"}. ${channels}`));
    card.append(element("code", "", "one bounded reward vote ∈ [-1, 1] r/s"));
    ui.sensorCatalog.append(card);
  }
}

async function loadCatalog() {
  try {
    const response = await fetch("/api/catalog", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderCatalog(await response.json());
  } catch (error) {
    ui.catalogSummary.textContent = "5 optional reward sensor families · 3 mounted reward sensor instances · 4 actuator-feedback sets · catalog offline";
  }
}

function acceptState(state) {
  const sessionKey = state.playground ? `${state.playground.session_id}:${state.playground.mode}` : "legacy";
  if (sessionKey !== lastSession || number(state.sim_time) < number(latest?.sim_time)) {
    const previousKey = trailKey(latest?.playground);
    if (previousKey && trail.length > 1) {
      previousTrails.set(previousKey, trail.slice());
    }
    trail = [];
    lastTrail = null;
    visualPosition = vector(state.robot?.position, 3);
    lastSession = sessionKey;
  }
  latest = state;
  const position = vector(state.robot?.position, 3);
  if (!lastTrail || Math.hypot(position[0] - lastTrail[0], position[1] - lastTrail[1]) > 0.04) {
    trail.push(position);
    lastTrail = position;
    if (trail.length > 600) trail.shift();
  }
  updateTelemetry(state);
  updateControlAvailability();
}

function trailKey(session) {
  if (session?.mode === "learned") return `learned:${session.side}`;
  if (session?.mode === "body_lab") return `body:${session.side}:${session.body_experiment?.variant}:${session.body_experiment?.strategy}`;
  return null;
}

async function pollState() {
  if (polling || controlling || document.hidden) return;
  polling = true;
  const generation = controlGeneration;
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), 5000);
  try {
    const response = await fetch("/api/state", { cache: "no-store", signal: abort.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    if (generation !== controlGeneration) return;
    const wasConnected = connection === "connected";
    acceptState(state);
    setConnection("connected");
    if (!wasConnected) announce(`${state.engine || "Simulation"} connected.`);
  } catch (error) {
    if (generation === controlGeneration) setConnection("disconnected", `No response from /api/state (${error.message}).`);
  } finally {
    window.clearTimeout(timeout);
    polling = false;
  }
}

async function sendControl(action, side, legacy = false, options = {}) {
  if (controlling || connection !== "connected") return;
  controlling = true;
  ++controlGeneration;
  updateControlAvailability();
  controlFailure = "";
  ui.controlError.hidden = true;
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), 5000);
  try {
    const response = await fetch(legacy ? "/api/control" : "/api/playground/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(side === undefined ? { action, ...options } : { action, side, ...options }),
      signal: abort.signal,
    });
    const state = await response.json();
    if (!response.ok) throw new Error(state.error || `HTTP ${response.status}`);
    acceptState(state);
  } catch (error) {
    controlFailure = `The command could not be confirmed: ${error.message}. Checking the workshop state.`;
    ui.controlError.textContent = controlFailure;
    ui.controlError.hidden = false;
  } finally {
    window.clearTimeout(timeout);
    controlling = false;
    updateControlAvailability();
  }
}

function prepareBody(options = {}) {
  return sendControl("body", options.side ?? latest?.playground?.side ?? "left", false, {
    variant: options.variant ?? bodySelection.variant,
    strategy: options.strategy ?? bodySelection.strategy,
  });
}

function resetCurrentExperiment() {
  const session = latest?.playground;
  if (session?.mode === "body_lab") prepareBody();
  else if (session?.mode === "demo") sendControl("reset", undefined, true);
  else sendControl("reset", session?.side ?? "left");
}

ui.playPause.addEventListener("click", () => {
  const session = latest?.playground;
  if (session?.status === "completed") resetCurrentExperiment();
  else sendControl(session?.status === "running" ? "pause" : "play");
});
ui.reset.addEventListener("click", resetCurrentExperiment);
ui.sunSides.forEach((button) => button.addEventListener("click", () => {
  if (latest?.playground?.mode === "body_lab") prepareBody({ side: button.dataset.sunSide });
  else sendControl("reset", button.dataset.sunSide);
}));
ui.sunseekerEntry.addEventListener("click", () => sendControl("reset", latest?.playground?.side ?? "left"));
ui.bodyEntry.addEventListener("click", () => prepareBody());
ui.bodyVariants.forEach((button) => button.addEventListener("click", () => prepareBody({ variant: button.dataset.bodyVariant })));
ui.bodyStrategies.forEach((button) => button.addEventListener("click", () => prepareBody({ strategy: button.dataset.bodyStrategy })));
ui.learnedMode.addEventListener("click", () => sendControl("reset", latest?.playground?.side ?? "right"));
ui.demoMode.addEventListener("click", () => sendControl("demo"));
ui.viewToggle.addEventListener("click", () => {
  const workbench = document.body.classList.toggle("workbench-view");
  ui.viewToggle.setAttribute("aria-pressed", String(!workbench));
  ui.viewToggle.textContent = workbench ? "Airship view" : "Close-up view";
  resizeCanvas();
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) pollState(); });
ui.detailsToggle.addEventListener("click", () => {
  if (typeof ui.detailsDialog.showModal === "function") {
    ui.detailsDialog.showModal();
  } else {
    ui.detailsDialog.setAttribute("open", "");
  }
  ui.detailsToggle.setAttribute("aria-expanded", "true");
  ui.detailsClose.focus();
});
ui.detailsDialog.addEventListener("close", () => {
  ui.detailsToggle.setAttribute("aria-expanded", "false");
  ui.detailsToggle.focus();
});

function resizeCanvas() {
  const rectangle = ui.canvas.getBoundingClientRect();
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rectangle.width * ratio));
  const height = Math.max(1, Math.round(rectangle.height * ratio));
  if (ui.canvas.width !== width || ui.canvas.height !== height) {
    ui.canvas.width = width;
    ui.canvas.height = height;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width: rectangle.width, height: rectangle.height };
}

new ResizeObserver(resizeCanvas).observe(ui.canvasShell);

function pathRoundRect(x, y, width, height, radius) {
  const r = Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2);
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, r);
}

function projection(width, height, camera) {
  const closeUp = document.body.classList.contains("workbench-view");
  const scale = closeUp ? Math.max(100, Math.min(190, width / 5)) : Math.max(62, Math.min(104, width / 9));
  return {
    scale,
    point(position) {
      const [x, y, z] = vector(position, 3);
      return [
        width * 0.49 + (x - camera[0]) * scale + (y - camera[1]) * scale * 0.34,
        height * 0.75 - z * scale - (y - camera[1]) * scale * 0.16,
      ];
    },
  };
}

function drawGround(width, height, project, camera) {
  const horizon = Math.max(110, height * 0.28);
  const gradient = ctx.createLinearGradient(0, horizon, 0, height);
  gradient.addColorStop(0, colors.groundFar);
  gradient.addColorStop(1, colors.ground);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, horizon, width, height - horizon);
  ctx.lineWidth = 1;
  ctx.strokeStyle = colors.line;
  const startX = Math.floor(camera[0] - width / project.scale) - 2;
  const endX = Math.ceil(camera[0] + width / project.scale) + 2;
  const gridCenterY = Math.floor(camera[1]);
  const gridStartY = gridCenterY - 6;
  const gridEndY = gridCenterY + 6;
  for (let x = startX; x <= endX; x += 1) {
    const a = project.point([x, camera[1] - 4, 0]);
    const b = project.point([x, camera[1] + 4, 0]);
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  }
  for (let y = gridStartY; y <= gridEndY; y += 1) {
    const a = project.point([startX, y, 0]);
    const b = project.point([endX, y, 0]);
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(201,255,103,.38)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([7, 10]);
  const centerA = project.point([startX, 0, 0.004]);
  const centerB = project.point([endX, 0, 0.004]);
  ctx.beginPath(); ctx.moveTo(...centerA); ctx.lineTo(...centerB); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = colors.muted;
  ctx.font = "10px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "center";
  for (let x = startX; x <= endX; x += 2) {
    const point = project.point([x, camera[1] - 2.55, 0]);
    ctx.fillText(`${x}m`, point[0], point[1] + 14);
  }
}

function drawTrail(project) {
  const previous = previousTrails.get(trailKey(latest?.playground));
  if (previous?.length > 1) {
    ctx.save();
    ctx.strokeStyle = "rgba(233,211,158,.28)";
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 7]);
    ctx.beginPath();
    previous.forEach((position, index) => {
      const point = project.point([position[0], position[1], 0.025]);
      if (index === 0) ctx.moveTo(...point); else ctx.lineTo(...point);
    });
    ctx.stroke();
    ctx.restore();
  }
  if (trail.length < 2) return;
  ctx.lineCap = "round";
  for (let index = 1; index < trail.length; index += 1) {
    const a = project.point([trail[index - 1][0], trail[index - 1][1], 0.014]);
    const b = project.point([trail[index][0], trail[index][1], 0.014]);
    ctx.strokeStyle = `rgba(101,232,207,${index / trail.length * 0.28})`;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  }
}

function drawWheel(center, radius, angle, faded) {
  ctx.save();
  ctx.translate(center[0], center[1]);
  ctx.globalAlpha = faded ? 0.58 : 1;
  ctx.fillStyle = colors.wheel;
  ctx.strokeStyle = colors.wheelEdge;
  ctx.lineWidth = Math.max(1.5, radius * 0.1);
  ctx.beginPath(); ctx.arc(0, 0, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.rotate(angle);
  ctx.strokeStyle = colors.cyan;
  ctx.lineWidth = Math.max(1, radius * 0.075);
  for (let spoke = 0; spoke < 3; spoke += 1) {
    ctx.rotate(Math.PI * 2 / 3);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(radius * 0.72, 0); ctx.stroke();
  }
  ctx.fillStyle = colors.signal;
  ctx.beginPath(); ctx.arc(0, 0, radius * 0.14, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawBlock(center, width, height, fill, edge) {
  ctx.fillStyle = "rgba(0,0,0,.24)";
  pathRoundRect(center[0] - width * 0.52, center[1] + height * 0.46, width * 1.04, height * 0.25, 7);
  ctx.fill();
  ctx.fillStyle = fill;
  pathRoundRect(center[0] - width / 2, center[1] - height / 2, width, height, 7);
  ctx.fill();
  ctx.strokeStyle = edge;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,.12)";
  ctx.fillRect(center[0] - width / 2 + 7, center[1] - height / 2 + 2, width - 14, 3);
}

function robotLocalPosition(robot, forward, lateral, height) {
  const base = vector(robot.position, 3);
  const yaw = number(robot.yaw);
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  return [
    base[0] + forward * cosine - lateral * sine,
    base[1] + forward * sine + lateral * cosine,
    base[2] + height,
  ];
}

function drawMountedSensor(center, kind, active, illuminance = 0) {
  ctx.save();
  ctx.translate(center[0], center[1] - 4);
  if (kind === "light") {
    const intensity = Math.sqrt(Math.max(0, Math.min(1, number(illuminance) / 1000)));
    const radius = 11 + intensity * 25;
    const glow = ctx.createRadialGradient(0, 0, 3, 0, 0, radius);
    glow.addColorStop(0, `rgba(242,233,126,${0.2 + intensity * 0.55})`);
    glow.addColorStop(1, "rgba(232,225,112,0)");
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(0, 0, radius, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = `rgba(222,222,126,${0.18 + intensity * 0.45})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(0, 0, 10 + intensity * 9, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = `rgb(${Math.round(130 + intensity * 125)},${Math.round(151 + intensity * 95)},90)`;
    ctx.beginPath(); ctx.arc(0, 0, 6 + intensity * 2, 0, Math.PI * 2); ctx.fill();
  } else {
    if (active) {
      ctx.fillStyle = "rgba(101,232,207,.13)";
      ctx.beginPath(); ctx.arc(0, 0, 13, 0, Math.PI * 2); ctx.fill();
    }
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = active ? colors.signal : colors.cyan;
    ctx.fillRect(-4.5, -4.5, 9, 9);
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(-4.5, -4.5, 9, 9);
  }
  ctx.restore();
}

function drawRobot(project, state, now) {
  const robot = state?.robot ?? {};
  const modules = Array.isArray(robot.modules) ? robot.modules : [];
  const wheels = modules.filter((module) => module.type === "wheel");
  const wheelAngles = vector(robot.wheel_angles, 4);
  const experiment = state.playground?.mode === "body_lab" ? state.playground.body_experiment : null;
  const center = project.point(robot.position);
  const sorted = wheels.map((wheel, index) => ({ ...wheel, index }))
    .sort((a, b) => number(b.position?.[1]) - number(a.position?.[1]));

  sorted.filter((wheel) => number(wheel.position?.[1]) > 0).forEach((wheel) =>
    drawWheel(project.point(wheel.position), project.scale * 0.12, wheelAngles[wheel.index], true));
  ctx.strokeStyle = "rgba(172,212,195,.34)";
  ctx.lineWidth = 3;
  for (const wheel of wheels) {
    const point = project.point(wheel.position);
    ctx.beginPath(); ctx.moveTo(...center); ctx.lineTo(...point); ctx.stroke();
  }
  drawBlock(center, project.scale * 0.72, project.scale * 0.24, colors.chassis, colors.chassisLight);
  modules.filter((module) => module.type === "power").forEach((module) =>
    drawBlock(project.point(module.position), project.scale * 0.23, project.scale * 0.1, colors.power, "#ffc46f"));
  const sensorModule = modules.find((item) => item.type === "sensor");
  const ambientPosition = sensorModule?.position ?? robotLocalPosition(robot, 0.1, 0, 0.13);
  const touchForces = vector(firstDefined(state.sensors?.touch_force_n, state.sensors?.touch_forces_n), 2);
  drawMountedSensor(project.point(ambientPosition), "light", false, firstDefined(state.playground?.illuminance_lux, state.sensors?.ambient_light_lux));
  drawMountedSensor(project.point(robotLocalPosition(robot, 0.38, 0, 0.02)), "touch", touchForces[0] > 0.02);
  drawMountedSensor(project.point(robotLocalPosition(robot, -0.38, 0, 0.02)), "touch", touchForces[1] > 0.02);
  sorted.filter((wheel) => number(wheel.position?.[1]) <= 0).forEach((wheel) =>
    drawWheel(project.point(wheel.position), project.scale * 0.12, wheelAngles[wheel.index], false));
  ctx.fillStyle = colors.ink;
  ctx.font = "600 10px SFMono-Regular, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.fillText("DB–01", center[0], center[1] + 3);
  if (experiment) drawBodyConnections(project, wheels, center, experiment, state.body_experiment_physics, state.playground.status);
}

function drawBodyConnections(project, wheels, center, experiment, physics, status) {
  const motors = Array.isArray(experiment.motors) ? experiment.motors : [];
  ctx.save();
  wheels.forEach((wheel, index) => {
    const point = project.point(wheel.position);
    const radius = project.scale * 0.12;
    const active = ["running", "paused"].includes(status) && experiment.phase === "probing" && (experiment.active_motor_id === motors[index]?.module_id || experiment.strategy === "together");
    const changed = wheel.name === "module_front_left_wheel" && physics?.changed_motor_id === "motor-0001";
    const coupling = physics?.connected === false ? "disconnected" : number(physics?.shaft_to_wheel_ratio) < 0 ? "reversed" : "normal";
    ctx.strokeStyle = active ? "#f9da89" : changed ? "#d0acfa" : "#637a71";
    ctx.lineWidth = active ? 2.5 : 1;
    if (active || changed) {
      ctx.beginPath(); ctx.arc(point[0], point[1], radius + 5, 0, Math.PI * 2); ctx.stroke();
    }
    const labelY = point[1] + (number(wheel.position?.[1]) > 0 ? -radius - 13 : radius + 17);
    ctx.font = `${active ? "700" : "500"} 10px Consolas, monospace`;
    ctx.fillStyle = active ? "#f9da89" : changed ? "#d0acfa" : "#bdc9ba";
    ctx.textAlign = "center";
    const symbol = changed ? coupling === "reversed" ? " ↶" : coupling === "disconnected" ? " ⋯" : " =" : "";
    ctx.fillText(`M${String(index + 1).padStart(2, "0")}${symbol}`, point[0], labelY);
    if (!changed) return;
    // This drawing uses privileged physics telemetry, never a controller inference.
    const x = center[0] * 0.4 + point[0] * 0.6;
    const y = center[1] * 0.4 + point[1] * 0.6;
    const angle = Math.atan2(point[1] - center[1], point[0] - center[0]);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = "#192b2a";
    pathRoundRect(-15, -8, 30, 16, 4); ctx.fill();
    ctx.strokeStyle = "#d0acfa";
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (coupling === "disconnected") {
      for (const offset of [-3, 3]) {
        ctx.moveTo(-11, offset); ctx.lineTo(-4, offset);
        ctx.moveTo(4, offset); ctx.lineTo(11, offset);
      }
    } else {
      ctx.moveTo(-11, -3); ctx.lineTo(11, coupling === "reversed" ? 3 : -3);
      ctx.moveTo(-11, 3); ctx.lineTo(11, coupling === "reversed" ? -3 : 3);
    }
    ctx.stroke();
    // The little shaft can spin independently of the wheel when disconnected.
    ctx.translate(-11, 0);
    ctx.rotate(number(physics.shaft_position_rad));
    ctx.fillStyle = "#192b2a";
    ctx.beginPath(); ctx.arc(0, 0, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-3.5, 0); ctx.lineTo(3.5, 0); ctx.stroke();
    ctx.restore();
  });
  ctx.restore();
}

function drawSunshine(project, state, width, height) {
  const source = state.playground?.light_position_m;
  if (!Array.isArray(source)) return;
  const mapped = project.point(source);
  const margin = width < 500 ? 55 : 95;
  const x = Math.max(margin, Math.min(width - margin, mapped[0]));
  const y = Math.max(108, height * 0.44);
  const offscreen = Math.abs(x - mapped[0]) > 1;
  const robotX = number(state.robot?.position?.[0]);
  const distance = Math.abs(number(source[0]) - robotX);
  ctx.save();
  const halo = ctx.createRadialGradient(x, y, 10, x, y, 70);
  halo.addColorStop(0, "rgba(253,205,101,.23)");
  halo.addColorStop(1, "rgba(253,205,101,0)");
  ctx.fillStyle = halo;
  ctx.beginPath(); ctx.arc(x, y, 70, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#d8b567";
  ctx.lineWidth = 1.5;
  for (let ray = 0; ray < 10; ray += 1) {
    const angle = ray * Math.PI / 5;
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(angle) * 25, y + Math.sin(angle) * 25);
    ctx.lineTo(x + Math.cos(angle) * 31, y + Math.sin(angle) * 31);
    ctx.stroke();
  }
  ctx.fillStyle = "#edcd7e";
  ctx.beginPath(); ctx.arc(x, y, 17, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#e6d0a0";
  ctx.font = "12px Georgia, serif";
  ctx.textAlign = "center";
  ctx.fillText(offscreen ? `${number(source[0]) < robotX ? "← " : ""}Sunshine${number(source[0]) > robotX ? " →" : ""}` : "Sunshine", x, y + 49);
  ctx.font = "10px Consolas, monospace";
  ctx.fillStyle = "#b7b18b";
  ctx.fillText(`${fixed(distance, 1)} m away`, x, y + 65);
  ctx.restore();
}

function drawWorldMap(state, width) {
  const source = state.playground?.light_position_m;
  if (!Array.isArray(source) || !["learned", "body_lab"].includes(state.playground?.mode)) return;
  const edge = width < 500 ? 28 : 50;
  const mapX = (x) => edge + Math.max(0, Math.min(1, (number(x) + 22) / 44)) * (width - edge * 2);
  const y = 69;
  ctx.save();
  ctx.strokeStyle = "rgba(223,210,162,.23)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(edge, y); ctx.lineTo(width - edge, y); ctx.stroke();
  ctx.textAlign = "center";
  ctx.font = "10px Consolas, monospace";
  ctx.fillStyle = "#96a28a";
  for (const x of [-20, 0, 20]) {
    ctx.beginPath(); ctx.moveTo(mapX(x), y - 3); ctx.lineTo(mapX(x), y + 4); ctx.stroke();
    ctx.fillText(x === 0 ? "start" : `${x} m`, mapX(x), y + 20);
  }
  ctx.fillStyle = "#ecc879";
  ctx.beginPath(); ctx.arc(mapX(source[0]), y, 6, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = colors.cyan;
  pathRoundRect(mapX(state.robot?.position?.[0]) - 5, y - 5, 10, 10, 2);
  ctx.fill();
  ctx.textAlign = "left";
  ctx.font = "11px Georgia, serif";
  ctx.fillStyle = "#adc0a8";
  ctx.fillText("A little map of its world", edge, y - 20);
  if (previousTrails.has(trailKey(state.playground))) {
    ctx.font = "10px Georgia, serif";
    ctx.fillText(state.playground.mode === "body_lab" ? "Dotted trail: last try with these choices" : "Dotted trail: your last try on this side", edge, y + 37);
  }
  ctx.restore();
}

function drawEmpty(width, height, now) {
  const gradient = ctx.createRadialGradient(width / 2, height / 2, 10, width / 2, height / 2, width * 0.55);
  gradient.addColorStop(0, "#10231d");
  gradient.addColorStop(1, colors.page);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(201,255,103,.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(width / 2, height / 2, 38 + (reducedMotion.matches ? 0 : now / 30 % 34), 0, Math.PI * 2);
  ctx.stroke();
}

function render(now) {
  const { width, height } = resizeCanvas();
  const delta = Math.min(0.08, Math.max(0, (now - lastFrame) / 1000));
  lastFrame = now;
  ctx.clearRect(0, 0, width, height);
  if (!latest) {
    drawEmpty(width, height, now);
  } else {
    const target = vector(latest.robot?.position, 3);
    const blend = reducedMotion.matches ? 1 : 1 - Math.exp(-delta * 9);
    visualPosition = visualPosition.map((value, index) => value + (target[index] - value) * blend);
    const project = projection(width, height, visualPosition);
    ctx.fillStyle = colors.page;
    ctx.fillRect(0, 0, width, height);
    drawGround(width, height, project, visualPosition);
    drawSunshine(project, latest, width, height);
    drawTrail(project);
    drawRobot(project, latest, now);
    drawWorldMap(latest, width);
  }
  window.requestAnimationFrame(render);
}

setConnection("connecting");
loadCatalog();
pollState();
window.setInterval(pollState, 100);
window.requestAnimationFrame(render);
