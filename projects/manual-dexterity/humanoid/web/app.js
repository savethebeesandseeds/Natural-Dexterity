"use strict";
const $ = id => document.getElementById(id);
let playing = true;
let frameBusy = false;
let commandError = "";
let pendingReset = null;
const first = (state, ...keys) => {
  for (const key of keys) if (state[key] !== undefined) return state[key];
  return null;
};
const number = (value, unit = "", places = 2) =>
  Number.isFinite(value) ? value.toFixed(places) + unit : "—";

async function control(action) {
  try {
    const payload = {action};
    if (action === "reset") Object.assign(payload, {
      command_x: Number($("speed").value), policy: $("policy").value, seed: 0
    });
    if (action === "reset") pendingReset = payload;
    const response = await fetch("/api/control", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error((await response.json()).error || "Control request failed");
    commandError = "";
    $("error").textContent = "";
  } catch (error) {
    pendingReset = null;
    commandError = error.message;
    $("error").textContent = commandError;
  }
}
$("toggle").onclick = () => control(playing ? "pause" : "play");
$("reset").onclick = () => control("reset");
$("policy").onchange = () => control("reset");
$("speed").oninput = () => $("speed-value").textContent = number(Number($("speed").value), " m/s");
$("speed").onchange = () => control("reset");

function foot(state, side) {
  const contacts = state.feet_contacts || state.foot_contacts || state.contacts || {};
  return Boolean(first(state, side + "_contact", side + "_foot_contact") ?? contacts[side]);
}

function drawContacts(history) {
  const canvas = $("contacts"), context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#e3e7de";
  context.fillRect(0, 10, canvas.width, 18);
  context.fillRect(0, 38, canvas.width, 18);
  const width = canvas.width / 160;
  history.slice(-160).forEach((state, index) => {
    context.fillStyle = "#608a58";
    if (foot(state, "left")) context.fillRect(index * width, 10, width + 1, 18);
    context.fillStyle = "#9baf63";
    if (foot(state, "right")) context.fillRect(index * width, 38, width + 1, 18);
  });
}

async function update() {
  try {
    const response = await fetch("/api/state");
    if (!response.ok) throw new Error("Viewer unavailable");
    const state = await response.json();
    if (state.error) throw new Error(state.error);
    $("connection").textContent = state.ready ? "● MuJoCo connected" : "Preparing simulation";
    $("toggle").disabled = !state.ready;
    $("reset").disabled = !state.ready;
    $("trained-policy").disabled = !state.trained_policy_available;
    $("distilled-policy").disabled = !state.distilled_policy_available;
    $("trained-availability").textContent = "Local checkpoints: PPO "
      + (state.trained_policy_available ? "available" : "unavailable") + " · imitation "
      + (state.distilled_policy_available ? "available" : "unavailable") + ".";
    if (!state.ready) return;
    if (state.control_error || (pendingReset && state.policy === pendingReset.policy
      && Math.abs(state.command_x - pendingReset.command_x) < 1e-6)) pendingReset = null;
    if (!pendingReset && document.activeElement !== $("policy")) $("policy").value = state.policy;
    if (!pendingReset && document.activeElement !== $("speed") && Number.isFinite(state.command_x)) {
      $("speed").value = state.command_x;
      $("speed-value").textContent = number(state.command_x, " m/s");
    }
    playing = state.playing;
    const time = first(state, "time_s", "time");
    const horizon = state.horizon_s || 30;
    const fallen = Boolean(first(state, "fallen", "fall"));
    const complete = Number.isFinite(time) && time + 1e-7 >= horizon;
    $("toggle").textContent = playing ? "Pause" : "Play";
    $("toggle").disabled = fallen || complete;
    $("clock").textContent = number(time, " s") + " / " + horizon + " s";
    const realtime = state.viewer_performance?.realtime_factor;
    $("playback-rate").textContent = playing && Number.isFinite(realtime)
      ? "LIVE · " + number(realtime, "× REAL TIME") : "LIVE PHYSICS";
    $("run-status").textContent = fallen ? "FALL · TRIAL STOPPED"
      : complete ? "TRIAL COMPLETE" : playing ? "SIMULATION RUNNING" : "PAUSED";
    const distance = first(state, "forward_distance_m", "distance_m", "distance", "displacement_x_m");
    $("distance").textContent = number(distance, " m");
    $("velocity").textContent = number(first(state, "mean_forward_speed_m_s", "mean_speed_m_s",
      "mean_speed_mps", "mean_speed") ?? (time > 0 ? distance / time : 0), " m/s");
    $("height").textContent = number(first(state, "torso_height_m", "pelvis_height_m", "height_m", "torso_height"), " m");
    $("steps").textContent = first(state, "alternating_count", "alternating_contacts",
      "alternating_contact_count", "alternations", "alternating_steps") ?? "—";
    $("left-foot").classList.toggle("on", foot(state, "left"));
    $("right-foot").classList.toggle("on", foot(state, "right"));
    drawContacts(state.history || []);
    const provenance = state.provenance || {};
    const inferenceDevice = provenance.inference_device || provenance.device || provenance.policy_device || "loaded";
    const trained = state.policy === "trained";
    const distilled = state.policy === "distilled";
    const passive = state.policy === "zero";
    $("evaluation-link").href = distilled ? "/api/evaluation-distilled"
      : trained ? "/api/evaluation-trained" : "/api/evaluation";
    $("evaluation-link").textContent = (distilled ? "Imitation policy" : trained ? "PPO policy" : "Pretrained baseline")
      + " evaluation ↗";
    $("policy-description").textContent = passive
      ? "All twelve motor torques are zero. This passive baseline shows what happens without a controller."
      : distilled ? "A local neural policy trained in C++ to imitate Unitree’s controller at 0.50 m/s. Its saved evaluation measures whether the learned policy walks."
      : trained ? "Latest PPO attempt, trained in C++ from random weights. The saved 20-second replay advanced 0.85 m without falling, but showed swivelling and slipping. Walking is not yet convincingly demonstrated."
      : "Unitree’s published pretrained policy. This baseline tests execution in MuJoCo; it was not trained by this project.";
    $("runtime").textContent = "MuJoCo physics on " + (provenance.physics_device || "cpu") + " · "
      + (passive ? "zero motor torques" : (distilled ? "Our native imitation policy on " : trained ? "Our native PPO policy on " : "Unitree pretrained policy on ")
        + inferenceDevice);
    $("error").textContent = state.control_error || commandError;
  } catch (error) {
    $("connection").textContent = "Connection interrupted";
    $("toggle").disabled = true;
    $("reset").disabled = true;
    $("error").textContent = error.message;
  } finally {
    setTimeout(update, 200);
  }
}

async function frame() {
  if (!frameBusy) {
    frameBusy = true;
    try {
      const response = await fetch("/frame.jpg");
      if (response.ok) {
        const url = URL.createObjectURL(await response.blob());
        const image = $("viewport"), old = image.dataset.url;
        image.src = url;
        image.dataset.url = url;
        if (old) URL.revokeObjectURL(old);
      }
    } catch {
      // State polling displays connection failures; a dropped frame can retry.
    } finally {
      frameBusy = false;
    }
  }
  setTimeout(frame, 80);
}
update();
frame();
