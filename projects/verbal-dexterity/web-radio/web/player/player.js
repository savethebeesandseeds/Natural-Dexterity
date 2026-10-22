"use strict";

const $ = id => document.getElementById(id);
const live = $("live-audio");
const button = $("live-button");
const note = $("player-note");
const canvas = $("sound-wave");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const spiritScene = window.RadioSpirits ? new RadioSpirits.Scene(canvas, {reducedMotion}) : null;
const visualPerformance = spiritScene && window.RadioPerformance
  ? new RadioPerformance.Timeline(spiritScene, api) : null;
const previewId = new URLSearchParams(window.location.search).get("preview");
const audition = previewId && spiritScene && window.RadioSemantic
  ? new RadioSemantic.Preview(spiritScene, api) : null;
let previewRetryAfter = 0;
let hls = null;
let ready = false;
let updating = false;
let stationUnavailable = false;
let connecting = false;
let playing = false;
let playAttempt = 0;
let audioContext = null;
let analyser = null;
let mediaSource = null;
let directAudio = false;
let samples = null;
let frame = null;
let frequencySamples = null;
let lastVisualFrame = 0;
let lastBoundaryRefresh = 0;
let waveformReadyAfter = 0;

live.volume = Number($("volume").value);

async function api(path) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 7000);
  try {
    const response = await fetch(path, {cache: "no-store", signal: controller.signal});
    if (!response.ok) throw new Error("Station unavailable");
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function updateButton() {
  const active = !live.paused || connecting;
  const label = active ? (previewId ? "Pause" : "Pause live") : (previewId ? "Play" : "Listen live");
  $("button-label").textContent = connecting && live.paused ? "Connecting…" : label;
  $("play-symbol").toggleAttribute("hidden", active);
  $("pause-symbol").toggleAttribute("hidden", !active);
  button.setAttribute("aria-label", label);
  button.title = label;
  button.disabled = !ready && !active;
}

function exactPlaybackTime() {
  // Only the player's HLS programme date follows the actual playhead.
  // A server-clock/live-window estimate is insufficient for character cues.
  const date = hls && hls.playingDate;
  const timestamp = date instanceof Date ? date.getTime() : NaN;
  return Number.isFinite(timestamp) ? timestamp : null;
}

function mixedFeatures() {
  if (!analyser || !samples || !frequencySamples || live.seeking ||
      performance.now() < waveformReadyAfter) return null;
  analyser.getFloatTimeDomainData(samples);
  let sum = 0;
  for (const value of samples) sum += value * value;
  const rms = Math.sqrt(sum / samples.length);
  analyser.getFloatFrequencyData(frequencySamples);
  const nyquist = audioContext.sampleRate / 2;
  const band = (from, to) => {
    const first = Math.max(0, Math.floor(from / nyquist * frequencySamples.length));
    const last = Math.min(frequencySamples.length, Math.ceil(to / nyquist * frequencySamples.length));
    let amplitude = 0;
    for (let i = first; i < last; i++) {
      if (Number.isFinite(frequencySamples[i])) amplitude += Math.pow(10, frequencySamples[i] / 20);
    }
    return Math.min(1, amplitude / Math.max(1, last - first) * 6);
  };
  // A short signed time-domain profile shapes radius along the local axis.
  // Box averaging limits spatial aliasing; fixed gain preserves quiet/loud contrast.
  const waveform = Array.from({length:64}, (_unused, bin) => {
    const first = Math.floor(bin * samples.length / 64);
    const last = Math.floor((bin + 1) * samples.length / 64);
    let sum = 0;
    for (let index = first; index < last; index++) sum += Number.isFinite(samples[index]) ? samples[index] : 0;
    return Math.max(-1, Math.min(1, sum / Math.max(1,last-first) * 7));
  });
  return {
    energy: Math.min(1, 1 - Math.exp(-rms * 7)),
    bands: [band(80, 300), band(300, 2500), band(2500, Math.min(8000, nyquist))],
    waveform,
    waveform_window_ms: samples.length * 1000 / audioContext.sampleRate
  };
}

function stopWave() {
  if (frame !== null) cancelAnimationFrame(frame);
  frame = null;
  lastVisualFrame = 0;
  if (spiritScene) {
    spiritScene.setPlaybackActive(false);
    spiritScene.setFeatures({});
  }
}

function drawWave(timestamp) {
  frame = null;
  if (!playing || reducedMotion.matches || document.hidden || !visualPerformance) return;
  if (!lastVisualFrame || timestamp - lastVisualFrame >= 1000 / 30) {
    lastVisualFrame = timestamp;
    const boundary = audition
      ? audition.tick(live.currentTime * 1000, mixedFeatures(), true)
      : visualPerformance.tick(exactPlaybackTime(), mixedFeatures(), true);
    if (boundary && timestamp - lastBoundaryRefresh > 1000) {
      lastBoundaryRefresh = timestamp;
      refresh();
    }
  }
  frame = requestAnimationFrame(drawWave);
}

function updateWave() {
  stopWave();
  const active = playing && !reducedMotion.matches && !document.hidden;
  if (spiritScene) spiritScene.setPlaybackActive(active);
  if (active && visualPerformance) frame = requestAnimationFrame(drawWave);
}

async function prepareAnalyser() {
  // The audio graph is created only after a listener's play gesture.
  try {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!Audio) return;
    if (!audioContext) audioContext = new Audio();
    if (!mediaSource) mediaSource = audioContext.createMediaElementSource(live);
    if (!analyser && !directAudio) {
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = .65;
      samples = new Float32Array(analyser.fftSize);
      frequencySamples = new Float32Array(analyser.frequencyBinCount);
      mediaSource.connect(analyser);
      analyser.connect(audioContext.destination);
    }
    if (audioContext.state === "suspended") await audioContext.resume();
  } catch {
    // Playback still matters when a browser cannot provide visualization.
    if (mediaSource && audioContext && !directAudio) {
      try {
        // Remove any partially connected analyser graph before the direct path.
        mediaSource.disconnect();
        if (analyser) analyser.disconnect();
        analyser = null;
        samples = null;
        frequencySamples = null;
        mediaSource.connect(audioContext.destination);
        directAudio = true;
        stopWave();
      } catch {}
    }
  }
}

function resetStream(message) {
  playAttempt += 1;
  connecting = false;
  playing = false;
  if (hls) { hls.destroy(); hls = null; }
  live.pause();
  live.removeAttribute("src");
  live.load();
  updateButton();
  stopWave();
  note.textContent = message;
}

function prepareStream() {
  if (previewId) {
    if (!audition || !audition.metadata) return false;
    if (!live.getAttribute("src")) live.src = audition.metadata.audio.url;
    return true;
  }
  if (hls || live.getAttribute("src")) return true;
  if (window.Hls && window.Hls.isSupported()) {
    hls = new Hls({liveSyncDurationCount: 3, maxBufferLength: 20, backBufferLength: 30});
    hls.on(Hls.Events.ERROR, (_event, data) => {
      if (data.fatal) resetStream("The connection paused. Select Listen live to reconnect.");
    });
    hls.loadSource("/hls/live.m3u8");
    hls.attachMedia(live);
    return true;
  }
  if (live.canPlayType("application/vnd.apple.mpegurl")) {
    live.src = "/hls/live.m3u8";
    return true;
  }
  note.textContent = "Live audio is not supported by this browser.";
  return false;
}

function joinLiveEdge() {
  const target = hls && hls.liveSyncPosition;
  if (Number.isFinite(target)) live.currentTime = target;
  else if (live.seekable.length) {
    const end = live.seekable.end(live.seekable.length - 1);
    const start = live.seekable.start(live.seekable.length - 1);
    live.currentTime = Math.max(start, end - 3);
  }
}

async function listen() {
  const attempt = ++playAttempt;
  if (!live.paused || connecting) {
    connecting = false;
    live.pause();
    note.textContent = ready ? "Paused" : "";
    updateButton();
    return;
  }
  connecting = true;
  note.textContent = "Connecting…";
  updateButton();
  await prepareAnalyser();
  if (attempt !== playAttempt) return;
  if (!prepareStream()) { connecting = false; updateButton(); return; }
  try {
    if (previewId) { if(live.ended)live.currentTime=0; }
    else joinLiveEdge();
    await live.play();
  } catch {
    if (attempt === playAttempt) note.textContent = previewId
      ? "Select Play to try the recording again."
      : "Select Listen live to try the connection again.";
  } finally {
    if (attempt === playAttempt) {
      connecting = false;
      updateButton();
    }
  }
}

async function playbackTime() {
  const exact = exactPlaybackTime();
  if (exact !== null) return new Date(exact).toISOString();
  if (!live.paused && live.seekable.length) {
    try {
      const windowInfo = await api("/api/live-window");
      const behindEdge = Math.max(0, live.seekable.end(live.seekable.length - 1) - live.currentTime);
      return new Date(Date.parse(windowInfo.end) - behindEdge * 1000).toISOString();
    } catch {}
  }
  return null;
}

function mediaTime(seconds) {
  const value = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  return Math.floor(value / 60) + ":" + String(value % 60).padStart(2, "0");
}

function updatePreviewPosition() {
  if (!audition || !audition.metadata) return;
  $("preview-elapsed").textContent=mediaTime(live.currentTime);
  $("preview-seek").value=String(live.currentTime);
  if (!live.ended && note.textContent === "End of preview") note.textContent = live.paused ? "Paused" : "";
  if (!playing || reducedMotion.matches) audition.tick(live.currentTime * 1000, null, false);
}

async function refreshPreview() {
  if (!audition) { note.textContent="This preview is unavailable."; return; }
  if (audition.pending || audition.metadata || Date.now()<previewRetryAfter) return;
  try {
    const metadata=await audition.load(previewId);
    ready=true;
    document.querySelector("main").setAttribute("aria-label","Radio Waajacu local preview");
    document.querySelector(".tuner").setAttribute("aria-label","Preview recording");
    live.src=metadata.audio.url;
    $("now-label").textContent="LOCAL PREVIEW";
    $("now-title").textContent=metadata.title;
    $("programme").hidden=false;
    $("signal-text").textContent="Preview";
    $("signal").classList.add("on");
    $("preview-progress").hidden=false;
    $("preview-seek").max=String(metadata.duration_ms/1000);
    $("preview-duration").textContent=mediaTime(metadata.duration_ms/1000);
    note.textContent="";
    updateButton();
  } catch {
    ready=false;
    previewRetryAfter=Date.now()+30000;
    $("signal-text").textContent="Preview unavailable";
    note.textContent="The recording could not be loaded.";
    updateButton();
  }
}

async function refresh() {
  if (previewId) { await refreshPreview(); return; }
  if (updating) return;
  updating = true;
  try {
    const state = await api("/api/status");
    if (stationUnavailable && live.paused && !connecting) note.textContent = "";
    stationUnavailable = false;
    ready = Boolean(state.stream_ready);
    if (!ready) $("programme").hidden = true;
    $("signal-text").textContent = ready ? "On air" : state.state === "recovering" ? "Reconnecting" : "Off air";
    $("signal").classList.toggle("on", ready);
    updateButton();
    if (!ready && live.paused && !connecting) {
      if (visualPerformance) visualPerformance.setAiring(null);
      $("now-label").textContent = "LIVE RADIO";
      $("now-title").textContent = "Waiting for the broadcast";
      note.textContent = "";
      return;
    }
    const exactPosition = !live.paused ? exactPlaybackTime() : null;
    const position = !live.paused
      ? exactPosition !== null ? new Date(exactPosition).toISOString() : await playbackTime()
      : null;
    const current = await api("/api/now" + (position ? "?at=" + encodeURIComponent(position) : ""));
    $("now-label").textContent = playing && position ? "NOW PLAYING" : "ON AIR NOW";
    $("now-title").textContent = current.airing ? current.airing.title : "Radio Waajacu";
    $("programme").hidden = !current.airing;
    // Preserve a paused cast, but never assign a character from an estimated clock.
    if (visualPerformance && !live.paused) {
      visualPerformance.setAiring(exactPosition !== null ? current.airing : null);
    }
  } catch {
    stationUnavailable = true;
    $("programme").hidden = true;
    $("signal-text").textContent = "Connection unavailable";
    $("signal").classList.remove("on");
    if (live.paused && !connecting) {
      ready = false;
      updateButton();
      note.textContent = "The station is temporarily unavailable.";
    }
  } finally {
    updating = false;
  }
}

button.addEventListener("click", listen);
$("volume").addEventListener("input", event => { live.volume = Number(event.target.value); });
live.addEventListener("play", updateButton);
live.addEventListener("playing", () => {
  playing = true;
  connecting = false;
  note.textContent = "";
  updateButton();
  updateWave();
  refresh();
});
live.addEventListener("pause", () => {
  playing = false;
  updateButton();
  stopWave();
  if (ready) note.textContent = "Paused";
});
for (const name of ["waiting", "stalled"]) {
  live.addEventListener(name, () => {
    playing = false;
    stopWave();
    if (!live.paused) note.textContent = "Buffering…";
  });
}
live.addEventListener("error", () => {
  if (live.getAttribute("src")) resetStream(previewId
    ? "The recording paused. Select Play to try again."
    : "The connection paused. Select Listen live to reconnect.");
});
live.addEventListener("ended", () => {
  if (!previewId) { resetStream("The broadcast paused. Select Listen live to reconnect."); return; }
  playing=false; connecting=false; stopWave(); updateButton();
  if(audition)audition.tick(live.currentTime*1000,null,false);
  note.textContent="End of preview";
});
live.addEventListener("timeupdate", updatePreviewPosition);
live.addEventListener("seeking", () => {
  stopWave();
  waveformReadyAfter = performance.now() + 100;
  if (audition && spiritScene) {
    spiritScene.setPositions(null);
    spiritScene.setRotations(null);
    audition.tick(live.currentTime * 1000, null, false);
  }
});
live.addEventListener("seeked", () => {
  const windowMs = samples && audioContext ? samples.length * 1000 / audioContext.sampleRate : 0;
  waveformReadyAfter = performance.now() + Math.max(60, windowMs);
  updatePreviewPosition();
  updateWave();
});
$("preview-seek").addEventListener("input", event => {
  if(previewId && ready) { live.currentTime=Number(event.target.value); updatePreviewPosition(); }
});
live.addEventListener("volumechange", () => { $("volume").value = String(live.volume); });
document.addEventListener("visibilitychange", () => { updateWave(); if (!document.hidden) refresh(); });
reducedMotion.addEventListener("change", updateWave);
window.addEventListener("pagehide", event => {
  stopWave();
  if (spiritScene && !event.persisted) spiritScene.destroy();
});
window.addEventListener("pageshow", event => {
  if (event.persisted) { updateWave(); refresh(); }
});
refresh();
setInterval(refresh, 2500);
