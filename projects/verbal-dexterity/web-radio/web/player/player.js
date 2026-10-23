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
const staticSite = Boolean(document.querySelector("main").dataset.staticCatalogue);
const playerParams = new URLSearchParams(window.location.search);
const creationId = playerParams.get("creation");
const recordingId = creationId || playerParams.get("recording") || playerParams.get("preview") ||
  document.querySelector("main").dataset.defaultRecording || null;
const libraryRecording = Boolean(creationId) ||
  (staticSite && /^(?:creation|project)-[0-9a-f]{24}$/.test(recordingId || "")) ||
  (!playerParams.get("recording") && !playerParams.get("preview") &&
    /^(?:creation|project)-[0-9a-f]{24}$/.test(recordingId || ""));
const recording = recordingId && spiritScene && window.RadioSemantic
  ? new RadioSemantic.Recording(spiritScene, api) : null;
let recordingRetryAfter = 0;
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
let outputGain = null;
let directAudio = false;
let samples = null;
let waveformSums = null;
let frame = null;
let frequencySamples = null;
let lastVisualFrame = 0;
let lastBoundaryRefresh = 0;
let waveformReadyAfter = 0;

// Keep this status row empty for programme playback.
if (recordingId) {
  $("signal").classList.add("quiet");
  $("signal-text").textContent = "";
}

// Keep volume after the analyser so listening level does not reshape a voice.
let listeningVolume = 1;
let listeningMuted = false;
let lastAudibleVolume = 1;
try {
  const saved = JSON.parse(sessionStorage.getItem("radio-waajacu:audio-level") || "null");
  if (saved && Number.isFinite(saved.volume) && saved.volume >= 0 && saved.volume <= 1 &&
      typeof saved.muted === "boolean") {
    listeningVolume = saved.volume;
    listeningMuted = saved.muted;
    if (Number.isFinite(saved.lastAudible) && saved.lastAudible > 0 && saved.lastAudible <= 1) {
      lastAudibleVolume = saved.lastAudible;
    }
  }
} catch {}

function updateAudioLevel(persist = false) {
  const silent = listeningMuted || listeningVolume === 0;
  if (outputGain && audioContext) {
    // The graph measures the unattenuated recording; only its output is quieted.
    live.volume = 1;
    live.muted = false;
    const now = audioContext.currentTime;
    const previousGain = outputGain.gain.value;
    outputGain.gain.cancelScheduledValues(now);
    outputGain.gain.setValueAtTime(previousGain, now);
    outputGain.gain.linearRampToValueAtTime(silent ? 0 : listeningVolume, now + .020);
  } else {
    live.volume = listeningVolume;
    live.muted = listeningMuted;
  }
  $("audio-volume").value = String(listeningVolume);
  $("audio-volume").setAttribute("aria-valuetext", Math.round(listeningVolume * 100) + "%");
  $("volume-toggle").setAttribute("aria-label", silent ? "Unmute" : "Mute");
  $("volume-toggle").setAttribute("aria-pressed", String(silent));
  $("volume-toggle").title = silent ? "Unmute" : "Mute";
  $("volume-symbol").toggleAttribute("hidden", silent);
  $("muted-symbol").toggleAttribute("hidden", !silent);
  if (persist) {
    try {
      sessionStorage.setItem("radio-waajacu:audio-level", JSON.stringify({
        volume: listeningVolume, muted: listeningMuted, lastAudible: lastAudibleVolume
      }));
    } catch {}
  }
}

function prepareOutputGain() {
  if (!outputGain) {
    const gain = audioContext.createGain();
    gain.gain.value = listeningMuted ? 0 : listeningVolume;
    gain.connect(audioContext.destination);
    outputGain = gain;
  }
  updateAudioLevel();
}

async function api(path) {
  if (staticSite && path.startsWith("/api/")) throw new Error("Live services unavailable");
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
  const label = active ? (recordingId || staticSite ? "Pause" : "Pause live") : (recordingId || staticSite ? "Play" : "Listen live");
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
  if (!analyser || !samples || !waveformSums || !frequencySamples || live.seeking ||
      performance.now() < waveformReadyAfter) return null;
  analyser.getFloatTimeDomainData(samples);
  // Prefix sums let every contour point measure a full local RMS window.
  waveformSums[0] = 0;
  for (let index = 0; index < samples.length; index++) {
    const value = Number.isFinite(samples[index]) ? samples[index] : 0;
    waveformSums[index + 1] = waveformSums[index] + value * value;
  }
  const energySamples = Math.min(samples.length, Math.max(1, Math.round(audioContext.sampleRate * .020)));
  const rms = Math.sqrt(Math.max(0, waveformSums[samples.length] -
    waveformSums[samples.length - energySamples]) / energySamples);
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
  // The changing amplitude outline of real audio, without signed cancellation,
  // peak normalization, hard clipping, or decorative repeating ripples.
  const halfWindow = Math.max(1, Math.round(audioContext.sampleRate * .020 / 2));
  const waveformRanges = [];
  const waveform = Array.from({length:64}, (_unused, bin) => {
    const center = Math.round((bin + .5) * samples.length / 64);
    const first = Math.max(0, center - halfWindow);
    const last = Math.min(samples.length, center + halfWindow);
    const meanSquare = Math.max(0, waveformSums[last] - waveformSums[first]) / (last - first);
    waveformRanges.push([
      (first - samples.length) * 1000 / audioContext.sampleRate,
      (last - samples.length) * 1000 / audioContext.sampleRate
    ]);
    return 1 - Math.exp(-7 * Math.sqrt(meanSquare));
  });
  return {
    energy: Math.min(1, 1 - Math.exp(-rms * 7)),
    energy_window_ms: energySamples * 1000 / audioContext.sampleRate,
    bands_window_ms: samples.length * 1000 / audioContext.sampleRate,
    bands: [band(80, 300), band(300, 2500), band(2500, Math.min(8000, nyquist))],
    waveform,
    waveform_ranges_ms: waveformRanges,
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
    const boundary = recording
      ? recording.tick(live.currentTime * 1000, mixedFeatures(), true)
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
    prepareOutputGain();
    if (!analyser && !directAudio) {
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 16384;
      analyser.smoothingTimeConstant = 0;
      samples = new Float32Array(analyser.fftSize);
      waveformSums = new Float64Array(analyser.fftSize + 1);
      frequencySamples = new Float32Array(analyser.frequencyBinCount);
      mediaSource.connect(analyser);
      analyser.connect(outputGain);
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
        waveformSums = null;
        frequencySamples = null;
        prepareOutputGain();
        mediaSource.connect(outputGain);
        directAudio = true;
        stopWave();
      } catch {
        // Preserve the original audible bypass if the browser cannot create a
        // usable gain node after the media element has already been rerouted.
        try {
          mediaSource.disconnect();
          if (outputGain) outputGain.disconnect();
          outputGain = null;
          mediaSource.connect(audioContext.destination);
          directAudio = true;
          updateAudioLevel();
          stopWave();
        } catch {}
      }
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
  if (recordingId) {
    if (!recording || !recording.metadata) return false;
    if (!live.getAttribute("src")) live.src = recording.metadata.audio.url;
    return true;
  }
  if (staticSite) return false;
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
    if (recordingId) { if(live.ended)live.currentTime=0; }
    else joinLiveEdge();
    await live.play();
  } catch {
    if (attempt === playAttempt) note.textContent = recordingId
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
  if (staticSite) return null;
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

function updateRecordingPosition() {
  if (!recording || !recording.metadata) return;
  $("recording-elapsed").textContent=mediaTime(live.currentTime);
  $("recording-seek").value=String(live.currentTime);
  $("recording-played").value=live.currentTime;
  if (!live.ended && note.textContent === "End of recording") note.textContent = live.paused ? "Paused" : "";
  if (!playing || reducedMotion.matches) recording.tick(live.currentTime * 1000, null, false);
}

async function refreshRecording() {
  if (!recording) { note.textContent="This recording is unavailable."; return; }
  if (recording.pending || recording.metadata || Date.now()<recordingRetryAfter) return;
  try {
    const metadata=await recording.load(recordingId, {library: libraryRecording, staticSite});
    ready=true;
    document.querySelector("main").setAttribute("aria-label","Radio Waajacu radio");
    document.querySelector(".tuner").setAttribute("aria-label","Radio player");
    live.src=metadata.audio.url;
    $("now-label").textContent="RADIO WAAJACU";
    $("now-title").textContent=metadata.title;
    $("programme").hidden=false;
    $("signal-text").textContent="";
    $("signal").classList.add("on");
    $("recording-progress").hidden=false;
    $("recording-time").hidden=false;
    $("recording-seek").max=String(metadata.duration_ms/1000);
    $("recording-played").max=metadata.duration_ms/1000;
    $("recording-played").value=live.currentTime;
    $("recording-duration").textContent=mediaTime(metadata.duration_ms/1000);
    note.textContent="";
    updateButton();
  } catch {
    ready=false;
    recordingRetryAfter=Date.now()+30000;
    $("signal-text").textContent="Recording unavailable";
    note.textContent="The recording could not be loaded.";
    updateButton();
  }
}

async function refresh() {
  if (recordingId) { await refreshRecording(); return; }
  if (staticSite) {
    ready = false;
    $("programme").hidden = true;
    $("signal").classList.add("quiet");
    $("signal-text").textContent = "";
    note.textContent = "Choose a programme from the library.";
    updateButton();
    return;
  }
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

$("volume-toggle").addEventListener("click", () => {
  if (listeningMuted || listeningVolume === 0) {
    listeningMuted = false;
    if (listeningVolume === 0) listeningVolume = lastAudibleVolume;
  } else {
    listeningMuted = true;
  }
  updateAudioLevel(true);
});
$("audio-volume").addEventListener("input", event => {
  const value = Number(event.target.value);
  if (!Number.isFinite(value)) return;
  listeningVolume = Math.max(0, Math.min(1, value));
  if (listeningVolume > 0) {
    listeningMuted = false;
    lastAudibleVolume = listeningVolume;
  }
  updateAudioLevel(true);
});
updateAudioLevel();

button.addEventListener("click", listen);
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
  if (live.getAttribute("src")) resetStream(recordingId
    ? "The recording paused. Select Play to try again."
    : "The connection paused. Select Listen live to reconnect.");
});
live.addEventListener("ended", () => {
  if (!recordingId) { resetStream("The broadcast paused. Select Listen live to reconnect."); return; }
  playing=false; connecting=false; stopWave(); updateButton();
  if(recording)recording.tick(live.currentTime*1000,null,false);
  note.textContent="End of recording";
});
live.addEventListener("timeupdate", updateRecordingPosition);
live.addEventListener("seeking", () => {
  stopWave();
  waveformReadyAfter = performance.now() + 100;
  if (recording && spiritScene) {
    spiritScene.setPositions(null);
    spiritScene.setRotations(null);
    spiritScene.setDepthRotations?.(null);
    recording.tick(live.currentTime * 1000, null, false);
  }
});
live.addEventListener("seeked", () => {
  const windowMs = samples && audioContext ? samples.length * 1000 / audioContext.sampleRate : 0;
  waveformReadyAfter = performance.now() + Math.max(60, windowMs);
  updateRecordingPosition();
  updateWave();
});
$("recording-seek").addEventListener("input", event => {
  if(recordingId && ready) { live.currentTime=Number(event.target.value); updateRecordingPosition(); }
});
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
if (!staticSite) setInterval(refresh, 2500);
