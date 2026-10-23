"use strict";

/* A continuous local-space spring projected into a quiet scene.
 * Audio features deform the local body; prepared semantic coordinates place
 * its center and orientation independently. This renderer derives neither from text.
 */
(() => {
  const TAU = Math.PI * 2;
  const NEUTRAL_ID = "__unassigned__";
  const NEUTRAL_COLOR = "#7b805f";
  const FRAME_MS = 1000 / 30;
  const WAVEFORM_SIZE = 64;
  // Flat, immutable options are shared by the Scene and pure geometry helper.
  // Application code configures these; the public player needs no controls.
  const DEFAULT_OPTIONS = Object.freeze({
    baseAngleDegrees: 90,
    semanticRotationDegrees: 150,
    semanticDepthDegrees: 75,
    axialTiltDegrees: 0,
    yawDegrees: 0,
    turns: 5.5,
    lengthScale: 1.15,
    radiusScale: 1,
    waveformGain: 1,
    axialEnergyKnee: .12,
    semanticSmoothingSeconds: .6,
    featureAttackSeconds: .075,
    featureReleaseSeconds: .28,
    waveformSmoothingSeconds: .035,
    tailFraction: .12
  });
  const OPTION_RANGES = Object.freeze({
    baseAngleDegrees: [-360, 360],
    semanticRotationDegrees: [0, 180],
    semanticDepthDegrees: [0, 89],
    axialTiltDegrees: [-75, 75],
    yawDegrees: [-75, 75],
    turns: [1, 10],
    lengthScale: [.25, 3],
    radiusScale: [.1, 3],
    waveformGain: [0, 8],
    axialEnergyKnee: [.001, 2],
    semanticSmoothingSeconds: [0, 5],
    featureAttackSeconds: [.001, 2],
    featureReleaseSeconds: [.001, 5],
    waveformSmoothingSeconds: [0, 1],
    tailFraction: [0, .25]
  });

  const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, value));
  const unit = value => Number.isFinite(value) ? clamp(value) : 0;
  const smoothstep = value => value * value * (3 - 2 * value);
  const easingAmount = (delta, seconds, immediate) => immediate || seconds === 0
    ? 1 : 1 - Math.exp(-Math.max(0, delta) / seconds);

  function ownFinite(object, key) {
    if (!object || typeof object !== "object" || Array.isArray(object)) return undefined;
    // Ignore inherited values and accessor properties as well as NaN/Infinity.
    const property = Object.getOwnPropertyDescriptor(object, key);
    return property && Number.isFinite(property.value) ? property.value : undefined;
  }

  function normalizeOptions(options, base = DEFAULT_OPTIONS) {
    const normalized = {};
    for (const [key, limits] of Object.entries(OPTION_RANGES)) {
      const previous = ownFinite(base, key);
      const supplied = ownFinite(options, key);
      normalized[key] = clamp(supplied === undefined
        ? (previous === undefined ? DEFAULT_OPTIONS[key] : previous) : supplied, ...limits);
    }
    return Object.freeze(normalized);
  }
  const hash = value => {
    let result = 2166136261;
    for (let i = 0; i < value.length; i++) {
      result ^= value.charCodeAt(i);
      result = Math.imul(result, 16777619);
    }
    return result >>> 0;
  };

  function fallbackColor(id) {
    const hue = (hash(id) % 360) / 60;
    const chroma = .30;
    const middle = chroma * (1 - Math.abs(hue % 2 - 1));
    const segments = [
      [chroma, middle, 0], [middle, chroma, 0],
      [0, chroma, middle], [0, middle, chroma],
      [middle, 0, chroma], [chroma, 0, middle]
    ];
    return "#" + segments[Math.floor(hue)].map(value =>
      Math.round((value + .28) * 255).toString(16).padStart(2, "0")).join("");
  }

  function normalizeColor(value, fallback) {
    if (typeof value !== "string") return fallback;
    if (/^#[0-9a-f]{6}$/i.test(value)) return value.toLowerCase();
    if (/^#[0-9a-f]{3}$/i.test(value)) {
      return "#" + value.slice(1).split("").map(character => character + character).join("").toLowerCase();
    }
    return fallback;
  }

  function rgb(hex) {
    return [1, 3, 5].map(index => parseInt(hex.slice(index, index + 2), 16));
  }

  function rgba(channels, alpha) {
    return "rgba(" + channels.map(value => Math.round(value)).join(",") + "," + clamp(alpha).toFixed(4) + ")";
  }

  function smoothEnvelope(current, target, deltaSeconds, attack = DEFAULT_OPTIONS.featureAttackSeconds, release = DEFAULT_OPTIONS.featureReleaseSeconds) {
    const start = unit(current);
    const end = unit(target);
    const delta = Number.isFinite(deltaSeconds) ? clamp(deltaSeconds, 0, .25) : 0;
    const attackTime = Number.isFinite(attack) ? Math.max(.001, attack) : DEFAULT_OPTIONS.featureAttackSeconds;
    const releaseTime = Number.isFinite(release) ? Math.max(.001, release) : DEFAULT_OPTIONS.featureReleaseSeconds;
    const duration = end > start ? attackTime : releaseTime;
    return start + (end - start) * (1 - Math.exp(-delta / duration));
  }

  /* Returns samples along ONE unbroken curve. The y coordinate is the spring's
   * elongated axis; only the supplied audio profile shapes its local radius.
   */
  function projectHelix(options = {}) {
    options = options && typeof options === "object" ? options : {};
    const renderer = normalizeOptions(options.renderer);
    const radius = (Number.isFinite(options.radius) ? Math.max(0, options.radius) : 20) * renderer.radiusScale;
    const length = (Number.isFinite(options.length) ? Math.max(0, options.length) : 120) * renderer.lengthScale;
    const turns = Number.isFinite(options.turns) ? clamp(options.turns, 1, 10) : renderer.turns;
    const phase = Number.isFinite(options.phase) ? options.phase : 0;
    const rotation = Number.isFinite(options.rotation) ? clamp(options.rotation, -1, 1) : 0;
    const tilt = (Number.isFinite(options.tilt) ? options.tilt : -.12) +
      (renderer.baseAngleDegrees + rotation * renderer.semanticRotationDegrees) * Math.PI / 180;
    const depthRotation = Number.isFinite(options.depthRotation) ? clamp(options.depthRotation, -1, 1) : 0;
    // Two coordinates orient the long axis: screen heading and signed depth
    // inclination. Zero depth lies in the screen plane; no axial spin is added.
    const pitch = Number.isFinite(options.pitch) ? options.pitch :
      clamp(renderer.axialTiltDegrees + depthRotation * renderer.semanticDepthDegrees, -89, 89) * Math.PI / 180;
    const yaw = Number.isFinite(options.yaw) ? options.yaw : renderer.yawDegrees * Math.PI / 180;
    const profile = Array.isArray(options.waveform) && options.waveform.length === WAVEFORM_SIZE &&
      options.waveform.every(Number.isFinite)
      ? options.waveform.map(value => clamp(value, -1, 1)) : null;
    const count = Math.round(clamp(Number(options.points) || 150, 32, 240));
    const camera = Number.isFinite(options.cameraDistance) && options.cameraDistance > 0
      ? Math.max(1, options.cameraDistance, Math.hypot(radius, length * .5) * 1.01)
      : Math.max(1, length * 4, radius * 9);
    const cosPitch = Math.cos(pitch), sinPitch = Math.sin(pitch);
    const cosYaw = Math.cos(yaw), sinYaw = Math.sin(yaw);
    const cosTilt = Math.cos(tilt), sinTilt = Math.sin(tilt);
    const points = [];
    for (let index = 0; index <= count; index++) {
      const u = index / count;
      // Radius is the waveform magnitude, with fixed gain and no baseline.
      // Zero sound has x=z=0: an exact straight axis at nonzero length.
      const profilePosition = u * (WAVEFORM_SIZE - 1);
      const profileLeft = Math.floor(profilePosition);
      const profileRight = Math.min(profileLeft + 1, WAVEFORM_SIZE - 1);
      const radialWave = profile ? profile[profileLeft] +
        (profile[profileRight] - profile[profileLeft]) * (profilePosition - profileLeft) : 0;
      // Smooth radial tails meet the axis tangentially while axial progress
      // continues. There are no terminal winding planes or separate end rings.
      // Zero disables this optional endpoint fade; waveform silence still draws a line.
      const tail = renderer.tailFraction > 0
        ? smoothstep(clamp(Math.min(u, 1 - u) / renderer.tailFraction)) : 1;
      const localRadius = radius * tail *
        Math.min(1, renderer.waveformGain * Math.abs(radialWave));
      // This is the single required helix carrier. No decorative periodic
      // radius, phase, or spacing patterns are added to the actual audio contour.
      const angle = TAU * turns * u + phase;
      const radialX = Math.cos(angle), radialZ = Math.sin(angle);
      const x = localRadius * radialX;
      const y = (u - .5) * length;
      const z = localRadius * radialZ;
      const pitchedY = y * cosPitch - z * sinPitch;
      const pitchedZ = y * sinPitch + z * cosPitch;
      const yawedX = x * cosYaw + pitchedZ * sinYaw;
      const yawedZ = -x * sinYaw + pitchedZ * cosYaw;
      const projectedScale = camera / (camera + yawedZ);
      // Rotate the radial normal with the actual out-of-plane view. Near
      // sections face the camera at negative z; roll does not change this depth.
      const radialNormalZ = -radialX * sinYaw + radialZ * cosPitch * cosYaw;
      points.push({
        x: (yawedX * cosTilt - pitchedY * sinTilt) * projectedScale,
        y: (yawedX * sinTilt + pitchedY * cosTilt) * projectedScale,
        z: yawedZ,
        front: localRadius > 1e-9 ? clamp((1 - radialNormalZ) / 2) : .5
      });
    }
    return points;
  }

  function cleanFeatures(value) {
    const bands = value && Array.isArray(value.bands) ? value.bands : [];
    const profile = value && value.waveform;
    const waveform = Array.isArray(profile) && profile.length === WAVEFORM_SIZE &&
      profile.every(Number.isFinite) ? profile.map(sample => clamp(sample, -1, 1))
      : Array(WAVEFORM_SIZE).fill(0);
    return {
      energy: unit(value && value.energy),
      bands: [unit(bands[0]), unit(bands[1]), unit(bands[2])],
      waveform,
      waveformMask: value && Array.isArray(value.waveform_mask) &&
        value.waveform_mask.length === WAVEFORM_SIZE
        ? value.waveform_mask.map(valid => valid === true) : Array(WAVEFORM_SIZE).fill(true)
    };
  }

  class Scene {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.context = canvas && canvas.getContext ? canvas.getContext("2d") : null;
      options = options && typeof options === "object" ? options : {};
      this.rendererOptions = normalizeOptions(options.renderer);
      this.reducedMotion = options.reducedMotion || window.matchMedia("(prefers-reduced-motion: reduce)");
      this.playbackActive = false;
      this.destroyed = false;
      this.cast = [];
      this.identities = new Map();
      this.targets = new Map();
      this.positionTargets = new Map();
      this.rotationTargets = new Map();
      this.depthRotationTargets = new Map();
      this.frame = null;
      this.lastFrame = null;
      this.elapsed = 0;
      this.width = 1;
      this.height = 1;
      this.scale = 1;
      this.tick = this.tick.bind(this);
      this.onVisibility = this.onVisibility.bind(this);
      this.onMotion = this.onMotion.bind(this);
      this.onResize = this.resize.bind(this);
      this.getIdentity(NEUTRAL_ID, NEUTRAL_COLOR);
      document.addEventListener("visibilitychange", this.onVisibility);
      if (this.reducedMotion.addEventListener) this.reducedMotion.addEventListener("change", this.onMotion);
      else if (this.reducedMotion.addListener) this.reducedMotion.addListener(this.onMotion);
      if (window.ResizeObserver) {
        this.observer = new ResizeObserver(this.onResize);
        this.observer.observe(canvas);
      } else {
        window.addEventListener("resize", this.onResize);
      }
      this.resize();
    }

    configure(partial) {
      if (this.destroyed) return this;
      const next = normalizeOptions(partial, this.rendererOptions);
      if (Object.keys(DEFAULT_OPTIONS).every(key => next[key] === this.rendererOptions[key])) return this;
      this.rendererOptions = next;
      this.clearTrails();
      this.renderOnce();
      this.schedule();
      return this;
    }

    getIdentity(id, color) {
      let identity = this.identities.get(id);
      if (!identity) {
        const seed = hash(id);
        identity = {
          id, color: normalizeColor(color, id === NEUTRAL_ID ? NEUTRAL_COLOR : fallbackColor(id)),
          phase: (seed % 6283) / 1000,
          tilt: ((seed >>> 8) % 1000 / 1000 - .5) * .28,
          energy: 0, bands: [0, 0, 0], trail: [], trailAt: -Infinity,
          position: null, rotation: 0, depthRotation: 0, waveform: Array(WAVEFORM_SIZE).fill(0)
        };
        this.identities.set(id, identity);
      } else if (color !== undefined) {
        identity.color = normalizeColor(color, identity.color);
      }
      return identity;
    }

    setCast(cast) {
      if (this.destroyed) return;
      const seen = new Set();
      const next = [];
      if (Array.isArray(cast)) {
        for (const entry of cast) {
          if (!entry || typeof entry.id !== "string" || !entry.id || entry.id.length > 128 ||
              entry.id === NEUTRAL_ID || seen.has(entry.id)) continue;
          seen.add(entry.id);
          next.push(this.getIdentity(entry.id, entry.color));
          if (next.length === 12) break;
        }
      }
      const changed = next.length !== this.cast.length || next.some((value, index) => value.id !== this.cast[index].id);
      this.cast = next;
      // Retain semantic state only for identities still in this cast.
      for (const id of this.positionTargets.keys()) {
        if (!seen.has(id)) this.positionTargets.delete(id);
      }
      for (const id of this.rotationTargets.keys()) {
        if (!seen.has(id)) this.rotationTargets.delete(id);
      }
      for (const id of this.depthRotationTargets.keys()) {
        if (!seen.has(id)) this.depthRotationTargets.delete(id);
      }
      for (const [id, identity] of this.identities) {
        if (!seen.has(id)) { identity.position = null; identity.rotation = 0; identity.depthRotation = 0; }
      }
      if (changed) {
        this.targets.clear();
        this.clearTrails();
        // Identity phase and color are retained; a new cast starts quietly.
        for (const identity of this.cast) {
          identity.energy = 0; identity.bands = [0, 0, 0];
          identity.waveform = Array(WAVEFORM_SIZE).fill(0);
        }
      }
      // Bound the cache across many programmes while retaining current identities.
      if (this.identities.size > 256) {
        for (const id of this.identities.keys()) {
          if (id !== NEUTRAL_ID && !seen.has(id)) this.identities.delete(id);
          if (this.identities.size <= 128) break;
        }
      }
      this.renderOnce();
      this.schedule();
    }

    setPlaybackActive(active) {
      if (this.destroyed) return;
      this.playbackActive = Boolean(active);
      if (!this.playbackActive) {
        this.targets.clear();
        this.clearTrails();
      }
      this.renderOnce();
      this.schedule();
    }

    setFeatures(features) {
      if (this.destroyed) return;
      const targets = new Map();
      const entries = Array.isArray(features)
        ? features.filter(value => value && typeof value.id === "string").map(value => [value.id, value])
        : features && typeof features === "object" ? Object.entries(features) : [];
      const present = new Set(this.cast.length ? this.cast.map(identity => identity.id) : [NEUTRAL_ID]);
      for (const [id, value] of entries) {
        if (present.has(id) && value && typeof value === "object") targets.set(id, cleanFeatures(value));
      }
      this.targets = targets;
      // Feature updates never animate reduced-motion or hidden documents.
      this.schedule();
    }

    setPositions(positions) {
      if (this.destroyed) return;
      // Every call is a complete playback-timed snapshot. Missing IDs return
      // to fixed anchors, including when seeking before their first meaning.
      const next = new Map();
      const present = new Set(this.cast.map(identity => identity.id));
      if (positions && typeof positions === "object" && !Array.isArray(positions)) {
        for (const [id, value] of Object.entries(positions)) {
          if (present.has(id) && Array.isArray(value) && value.length === 3 &&
              value.every(Number.isFinite)) {
            next.set(id, value.map(coordinate => clamp(coordinate, -1, 1)));
          }
        }
      }
      this.positionTargets = next;
      for (const [id, identity] of this.identities) {
        if (!next.has(id)) identity.position = null;
      }
      // A reduced-motion listener still sees the current semantic location,
      // without an interpolation loop or time-driven camera movement.
      if (this.reducedMotion.matches || !next.size) this.renderOnce();
      this.schedule();
    }

    setRotations(rotations) {
      if (this.destroyed) return;
      // A complete fourth-dimension snapshot; absence resets to the identity's
      // original orientation, including before the first turn after a seek.
      const next = new Map();
      const present = new Set(this.cast.map(identity => identity.id));
      if (rotations && typeof rotations === "object" && !Array.isArray(rotations)) {
        for (const [id, value] of Object.entries(rotations)) {
          if (present.has(id) && Number.isFinite(value)) next.set(id, clamp(value, -1, 1));
        }
      }
      this.rotationTargets = next;
      for (const [id, identity] of this.identities) {
        if (!next.has(id)) identity.rotation = 0;
      }
      if (this.reducedMotion.matches || !next.size) this.renderOnce();
      this.schedule();
    }

    setDepthRotations(rotations) {
      if (this.destroyed) return;
      // A complete fifth-dimension snapshot. Missing coordinates return to
      // zero inclination instead of retaining a future turn after a seek.
      const next = new Map();
      const present = new Set(this.cast.map(identity => identity.id));
      if (rotations && typeof rotations === "object" && !Array.isArray(rotations)) {
        for (const [id, value] of Object.entries(rotations)) {
          if (present.has(id) && Number.isFinite(value)) next.set(id, clamp(value, -1, 1));
        }
      }
      this.depthRotationTargets = next;
      for (const [id, identity] of this.identities) {
        if (!next.has(id)) identity.depthRotation = 0;
      }
      if (this.reducedMotion.matches || !next.size) this.renderOnce();
      this.schedule();
    }

    clearTrails() {
      for (const identity of this.identities.values()) {
        identity.trail.length = 0;
        identity.trailAt = -Infinity;
      }
    }

    resize() {
      if (this.destroyed || !this.canvas) return;
      const bounds = this.canvas.getBoundingClientRect();
      this.width = Math.max(1, bounds.width || 1);
      this.height = Math.max(1, bounds.height || 1);
      this.scale = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.round(this.width * this.scale);
      this.canvas.height = Math.round(this.height * this.scale);
      if (this.context) this.context.setTransform(this.scale, 0, 0, this.scale, 0, 0);
      this.clearTrails();
      this.renderOnce();
      this.schedule();
    }

    onVisibility() {
      this.cancel();
      this.clearTrails();
      if (!document.hidden) {
        this.renderOnce();
        this.schedule();
      }
    }

    onMotion() {
      this.cancel();
      this.clearTrails();
      this.renderOnce();
      this.schedule();
    }

    cancel() {
      if (this.frame !== null) cancelAnimationFrame(this.frame);
      this.frame = null;
      this.lastFrame = null;
    }

    schedule() {
      if (!this.destroyed && this.context && !document.hidden && !this.reducedMotion.matches && this.frame === null) {
        this.frame = requestAnimationFrame(this.tick);
      }
    }

    tick(timestamp) {
      this.frame = null;
      if (this.destroyed || document.hidden || this.reducedMotion.matches) {
        this.lastFrame = null;
        return;
      }
      const elapsedMs = this.lastFrame === null ? FRAME_MS : timestamp - this.lastFrame;
      if (elapsedMs + .25 < FRAME_MS) {
        this.schedule();
        return;
      }
      const delta = Math.min(elapsedMs / 1000, .12);
      this.lastFrame = timestamp;
      this.elapsed += delta;
      this.draw(delta);
      this.schedule();
    }

    renderOnce() {
      if (!this.destroyed && this.context && !document.hidden) this.draw(0);
    }

    path(points, originX, originY) {
      const context = this.context;
      context.beginPath();
      for (let index = 0; index < points.length; index++) {
        const point = points[index];
        if (index === 0) context.moveTo(originX + point.x, originY + point.y);
        else context.lineTo(originX + point.x, originY + point.y);
      }
    }

    draw(delta) {
      const context = this.context;
      const reduced = this.reducedMotion.matches;
      const renderer = this.rendererOptions;
      context.clearRect(0, 0, this.width, this.height);
      const identities = this.cast.length ? this.cast : [this.getIdentity(NEUTRAL_ID)];
      const count = identities.length;
      const columns = count <= 4 ? count : Math.ceil(Math.sqrt(count * this.width / Math.max(this.height, 1)));
      const safeColumns = Math.max(1, Math.min(count, 4, columns));
      const rows = Math.ceil(count / safeColumns);
      const cellWidth = this.width * .94 / safeColumns;
      const cellHeight = this.height * .94 / rows;
      const time = reduced ? 0 : this.elapsed;
      context.lineCap = "round";
      context.lineJoin = "round";

      // One common scene camera. These conservative extents depend only on
      // the viewport/cast and the existing maximum local deformation, never on
      // the current sound. Audio therefore cannot move a semantic center.
      const intrinsicRadius = Math.max(1, Math.min(cellWidth * .22, cellHeight * .13));
      const intrinsicLength = Math.max(2, cellHeight * .71);
      const baseRadius = intrinsicRadius * renderer.radiusScale;
      const baseLength = intrinsicLength * renderer.lengthScale;
      // Every configured roll, including the base 90-degree offset without a
      // semantic cue, preserves this sphere. Tail fades cannot enlarge it.
      const maximumRadius = baseRadius * Math.min(1, renderer.waveformGain);
      const maximumAxial = .5 * baseLength;
      const sphere = Math.hypot(maximumRadius, maximumAxial);
      const localCamera = Math.max(1, baseLength * 4, baseRadius * 9);
      // Tangent silhouette under the same fixed local camera used below. Full
      // energy/waveform bounds are reserved even for a silent character.
      const projectedSphere = localCamera * sphere /
        Math.sqrt(localCamera * localCamera - sphere * sphere);
      const extentX = projectedSphere;
      const extentY = projectedSphere;
      const maximumPerspective = 1.2;
      const fit = Math.min(1,
        this.width * .30 / (extentX * maximumPerspective),
        this.height * .34 / (extentY * maximumPerspective));
      const margin = Math.min(this.width, this.height) * .025;
      const rangeX = Math.max(0, (this.width / 2 - margin - extentX * fit * maximumPerspective) / maximumPerspective);
      const rangeY = Math.max(0, (this.height / 2 - margin - extentY * fit * maximumPerspective) / maximumPerspective);
      const positionAmount = easingAmount(delta, renderer.semanticSmoothingSeconds, reduced);
      const placements = identities.map((identity, index) => {
        const row = Math.floor(index / safeColumns);
        const rowCount = Math.min(safeColumns, count - row * safeColumns);
        const column = index % safeColumns;
        const fixedX = this.width / 2 + (column - (rowCount - 1) / 2) * cellWidth;
        const fixedY = this.height / 2 + (row - (rows - 1) / 2) * cellHeight;
        const semanticTarget = this.positionTargets.get(identity.id);
        if (!semanticTarget) {
          // Fixed anchors reserve the same complete orientation envelope,
          // including the configured base angle before semantic cues arrive.
          const fixedFit = Math.max(0, Math.min(1,
            (Math.min(fixedX, this.width - fixedX) - margin) / extentX,
            (Math.min(fixedY, this.height - fixedY) - margin) / extentY));
          return {identity, index, originX: fixedX, originY: fixedY, sceneScale: fixedFit, sceneDepth: 0};
        }
        if (!identity.position) {
          // Begin at the old fixed anchor, then approach the supplied center.
          // Reduced motion uses the represented point immediately.
          identity.position = reduced ? semanticTarget.slice() : [
            rangeX ? (fixedX - this.width / 2) / rangeX : 0,
            rangeY ? -(fixedY - this.height / 2) / rangeY : 0,
            0
          ];
        }
        for (let axis = 0; axis < 3; axis++) {
          const difference = semanticTarget[axis] - identity.position[axis];
          identity.position[axis] = reduced || Math.abs(difference) < .00001
            ? semanticTarget[axis] : identity.position[axis] + difference * positionAmount;
        }
        const [x, y, z] = identity.position;
        // x points right, y up, and positive z away from the shared camera.
        const perspective = clamp(3 / (3 + z), .75, maximumPerspective);
        return {
          identity, index,
          originX: this.width / 2 + x * rangeX * perspective,
          originY: this.height / 2 - y * rangeY * perspective,
          sceneScale: fit * perspective,
          sceneDepth: z
        };
      });
      // Draw farther bodies first; no collision offsets or narrative movement.
      placements.sort((a, b) => b.sceneDepth - a.sceneDepth || a.index - b.index);

      const waveformAmount = easingAmount(delta, renderer.waveformSmoothingSeconds, reduced);
      for (const {identity, originX, originY, sceneScale} of placements) {
        const rotationTarget = this.rotationTargets.get(identity.id) || 0;
        const rotationDifference = rotationTarget - identity.rotation;
        identity.rotation = reduced || Math.abs(rotationDifference) < .00001
          ? rotationTarget : identity.rotation + rotationDifference * positionAmount;
        const depthTarget = this.depthRotationTargets.get(identity.id) || 0;
        const depthDifference = depthTarget - identity.depthRotation;
        identity.depthRotation = reduced || Math.abs(depthDifference) < .00001
          ? depthTarget : identity.depthRotation + depthDifference * positionAmount;
        const target = !reduced && this.playbackActive ? this.targets.get(identity.id) : null;
        const targetEnergy = target ? target.energy : 0;
        identity.energy = reduced ? 0 : smoothEnvelope(identity.energy, targetEnergy, delta,
          renderer.featureAttackSeconds, renderer.featureReleaseSeconds);
        for (let band = 0; band < 3; band++) {
          identity.bands[band] = reduced ? 0 : smoothEnvelope(identity.bands[band], target ? target.bands[band] : 0, delta, .09, .24);
        }
        for (let bin = 0; bin < WAVEFORM_SIZE; bin++) {
          const soundTarget = !reduced && target ? target.waveform[bin] : 0;
          const difference = soundTarget - identity.waveform[bin];
          // Reapply ownership after smoothing so stale samples cannot bleed
          // across cue boundaries when a character begins another turn.
          identity.waveform[bin] = target && !target.waveformMask[bin] ? 0
            : reduced || Math.abs(difference) < .00001
              ? soundTarget : identity.waveform[bin] + difference * waveformAmount;
        }
        const color = rgb(identity.color);
        if (identity.energy < .005) {
          // Silence has no independent movement or full resting coil: retain
          // only the character's small colored point at its represented center.
          identity.trail.length = 0;
          identity.trailAt = -Infinity;
          context.beginPath();
          context.arc(originX, originY, 2, 0, TAU);
          context.fillStyle = rgba(color, .65);
          context.fill();
          continue;
        }
        // Energy controls axial elongation. A short smooth activation gate
        // opens radius near silence; it does not rescale ordinary speech's
        // fixed-gain amplitude profile a second time by the current RMS level.
        const radiusGate = clamp(identity.energy / .05);
        const radiusActivation = smoothstep(radiusGate);
        const radius = intrinsicRadius * radiusActivation;
        const knee = renderer.axialEnergyKnee;
        const axialActivation = (1 + knee) * identity.energy / (identity.energy + knee);
        const length = intrinsicLength * axialActivation;
        const points = projectHelix({
          radius, length, renderer, cameraDistance: localCamera,
          phase: identity.phase,
          // PC4 sets screen heading; PC5 sets depth inclination. Both move
          // the whole audio body and neither introduces a spin around its axis.
          tilt: identity.tilt, rotation: identity.rotation, depthRotation: identity.depthRotation,
          waveform: identity.waveform, points: count > 6 ? 110 : 150
        });
        // Camera placement scales the whole unchanged local body, including
        // its length and radius. Strand, shading and trail styles stay intact.
        if (sceneScale !== 1) {
          for (const point of points) {
            point.x *= sceneScale;
            point.y *= sceneScale;
            point.z *= sceneScale;
          }
        }
        const highlight = color.map((value, channel) => value * .56 + [250, 239, 207][channel] * .44);
        const activity = Math.max(identity.energy, ...identity.bands);
        const opacity = .52 + activity * .33;

        if (reduced || !this.playbackActive || activity < .012) {
          identity.trail.length = 0;
        } else {
          identity.trail = identity.trail.filter(sample => time - sample.at < .22);
          for (const sample of identity.trail) {
            const fade = Math.max(0, 1 - (time - sample.at) / .22);
            this.path(sample.points, originX, originY);
            context.strokeStyle = rgba(color, .055 * fade * activity);
            context.lineWidth = 1.35;
            context.stroke();
          }
          if (time - identity.trailAt >= .065) {
            identity.trail.push({at: time, points});
            identity.trailAt = time;
          }
        }

        // A single under-strand guarantees continuous, unbroken geometry.
        this.path(points, originX, originY);
        context.strokeStyle = rgba(color, opacity * .22);
        context.lineWidth = 3.1;
        context.stroke();

        const segments = [];
        for (let point = 1; point < points.length; point++) {
          const a = points[point - 1], b = points[point];
          // Degenerate curves must not stack rounded zero-length strokes
          // into dark dots (including an explicitly zero-length helper curve).
          if (Math.hypot(b.x - a.x, b.y - a.y) < 1e-6) continue;
          segments.push({a, b, z: (a.z + b.z) / 2, front: (a.front + b.front) / 2});
        }
        // Back sections first; the nearer portions of the same coil remain crisp.
        segments.sort((a, b) => b.z - a.z);
        for (const segment of segments) {
          const depth = .36 + segment.front * .64;
          context.beginPath();
          context.moveTo(originX + segment.a.x, originY + segment.a.y);
          context.lineTo(originX + segment.b.x, originY + segment.b.y);
          context.strokeStyle = rgba(color, opacity * depth);
          context.lineWidth = 1.1 + segment.front * .65;
          context.stroke();
          if (segment.front > .38) {
            context.strokeStyle = rgba(highlight, opacity * .55 * segment.front);
            context.lineWidth = .50;
            context.stroke();
          }
        }
      }
    }

    destroy() {
      if (this.destroyed) return;
      this.destroyed = true;
      this.cancel();
      if (this.observer) this.observer.disconnect();
      else window.removeEventListener("resize", this.onResize);
      document.removeEventListener("visibilitychange", this.onVisibility);
      if (this.reducedMotion.removeEventListener) this.reducedMotion.removeEventListener("change", this.onMotion);
      else if (this.reducedMotion.removeListener) this.reducedMotion.removeListener(this.onMotion);
      this.targets.clear();
      this.positionTargets.clear();
      this.rotationTargets.clear();
      this.depthRotationTargets.clear();
      this.identities.clear();
      if (this.context) this.context.clearRect(0, 0, this.width, this.height);
    }
  }

  window.RadioSpirits = Object.freeze({Scene, projectHelix, smoothEnvelope, NEUTRAL_ID, DEFAULT_OPTIONS, normalizeOptions});
})();
