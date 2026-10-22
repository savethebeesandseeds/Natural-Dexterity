# Radio spirit renderer

`web/player/spirits.js` is the shared Canvas2D renderer for live radio and recorded
programme playback. It has no network, provider, text-embedding, or generation
logic. The caller supplies the current cast, playback-timed sound features, and
optional semantic center/orientation snapshots. No particular programme or voice
names are built into the renderer. The current live adapter supplies RMS/bands and
an exact HLS playback clock; the recording adapter additionally supplies prepared
semantic positions and orientation. Reusing this renderer does not automatically
prepare semantic data for a new programme. Independently fitted per-programme PCA
coordinates are not comparable across programmes unless a shared projection basis
is reused.

## Application integration

Load `spirits.js` before the player controller. Its browser global is
`window.RadioSpirits`.

```js
const {Scene, DEFAULT_OPTIONS, normalizeOptions} = window.RadioSpirits;
const scene = new Scene(document.querySelector("canvas"), {
  reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)"),
  renderer: {baseAngleDegrees: 90, tailFraction: .12}
});
scene.setCast([{id: "voice-id", color: "#789287"}]);
scene.setPlaybackActive(true);
scene.setFeatures({"voice-id": {
  energy: .4,
  bands: [.2, .3, .1],
  waveform: Array(64).fill(0) // Replace with actual signed audio samples.
}});
scene.setPositions({"voice-id": [.2, -.1, .3]});
scene.setRotations({"voice-id": -.25});

// Application configuration, without adding public player controls:
scene.configure({lengthScale: 1.3, tailFraction: .16});
```

`DEFAULT_OPTIONS`, the result of `normalizeOptions(partial, base)`, and
`scene.rendererOptions` are frozen flat objects. `configure(partial)` merges into
the current configuration and returns the scene. Unknown keys, strings,
inherited/accessor properties, `NaN`, and infinities are ignored; finite numeric
values are clamped to the ranges below. Invalid values preserve the previous
setting (or the default when normalizing without a base). Configuration changes
clear old motion trails and reframe the static camera envelope.

| Option | Default | Allowed range | Meaning |
| --- | ---: | ---: | --- |
| `baseAngleDegrees` | 90 | -360 to 360 | Fixed rotation about the view axis; added to the stable identity tilt. |
| `semanticRotationDegrees` | 150 | 0 to 180 | Fourth-coordinate value -1 to 1 maps to this signed angular range. |
| `turns` | 5.5 | 1 to 10 | Continuous coil winding count. |
| `lengthScale` | 1.15 | .25 to 3 | Multiplier of the viewport/cast axial base length. |
| `radiusScale` | 1 | .1 to 3 | Multiplier of the viewport/cast base radius. |
| `waveformGain` | 2 | 0 to 8 | Fixed gain of absolute signed waveform samples; radial multiplier caps at 1. |
| `axialEnergyKnee` | .12 | .001 to 2 | Energy-to-length response knee; smaller values extend quiet speech more. |
| `semanticSmoothingSeconds` | .6 | 0 to 5 | Exponential position/orientation time constant; 0 snaps. |
| `featureAttackSeconds` | .075 | .001 to 2 | Energy attack time constant. |
| `featureReleaseSeconds` | .28 | .001 to 5 | Energy release time constant. |
| `waveformSmoothingSeconds` | .035 | 0 to 1 | Signed per-sample waveform smoothing; 0 snaps. |
| `tailFraction` | .12 | 0 to .25 | Fraction of each end used for a smooth radial fade to the axis; 0 disables it. |

Length and radius changes remain subject to viewport fitting. A larger length
scale changes the local aspect ratio; it cannot force a body outside the reserved
camera envelope. Styling (strand width, highlight, color, depth shading, and
short motion trails) remains independent of these geometry options.

## Data and timing contract

- `setCast([{id, color}])` accepts up to 12 unique nonempty string IDs. Colors are
  six- or three-digit hex; a stable ID-derived color is used when omitted. An
  empty cast shows one neutral point with ID `RadioSpirits.NEUTRAL_ID`, whose
  value is `"__unassigned__"`; this is not an invented speaker.
- `setPlaybackActive(boolean)` gates audio deformation. Pausing clears feature
  targets and trails, then releases the body to its silent point.
- `setFeatures(map)` is a full snapshot keyed by current character ID; an array
  of `{id, energy, bands, waveform}` entries is also accepted. Energy and the
  three low/mid/high bands are finite numbers clamped to [0, 1]. `waveform` is
  exactly 64 finite signed values clamped to [-1, 1]; a missing/invalid profile
  becomes zero. Omitted characters release to silence.
- `setPositions(map)` is a full snapshot keyed by ID with finite `[x, y, z]`
  coordinates clamped to [-1, 1]. Positive x is right, y is up, z is away.
  Omitted IDs use fixed scene anchors. `null` clears all semantic positions.
- `setRotations(map)` is a full snapshot keyed by ID with finite scalars clamped
  to [-1, 1]. Omitted IDs return to the base angle plus their stable identity
  tilt. `null` clears all semantic rotations.
- `resize()` updates the backing canvas; a ResizeObserver normally does this.
  `renderOnce()` draws without advancing time. `destroy()` releases listeners,
  observers, animation frames, and scene state.

The caller must sample cues, features, and semantics using actual audio playback
time. It must reset or replace snapshots on seeks so future meanings do not
persist. It must only attribute waveform samples to an identified voice when the
source features or cue/audio window support that attribution. A mixed waveform
cannot independently animate overlapping voices; use prepared per-speaker
features for those intervals. The renderer neither infers speakers nor supplies
synthetic sound energy.

## Geometry and accessibility

Silence (smoothed energy below .005) is a small colored point at the represented
center. Energy elongates the main axis by
`(1 + axialEnergyKnee) * E / (E + axialEnergyKnee)`. A short smooth radius gate
near silence opens the body without multiplying ordinary speech amplitude by
energy a second time. Along the axis, radius is the absolute linearly sampled
waveform, multiplied by fixed `waveformGain`, capped at one, and shaped by the
existing bounded band ripples. **A zero waveform at nonzero energy is a straight
line**, not a normal resting spiral.

The tail envelope is smoothstep of distance from either endpoint divided by
`tailFraction`. Its radius and radial derivative reach zero at the endpoint, so
the continuous strand meets the axis with a straight tangent. Axial coordinates
continue across the full range: there are no forced flat winding planes, extra
rings, disconnected tails, or separate discs. `tailFraction: 0` preserves the raw
waveform through the endpoints.

Prepared xyz values place the entire local body with a common perspective camera.
The fourth value sets a target roll; it is not a spin rate. Shape deformation
cannot move its semantic center. Static camera bounds reserve the full configured
length, radius, waveform cap, band ripples, and all view-axis orientations. Camera
fit depends on viewport, cast, and configuration, never the current audio sample.

The scene owns a 30 fps-capped requestAnimationFrame loop. Hidden documents stop
it and clear trails. Reduced motion disables the loop and all audio deformation,
shows quiet points, and represents current semantic positions/orientations
immediately without interpolated motion. Identity colors and phases remain stable
across programmes using the same IDs.

## Pure helpers

`projectHelix(options)` returns local projected `{x, y, z, front}` samples along
one continuous strand. It accepts `renderer` with the same normalized options;
`radius` (default 20) and `length` (default 120) are base dimensions before
`radiusScale`/`lengthScale`. Optional geometry inputs are `phase`, `tilt`, `pitch`,
`yaw` (radians), `rotation` (-1 to 1), `energy`, `bands`, `waveform`, `time`,
`points` (32 to 240), and `cameraDistance`. An explicit finite `turns` overrides
the normalized winding count. The configured base angle and signed semantic
rotation are added to `tilt`. An explicit camera distance is bounded away from
the geometry to prevent a perspective singularity; Scene passes its fixed full
body camera. The helper does not apply energy elongation itself, allowing callers
to inspect a full-length zero-waveform line.

`smoothEnvelope(current, target, deltaSeconds, attack, release)` is the bounded,
frame-rate-independent unit feature envelope. Its default attack/release values
match `DEFAULT_OPTIONS`; Scene supplies its configured energy values.
