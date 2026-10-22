# Local audio and semantic audition

Open http://127.0.0.1:43120/?recording=empty-chair-first-guest-v2 to hear
*The Mouse Who Brought a Monster*. The ordinary entrance still follows the
station schedule. Preparing this audition does not publish or schedule it.

The existing botanical frame, paper texture, Waajacu macaw footer, and continuous
textured helix strands remain in place. The cast contains Nell, Rusk, and Orro:
each character keeps its own color and a quietly present dot when silent.

## Audio body and semantic placement

The spring body follows measured RMS energy and three frequency bands from the
matching final WAV, with 50 ms samples, character cues, smoothed attack/release,
and the existing short fading trail. The final audio is a verified concatenation
of sequential character takes; these cues identify turns, not individual words.
A 64-point signed time-domain profile from the actual Web Audio output also
shapes the radius along the spring's own axis. Audio energy stretches a quiet
dot along that axis into the coil; fading speech contracts it again. The
strand fades smoothly into the axis at both ends. Tail amount is configurable,
including zero, and axial progress continues across the full range without
forced winding planes. Semantic rotation carries the entire axis and waveform together. It box-averages the analyser's
2048-sample window with fixed gain, interpolates spatially along the coil, and
smooths briefly in time. Its absolute magnitude, with a fixed gain of two and
a unit ceiling, gives radius with no constant offset. A zero profile is exactly
a straight axis; zero energy contracts that axis to the silent dot. A monotonic
energy response opens axial length more fully during ordinary speech. The canvas
fills the existing animation slot without moving the controls. This waveform is attached
only to the sole active timed speaker. Overlapping voices require separate
waveforms; a mixed waveform is never copied to multiple character bodies.
A brief post-seek hold excludes samples from the previous playback location.

Each spoken turn also has a Gemini Embedding 2 text embedding. All turns share
one L2-normalized, mean-centered PCA fit, reduced from 3072 dimensions to four.
The first three components position the center; the fourth smoothly rotates the
whole spring about the viewing axis by up to 150 degrees each way, added to the
90-degree base orientation. All components
use the same score scale, preserving the original three coordinates. Rotation
encodes a semantic value and adds no time-driven spin. A character's spring center approaches the current turn's
coordinate with a 0.6 second time constant and retains its latest coordinate when
inactive. Before first speech it rests at its original anchor. Seeking backward
reconstructs positions from the timeline prefix, without retaining future turns.

The projection is a relative semantic space fitted to this recording; its first
three components retain 28.07% of the normalized embedding variance; adding
rotation raises this to 35.35%. Its axes
have no assigned narrative meaning. It adds no collision avoidance, per-character
offsets, or camera choreography. Rank-deficient axes remain zero. Raw embeddings
and the fitted basis remain separate, so another projection can be compared
without more paid calls. Audio embeddings have not been requested.

Playback uses the exact final PCM WAV and audio.currentTime, including pause,
seek, and replay. Reduced motion shows the current semantic position without
continuous body animation. Backend preparation and credentials never appear as
public generation controls.

## Preparation and evidence

All project code runs in the documented managed Linux development container.
tools/prepare_audition_preview.py verifies source hashes, exact concatenation,
the final loudness transform, transcript slices, and character feature tracks.
It writes .runtime/visual-previews/<id>/ and an isolated preparation release.

The existing EmbeddingService prepares text windows against that isolated
release. Paid requests require explicit episode/series caps, durable reservation,
and retained uncertain outcomes. This audition was explicitly authorized for text
only with a US$0.10 total cap. Nineteen requests completed; the conservative
serving-ceiling accounting is US$0.031141, not a reconciled provider invoice.
No additional provider work occurs when opening or replaying the page.

tools/project_preview_semantics.py validates the raw bundle and source identity,
fits one PCA basis, and writes semantic.json. The v1 preview and evidence are
preserved; v2 adds the fourth component without additional provider calls. Private receipts, the mean/basis,
and source alignment evidence stay under evidence/. The only served assets are
the fixed audio.wav, performance.json, and optional semantic.json, checked
against the manifest. Metadata and assets use no-store and noindex headers.
Source takes, credentials, ledgers, raw vectors, and preparation files are not
available through these routes.

The current alignment includes natural leading/trailing silence in each source
turn. More granular phrase/word timing can be prepared later; this first test
moves by spoken turn.

## Shared rendering and listening controls

The listening page uses the common configurable renderer described in
[renderer integration and options](RENDERER.md). Its defaults and runtime
configuration apply to any supplied cast and playback-timed feature/semantic
snapshots. The current recording adapter is named Recording; the previous
Preview alias and query remain compatible with existing links and retained data.
Public labels use Radio Waajacu and Recording, and the page has no volume
control. Playback uses unity gain and follows device master volume. The seek
line is wider with a one-pixel track and a larger transparent pointer target.

The visible name does not alter the private source identities, immutable
recording evidence, preparation journals, or station schedule.
