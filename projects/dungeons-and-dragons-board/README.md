# Dungeons & Dragons Board

**Status:** Idea record with initial research. No development requested.
**Recorded and researched:** 16 October 2026.

A physical tabletop board for playing Dungeons & Dragons with friends, combining
real miniatures and terrain, a robotic component, and an OpenAI live voice model
acting as the game master. Players talk naturally, make decisions, roll dice,
and handle their characters while the game master describes the world, plays
non-player characters, manages encounters, and coordinates physical effects.

The desired experience is a shared adventure in which the board responds to the
story: a creature advances, a door opens, or lighting reveals a newly discovered
room. The board supports exploration and conversation as well as combat.

This folder currently contains only this README. All hardware and software
choices below are proposals for investigation; no prototype or performance
claims are implied.

The concept is divided into three parts:

| Part | Main responsibility | Connection to the other parts |
| --- | --- | --- |
| Board-master agent | Run the adventure and resolve what happens in the game. | Receives player intentions and board observations; produces recorded game events. |
| Board design | Provide the physical playing surface, sensing, robotics, and controls. | Reports observations and capabilities; executes validated physical requests. |
| Story animation | Bring game events to life through coordinated narration, sound, visuals, and movement. | Turns recorded events into presentation cues and requests supported board effects. |

## 1. Board-master agent

### Role and game state

The agent takes the game-master role: understands player intentions, describes
situations, plays non-player characters, prepares encounters, and guides the
adventure. It decides what happens with support from rules and recorded state.
The story-animation part coordinates how those outcomes are presented.

D&D combines social interaction, exploration, and combat. The game master
describes a situation, players choose actions, and the game master resolves the
consequences. A successful board must preserve that freedom, including actions
that do not fit a predefined menu. The SRD's square-grid rules use one square
to represent 5 feet. [1]

Proposed starting ruleset: **SRD 5.2.1**, with the edition and any house rules
recorded per campaign. Avoid accidentally mixing rules from different editions.
The game system would maintain characters, initiative, movement, actions,
reactions, hit points, conditions, and resources. An incomplete rules
implementation must expose unsupported cases for a human ruling.

Use physical dice with confirmed result entry initially; a logged software dice
roller could be optional. The model requests a roll and narrates its outcome
after receiving the recorded result.

The SRD is available under CC BY 4.0. Any implementation distributing adapted SRD
material must include the attribution required by its legal preamble. D&D Beyond
also identifies content excluded from the SRD; a freely readable rulebook or
adventure should not be assumed to have the same reuse permission. [1]

### OpenAI live conversation and reasoning

The current official documentation describes **GPT-Live (`gpt-live-1`)** as a
voice frontend that can listen while speaking and delegate reasoning and tools
to a separate backend. WebRTC is the recommended audio connection; keep the
project API key on a trusted server. [5][6]

Proposed division within the board-master system:

- **Live conversation:** player questions, character dialogue, and delivery of
  narration coordinated with the story-animation part.
- **Game-master backend:** campaign context, rules retrieval, encounter decisions,
  and requests to application tools. Client delegation is a candidate because
  it lets the application control context and validate returned results.
- **Game application:** authoritative state, dice records, supported rule checks,
  visibility, saves, and an event history that supports correction and undo.

GPT-Live's audio frontend does **not** accept images directly. Send any board
image needed for interpretation to a vision-capable backend and return concise
findings to the conversation. [7] The board supplies structured observations
with timestamps and confidence.

Expose bounded application operations such as inspecting the encounter,
requesting a roll, proposing a move, and saving a ruling. Keep campaign memory
outside the voice session so a reconnection does not reset the adventure or
expose game-master secrets. Players need a way to pause and correct a ruling.

### First investigation

Run a short adventure with ordinary miniatures and manual movement. Assess
player agency, rule accuracy, multiple speakers, interruptions, continuity,
and session cost before connecting robotics. Open decisions include rules
coverage, campaign preparation, human overrides, and acceptable cloud
dependency. Measure voice and backend usage over a realistic session before
estimating ongoing costs.

## 2. Board design

### Playing surface and robotic mechanism

**Candidate to investigate first:** a hidden XY carriage with a controllable
magnetic coupling beneath a flat, nonmagnetic playing surface. Compatible bases
would allow selected miniatures or tokens to slide across it. Robotic chess
projects demonstrate this principle: Gambit documents an electromagnet and
magnetic piece bases, while Phantom describes the tradeoffs among coupling,
surface thickness, piece weight, and clearance. [2][3]

Applying it to D&D remains an engineering hypothesis. Terrain, crowded rooms,
large creatures, overhanging miniatures, and different base sizes make movement
harder than on an open chessboard. A single carriage would move one piece at a
time. Its disengaged travel must also avoid disturbing other magnetic bases.
Do not assume a display panel can sit between the magnet and the pieces without
testing the full material stack and magnetic separation.

The smallest useful robotic demonstration would move one enemy token reliably
while players move their own pieces. Actuated doors or traps are possible later
capabilities for story animation. Self-propelled miniature bases and an overhead
pick-and-place mechanism are alternatives to compare if sliding cannot handle
the required terrain; neither is selected yet.

### Sensing and presentation hardware

For sensing, investigate an overhead camera with visible identifiers on token
bases or small stands and fixed board calibration markers. OpenCV documents
ArUco marker identification and pose estimation. [4] Marker visibility, camera
resolution, hands, shadows, and tall terrain must be tested. Hidden markers
under an opaque miniature are not visible to an overhead camera. Require a
manual correction path when identity or position is uncertain.

Begin with printed maps and removable terrain. Projection, dynamic lighting,
or a screen are later options. Physical terrain must match the recorded map;
shared lighting cannot conceal a visible miniature from only one player, so
private information needs a separate player view or delayed physical placement.

Board design also covers microphones, speakers, echo control, power, and local
computing hardware. Players should know when microphones and cameras are active
and have simple mute controls.

### Board control

The board controller owns calibrated motion, route clearance, limits, and
feedback. The language model should never supply raw motor commands. The game
application checks the legal game path; the board checks current observations
and whether the requested movement is physically executable.

An enclosed mechanism, conservative speed and force limits, end stops, a local
physical stop, and a deliberate motion-enable control are proposed requirements.
Cloud connectivity or camera inference must not be the sole stop mechanism.
Players need an immediate way to stop physical motion and continue manually.

### First investigation

Test one physical move: measure coupling through the proposed surface,
repeatability, noise, heating, release, and disturbance of adjacent tokens.
Then recognize a few marked tokens, detect manual changes, and recover from
occlusion or incorrect observations. Choose dimensions and base spacing from
measurements. Open decisions include board size, flat versus raised terrain,
miniature compatibility, mechanism, sensors, and hardware budget.

## 3. Story animation during play

### Bringing the adventure to life

This part concerns how the story is enacted and experienced at the table:
coordinating spoken narration, character performances, ambient sound, lighting,
map reveals, visual effects, and supported robotic movement. It gives a recorded
game event a physical and audiovisual presentation.

Possible effects to investigate include:

- A discovered room appears on the map as the game master describes it.
- An enemy advances on the board, accompanied by footsteps and narration.
- A spell is represented by a light or projected area, with matching sound.
- A door or trap moves at the appropriate moment if the board supports it.
- A scene transition changes the atmosphere and prepares the next encounter.

These are presentation ideas, not selected hardware or implemented features.
Animation must use capabilities reported by the board and respect each player's
visibility. It must not independently change dice results, damage, positions,
or other game outcomes. Effects can be simplified or skipped while preserving
the recorded game state.

### Coordination with the agent and board

Example flow:

1. A player declares an action to the board-master agent.
2. The game application records any roll, resolves supported mechanics, and
   records the outcome, including the intended movement or reveal.
3. Story animation prepares narration and presentation cues, requesting any
   physical movement through the board controller.
4. The board reports completion, failure, or a need for manual placement.
5. Narration reflects the confirmed result, and the application records any
   reconciliation needed between physical positions and game state.

The legal game path and physical execution must stay consistent. An arbitrary
mechanical detour must not silently change movement costs or trigger game
effects. If the approved path cannot be executed, use manual placement and
record the reconciliation.

Interrupted speech does not establish whether a physical action was cancelled.
Track game actions and their presentation cues with identifiers, reject
duplicate or stale requests, and distinguish proposed, accepted, executing,
completed, and failed actions. Replaying narration must not repeat a completed
physical move. After a dropped connection or uncertain move, stop and reconcile
observed positions before resuming.

### First investigation

Stage one short encounter using spoken narration, simple sound or map cues,
and manually moved pieces. Evaluate pacing, clarity, interruptions, and whether
the effects help players follow the story. Then connect a few enemy moves and
exercise blocked routes, duplicate requests, interruption, stop, and
reconnection. Set acceptance thresholds before evaluating the prototype.

Open decisions include the visual style, sound design, scene-authoring method,
live-generated versus prepared assets, cue timing, and which effects add enough
to play to justify their complexity. Development of all three parts remains
future work.

## Research sources

Sources checked on 16 October 2026. They establish the documented capabilities
above; the proposed D&D integration has not been validated.

1. [Wizards of the Coast / D&D Beyond: SRD and edition information](https://www.dndbeyond.com/srd)
   and [SRD 5.2.1 PDF](https://media.dndbeyond.com/compendium-images/srd/5.2/SRD_CC_v5.2.1.pdf),
   especially Legal Information, Rhythm of Play, and Playing on a Grid.
2. [Gambit: robotic chessboard project and engineering logs](https://hackaday.io/project/192559-gambit).
3. [Phantom Chess: how robotic chessboards work](https://www.phantomchessboard.com/blog/how-robotic-chess-boards-work).
4. [OpenCV: detection of ArUco markers](https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html).
5. [OpenAI: getting started with GPT-Live](https://developers.openai.com/api/docs/guides/live).
6. [OpenAI: WebRTC connections](https://developers.openai.com/api/docs/guides/voice-webrtc).
7. [OpenAI: delegation, tools, and visual context in GPT-Live](https://developers.openai.com/api/docs/guides/live-delegation).
