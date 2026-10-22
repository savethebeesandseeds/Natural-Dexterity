# Does continued learning help after the sun moves?

## Adopted goal and protocol, before execution

Adopted 2026-09-06 after the owner asked to define and pursue the next goal.
Determine the contribution of continued action-value learning to adaptation
in the existing constructed creature. Preserve a complete reproducible result,
whether continued learning helps, hurts, or has inconsistent effects. This is
a bounded engine experiment before a new controller or more elaborate body.

### Fixed comparison

Use the same two exposed development bodies as light-learning v2: the default
light-enabled [3,3] assembly and that assembly with a .012 kg block at segment
1, slot 0, side +1. Use learner seeds 1, 2 and 3 on each. Six pairs, twelve
120-second runs, each independently replayed. No parameter selection, extra
seeds, retries to improve outcomes, or new physical scenarios in this protocol.
The frozen original rover-v3 challenge remains outside this work.

Each branch starts from the same deterministic world initialization and seed,
then executes the unmodified v2 learner for 3,000 control steps (60 seconds).
Compare the complete command/observation prefix, public physical state, full
controller-state fingerprint and learned-parameter fingerprint. Reconstructing
the same prefix avoids adding a simulator save/restore mechanism. Public state
is not a complete native MuJoCo checkpoint; deterministic reconstruction,
identical full prefixes and independent replay provide the corresponding
reproducibility evidence. Require these checks before interpreting a pair.

Immediately before action 3,001, move the sun from [0.30,0,0.20] to
[-0.30,0,0.20] m, intensity 1000 lux, in both branches. One continues learning;
the other freezes its current value parameters. Freeze occurs between a
completed act/observe pair, and can fall within a .4-second option. Do not end
or restart that option. Its accumulated reward, current motor command, world,
sensor delivery queues and all history continue across the intervention.

Freezing suppresses both online and replay writes to value parameters. It
preserves sensing, feature computation, reward history, eligibility/replay
bookkeeping, seeded random draws, exploration schedule, action selection,
local feedback governor, and native protection. Thus frozen values can still
produce changing actions in response to new measurements. Reset enables
learning again; the experiment uses a one-way freeze within a run. The
controller receives no sun coordinates or sun-move notification.

The existing v2 initialization, 56 features, .92 option discount, .4-second
options, .02-second control and .002-second physics, .15 effort requests,
.35 cap, .1 slew and 3 rad/s soft governor stay fixed. No contact is added.
Require every continued branch to reproduce the corresponding preserved v2
commands, complete observation hash and summed return exactly in this runtime.

### Measures declared before reading the new outcome

Primary: paired difference (continued minus frozen) in integrated summed
sensor reward during seconds 60-120. Report all six differences and their
median within each body. Do not substitute a difference of medians for the
median of paired differences. The report horizon is undiscounted; the online
learner retains its existing discounted continuing objective.

Secondary: mean delivered light over 60-80 and 100-120 seconds, post-move
electrical energy, reward-component sums, maximum current/temperature/speed,
native vetoes and feedback flags. Report every run, including failures. Energy
is a cost measurement, not an additional reward vote. Curves at .1-second
spacing are descriptive displays of delivered readings, not independent trials.

No population-level significance or universal learning claim follows from
three known seeds on each of two known bodies. This intervention estimates
the total effect of continuing value updates in this particular moving-sun
scenario, including consequent changes in actions and visited states. It does
not separately identify the value of learning specifically caused by moving
the sun versus the effect of further training in an unchanged world; that
would need an additional stationary-sun control.

### Integrity and completion

The output directory must be new and must refuse overwrite. Archive exact
source, protocol, relevant build/runtime definitions, frozen inputs and the
reference v2 report with SHA-256 checks. Replay each entire controller and
observation trace with its same intervention. Require frozen parameter hashes
to remain identical after every post-freeze transition, and confirm that
sensing and control continue. Preserve failed reports and implementation
versions if a correction becomes necessary; do not silently replace evidence.

Success means the freeze contract is tested, all pairs have interpretable
integrity checks, results and limits are documented, and the result directs
the next engine decision. Improvement is an empirical outcome, not a required
acceptance condition. A read-only local result page will expose paired light
curves and metrics without controlling the owner's active creature.
