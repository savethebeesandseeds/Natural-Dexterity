# Droid Blocks: let rebuilding lead the roadmap

**Contributor:** Codex, 2026-09-05  
**Status:** discussion proposal, not an adopted specification or experiment protocol.  
**Basis:** the open handoff, contracts, implementation, and a read-only look at the running playground. No training, validation, or challenge rollouts were run for this review.

The idea that excites me is that building a body becomes a way of composing behavior. A child moves an eye, reverses a motor, or adds a new preference, and discovers something about the creature they made.

I would put this toy north star beside the existing research north star:

> A child can build, wake, observe, and rebuild a creature; their changes have understandable consequences, and the creature can discover useful ways to act through its new body.

Keep the scientific care already present: independent safety, physical observations, tiny baselines, immutable evidence, and honest exposure records. Bring the child's construction loop forward. Sophisticated continual learning should grow in response to that loop's failures.

## The first experience I would aim for

A child wakes a little sun-seeker. Its sensor visibly responds to the light. It tries a movement, then approaches brighter conditions. The child puts the light somewhere else and observes a response. They pause, reverse a motor connection, and wake it again. It discovers how to cooperate with the changed body. Later, the same appetite animates a stationary flower that turns its sensor toward sunlight.

That story contains several milestones, not one feature we already have. In particular, the current API can select fixed development light worlds at reset, but moving the lamp during a run requires a new environment capability. The flower requires directional light sensing. Every changed environment or topology needs a successor version; frozen v3 is body-locked.

The child should be able to ask through construction: what happens if this part goes here? Useful surprises include turning, rocking, reaching, or remaining still. An assembly with no effective transmission or no observable action consequence cannot be promised locomotion. Make such outcomes understandable and easy to revise.

## Proposed order

| Next step | Concrete deliverable | Question it answers |
| --- | --- | --- |
| Make the learned behavior tangible | A clearly labeled frozen-policy run on known development worlds, visible lamp, sensor glow driven by actual readings, and repeatable before/after trails | Can someone see that sensing causes the behavior and deliberately explore it? |
| Make one rebuild real | A minimal assembly description and generated two-motor cart family: normal, reversed transmission, and disconnected transmission | Can the system discover changed action consequences instead of depending on one cooperative motor layout? |
| Make sensing temporal | A separately versioned sample-and-hold sensor model with age, sequence, latency, and validity | Does useful behavior survive the declared sensor clock? |
| Make the same preference inhabit a different body | A stationary motor that swivels a directional light sensor | Does the approach produce more than variations of driving? |
| Make experience survive rebuilding | Compare scratch learning, a parent with small adapters, and bounded replay on the same adaptation budget | What should be remembered, and does remembering actually help? |

The timing work can proceed alongside the first assembly work. Close the existing v3 experiment under its existing approval and preservation procedure as a separate workstream. Its outcome should not gate development-only toy exploration. Begin lineage with a small record of parents, environment and topology versions, available data, and sealed challenges; expand it when needed.

## Experiments that can change our minds

**Reverse a motor before scaling the network.** The four wheel axes currently all point along `0 1 0` in [the model](../models/droid.xml). This makes common effort signs cooperate. A reversed transmission tests whether shared control can recover a useful relationship between effort and motion. A disconnected transmission distinguishes shaft movement from useful movement of the body.

Compare the current shared-rule idea with a simple successor baseline that tries bounded, distinct motor probes and keeps a small per-motor memory of their effects. Compare an observation-responsive hand-written controller as well as zero and random control. These are new comparisons, not changes to v3's acceptance rule. If two motors receive identical observations and memory, one deterministic shared rule initially gives identical actions; more parameters alone do not provide missing distinguishing information. Connector relationships, differentiated probes, and remembered consequences are candidate ways to obtain it. Keep ID renaming equivariance explicit, including how probe randomness follows module correspondence.

**Exercise the sensor clock.** The [catalog](../config/module_catalog.json) declares ambient-light sampling every 0.1 seconds with 0.02 seconds nominal latency. `policy_observation_locked()` in [simulation.cpp](../src/simulation.cpp) instead obtains fresh analytic light each observation, marks it valid, and provides no sample timestamp or sequence. [The policy](../src/policy.cpp) amplifies adjacent normalized light changes by 8192. This is a specific reason to test sample-and-hold and latency early. It is not evidence that the policy will fail. Add controlled quantization and noise sensitivity experiments; call their ranges exploratory until measurements justify physical ranges.

**Make aiming an eye matter.** `ambient_light_lux()` currently depends only on XY distance; it ignores sensor orientation, height, and occlusion. The catalog's sensing-hemisphere description is therefore not yet realized. A directional sensing model makes a swiveling flower possible and makes placement a consequential building choice. Test directionality separately before adding occlusion. Preserve v3's original light model and evidence.

**Keep relationships when preserving symmetry.** Current light observations are pooled into a mean. Two eyes can provide different information, but averaging immediately discards their correspondence. A successor should retain per-sensor histories and their relations to motors, with pooled summaries available alongside them. Renaming IDs should only relabel results; moving a sensor should be allowed to change behavior.

## Architecture choices I would keep open

Use connector information that a proposed physical connection can reliably expose. Local attachment frames are useful structural information; they do not imply authored roles such as front wheel. Separate this known structure from unmeasured properties such as transmission effects, slip, and load. Explicitly label simulated-only descriptors until there is a hardware path for obtaining them.

My first candidate is a shared core, small per-module memory, and a temporary assembly-specific adaptation state. A graph network is one option to compare when relational baselines reach their limits. Shared parameters also do not require a distributed processor in every brick: the first assembly can use one computing hub while keeping the module interface compatible with other arrangements.

Learning from scratch after every rebuild is an experiment, not a product requirement. Measure time to the first informative action, time to recover useful behavior, adaptation samples, retained competence, and safety interventions. Choose concrete responsiveness targets after identifying the intended age range and observing play; proposed seconds are not established child-development facts. Distinguish simulation acceleration from the waiting time available on a physical toy.

Define three different operations: restart the world, rebuild the body, and forget learned experience. Each should have predictable memory semantics. Re-enumeration and bounded probing belong to a deliberate wake sequence after rebuilding; moving connections should not silently preserve an obsolete action map.

## Preferences are part of the construction kit

The sensor-vote idea can make motivation tangible. Explore visibly distinct, versioned light-seeking and shade-seeking preference modules. Later, a gentle-contact preference might support a creature that leans against a hand. These would be new reward families, not reinterpretations of existing ones.

Test duplicate votes and competing wishes before adding many families. Equal opposing rewards can cancel into an uninformative objective; that is a possible experiment outcome, not automatically an interesting personality. More sensing also need not mean a stronger desire for that sensation. A module variant that observes without voting is worth discussing if compulsory preference changes make construction confusing.

Keep three questions separate: can this assembly act, can it sense the consequences, and does it prefer a different condition? No sensors means no reward under the current contract; energy conservation and impact avoidance may legitimately favor stillness. Any additional drive to explore should be an explicit design decision with its own budget and provenance.

New preference identities must be distinguishable to adaptation, through a declared versioned descriptor or subsequent learning feedback. Identical inference inputs cannot tell a frozen controller to pursue two different hidden wishes. The sum alone also should not be used to compare bodies with different numbers of voters; retain components and behavior measures.

## Let the workshop frame the child's creation

The airship console has a memorable world and could be the workshop that carries the child's inventions. In the inspected view, however, the creature occupies a small fraction of the screen. Give building and interaction a larger stage, with instruments available when wanted.

The current renderer draws a time-based pulse around the light sensor, independent of illuminance, and renders no light source: see `drawMountedSensor()` and `render()` in [app.js](../web/app.js). Showing the stimulus and actual sensor response would make causality visible. The existing details panel is a natural home for reward accounting, lineage, and engineering telemetry.

Make the active controller clear throughout interaction. The existing play control can return evaluation to privileged demo motion; a learned play session needs controls that preserve its selected mode. Describe visible probing as probing, and learning as learning only when parameters or adaptation state actually update.

A small play study should observe whether children make predictions, notice a causal change, and choose to rebuild. Returning to make another creature is a product signal worth tracking alongside reward. Avoid making the child responsible for a creature's supposed suffering; preferences and protective stops can be understandable without emotional pressure.

This emphasis on authorship and experimentation is inspired by Mitchel Resnick's account of creative learning through designing, trying, and revising meaningful projects: [MIT, “Wide-Open Spaces”](https://betterworld.mit.edu/?p=20919). It motivates a design hypothesis here, not a claim that Droid Blocks has demonstrated educational benefits.

## Smallest proposed next contribution

**Assumption challenged:** deeper continual learning must precede a convincing rebuilding experience.

**Experiment:** the two-motor cart family, a visible development-only lamp task, and shared control versus bounded per-motor probing with small persistent adapters.

**Expected observation:** normal mounting may be easy; reversing or disconnecting a transmission should reveal whether the controller identifies action consequences and whether that discovery is understandable to a player. No outcome is assumed.

**Failure boundary:** record non-movement, symmetry failures, sensor-timing sensitivity, and every simulated safety intervention. A simulated success does not qualify physical exploration.

**Exposure:** only newly declared development worlds and topology variants. No v3 challenge execution or outcomes are needed. Declare a fresh sealed set before any successor generalization claim.

**Possible artifacts:** versioned assembly schema, generated cart fixtures, temporal sensor schema, simple adaptation baseline, development traces, compact exposure record, and a visible comparison. These are proposed deliverables; this review produced only this note and its handoff link.

**Open decisions:** intended age range, first physical scale and compute envelope, what connections can measure, how quickly waking must become interesting, and which effects children find worth rebuilding for.
