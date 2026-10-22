# Pinned upstream recipe audit

Inspected 2026-09-17 at Unitree commit
`276801e46c5d433564f24658bac64f254b7d2d4b`, the existing asset provenance commit.
This was a source/configuration review; no upstream training runtime was executed.

The generic airtime reward is enabled by a nonzero requested command, rather
than measured forward movement. However, the G1 configuration sets that reward's
weight to zero, so it does not establish that airtime alone trained the published
G1 controller. [Generic reward implementation](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/base/legged_robot.py#L640),
[G1 configuration](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/g1/g1_config.py#L62).

The G1-specific implementation rewards contact matching a predetermined leg
phase and penalizes swing-foot height error. Its phase period is 0.8 seconds,
with opposing leg phases. These are material gait-shaping assumptions when
interpreting the upstream controller as a baseline. The current native
experiment uses measured contact events without adding a prescribed contact
schedule. [G1 environment implementation](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/g1/g1_env.py#L49).

The pinned G1 configuration uses a recurrent policy with a 64-unit LSTM,
32-unit actor/critic hidden layers, initial action-noise standard deviation
0.8 and entropy coefficient 0.01. The native history-PPO architecture and its
training recipe differ, so its present failure does not isolate PPO or history
as the cause. These settings are context, not additional changes to the recorded
v6 experiment. [Pinned policy and algorithm settings](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/g1/g1_config.py#L83).

The v6 hypothesis remains narrower: crediting an upright exploratory touchdown
without requiring existing forward speed may help escape standing. Measurements
must still distinguish useful steps, marching, chatter, sliding and falls.
