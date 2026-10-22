#!/bin/sh
# Two bounded native shared-learning measurements in the existing container.
# No Docker lifecycle, dependency installation, or modification of prior runs.
set -eu
umask 077

[ "$#" -eq 2 ] || {
    printf '%s\n' 'Usage: sh /workspace/run-shared-pilots.sh PROVEN_PONG_CHECKPOINT PONG_PROOF_EVALUATION' >&2
    exit 2
}
for utility in flock sed date readlink sha256sum mkdir mv dirname cat; do
    command -v "$utility" >/dev/null 2>&1 || { printf 'Required utility unavailable: %s\n' "$utility" >&2; exit 2; }
done
binary=/workspace/build/native/atari
[ -x "$binary" ] || { printf '%s\n' 'Native Atari executable is unavailable.' >&2; exit 2; }
[ "$(readlink -e /workspace/runs)" = /workspace/runs ] || {
    printf '%s\n' 'Expected canonical experiment directory /workspace/runs.' >&2; exit 2;
}
confined() {
    resolved=$(readlink -e -- "$1") || return 2
    case "$resolved" in /workspace/runs/*) ;; *) return 2 ;; esac
    # These paths are safe for the fixed JSON manifest below without escaping.
    case "$resolved" in *[!A-Za-z0-9_./-]*) return 2 ;; esac
    printf '%s\n' "$resolved"
}
file_hash() { digest=$(sha256sum "$1"); printf '%s\n' "${digest%% *}"; }
word() { sed -n "s/^  \"$1\": \"\\([a-z_]*\\)\"[,]\\{0,1\\}$/\\1/p" "$2"; }
integer() { sed -n "s/^  \"$1\": \\([0-9][0-9]*\\)[,]\\{0,1\\}$/\\1/p" "$2"; }
json_hash() { sed -n "s/^  \"$1\": \"\\([a-f0-9]\\{64\\}\\)\"[,]\\{0,1\\}$/\\1/p" "$2"; }
source_checkpoint=$(confined "$1") || { printf '%s\n' 'Checkpoint must resolve beneath /workspace/runs.' >&2; exit 2; }
source_evaluation=$(confined "$2") || { printf '%s\n' 'Proof evaluation must resolve beneath /workspace/runs.' >&2; exit 2; }
source_config=$(confined "$(dirname "$source_checkpoint")/config.json") || exit 2
for input in "$source_checkpoint" "$source_evaluation" "$source_config"; do
    [ -f "$input" ] && [ -r "$input" ] || { printf 'Unreadable source artifact: %s\n' "$input" >&2; exit 2; }
done
checkpoint_hash=$(file_hash "$source_checkpoint")
evaluation_hash=$(file_hash "$source_evaluation")
config_hash=$(file_hash "$source_config")
# This recovery sequence uses the same frozen Pong proof as the completed
# independent pilots. A different learned policy is a different experiment.
[ "$checkpoint_hash" = ea3106fbead715ef54fe144121934a0b10787962a735ac68b84fc7680f0867c8 ] &&
    [ "$evaluation_hash" = 6394cb332da333e81a06004064aa96aa3b270122a468c96299dec31f8152bc55 ] &&
    [ "$config_hash" = 260eb56c03846c0dc4e9be36e2f4df3c738f1eb820f4fbd41dc014c323995a10 ] || {
    printf '%s\n' 'Source must be the preserved 3,031,808-decision Pong proof and its exact configuration/evaluation.' >&2
    exit 2
}
[ "$(json_hash checkpoint_sha256 "$source_evaluation")" = "$checkpoint_hash" ] &&
    [ "$(word status "$source_evaluation")" = complete ] &&
    [ "$(integer episodes_completed "$source_evaluation")" = 20 ] &&
    [ "$(integer truncated_games "$source_evaluation")" = 0 ] || exit 2

# Reject every overlapping queue before creating new evidence. The pilot lease
# also excludes legacy transfer/shared queues; the extension lease excludes an
# old extension still waiting for a prerequisite. Never signal those processes.
exec 9>> /tmp/atari-shared-suite.lock
flock --nonblock --exclusive 9 || { printf '%s\n' 'A shared pilot suite is active; preserving it.' >&2; exit 2; }
exec 8>> /tmp/atari-pilot-suite.lock
flock --nonblock --exclusive 8 || { printf '%s\n' 'A pilot queue is active; preserve it and wait for its exit.' >&2; exit 2; }
exec 7>> /tmp/atari-extension-suite.lock
flock --nonblock --exclusive 7 || { printf '%s\n' 'An extension queue is active; preserve it and wait for its exit.' >&2; exit 2; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
suite=/workspace/runs/shared-suite-$stamp-$$
two_run=/workspace/runs/shared-pong-breakout-$stamp-seed41-$$
three_run=/workspace/runs/shared-pong-breakout-space-invaders-$stamp-seed51-$$
mkdir "$suite"
printf 'Shared pilot suite: %s\n' "$suite"
printf '%s\n' "$$" > "$suite/runner.pid"
exec >> "$suite/runner.log" 2>&1
queue_status=starting
phase=initialization
active_run=
child_pid=
reason=
last_child_exit=0
queue_exit=0
write_status() {
    updated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    cat > "$suite/status.json.tmp" <<EOF
{
  "status": "$queue_status",
  "phase": "$phase",
  "reason": "$reason",
  "runner_pid": $$,
  "child_pid": ${child_pid:-0},
  "active_run": "$active_run",
  "last_child_exit_code": $last_child_exit,
  "queue_exit_code": $queue_exit,
  "updated_at": "$updated"
}
EOF
    mv "$suite/status.json.tmp" "$suite/status.json"
    printf '%s\n' "$phase" > "$suite/phase"
    printf '%s\n' "${child_pid:-0}" > "$suite/child.pid"
    printf '%s %s %s %s\n' "$updated" "$queue_status" "$phase" "$reason" >> "$suite/events.log"
}
stop_child() {
    if [ -n "$child_pid" ]; then
        if [ -n "$active_run" ] && [ -d "$active_run" ]; then
            printf '%s\n' 'Shared pilot suite stopped; preserve this experiment.' > "$active_run/STOP"
        fi
        # This PID is only the direct child created below: a bounded lock waiter
        # or our native learner. Never infer another process from a PID file.
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
        child_pid=
    fi
}
on_exit() {
    queue_exit=$?
    trap - 0 HUP INT TERM
    stop_child
    if [ "$queue_status" != complete ] && [ "$queue_status" != stopped ]; then
        queue_status=failed
        [ -n "$reason" ] || reason=unexpected_runner_exit
        [ "$queue_exit" -ne 0 ] || queue_exit=1
    fi
    write_status || true
    printf 'Shared pilot suite %s: %s (exit %s)\n' "$queue_status" "$reason" "$queue_exit"
    exit "$queue_exit"
}
on_signal() { trap '' HUP INT TERM; queue_status=stopped; reason=signal_$1; stop_child; exit "$2"; }
trap on_exit 0
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
fail() { reason=$1; printf 'Shared pilot suite aborted: %s\n' "$reason" >&2; exit 1; }
check_stop() {
    if [ -e "$suite/STOP" ]; then queue_status=stopped; reason=suite_STOP; exit 130; fi
    if [ -n "$active_run" ] && [ -e "$active_run/STOP" ]; then queue_status=stopped; reason=phase_STOP; exit 130; fi
}
wait_child() {
    if wait "$child_pid"; then last_child_exit=0; else last_child_exit=$?; fi
    child_pid=
    write_status
    check_stop
    [ "$last_child_exit" -eq 0 ] || fail child_failed_or_wait_timed_out
}
validate_source_hashes() {
    [ "$(confined "$source_checkpoint")" = "$source_checkpoint" ] &&
        [ "$(confined "$source_config")" = "$source_config" ] &&
        [ "$(confined "$source_evaluation")" = "$source_evaluation" ] &&
        [ "$(file_hash "$source_checkpoint")" = "$checkpoint_hash" ] &&
        [ "$(file_hash "$source_config")" = "$config_hash" ] &&
        [ "$(file_hash "$source_evaluation")" = "$evaluation_hash" ] || fail proven_source_changed
}
require_complete() {
    checked_run=$1
    required_steps=$2
    check_stop
    for name in config.json progress.json retention.json checkpoint-manifest.json shared-final.pt; do
        [ -f "$checked_run/$name" ] && [ -r "$checked_run/$name" ] || fail completed_phase_artifact_missing
        [ "$(confined "$checked_run/$name")" = "$checked_run/$name" ] || fail completed_phase_path_changed
    done
    case "$(word status "$checked_run/progress.json")" in
        complete) ;;
        stopped) queue_status=stopped; reason=native_phase_stopped; exit 130 ;;
        *) fail native_phase_not_complete ;;
    esac
    [ "$(word status "$checked_run/retention.json")" = complete ] || fail retention_measurement_not_complete
    [ "$(integer steps "$checked_run/progress.json")" = "$required_steps" ] &&
        [ "$(integer total_new_decisions "$checked_run/retention.json")" = "$required_steps" ] &&
        [ "$(integer per_game_new_decisions "$checked_run/retention.json")" = 1000000 ] &&
        [ "$(integer total_new_decisions "$checked_run/checkpoint-manifest.json")" = "$required_steps" ] &&
        [ "$(integer per_game_new_decisions "$checked_run/checkpoint-manifest.json")" = 1000000 ] || fail phase_decision_budget_not_completed
    [ "$(integer max_seconds "$checked_run/config.json")" = 7200 ] || fail phase_training_time_budget_mismatch
    [ "$(integer evaluation_max_seconds_per_game "$checked_run/config.json")" = 3600 ] || fail phase_evaluation_time_budget_mismatch
    final_hash=$(file_hash "$checked_run/shared-final.pt")
    [ "$(json_hash shared_checkpoint_sha256 "$checked_run/retention.json")" = "$final_hash" ] || fail retention_checkpoint_hash_mismatch
    # Native retention validates every game's paired before/after evaluation,
    # exact checkpoint/config hashes, seeds, and score criteria. Completion here
    # never requires a passed score threshold and never rescues a failed one.
    cat > "$suite/$phase.evidence.json" <<EOF
{
  "run_dir": "$checked_run",
  "shared_checkpoint_sha256": "$final_hash",
  "config_sha256": "$(file_hash "$checked_run/config.json")",
  "progress_sha256": "$(file_hash "$checked_run/progress.json")",
  "retention_sha256": "$(file_hash "$checked_run/retention.json")",
  "checkpoint_manifest_sha256": "$(file_hash "$checked_run/checkpoint-manifest.json")",
  "required_new_decisions": $required_steps,
  "required_per_game_new_decisions": 1000000
}
EOF
}
run_phase() {
    phase=$1; active_run=$2; required_steps=$3
    shift 3
    check_stop
    [ ! -e "$active_run" ] && [ ! -e "$active_run.log" ] || fail preserving_existing_phase_artifacts
    queue_status=running; reason=
    write_status
    # Queue lock descriptors are closed in children; only this supervisor owns
    # those leases. The native executable acquires its own training lease.
    "$binary" "$@" 7>&- 8>&- 9>&- > "$suite/$phase.launch.log" 2>&1 &
    child_pid=$!
    write_status
    wait_child
    require_complete "$active_run" "$required_steps"
    printf 'Completed %s: %s\n' "$phase" "$active_run"
}

cat > "$suite/manifest.json" <<EOF
{
  "schema_version": 1,
  "created_at": "$stamp",
  "scope": "bounded shared2 then shared3 measurements inside the existing container; independent pilots are preserved",
  "source_checkpoint": "$source_checkpoint",
  "source_checkpoint_sha256": "$checkpoint_hash",
  "source_config": "$source_config",
  "source_config_sha256": "$config_hash",
  "source_evaluation": "$source_evaluation",
  "source_evaluation_sha256": "$evaluation_hash",
  "source_pretraining_decisions": 3031808,
  "native_binary_sha256_at_dispatch": "$(file_hash "$binary")",
  "wait_timeout_seconds": 3600,
  "evaluation_seconds_per_game": 3600,
  "evaluation_episodes_per_game_per_stage": 20,
  "max_episode_raw_frames": 108000,
  "phases": [
    {"id": "shared_two", "run_dir": "$two_run", "games": ["pong", "breakout"], "steps_total": 2000000, "steps_per_game": 1000000, "seed": 41, "environments": 8, "environments_per_game": 4, "minibatch_size": 256, "training_hours": 2, "evaluation_seed_starts": [4000100, 4100100]},
    {"id": "shared_three", "run_dir": "$three_run", "source_checkpoint": "$two_run/shared-final.pt", "games": ["pong", "breakout", "space_invaders"], "steps_total": 3000000, "steps_per_game": 1000000, "seed": 51, "environments": 12, "environments_per_game": 4, "minibatch_size": 384, "training_hours": 2, "evaluation_seed_starts": [5000100, 5100100, 5200100]}
  ],
  "budget_change": "Only wall-time guards increased: 7200 training seconds and 3600 evaluation seconds per game. Decision counts, frame caps, seeds, scoring and zero-truncation success criteria are unchanged.",
  "completion_gate": "Each phase must finish its complete decision budget and all paired before/after measurements. Missing score milestones remains a failure of that milestone, not a runner failure.",
  "stop_behavior": "Signal this supervisor for immediate direct-child cancellation, or stop its active native run. Suite STOP is checked between phases. Prior experiments are never modified."
}
EOF
phase=training_lease
queue_status=waiting
reason=waiting_for_existing_learner_lease
write_status
check_stop
# This waiter releases the training lock when true exits. Our learner then
# acquires its own lock; a racing unrelated learner makes the native CLI refuse.
flock --exclusive --timeout 3600 /tmp/atari-training.lock true 7>&- 8>&- 9>&- &
child_pid=$!
write_status
wait_child
validate_source_hashes
run_phase shared_two "$two_run" 2000000 shared-train --root /workspace --game pong \
    --source-checkpoint "$source_checkpoint" --source-evaluation "$source_evaluation" \
    --run-dir "$two_run" --steps 2000000 --seed 41 --envs 8 --hours 2 --evaluate true --evaluation-seconds 3600
check_stop
# Native shared-extend validates the full two-game checkpoint/config/retention
# lineage again before copying its encoder and heads into the three-game model.
run_phase shared_three "$three_run" 3000000 shared-extend --root /workspace --game pong \
    --source-checkpoint "$two_run/shared-final.pt" --run-dir "$three_run" \
    --steps 3000000 --seed 51 --envs 12 --hours 2 --evaluate true --evaluation-seconds 3600
check_stop
phase=finished
active_run=
queue_status=complete
reason=both_shared_measurements_completed
write_status
