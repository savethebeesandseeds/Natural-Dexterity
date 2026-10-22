#!/bin/sh
# Sequential, bounded experiments inside the existing Atari container.
# This runner never builds dependencies or manages Docker/container lifecycle.
set -eu
umask 077

if [ "$#" -ne 3 ]; then
    printf '%s\n' 'Usage: sh /workspace/run-pilots.sh SPACE_INVADERS_RUN PONG_CHECKPOINT PONG_EVALUATION' >&2
    exit 2
fi
for utility in flock sed date readlink sha256sum mkdir mv dirname cat; do
    command -v "$utility" >/dev/null 2>&1 || { printf 'Required utility unavailable: %s\n' "$utility" >&2; exit 2; }
done
[ -x /workspace/build/native/atari ] || { printf '%s\n' 'Native Atari executable is unavailable.' >&2; exit 2; }
[ "$(readlink -e /workspace/runs)" = /workspace/runs ] || { printf '%s\n' 'Expected /workspace/runs as the canonical experiment directory.' >&2; exit 2; }

confined_path() {
    resolved=$(readlink -e -- "$1") || { printf 'Cannot resolve experiment input: %s\n' "$1" >&2; return 2; }
    case "$resolved" in
        /workspace/runs/*) ;;
        *) printf 'Input is outside /workspace/runs: %s\n' "$resolved" >&2; return 2 ;;
    esac
    # All accepted paths can be included verbatim in the manifest's JSON strings.
    # Reject spaces, quotes, backslashes, control characters and shell metacharacters.
    case "$resolved" in
        *[!A-Za-z0-9_./-]*) printf 'Unsupported characters in experiment path: %s\n' "$resolved" >&2; return 2 ;;
    esac
    printf '%s\n' "$resolved"
}
prerequisite=$(confined_path "$1")
source_checkpoint=$(confined_path "$2")
source_evaluation=$(confined_path "$3")
[ -d "$prerequisite" ] && [ -f "$prerequisite/progress.json" ] && [ -f "$prerequisite/config.json" ] || {
    printf '%s\n' 'Prerequisite must be an existing native run with progress.json and config.json.' >&2; exit 2;
}
[ -f "$source_checkpoint" ] && [ -r "$source_checkpoint" ] && [ -f "$source_evaluation" ] && [ -r "$source_evaluation" ] || {
    printf '%s\n' 'Source checkpoint and evaluation must be readable regular files.' >&2; exit 2;
}
source_config=$(confined_path "$(dirname "$source_checkpoint")/config.json")
[ -f "$source_config" ] || { printf '%s\n' 'Source checkpoint requires an adjacent config.json.' >&2; exit 2; }
# Native JSON is pretty-printed with exactly two spaces for top-level fields.
# Anchoring indentation prevents a nested evaluation status from passing a gate.
native_status() { sed -n 's/^  "status": "\([a-z_]*\)"[,]\{0,1\}$/\1/p' "$1/progress.json"; }
native_word() { sed -n "s/^  \"$1\": \"\\([a-z_]*\\)\"[,]\\{0,1\\}$/\\1/p" "$2"; }
native_integer() { sed -n "s/^  \"$1\": \\([0-9][0-9]*\\)[,]\\{0,1\\}$/\\1/p" "$2"; }
prerequisite_game=$(native_word game_id "$prerequisite/config.json")
[ "$prerequisite_game" = space_invaders ] || { printf '%s\n' 'The prerequisite must be the Space Invaders scratch run.' >&2; exit 2; }
[ "$(native_word initialization_mode "$prerequisite/config.json")" = cold_start ] &&
    [ "$(native_integer seed "$prerequisite/config.json")" = 31 ] &&
    [ "$(native_integer environments "$prerequisite/config.json")" = 8 ] &&
    [ "$(native_integer max_agent_decisions "$prerequisite/config.json")" = 1000000 ] &&
    [ "$(native_integer previous_trained_steps "$prerequisite/config.json")" = 0 ] || {
    printf '%s\n' 'Prerequisite must be the declared cold-start Space Invaders pilot: seed 31, 8 environments, 1M new decisions.' >&2; exit 2;
}

# Hold a separate kernel lease for the entire queue. Reject duplicate dispatch
# before creating any suite or run; the learner lease remains independently owned.
exec 9>> /tmp/atari-pilot-suite.lock
flock --nonblock --exclusive 9 || { printf '%s\n' 'A pilot suite is already active; preserving that queue and its runs.' >&2; exit 2; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
suite=/workspace/runs/pilot-suite-$stamp-$$
breakout_run=/workspace/runs/breakout-transfer-$stamp-seed21-$$
invaders_run=/workspace/runs/space_invaders-transfer-$stamp-seed31-$$
shared_run=/workspace/runs/shared-pong-breakout-$stamp-seed41-$$
# mkdir without -p refuses to replace any previous suite or evidence.
mkdir "$suite"
printf 'Pilot queue: %s\n' "$suite"
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
        # This is our direct child only: either the flock waiter or native phase.
        # Never signal the prerequisite learner or infer a PID from a lock file.
        if [ -n "$active_run" ] && [ -d "$active_run" ]; then
            printf '%s\n' 'Pilot queue stopped; preserve this run.' > "$active_run/STOP"
        fi
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
    printf 'Queue %s: %s (exit %s)\n' "$queue_status" "$reason" "$queue_exit"
    exit "$queue_exit"
}
on_signal() {
    trap '' HUP INT TERM
    queue_status=stopped
    reason=signal_$1
    stop_child
    exit "$2"
}
trap on_exit 0
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
fail() { reason=$1; printf 'Queue aborted: %s\n' "$reason" >&2; exit 1; }
check_queue_stop() {
    if [ -e "$suite/STOP" ]; then queue_status=stopped; reason=suite_STOP; exit 130; fi
}
require_complete() {
    check_queue_stop
    if [ -e "$1/STOP" ]; then queue_status=stopped; reason=phase_STOP; exit 130; fi
    [ -f "$1/progress.json" ] || fail phase_progress_missing
    reported_status=$(native_status "$1")
    case "$reported_status" in
        complete) ;;
        stopped) queue_status=stopped; reason=phase_stopped; exit 130 ;;
        *) printf 'Rejected top-level phase status: %s\n' "$reported_status"; fail phase_not_complete ;;
    esac
}
wait_child() {
    if wait "$child_pid"; then last_child_exit=0; else last_child_exit=$?; fi
    child_pid=
    write_status
    [ "$last_child_exit" -eq 0 ] || fail child_process_failed
}
run_phase() {
    phase=$1
    active_run=$2
    shift 2
    check_queue_stop
    [ ! -e "$active_run" ] && [ ! -e "$active_run.log" ] || fail preserving_existing_phase_artifacts
    queue_status=running
    reason=
    write_status
    # Obtain a direct child PID for responsive signal handling, then wait here.
    # Exactly one native phase runs at a time; no detached follow-up is launched.
    /workspace/build/native/atari "$@" > "$suite/$phase.launch.log" 2>&1 &
    child_pid=$!
    write_status
    wait_child
    require_complete "$active_run"
    printf 'Completed %s: %s\n' "$phase" "$active_run"
}

checkpoint_hash=$(sha256sum "$source_checkpoint"); checkpoint_hash=${checkpoint_hash%% *}
evaluation_hash=$(sha256sum "$source_evaluation"); evaluation_hash=${evaluation_hash%% *}
config_hash=$(sha256sum "$source_config"); config_hash=${config_hash%% *}
source_game=$(native_word game_id "$source_config")
[ -z "$source_game" ] || [ "$source_game" = pong ] || fail source_game_is_not_pong
source_rom=$(sed -n 's/^  "rom": "\([^"]*\)"[,]\{0,1\}$/\1/p' "$source_config")
case "$source_rom" in */pong.bin) ;; *) fail source_rom_is_not_pong ;; esac
source_passed=$(sed -n 's/^  "passed": \(true\|false\)[,]\{0,1\}$/\1/p' "$source_evaluation")
[ "$source_passed" = true ] || fail source_evaluation_did_not_pass
proof_checkpoint_hash=$(sed -n 's/^  "checkpoint_sha256": "\([a-f0-9]\{64\}\)"[,]\{0,1\}$/\1/p' "$source_evaluation")
[ "$proof_checkpoint_hash" = "$checkpoint_hash" ] || fail source_checkpoint_does_not_match_passed_evaluation
cat > "$suite/manifest.json" <<EOF
{
  "schema_version": 1,
  "created_at": "$stamp",
  "scope": "current-task bounded sequential pilots; no container lifecycle operations",
  "prerequisite_run": "$prerequisite",
  "source_checkpoint": "$source_checkpoint",
  "source_checkpoint_sha256": "$checkpoint_hash",
  "source_config": "$source_config",
  "source_config_sha256": "$config_hash",
  "source_evaluation": "$source_evaluation",
  "source_evaluation_sha256": "$evaluation_hash",
  "source_training_cost": "Read source_pretraining_decisions from each native run config; separate from target budgets.",
  "phases": [
    {"id": "breakout_transfer", "run_dir": "$breakout_run", "game": "breakout", "steps": 1000000, "seed": 21, "environments": 8, "hours": 1, "initialization": "Pong encoder only; fresh target heads and Adam"},
    {"id": "space_invaders_transfer", "run_dir": "$invaders_run", "game": "space_invaders", "steps": 1000000, "seed": 31, "environments": 8, "hours": 1, "initialization": "Pong encoder only; fresh target heads and Adam"},
    {"id": "shared", "run_dir": "$shared_run", "games": ["pong", "breakout"], "steps_total": 2000000, "steps_per_game": 1000000, "seed": 41, "environments": 8, "hours": 1, "initialization": "Pong encoder and heads; fresh Breakout heads and Adam"}
  ],
  "completion_gate": "Native top-level progress.status must be complete and STOP absent. This records completed measurements, not necessarily passed game criteria.",
  "stop_behavior": "Signal the runner for immediate graceful cancellation, or stop the active native run. A suite STOP file is checked between phases."
}
EOF
phase=prerequisite
queue_status=waiting
reason=waiting_for_existing_learner_lease
write_status
check_queue_stop
# flock blocks on the same kernel lease as main.cpp, releases it when true exits,
# and never retains the lock while a subsequent native learner acquires its own.
flock --exclusive /tmp/atari-training.lock true &
child_pid=$!
write_status
wait_child
require_complete "$prerequisite"
[ "$(native_integer steps "$prerequisite/progress.json")" = 1000000 ] || fail prerequisite_decision_budget_not_completed

run_phase breakout_transfer "$breakout_run" train --root /workspace --game breakout \
    --transfer-features "$source_checkpoint" --run-dir "$breakout_run" \
    --steps 1000000 --seed 21 --envs 8 --hours 1
run_phase space_invaders_transfer "$invaders_run" train --root /workspace --game space_invaders \
    --transfer-features "$source_checkpoint" --run-dir "$invaders_run" \
    --steps 1000000 --seed 31 --envs 8 --hours 1
run_phase shared "$shared_run" shared-train --root /workspace --game pong \
    --source-checkpoint "$source_checkpoint" --source-evaluation "$source_evaluation" \
    --run-dir "$shared_run" --steps 2000000 --seed 41 --envs 8 --hours 1 --evaluate true
check_queue_stop
phase=finished
active_run=
queue_status=complete
reason=all_declared_pilots_completed
write_status
