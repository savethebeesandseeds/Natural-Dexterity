#!/bin/sh
# One bounded three-game follow-up inside the existing /workspace container.
# No container lifecycle, dependency installation, or source-run mutation.
set -eu
umask 077
[ "$#" -eq 1 ] || { printf '%s\n' 'Usage: sh /workspace/run-extension.sh PILOT_SUITE' >&2; exit 2; }
for utility in flock sed date readlink sha256sum mkdir mv cat; do
    command -v "$utility" >/dev/null 2>&1 || { printf 'Required utility unavailable: %s\n' "$utility" >&2; exit 2; }
done
[ -x /workspace/build/native/atari ] || { printf '%s\n' 'Native Atari executable is unavailable.' >&2; exit 2; }
[ "$(readlink -e /workspace/runs)" = /workspace/runs ] || exit 2
confined_existing() {
    resolved=$(readlink -e -- "$1") || return 2
    case "$resolved" in /workspace/runs/*) ;; *) return 2 ;; esac
    case "$resolved" in *[!A-Za-z0-9_./-]*) return 2 ;; esac
    printf '%s\n' "$resolved"
}
word() { sed -n "s/^  \"$1\": \"\\([a-z_]*\\)\"[,]\\{0,1\\}$/\\1/p" "$2"; }
integer() { sed -n "s/^  \"$1\": \\([0-9][0-9]*\\)[,]\\{0,1\\}$/\\1/p" "$2"; }
file_hash() { hash=$(sha256sum "$1"); printf '%s\n' "${hash%% *}"; }
prerequisite=$(confined_existing "$1") || { printf '%s\n' 'Pilot suite must resolve beneath /workspace/runs with safe path characters.' >&2; exit 2; }
[ -d "$prerequisite" ] && [ -f "$prerequisite/manifest.json" ] && [ -f "$prerequisite/status.json" ] || exit 2
# The preceding runner writes each phase as one fixed-format JSON line. Accept
# exactly its shared phase; never use a nested status or infer the current run.
source_run=$(sed -n 's/^    {"id": "shared", "run_dir": "\([A-Za-z0-9_./-]*\)", .*$/\1/p' "$prerequisite/manifest.json")
case "$source_run" in /workspace/runs/shared-pong-breakout-*) ;; *) printf '%s\n' 'Pilot manifest has no unique shared phase.' >&2; exit 2 ;; esac
source_basename=${source_run#/workspace/runs/}
case "$source_basename" in *[!A-Za-z0-9_-]*) printf '%s\n' 'Shared run must be a direct child of /workspace/runs.' >&2; exit 2 ;; esac
prerequisite_manifest_hash=$(file_hash "$prerequisite/manifest.json")
source_checkpoint=$source_run/shared-final.pt
exec 9>> /tmp/atari-extension-suite.lock
flock --nonblock --exclusive 9 || { printf '%s\n' 'An extension queue is already active; preserving it.' >&2; exit 2; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
suite=/workspace/runs/extension-suite-$stamp-$$
run=/workspace/runs/shared-pong-breakout-space-invaders-$stamp-seed51-$$
mkdir "$suite"
printf 'Extension queue: %s\n' "$suite"
printf '%s\n' "$$" > "$suite/runner.pid"
exec >> "$suite/runner.log" 2>&1
queue_status=starting
phase=initialization
reason=
active_run=
child_pid=
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
        # Only our own direct child is signalled, never the prerequisite runner.
        if [ -n "$active_run" ] && [ -d "$active_run" ]; then
            printf '%s\n' 'Extension queue stopped; preserve this run.' > "$active_run/STOP"
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
    printf 'Extension queue %s: %s (exit %s)\n' "$queue_status" "$reason" "$queue_exit"
    exit "$queue_exit"
}
on_signal() { trap '' HUP INT TERM; queue_status=stopped; reason=signal_$1; stop_child; exit "$2"; }
trap on_exit 0
trap 'on_signal HUP 129' HUP
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
fail() { reason=$1; printf 'Extension aborted: %s\n' "$reason" >&2; exit 1; }
check_stops() {
    for stop in "$suite/STOP" "$prerequisite/STOP" "$source_run/STOP"; do
        if [ -e "$stop" ]; then queue_status=stopped; reason=STOP_requested; exit 130; fi
    done
}
wait_child() {
    if wait "$child_pid"; then last_child_exit=0; else last_child_exit=$?; fi
    child_pid=
    write_status
    if [ "$last_child_exit" -ne 0 ]; then
        if [ "$phase" = prerequisite ]; then fail prerequisite_lease_failed_or_timed_out; fi
        fail native_extension_failed
    fi
}
cat > "$suite/manifest.json" <<EOF
{
  "schema_version": 1,
  "created_at": "$stamp",
  "scope": "one bounded current-task three-game extension inside the existing container",
  "prerequisite_suite": "$prerequisite",
  "prerequisite_manifest_sha256": "$prerequisite_manifest_hash",
  "source_run": "$source_run",
  "source_checkpoint": "$source_checkpoint",
  "source_required_steps": 2000000,
  "source_required_seed": 41,
  "wait_timeout_seconds": 10800,
  "run_dir": "$run",
  "mode": "shared-extend",
  "games": ["pong", "breakout", "space_invaders"],
  "steps_total": 3000000,
  "steps_per_game": 1000000,
  "seed": 51,
  "environments": 12,
  "environments_per_game": 4,
  "minibatch_size": 384,
  "rollout_steps": 128,
  "epochs": 4,
  "training_hours": 1,
  "evaluate": true,
  "initialization": "Restore learned Pong and Breakout encoder/heads; fresh Space Invaders heads and Adam. Record source costs separately.",
  "completion_gate": "Suite, source progress and retention must be complete with no STOP; native code validates full lineage. Extension must finish all 3M decisions and finalized paired evaluations. Criteria may still fail.",
  "stop_behavior": "Signal this runner for immediate graceful cancellation or stop its active native run. STOP files are checked before dispatch and after completion."
}
EOF
phase=prerequisite
queue_status=waiting
reason=waiting_for_pilot_suite_lease
write_status
check_stops
# Hold fd8 after acquisition so another pilot queue cannot race this extension.
# The native learner still acquires its own independent training lease.
exec 8>> /tmp/atari-pilot-suite.lock
flock --exclusive --timeout 10800 8 &
child_pid=$!
write_status
wait_child
check_stops
[ "$(file_hash "$prerequisite/manifest.json")" = "$prerequisite_manifest_hash" ] || fail prerequisite_manifest_changed
case "$(word status "$prerequisite/status.json")" in
    complete) ;;
    stopped) queue_status=stopped; reason=prerequisite_suite_stopped; exit 130 ;;
    *) fail prerequisite_suite_not_complete ;;
esac
[ "$(confined_existing "$source_run")" = "$source_run" ] || fail source_run_path_changed
[ "$(confined_existing "$source_checkpoint")" = "$source_checkpoint" ] || fail source_checkpoint_path_changed
for file in progress.json retention.json config.json checkpoint-manifest.json; do
    [ -f "$source_run/$file" ] && [ -r "$source_run/$file" ] || fail source_evidence_missing
    [ "$(confined_existing "$source_run/$file")" = "$source_run/$file" ] || fail source_evidence_path_changed
done
[ "$(word status "$source_run/progress.json")" = complete ] &&
    [ "$(word status "$source_run/retention.json")" = complete ] || fail source_measurements_not_complete
[ "$(integer steps "$source_run/progress.json")" = 2000000 ] &&
    [ "$(integer total_new_decisions "$source_run/retention.json")" = 2000000 ] &&
    [ "$(integer total_new_decisions "$source_run/checkpoint-manifest.json")" = 2000000 ] || fail source_decision_budget_mismatch
[ "$(word training_mode "$source_run/config.json")" = shared ] &&
    [ "$(word game_id "$source_run/config.json")" = pong ] &&
    [ "$(word initialization_mode "$source_run/config.json")" = source_encoder_and_pong_heads_fresh_breakout_heads_and_adam ] &&
    [ "$(integer seed "$source_run/config.json")" = 41 ] &&
    [ "$(integer environments "$source_run/config.json")" = 8 ] &&
    [ "$(integer environments_per_game "$source_run/config.json")" = 4 ] &&
    [ "$(integer max_agent_decisions "$source_run/config.json")" = 2000000 ] &&
    [ "$(integer max_seconds "$source_run/config.json")" = 3600 ] &&
    [ "$(integer rollout_steps "$source_run/config.json")" = 128 ] &&
    [ "$(integer epochs "$source_run/config.json")" = 4 ] &&
    [ "$(integer minibatch_size "$source_run/config.json")" = 256 ] || fail source_configuration_mismatch
checkpoint_hash=$(file_hash "$source_checkpoint")
retention_checkpoint_hash=$(sed -n 's/^  "shared_checkpoint_sha256": "\([a-f0-9]\{64\}\)"[,]\{0,1\}$/\1/p' "$source_run/retention.json")
[ "$checkpoint_hash" = "$retention_checkpoint_hash" ] || fail source_checkpoint_evidence_hash_mismatch
cat > "$suite/source-validation.json" <<EOF
{
  "checkpoint_sha256": "$checkpoint_hash",
  "config_sha256": "$(file_hash "$source_run/config.json")",
  "progress_sha256": "$(file_hash "$source_run/progress.json")",
  "retention_sha256": "$(file_hash "$source_run/retention.json")",
  "checkpoint_manifest_sha256": "$(file_hash "$source_run/checkpoint-manifest.json")",
  "native_binary_sha256": "$(file_hash /workspace/build/native/atari)"
}
EOF
check_stops
[ ! -e "$run" ] && [ ! -e "$run.log" ] || fail preserving_existing_extension_artifacts
phase=shared_extension
active_run=$run
queue_status=running
reason=
write_status
/workspace/build/native/atari shared-extend --root /workspace --game pong \
    --source-checkpoint "$source_checkpoint" --run-dir "$run" \
    --steps 3000000 --seed 51 --envs 12 --hours 1 --evaluate true > "$suite/shared-extension.launch.log" 2>&1 &
child_pid=$!
write_status
wait_child
check_stops
if [ -e "$run/STOP" ] || [ "$(word status "$run/progress.json")" = stopped ]; then
    queue_status=stopped; reason=extension_stopped; exit 130
fi
[ "$(word status "$run/progress.json")" = complete ] &&
    [ "$(word status "$run/retention.json")" = complete ] || fail extension_measurements_not_complete
[ "$(integer steps "$run/progress.json")" = 3000000 ] &&
    [ "$(integer per_game_new_decisions "$run/retention.json")" = 1000000 ] || fail extension_decision_budget_not_completed
phase=finished
active_run=
queue_status=complete
reason=declared_three_game_extension_completed
write_status
