param(
    [string[]]$Run = @('reward-v3-control-seed1', 'reward-v4-slip-seed1')
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$artifactRoot = Join-Path $projectRoot 'artifacts/ppo-milestone'

function Read-RunJson([string]$Path) {
    # The trainer flushes a live manifest; a read can coincide with its rewrite.
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
        catch {
            if ($attempt -eq 2) { throw }
            Start-Sleep -Milliseconds 20
        }
    }
}

$runs = @(foreach ($name in $Run) {
    $path = if ([IO.Path]::IsPathRooted($name)) { $name } else { Join-Path $artifactRoot $name }
    try {
        $manifest = Read-RunJson (Join-Path $path 'run.json')
        $screens = @(Get-ChildItem -LiteralPath (Join-Path $path 'checkpoints') -Directory |
            Sort-Object { [long]$_.Name } | ForEach-Object {
                $file = Join-Path $_.FullName 'screen.json'
                if (Test-Path -LiteralPath $file) {
                    $trials = @(Read-RunJson $file)
                    # Interrupted shutdown can save a zero-duration screen.
                    if (@($trials | Where-Object { $_.time -gt 0 }).Count -gt 0) {
                        [pscustomobject]@{
                            checkpoint = [long]$_.Name
                            path = $file
                            trials = @($trials | ForEach-Object {
                                [pscustomobject]@{
                                    command_mps = $_.command_x
                                    seconds = $_.time
                                    distance_m = $_.distance
                                    fell = $_.fall
                                    left_touchdowns = $_.left_steps
                                    right_touchdowns = $_.right_steps
                                    alternations = $_.alternating_count
                                    touchdowns_after_5s = @($_.touchdown_events | Where-Object { $_.time -gt 5 }).Count
                                    swing_diagnostics_available = ($null -ne $_.swing_diagnostics)
                                    qualifying_lifts = $_.swing_diagnostics.summary.qualifying_lifts.total
                                    late_qualifying_lifts = $_.swing_diagnostics.summary.late_qualifying_lifts.total
                                    qualifying_alternations = $_.swing_diagnostics.summary.qualifying_alternations
                                    late_qualifying_alternations = $_.swing_diagnostics.summary.late_qualifying_alternations
                                    speed_rmse_mps = $_.speed_tracking_rmse
                                    support_slip_mean_mps = $_.foot_slip.combined.mean_speed
                                    support_slip_rms_mps = $_.foot_slip.combined.rms_speed
                                }
                            })
                        }
                    }
                }
            })
        [pscustomobject]@{
            run = $path
            status = $manifest.status
            reward = $manifest.reward_version
            training_seed = $manifest.seed
            completed_transitions = $manifest.completed_steps
            additional_transitions = $manifest.additional_steps
            invocation_seconds = $manifest.seconds
            latest_training_rollout = $manifest.latest_rollout
            latest_completed_screen = $screens | Select-Object -Last 1
        }
    } catch {
        [pscustomobject]@{ run = $path; read_error = $_.Exception.Message }
    }
})

[pscustomobject]@{
    observed_utc = [DateTime]::UtcNow.ToString('o')
    scope = 'Read-only development snapshot; no final-goal acceptance or training control'
    timing_scope = 'invocation_seconds reports the trainer timer: training and export work after model/world initialization, including development screens; initialization and compilation are excluded'
    runs = $runs
} | ConvertTo-Json -Depth 8
