param(
    [string[]]$Run = @('v6-probe-mean', 'v6-probe-native', 'v6-probe-larger', 'v6-probe-persistent')
)

$ErrorActionPreference = 'Stop'
$artifactRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'artifacts/ppo-milestone'
$runs = @(foreach ($name in $Run) {
    $path = if ([IO.Path]::IsPathRooted($name)) { $name } else { Join-Path $artifactRoot $name }
    $manifest = Get-Content -LiteralPath (Join-Path $path 'run.json') -Raw | ConvertFrom-Json
    if ($manifest.status -ne 'diagnosed' -or $manifest.training_performed -ne $false -or $manifest.optimizer_updates -ne 0) {
        throw "Not a completed load-only diagnostic: $path"
    }
    $screens = @(Get-Content -LiteralPath (Join-Path $path 'screen.json') -Raw | ConvertFrom-Json)
    $trials = @($screens | ForEach-Object {
        if ($null -eq $_.diagnostic_playback -or $null -eq $_.after_five_seconds) { throw "Missing playback diagnostics: $path" }
        [pscustomobject]@{
            command_mps = $_.command_x
            trial = $_.diagnostic_playback.trial
            noise_seed = $_.diagnostic_playback.derived_noise_seed
            physical_reset_seed = $_.seed
            seconds = $_.time
            fell = $_.fall
            distance_m = $_.distance
            late_qualifying_lifts = $_.swing_diagnostics.summary.late_qualifying_lifts.total
            late_qualifying_alternations = $_.swing_diagnostics.summary.late_qualifying_alternations
            reached_five_seconds = $_.after_five_seconds.reached_five_seconds
            after_five = $_.after_five_seconds
            target_perturbation_rms_rad = $_.diagnostic_playback.same_state_target_delta_rms_rad_combined
            target_perturbation_rms_rad_per_joint = $_.diagnostic_playback.same_state_target_delta_rms_rad_per_joint
            near_action_bound_fraction = $_.diagnostic_playback.near_action_bound_fraction
        }
    })
    [pscustomobject]@{
        run = $path
        status = $manifest.status
        model_sha256 = $manifest.checkpoint.model_sha256
        executable_sha256 = $manifest.executable_sha256
        source_sha256 = $manifest.source_sha256
        screen_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $path 'screen.json')).Hash.ToLowerInvariant()
        noise_scale = $screens[0].diagnostic_playback.noise_scale
        noise_hold_intervals = $screens[0].diagnostic_playback.noise_hold_control_steps
        trial_count = $trials.Count
        fall_count = @($trials | Where-Object fell).Count
        trials_reaching_five_seconds = @($trials | Where-Object reached_five_seconds).Count
        trials_with_late_lifts = @($trials | Where-Object { $_.late_qualifying_lifts -gt 0 }).Count
        trials_with_late_alternations = @($trials | Where-Object { $_.late_qualifying_alternations -gt 0 }).Count
        late_qualifying_lifts_total = ($trials | Measure-Object -Property late_qualifying_lifts -Sum).Sum
        late_qualifying_alternations_total = ($trials | Measure-Object -Property late_qualifying_alternations -Sum).Sum
        observed_alive_after_five_seconds = ($trials.after_five | Measure-Object -Property alive_after_5_exposure_seconds -Sum).Sum
        # This is a trial-level mean, not a pooled interval-weighted RMS.
        mean_trial_target_perturbation_rms_rad = ($trials | Measure-Object -Property target_perturbation_rms_rad -Average).Average
        trials = $trials
    }
})
[pscustomobject]@{
    observed_utc = [DateTime]::UtcNow.ToString('o')
    scope = 'Frozen-checkpoint diagnostic playback, no optimizer updates; repeated deterministic means are references, not independent training or reset evidence'
    interpretation = 'Late lifts require completed noninitial force-support loss for at least 60ms and peak sampled sole clearance at least 1cm, landing after 5s. These diagnostics do not replace the walking gate.'
    runs = $runs
} | ConvertTo-Json -Depth 12
