# Integration checks against the real local C++ service.
# Exercises both controllers, then starts a fresh run in the original mode.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$baseUri = 'http://127.0.0.1:43118'
$historyWaitSeconds = 30

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Assert-Finite($Value, [string]$Name) {
    Assert-True ($null -ne $Value -and $Value -isnot [string] -and $Value -isnot [bool]) "$Name is not numeric."
    try { $numeric = [double]$Value } catch { throw "$Name is not numeric." }
    Assert-True (-not [double]::IsNaN($numeric) -and -not [double]::IsInfinity($numeric)) "$Name is non-finite."
}
function Assert-Vector($Value, [int]$Length, [string]$Name) {
    $components = @($Value)
    Assert-True ($components.Count -eq $Length) "$Name must have $Length components."
    foreach ($component in $components) { Assert-Finite $component $Name }
}
function Control([string]$Action) {
    Invoke-RestMethod "$baseUri/api/control" -Method Post -ContentType application/json `
        -Body (@{ action = $Action } | ConvertTo-Json -Compress) -TimeoutSec 30
}
function Get-Json([string]$Path, [int]$TimeoutSeconds = 30) {
    Invoke-RestMethod "$baseUri$Path" -TimeoutSec $TimeoutSeconds
}
function Request-Status([string]$Body, [hashtable]$Headers = @{}) {
    try {
        $response = Invoke-WebRequest "$baseUri/api/control" -Method Post -ContentType application/json `
            -Body $Body -Headers $Headers -UseBasicParsing -TimeoutSec 30
        return [int]$response.StatusCode
    } catch {
        if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
        throw
    }
}
function Wait-History($RunId, [int]$MinimumSamples = 4) {
    # Advance by observed simulation samples, not an assumed real-time ratio.
    $elapsed = [Diagnostics.Stopwatch]::StartNew()
    $lastObservation = 'No history response received.'
    while ($elapsed.Elapsed.TotalSeconds -lt $historyWaitSeconds) {
        $remaining = $historyWaitSeconds - $elapsed.Elapsed.TotalSeconds
        if ($remaining -lt 1) { break }
        $timeout = [int][Math]::Min(5, [Math]::Floor($remaining))
        $candidate = $null
        try { $candidate = Get-Json '/api/telemetry' $timeout } catch {
            # A slow solve can temporarily delay the shared snapshot lock.
            # An explicit HTTP failure is a service failure, not a timing delay.
            if ($_.Exception.Response) { throw }
            $lastObservation = $_.Exception.Message
        }
        if ($null -ne $candidate) {
            Assert-True ($candidate.run_id -eq $RunId) 'Run changed unexpectedly while waiting for history.'
            $count = @($candidate.samples).Count
            $lastObservation = "$count native samples received for run $RunId."
            if ($count -ge $MinimumSamples) {
                return [pscustomobject]@{ History = $candidate; WaitSeconds = $elapsed.Elapsed.TotalSeconds }
            }
            Assert-True (-not $candidate.finished) 'Mission ended before sufficient history samples accumulated.'
        }
        $remainingMilliseconds = [int](1000 * ($historyWaitSeconds - $elapsed.Elapsed.TotalSeconds))
        if ($remainingMilliseconds -gt 0) { Start-Sleep -Milliseconds ([Math]::Min(150, $remainingMilliseconds)) }
    }
    throw "Native telemetry did not reach $MinimumSamples samples within $historyWaitSeconds seconds. $lastObservation"
}
function Assert-Thermodynamics($Thermal, [string]$Context) {
    Assert-True ($null -ne $Thermal) "$Context has no native thermodynamic measurements."
    foreach ($field in @('entropy_rate_w_per_k', 'entropy_motor_w_per_k', 'entropy_heat_transfer_w_per_k',
        'entropy_wake_w_per_k', 'entropy_drag_w_per_k', 'cumulative_entropy_j_per_k',
        'electrical_power_w', 'electrical_energy_j', 'air_dissipation_w', 'motor_loss_w',
        'heat_rejection_w', 'stored_thermal_energy_j', 'stored_wake_energy_j',
        'terminal_cooling_entropy_j_per_k', 'terminal_entropy_commitment_j_per_k',
        'entropy_with_terminal_commitment_j_per_k', 'energy_balance_residual_w',
        'integrated_energy_residual_j', 'wind_work_j', 'contact_work_j')) {
        Assert-Finite $Thermal.$field "$Context.$field"
    }
    Assert-Vector $Thermal.motor_temperature_k 2 "$Context.motor_temperature_k"
    Assert-Vector $Thermal.motor_current_a 2 "$Context.motor_current_a"
    foreach ($temperature in $Thermal.motor_temperature_k) {
        Assert-True ([double]$temperature -gt 0) "$Context uses an invalid absolute temperature."
    }
    Assert-True ([double]$Thermal.entropy_rate_w_per_k -ge -1e-6) "$Context has negative entropy generation."
}
function Assert-Hinf($Controller, [string]$Mode, [string]$Context, [switch]$History) {
    Assert-True ($null -ne $Controller -and $Controller.available -is [bool]) "$Context has no explicit availability."
    foreach ($field in @('control_decisions', 'applied_decisions', 'fallback_decisions', 'landing_support_decisions', 'observation_hold_decisions', 'saturation_steps')) {
        Assert-Finite $Controller.$field "$Context.$field"
        Assert-True ($Controller.$field -ge 0 -and [Math]::Floor($Controller.$field) -eq $Controller.$field) "$Context has invalid decision counts."
    }
    if ($Mode -eq 'minimum_entropy_hinf') {
        Assert-True (($Controller.applied_decisions + $Controller.fallback_decisions + $Controller.landing_support_decisions + $Controller.observation_hold_decisions) -eq $Controller.control_decisions) "$Context decision accounting is inconsistent."
    }
    if ($Controller.available) {
        Assert-True ($Controller.applied_controller -eq 'minimum_entropy_hinf') "$Context claims active H-infinity diagnostics under a different applied law."
        Assert-Vector $Controller.controller_state 23 "$Context.controller_state"
        Assert-Vector $Controller.measurement 12 "$Context.measurement"
        Assert-Vector $Controller.raw_command 4 "$Context.raw_command"
        Assert-Vector $Controller.applied_command 4 "$Context.applied_command"
        Assert-Finite $Controller.update_ms "$Context.update_ms"
        Assert-True ($Controller.local_scope.within_declared_region -is [bool]) "$Context omits local scope."
    }
}
function Assert-ParameterEstimate($Estimate, [string]$Context, [switch]$History) {
    Assert-True ($null -ne $Estimate) "$Context is missing from the native snapshot."
    foreach ($field in @('mass_kg', 'mass_scale', 'accepted_samples')) { Assert-Finite $Estimate.$field "$Context.$field" }
    Assert-True ($Estimate.mass_kg -gt 0 -and $Estimate.mass_scale -gt 0) "$Context contains a nonpositive mass estimate."
    Assert-True ($Estimate.accepted_samples -ge 0 -and [Math]::Floor($Estimate.accepted_samples) -eq $Estimate.accepted_samples) "$Context accepted-sample count is invalid."
    if (-not $History) {
        foreach ($field in @('method', 'inertia_assumption')) {
            Assert-True (-not [string]::IsNullOrWhiteSpace([string]$Estimate.$field)) "$Context.$field is not documented."
        }
    }
}
function Assert-State($State, [string]$Mode) {
    Assert-True ($State.controller_mode -eq $Mode) 'State reports the wrong active controller mode.'
    Assert-Finite $State.time 'state.time'
    foreach ($field in @('position', 'target', 'velocity', 'attitude_deg', 'body_rates', 'wind')) {
        Assert-Vector $State.$field 3 "state.$field"
    }
    Assert-Vector $State.quaternion 4 'state.quaternion'
    Assert-Vector $State.input_rad 4 'state.input_rad'
    Assert-Vector $State.actuator_pitch_rad 4 'state.actuator_pitch_rad'
    Assert-Vector $State.inflow_m_s 2 'state.inflow_m_s'
    Assert-Vector $State.flap_rad 2 'state.flap_rad'
    Assert-Vector $State.model_state 23 'state.model_state'
    Assert-Thermodynamics $State.thermodynamics 'state.thermodynamics'
    Assert-Hinf $State.hinf $Mode 'state.hinf'
    if ($null -ne $model.parameter_estimation -or $null -ne $State.parameter_estimation) {
        Assert-ParameterEstimate $State.parameter_estimation 'state.parameter_estimation'
        Assert-Finite $State.mass_kg 'state.simulated_plant_mass_kg'
    }
    Assert-True ($State.execution.physics -eq 'CPU' -and $State.execution.controller -eq 'CPU') 'GPU device access was incorrectly presented as native GPU computation.'
    Assert-True ($State.execution.renderer -eq 'Browser WebGL') 'Rendering execution metadata is incorrect.'
}
function Assert-History($History, $State, [string]$Mode) {
    Assert-True ($History.schema_version -ge 2) 'Thermodynamic telemetry schema is missing.'
    Assert-True ($History.run_id -eq $State.run_id) 'History and state belong to different missions.'
    $samples = @($History.samples)
    Assert-True ($samples.Count -gt 0 -and $samples.Count -le $History.max_samples) 'History sample bound is invalid.'
    Assert-True ($History.sample_count -eq $samples.Count) 'Reported history count differs from sample count.'
    Assert-True $History.history_complete 'Short test run unexpectedly lost history.'
    Assert-True ([Math]::Abs([double]$samples[0].time) -lt 1e-6) 'History omitted the initial state.'
    $lastTime = -1.0
    foreach ($sample in $samples) {
        Assert-Finite $sample.time 'history.time'
        Assert-True ([double]$sample.time -gt $lastTime) 'History includes duplicate or unordered times.'
        $lastTime = [double]$sample.time
        Assert-True ($sample.controller_mode -eq $Mode) 'History mixes controller modes across a reset.'
        foreach ($field in @('position', 'target', 'velocity', 'attitude_deg', 'body_rates', 'wind')) {
            Assert-Vector $sample.$field 3 "history.$field"
        }
        foreach ($field in @('input_rad', 'actuator_pitch_rad')) { Assert-Vector $sample.$field 4 "history.$field" }
        foreach ($field in @('inflow_m_s', 'flap_rad')) { Assert-Vector $sample.$field 2 "history.$field" }
        Assert-Thermodynamics $sample.thermodynamics 'history.thermodynamics'
        Assert-Hinf $sample.hinf $Mode 'history.hinf' -History
        if ($null -ne $model.parameter_estimation -or $null -ne $sample.parameter_estimation) {
            Assert-ParameterEstimate $sample.parameter_estimation 'history.parameter_estimation' -History
            Assert-Finite $sample.mass_kg 'history.simulated_plant_mass_kg'
        }
    }
    # Do not assume a fresh run has fewer samples after HTTP/solver delays.
    # Its new identity, ordered initial point and paused time bound establish reset.
    Assert-True ($lastTime -le ([double]$State.time + 1e-6)) 'History contains future data leaked from the previous run.'
}

$health = Get-Json '/healthz'
Assert-True ($health.status -eq 'ok') 'Service is not healthy.'
$model = Get-Json '/api/model'
Assert-True ($model.schema_version -eq 4 -and $model.plant.state_dimension -eq 23) 'Current physical model metadata is missing.'
Assert-True ($model.provenance.model_source_sha256 -match '^[a-f0-9]{64}$') 'Native model source fingerprint is missing.'
Assert-True ($model.provenance.evaluator_source_sha256 -match '^[a-f0-9]{64}$') 'Native evaluator source fingerprint is missing.'
Assert-True ($model.plant.equations.entropy -and $model.plant.equations.first_law) 'Physical entropy/energy equations are missing.'
Assert-Finite $model.plant.motors.main.thermal_capacity_j_k 'model.main_motor.thermal_capacity_j_k'
Assert-Finite $model.plant.environment.ambient_temperature_k 'model.ambient_temperature_k'
$validation = Get-Json '/api/evaluation'
Assert-True ($validation.validation_saved -and -not $validation.live) 'Saved validation is incorrectly marked as live.'
Assert-True $validation.source_verified 'Saved evaluation does not match the current compiled model/evaluator.'
Assert-True ($validation.report.passed -is [bool]) 'Saved flight result is not an explicit Boolean.'
Assert-True ($validation.report.complete_evaluation -eq $true) 'Only a filtered evaluation is saved; the complete suite has not been established.'
$failedTrials = @($validation.report.scenarios | Where-Object { $_.passed -ne $true })
Assert-True ($validation.report.passed -eq ($failedTrials.Count -eq 0 -and $validation.report.reset_replay_identical -and $validation.report.design_passed)) 'Aggregate result contradicts its component evidence.'
Assert-True ($validation.report.hinf_trial_count -eq 7 -and $validation.report.hinf_trials_passed) 'Corrected H-infinity controller has not passed all seven required missions.'
Assert-True (@($validation.report.scenarios | Where-Object { $_.controller_mode -eq 'minimum_entropy_hinf' }).Count -gt 0) 'Saved report contains no entropy-controller evidence.'
Assert-True (@($validation.report.scenarios | Where-Object { $_.controller_mode -eq 'pid_baseline' -and $_.autopilot -ne $false }).Count -gt 0) 'Saved report contains no enabled PID comparison.'
Assert-True ($null -ne $validation.report.comparisons -and @($validation.report.comparisons).Count -gt 0) 'Saved report has no paired comparisons.'
Assert-True ($validation.report.schema_version -eq 4 -and $validation.report.method -eq 'minimum_entropy_hinf') 'Saved report describes a different method.'
Assert-True $model.hinf.passed 'Offline H-infinity synthesis checks failed.'
Assert-True ($model.hinf.selected_design.hinf_norm_upper_bound -lt $model.hinf.selected_design.gamma) 'Closed-loop H-infinity upper bound does not satisfy gamma.'
Assert-True ($model.hinf.selected_design.sampled_closed_loop_spectral_radius -lt 1) 'Sampled closed-loop stability check failed.'
foreach ($trial in @($validation.report.scenarios | Where-Object { $_.mode -eq 'hinf' })) {
    $summary = $trial.controller_checks
    Assert-True ($summary.control_decisions -eq 7000) 'Saved trial omitted full-mission decisions.'
    Assert-True (($summary.applied_decisions + $summary.fallback_decisions + $summary.landing_support_decisions + $summary.observation_hold_decisions) -eq 7000) 'Saved applied-category accounting is inconsistent.'
    Assert-True ($summary.observation_hold_decisions -eq 10) 'Saved trial omitted its initial 100 Hz hold decisions.'
    Assert-True ([Math]::Abs($summary.applied_fraction - $summary.applied_decisions / 7000) -lt 1e-10) 'Saved H-infinity share uses the wrong denominator.'
}
$initial = Get-Json '/api/state'
$originalMode = [string]$initial.controller_mode
Assert-True ($originalMode -in @('minimum_entropy_hinf', 'pid_baseline')) 'Initial controller mode is unknown.'
try {
$reset = Control 'reset'
Assert-True ($reset.run_id -gt $initial.run_id -and $reset.controller_mode -eq $originalMode) 'Ordinary reset did not advance run_id while preserving the controller.'
Assert-True ([Math]::Abs([double]$reset.time) -lt 1e-6) 'Reset did not return to the initial simulation time.'

$modeChecks = @()
$previousRun = $reset.run_id
foreach ($entry in @(@{ Action='pid'; Mode='pid_baseline' }, @{ Action='hinf'; Mode='minimum_entropy_hinf' })) {
    $started = Control $entry.Action
    Assert-True ($started.run_id -gt $previousRun -and $started.running) 'Mode switch did not start a fresh running mission.'
    Assert-True ([Math]::Abs([double]$started.time) -lt 1e-6) 'Mode switch retained the previous simulation time.'
    Assert-State $started $entry.Mode
    $waited = Wait-History $started.run_id
    $paused = Control 'pause'
    $history = Get-Json '/api/telemetry'
    Assert-State $paused $entry.Mode
    Assert-History $history $paused $entry.Mode
    Assert-True (@($history.samples).Count -ge 4) 'Pause lost accumulated native samples.'
    Start-Sleep -Milliseconds 250
    $stillPaused = Get-Json '/api/state'
    $stillHistory = Get-Json '/api/telemetry'
    Assert-True (-not $stillPaused.running -and $stillPaused.time -eq $paused.time) 'Pause changed simulation time.'
    Assert-True (@($stillHistory.samples).Count -eq @($history.samples).Count) 'Pause added duplicate graph samples.'
    Assert-History $stillHistory $stillPaused $entry.Mode
    $modeChecks += [ordered]@{
        mode = $entry.Mode
        run_id = $paused.run_id
        samples = @($history.samples).Count
        history_wait_seconds = [Math]::Round($waited.WaitSeconds, 3)
        paused_simulation_time = $paused.time
        controller_status = $paused.hinf.status
        applied_decisions_observed = $paused.hinf.applied_decisions
        fallback_decisions_observed = $paused.hinf.fallback_decisions
    }
    $preserved = Control 'reset'
    Assert-True ($preserved.run_id -gt $paused.run_id -and $preserved.controller_mode -eq $entry.Mode) 'Reset changed controller mode or retained run identity.'
    Assert-True ([Math]::Abs([double]$preserved.time) -lt 1e-6) 'Reset did not clear the previous run time.'
    $resetPaused = Control 'pause'
    $fresh = Get-Json '/api/telemetry'
    Assert-History $fresh $resetPaused $entry.Mode
    $previousRun = $preserved.run_id
}

Assert-True ((Request-Status '{"action":"invalid"}') -eq 400) 'Invalid action was accepted.'
Assert-True ((Request-Status '{"action":"entropy"}') -eq 400) 'Obsolete entropy-NMPC action was accepted.'
Assert-True ((Request-Status '{"action":"entropy","extra":true}') -eq 400) 'Extra control fields were accepted.'
Assert-True ((Request-Status '{"action":"entropy","action":"pid"}') -eq 400) 'Duplicate action fields were accepted.'
Assert-True ((Request-Status '{"action":"play"}' @{ Origin='https://example.com' }) -eq 403) 'External control origin was accepted.'
} finally {
# Restore the requested controller even when an integration assertion fails.
$finalAction = if ($originalMode -eq 'minimum_entropy_hinf') { 'hinf' } else { 'pid' }
$finalReset = Control $finalAction
Assert-True ($finalReset.run_id -gt $previousRun -and $finalReset.running) 'Final fresh mission did not start.'
Assert-State $finalReset $originalMode
}

$report = [ordered]@{
    checked_at_utc = [DateTime]::UtcNow.ToString('o')
    passed = $true
    source_verified = $validation.source_verified
    saved_evaluation_passed = $validation.report.passed
    hinf_trials_passed = $validation.report.hinf_trials_passed
    pid_trials_passed = $validation.report.pid_trials_passed
    complete_evaluation = $validation.report.complete_evaluation
    model_source_sha256 = $model.provenance.model_source_sha256
    evaluator_source_sha256 = $model.provenance.evaluator_source_sha256
    history_wait_budget_seconds = $historyWaitSeconds
    original_controller_mode = $originalMode
    final_controller_mode = $finalReset.controller_mode
    mode_checks = $modeChecks
    checks = @('native physical model metadata', 'saved paired evaluation matches source and reports both passing and failing trials honestly',
        'PID and minimum-entropy H-infinity mode switches reset state and history', 'ordinary reset preserves each controller mode',
        'adaptive wait for at least four native samples', 'ordered finite physical and thermodynamic telemetry',
        'H-infinity, hold, landing support and fallback decisions remain distinct', 'offline gain and sampled stability certificates are exposed',
        'pause freezes time and history', 'malformed controls rejected', 'external origin rejected',
        'original controller restored in a fresh running mission')
}
$outputPath = Join-Path $PSScriptRoot '../artifacts/http-check.json'
$report | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $outputPath -Encoding utf8
$report | ConvertTo-Json -Depth 7
