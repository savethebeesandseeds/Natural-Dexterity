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
function Assert-Optimizer($Optimizer, [string]$Mode, [string]$Context, [switch]$History) {
    Assert-True ($null -ne $Optimizer -and $Optimizer.available -is [bool]) "$Context has no explicit optimizer availability."
    Assert-True ($Optimizer.accepted -is [bool] -and $Optimizer.fallback -is [bool]) "$Context omits acceptance/fallback flags."
    $counterFields = @('solves', 'accepted_solves', 'fallback_solves')
    $hasDecisionCounters = $Optimizer.available -or $Optimizer.PSObject.Properties.Name -contains 'control_decisions' -or $Optimizer.status -in @('landing_support', 'observation_hold')
    if ($hasDecisionCounters) { $counterFields += @('control_decisions', 'landing_support_decisions', 'observation_hold_decisions') }
    foreach ($field in $counterFields) {
        Assert-Finite $Optimizer.$field "$Context.$field"
        Assert-True ([double]$Optimizer.$field -ge 0) "$Context.$field is negative."
        Assert-True ([Math]::Floor([double]$Optimizer.$field) -eq [double]$Optimizer.$field) "$Context.$field is not an integer count."
    }
    Assert-True (($Optimizer.accepted_solves + $Optimizer.fallback_solves) -eq $Optimizer.solves) "$Context solve accounting is inconsistent."
    if ($hasDecisionCounters) {
        Assert-True (($Optimizer.solves + $Optimizer.landing_support_decisions + $Optimizer.observation_hold_decisions) -eq $Optimizer.control_decisions) "$Context control intervals omit initial observation or planned PID landing support."
    }
    $prediction = @()
    if ($null -ne $Optimizer.prediction) { $prediction = @($Optimizer.prediction) }
    if ($History) {
        Assert-True ($Optimizer.PSObject.Properties.Name -notcontains 'prediction') 'History redundantly contains full prediction trajectories.'
    }
    $observationHold = $Optimizer.applied_controller -eq 'trim_observation_hold' -or $Optimizer.status -eq 'observation_hold'
    if ($observationHold) {
        Assert-True ($Optimizer.applied_controller -eq 'trim_observation_hold' -and $Optimizer.status -eq 'observation_hold') "$Context does not identify initial trim observation consistently."
        Assert-True ($Optimizer.optimizer_attempted -is [bool] -and -not $Optimizer.optimizer_attempted) 'Initial trim observation was counted as an optimization attempt.'
        Assert-True (-not $Optimizer.accepted -and -not $Optimizer.converged -and $prediction.Count -eq 0) 'Initial trim observation retained an accepted/converged optimizer result or trajectory.'
        Assert-True ($Optimizer.solves -eq 0 -and $Optimizer.landing_support_decisions -eq 0) 'Initial trim observation occurred after optimization or landing support.'
        if ($Mode -eq 'entropy_nmpc') {
            Assert-True ($Optimizer.available -and $Optimizer.fallback -and $Optimizer.control_decisions -eq 1 -and $Optimizer.observation_hold_decisions -eq 1) 'NMPC participation omitted or miscounted its initial trim observation interval.'
            Assert-True ($Optimizer.iterations -eq 0 -and $Optimizer.solve_ms -eq 0) 'Initial trim observation reports solver work despite no attempt.'
        } else {
            Assert-True ($Mode -eq 'pid_baseline' -and -not $Optimizer.available -and $Optimizer.control_decisions -eq 0 -and $Optimizer.observation_hold_decisions -eq 0) 'PID trim observation retained NMPC decision counters.'
        }
        foreach ($field in @('objective_j_per_k', 'seed_objective_j_per_k', 'max_constraint_violation', 'dynamics_residual', 'action_difference_norm')) {
            if ($Mode -eq 'entropy_nmpc') { Assert-True ($Optimizer.PSObject.Properties.Name -contains $field) "$Context.$field must be explicitly null during NMPC-mode observation." }
            Assert-True ($null -eq $Optimizer.$field) "$Context.$field claims an optimizer result during the trim hold."
        }
        return
    }
    if ($Mode -eq 'pid_baseline') {
        Assert-True (-not $Optimizer.available -and -not $Optimizer.accepted -and -not $Optimizer.fallback) 'PID baseline is incorrectly advertised as an optimizer result.'
        Assert-True ((-not $hasDecisionCounters -or $Optimizer.control_decisions -eq 0) -and $Optimizer.solves -eq 0 -and $prediction.Count -eq 0) 'PID baseline retained optimizer state from another run.'
        return
    }
    Assert-True ($Mode -eq 'entropy_nmpc') "$Context has an unknown controller mode."
    $landingSupport = $Optimizer.applied_controller -eq 'pid_landing_support' -or $Optimizer.status -eq 'landing_support'
    if ($landingSupport) {
        Assert-True ($Optimizer.applied_controller -eq 'pid_landing_support' -and $Optimizer.status -eq 'landing_support') "$Context does not identify planned PID landing support consistently."
        Assert-True ($Optimizer.optimizer_attempted -is [bool] -and -not $Optimizer.optimizer_attempted) 'Planned PID landing was counted as an optimization attempt.'
        Assert-True (-not $Optimizer.accepted -and -not $Optimizer.converged -and $Optimizer.fallback) 'Planned landing must select PID without claiming accepted/converged NMPC.'
        Assert-True ($Optimizer.landing_support_decisions -gt 0) 'Planned PID landing omitted its decision count.'
        Assert-True ($Optimizer.iterations -eq 0 -and $Optimizer.solve_ms -eq 0) 'Planned PID landing reports solver work despite no attempt.'
        Assert-True ($prediction.Count -eq 0) 'Planned PID landing retained an optimizer trajectory.'
        foreach ($field in @('objective_j_per_k', 'seed_objective_j_per_k', 'max_constraint_violation', 'dynamics_residual')) {
            Assert-True ($Optimizer.PSObject.Properties.Name -contains $field -and $null -eq $Optimizer.$field) "$Context.$field must be explicitly null when no optimizer ran."
        }
        return
    }
    if ($Optimizer.available) {
        Assert-True ($Optimizer.optimizer_attempted -is [bool] -and $Optimizer.optimizer_attempted) "$Context available solve was not marked as an actual attempt."
        Assert-True ($Optimizer.converged -is [bool]) "$Context conflates solver convergence with acceptance."
        Assert-True ([bool]$Optimizer.fallback -eq (-not [bool]$Optimizer.accepted)) "$Context rejected/accepted state does not match explicit fallback."
        $expectedController = if ($Optimizer.accepted) { 'entropy_nmpc' } else { 'pid_fallback' }
        Assert-True ($Optimizer.applied_controller -eq $expectedController) "$Context names the wrong applied controller."
        Assert-True (-not [string]::IsNullOrWhiteSpace([string]$Optimizer.status)) "$Context has no solver termination status."
        foreach ($field in @('iterations', 'solve_ms', 'control_interval_ms')) { Assert-Finite $Optimizer.$field "$Context.$field" }
        Assert-True ($Optimizer.solve_ms -ge 0) "$Context has a negative solve duration."
        Assert-True ($Optimizer.control_interval_ms -gt 0) "$Context has an invalid control interval."
        Assert-True ($Optimizer.within_control_interval -is [bool]) "$Context omits measured timing status."
        # History values are rounded to six decimal places; allow a half-unit
        # rounding boundary without changing the native timing decision.
        if ([Math]::Abs($Optimizer.solve_ms - $Optimizer.control_interval_ms) -gt 1e-6) {
            Assert-True ($Optimizer.within_control_interval -eq ($Optimizer.solve_ms -le $Optimizer.control_interval_ms)) "$Context timing decision is inconsistent."
        }
        if ($Optimizer.fallback) {
            Assert-True (-not [string]::IsNullOrWhiteSpace([string]$Optimizer.fallback_reason)) 'Rejected NMPC did not state why PID fallback is active.'
        } else {
            foreach ($field in @('objective_j_per_k', 'max_constraint_violation', 'dynamics_residual', 'intermediate_planning_violation')) {
                Assert-Finite $Optimizer.$field "$Context.$field"
            }
        }
    }
    if ($prediction.Count -gt 0) {
        Assert-True ($Optimizer.available -and $Optimizer.accepted -and -not $Optimizer.fallback) 'An unaccepted or fallback trajectory is being exposed as an accepted prediction.'
        $previousTime = [double]::NegativeInfinity
        foreach ($point in $prediction) {
            Assert-Finite $point.time "$Context.prediction.time"
            Assert-True ([double]$point.time -gt $previousTime) 'Prediction timestamps are not strictly ordered.'
            $previousTime = [double]$point.time
            Assert-Vector $point.position 3 "$Context.prediction.position"
            Assert-Vector $point.temperature_k 2 "$Context.prediction.temperature_k"
        }
    }
    if ($Optimizer.fallback) { Assert-True ($prediction.Count -eq 0) 'PID fallback retained a rejected prediction.' }
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
    Assert-Optimizer $State.optimizer $Mode 'state.optimizer'
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
        Assert-Optimizer $sample.optimizer $Mode 'history.optimizer' -History
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
Assert-True ($model.schema_version -ge 3 -and $model.plant.state_dimension -eq 23) 'Current physical model metadata is missing.'
Assert-True ($model.provenance.model_source_sha256 -match '^[a-f0-9]{64}$') 'Native model source fingerprint is missing.'
Assert-True ($model.provenance.evaluator_source_sha256 -match '^[a-f0-9]{64}$') 'Native evaluator source fingerprint is missing.'
Assert-True ($model.controller.objective -match 'entropy') 'Native metadata does not declare the physical entropy objective.'
Assert-True ($model.plant.equations.entropy -and $model.plant.equations.first_law) 'Physical entropy/energy equations are missing.'
Assert-Finite $model.plant.motors.main.thermal_capacity_j_k 'model.main_motor.thermal_capacity_j_k'
Assert-Finite $model.plant.environment.ambient_temperature_k 'model.ambient_temperature_k'
$validation = Get-Json '/api/evaluation'
Assert-True ($validation.validation_saved -and -not $validation.live) 'Saved validation is incorrectly marked as live.'
Assert-True $validation.source_verified 'Saved evaluation does not match the current compiled model/evaluator.'
Assert-True $validation.report.passed 'Flight evaluation failed.'
Assert-True ($validation.report.complete_evaluation -eq $true) 'Only a filtered evaluation is saved; the complete suite has not been established.'
Assert-True (@($validation.report.scenarios | Where-Object { $_.passed -ne $true }).Count -eq 0) 'Saved suite success contradicts a failed trial.'
Assert-True (@($validation.report.scenarios | Where-Object { $_.controller_mode -eq 'entropy_nmpc' }).Count -gt 0) 'Saved report contains no entropy-controller evidence.'
Assert-True (@($validation.report.scenarios | Where-Object { $_.controller_mode -eq 'pid_baseline' -and $_.autopilot -ne $false }).Count -gt 0) 'Saved report contains no enabled PID comparison.'
Assert-True ($null -ne $validation.report.comparisons -and @($validation.report.comparisons).Count -gt 0) 'Saved report has no paired comparisons.'
foreach ($trial in @($validation.report.scenarios | Where-Object { $_.controller_mode -eq 'entropy_nmpc' -and $_.autopilot -ne $false })) {
    $summary = $trial.optimizer
    foreach ($field in @('control_decisions', 'solves', 'accepted_solves', 'fallback_solves', 'landing_support_decisions', 'observation_hold_decisions', 'accepted_fraction', 'fallback_fraction', 'pid_fraction')) {
        Assert-Finite $summary.$field "saved.$($trial.name).optimizer.$field"
    }
    Assert-True ($summary.control_decisions -gt 0 -and $summary.control_decisions -eq ($summary.solves + $summary.landing_support_decisions + $summary.observation_hold_decisions)) 'Saved participation excludes observation or planned PID landing intervals.'
    Assert-True ($summary.observation_hold_decisions -eq 1) 'Saved NMPC trial omitted or repeated its initial observation interval.'
    Assert-True ($summary.solves -eq ($summary.accepted_solves + $summary.fallback_solves)) 'Saved optimizer attempt accounting is inconsistent.'
    Assert-True ([Math]::Abs($summary.accepted_fraction - ($summary.accepted_solves / $summary.control_decisions)) -lt 1e-10) 'Saved NMPC share uses the wrong denominator.'
    Assert-True ([Math]::Abs($summary.fallback_fraction - (($summary.fallback_solves + $summary.landing_support_decisions + $summary.observation_hold_decisions) / $summary.control_decisions)) -lt 1e-10) 'Saved non-NMPC share omits rejected solves, observation or planned landing support.'
    Assert-True ([Math]::Abs($summary.pid_fraction - (($summary.fallback_solves + $summary.landing_support_decisions) / $summary.control_decisions)) -lt 1e-10) 'Saved PID share incorrectly includes observation hold or omits actual PID support.'
}

$initial = Get-Json '/api/state'
$originalMode = [string]$initial.controller_mode
Assert-True ($originalMode -in @('entropy_nmpc', 'pid_baseline')) 'Initial controller mode is unknown.'
try {
$reset = Control 'reset'
Assert-True ($reset.run_id -gt $initial.run_id -and $reset.controller_mode -eq $originalMode) 'Ordinary reset did not advance run_id while preserving the controller.'
Assert-True ([Math]::Abs([double]$reset.time) -lt 1e-6) 'Reset did not return to the initial simulation time.'

$modeChecks = @()
$previousRun = $reset.run_id
foreach ($entry in @(@{ Action='pid'; Mode='pid_baseline' }, @{ Action='entropy'; Mode='entropy_nmpc' })) {
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
        solver_status = $paused.optimizer.status
        accepted_solves_observed = $paused.optimizer.accepted_solves
        fallback_solves_observed = $paused.optimizer.fallback_solves
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
Assert-True ((Request-Status '{"action":"entropy","extra":true}') -eq 400) 'Extra control fields were accepted.'
Assert-True ((Request-Status '{"action":"entropy","action":"pid"}') -eq 400) 'Duplicate action fields were accepted.'
Assert-True ((Request-Status '{"action":"play"}' @{ Origin='https://example.com' }) -eq 403) 'External control origin was accepted.'
} finally {
# Restore the requested controller even when an integration assertion fails.
$finalAction = if ($originalMode -eq 'entropy_nmpc') { 'entropy' } else { 'pid' }
$finalReset = Control $finalAction
Assert-True ($finalReset.run_id -gt $previousRun -and $finalReset.running) 'Final fresh mission did not start.'
Assert-State $finalReset $originalMode
}

$report = [ordered]@{
    checked_at_utc = [DateTime]::UtcNow.ToString('o')
    passed = $true
    source_verified = $validation.source_verified
    saved_evaluation_passed = $validation.report.passed
    complete_evaluation = $validation.report.complete_evaluation
    model_source_sha256 = $model.provenance.model_source_sha256
    evaluator_source_sha256 = $model.provenance.evaluator_source_sha256
    history_wait_budget_seconds = $historyWaitSeconds
    original_controller_mode = $originalMode
    final_controller_mode = $finalReset.controller_mode
    mode_checks = $modeChecks
    checks = @('native physical model metadata', 'saved paired evaluation passes and matches compiled source',
        'PID and entropy mode switches reset state and history', 'ordinary reset preserves each controller mode',
        'adaptive wait for at least four native samples', 'ordered finite physical and thermodynamic telemetry',
        'acceptance and fallback remain distinct', 'only accepted nonfallback predictions are exposed',
        'pause freezes time and history', 'malformed controls rejected', 'external origin rejected',
        'original controller restored in a fresh running mission')
}
$outputPath = Join-Path $PSScriptRoot '../artifacts/http-check.json'
$report | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $outputPath -Encoding utf8
$report | ConvertTo-Json -Depth 7
