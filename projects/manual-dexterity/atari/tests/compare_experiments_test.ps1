#requires -Version 7.0
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runs = Join-Path $project 'runs'
$tool = Join-Path $project 'compare-experiments.ps1'
function Read-Json([string]$Path) { return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json -AsHashtable }
function Write-Json($Value, [string]$Path) { $Value | ConvertTo-Json -Depth 60 | Set-Content -LiteralPath $Path -Encoding utf8NoBOM }
function Sha([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Host-Path([string]$Path) { return Join-Path $project $Path.Substring('/workspace/'.Length) }
function Container-Path([string]$Path) { return '/workspace/' + [IO.Path]::GetRelativePath($project, $Path).Replace('\', '/') }
function Clone($Value) { return $Value | ConvertTo-Json -Depth 60 | ConvertFrom-Json -AsHashtable }
function Require([bool]$Condition, [string]$Message) { if (!$Condition) { throw $Message } }
# Integration fixtures use the actual preserved Pong proof but synthetic target
# checkpoints. The host tool checks provenance/hash integrity, not LibTorch bytes.
# Fixtures remain clearly named under runs; no experiment data is deleted.
$proof = (Read-Json (Join-Path $runs 'proofs.json')).pong
$sourceCheckpoint = Host-Path $proof.checkpoint
$sourceConfigPath = Join-Path ([IO.Path]::GetDirectoryName($sourceCheckpoint)) 'config.json'
$sourceConfig = Read-Json $sourceConfigPath
$registry = Read-Json (Join-Path $project 'games.json')
$pong = @($registry.games | Where-Object id -CEQ pong)[0]
$breakout = @($registry.games | Where-Object id -CEQ breakout)[0]
$fixtureRoot = Join-Path $runs ('.comparison-test-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($fixtureRoot) | Out-Null
$heads = [ordered]@{}
foreach ($entry in @(@('actor.weight', @(4,512)), @('actor.bias', @(4)), @('critic.weight', @(1,512)), @('critic.bias', @(1)))) {
    $heads[$entry[0]] = @{ shape = $entry[1]; dtype = 'Float'; sha256 = 'a' * 64 }
}
$config = Clone $sourceConfig
$config.game_id = 'breakout'; $config.game_title = 'Breakout'; $config.game_spec = $breakout
$config.action_count = 4; $config.rom = '/opt/ale/roms/breakout.bin'; $config.rom_sha256 = $breakout.rom_sha256
$config.seed = 21; $config.max_agent_decisions = 1000000; $config.max_seconds = 3600
$config.previous_trained_steps = 0; $config.parent_checkpoint = $null; $config.parent_checkpoint_sha256 = $null
$config.parent_config_sha256 = $null; $config.evaluation_seed = $breakout.final_eval_seed
$config.initialization_mode = 'cold_start'; $config.continuation = 'cold_start'; $config.transfer = $null
$config.initial_adam_state_entries = 0; $config.initial_head_fingerprints = $heads
$config.tensor_fingerprint_format = 'SHA256 of contiguous CPU tensor bytes; shape and dtype recorded separately'
$shapes = [ordered]@{ 'conv1.weight' = @(32,4,8,8); 'conv1.bias' = @(32); 'conv2.weight' = @(64,32,4,4);
    'conv2.bias' = @(64); 'conv3.weight' = @(64,64,3,3); 'conv3.bias' = @(64); 'hidden.weight' = @(512,3136); 'hidden.bias' = @(512) }
$tensors = @()
foreach ($name in $shapes.Keys) {
    $fingerprint = @{ shape = $shapes[$name]; dtype = 'Float'; sha256 = 'b' * 64 }
    $tensors += @{ name = $name; source = $fingerprint; target = $fingerprint }
}
$transferConfig = Clone $config
$transferConfig.initialization_mode = 'feature_transfer_fresh_heads_and_adam'
$transferConfig.continuation = $transferConfig.initialization_mode
$transferConfig.transfer = @{ source_checkpoint = $proof.checkpoint; source_config = (Container-Path $sourceConfigPath);
    source_checkpoint_sha256 = (Sha $sourceCheckpoint); source_config_sha256 = (Sha $sourceConfigPath);
    source_pretraining_decisions = $proof.trained_decisions; source_game_id = 'pong'; source_game_spec = $pong;
    source_rom_sha256 = $pong.rom_sha256; target_step_count_at_initialization = 0; optimizer_restored = $false;
    heads_unchanged = $true; head_fingerprints = $heads; tensor_count = 8; tensors = $tensors }
function Make-Run([string]$Name, $Config, [double]$Score) {
    $directory = Join-Path $fixtureRoot $Name
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    Write-Json $Config (Join-Path $directory 'config.json')
    [IO.File]::WriteAllText((Join-Path $directory 'final.pt'), "Synthetic host comparison fixture: $Name")
    $episodes = @()
    for ($i = 0; $i -lt 20; $i++) {
        $episodes += @{ episode = $i + 1; seed = 1100100 + $i; terminated = $true; truncated = $false;
            agent_decisions = 10; raw_frames = 40; raw_return = $Score; positive_game = $true; won = $null }
    }
    $evaluation = @{ status = 'complete'; game_id = 'breakout'; game_spec = $breakout;
        rom_sha256 = $breakout.rom_sha256; preprocessing = $Config.preprocessing;
        episodes_requested = 20; episodes_completed = 20; truncated_games = 0; policy = 'deterministic_ppo';
        checkpoint = (Container-Path (Join-Path $directory 'final.pt')); checkpoint_sha256 = (Sha (Join-Path $directory 'final.pt'));
        checkpoint_config = (Container-Path (Join-Path $directory 'config.json')); checkpoint_config_sha256 = (Sha (Join-Path $directory 'config.json'));
        seed_start = 1100100; episodes = $episodes; mean_return = $Score; positive_games = 20; wins = $null;
        passed = $false; verdict = 'target_not_met'; criterion_evidence = @{ episodes_required = 20;
            episodes_completed = 20; positive_games = 20; truncated_games = 0; mean_return = $Score } }
    Write-Json $evaluation (Join-Path $directory 'evaluation.json')
    Write-Json @{ status = 'complete'; stop_reason = 'target_steps'; agent_decisions = 1000000; steps = 1000000;
        target_steps = 1000000; cumulative_agent_decisions = 1000000; checkpoint_steps = 1000000; checkpoint_run_steps = 1000000;
        previous_trained_steps = 0; game_id = 'breakout'; run_dir = (Container-Path $directory); training_elapsed_seconds = 800;
        evaluation = $evaluation } (Join-Path $directory 'progress.json')
    return $directory
}
$scratch = Make-Run 'scratch' $config 3
$transfer = Make-Run 'transfer' $transferConfig 5
$output = Join-Path $fixtureRoot 'valid.json'
$null = & $tool -ScratchRun $scratch -TransferRun $transfer -Output $output
$report = Read-Json $output
Require ($report.comparable -and $report.result.paired_mean_transfer_minus_scratch -eq 2) 'Valid paired return comparison failed.'
Require ($report.source_pretraining.agent_decisions -eq $proof.trained_decisions) 'Source cost was lost.'
Require ($report.result.paired_returns.Count -eq 20) 'Missing paired episode evidence.'
$rejections = 0
function Reject([string]$Label, [scriptblock]$Action) {
    $failed = $false
    try { & $Action | Out-Null } catch { $failed = $true }
    Require $failed "Invalid evidence accepted: $Label"
    $script:rejections++
}
Reject 'overwrite existing report' { & $tool -ScratchRun $scratch -TransferRun $transfer -Output $output }
Reject 'outside runs output' { & $tool -ScratchRun $scratch -TransferRun $transfer -Output (Join-Path $project 'illegal-comparison.json') }
$originalEvaluation = Get-Content -Raw -LiteralPath (Join-Path $transfer 'evaluation.json')
$originalProgress = Get-Content -Raw -LiteralPath (Join-Path $transfer 'progress.json')
$originalConfig = Get-Content -Raw -LiteralPath (Join-Path $transfer 'config.json')
foreach ($case in @('summary', 'seed', 'truncated', 'incomplete', 'checkpoint_hash', 'config_hash')) {
    $e = $originalEvaluation | ConvertFrom-Json -AsHashtable
    switch ($case) {
        summary { $e.mean_return = 999 }
        seed { $e.episodes[0].seed++ }
        truncated { $e.episodes[19].truncated = $true }
        incomplete { $e.status = 'incomplete'; $e.episodes_completed = 19 }
        checkpoint_hash { $e.checkpoint_sha256 = 'c' * 64 }
        config_hash { $e.checkpoint_config_sha256 = 'c' * 64 }
    }
    Write-Json $e (Join-Path $transfer 'evaluation.json')
    $p = $originalProgress | ConvertFrom-Json -AsHashtable; $p.evaluation = $e
    Write-Json $p (Join-Path $transfer 'progress.json')
    Reject $case { & $tool -ScratchRun $scratch -TransferRun $transfer -Output (Join-Path $fixtureRoot "$case.json") }
}
[IO.File]::WriteAllText((Join-Path $transfer 'evaluation.json'), $originalEvaluation)
[IO.File]::WriteAllText((Join-Path $transfer 'progress.json'), $originalProgress)
foreach ($case in @('head', 'hyperparameter', 'budget', 'copied_tensor', 'source_hash', 'source_steps')) {
    $c = Clone $transferConfig
    switch ($case) {
        head { $c.initial_head_fingerprints.'actor.weight'.sha256 = 'c' * 64 }
        hyperparameter { $c.learning_rate = .5 }
        budget { $c.max_agent_decisions = 999999 }
        copied_tensor { $c.transfer.tensors[7].target.sha256 = 'c' * 64 }
        source_hash { $c.transfer.source_checkpoint_sha256 = 'c' * 64 }
        source_steps { $c.transfer.source_pretraining_decisions++ }
    }
    Write-Json $c (Join-Path $transfer 'config.json')
    $e = $originalEvaluation | ConvertFrom-Json -AsHashtable
    $e.checkpoint_config_sha256 = Sha (Join-Path $transfer 'config.json')
    Write-Json $e (Join-Path $transfer 'evaluation.json')
    $p = $originalProgress | ConvertFrom-Json -AsHashtable; $p.evaluation = $e
    Write-Json $p (Join-Path $transfer 'progress.json')
    Reject $case { & $tool -ScratchRun $scratch -TransferRun $transfer -Output (Join-Path $fixtureRoot "$case.json") }
}
[IO.File]::WriteAllText((Join-Path $transfer 'config.json'), $originalConfig)
[IO.File]::WriteAllText((Join-Path $transfer 'evaluation.json'), $originalEvaluation)
[IO.File]::WriteAllText((Join-Path $transfer 'progress.json'), $originalProgress)
$checkpointPath = Join-Path $transfer 'final.pt'
$checkpointBytes = [IO.File]::ReadAllBytes($checkpointPath)
[IO.File]::AppendAllText($checkpointPath, 'corruption')
Reject 'checkpoint bytes changed' { & $tool -ScratchRun $scratch -TransferRun $transfer -Output (Join-Path $fixtureRoot 'checkpoint_bytes.json') }
[IO.File]::WriteAllBytes($checkpointPath, $checkpointBytes)
$capped = Make-Run 'capped-transfer' $transferConfig 120
$cappedEvaluation = Read-Json (Join-Path $capped 'evaluation.json')
$cappedEvaluation.episodes[19].terminated = $false
$cappedEvaluation.episodes[19].truncated = $true
$cappedEvaluation.episodes[19].raw_frames = 108000
$cappedEvaluation.episodes[19].positive_game = $false
$cappedEvaluation.truncated_games = 1; $cappedEvaluation.positive_games = 19
$cappedEvaluation.criterion_evidence.truncated_games = 1
$cappedEvaluation.criterion_evidence.positive_games = 19
$cappedProgress = Read-Json (Join-Path $capped 'progress.json')
$cappedProgress.evaluation = $cappedEvaluation
Write-Json $cappedEvaluation (Join-Path $capped 'evaluation.json')
Write-Json $cappedProgress (Join-Path $capped 'progress.json')
$null = & $tool -ScratchRun $scratch -TransferRun $capped -Output (Join-Path $fixtureRoot 'valid-capped.json')
$cappedReport = Read-Json (Join-Path $fixtureRoot 'valid-capped.json')
Require (!$cappedReport.full_game_comparison -and $cappedReport.bounded_episode_comparison) 'Truncated run was represented as a full-game comparison.'
Require ($cappedReport.comparability.paired_both_natural_game_count -eq 19 -and $cappedReport.transfer.truncated_games -eq 1) 'Capped report lost natural/truncated episode counts.'
Require (!$cappedReport.transfer.milestone_passed -and $cappedReport.transfer.mean_raw_return -eq 120) 'Truncation was allowed to rescue a milestone pass.'
foreach ($case in @('truncation_summary', 'rescue_pass')) {
    $invalid = Clone $cappedEvaluation
    if ($case -eq 'truncation_summary') { $invalid.truncated_games = 0 }
    else { $invalid.passed = $true; $invalid.verdict = 'target_met' }
    Write-Json $invalid (Join-Path $capped 'evaluation.json')
    $invalidProgress = Clone $cappedProgress; $invalidProgress.evaluation = $invalid
    Write-Json $invalidProgress (Join-Path $capped 'progress.json')
    Reject $case { & $tool -ScratchRun $scratch -TransferRun $capped -Output (Join-Path $fixtureRoot "$case.json") }
}
Write-Json $cappedEvaluation (Join-Path $capped 'evaluation.json')
Write-Json $cappedProgress (Join-Path $capped 'progress.json')
Write-Output "Comparison integration tests passed: valid full-game and capped-episode reports plus $rejections rejection cases. Synthetic fixtures preserved at $fixtureRoot"
