#requires -Version 7.0
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Run)
# Read-only verification of native shared-pilot artifacts. No publication,
# training, checkpoint deserialization, or modification of experiment evidence.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectPath = [IO.Path]::GetFullPath($PSScriptRoot)
$RunsPath = Join-Path $ProjectPath 'runs'
$Evidence = [Collections.Generic.List[hashtable]]::new()
function Require([bool]$Condition, [string]$Message) { if (!$Condition) { throw $Message } }
function Field($Object, [string]$Key) {
    Require ($Object -is [Collections.IDictionary] -and $Object.Contains($Key)) "Missing field: $Key"
    return $Object[$Key]
}
function Safe-Path([string]$Value) {
    Require (![string]::IsNullOrWhiteSpace($Value)) 'An evidence path is empty.'
    $path = if ($Value.StartsWith('/workspace/')) { Join-Path $ProjectPath $Value.Substring(11) }
        elseif ([IO.Path]::IsPathRooted($Value)) { $Value }
        elseif ($Value.Replace('\', '/').StartsWith('runs/')) { Join-Path $ProjectPath $Value }
        else { Join-Path $RunsPath $Value }
    $path = [IO.Path]::GetFullPath($path)
    Require ($path.StartsWith($RunsPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) "Evidence must remain beneath project runs: $Value"
    Require (Test-Path -LiteralPath $path) "Missing evidence: $path"
    $part = $path
    while ($part.Length -ge $RunsPath.Length) {
        Require (((Get-Item -Force -LiteralPath $part).Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "Reparse point is not an evidence path: $part"
        if ($part -ieq $RunsPath) { break }
        $part = [IO.Path]::GetDirectoryName($part)
    }
    return $path
}
function Hash([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Read-Evidence([string]$Path, [switch]$Binary) {
    $path = Safe-Path $Path; $hash = Hash $path
    $result = @{ path = $path; sha256 = $hash }
    if (!$Binary) {
        $result['data'] = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json -AsHashtable
        Require ($result.data -is [Collections.IDictionary]) "Expected JSON object: $path"
    }
    Require ((Hash $path) -ceq $hash) "Evidence changed while reading: $path"
    $Evidence.Add(@{ path = $path; sha256 = $hash }); return $result
}
function Identity($Artifact) { return [ordered]@{ path = $Artifact.path; sha256 = $Artifact.sha256 } }
function Number($Value, [string]$Label) {
    Require (($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]) -and
        [double]::IsFinite([double]$Value)) "Invalid finite number: $Label"
    return [double]$Value
}
function Integer($Value, [string]$Label) {
    $n = Number $Value $Label
    Require ($n -eq [Math]::Truncate($n) -and [Math]::Abs($n) -lt 9007199254740992) "Invalid exact integer: $Label"
    return [long]$n
}
function Equal($Left, $Right, [string]$Label) {
    if ($null -eq $Left -or $null -eq $Right) { Require ($null -eq $Left -and $null -eq $Right) "Mismatch: $Label"; return }
    if ($Left -is [Collections.IDictionary]) {
        Require ($Right -is [Collections.IDictionary] -and $Left.Count -eq $Right.Count) "Mismatch: $Label"
        foreach ($key in $Left.Keys) { Require ($Right.Contains($key)) "Missing $Label.$key"; Equal $Left[$key] $Right[$key] "$Label.$key" }
    } elseif ($Left -is [array]) {
        Require ($Right -is [array] -and $Left.Count -eq $Right.Count) "Mismatch: $Label"
        for ($i = 0; $i -lt $Left.Count; $i++) { Equal $Left[$i] $Right[$i] "$Label[$i]" }
    } elseif ($Left -is [string] -or $Left -is [bool]) {
        Require ($Left.GetType() -eq $Right.GetType() -and $Left -ceq $Right) "Mismatch: $Label"
    } else { Require ((Number $Left $Label) -eq (Number $Right $Label)) "Mismatch: $Label" }
}
function Near($Actual, [double]$Expected, [string]$Label) {
    Require ([Math]::Abs((Number $Actual $Label) - $Expected) -le 1e-9 * [Math]::Max(1, [Math]::Abs($Expected))) "Recomputed value differs: $Label"
}
$registryPath = Join-Path $ProjectPath 'games.json'
$registryHash = Hash $registryPath
$registry = Get-Content -Raw -LiteralPath $registryPath | ConvertFrom-Json -AsHashtable
Equal (Field $registry 'schema_version') 1 'registry schema'
$Games = @{}
foreach ($game in $registry.games) { Require (!$Games.ContainsKey($game.id)) 'Duplicate registered game.'; $Games[$game.id] = $game }

function Verify-Evaluation([string]$Directory, [string]$CheckpointName, $Game, [long]$Seed, $Reference, $ParentConfig, [string]$ParentRun, [long]$PreviousSteps) {
    $config = Read-Evidence (Join-Path $Directory 'config.json')
    $evaluation = Read-Evidence (Join-Path $Directory 'evaluation.json')
    $checkpoint = Read-Evidence (Join-Path $Directory $CheckpointName) -Binary
    $c = $config.data; $e = $evaluation.data
    Equal (Safe-Path (Field $Reference 'checkpoint')) $checkpoint.path 'retention checkpoint path'
    foreach ($pair in @(@('checkpoint_sha256', $checkpoint), @('config_sha256', $config), @('evaluation_sha256', $evaluation))) {
        Equal (Field $Reference $pair[0]) $pair[1].sha256 "retention $($pair[0])"
    }
    Equal (Safe-Path (Field $e 'checkpoint')) $checkpoint.path 'evaluation checkpoint path'
    Equal (Safe-Path (Field $e 'checkpoint_config')) $config.path 'evaluation config path'
    Equal (Field $e 'checkpoint_sha256') $checkpoint.sha256 'evaluated checkpoint hash'
    Equal (Field $e 'checkpoint_config_sha256') $config.sha256 'evaluated config hash'
    Equal (Field $c 'game_id') $Game.id 'export game'
    Equal (Field $c 'game_spec') $Game 'export game specification'
    Equal (Field $c 'action_count') $Game.expected_actions 'export actions'
    Equal (Field $c 'rom_sha256') $Game.rom_sha256 'export ROM hash'
    Equal (Field $c 'training_mode') 'shared_export' 'export type'
    Equal (Safe-Path (Field $c 'shared_parent_run')) $ParentRun 'export parent run'
    Equal (Safe-Path (Field $c 'joint_checkpoint')) (Join-Path $ParentRun 'shared-final.pt') 'export joint checkpoint'
    Equal (Field $c 'previous_trained_steps') $PreviousSteps 'export source decisions'
    Equal (Field $c 'optimizer_available') $false 'inference-only export'
    foreach ($key in @('network', 'preprocessing', 'algorithm', 'language', 'torch_version', 'seed')) {
        Equal (Field $c $key) (Field $ParentConfig $key) "export $key"
    }
    foreach ($key in @('game_id', 'game_spec', 'rom_sha256', 'preprocessing')) { Equal (Field $e $key) (Field $c $key) "evaluation $key" }
    Equal (Field $e 'status') 'complete' 'evaluation completion'
    Equal (Field $e 'policy') 'deterministic_ppo' 'evaluation policy'
    Equal (Field $e 'episodes_requested') 20 'requested episodes'
    Equal (Field $e 'episodes_completed') 20 'completed episodes'
    Equal (Field $e 'seed_start') $Seed 'evaluation seed'
    $rows = @(Field $e 'episodes'); Require ($rows.Count -eq 20) 'Expected exactly 20 episode records.'
    $sum = 0.0; $caps = 0; $positive = 0
    for ($i = 0; $i -lt 20; $i++) {
        $row = $rows[$i]
        Equal (Field $row 'episode') ($i + 1) 'episode order'
        Equal (Field $row 'seed') ($Seed + $i) 'episode seed'
        $terminated = Field $row 'terminated'; $truncated = Field $row 'truncated'
        Require ($terminated -is [bool] -and $truncated -is [bool] -and ($terminated -or $truncated)) 'Unfinalized episode.'
        $frames = Integer (Field $row 'raw_frames') 'episode frames'
        Require ($frames -gt 0 -and $frames -le $c.preprocessing.max_episode_raw_frames) 'Invalid episode frame count.'
        Require ((Integer (Field $row 'agent_decisions') 'episode decisions') -gt 0) 'Invalid episode decision count.'
        if ($truncated) { Equal $frames $c.preprocessing.max_episode_raw_frames 'truncation frame cap'; $caps++ }
        $score = Number (Field $row 'raw_return') 'episode return'; $sum += $score
        $positiveGame = $terminated -and !$truncated -and $score -gt 0
        Equal (Field $row 'positive_game') $positiveGame 'positive game flag'
        Equal (Field $row 'won') $(if ($Game.id -ceq 'pong') { $positiveGame } else { $null }) 'game win meaning'
        if ($positiveGame) { $positive++ }
    }
    $mean = $sum / 20; $criterion = $Game.criterion
    Equal $criterion.episodes 20 'registered criterion episode count'
    Equal $criterion.max_truncated_episodes 0 'registered zero-truncation criterion'
    $scorePassed = if ($criterion.mean_strict) { $mean -gt $criterion.mean_raw_return_threshold } else { $mean -ge $criterion.mean_raw_return_threshold }
    $passed = $caps -eq 0 -and $positive -ge $criterion.min_positive_games -and $scorePassed
    Near (Field $e 'mean_return') $mean 'evaluation mean'
    Equal (Field $e 'truncated_games') $caps 'evaluation cap count'
    Equal (Field $e 'positive_games') $positive 'evaluation positive count'
    Equal (Field $e 'wins') $(if ($Game.id -ceq 'pong') { $positive } else { $null }) 'evaluation win count'
    Equal (Field $e 'passed') $passed 'evaluation criterion'
    Equal (Field $e 'verdict') $(if ($passed) { 'target_met' } else { 'target_not_met' }) 'evaluation verdict'
    $primary = Field $e 'criterion_evidence'
    Equal (Field $primary 'episodes_required') 20 'criterion episode requirement'
    Equal (Field $primary 'episodes_completed') 20 'criterion episode completion'
    Equal (Field $primary 'positive_games') $positive 'criterion positive count'
    Equal (Field $primary 'truncated_games') $caps 'criterion cap count'
    Near (Field $primary 'mean_return') $mean 'criterion mean'
    return @{ mean = $mean; caps = $caps; positive = $positive; passed = $passed; rows = $rows;
        artifacts = [ordered]@{ checkpoint = (Identity $checkpoint); config = (Identity $config); evaluation = (Identity $evaluation) } }
}

function Verify-Run([string]$Path, [int]$Depth = 0) {
    Require ($Depth -le 1) 'Unexpected shared source recursion.'
    $directory = Safe-Path $Path
    Require (!(Test-Path -LiteralPath (Join-Path $directory 'STOP'))) "Stopped run is not completion evidence: $directory"
    $progress = Read-Evidence (Join-Path $directory 'progress.json')
    Equal (Field $progress.data 'status') 'complete' 'completed shared run required'
    $config = Read-Evidence (Join-Path $directory 'config.json')
    $retention = Read-Evidence (Join-Path $directory 'retention.json')
    $manifest = Read-Evidence (Join-Path $directory 'checkpoint-manifest.json')
    $joint = Read-Evidence (Join-Path $directory 'shared-final.pt') -Binary
    $c = $config.data; $p = $progress.data; $r = $retention.data; $m = $manifest.data
    $stage = Field $c 'shared_stage'
    Require ($stage -cin @('two_game_pilot', 'three_game_extension')) 'Unknown shared stage.'
    Require ($Depth -eq 0 -or $stage -ceq 'two_game_pilot') 'Three-game source must be the two-game pilot.'
    $ids = if ($stage -ceq 'two_game_pilot') { @('pong', 'breakout') } else { @('pong', 'breakout', 'space_invaders') }
    $total = 1000000 * $ids.Count; $trainingSeed = if ($ids.Count -eq 2) { 41 } else { 51 }
    $seedBase = if ($ids.Count -eq 2) { 4000100 } else { 5000100 }
    Equal (Field $c 'training_mode') 'shared' 'shared configuration'
    Equal (Field $c 'device') 'cuda' 'GPU training'
    Equal (Field $c 'initialization_mode') $(if ($ids.Count -eq 2) { 'source_encoder_and_pong_heads_fresh_breakout_heads_and_adam' }
        else { 'source_joint_encoder_pong_and_breakout_heads_fresh_space_invaders_heads_and_adam' }) 'shared initialization'
    Equal (Field $c 'algorithm') 'PPO' 'training algorithm'
    Equal (Field $c 'language') 'C++20' 'native runtime'
    Equal (Field $c 'advantage_normalization') 'separate within each game and rollout' 'balanced advantage normalization'
    Equal (Field $c 'rollout_steps') 128 'shared rollout'
    Equal (Field $c 'epochs') 4 'shared epochs'
    Equal (Field $c 'minibatch_size') ($ids.Count * 128) 'balanced minibatch'
    Equal (Field $c.preprocessing 'max_episode_raw_frames') 108000 'declared frame cap'
    Equal (Field $c.preprocessing 'frame_skip') 4 'declared action repeat'
    Equal (Field $c.preprocessing 'sticky_action_probability') .25 'declared sticky actions'
    Equal (Field $c.preprocessing 'terminal_on_life_loss') $false 'complete-game episode protocol'
    Equal (Field $c 'seed') $trainingSeed 'declared training seed'
    Equal (Field $c 'initial_adam_state_entries') 0 'fresh shared Adam'
    Equal (Field $c 'max_agent_decisions') $total 'declared total decisions'
    Equal (Field $c 'environments') ($ids.Count * 4) 'balanced environment count'
    Equal (Field $c 'environments_per_game') 4 'environments per game'
    Equal (Field $c 'evaluation_episodes') 20 'paired evaluation count'
    Equal (Field $c 'joint_checkpoint_game_order') $ids 'declared game order'
    Equal (Field $m 'game_order') $ids 'manifest game order'
    Equal (Field $r 'joint_checkpoint_game_order') $ids 'retention game order'
    Equal (Safe-Path (Field $p 'run_dir')) $directory 'progress run identity'
    Equal (Field $p 'stop_reason') 'target_steps' 'complete decision budget'
    Require ((Number (Field $p 'training_elapsed_seconds') 'training elapsed seconds') -gt 0) 'Shared training duration must be positive.'
    foreach ($key in @('steps', 'agent_decisions', 'target_steps', 'checkpoint_run_steps')) { Equal (Field $p $key) $total "progress $key" }
    Equal (Field $r 'status') 'complete' 'complete retention measurements'
    Equal (Field $r 'training_seed') $trainingSeed 'retention training seed'
    foreach ($document in @($m, $r)) {
        Equal (Field $document 'total_new_decisions') $total 'artifact total decisions'
        Equal (Field $document 'per_game_new_decisions') 1000000 'artifact per-game decisions'
        Equal (Field $document 'shared_checkpoint_sha256') $joint.sha256 'joint checkpoint hash'
    }
    Equal (Safe-Path (Field $m 'shared_checkpoint')) $joint.path 'manifest joint path'
    Equal (Field $m 'config_sha256') $config.sha256 'manifest config hash'
    $transfer = Field $c 'transfer'
    $sourceCheckpoint = Read-Evidence (Field $transfer 'source_checkpoint') -Binary
    $sourceConfig = Read-Evidence (Field $transfer 'source_config')
    Equal $sourceConfig.path (Join-Path ([IO.Path]::GetDirectoryName($sourceCheckpoint.path)) 'config.json') 'source adjacent config'
    Equal $sourceCheckpoint.sha256 (Field $transfer 'source_checkpoint_sha256') 'source checkpoint hash'
    Equal $sourceConfig.sha256 (Field $transfer 'source_config_sha256') 'source config hash'
    Equal (Field $r 'source_checkpoint_sha256') $sourceCheckpoint.sha256 'retention source hash'
    Equal (Field $transfer 'optimizer_restored') $false 'source optimizer not restored'
    Equal (Field $transfer 'pong_heads_restored') $true 'Pong source heads restored'
    $sourceSummary = $null
    if ($ids.Count -eq 3) {
        $sourceSummary = Verify-Run ([IO.Path]::GetDirectoryName($sourceCheckpoint.path)) ($Depth + 1)
        Equal $sourceSummary.artifacts.joint_checkpoint.sha256 $sourceCheckpoint.sha256 'verified two-game source checkpoint'
        $sourceCost = $sourceSummary.source_pretraining_decisions + $sourceSummary.new_decisions
        $previous = @{ pong = $sourceSummary.previous_game_decisions.pong + 1000000; breakout = 1000000; space_invaders = 0 }
        Equal (Field $transfer 'original_pong_pretraining_decisions') $sourceSummary.source_pretraining_decisions 'original Pong cost'
        Equal (Field $transfer 'source_shared_new_decisions') $sourceSummary.new_decisions 'source shared cost'
        Equal (Field $transfer 'breakout_heads_restored') $true 'learned Breakout head restoration'
        Equal (Field $transfer 'space_invaders_heads_restored') $false 'fresh Space Invaders heads'
        Equal (Field $transfer 'source_completion_validated') $true 'native source completion validation'
        Equal (Field $transfer 'source_retention_sha256') $sourceSummary.artifacts.retention.sha256 'source retention hash'
        Equal (Safe-Path (Field $transfer 'source_retention')) $sourceSummary.artifacts.retention.path 'source retention path'
        foreach ($item in (Field $transfer 'source_evidence_hashes').GetEnumerator()) {
            $artifact = Read-Evidence $item.Key -Binary; Equal $artifact.sha256 $item.Value 'native source evidence hash'
        }
    } else {
        Equal (Field $transfer 'breakout_heads_restored') $false 'fresh Breakout heads'
        $proofs = Read-Evidence (Join-Path $RunsPath 'proofs.json')
        $proof = Field $proofs.data 'pong'
        Equal (Field $proof 'passed') $true 'preserved Pong proof'
        Equal (Field $proof 'checkpoint_sha256') $sourceCheckpoint.sha256 'proven Pong source hash'
        Equal (Safe-Path (Field $proof 'checkpoint')) $sourceCheckpoint.path 'proven Pong source path'
        $sourceCost = Integer (Field $proof 'trained_decisions') 'source Pong decisions'
        Require ($sourceCost -gt 0) 'Source pretraining must be positive.'
        $previous = @{ pong = $sourceCost; breakout = 0 }
        $sourceEvaluation = Read-Evidence (Field $transfer 'source_evaluation')
        Equal $sourceEvaluation.sha256 (Field $transfer 'source_evaluation_sha256') 'Pong source evaluation hash'
        Equal (Safe-Path (Field $proof 'evaluation')) $sourceEvaluation.path 'preserved proof evaluation path'
        Equal (Field $sourceEvaluation.data 'checkpoint_sha256') $sourceCheckpoint.sha256 'proof evaluated source'
        Equal (Field $sourceEvaluation.data 'passed') $true 'Pong proof verdict'
        Equal (Field $sourceEvaluation.data 'episodes_completed') 20 'Pong proof count'
        Equal (Field $sourceEvaluation.data 'truncated_games') 0 'Pong proof truncations'
    }
    foreach ($document in @($transfer, $r, $m)) { Equal (Field $document 'source_pretraining_decisions') $sourceCost 'separate source training cost' }
    Equal (Field $c 'previous_trained_steps') $sourceCost 'config source cost'
    Equal (Field $p 'previous_trained_steps') $sourceCost 'progress source cost'
    Equal (Field $p 'cumulative_agent_decisions') ($sourceCost + $total) 'cumulative all-game cost'
    foreach ($document in @($c, $r, $m)) { Equal (Field $document 'previous_game_decisions') $previous 'per-game source cost' }
    Equal (Field $p 'checkpoint_steps') ($previous.pong + 1000000) 'Pong viewer checkpoint decisions'
    Require ($c.shared_games.Count -eq $ids.Count -and $r.games.Count -eq $ids.Count -and $r.artifacts.Count -eq $ids.Count -and $p.per_game.Count -eq $ids.Count) 'Missing or extra game records.'
    $summaries = [ordered]@{}; $allFull = $true
    for ($i = 0; $i -lt $ids.Count; $i++) {
        $id = $ids[$i]; Require ($Games.ContainsKey($id)) "Unknown registered game: $id"
        $game = $Games[$id]; $seed = $seedBase + 100000 * $i
        Equal $c.shared_games[$i] $game 'shared registry game'
        Equal (Field $c.evaluation_seeds $id) $seed 'paired evaluation seed'
        Near (Field $c.minibatch_game_weights $id) (1.0 / $ids.Count) 'balanced optimization weight'
        $gp = Field $p.per_game $id
        Equal (Field $gp 'agent_decisions') 1000000 'per-game completed decisions'
        Equal (Field $gp 'target_steps') 1000000 'per-game target decisions'
        Equal (Field $gp 'previous_game_decisions') $previous[$id] 'progress per-game source decisions'
        Equal (Field $gp 'cumulative_game_decisions') ($previous[$id] + 1000000) 'per-game cumulative decisions'
        $before = Verify-Evaluation (Join-Path $directory "before/$id") 'initial.pt' $game $seed $r.artifacts[$id].before $c $directory $previous[$id]
        $after = Verify-Evaluation (Join-Path $directory $id) 'final.pt' $game $seed $r.artifacts[$id].after $c $directory $previous[$id]
        $paired = Field $r.games $id
        Equal (Field $paired 'game_id') $id 'retention game identity'
        Equal (Field $paired 'complete') $true 'paired result completion'
        Equal (Field $paired 'seed_start') $seed 'paired result seed'
        Equal (Field $paired 'paired_episodes') 20 'paired result count'
        $differences = @(Field $paired 'paired_differences'); Require ($differences.Count -eq 20) 'Missing paired differences.'
        $deltaSum = 0.0; $bothNatural = 0
        for ($n = 0; $n -lt 20; $n++) {
            $delta = $after.rows[$n].raw_return - $before.rows[$n].raw_return; $deltaSum += $delta
            Equal (Field $differences[$n] 'seed') ($seed + $n) 'paired difference seed'
            Near (Field $differences[$n] 'raw_return_delta') $delta 'paired episode difference'
            Equal (Field $differences[$n] 'before_truncated') $before.rows[$n].truncated 'paired before cap'
            Equal (Field $differences[$n] 'after_truncated') $after.rows[$n].truncated 'paired after cap'
            if (!$before.rows[$n].truncated -and !$after.rows[$n].truncated) { $bothNatural++ }
        }
        foreach ($label in @('before', 'after')) {
            $value = if ($label -ceq 'before') { $before } else { $after }
            Near (Field $paired ($label + '_mean_raw_return')) $value.mean 'retention mean'
            Equal (Field $paired ($label + '_truncated_games')) $value.caps 'retention caps'
            Equal (Field $paired ($label + '_passed')) $value.passed 'retention criterion'
            if ($id -ceq 'pong') { Equal (Field $paired ($label + '_wins')) $value.positive 'retention Pong wins' }
        }
        $full = $before.caps -eq 0 -and $after.caps -eq 0; $allFull = $allFull -and $full
        $retained = $full -and $before.passed -and $after.passed
        Near (Field $paired 'paired_mean_raw_return_delta') ($deltaSum / 20) 'paired mean difference'
        Equal (Field $paired 'full_game_comparison') $full 'full-game comparison'
        Equal (Field $paired 'bounded_episode_comparison') $true 'complete bounded comparison'
        Equal (Field $paired 'comparison_scope') $(if ($full) { 'full_games' } else { 'frame_capped_episodes' }) 'paired comparison scope'
        Equal (Field $paired 'criterion_retention_applicable') $before.passed 'criterion retention applicability'
        Equal (Field $paired 'criterion_retained') $retained 'criterion retention result'
        $summaries[$id] = [ordered]@{ title = $game.title; evaluation_seed_start = $seed; paired_episodes = 20;
            before_mean = $before.mean; after_mean = $after.mean; paired_mean_delta = $deltaSum / 20;
            before_truncated = $before.caps; after_truncated = $after.caps; both_natural_pairs = $bothNatural;
            before_passed = $before.passed; after_passed = $after.passed; full_game_comparison = $full;
            criterion_retention_applicable = $before.passed; criterion_retained = $retained;
            new_decisions = 1000000; before_artifacts = $before.artifacts; after_artifacts = $after.artifacts }
    }
    Equal (Field $r 'bounded_episode_comparison') $true 'overall bounded comparison'
    Equal (Field $r 'full_game_comparison') $allFull 'overall full-game comparison'
    if ($p.Contains('retention')) { Equal $p.retention $r 'embedded progress retention' }
    return [ordered]@{ verified = $true; stage = $stage; run_dir = $directory; training_seed = $trainingSeed;
        new_decisions = $total; new_decisions_per_game = 1000000; source_pretraining_decisions = $sourceCost;
        cumulative_source_plus_new_decisions = $sourceCost + $total; previous_game_decisions = $previous;
        bounded_episode_comparison = $true; full_game_comparison = $allFull; games = $summaries;
        artifacts = [ordered]@{ config = (Identity $config); progress = (Identity $progress); retention = (Identity $retention);
            joint_checkpoint = (Identity $joint); checkpoint_manifest = (Identity $manifest);
            source_checkpoint = (Identity $sourceCheckpoint); source_config = (Identity $sourceConfig) };
        source_shared_verification = $(if ($null -ne $sourceSummary) { [ordered]@{ run_dir = $sourceSummary.run_dir; verified = $true; artifacts = $sourceSummary.artifacts } } else { $null });
        limitations = @('One training seed; descriptive paired retention, not a generalization claim.',
            'Capped episode differences do not establish full-game retention. Criterion retention applies only when the initial policy passed.',
            'Hashes bind native checkpoint/export evidence; this host verifier does not deserialize or re-execute LibTorch policies.') }
}
$result = Verify-Run $Run
foreach ($item in $Evidence) { Require ((Hash $item.path) -ceq $item.sha256) "Evidence changed during verification: $($item.path)" }
Require ((Hash $registryPath) -ceq $registryHash) 'Game registry changed during verification.'
$result['registry'] = [ordered]@{ path = $registryPath; sha256 = $registryHash }
$result | ConvertTo-Json -Depth 40
