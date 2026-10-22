#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ScratchRun,
    [Parameter(Mandatory)][string]$TransferRun,
    [string]$Output = ''
)
# Read-only evidence analysis; this command does not start or change experiments.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectPath = [IO.Path]::GetFullPath($PSScriptRoot)
$RunsPath = [IO.Path]::GetFullPath((Join-Path $ProjectPath 'runs'))
$RunsPrefix = $RunsPath + [IO.Path]::DirectorySeparatorChar
$Snapshots = [Collections.Generic.List[hashtable]]::new()

function Assert-That([bool]$Condition, [string]$Message) {
    if (!$Condition) { throw $Message }
}
function Field($Object, [string]$Name) {
    Assert-That ($Object -is [Collections.IDictionary] -and $Object.Contains($Name)) "Missing field: $Name"
    return $Object[$Name]
}
function Is-Number($Value) {
    return $Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]
}
function Number($Value, [string]$Name) {
    Assert-That ((Is-Number $Value) -and [double]::IsFinite([double]$Value)) "Expected finite number: $Name"
    return [double]$Value
}
function Integer($Value, [string]$Name) {
    $number = Number $Value $Name
    Assert-That ($number -eq [Math]::Truncate($number) -and [Math]::Abs($number) -lt 9007199254740992) "Expected exact integer: $Name"
    return [long]$number
}
function Same-Json($Left, $Right) {
    if ($null -eq $Left -or $null -eq $Right) { return $null -eq $Left -and $null -eq $Right }
    if ($Left -is [Collections.IDictionary]) {
        if ($Right -isnot [Collections.IDictionary] -or $Left.Count -ne $Right.Count) { return $false }
        foreach ($key in $Left.Keys) {
            if (!$Right.Contains($key) -or !(Same-Json $Left[$key] $Right[$key])) { return $false }
        }
        return $true
    }
    if ($Left -is [array]) {
        if ($Right -isnot [array] -or $Left.Count -ne $Right.Count) { return $false }
        for ($i = 0; $i -lt $Left.Count; $i++) { if (!(Same-Json $Left[$i] $Right[$i])) { return $false } }
        return $true
    }
    if (Is-Number $Left) { return (Is-Number $Right) -and [double]$Left -eq [double]$Right }
    return $Left.GetType() -eq $Right.GetType() -and $Left -ceq $Right
}
function Assert-Same($Left, $Right, [string]$Name) {
    Assert-That (Same-Json $Left $Right) "Evidence mismatch: $Name"
}
function Assert-Near($Left, $Right, [string]$Name) {
    $a = Number $Left $Name; $b = Number $Right $Name
    Assert-That ([Math]::Abs($a - $b) -le 1e-9 * [Math]::Max(1, [Math]::Abs($b))) "Summary disagrees with episodes: $Name"
}
function Safe-Path([string]$Value, [switch]$Directory, [switch]$MayNotExist) {
    Assert-That (![string]::IsNullOrWhiteSpace($Value)) 'An evidence path is empty.'
    $candidate = if ($Value.StartsWith('/workspace/')) { Join-Path $ProjectPath $Value.Substring(11) }
        elseif ([IO.Path]::IsPathRooted($Value)) { $Value }
        elseif ($Value.Replace('\', '/').StartsWith('runs/')) { Join-Path $ProjectPath $Value }
        else { Join-Path $RunsPath $Value }
    $full = [IO.Path]::GetFullPath($candidate)
    Assert-That ($full.StartsWith($RunsPrefix, [StringComparison]::OrdinalIgnoreCase)) "Evidence/output path must stay inside project runs: $Value"
    # Reject junctions/symlinks at every component, including the runs directory.
    # Lexical containment alone would permit evidence or output outside runs.
    $part = $full
    while ($part.Length -ge $RunsPath.Length) {
        if (Test-Path -LiteralPath $part) {
            $item = Get-Item -Force -LiteralPath $part
            Assert-That (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "Reparse points are not evidence paths: $part"
        }
        if ($part -ieq $RunsPath) { break }
        $part = [IO.Path]::GetDirectoryName($part)
    }
    if (!$MayNotExist) {
        $kind = if ($Directory) { 'Container' } else { 'Leaf' }
        Assert-That (Test-Path -LiteralPath $full -PathType $kind) "Missing evidence $kind`: $full"
    }
    return $full
}
function Hash([string]$Path) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }
function Check-Hash($Value, [string]$Name) {
    Assert-That ($Value -is [string] -and $Value -cmatch '^[0-9a-f]{64}$') "Invalid SHA256: $Name"
}
function Snapshot([string]$Path, [switch]$Json) {
    $safe = Safe-Path $Path
    $sha = Hash $safe
    $entry = @{ path = $safe; sha256 = $sha; bytes = (Get-Item -LiteralPath $safe).Length }
    if ($Json) {
        $entry['data'] = Get-Content -Raw -LiteralPath $safe | ConvertFrom-Json -AsHashtable
        Assert-That ($entry.data -is [Collections.IDictionary]) "Expected JSON object: $safe"
    }
    Assert-That ((Hash $safe) -ceq $sha) "Evidence changed while reading: $safe"
    $Snapshots.Add(@{ path = $safe; sha256 = $sha; bytes = $entry.bytes })
    return $entry
}
function Artifact($Snapshot) { return @{ path = $Snapshot.path; sha256 = $Snapshot.sha256; bytes = $Snapshot.bytes } }
function Fingerprint($Value, [long[]]$Shape, [string]$Name) {
    Assert-That ($Value -is [Collections.IDictionary]) "Missing tensor fingerprint: $Name"
    Assert-Same (Field $Value 'dtype') 'Float' "$Name dtype"
    $null = Field $Value 'shape'
    Assert-Same $Value['shape'] $Shape "$Name shape"
    Check-Hash (Field $Value 'sha256') $Name
}
function Read-Run([string]$Argument, [string]$Mode) {
    $directory = Safe-Path $Argument -Directory
    $config = Snapshot (Join-Path $directory 'config.json') -Json
    $progress = Snapshot (Join-Path $directory 'progress.json') -Json
    $evaluation = Snapshot (Join-Path $directory 'evaluation.json') -Json
    $checkpoint = Snapshot (Join-Path $directory 'final.pt')
    $c = $config.data; $p = $progress.data; $e = $evaluation.data
    Assert-Same (Field $c 'initialization_mode') $Mode 'initialization mode'
    Assert-Same (Field $c 'previous_trained_steps') 0 'previous trained steps'
    Assert-Same (Field $c 'initial_adam_state_entries') 0 'fresh Adam state'
    Assert-Same (Field $c 'parent_checkpoint') $null 'unexpected continuation parent'
    Assert-Same (Field $c 'max_agent_decisions') 1000000 'declared pilot budget'
    Assert-Same (Field $p 'status') 'complete' 'terminal training status'
    Assert-Same (Field $p 'stop_reason') 'target_steps' 'training stop reason'
    foreach ($key in @('agent_decisions', 'steps', 'target_steps', 'cumulative_agent_decisions', 'checkpoint_steps', 'checkpoint_run_steps')) {
        Assert-Same (Field $p $key) 1000000 "actual $key"
    }
    Assert-Same (Field $p 'previous_trained_steps') 0 'progress previous trained steps'
    Assert-Same (Field $p 'game_id') (Field $c 'game_id') 'progress game'
    Assert-Same (Safe-Path (Field $p 'run_dir') -Directory) $directory 'progress run directory'
    $seconds = Number (Field $p 'training_elapsed_seconds') 'training elapsed seconds'
    Assert-That ($seconds -gt 0) 'Training time must be positive.'
    Assert-Same (Field $e 'status') 'complete' 'complete evaluation'
    Assert-Same (Field $e 'episodes_requested') 20 'requested evaluation games'
    Assert-Same (Field $e 'episodes_completed') 20 'completed evaluation games'
    Assert-Same (Field $e 'policy') 'deterministic_ppo' 'evaluation policy'
    Assert-Same (Safe-Path (Field $e 'checkpoint')) $checkpoint.path 'evaluated final checkpoint path'
    Assert-Same (Field $e 'checkpoint_sha256') $checkpoint.sha256 'evaluated final checkpoint hash'
    Assert-Same (Safe-Path (Field $e 'checkpoint_config')) $config.path 'evaluated config path'
    Assert-Same (Field $e 'checkpoint_config_sha256') $config.sha256 'evaluated config hash'
    foreach ($key in @('game_id', 'game_spec', 'rom_sha256', 'preprocessing')) {
        Assert-Same (Field $e $key) (Field $c $key) "evaluation $key"
    }
    if ($p.Contains('evaluation')) { Assert-Same $p.evaluation $e 'embedded progress evaluation' }
    $spec = Field $c 'game_spec'
    Assert-Same (Field $spec 'id') $c.game_id 'game specification identity'
    Assert-Same (Field $spec 'rom_sha256') $c.rom_sha256 'game specification ROM'
    Assert-Same (Field $spec 'expected_actions') (Field $c 'action_count') 'game action count'
    $seed = Integer (Field $e 'seed_start') 'evaluation seed'
    Assert-Same $seed (Field $c 'evaluation_seed') 'configured evaluation seed'
    Assert-Same $seed (Field $spec 'final_eval_seed') 'registered evaluation seed'
    Assert-Same (Field $c 'evaluation_episodes') 20 'configured evaluation count'
    $episodes = @(Field $e 'episodes')
    Assert-That ($episodes.Count -eq 20) 'Evaluation must contain exactly 20 episode records.'
    $sum = 0.0; $positive = 0; $truncated = 0; $returns = [Collections.Generic.List[hashtable]]::new()
    $frameLimit = Integer (Field $c.preprocessing 'max_episode_raw_frames') 'episode frame limit'
    Assert-That ($frameLimit -gt 0) 'The bounded evaluation protocol needs a positive frame limit.'
    for ($i = 0; $i -lt 20; $i++) {
        $row = $episodes[$i]
        Assert-Same (Field $row 'episode') ($i + 1) 'episode order'
        Assert-Same (Field $row 'seed') ($seed + $i) 'episode seed order'
        $terminated = Field $row 'terminated'; $capped = Field $row 'truncated'
        Assert-That ($terminated -is [bool] -and $capped -is [bool] -and ($terminated -or $capped)) 'Episode must have a truthful terminal or truncation flag.'
        Assert-That ((Integer (Field $row 'agent_decisions') 'episode decisions') -gt 0) 'Episode needs positive decisions.'
        $rawFrames = Integer (Field $row 'raw_frames') 'episode frames'
        Assert-That ($rawFrames -gt 0 -and $rawFrames -le $frameLimit) 'Episode raw frames must stay within the configured cap.'
        if ($capped) { Assert-Same $rawFrames $frameLimit 'truncated episode reached its exact frame cap'; $truncated++ }
        $score = Number (Field $row 'raw_return') 'episode raw return'
        $positiveGame = $terminated -and !$capped -and $score -gt 0
        Assert-Same (Field $row 'positive_game') $positiveGame 'positive game flag'
        if ($c.game_id -ceq 'pong') { Assert-Same (Field $row 'won') $positiveGame 'Pong win' }
        else { Assert-Same (Field $row 'won') $null 'non-Pong win must be null' }
        $sum += $score; if ($positiveGame) { $positive++ }
        $returns.Add(@{ seed = $seed + $i; raw_return = $score; terminated = $terminated; truncated = $capped; raw_frames = $rawFrames })
    }
    $mean = $sum / 20
    Assert-Near (Field $e 'mean_return') $mean 'mean return'
    Assert-Same (Field $e 'positive_games') $positive 'positive games summary'
    Assert-Same (Field $e 'truncated_games') $truncated 'truncated episodes summary'
    $criterion = Field $spec 'criterion'
    Assert-Same (Field $criterion 'episodes') 20 'criterion game count'
    Assert-Same (Field $criterion 'max_truncated_episodes') 0 'criterion truncations'
    $threshold = Number (Field $criterion 'mean_raw_return_threshold') 'score threshold'
    $strict = Field $criterion 'mean_strict'
    Assert-That ($strict -is [bool]) 'mean_strict must be boolean.'
    $passed = $truncated -eq 0 -and $positive -ge (Integer (Field $criterion 'min_positive_games') 'minimum positive games') -and
        $(if ($strict) { $mean -gt $threshold } else { $mean -ge $threshold })
    Assert-Same (Field $e 'passed') $passed 'recomputed milestone verdict'
    Assert-Same (Field $e 'verdict') $(if ($passed) { 'target_met' } else { 'target_not_met' }) 'milestone verdict label'
    $primary = Field $e 'criterion_evidence'
    Assert-Same (Field $primary 'episodes_required') 20 'criterion required games'
    Assert-Same (Field $primary 'episodes_completed') 20 'criterion completed games'
    Assert-Same (Field $primary 'positive_games') $positive 'criterion positive games'
    Assert-Same (Field $primary 'truncated_games') $truncated 'criterion truncations'
    Assert-Near (Field $primary 'mean_return') $mean 'criterion mean return'
    $heads = Field $c 'initial_head_fingerprints'
    Assert-That ($heads -is [Collections.IDictionary] -and $heads.Count -eq 4) 'Expected all four initial head fingerprints.'
    $actions = Integer $c.action_count 'action count'
    Fingerprint (Field $heads 'actor.weight') @($actions, 512) 'actor.weight'
    Fingerprint (Field $heads 'actor.bias') @($actions) 'actor.bias'
    Fingerprint (Field $heads 'critic.weight') @(1, 512) 'critic.weight'
    Fingerprint (Field $heads 'critic.bias') @(1) 'critic.bias'
    return @{ directory = $directory; config = $config; progress = $progress; evaluation = $evaluation;
        checkpoint = $checkpoint; returns = $returns.ToArray(); mean = $mean; positive_games = $positive;
        passed = $passed; truncated_games = $truncated; training_seconds = $seconds }
}

$scratch = Read-Run $ScratchRun 'cold_start'
$transfer = Read-Run $TransferRun 'feature_transfer_fresh_heads_and_adam'
Assert-That ($scratch.directory -ine $transfer.directory) 'Scratch and transfer runs must be distinct.'
$sc = $scratch.config.data; $tc = $transfer.config.data
$matchedKeys = @('game_id', 'game_spec', 'algorithm', 'language', 'device', 'torch_version',
    'training_action_sampling', 'rom_sha256', 'action_count', 'seed', 'max_agent_decisions', 'max_seconds',
    'environments', 'rollout_steps', 'epochs', 'minibatch_size', 'gamma', 'gae_lambda', 'clip_range',
    'entropy_coefficient', 'learning_rate', 'value_coefficient', 'max_gradient_norm', 'preprocessing',
    'evaluation_seed', 'evaluation_episodes', 'network', 'budget_scope', 'checkpoint_steps_scope',
    'initial_head_fingerprints', 'tensor_fingerprint_format')
foreach ($key in $matchedKeys) { Assert-Same (Field $sc $key) (Field $tc $key) "paired configuration $key" }
Assert-Same (Field $sc 'transfer') $null 'scratch run transfer metadata'
Assert-That ($sc.game_id -cin @('breakout', 'space_invaders')) 'This protocol compares the declared Breakout/Space Invaders pilots.'
Assert-Same $sc.algorithm 'PPO' 'pilot algorithm'
Assert-Same $sc.language 'C++20' 'native runtime language'
Assert-Same $sc.device 'cuda' 'GPU training'
Assert-Same $sc.environments 8 'pilot worker count'
Assert-Same $sc.seed $(if ($sc.game_id -ceq 'breakout') { 21 } else { 31 }) 'declared training seed'
$provenance = Field $tc 'transfer'
Assert-Same (Field $provenance 'target_step_count_at_initialization') 0 'transfer target starts at zero decisions'
Assert-Same (Field $provenance 'optimizer_restored') $false 'transfer optimizer reset'
Assert-Same (Field $provenance 'heads_unchanged') $true 'transfer heads unchanged'
Assert-Same (Field $provenance 'head_fingerprints') $tc.initial_head_fingerprints 'transfer initial head fingerprints'
Assert-Same (Field $provenance 'tensor_count') 8 'copied feature tensor count'
$featureShapes = [ordered]@{ 'conv1.weight' = @(32, 4, 8, 8); 'conv1.bias' = @(32);
    'conv2.weight' = @(64, 32, 4, 4); 'conv2.bias' = @(64); 'conv3.weight' = @(64, 64, 3, 3);
    'conv3.bias' = @(64); 'hidden.weight' = @(512, 3136); 'hidden.bias' = @(512) }
$features = @(Field $provenance 'tensors')
Assert-That ($features.Count -eq 8) 'Expected eight feature-copy records.'
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($feature in $features) {
    $name = Field $feature 'name'
    Assert-That ($name -is [string] -and $featureShapes.Contains($name) -and $seen.Add($name)) "Unexpected or duplicate feature tensor: $name"
    Fingerprint (Field $feature 'source') $featureShapes[$name] "source $name"
    Fingerprint (Field $feature 'target') $featureShapes[$name] "target $name"
    Assert-Same $feature.source $feature.target "exact copied feature $name"
}
$sourceCheckpoint = Snapshot (Field $provenance 'source_checkpoint')
$sourceConfig = Snapshot (Field $provenance 'source_config') -Json
Assert-Same $sourceConfig.path (Join-Path ([IO.Path]::GetDirectoryName($sourceCheckpoint.path)) 'config.json') 'adjacent source config'
Assert-Same $sourceCheckpoint.sha256 (Field $provenance 'source_checkpoint_sha256') 'transfer source checkpoint hash'
Assert-Same $sourceConfig.sha256 (Field $provenance 'source_config_sha256') 'transfer source config hash'
Assert-Same (Field $provenance 'source_game_id') 'pong' 'declared source game'
$sourceSpec = Field $provenance 'source_game_spec'
Assert-Same (Field $sourceSpec 'id') 'pong' 'source game specification'
Assert-Same (Field $sourceConfig.data 'action_count') (Field $sourceSpec 'expected_actions') 'source action count'
Assert-Same (Field $sourceConfig.data 'rom_sha256') (Field $sourceSpec 'rom_sha256') 'source ROM hash'
Assert-Same (Field $provenance 'source_rom_sha256') $sourceSpec.rom_sha256 'source provenance ROM hash'
foreach ($key in @('algorithm', 'language', 'torch_version', 'network', 'preprocessing')) {
    Assert-Same (Field $sourceConfig.data $key) (Field $tc $key) "source compatible $key"
}
$pretraining = Integer (Field $provenance 'source_pretraining_decisions') 'source pretraining decisions'
Assert-That ($pretraining -gt 0) 'Source needs a positive pretraining decision count.'
# Bind the native-recorded source step count to the preserved, independently
# evaluated Pong proof. PowerShell deliberately does not deserialize LibTorch.
$proofs = Snapshot (Join-Path $RunsPath 'proofs.json') -Json
$proof = Field $proofs.data 'pong'
Assert-Same (Field $proof 'passed') $true 'Pong source proof'
Assert-Same (Safe-Path (Field $proof 'checkpoint')) $sourceCheckpoint.path 'proven source checkpoint path'
Assert-Same (Field $proof 'checkpoint_sha256') $sourceCheckpoint.sha256 'proven source checkpoint hash'
Assert-Same (Field $proof 'trained_decisions') $pretraining 'proven source pretraining decisions'
$sourceEvaluation = Snapshot (Field $proof 'evaluation') -Json
Assert-Same (Field $sourceEvaluation.data 'checkpoint_sha256') $sourceCheckpoint.sha256 'source evaluation checkpoint hash'
Assert-Same (Field $sourceEvaluation.data 'passed') $true 'source evaluation pass'
Assert-Same (Field $sourceEvaluation.data 'episodes_completed') 20 'source proof full evaluation count'
Assert-Same (Field $sourceEvaluation.data 'truncated_games') 0 'source proof truncations'
Assert-Same (Field $sourceEvaluation.data 'status') 'complete' 'source proof terminal evaluation'
Assert-Same (Safe-Path (Field $sourceEvaluation.data 'checkpoint')) $sourceCheckpoint.path 'source evaluation checkpoint path'
$sourceEpisodes = @(Field $sourceEvaluation.data 'episodes')
Assert-That ($sourceEpisodes.Count -eq 20) 'Source proof needs 20 episode records.'
$sourceSum = 0.0; $sourceWins = 0; $sourceSeed = Integer (Field $sourceEvaluation.data 'seed_start') 'source proof seed'
for ($i = 0; $i -lt 20; $i++) {
    $row = $sourceEpisodes[$i]
    Assert-Same (Field $row 'episode') ($i + 1) 'source episode order'
    Assert-Same (Field $row 'seed') ($sourceSeed + $i) 'source episode seed'
    Assert-Same (Field $row 'terminated') $true 'source full natural game termination'
    Assert-Same (Field $row 'truncated') $false 'source episode truncation'
    $score = Number (Field $row 'raw_return') 'source episode return'
    Assert-Same (Field $row 'won') ($score -gt 0) 'source episode win'
    $sourceSum += $score; if ($score -gt 0) { $sourceWins++ }
}
Assert-That ($sourceWins -ge 16 -and $sourceSum -gt 0) 'Source episodes do not meet the Pong proof criterion.'
Assert-Same (Field $sourceEvaluation.data 'wins') $sourceWins 'source evaluation wins'
Assert-Near (Field $sourceEvaluation.data 'mean_return') ($sourceSum / 20) 'source evaluation mean'
Assert-Same (Field $proof 'wins') $sourceWins 'proof registry wins'
Assert-Near (Field $proof 'mean_return') ($sourceSum / 20) 'proof registry mean'

$pairs = @(); $deltaSum = 0.0; $higher = 0; $equal = 0; $lower = 0; $bothNatural = 0
for ($i = 0; $i -lt 20; $i++) {
    $a = $scratch.returns[$i]; $b = $transfer.returns[$i]
    Assert-Same $a.seed $b.seed 'paired evaluation seed'
    $delta = $b.raw_return - $a.raw_return; $deltaSum += $delta
    $natural = !$a.truncated -and !$b.truncated
    if ($natural) { $bothNatural++ }
    if ($delta -gt 0) { $higher++ } elseif ($delta -lt 0) { $lower++ } else { $equal++ }
    $pairs += [ordered]@{ seed = $a.seed; scratch_raw_return = $a.raw_return;
        transfer_raw_return = $b.raw_return; transfer_minus_scratch = $delta;
        scratch_truncated = $a.truncated; transfer_truncated = $b.truncated;
        scratch_raw_frames = $a.raw_frames; transfer_raw_frames = $b.raw_frames; both_natural_games = $natural }
}
$fullGames = $scratch.truncated_games -eq 0 -and $transfer.truncated_games -eq 0
function Run-Report($Run) {
    return [ordered]@{ run_directory = $Run.directory; config = (Artifact $Run.config);
        progress = (Artifact $Run.progress); evaluation = (Artifact $Run.evaluation); final_checkpoint = (Artifact $Run.checkpoint);
        configuration = $Run.config.data; target_agent_decisions = 1000000;
        target_training_seconds = $Run.training_seconds; mean_raw_return = $Run.mean;
        positive_games = $Run.positive_games; truncated_games = $Run.truncated_games;
        natural_games = 20 - $Run.truncated_games; milestone_passed = $Run.passed }
}
$report = [ordered]@{
    schema_version = 1; created_utc = [DateTime]::UtcNow.ToString('o'); game_id = $sc.game_id
    comparison = 'matched_feature_transfer_pilot'; comparable = $true
    bounded_episode_comparison = $true; full_game_comparison = $fullGames
    comparability = [ordered]@{ matched_config_fields = $matchedKeys; identical_fresh_heads = $true;
        fresh_adam = $true; target_agent_decisions_each = 1000000; same_training_seed = $sc.seed;
        paired_evaluation_seed_start = $sc.evaluation_seed; paired_finalized_episodes = 20;
        raw_frame_cap_per_episode = $sc.preprocessing.max_episode_raw_frames;
        scratch_truncated_games = $scratch.truncated_games; transfer_truncated_games = $transfer.truncated_games;
        paired_both_natural_game_count = $bothNatural;
        feature_tensors_copied_exactly = 8; returns_recomputed_from_episodes = $true;
        checkpoint_integrity = 'SHA256 matches native evaluation/transfer provenance; no host LibTorch deserialization' }
    scratch = (Run-Report $scratch); transfer = (Run-Report $transfer)
    source_pretraining = [ordered]@{ game_id = 'pong'; agent_decisions = $pretraining;
        checkpoint = (Artifact $sourceCheckpoint); config = (Artifact $sourceConfig);
        proof_registry = (Artifact $proofs); evaluation = (Artifact $sourceEvaluation);
        charged_to_target_budget = $false; total_source_plus_transfer_target_decisions = $pretraining + 1000000;
        training_seconds = $null; training_time_note = 'Source checkpoint spans earlier runs; no verified source wall-time total is inferred.' }
    result = [ordered]@{ scratch_mean_raw_return = $scratch.mean; transfer_mean_raw_return = $transfer.mean;
        paired_mean_transfer_minus_scratch = $deltaSum / 20; higher_score_pairs = $higher;
        paired_mean_capped_episode_return_delta = $deltaSum / 20;
        score_delta_scope = $(if ($fullGames) { '20 naturally completed games under the identical frame-capped protocol' }
            else { '20 finalized episodes under the identical frame cap; includes truncated episode returns' });
        equal_score_pairs = $equal; lower_score_pairs = $lower;
        target_training_seconds_transfer_minus_scratch = $transfer.training_seconds - $scratch.training_seconds;
        paired_returns = $pairs }
    limitations = @('One training seed per game: this is a pilot, not a robust claim of transfer advantage.',
        'Paired evaluation episodes do not constitute independent training replications.',
        'Source pretraining decisions are additional cost; equal target budgets do not mean equal total compute.',
        'Elapsed training time includes host scheduling and concurrent viewer/evaluation load.',
        'Any truncated episode prevents a full-game superiority claim; capped-episode deltas do not establish full-game performance.',
        'Both-natural pair counts are descriptive; conditioning on termination can select different policy outcomes.',
        'Milestones are engineering thresholds, not human baselines or claims of literal game completion.')
}
if (!$Output) {
    $Output = Join-Path $RunsPath ('comparisons/{0}-{1}-{2}.json' -f $sc.game_id,
        [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'), [Guid]::NewGuid().ToString('N').Substring(0, 8))
}
$destination = Safe-Path $Output -MayNotExist
Assert-That (!(Test-Path -LiteralPath $destination)) "Comparison output already exists; preserving it: $destination"
foreach ($snapshot in $Snapshots) {
    $null = Safe-Path $snapshot.path
    Assert-That ((Hash $snapshot.path) -ceq $snapshot.sha256) "Evidence changed before report creation: $($snapshot.path)"
}
$parent = [IO.Path]::GetDirectoryName($destination)
[IO.Directory]::CreateDirectory($parent) | Out-Null
$null = Safe-Path $destination -MayNotExist
$bytes = [Text.UTF8Encoding]::new($false).GetBytes(($report | ConvertTo-Json -Depth 60) + "`n")
$file = [IO.File]::Open($destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try { $file.Write($bytes, 0, $bytes.Length) } finally { $file.Dispose() }
[ordered]@{ report = $destination; sha256 = (Hash $destination); game_id = $sc.game_id;
    scratch_mean = $scratch.mean; transfer_mean = $transfer.mean; paired_mean_delta = $deltaSum / 20 } | ConvertTo-Json
