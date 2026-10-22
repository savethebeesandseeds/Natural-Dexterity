#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$SuiteRun,
    [Parameter(Mandatory)][string]$BreakoutScratchRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectPath = [IO.Path]::GetFullPath($PSScriptRoot)
$RunsPath = Join-Path $ProjectPath 'runs'
$RunsPrefix = $RunsPath + [IO.Path]::DirectorySeparatorChar
$Evidence = [Collections.Generic.List[hashtable]]::new()
$Diagnostics = [Collections.Generic.List[hashtable]]::new()
function Require([bool]$Value, [string]$Message) { if (!$Value) { throw $Message } }
function Field($Object, [string]$Key) {
    Require ($Object -is [Collections.IDictionary] -and $Object.Contains($Key)) "Missing evidence field: $Key"
    return $Object[$Key]
}
function Safe-Path([string]$Path, [switch]$MayNotExist) {
    Require (![string]::IsNullOrWhiteSpace($Path)) 'Empty evidence path.'
    $candidate = if ($Path.StartsWith('/workspace/')) { Join-Path $ProjectPath $Path.Substring(11) }
        elseif ([IO.Path]::IsPathRooted($Path)) { $Path }
        elseif ($Path.Replace('\', '/').StartsWith('runs/')) { Join-Path $ProjectPath $Path }
        else { Join-Path $RunsPath $Path }
    $full = [IO.Path]::GetFullPath($candidate)
    Require ($full.StartsWith($RunsPrefix, [StringComparison]::OrdinalIgnoreCase)) "Path must stay within project runs: $Path"
    $part = $full
    while ($part.Length -ge $RunsPath.Length) {
        if (Test-Path -LiteralPath $part) {
            Require (((Get-Item -Force -LiteralPath $part).Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) "Reparse point is not an evidence path: $part"
        }
        if ($part -ieq $RunsPath) { break }
        $part = [IO.Path]::GetDirectoryName($part)
    }
    if (!$MayNotExist) { Require (Test-Path -LiteralPath $full) "Missing evidence: $full" }
    return $full
}
function Container-Path([string]$Path) { return '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $Path).Replace('\', '/') }
function Hash([string]$Path) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }
function Read-Evidence([string]$Path, [switch]$Binary, [switch]$Changing) {
    $path = Safe-Path $Path
    $hash = Hash $path
    $snapshot = @{ path = $path; sha256 = $hash }
    if (!$Binary) {
        $snapshot['data'] = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json -AsHashtable
        Require ($snapshot.data -is [Collections.IDictionary]) "Expected JSON object: $path"
    }
    if ((Hash $path) -cne $hash) {
        if ($Changing) { return $null }
        throw "Evidence changed while reading: $path"
    }
    $Evidence.Add(@{ path = $path; sha256 = $hash })
    return $snapshot
}
function Number($Value, [string]$Label) {
    Require (($Value -is [int] -or $Value -is [long] -or $Value -is [double] -or $Value -is [decimal]) -and
        [double]::IsFinite([double]$Value)) "Invalid finite number: $Label"
    return [double]$Value
}
function Equal($Left, $Right, [string]$Label) {
    if ($null -eq $Left -or $null -eq $Right) { Require ($null -eq $Left -and $null -eq $Right) "Mismatch: $Label"; return }
    if ($Left -is [Collections.IDictionary]) {
        Require ($Right -is [Collections.IDictionary] -and $Left.Count -eq $Right.Count) "Mismatch: $Label"
        foreach ($key in $Left.Keys) { Equal $Left[$key] (Field $Right $key) "$Label.$key" }
    } elseif ($Left -is [array]) {
        Require ($Right -is [array] -and $Left.Count -eq $Right.Count) "Mismatch: $Label"
        for ($i = 0; $i -lt $Left.Count; $i++) { Equal $Left[$i] $Right[$i] "$Label[$i]" }
    } elseif ($Left -is [string] -or $Left -is [bool]) {
        Require ($Left.GetType() -eq $Right.GetType() -and $Left -ceq $Right) "Mismatch: $Label"
    } else { Require ((Number $Left $Label) -eq (Number $Right $Label)) "Mismatch: $Label" }
}
function Near($Value, [double]$Expected, [string]$Label) {
    Require ([Math]::Abs((Number $Value $Label) - $Expected) -le 1e-9 * [Math]::Max(1, [Math]::Abs($Expected))) "Episode/summary mismatch: $Label"
}
function Validate-Run([string]$RunPath, [string]$Game, [string]$Mode) {
    $directory = Safe-Path $RunPath -MayNotExist
    $progressPath = Join-Path $directory 'progress.json'
    if (!(Test-Path -LiteralPath $progressPath -PathType Leaf)) {
        $Diagnostics.Add(@{ run_dir = (Container-Path $directory); status = 'not_started'; published = $false }); return $null
    }
    $progress = Read-Evidence $progressPath -Changing
    if ($null -eq $progress) {
        $Diagnostics.Add(@{ run_dir = (Container-Path $directory); status = 'progress_changing'; published = $false }); return $null
    }
    $p = $progress.data
    if ((Field $p 'status') -cne 'complete') {
        # Active progress can change after this read and is not publication evidence.
        $Evidence.RemoveAt($Evidence.Count - 1)
        $Diagnostics.Add(@{ run_dir = (Container-Path $directory); status = $p.status; published = $false }); return $null
    }
    $config = Read-Evidence (Join-Path $directory 'config.json')
    $evaluation = Read-Evidence (Join-Path $directory 'evaluation.json')
    $checkpoint = Read-Evidence (Join-Path $directory 'final.pt') -Binary
    $c = $config.data; $e = $evaluation.data
    Equal (Field $e 'status') 'complete' 'evaluation status'
    foreach ($key in @('game_id', 'rom_sha256', 'game_spec', 'preprocessing')) { Equal (Field $e $key) (Field $c $key) "evaluation $key" }
    Equal (Field $c 'game_id') $Game 'expected game'
    Equal (Field $p 'game_id') $Game 'progress game'
    Equal (Field $c 'initialization_mode') $Mode 'expected initialization'
    Equal (Field $c 'previous_trained_steps') 0 'no continuation'
    Equal (Field $p 'previous_trained_steps') 0 'progress no continuation'
    Equal (Field $c 'parent_checkpoint') $null 'no parent checkpoint'
    Equal (Field $c 'initial_adam_state_entries') 0 'fresh Adam'
    Equal (Field $c 'seed') $(if ($Game -ceq 'breakout') { 21 } else { 31 }) 'pilot seed'
    Equal (Field $c 'environments') 8 'pilot environments'
    Equal (Field $c 'max_agent_decisions') 1000000 'pilot budget'
    foreach ($key in @('agent_decisions', 'steps', 'target_steps', 'cumulative_agent_decisions', 'checkpoint_steps', 'checkpoint_run_steps')) {
        Equal (Field $p $key) 1000000 "actual $key"
    }
    Equal (Field $p 'stop_reason') 'target_steps' 'training termination'
    Equal (Safe-Path (Field $p 'run_dir')) $directory 'progress run directory'
    Equal (Safe-Path (Field $e 'checkpoint')) $checkpoint.path 'evaluated final checkpoint'
    Equal (Field $e 'checkpoint_sha256') $checkpoint.sha256 'final checkpoint SHA256'
    Equal (Safe-Path (Field $e 'checkpoint_config')) $config.path 'evaluated config'
    Equal (Field $e 'checkpoint_config_sha256') $config.sha256 'config SHA256'
    Equal (Field $e 'policy') 'deterministic_ppo' 'evaluation policy'
    Equal (Field $e 'episodes_requested') 20 'requested episodes'
    Equal (Field $e 'episodes_completed') 20 'completed episodes'
    Equal (Field $c 'evaluation_episodes') 20 'configured evaluation episodes'
    $spec = Field $c 'game_spec'; $criterion = Field $spec 'criterion'
    Equal (Field $spec 'id') $Game 'registered game'
    Equal (Field $spec 'rom_sha256') $c.rom_sha256 'registered ROM hash'
    Equal (Field $spec 'expected_actions') (Field $c 'action_count') 'registered actions'
    $seed = if ($Game -ceq 'breakout') { 1100100 } else { 1200100 }
    Equal (Field $e 'seed_start') $seed 'pilot evaluation seed'
    Equal (Field $c 'evaluation_seed') $seed 'configured evaluation seed'
    Equal (Field $spec 'final_eval_seed') $seed 'registered evaluation seed'
    Equal (Field $criterion 'episodes') 20 'criterion episodes'
    Equal (Field $criterion 'max_truncated_episodes') 0 'criterion forbids truncations'
    Equal (Field $criterion 'mean_raw_return_threshold') $(if ($Game -ceq 'breakout') { 100 } else { 1000 }) 'declared milestone'
    Equal (Field $criterion 'mean_strict') $false 'declared inclusive threshold'
    Equal (Field $criterion 'min_positive_games') 0 'declared minimum positive games'
    $cap = Number (Field $c.preprocessing 'max_episode_raw_frames') 'frame cap'
    Equal $cap 108000 'declared frame cap'
    $rows = @(Field $e 'episodes'); Require ($rows.Count -eq 20) 'Exactly 20 episode records are required.'
    $sum = 0.0; $truncated = 0; $positive = 0
    for ($i = 0; $i -lt 20; $i++) {
        $row = $rows[$i]
        Equal (Field $row 'episode') ($i + 1) 'episode index'
        Equal (Field $row 'seed') ($seed + $i) 'episode seed'
        $done = Field $row 'terminated'; $capped = Field $row 'truncated'
        Require ($done -is [bool] -and $capped -is [bool] -and ($done -or $capped)) 'Episode must end naturally or at the frame cap.'
        $frames = Number (Field $row 'raw_frames') 'episode frames'
        $decisions = Number (Field $row 'agent_decisions') 'episode decisions'
        Require ($frames -gt 0 -and $frames -le $cap -and $frames -eq [Math]::Truncate($frames) -and
            $decisions -gt 0 -and $decisions -eq [Math]::Truncate($decisions)) 'Invalid episode frame/decision count.'
        if ($capped) { Equal $frames $cap 'truncated frame count'; $truncated++ }
        $score = Number (Field $row 'raw_return') 'raw return'; $sum += $score
        $naturalPositive = $done -and !$capped -and $score -gt 0
        Equal (Field $row 'positive_game') $naturalPositive 'positive natural game'
        Equal (Field $row 'won') $null 'non-Pong wins are undefined'
        if ($naturalPositive) { $positive++ }
    }
    $mean = $sum / 20; $passed = $truncated -eq 0 -and $mean -ge $criterion.mean_raw_return_threshold
    Near (Field $e 'mean_return') $mean 'mean return'
    Equal (Field $e 'truncated_games') $truncated 'truncated summary'
    Equal (Field $e 'positive_games') $positive 'positive summary'
    Equal (Field $e 'wins') $null 'non-Pong win summary'
    Equal (Field $e 'passed') $passed 'recomputed milestone'
    Equal (Field $e 'verdict') $(if ($passed) { 'target_met' } else { 'target_not_met' }) 'milestone verdict'
    $primary = Field $e 'criterion_evidence'
    Equal (Field $primary 'episodes_required') 20 'criterion required episodes'
    Equal (Field $primary 'episodes_completed') 20 'criterion completed episodes'
    Equal (Field $primary 'truncated_games') $truncated 'criterion truncated summary'
    Equal (Field $primary 'positive_games') $positive 'criterion positive summary'
    Near (Field $primary 'mean_return') $mean 'criterion mean'
    if ($p.Contains('evaluation')) { Equal $p.evaluation $e 'embedded progress evaluation' }
    $seconds = Number (Field $p 'training_elapsed_seconds') 'training seconds'
    Require ($seconds -gt 0) 'Positive training duration required.'
    if ($Mode -ceq 'cold_start') { Equal (Field $c 'transfer') $null 'scratch transfer metadata' }
    else {
        $source = Field $c 'transfer'
        Equal (Field $source 'source_checkpoint_sha256') $Manifest.data.source_checkpoint_sha256 'suite source hash'
        Equal (Safe-Path (Field $source 'source_checkpoint')) $SourceCheckpoint.path 'suite source path'
        Equal (Field $source 'source_config_sha256') $SourceConfig.sha256 'suite source config hash'
        Equal (Field $source 'target_step_count_at_initialization') 0 'fresh target decisions'
        Equal (Field $source 'optimizer_restored') $false 'fresh transfer Adam'
    }
    return @{ directory = $directory; config = $config; evaluation = $evaluation; progress = $progress; checkpoint = $checkpoint;
        row = [ordered]@{ run_dir = (Container-Path $directory); game_id = $Game; game_title = $c.game_title;
            initialization = $(if ($Mode -ceq 'cold_start') { 'Scratch' } else { 'Pong features / fresh heads' });
            initialization_mode = $Mode; training_seed = $c.seed; target_decisions = 1000000;
            mean_return = $mean; episodes_completed = 20; truncated_games = $truncated; passed = $passed;
            training_seconds = $seconds; evaluation_seed = $seed; checkpoint_sha256 = $checkpoint.sha256;
            config_sha256 = $config.sha256; progress_sha256 = $progress.sha256; evaluation_sha256 = $evaluation.sha256;
            evaluation_path = (Container-Path $evaluation.path); checkpoint_path = (Container-Path $checkpoint.path) } }
}
function Atomic-Json([string]$Path, $Value) {
    $path = Safe-Path $Path -MayNotExist
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path)) | Out-Null
    $temporary = Safe-Path ($path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp') -MayNotExist
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 60) + "`n")
    $file = [IO.File]::Open($temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $file.Write($bytes, 0, $bytes.Length); $file.Flush($true) } finally { $file.Dispose() }
    [IO.File]::Move($temporary, $path, $true)
}

$lockPath = Safe-Path (Join-Path $RunsPath '.publish-results.lock') -MayNotExist
$lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $suite = Safe-Path $SuiteRun
    $Manifest = Read-Evidence (Join-Path $suite 'manifest.json')
    Equal (Field $Manifest.data 'schema_version') 1 'suite schema'
    $SourceCheckpoint = Read-Evidence (Field $Manifest.data 'source_checkpoint') -Binary
    $SourceConfig = Read-Evidence (Field $Manifest.data 'source_config')
    $sourceEvaluation = Read-Evidence (Field $Manifest.data 'source_evaluation')
    Equal $SourceCheckpoint.sha256 (Field $Manifest.data 'source_checkpoint_sha256') 'suite source checkpoint hash'
    Equal $SourceConfig.sha256 (Field $Manifest.data 'source_config_sha256') 'suite source config hash'
    Equal $sourceEvaluation.sha256 (Field $Manifest.data 'source_evaluation_sha256') 'suite source evaluation hash'
    $known = @(
        @{ path = $BreakoutScratchRun; game = 'breakout'; mode = 'cold_start'; id = 'breakout_scratch' },
        @{ path = (Field $Manifest.data 'prerequisite_run'); game = 'space_invaders'; mode = 'cold_start'; id = 'space_invaders_scratch' }
    )
    foreach ($game in @('breakout', 'space_invaders')) {
        $phase = @($Manifest.data.phases | Where-Object { $_.id -ceq ($game + '_transfer') })
        Require ($phase.Count -eq 1) "Suite needs one $game transfer phase."
        Equal (Field $phase[0] 'game') $game 'suite phase game'
        Equal (Field $phase[0] 'steps') 1000000 'suite phase budget'
        Equal (Field $phase[0] 'seed') $(if ($game -ceq 'breakout') { 21 } else { 31 }) 'suite phase seed'
        Equal (Field $phase[0] 'environments') 8 'suite phase workers'
        $known += @{ path = (Field $phase[0] 'run_dir'); game = $game;
            mode = 'feature_transfer_fresh_heads_and_adam'; id = $game + '_transfer' }
    }
    $completed = [ordered]@{}
    foreach ($candidate in $known) {
        $run = Validate-Run $candidate.path $candidate.game $candidate.mode
        if ($null -ne $run) { $completed[$candidate.id] = $run }
    }
    $experimentPath = Safe-Path (Join-Path $RunsPath 'experiments.json') -MayNotExist
    $experiments = if (Test-Path -LiteralPath $experimentPath) { (Read-Evidence $experimentPath).data } else { @{ schema_version = 1; runs = @() } }
    Equal (Field $experiments 'schema_version') 1 'experiment index schema'
    $rows = [Collections.Generic.List[object]]::new()
    foreach ($row in @(Field $experiments 'runs')) { $rows.Add($row) }
    foreach ($run in $completed.Values) {
        $matches = @(); for ($i = 0; $i -lt $rows.Count; $i++) {
            if ((Safe-Path (Field $rows[$i] 'run_dir') -MayNotExist) -ieq $run.directory) { $matches += $i }
        }
        Require ($matches.Count -le 1) 'Duplicate existing experiment rows are preserved; resolve the ambiguity before publication.'
        if ($matches.Count) { $rows[$matches[0]] = $run.row } else { $rows.Add($run.row) }
    }
    $indexPath = Safe-Path (Join-Path $RunsPath 'comparisons/index.json') -MayNotExist
    $index = if (Test-Path -LiteralPath $indexPath) { (Read-Evidence $indexPath).data } else { @{ schema_version = 1; reports = @() } }
    Equal (Field $index 'schema_version') 1 'comparison index schema'
    $reports = [Collections.Generic.List[object]]::new()
    foreach ($row in @(Field $index 'reports')) { $reports.Add($row) }
    foreach ($game in @('breakout', 'space_invaders')) {
        if (!$completed.Contains($game + '_scratch') -or !$completed.Contains($game + '_transfer')) { continue }
        $scratch = $completed[$game + '_scratch']; $transfer = $completed[$game + '_transfer']
        $matchingReport = $null; $alreadyIndexed = $false
        $candidates = [Collections.Generic.List[hashtable]]::new()
        $candidatePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($entry in $reports) {
            if ($entry.game_id -cne $game) { continue }
            $saved = Read-Evidence (Field $entry 'report_path')
            Equal $saved.sha256 (Field $entry 'report_sha256') 'indexed immutable comparison hash'
            $null = $candidatePaths.Add($saved.path)
            $candidates.Add(@{ snapshot = $saved; indexed = $true; entry = $entry })
        }
        $comparisonDirectory = Safe-Path (Join-Path $RunsPath 'comparisons') -MayNotExist
        if (Test-Path -LiteralPath $comparisonDirectory -PathType Container) {
            foreach ($file in Get-ChildItem -LiteralPath $comparisonDirectory -Filter '*.json' -File) {
                if ($file.FullName -ieq $indexPath -or !$candidatePaths.Add($file.FullName)) { continue }
                $saved = Read-Evidence $file.FullName
                if ($saved.data.Contains('comparison') -and $saved.data.comparison -ceq 'matched_feature_transfer_pilot') {
                    $candidates.Add(@{ snapshot = $saved; indexed = $false })
                }
            }
        }
        foreach ($candidate in $candidates) {
            $saved = $candidate.snapshot
            $r = $saved.data
            if ($r.game_id -ceq $game -and $r.scratch.final_checkpoint.sha256 -ceq $scratch.checkpoint.sha256 -and
                $r.transfer.final_checkpoint.sha256 -ceq $transfer.checkpoint.sha256 -and
                $r.scratch.config.sha256 -ceq $scratch.config.sha256 -and $r.transfer.config.sha256 -ceq $transfer.config.sha256 -and
                $r.scratch.progress.sha256 -ceq $scratch.progress.sha256 -and $r.transfer.progress.sha256 -ceq $transfer.progress.sha256 -and
                $r.scratch.evaluation.sha256 -ceq $scratch.evaluation.sha256 -and $r.transfer.evaluation.sha256 -ceq $transfer.evaluation.sha256 -and
                (Safe-Path $r.scratch.run_directory) -ieq $scratch.directory -and (Safe-Path $r.transfer.run_directory) -ieq $transfer.directory) {
                Equal $r.comparable $true 'existing comparison comparability'
                Equal $r.result.scratch_mean_raw_return $scratch.row.mean_return 'existing scratch comparison mean'
                Equal $r.result.transfer_mean_raw_return $transfer.row.mean_return 'existing transfer comparison mean'
                Near $r.result.paired_mean_transfer_minus_scratch ($transfer.row.mean_return - $scratch.row.mean_return) 'existing comparison delta'
                Equal $r.full_game_comparison ($scratch.row.truncated_games -eq 0 -and $transfer.row.truncated_games -eq 0) 'existing comparison full-game flag'
                $matchingReport = $saved; $alreadyIndexed = $candidate.indexed; break
            }
        }
        if ($alreadyIndexed) { continue }
        if ($null -eq $matchingReport) {
            $resultText = & (Join-Path $ProjectPath 'compare-experiments.ps1') -ScratchRun $scratch.directory -TransferRun $transfer.directory
            $result = ($resultText -join "`n") | ConvertFrom-Json -AsHashtable
            $saved = Read-Evidence $result.report
            Equal $saved.sha256 $result.sha256 'new comparison report hash'
        } else { $saved = $matchingReport }
        $r = $saved.data
        $reports.Add([ordered]@{ game_id = $game; game_title = $scratch.row.game_title;
            scratch_run = $scratch.row.run_dir; transfer_run = $transfer.row.run_dir;
            scratch_mean = $r.result.scratch_mean_raw_return; transfer_mean = $r.result.transfer_mean_raw_return;
            delta = $r.result.paired_mean_transfer_minus_scratch; full_game_comparison = $r.full_game_comparison;
            scratch_truncated = $scratch.row.truncated_games; transfer_truncated = $transfer.row.truncated_games;
            source_pretraining_decisions = $r.source_pretraining.agent_decisions; target_decisions = 1000000;
            report_path = (Container-Path $saved.path); report_sha256 = $saved.sha256 })
    }
    foreach ($item in $Evidence) { Require ((Hash $item.path) -ceq $item.sha256) "Evidence changed before publication: $($item.path)" }
    $experiments.runs = $rows.ToArray(); $index.reports = $reports.ToArray()
    Atomic-Json $experimentPath $experiments
    Atomic-Json $indexPath $index
    [ordered]@{ experiments_path = $experimentPath; experiments_sha256 = (Hash $experimentPath);
        published_completed_runs = @($completed.Values | ForEach-Object { $_.row });
        comparisons_index = $indexPath; comparison_count = $reports.Count; skipped = $Diagnostics.ToArray() } | ConvertTo-Json -Depth 20
} finally { $lock.Dispose() }
