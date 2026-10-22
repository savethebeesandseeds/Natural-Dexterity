$ErrorActionPreference = 'Stop'
$taskProject = 'C:\Work\Universalis-Materiae-Dexteritas\projects\droid-blocks'
$taskBatch = Join-Path $taskProject 'artifacts\reports\light-headroom-v1-development-20261004'
$taskWeb = Join-Path $taskProject 'web'
$taskIndexPath = Join-Path $taskWeb 'headroom-data.json'
$taskIndex = Get-Content -Raw -LiteralPath $taskIndexPath | ConvertFrom-Json
$taskReceiptPath = Join-Path $taskBatch 'verification\export-check.json'
if (Test-Path -LiteralPath $taskReceiptPath) { throw 'Export receipt already exists.' }
$taskProblems = [System.Collections.Generic.List[string]]::new()
if ($taskIndex.schema -ne 'light_headroom_view_v1' -or $taskIndex.cases.Count -ne 24 -or $taskIndex.controllers.Count -ne 9 -or $taskIndex.trials.Count -ne 216) { $taskProblems.Add('Export matrix differs.') }
$taskReportHash = (Get-FileHash -LiteralPath (Join-Path $taskBatch 'report.json')).Hash.ToLowerInvariant()
$taskAuditHash = (Get-FileHash -LiteralPath (Join-Path $taskBatch 'verification\independent-audit.json')).Hash.ToLowerInvariant()
if ($taskReportHash -ne $taskIndex.provenance.report_sha256 -or $taskAuditHash -ne $taskIndex.provenance.audit_sha256) { $taskProblems.Add('Report/audit provenance differs.') }
$taskFrameCount = 0
$taskSafeComplete = 0
foreach ($taskTrial in $taskIndex.trials) {
    $taskPlaybackPath = Join-Path $taskWeb $taskTrial.trace_url.TrimStart('/')
    $taskPlaybackHash = (Get-FileHash -LiteralPath $taskPlaybackPath).Hash.ToLowerInvariant()
    $taskExpectedHash = ($taskIndex.provenance.exported_playbacks | Where-Object trial_id -eq $taskTrial.id).sha256
    if ($taskPlaybackHash -ne $taskExpectedHash) { $taskProblems.Add('Playback hash differs: ' + $taskTrial.id) }
    $taskPlayback = Get-Content -Raw -LiteralPath $taskPlaybackPath | ConvertFrom-Json
    $taskSnapshots = Get-Content -Raw -LiteralPath (Join-Path $taskBatch $taskTrial.source_snapshots.path) | ConvertFrom-Json
    if ($taskPlayback.schema -ne 'light_headroom_playback_v1' -or $taskPlayback.id -ne $taskTrial.id -or $taskPlayback.frames.Count -ne $taskSnapshots.Count) { $taskProblems.Add('Playback identity/count differs: ' + $taskTrial.id) }
    $taskFrameCount += $taskPlayback.frames.Count
    foreach ($taskPosition in @(0,($taskSnapshots.Count - 1))) {
        $taskFrame = $taskPlayback.frames[$taskPosition]
        $taskSnapshot = $taskSnapshots[$taskPosition]
        if ([math]::Abs($taskFrame.time_s - ($taskSnapshot.step * 0.02)) -gt 1e-9 -or $taskFrame.cumulative_reward -ne $taskSnapshot.physics.reward.cumulative) { $taskProblems.Add('Playback endpoint clock/reward differs: ' + $taskTrial.id) }
        $taskActualPhysics = $taskFrame.physics | ConvertTo-Json -Depth 50 -Compress
        $taskSourcePhysics = $taskSnapshot.physics | ConvertTo-Json -Depth 50 -Compress
        if ($taskActualPhysics -cne $taskSourcePhysics) { $taskProblems.Add('Playback endpoint physics differs: ' + $taskTrial.id) }
    }
    if ($taskTrial.metrics.complete -and $taskTrial.metrics.safe) {
        $taskSafeComplete++
        if ($taskPlayback.frames.Count -ne 321 -or $taskPlayback.frames[-1].time_s -ne 64 -or [math]::Abs($taskPlayback.frames[-1].cumulative_reward - $taskTrial.metrics.native_reward) -gt 1e-9) { $taskProblems.Add('Complete playback endpoint differs: ' + $taskTrial.id) }
    } elseif ($taskPlayback.frames[-1].time_s -gt $taskTrial.metrics.duration_s -or ($taskTrial.metrics.duration_s - $taskPlayback.frames[-1].time_s) -ge 0.2) { $taskProblems.Add('Stopped playback exposure differs: ' + $taskTrial.id) }
}
$taskValue = [ordered]@{
    schema='light_headroom_export_check_v1'
    index_sha256=(Get-FileHash -LiteralPath $taskIndexPath).Hash.ToLowerInvariant()
    report_sha256=$taskReportHash
    audit_sha256=$taskAuditHash
    cases_checked=$taskIndex.cases.Count
    controllers_checked=$taskIndex.controllers.Count
    playback_files_checked=$taskIndex.trials.Count
    total_saved_frames=$taskFrameCount
    complete_safe_playbacks_checked=$taskSafeComplete
    scope='Read-only file inspection: published hashes, all frame counts, first/last physical snapshot equality, complete/stopped exposure and native endpoint reward. Exporter separately validates every snapshot clock and scene before copying.'
    passed=($taskProblems.Count -eq 0)
    issues=$taskProblems.ToArray()
}
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskBatch 'verification\check-export.ps1')
[System.IO.File]::WriteAllText($taskReceiptPath, ($taskValue | ConvertTo-Json -Depth 6) + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
$taskValue | ConvertTo-Json -Depth 6
if ($taskProblems.Count -ne 0) { throw 'Export checks failed; retain the receipt.' }
