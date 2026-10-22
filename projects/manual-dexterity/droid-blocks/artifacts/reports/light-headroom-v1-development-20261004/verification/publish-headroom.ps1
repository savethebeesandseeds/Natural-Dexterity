# Host filesystem integration only. Simulation, compilation and auditing belong
# in the managed Linux container. Recover a fully staged native export when the
# Docker Desktop shared mount rejects Linux renameat2(RENAME_NOREPLACE).
$ErrorActionPreference = 'Stop'
$taskProject = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskWeb = Join-Path $taskProject 'web'
$taskBatch = Join-Path $taskProject 'artifacts\reports\light-headroom-v1-development-20261004'
$taskStage = [System.IO.Path]::GetFullPath((Join-Path $taskWeb 'headroom-data.next.json'))
$taskIndex = [System.IO.Path]::GetFullPath((Join-Path $taskWeb 'headroom-data.json'))
if ([System.IO.Path]::GetDirectoryName($taskStage) -cne [System.IO.Path]::GetDirectoryName($taskIndex)) { throw 'Publication must remain in the same web directory.' }
if (!(Test-Path -LiteralPath $taskStage -PathType Leaf) -or (Test-Path -LiteralPath $taskIndex)) { throw 'Require a staged index and no published destination.' }
$taskView = Get-Content -Raw -LiteralPath $taskStage | ConvertFrom-Json
if ($taskView.schema -ne 'light_headroom_view_v1' -or $taskView.cases.Count -ne 24 -or $taskView.controllers.Count -ne 9 -or $taskView.trials.Count -ne 216 -or !$taskView.summary.integrity_passed) { throw 'Staged index has not passed full native export validation.' }
$taskVerification = Join-Path $taskBatch 'verification'
$taskReplay = Get-Content -Raw -LiteralPath (Join-Path $taskVerification 'native-verify.log') | ConvertFrom-Json
if (!$taskReplay.verified -or !$taskReplay.complete -or $taskReplay.retained_runs -ne 216) { throw 'Completed native verification receipt required.' }
foreach ($taskEvidence in @(@{path='report.json';sha=$taskView.provenance.report_sha256},@{path='verification\independent-audit.json';sha=$taskView.provenance.audit_sha256})) {
    if ((Get-FileHash -LiteralPath (Join-Path $taskBatch $taskEvidence.path)).Hash.ToLowerInvariant() -ne $taskEvidence.sha) { throw 'Staged evidence provenance differs.' }
}
if ($taskView.provenance.exported_playbacks.Count -ne 216) { throw 'Require all playback fingerprints.' }
foreach ($taskPlayback in $taskView.provenance.exported_playbacks) {
    $taskName = $taskPlayback.trial_id + '.json'
    if ($taskName -notmatch '^[a-zA-Z0-9_-]+\.json$' -or $taskPlayback.url -cne ('/headroom-traces/' + $taskName)) { throw 'Unsafe or unexpected playback path.' }
    if ((Get-FileHash -LiteralPath (Join-Path (Join-Path $taskWeb 'headroom-traces') $taskName)).Hash.ToLowerInvariant() -ne $taskPlayback.sha256) { throw 'Staged playback hash differs.' }
}
$taskStagedHash = (Get-FileHash -LiteralPath $taskStage).Hash.ToLowerInvariant()
$taskArchive = Join-Path $taskVerification 'unpublished-index.json'
$taskReceipt = Join-Path $taskVerification 'publication-recovery.json'
if ((Test-Path -LiteralPath $taskArchive) -or (Test-Path -LiteralPath $taskReceipt)) { throw 'Publication recovery evidence already exists.' }
Copy-Item -LiteralPath $taskStage -Destination $taskArchive
# Same-directory NTFS file rename; File.Move's two-argument overload never
# replaces an existing destination. A race fails and leaves the staged source.
[System.IO.File]::Move($taskStage, $taskIndex)
$taskPublishedHash = (Get-FileHash -LiteralPath $taskIndex).Hash.ToLowerInvariant()
if ($taskPublishedHash -ne $taskStagedHash) { throw 'Published bytes differ; preserve all evidence.' }
$taskValue = [ordered]@{
    schema='light_headroom_publication_recovery_v1'
    reason='Linux renameat2(RENAME_NOREPLACE) rejected by the existing Docker Desktop Windows shared mount after full export staging.'
    native_export_exit_code=2
    method='System.IO.File.Move, two arguments, same-directory native Windows no-overwrite rename'
    source=$taskStage
    destination=$taskIndex
    index_sha256=$taskPublishedHash
    playback_hashes_checked=216
    report_sha256=$taskView.provenance.report_sha256
    audit_sha256=$taskView.provenance.audit_sha256
    staged_copy='verification/unpublished-index.json'
    script_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    host_scope='File inspection and OS filesystem integration only; no project build or runtime.'
    passed=$true
}
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskVerification 'publish-headroom.ps1')
[System.IO.File]::WriteAllText($taskReceipt, ($taskValue | ConvertTo-Json -Depth 6) + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
$taskValue | ConvertTo-Json -Depth 6
