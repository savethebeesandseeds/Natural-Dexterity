$ErrorActionPreference = 'Stop'
$taskRepo = 'C:\Work\Universalis-Materiae-Dexteritas'
$taskProject = Join-Path $taskRepo 'projects\droid-blocks'
$taskBatch = Join-Path $taskProject 'artifacts\reports\light-headroom-v1-development-20261004'
$taskQa = Join-Path $taskProject 'build\light-headroom-qa-20261004'
$taskReportPath = Join-Path $taskBatch 'report.json'
$taskReport = Get-Content -Raw -LiteralPath $taskReportPath | ConvertFrom-Json
if (!$taskReport.complete -or $taskReport.completed_runs -ne 216) { throw 'Collection is not complete.' }
$taskVerification = Join-Path $taskBatch 'verification'
$taskReceipt = Join-Path $taskVerification 'collection-preservation.json'
if (Test-Path -LiteralPath $taskReceipt) { throw 'Preservation receipt already exists.' }
[void](New-Item -ItemType Directory -Path $taskVerification -Force)
$taskProblems = [System.Collections.Generic.List[string]]::new()
function Confirm-TaskHash([string]$taskPath, [string]$taskExpected) {
    if (!(Test-Path -LiteralPath $taskPath -PathType Leaf)) {
        $taskProblems.Add('Missing file: ' + $taskPath)
    } elseif ((Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskExpected) {
        $taskProblems.Add('Changed file: ' + $taskPath)
    }
}
$taskPrior = Get-Content -Raw -LiteralPath (Join-Path $taskQa 'prior-artifact-hashes.json') | ConvertFrom-Json
foreach ($taskItem in $taskPrior) { Confirm-TaskHash (Join-Path $taskRepo $taskItem.path) $taskItem.sha256 }
$taskSourceCount = 0
foreach ($taskProperty in $taskReport.source_sha256.PSObject.Properties) {
    Confirm-TaskHash (Join-Path (Join-Path $taskBatch 'source') $taskProperty.Name) $taskProperty.Value
    Confirm-TaskHash (Join-Path $taskProject $taskProperty.Name) $taskProperty.Value
    $taskSourceCount++
}
$taskTraceCount = 0
$taskSnapshotCount = 0
foreach ($taskCase in $taskReport.cases) {
    foreach ($taskTrial in $taskCase.trials) {
        Confirm-TaskHash (Join-Path $taskBatch $taskTrial.trace.path) $taskTrial.trace.sha256
        Confirm-TaskHash (Join-Path $taskBatch $taskTrial.snapshots.path) $taskTrial.snapshots.sha256
        $taskTraceCount++
        $taskSnapshotCount++
    }
}
$taskLive = [System.Collections.Generic.List[object]]::new()
$taskClient = [System.Net.WebClient]::new()
try {
    foreach ($taskSession in @('light','construction')) {
        $taskAfter = Join-Path $taskQa ($taskSession + '-after.json')
        if (Test-Path -LiteralPath $taskAfter) { throw 'Live-state receipt already exists.' }
        [System.IO.File]::WriteAllBytes($taskAfter, $taskClient.DownloadData('http://127.0.0.1:43117/api/' + $taskSession + '/state'))
        $taskBefore = Join-Path $taskQa ($taskSession + '-before.json')
        $taskBeforeHash = (Get-FileHash -LiteralPath $taskBefore -Algorithm SHA256).Hash.ToLowerInvariant()
        $taskAfterHash = (Get-FileHash -LiteralPath $taskAfter -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($taskBeforeHash -ne $taskAfterHash) { $taskProblems.Add('Live session changed: ' + $taskSession) }
        $taskLive.Add([ordered]@{session=$taskSession; before_sha256=$taskBeforeHash; after_sha256=$taskAfterHash; identical=($taskBeforeHash -eq $taskAfterHash)})
    }
} finally { $taskClient.Dispose() }
foreach ($taskName in @('runtime.json','prior-artifact-hashes.json','native-ctest.log','collection.log','check-preservation.ps1','light-before.json','light-after.json','construction-before.json','construction-after.json')) {
    $taskDestination = Join-Path $taskVerification $taskName
    if (Test-Path -LiteralPath $taskDestination) { throw 'Verification evidence already exists.' }
    Copy-Item -LiteralPath (Join-Path $taskQa $taskName) -Destination $taskDestination
}
$taskValue = [ordered]@{
    schema='light_headroom_preservation_v1'
    report_sha256=(Get-FileHash -LiteralPath $taskReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    preexisting_artifacts_checked=$taskPrior.Count
    archived_and_live_sources_checked=$taskSourceCount
    raw_transition_tapes_checked=$taskTraceCount
    snapshot_files_checked=$taskSnapshotCount
    frozen_rover_inputs_checked=@('config/learning_experiment_v3.json','artifacts/light-search-policy-v3-seed0.json','models/droid.xml')
    live_sessions=$taskLive.ToArray()
    scope='Read-only host file hashing and byte-for-byte API GET comparison after native container collection, before verification-tool correction. No host build, project runtime or learning.'
    passed=($taskProblems.Count -eq 0)
    issues=$taskProblems.ToArray()
}
[System.IO.File]::WriteAllText($taskReceipt, ($taskValue | ConvertTo-Json -Depth 8) + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
$taskValue | ConvertTo-Json -Depth 8
if ($taskProblems.Count -ne 0) { throw 'Preservation checks failed; retain the receipt.' }
