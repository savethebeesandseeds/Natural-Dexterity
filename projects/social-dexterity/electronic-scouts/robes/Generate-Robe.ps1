param(
    [string]$Measurements = (Join-Path $PSScriptRoot 'measurements.example.json'),
    [string]$Output = (Join-Path $PSScriptRoot 'output'),
    [switch]$NoTiles
)
$ErrorActionPreference = 'Stop'
$taskRuntime = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $taskRuntime) {
    $taskPython = $taskRuntime
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $taskPython = (Get-Command python).Source
} else {
    throw 'Python 3 with reportlab is required. See README.md.'
}
$taskArgs = @((Join-Path $PSScriptRoot 'generate_pattern.py'), '--measurements', $Measurements, '--output', $Output)
if ($NoTiles) { $taskArgs += '--no-tiles' }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Pattern generation failed with exit code $LASTEXITCODE" }
