[CmdletBinding()]
param([switch] $Publish)

# Reuse the existing LaTeX environment without changing its configuration.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$containerName = 'documents-latex'
$manuscripts = @(Get-ChildItem -LiteralPath $paperRoot -Filter '*.tex' |
    Where-Object { $_.Name -ne 'ieee-preamble.tex' })
if ($manuscripts.Count -ne 1) { throw 'Expected exactly one main .tex manuscript.' }
$documentName = $manuscripts[0].BaseName

function Invoke-PaperDocker {
    param([Parameter(Mandatory = $true)][string[]] $Arguments)
    $result = @(& docker @Arguments 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0) {
        throw ("docker {0} failed:`n{1}" -f ($Arguments -join ' '), ($result -join "`n"))
    }
    return $result
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI is required, or run bash build.sh in an existing TeX environment.'
}
$inspection = (Invoke-PaperDocker -Arguments @('inspect', $containerName)) -join "`n" |
    ConvertFrom-Json
$runtime = @($inspection)[0]
if ($runtime.State.Status -ne 'running') {
    throw "Start the existing environment with: & 'C:\Work\documents\cv.ps1' start"
}
if ($runtime.Config.Labels.'org.local.documents-latex.managed' -ne 'true' -or
    $runtime.Config.Image -ne 'debian:12-slim') {
    throw 'The existing container does not match the documented LaTeX environment.'
}
$containerId = $runtime.Id
$snapshot = ((Invoke-PaperDocker -Arguments @('exec', $containerId, 'mktemp', '-d',
    "/tmp/$documentName-XXXXXX")) -join '').Trim()
if ($snapshot -notmatch ('^/tmp/' + [regex]::Escape($documentName) + '-[A-Za-z0-9]+$')) {
    throw "Unexpected build snapshot path: $snapshot"
}

foreach ($relativePath in @($manuscripts[0].Name, 'ieee-preamble.tex', 'references.bib',
        'build.sh', 'vendor')) {
    $source = Join-Path $paperRoot $relativePath
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    Invoke-PaperDocker -Arguments @('cp', $source, "${containerId}:$snapshot/") | Out-Host
}
# Windows checkouts can use CRLF; normalize only the disposable shell-script copy.
Invoke-PaperDocker -Arguments @('exec', $containerId, 'sed', '-i', 's/\r$//',
    "$snapshot/build.sh") | Out-Host
Invoke-PaperDocker -Arguments @('exec', $containerId, 'bash', "$snapshot/build.sh") |
    Select-Object -Last 14 | Out-Host
$buildDirectory = Join-Path $paperRoot 'build'
New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
Invoke-PaperDocker -Arguments @('cp', "${containerId}:$snapshot/build/.", $buildDirectory) | Out-Host
$builtPdf = Join-Path $buildDirectory "$documentName.pdf"
if (-not (Test-Path -LiteralPath $builtPdf -PathType Leaf)) { throw 'No compiled PDF returned.' }

if ($Publish) {
    Copy-Item -LiteralPath $builtPdf -Destination (Join-Path $paperRoot "$documentName.pdf") -Force
}
Write-Host "Verified one-page PDF: $builtPdf"
Write-Host "Build logs and bibliography: $buildDirectory"
Write-Host "Container build snapshot: $snapshot"
