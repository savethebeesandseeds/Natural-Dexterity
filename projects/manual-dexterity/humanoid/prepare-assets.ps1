# Pinned public C/C++ dependencies. No container lifecycle or repository creation.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$runtimeRoot = Join-Path $PSScriptRoot 'runtime'
$sources = @{
    'unitree_rl_gym-276801e.tar.gz' = 'https://codeload.github.com/unitreerobotics/unitree_rl_gym/tar.gz/276801e46c5d433564f24658bac64f254b7d2d4b'
    'mujoco-3.3.2-linux-x86_64.tar.gz' = 'https://github.com/google-deepmind/mujoco/releases/download/3.3.2/mujoco-3.3.2-linux-x86_64.tar.gz'
    'libtorch-cxx11-abi-shared-with-deps-2.7.1+cu126.zip' = 'https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu126.zip'
}
$manifest = Get-Content -LiteralPath (Join-Path $runtimeRoot 'dependencies.sha256')
if ($manifest.Count -ne 3) { throw 'Expected exactly three locked native dependency archives.' }
foreach ($line in $manifest) {
    if ($line -notmatch '^([a-f0-9]{64})  ([a-zA-Z0-9_.+-]+)$') { throw 'Invalid native dependency lock.' }
    $expectedHash = $Matches[1]
    $filename = $Matches[2]
    if (-not $sources.ContainsKey($filename)) { throw "No authoritative source for $filename" }
    $archive = Join-Path $runtimeRoot $filename
    if (Test-Path -LiteralPath $archive) {
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ine $expectedHash) {
            throw "Existing dependency $filename has the wrong checksum. Preserved for inspection."
        }
        Write-Output "Verified $filename"
        continue
    }
    $download = $archive + '.download-' + [Guid]::NewGuid().ToString('N')
    Invoke-WebRequest $sources[$filename] -OutFile $download
    if ((Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash -ine $expectedHash) {
        throw "Download checksum mismatch. Preserved $download for inspection."
    }
    [IO.File]::Move($download, $archive)
    Write-Output "Downloaded and verified $filename"
}
