[CmdletBinding()]
param(
    [ValidateSet('start', 'build', 'status', 'stop', 'logs', 'shell', 'games', 'check', 'train', 'train-shared', 'train-shared-extend', 'pilot-followups', 'shared-pilots', 'evaluate', 'stop-training')]
    [string]$Action = 'status',
    [ValidateRange(1, 1000000000)][int]$Steps = 5000000,
    [ValidateRange(0.001, 24)][double]$Hours = 3,
    [ValidateRange(1, 64)][int]$Envs = 8,
    [ValidateRange(0, 99999)][int]$Seed = 1,
    [string]$Resume = '',
    [string]$TransferFeatures = '',
    [string]$Checkpoint = '',
    [string]$SourceEvaluation = '',
    [string]$PrerequisiteRun = '',
    [ValidateRange(1, 19)][int]$Episodes = 5,
    [ValidateRange(100000, 999999)][int]$EvalSeed = 200000,
    [ValidateRange(1, 900)][int]$Seconds = 180,
    [ValidateRange(1, 7200)][int]$EvaluationSeconds = 3600,
    [string]$Game = ''
)
$ErrorActionPreference = 'Stop'
$ContainerName = 'atari-dev'
$ImageName = 'atari:dev'
$ProjectPath = [IO.Path]::GetFullPath($PSScriptRoot)
$ExpectedCommand = @('sh', '/workspace/run.sh')
$ExpectedRestartPolicy = 'no'
$GameWasSpecified = $PSBoundParameters.ContainsKey('Game')
$script:GameRegistry = $null

function Read-ArtifactText([string]$Path) {
    # Native writers replace JSON atomically; allow that replacement during a host read.
    $share = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $share)
    $reader = $null
    try {
        $reader = [IO.StreamReader]::new($stream)
        return $reader.ReadToEnd()
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
        else { $stream.Dispose() }
    }
}

function Get-GameRegistry {
    if ($null -eq $script:GameRegistry) {
        $document = Read-ArtifactText (Join-Path $ProjectPath 'games.json') | ConvertFrom-Json
        if ($document.schema_version -ne 1 -or !$document.games -or @($document.games).Count -eq 0) { throw 'Expected a nonempty version 1 game registry.' }
        $ids = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        $roms = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($entry in @($document.games)) {
            if ($entry.id -isnot [string] -or $entry.id -cnotmatch '^[a-z][a-z0-9_]{0,63}$' -or !$ids.Add($entry.id)) { throw 'Game registry has an unsafe or duplicate ID.' }
            if ($entry.rom_filename -isnot [string] -or $entry.rom_filename -cnotmatch '^[a-z][a-z0-9_]{0,63}\.bin$' -or !$roms.Add($entry.rom_filename)) { throw 'Game registry has an unsafe or duplicate ROM filename.' }
            if ($entry.rom_sha256 -isnot [string] -or $entry.rom_sha256 -cnotmatch '^[0-9a-f]{64}$') { throw 'Game registry needs pinned ROM SHA256 values.' }
        }
        $script:GameRegistry = @($document.games)
    }
    return $script:GameRegistry
}

function Resolve-Game([string]$Id) {
    $registry = @(Get-GameRegistry)
    foreach ($entry in $registry) { if ($entry.id -ceq $Id) { return $entry } }
    throw "Unknown game ID '$Id'. Available games: $(($registry.id) -join ', ')."
}

function Get-CheckpointGame([string]$CheckpointPath, [switch]$AllowDifferentGame) {
    $configPath = Join-Path ([IO.Path]::GetDirectoryName($CheckpointPath)) 'config.json'
    if (!(Test-Path -LiteralPath $configPath -PathType Leaf)) { throw 'The selected checkpoint needs an adjacent config.json.' }
    $config = Read-ArtifactText $configPath | ConvertFrom-Json
    $romFilename = if ($config.rom -is [string]) { ($config.rom.Replace('\', '/') -split '/')[-1] } else { '' }
    if ($config.PSObject.Properties.Name -contains 'game_id') {
        if ($config.game_id -isnot [string] -or !$config.game_id) { throw 'Checkpoint config has an invalid game_id.' }
        $inferred = Resolve-Game $config.game_id
        if ($romFilename -and $romFilename -cne $inferred.rom_filename) { throw 'Checkpoint game_id and ROM filename disagree.' }
    } else {
        $known = @(Get-GameRegistry | Where-Object { $_.rom_filename -ceq $romFilename })
        if ($known.Count -ne 1) { throw 'Cannot infer checkpoint game from its config; expected game_id or a recognized legacy ROM filename.' }
        $inferred = $known[0]
    }
    if ($config.rom_sha256 -and $config.rom_sha256 -cne $inferred.rom_sha256) { throw 'Checkpoint ROM SHA256 disagrees with the game registry.' }
    if (!$AllowDifferentGame -and $GameWasSpecified -and $Game -cne $inferred.id) { throw "Selected game '$Game' does not match checkpoint game '$($inferred.id)'." }
    return $inferred
}

function Invoke-DockerChecked([string[]]$DockerArguments) {
    & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) { throw "Docker failed ($LASTEXITCODE): $($DockerArguments -join ' ')" }
}

function Read-Container {
    $matches = @(Invoke-DockerChecked @('container', 'ls', '-a', '--filter', "name=^/$ContainerName$", '--format', '{{.ID}}'))
    if ($matches.Count -eq 0) { return $null }
    return @((Invoke-DockerChecked @('container', 'inspect', $ContainerName)) | ConvertFrom-Json)[0]
}

function Assert-Contract($Container) {
    $differences = [Collections.Generic.List[string]]::new()
    foreach ($pair in @(@('io.waajacu.managed', 'true'), @('io.waajacu.project', 'atari'), @('io.waajacu.config', 'native-v1'))) {
        if ($Container.Config.Labels.($pair[0]) -ne $pair[1]) { $differences.Add("label $($pair[0])") }
    }
    if ($Container.Config.Image -ne $ImageName) { $differences.Add('image name') }
    $image = @((Invoke-DockerChecked @('image', 'inspect', $ImageName)) | ConvertFrom-Json)[0]
    if ($Container.Image -ne $image.Id) { $differences.Add('immutable image ID differs from current tag') }
    if (($Container.Config.Cmd -join '|') -ne ($ExpectedCommand -join '|')) { $differences.Add('command') }
    if ($null -ne $Container.Config.Entrypoint -and @($Container.Config.Entrypoint).Count -gt 0) { $differences.Add('entrypoint') }
    if ($Container.Config.WorkingDir -ne '/workspace') { $differences.Add('working directory') }
    $mounts = @($Container.Mounts)
    if ($mounts.Count -ne 1 -or $mounts[0].Type -ne 'bind' -or $mounts[0].Destination -ne '/workspace' -or !$mounts[0].RW -or $mounts[0].Source.Replace('/', '\').TrimEnd('\') -ine $ProjectPath.TrimEnd('\')) { $differences.Add('bind mounts or volumes') }
    $ports = @($Container.HostConfig.PortBindings.PSObject.Properties)
    $binding = @($Container.HostConfig.PortBindings.'8080/tcp')
    if ($ports.Count -ne 1 -or $binding.Count -ne 1 -or $binding[0].HostIp -ne '127.0.0.1' -or $binding[0].HostPort -ne '43260') { $differences.Add('published ports') }
    if ($Container.HostConfig.RestartPolicy.Name -ne $ExpectedRestartPolicy) { $differences.Add('restart policy') }
    if (!$Container.HostConfig.Init -or $Container.HostConfig.ShmSize -ne 1073741824 -or $Container.Config.StopTimeout -ne 30) { $differences.Add('init/shared memory/stop timeout') }
    $gpu = @($Container.HostConfig.DeviceRequests)
    if ($gpu.Count -ne 1 -or $gpu[0].Count -ne -1 -or (@($gpu[0].Capabilities[0]) -notcontains 'gpu')) { $differences.Add('GPU request') }
    if (($null -ne $Container.HostConfig.Devices -and @($Container.HostConfig.Devices).Count -ne 0) -or $Container.HostConfig.Privileged -or $Container.HostConfig.NetworkMode -eq 'host') { $differences.Add('extra devices/privilege/network') }
    if (@($Container.Config.Env) -notcontains 'NVIDIA_DRIVER_CAPABILITIES=compute,utility') { $differences.Add('NVIDIA capabilities') }
    if ($differences.Count) { throw "Preserving conflicting container $($Container.Id). Differences: $($differences -join ', '). No replacement is permitted by this launcher." }
}

function Build-Image {
    [IO.Directory]::CreateDirectory((Join-Path $ProjectPath '.runtime/dependencies')) | Out-Null
    Invoke-DockerChecked @('build', '--tag', $ImageName, $ProjectPath)
}

function Ensure-Image {
    $imageIds = @(Invoke-DockerChecked @('image', 'ls', '--quiet', $ImageName))
    if ($imageIds.Count -eq 0) { Build-Image; return }
    $image = @((Invoke-DockerChecked @('image', 'inspect', $ImageName)) | ConvertFrom-Json)[0]
    if ($image.Config.Labels.'io.waajacu.managed' -ne 'true' -or $image.Config.Labels.'io.waajacu.project' -ne 'atari' -or $image.Config.Labels.'io.waajacu.config' -ne 'native-v1') {
        throw "Preserving unmanaged or incompatible image $ImageName ($($image.Id))."
    }
}

function Assert-FreePort {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 43260)
    try { $listener.Start() }
    catch { throw 'Port 43260 is in use. No container was created; select and approve a free port.' }
    finally { $listener.Stop() }
}

function Save-Record($Container) {
    $recordDir = Join-Path $ProjectPath '.runtime'
    [IO.Directory]::CreateDirectory($recordDir) | Out-Null
    $Container | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $recordDir 'container-inspect.json') -Encoding utf8
}

$selectedGame = $null
if ($GameWasSpecified) { $selectedGame = Resolve-Game $Game }
if ($Action -eq 'games') { @(Get-GameRegistry) | ConvertTo-Json -Depth 10; return }
if ($Action -in @('check', 'train') -and $null -eq $selectedGame) { $selectedGame = Resolve-Game 'pong' }

$container = Read-Container
if ($null -ne $container) { Assert-Contract $container }
switch ($Action) {
    'build' {
        if ($null -ne $container) { throw 'Preserving the existing container and its image tag. A replacement build requires a separately approved rebuild procedure.' }
        Build-Image
        break
    }
    'start' {
        if ($null -eq $container) {
            Ensure-Image
            Assert-FreePort
            Invoke-DockerChecked @('container', 'create', '--name', $ContainerName,
                '--label', 'io.waajacu.managed=true', '--label', 'io.waajacu.project=atari', '--label', 'io.waajacu.config=native-v1',
                '--mount', "type=bind,source=$ProjectPath,target=/workspace",
                '--publish', '127.0.0.1:43260:8080', '--gpus', 'all',
                '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
                '--init', '--shm-size', '1g', '--stop-timeout', '30', '--restart', $ExpectedRestartPolicy, $ImageName)
            $container = Read-Container
            Assert-Contract $container
        }
        if (!$container.State.Running) { Invoke-DockerChecked @('container', 'start', $container.Id) }
        $container = Read-Container
        Assert-Contract $container
        Save-Record $container
        Write-Host "Viewer: http://127.0.0.1:43260/ | ID: $($container.Id)"
        break
    }
    'status' {
        if ($null -eq $container) { Write-Host 'atari-dev does not exist.'; break }
        $container | Select-Object Id, Image, State, Mounts | ConvertTo-Json -Depth 12
        break
    }
    default {
        if ($null -eq $container) { throw 'No Atari container exists. Run .\atari.ps1 start after approving the contract.' }
        if ($Action -eq 'stop') { Invoke-DockerChecked @('container', 'stop', $container.Id); break }
        if ($Action -eq 'logs') { Invoke-DockerChecked @('container', 'logs', '--tail', '100', $container.Id); break }
        if (!$container.State.Running) { throw 'The matching container is stopped. Run .\atari.ps1 start to reuse it.' }
        switch ($Action) {
            'shell' { Invoke-DockerChecked @('exec', '-it', $container.Id, 'bash') }
            'check' {
                Invoke-DockerChecked @('exec', $container.Id, '/workspace/build/native/atari', 'check', '--game', $selectedGame.id)
                Invoke-DockerChecked @('exec', $container.Id, 'ctest', '--test-dir', '/workspace/build/native', '--output-on-failure')
                Invoke-DockerChecked @('exec', $container.Id, 'sh', '-c', '! command -v python && ! command -v python3')
            }
            'train' {
                if ($Resume -and $TransferFeatures) { throw 'Choose Resume or TransferFeatures, not both.' }
                $containerCheckpoint = $null
                if ($Resume) {
                    $checkpoint = if ($Resume.StartsWith('/workspace/')) {
                        Join-Path $ProjectPath $Resume.Substring('/workspace/'.Length)
                    } elseif ([IO.Path]::IsPathRooted($Resume)) { $Resume }
                    else { Join-Path $ProjectPath $Resume }
                    $checkpoint = [IO.Path]::GetFullPath($checkpoint)
                    $runsPrefix = (Join-Path $ProjectPath 'runs') + [IO.Path]::DirectorySeparatorChar
                    if (!$checkpoint.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase) -or !(Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                        throw 'Resume checkpoint must be an existing file under this project runs directory.'
                    }
                    $containerCheckpoint = '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $checkpoint).Replace('\', '/')
                    $selectedGame = Get-CheckpointGame $checkpoint
                }
                if (!$PSBoundParameters.ContainsKey('Steps')) { $Steps = [int]$selectedGame.training_decisions }
                $trainArguments = @('exec', '-d', $container.Id, '/workspace/build/native/atari', 'train', '--game', $selectedGame.id, '--steps', "$Steps", '--hours', $Hours.ToString([Globalization.CultureInfo]::InvariantCulture), '--envs', "$Envs", '--seed', "$Seed")
                if ($containerCheckpoint) { $trainArguments += @('--resume', $containerCheckpoint) }
                if ($TransferFeatures) {
                    $source = if ($TransferFeatures.StartsWith('/workspace/')) { Join-Path $ProjectPath $TransferFeatures.Substring('/workspace/'.Length) }
                        elseif ([IO.Path]::IsPathRooted($TransferFeatures)) { $TransferFeatures }
                        else { Join-Path $ProjectPath $TransferFeatures }
                    $source = [IO.Path]::GetFullPath($source)
                    $runsPrefix = (Join-Path $ProjectPath 'runs') + [IO.Path]::DirectorySeparatorChar
                    if (!$source.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase) -or !(Test-Path -LiteralPath $source -PathType Leaf)) {
                        throw 'Transfer checkpoint must be an existing file under this project runs directory.'
                    }
                    $null = Get-CheckpointGame $source -AllowDifferentGame
                    $sourceContainer = '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $source).Replace('\', '/')
                    $trainArguments += @('--transfer-features', $sourceContainer)
                }
                Invoke-DockerChecked $trainArguments
                Write-Host 'Experiment dispatched. Check runs/current.json, its .log file, and the viewer for startup and progress.'
            }
            'evaluate' {
                if ($EvalSeed + $Episodes - 1 -gt 999999) { throw 'Development evaluation seeds must remain between 100000 and 999999.' }
                if (!$Checkpoint) {
                    $current = Read-ArtifactText (Join-Path $ProjectPath 'runs/current.json') | ConvertFrom-Json
                    if (!$current.run_dir -or !$current.run_dir.StartsWith('/workspace/runs/')) { throw 'Current run must be under /workspace/runs.' }
                    $Checkpoint = $current.run_dir.TrimEnd('/') + '/latest.pt'
                }
                $sourceCheckpoint = if ($Checkpoint.StartsWith('/workspace/')) {
                    Join-Path $ProjectPath $Checkpoint.Substring('/workspace/'.Length)
                } elseif ([IO.Path]::IsPathRooted($Checkpoint)) { $Checkpoint }
                else { Join-Path $ProjectPath $Checkpoint }
                $sourceCheckpoint = [IO.Path]::GetFullPath($sourceCheckpoint)
                $runsPrefix = (Join-Path $ProjectPath 'runs') + [IO.Path]::DirectorySeparatorChar
                if (!$sourceCheckpoint.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase) -or !(Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
                    throw 'Evaluation checkpoint must be an existing file under this project runs directory.'
                }
                $sourceDirectory = [IO.Path]::GetDirectoryName($sourceCheckpoint)
                if (!(Test-Path -LiteralPath (Join-Path $sourceDirectory 'config.json') -PathType Leaf)) { throw 'The selected checkpoint needs an adjacent config.json.' }
                $selectedGame = Get-CheckpointGame $sourceCheckpoint
                $containerCheckpoint = '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $sourceCheckpoint).Replace('\', '/')
                $snapshotName = 'development-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
                $snapshotPath = Join-Path $sourceDirectory $snapshotName
                $containerSnapshot = '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $snapshotPath).Replace('\', '/')
                $manifest = [ordered]@{
                    role = 'development_evaluation'
                    game_id = $selectedGame.id
                    source_checkpoint = $containerCheckpoint
                    checkpoint = "$containerSnapshot/checkpoint.pt"
                    seed_start = $EvalSeed
                    seed_end = $EvalSeed + $Episodes - 1
                    episodes_requested = $Episodes
                    max_seconds = $Seconds
                    final_evaluation = $false
                } | ConvertTo-Json -Compress
                $snapshotScript = @'
set -eu
source=$(readlink -f -- "$1")
case "$source" in /workspace/runs/*) ;; *) echo 'Checkpoint resolved outside /workspace/runs.' >&2; exit 1 ;; esac
test -f "$source"
config="$(dirname -- "$source")/config.json"
test -f "$config"
mkdir -- "$2"
cp -- "$source" "$2/checkpoint.pt"
cp -- "$config" "$2/config.json"
printf '%s\n' "$3" > "$2/development.json"
/workspace/build/native/atari-checkpoint-check "$2/checkpoint.pt" > "$2/checkpoint-check.json"
'@
                Invoke-DockerChecked @('exec', $container.Id, 'sh', '-c', $snapshotScript.Replace("`r`n", "`n"), 'atari-development-snapshot', $containerCheckpoint, $containerSnapshot, $manifest)
                Write-Host "Development evaluation; this is not the final verdict. Snapshot: $snapshotPath"
                Invoke-DockerChecked @('exec', $container.Id, '/workspace/build/native/atari', 'evaluate', '--game', $selectedGame.id, '--checkpoint', "$containerSnapshot/checkpoint.pt", '--output', "$containerSnapshot/evaluation.json", '--episodes', "$Episodes", '--seed', "$EvalSeed", '--seconds', "$Seconds")
                Write-Host "Development result: $(Join-Path $snapshotPath 'evaluation.json')"
            }
            'train-shared' {
                if (!$Checkpoint) { throw 'Shared training needs -Checkpoint pointing to the preserved Pong source.' }
                if ($GameWasSpecified -or $Resume -or $TransferFeatures) { throw 'Shared training uses the declared Pong + Breakout mixture and its own source checkpoint.' }
                if ($Envs % 2 -ne 0 -or $Envs -lt 2) { throw 'Shared training needs an even environment count of at least two.' }
                if (!$PSBoundParameters.ContainsKey('Steps')) { $Steps = 2000000 }
                if (!$PSBoundParameters.ContainsKey('Hours')) { $Hours = 1 }
                if (!$PSBoundParameters.ContainsKey('Seed')) { $Seed = 41 }
                $sharedArguments = @('exec', '-d', $container.Id, '/workspace/build/native/atari', 'shared-train', '--steps', "$Steps", '--hours', $Hours.ToString([Globalization.CultureInfo]::InvariantCulture), '--envs', "$Envs", '--seed', "$Seed", '--evaluation-seconds', "$EvaluationSeconds")
                foreach ($item in @(@('--source-checkpoint', $Checkpoint), @('--source-evaluation', $SourceEvaluation))) {
                    if (!$item[1]) { continue }
                    $inputPath = [string]$item[1]
                    $source = if ($inputPath.StartsWith('/workspace/')) { Join-Path $ProjectPath $inputPath.Substring('/workspace/'.Length) }
                        elseif ([IO.Path]::IsPathRooted($inputPath)) { $inputPath }
                        else { Join-Path $ProjectPath $inputPath }
                    $source = [IO.Path]::GetFullPath($source)
                    $runsPrefix = (Join-Path $ProjectPath 'runs') + [IO.Path]::DirectorySeparatorChar
                    if (!$source.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase) -or !(Test-Path -LiteralPath $source -PathType Leaf)) {
                        throw 'Shared source files must exist under this project runs directory.'
                    }
                    if ($item[0] -eq '--source-checkpoint' -and (Get-CheckpointGame $source).id -ne 'pong') { throw 'Shared source must be a Pong checkpoint.' }
                    $sharedArguments += @($item[0], '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $source).Replace('\', '/'))
                }
                Invoke-DockerChecked $sharedArguments
                Write-Host 'Shared experiment dispatched. The budget is total new decisions, balanced equally between Pong and Breakout.'
            }
            'pilot-followups' {
                if (!$PrerequisiteRun -or !$Checkpoint -or !$SourceEvaluation) { throw 'Pilot followups need -PrerequisiteRun, -Checkpoint and -SourceEvaluation.' }
                if ($GameWasSpecified -or $Resume -or $TransferFeatures -or $PSBoundParameters.ContainsKey('Steps') -or $PSBoundParameters.ContainsKey('Hours') -or $PSBoundParameters.ContainsKey('Seed') -or $PSBoundParameters.ContainsKey('Envs')) {
                    throw 'Pilot followups use the fixed GOAL.md budgets, seeds and eight environments.'
                }
                $suiteArguments = @('exec', '-d', $container.Id, 'sh', '/workspace/run-pilots.sh')
                foreach ($inputPath in @($PrerequisiteRun, $Checkpoint, $SourceEvaluation)) {
                    $source = if ($inputPath.StartsWith('/workspace/')) { Join-Path $ProjectPath $inputPath.Substring('/workspace/'.Length) }
                        elseif ([IO.Path]::IsPathRooted($inputPath)) { $inputPath }
                        else { Join-Path $ProjectPath $inputPath }
                    $source = [IO.Path]::GetFullPath($source)
                    $runsPrefix = (Join-Path $ProjectPath 'runs') + [IO.Path]::DirectorySeparatorChar
                    if (!$source.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase) -or !(Test-Path -LiteralPath $source)) {
                        throw 'Pilot evidence must exist under this project runs directory.'
                    }
                    $suiteArguments += '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $source).Replace('\', '/')
                }
                Invoke-DockerChecked $suiteArguments
                Write-Host 'Pilot sequence dispatched. It waits for the prerequisite and preserves its manifest, status and logs under runs/pilot-suite-*.'
            }
            'shared-pilots' {
                if (!$Checkpoint -or !$SourceEvaluation) { throw 'Shared pilots need -Checkpoint and -SourceEvaluation for the preserved Pong proof.' }
                if ($GameWasSpecified -or $Resume -or $TransferFeatures -or $PrerequisiteRun -or $PSBoundParameters.ContainsKey('Steps') -or $PSBoundParameters.ContainsKey('Hours') -or $PSBoundParameters.ContainsKey('Seed') -or $PSBoundParameters.ContainsKey('Envs') -or $PSBoundParameters.ContainsKey('EvaluationSeconds')) {
                    throw 'Shared pilots use the fixed GOAL.md decisions/seeds/workers, two-hour training guards and 3600-second per-game evaluation guards.'
                }
                $suiteArguments = @('exec', '-d', $container.Id, 'sh', '/workspace/run-shared-pilots.sh')
                foreach ($inputPath in @($Checkpoint, $SourceEvaluation)) {
                    $source = if ($inputPath.StartsWith('/workspace/')) { Join-Path $ProjectPath $inputPath.Substring('/workspace/'.Length) }
                        elseif ([IO.Path]::IsPathRooted($inputPath)) { $inputPath }
                        else { Join-Path $ProjectPath $inputPath }
                    $source = [IO.Path]::GetFullPath($source)
                    $runsPrefix = (Join-Path $ProjectPath 'runs') + [IO.Path]::DirectorySeparatorChar
                    if (!$source.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase) -or !(Test-Path -LiteralPath $source -PathType Leaf)) {
                        throw 'Shared pilot source files must exist under this project runs directory.'
                    }
                    $suiteArguments += '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $source).Replace('\', '/')
                }
                Invoke-DockerChecked $suiteArguments
                Write-Host 'Shared two-game and three-game pilots dispatched. Fresh evidence, status and logs are preserved under runs/shared-suite-*.'
            }
            'train-shared-extend' {
                if (!$Checkpoint) { throw 'Shared extension needs -Checkpoint pointing to a completed two-game shared-final.pt.' }
                if ($GameWasSpecified -or $Resume -or $TransferFeatures -or $SourceEvaluation) { throw 'Shared extension uses all three registered games and reads the adjacent two-game evidence.' }
                if (!$PSBoundParameters.ContainsKey('Steps')) { $Steps = 3000000 }
                if (!$PSBoundParameters.ContainsKey('Hours')) { $Hours = 1 }
                if (!$PSBoundParameters.ContainsKey('Seed')) { $Seed = 51 }
                if (!$PSBoundParameters.ContainsKey('Envs')) { $Envs = 12 }
                if ($Envs -lt 3 -or $Envs % 3 -ne 0) { throw 'Shared extension needs an environment count divisible by three.' }
                $source = if ($Checkpoint.StartsWith('/workspace/')) { Join-Path $ProjectPath $Checkpoint.Substring('/workspace/'.Length) }
                    elseif ([IO.Path]::IsPathRooted($Checkpoint)) { $Checkpoint }
                    else { Join-Path $ProjectPath $Checkpoint }
                $source = [IO.Path]::GetFullPath($source)
                $runsPrefix = (Join-Path $ProjectPath 'runs') + [IO.Path]::DirectorySeparatorChar
                if (!$source.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase) -or !(Test-Path -LiteralPath $source -PathType Leaf) -or [IO.Path]::GetFileName($source) -ne 'shared-final.pt') {
                    throw 'Shared extension source must be a shared-final.pt under this project runs directory.'
                }
                $sourceDirectory = [IO.Path]::GetDirectoryName($source)
                foreach ($required in @('config.json', 'progress.json', 'retention.json', 'checkpoint-manifest.json')) {
                    if (!(Test-Path -LiteralPath (Join-Path $sourceDirectory $required) -PathType Leaf)) { throw "Shared source is missing $required." }
                }
                $sourceProgress = Read-ArtifactText (Join-Path $sourceDirectory 'progress.json') | ConvertFrom-Json
                $sourceRetention = Read-ArtifactText (Join-Path $sourceDirectory 'retention.json') | ConvertFrom-Json
                if ($sourceProgress.status -ne 'complete' -or $sourceRetention.status -ne 'complete') { throw 'The two-game experiment and its retention measurements must complete before extension.' }
                $sourceContainer = '/workspace/' + [IO.Path]::GetRelativePath($ProjectPath, $source).Replace('\', '/')
                Invoke-DockerChecked @('exec', '-d', $container.Id, '/workspace/build/native/atari', 'shared-extend', '--source-checkpoint', $sourceContainer, '--steps', "$Steps", '--hours', $Hours.ToString([Globalization.CultureInfo]::InvariantCulture), '--envs', "$Envs", '--seed', "$Seed", '--evaluation-seconds', "$EvaluationSeconds")
                Write-Host 'Three-game extension dispatched. The total new budget is divided equally among Pong, Breakout and Space Invaders.'
            }
            'stop-training' {
                $currentPath = Join-Path $ProjectPath 'runs/current.json'
                $current = Read-ArtifactText $currentPath | ConvertFrom-Json
                if ($current.run_dir -isnot [string] -or !$current.run_dir.StartsWith('/workspace/runs/', [StringComparison]::Ordinal)) { throw 'Current run must be under /workspace/runs; preserving state.' }
                $runName = $current.run_dir.Substring('/workspace/runs/'.Length)
                $prefixes = (@(Get-GameRegistry | ForEach-Object { [regex]::Escape($_.id) }) + @('shared-pong-breakout')) -join '|'
                if ($runName -cnotmatch "^(?:$prefixes)-[A-Za-z0-9-]+$") { throw 'Unexpected run name; preserving state.' }
                if ($GameWasSpecified -and !$runName.StartsWith($Game + '-', [StringComparison]::Ordinal)) { throw "Current run does not belong to selected game '$Game'; preserving state." }
                $runsPrefix = [IO.Path]::GetFullPath((Join-Path $ProjectPath 'runs')) + [IO.Path]::DirectorySeparatorChar
                $runPath = [IO.Path]::GetFullPath((Join-Path $runsPrefix $runName))
                if (!$runPath.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Current run resolved outside this project runs directory; preserving state.' }
                if (!(Test-Path -LiteralPath $runPath -PathType Container)) { throw 'Current run has not initialized yet.' }
                New-Item -ItemType File -Path (Join-Path $runPath 'STOP') -Force | Out-Null
                Write-Host 'Graceful stop requested. The learner will save its checkpoint.'
            }
        }
    }
}
