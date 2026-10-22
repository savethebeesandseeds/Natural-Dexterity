[CmdletBinding()]
param([ValidateSet('up', 'status', 'stop', 'logs', 'test', 'check', 'gpu')][string]$Action = 'up')
$ErrorActionPreference = 'Stop'
$containerName = 'helicopter-dev'
$imageName = 'helicopter:dev'
$projectPath = [IO.Path]::GetFullPath($PSScriptRoot)

function Invoke-Docker {
    param([string[]]$DockerArgs)
    & docker @DockerArgs
    if ($LASTEXITCODE -ne 0) { throw "Docker failed ($LASTEXITCODE): $($DockerArgs -join ' ')" }
}
function Read-Container {
    $names = @(Invoke-Docker @('container', 'ls', '-a', '--filter', "name=^/$containerName$", '--format', '{{.Names}}'))
    if ($names -notcontains $containerName) { return $null }
    $data = Invoke-Docker @('container', 'inspect', $containerName)
    return @($data | ConvertFrom-Json)[0]
}
function Assert-Container($container) {
    $problems = [Collections.Generic.List[string]]::new()
    if ($container.Config.Labels.'io.waajacu.managed' -ne 'true' -or
        $container.Config.Labels.'io.waajacu.project' -ne 'helicopter' -or
        $container.Config.Labels.'io.waajacu.configuration' -ne '1') { $problems.Add('ownership/configuration labels') }
    if ($container.Config.Image -ne $imageName) { $problems.Add('image name') }
    $expectedImage = @(Invoke-Docker @('image', 'inspect', $imageName, '--format', '{{.Id}}'))[0]
    if ($container.Image -ne $expectedImage) { $problems.Add('immutable image ID (image was rebuilt; replacement needs explicit review)') }
    if (($container.Config.Cmd -join '|') -ne '/bin/bash|/workspace/run.sh') { $problems.Add('startup command') }
    if ($container.Config.Entrypoint -and @($container.Config.Entrypoint).Count -gt 0) { $problems.Add('entrypoint') }
    if ($container.Config.WorkingDir -ne '/workspace') { $problems.Add('working directory') }
    if ($container.Config.Env -notcontains 'NVIDIA_DRIVER_CAPABILITIES=compute,utility') { $problems.Add('GPU environment') }
    $mounts = @($container.Mounts)
    if ($mounts.Count -ne 1 -or $mounts[0].Type -ne 'bind' -or
        $mounts[0].Destination -ne '/workspace' -or -not $mounts[0].RW -or
        [IO.Path]::GetFullPath($mounts[0].Source) -ne $projectPath) { $problems.Add('bind mount') }
    $ports = @($container.HostConfig.PortBindings.PSObject.Properties)
    $bindings = @($container.HostConfig.PortBindings.'8080/tcp')
    if ($ports.Count -ne 1 -or $bindings.Count -ne 1 -or
        $bindings[0].HostIp -ne '127.0.0.1' -or $bindings[0].HostPort -ne '43118') { $problems.Add('published port') }
    $gpu = @($container.HostConfig.DeviceRequests)
    if ($gpu.Count -ne 1 -or $gpu[0].Count -ne -1 -or
        @($gpu[0].Capabilities[0]) -notcontains 'gpu' -or $gpu[0].DeviceIDs) { $problems.Add('GPU access') }
    if ($container.HostConfig.RestartPolicy.Name -ne 'no') { $problems.Add('restart policy') }
    if (-not $container.HostConfig.Init -or $container.Config.StopTimeout -ne 15) { $problems.Add('init/stop timeout') }
    if ($container.HostConfig.Privileged -or $container.HostConfig.Devices -or
        $container.HostConfig.CapAdd -or $container.HostConfig.NetworkMode -ne 'bridge') { $problems.Add('extra privileges/devices/network') }
    if ($problems.Count -gt 0) {
        throw "Preserved conflicting container $($container.Id). Differences: $($problems -join ', '). No changes made."
    }
}

$container = Read-Container
if ($container) { Assert-Container $container }
if ($Action -eq 'up') {
    if (-not $container) {
        Write-Host 'Creating helicopter-dev; Debian 12 slim; /bin/bash /workspace/run.sh; project -> /workspace (rw); no volumes; 127.0.0.1:43118 -> 8080; restart=no; all NVIDIA GPUs; no extra devices.'
        Invoke-Docker @('build', '--tag', $imageName, $projectPath)
        Invoke-Docker @('container', 'create', '--name', $containerName,
            '--label', 'io.waajacu.managed=true', '--label', 'io.waajacu.project=helicopter',
            '--label', 'io.waajacu.configuration=1', '--mount', "type=bind,source=$projectPath,target=/workspace",
            '--publish', '127.0.0.1:43118:8080', '--gpus', 'all',
            '--env', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility', '--init', '--stop-timeout', '15',
            '--restart', 'no', $imageName)
        $container = Read-Container
        Assert-Container $container
    }
    if (-not $container.State.Running) { Invoke-Docker @('container', 'start', $container.Id) }
    $ready = $false
    # Startup builds and verifies the nominal offline H-infinity design.
    for ($attempt = 0; $attempt -lt 1800; $attempt++) {
        try {
            $health = Invoke-RestMethod 'http://127.0.0.1:43118/healthz' -TimeoutSec 2
            if ($health.ok -eq $true -or $health.status -eq 'ok') { $ready = $true; break }
        } catch { }
        $current = Read-Container
        if (-not $current.State.Running) {
            Invoke-Docker @('container', 'logs', '--tail', '60', $container.Id)
            throw 'Helicopter startup failed. The container and its data have been preserved.'
        }
        Start-Sleep -Milliseconds 1000
    }
    if (-not $ready) { throw 'Viewer did not become healthy in time; inspect logs. Container preserved.' }
    Write-Host 'Helicopter viewer: http://127.0.0.1:43118/'
} elseif (-not $container) {
    if ($Action -eq 'status') { Write-Host 'helicopter-dev does not exist.'; exit 0 }
    throw 'helicopter-dev does not exist. Run .\helicopter.ps1 up first.'
} elseif ($Action -eq 'status') {
    $container | Select-Object Id, Image, @{n='Status';e={$_.State.Status}}, @{n='Health';e={$_.State.Health.Status}}, Mounts
} elseif ($Action -eq 'stop') {
    Invoke-Docker @('container', 'stop', $container.Id)
} elseif ($Action -eq 'logs') {
    Invoke-Docker @('container', 'logs', '--tail', '100', $container.Id)
} elseif ($Action -eq 'test') {
    Invoke-Docker @('exec', $container.Id, 'cmake', '--build', '/workspace/build/native', '--parallel', '2')
    Invoke-Docker @('exec', $container.Id, '/workspace/build/native/helicopter-thermodynamics')
    Invoke-Docker @('exec', $container.Id, '/workspace/build/native/helicopter-mass-estimator-test')
    Invoke-Docker @('exec', '-w', '/workspace/build/native', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1', $container.Id, '/workspace/build/native/helicopter-hinf-test')
    Invoke-Docker @('exec', '-w', '/workspace/build/native', '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1', $container.Id, '/workspace/build/native/helicopter-evaluate', '--output', '/workspace/artifacts/evaluation.json')
} elseif ($Action -eq 'check') {
    & (Join-Path $projectPath 'tests/check-service.ps1')
} elseif ($Action -eq 'gpu') {
    Invoke-Docker @('exec', $container.Id, '/workspace/build/native/helicopter-gpu-check')
}
