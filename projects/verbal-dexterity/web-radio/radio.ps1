[CmdletBinding()]
param([ValidateSet('up','setup','start','status','stop','logs','test','check','gpu','studio-stop','studio-start','services-stop','backup')][string]$Action = 'up')
$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath($PSScriptRoot)
$contract = Get-Content -Raw -LiteralPath (Join-Path $projectPath 'infra/container.json') | ConvertFrom-Json
$containerName = $contract.name
function Invoke-Docker {
    param([string[]]$DockerArgs)
    & docker @DockerArgs
    if ($LASTEXITCODE -ne 0) { throw "Docker failed ($LASTEXITCODE): $($DockerArgs[0..([Math]::Min(2,$DockerArgs.Length-1))] -join ' ')" }
}
function Read-Container {
    $names = @(Invoke-Docker @('container','ls','-a','--filter',"name=^/$containerName$",'--format','{{.Names}}'))
    if ($names -notcontains $containerName) { return $null }
    return @(Invoke-Docker @('container','inspect',$containerName) | ConvertFrom-Json)[0]
}
function Assert-Container($item) {
    $issues = [Collections.Generic.List[string]]::new()
    foreach ($label in $contract.labels.PSObject.Properties) {
        if ($item.Config.Labels.($label.Name) -ne $label.Value) { $issues.Add("label $($label.Name)") }
    }
    if ($item.Config.Labels.'io.waajacu.project-root' -ne $projectPath) { $issues.Add('project root label') }
    if ($item.Config.Image -ne $contract.base_image) { $issues.Add('base image reference') }
    $imageId = @(Invoke-Docker @('image','inspect',$contract.base_image,'--format','{{.Id}}'))[0]
    if ($item.Image -ne $imageId) { $issues.Add('immutable base image ID') }
    if (($item.Config.Cmd -join '|') -ne ($contract.command -join '|')) { $issues.Add('command') }
    if ($item.Config.Entrypoint) { $issues.Add('entrypoint') }
    if ($item.Config.WorkingDir -ne '/workspace') { $issues.Add('working directory') }
    $mounts = @($item.Mounts)
    if ($mounts.Count -ne 1 -or $mounts[0].Type -ne 'bind' -or
        $mounts[0].Destination -ne '/workspace' -or -not $mounts[0].RW -or
        [IO.Path]::GetFullPath($mounts[0].Source) -ne $projectPath) { $issues.Add('sole project bind mount') }
    $bindings = @($item.HostConfig.PortBindings.PSObject.Properties)
    if ($bindings.Count -ne 2) { $issues.Add('port count') }
    foreach ($port in $contract.ports) {
        $binding = @($item.HostConfig.PortBindings.("$($port.container_port)/tcp"))
        if ($binding.Count -ne 1 -or $binding[0].HostIp -ne $port.host -or
            $binding[0].HostPort -ne [string]$port.host_port) { $issues.Add("matching port $($port.host_port)") }
    }
    $gpu = @($item.HostConfig.DeviceRequests)
    if ($gpu.Count -ne 1 -or $gpu[0].Count -ne -1 -or $gpu[0].DeviceIDs -or
        @($gpu[0].Capabilities[0]) -notcontains 'gpu') { $issues.Add('all-GPU request') }
    if ($item.Config.Env -notcontains 'NVIDIA_DRIVER_CAPABILITIES=compute,utility') { $issues.Add('GPU capabilities') }
    if ($item.HostConfig.RestartPolicy.Name -ne 'no' -or -not $item.HostConfig.Init -or
        $item.Config.StopTimeout -ne 20) { $issues.Add('lifecycle policy') }
    if ($item.HostConfig.Privileged -or $item.HostConfig.Devices -or $item.HostConfig.CapAdd -or
        $item.HostConfig.NetworkMode -ne 'bridge') { $issues.Add('privilege/device/network configuration') }
    if ($issues.Count) { throw "Preserved conflicting container $($item.Id): $($issues -join ', ')." }
}
function Assert-FreePorts {
    $ids = @(Invoke-Docker @('container','ls','-aq'))
    if ($ids.Count) {
        $allItems = @(Invoke-Docker (@('container','inspect') + $ids) | ConvertFrom-Json)
        foreach ($other in $allItems) {
            foreach ($property in $other.HostConfig.PortBindings.PSObject.Properties) {
                foreach ($binding in $property.Value) {
                    if ([int]$binding.HostPort -in @(43120,43121)) {
                        throw "Port $($binding.HostPort) is reserved by $($other.Name). Container preserved."
                    }
                }
            }
        }
    }
    foreach ($portNumber in @(43120,43121)) {
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$portNumber)
        try { $listener.Start() } finally { $listener.Stop() }
    }
}
$item = Read-Container
if ($item) { Assert-Container $item }
if ($Action -eq 'up') {
    if (-not $item) {
        Assert-FreePorts
        $null = Invoke-Docker @('image','inspect',$contract.base_image,'--format','{{.Id}}')
        Write-Host 'Creating radio-waajacu: pinned Debian13 slim; command /bin/bash /workspace/infra/container-entry.sh; sole project bind -> /workspace (rw); no named volumes; loopback 43120:43120,43121:43121; restart=no; init; all NVIDIA GPUs compute,utility; bridge; no extra devices.'
        $argsList = @('container','create','--name',$containerName,'--workdir','/workspace',
            '--mount',"type=bind,source=$projectPath,target=/workspace",
            '--publish','127.0.0.1:43120:43120','--publish','127.0.0.1:43121:43121',
            '--gpus','all','--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
            '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH=/workspace',
            '--env','WEB_RADIO_DATA=/workspace/.runtime','--network','bridge',
            '--init','--stop-timeout','20','--restart','no',
            '--label','io.waajacu.managed=true','--label','io.waajacu.project=web-radio',
            '--label','io.waajacu.configuration=1','--label',"io.waajacu.project-root=$projectPath",
            $contract.base_image) + @($contract.command)
        Invoke-Docker $argsList
        $item = Read-Container
        Assert-Container $item
    }
    if (-not $item.State.Running) { Invoke-Docker @('container','start',$item.Id) }
    $ready = $false
    for ($attempt = 0; $attempt -lt 300; $attempt++) {
        & docker exec $item.Id /bin/bash -c 'test -f /opt/web-radio/ready' 2>$null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
        $current = Read-Container
        if (-not $current.State.Running) {
            Invoke-Docker @('container','logs','--tail','40',$item.Id)
            throw 'Setup exited; container and data preserved.'
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw 'Setup is still running; container preserved. Inspect logs, then rerun up.' }
    Invoke-Docker @('exec',$item.Id,'/bin/bash','/workspace/run.sh','start')
    Write-Host 'Radio: http://127.0.0.1:43120/   Direct station: http://127.0.0.1:43121/'
} elseif (-not $item) {
    if ($Action -eq 'status') { Write-Host 'radio-waajacu does not exist.'; exit 0 }
    throw 'Run .\radio.ps1 up first.'
} elseif ($Action -eq 'status') {
    $item | Select-Object Id,Image,@{n='Status';e={$_.State.Status}},Mounts
    if ($item.State.Running) { Invoke-Docker @('exec',$item.Id,'/bin/bash','-c','if test -f /opt/web-radio/ready; then echo "Environment: ready"; else echo "Environment: setup"; fi; if test -f /workspace/run.sh; then /bin/bash /workspace/run.sh status; fi') }
} elseif ($Action -eq 'stop') {
    if ($item.State.Running) {
        & docker exec $item.Id /bin/bash -c 'test -f /opt/web-radio/ready' 2>$null
        if ($LASTEXITCODE -eq 0) { Invoke-Docker @('exec',$item.Id,'/bin/bash','/workspace/run.sh','stop') }
    }
    Invoke-Docker @('container','stop',$item.Id)
} elseif ($Action -eq 'logs') {
    if ($item.State.Running) {
        & docker exec $item.Id /bin/bash -c 'test -f /workspace/.runtime/control/studio.log' 2>$null
        if ($LASTEXITCODE -eq 0) { Invoke-Docker @('exec',$item.Id,'/bin/bash','/workspace/run.sh','logs') }
        else { Invoke-Docker @('container','logs','--tail','80',$item.Id) }
    } else { Invoke-Docker @('container','logs','--tail','80',$item.Id) }
} elseif ($Action -eq 'setup') {
    Invoke-Docker @('exec',$item.Id,'/bin/bash','/workspace/setup.sh')
} elseif ($Action -eq 'gpu') {
    Invoke-Docker @('exec',$item.Id,'nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv')
} else {
    Invoke-Docker @('exec',$item.Id,'/bin/bash','/workspace/run.sh',$Action)
}
