[CmdletBinding()]
param(
    [ValidateSet('up','status','stop','logs','exec','build')][string]$Action = 'status',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs
)
$ErrorActionPreference = 'Stop'
$containerName = 'humanoid-mujoco-native'
$imageName = 'humanoid-mujoco:dev'
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$expectedRoot = 'C:\Work\Natural-Dexterity\projects\humanoid'
if ($projectRoot -ne $expectedRoot) { throw "This definition is for $expectedRoot. Confirm a new definition before using another location." }
$configText = [ordered]@{
    schema=3; container=$containerName; root=$projectRoot; image=$imageName; command=@('bash','/workspace/run.sh')
    mount='/workspace'; port='127.0.0.1:43871:8080'; restart='no'; gpu='all'
    capabilities='compute,utility,graphics'; init=$true; shm=1073741824; stopTimeout=20
    dockerfile=(Get-FileHash "$PSScriptRoot/Dockerfile" -Algorithm SHA256).Hash
    setup=(Get-FileHash "$PSScriptRoot/setup.sh" -Algorithm SHA256).Hash
    dependencies=(Get-FileHash "$PSScriptRoot/runtime/dependencies.sha256" -Algorithm SHA256).Hash
} | ConvertTo-Json -Compress
$configDigest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($configText))).ToLowerInvariant()

function Invoke-Docker {
    param([string[]]$DockerArgs)
    & docker @DockerArgs
    if ($LASTEXITCODE -ne 0) { throw "Docker failed (exit $LASTEXITCODE): $($DockerArgs[0])" }
}
function Get-Container {
    $names = & docker container ls -a --format '{{.Names}}'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Docker; no mutation performed.' }
    if (@($names) -notcontains $containerName) { return $null }
    $json = & docker container inspect $containerName
    if ($LASTEXITCODE -ne 0) { throw 'Container inspection failed.' }
    return @($json | ConvertFrom-Json)[0]
}
function Assert-Contract($item) {
    $problems = [Collections.Generic.List[string]]::new()
    if ($item.Config.Labels.'io.waajacu.managed' -ne 'true' -or $item.Config.Labels.'io.waajacu.project' -ne 'humanoid') { $problems.Add('ownership labels') }
    if ($item.Config.Labels.'io.waajacu.configuration' -ne $configDigest) { $problems.Add('configuration digest') }
    if ($item.Config.Image -ne $imageName) { $problems.Add('image name') }
    if (($item.Config.Cmd -join '|') -ne 'bash|/workspace/run.sh' -or $item.Config.Entrypoint) { $problems.Add('startup command') }
    if ($item.Config.WorkingDir -ne '/workspace') { $problems.Add('working directory') }
    if (@($item.Mounts).Count -ne 1) { $problems.Add('mount count / unexpected volume') }
    else {
        $mount = $item.Mounts[0]
        $source = $mount.Source.Replace('/run/desktop/mnt/host/c/', 'C:/').Replace('\','/').TrimEnd('/')
        if ($mount.Type -ne 'bind' -or $mount.Destination -ne '/workspace' -or -not $mount.RW -or $source -ine $projectRoot.Replace('\','/')) { $problems.Add('project bind mount') }
    }
    $ports = $item.HostConfig.PortBindings
    $bindings = @($ports.'8080/tcp')
    if (@($ports.PSObject.Properties).Count -ne 1 -or $bindings.Count -ne 1 -or $bindings[0].HostIp -ne '127.0.0.1' -or $bindings[0].HostPort -ne '43871') { $problems.Add('port binding') }
    if ($item.HostConfig.RestartPolicy.Name -ne 'no') { $problems.Add('restart policy') }
    if (-not $item.HostConfig.Init -or $item.HostConfig.ShmSize -ne 1073741824 -or $item.Config.StopTimeout -ne 20) { $problems.Add('init/shared memory/stop timeout') }
    $requests = @($item.HostConfig.DeviceRequests)
    if ($requests.Count -ne 1 -or $requests[0].Count -ne -1 -or ($requests[0].Capabilities[0] -join ',') -ne 'gpu') { $problems.Add('GPU request') }
    if ($item.Config.Env -notcontains 'NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics') { $problems.Add('GPU capabilities') }
    if ($item.HostConfig.Privileged -or @($item.HostConfig.Devices).Count -gt 0) { $problems.Add('unexpected privileged/device access') }
    if ($problems.Count) { throw "Preserved container $($item.Id); configuration conflicts: $($problems -join ', '). No replacement or deletion is authorized." }
}

$existing = Get-Container
if ($existing) { Assert-Contract $existing }
switch ($Action) {
    'status' {
        if ($existing) { $existing | Select-Object Id,Image,State,Mounts,@{n='Ports';e={$_.HostConfig.PortBindings}},@{n='GPU';e={$_.HostConfig.DeviceRequests}} | ConvertTo-Json -Depth 12 }
        else { Write-Output "$containerName does not exist." }
    }
    'build' {
        if ($existing) { throw 'A managed container already exists. Source edits are bind-mounted; rebuilding/replacing its environment requires an explicit separate rebuild request.' }
        & "$PSScriptRoot/prepare-assets.ps1"
        Invoke-Docker -DockerArgs @('build','--tag',$imageName,'--label',"io.waajacu.configuration=$configDigest",$projectRoot)
    }
    'up' {
        if (-not $existing) {
            # No disposable probes, guessed volumes, or existing-container replacement.
            & "$PSScriptRoot/prepare-assets.ps1"
            Invoke-Docker -DockerArgs @('build','--tag',$imageName,'--label',"io.waajacu.configuration=$configDigest",$projectRoot)
            Invoke-Docker -DockerArgs @('container','create','--name',$containerName,
                '--label','io.waajacu.managed=true','--label','io.waajacu.project=humanoid',
                '--label',"io.waajacu.configuration=$configDigest",'--label',"io.waajacu.project-root=$projectRoot",
                '--mount',"type=bind,source=$projectRoot,target=/workspace",
                '--publish','127.0.0.1:43871:8080','--gpus','all',
                '--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics',
                '--init','--shm-size','1g','--stop-timeout','20','--restart','no',$imageName)
            $existing = Get-Container
            Assert-Contract $existing
        }
        Invoke-Docker -DockerArgs @('container','start',$existing.Id)
        $existing = Get-Container
        Assert-Contract $existing
        Write-Output "Container $($existing.Id). Viewer: http://127.0.0.1:43871/"
    }
    'stop' { if ($existing) { Invoke-Docker -DockerArgs @('container','stop',$existing.Id) } }
    'logs' { if (-not $existing) { throw 'Container does not exist.' }; Invoke-Docker -DockerArgs @('container','logs','--tail','100',$existing.Id) }
    'exec' {
        if (-not $existing -or -not $existing.State.Running) { throw 'Start the matching managed container first.' }
        if (-not $CommandArgs) { throw 'Pass the command to execute.' }
        Invoke-Docker -DockerArgs (@('container','exec',$existing.Id) + $CommandArgs)
    }
}
