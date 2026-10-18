[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$oldRoot = 'C:\Work\Universalis-Materiae-Dexteritas'
$newRoot = 'C:\Work\Natural-Dexterity'

function Test-ExactJunction($Item, [string]$Target) {
    return $null -ne $Item -and $Item.LinkType -eq 'Junction' -and
        [IO.Path]::GetFullPath($Item.Target).TrimEnd('\', '/') -ieq $Target
}

# Do not let this PowerShell process itself hold the source directory open.
Set-Location -LiteralPath 'C:\Work'
[Environment]::CurrentDirectory = 'C:\Work'
$old = Get-Item -LiteralPath $oldRoot -Force -ErrorAction SilentlyContinue
$new = Get-Item -LiteralPath $newRoot -Force -ErrorAction SilentlyContinue
if ((Test-ExactJunction $old $newRoot) -and $null -ne $new -and -not $new.LinkType) {
    Write-Output "Already complete: $newRoot; the old path is its compatibility junction."
    return
}
if ($null -eq $old -or $old.LinkType -or -not $old.PSIsContainer -or
    -not (Test-ExactJunction $new $oldRoot)) {
    throw 'Expected the original directory and the verified Natural-Dexterity junction. Nothing was changed.'
}
if ((Split-Path -Parent $old.FullName) -ne 'C:\Work' -or
    (Split-Path -Parent $new.FullName) -ne 'C:\Work') {
    throw 'Unexpected path outside C:\Work. Nothing was changed.'
}
$rootOwner = (Get-Acl -LiteralPath $oldRoot).Owner
$gitOwner = (Get-Acl -LiteralPath "$oldRoot\.git").Owner
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
if ($rootOwner -ine $identity -or $gitOwner -ine $identity) {
    throw "Run this as the existing workspace owner ($rootOwner). Ownership will not be changed."
}

# Delete only the verified junction itself, without traversing its target.
[IO.Directory]::Delete($newRoot)
try {
    Rename-Item -LiteralPath $oldRoot -NewName 'Natural-Dexterity'
} catch {
    if (-not (Test-Path -LiteralPath $newRoot)) {
        New-Item -ItemType Junction -Path $newRoot -Target $oldRoot | Out-Null
    }
    throw "Folder move failed; the new-name junction was restored. Close Codex and other processes using this directory, then retry. $($_.Exception.Message)"
}
try {
    New-Item -ItemType Junction -Path $oldRoot -Target $newRoot | Out-Null
} catch {
    Rename-Item -LiteralPath $newRoot -NewName 'Universalis-Materiae-Dexteritas'
    New-Item -ItemType Junction -Path $newRoot -Target $oldRoot | Out-Null
    throw 'Could not create the compatibility junction; the original directory layout was restored.'
}
if ((Get-Acl -LiteralPath $newRoot).Owner -ine $rootOwner -or
    (Get-Acl -LiteralPath "$newRoot\.git").Owner -ine $gitOwner) {
    throw 'Unexpected ownership change. Inspect the directory; no ownership change is authorized by this script.'
}
& git -C $newRoot status --short
if ($LASTEXITCODE -ne 0) { throw 'Directory moved, but Git status failed. Inspect the preserved checkout.' }
Write-Output "Complete: $newRoot; $oldRoot is its compatibility junction."
