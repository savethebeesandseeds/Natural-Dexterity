# Preserve existing Docker mount contracts after the repository directory rename.
function Resolve-ContainerProjectPath {
    param([Parameter(Mandatory=$true)][string]$ProjectPath)

    $project = [IO.Path]::GetFullPath($ProjectPath).TrimEnd('\', '/')
    $legacyRoot = 'C:\Work\Universalis-Materiae-Dexteritas'
    $currentRoot = 'C:\Work\Natural-Dexterity'
    if ($project.StartsWith($currentRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        $current = Get-Item -LiteralPath $currentRoot -Force -ErrorAction SilentlyContinue
        $legacy = Get-Item -LiteralPath $legacyRoot -Force -ErrorAction SilentlyContinue
        $currentLinksToLegacy = $null -ne $current -and $current.LinkType -eq 'Junction' -and
            [IO.Path]::GetFullPath($current.Target).TrimEnd('\', '/') -ieq $legacyRoot
        $legacyLinksToCurrent = $null -ne $legacy -and $legacy.LinkType -eq 'Junction' -and
            [IO.Path]::GetFullPath($legacy.Target).TrimEnd('\', '/') -ieq $currentRoot
        if ($currentLinksToLegacy -or $legacyLinksToCurrent) {
            return $legacyRoot + $project.Substring($currentRoot.Length)
        }
    }
    return $project
}
