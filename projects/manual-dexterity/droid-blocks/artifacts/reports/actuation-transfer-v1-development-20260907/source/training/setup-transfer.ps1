[CmdletBinding()]
param(
    [string]$BasePython = 'C:/Users/santi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvRoot = Join-Path $projectRoot 'build/transfer-venv'
$venvPython = Join-Path $venvRoot 'Scripts/python.exe'
$setupEvidence = Join-Path $projectRoot 'build/actuation-transfer-reference/setup'
$requirements = Join-Path $PSScriptRoot 'requirements-transfer.txt'

if (-not (Test-Path -LiteralPath $BasePython -PathType Leaf)) { throw 'The declared Python runtime is unavailable.' }
$runtimeVersion = & $BasePython -c 'import sys; print("%d.%d" % sys.version_info[:2])'
if ($LASTEXITCODE -ne 0 -or $runtimeVersion -ne '3.12') { throw 'This pinned Windows CPU setup requires Python 3.12.' }

if (-not (Test-Path -LiteralPath $venvRoot)) {
    & $BasePython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Creating the project-local virtual environment failed; existing state is preserved.' }
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) { throw 'The existing environment is incomplete; it was not replaced.' }
$prefix = & $venvPython -I -c 'import sys; print(sys.prefix)'
if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath($prefix) -ne [IO.Path]::GetFullPath($venvRoot)) {
    throw 'Interpreter prefix differs from the intended project-local environment.'
}
New-Item -ItemType Directory -Path $setupEvidence -Force | Out-Null

# One vendor registry for the explicit CPU wheel; dependencies come from official PyPI.
& $venvPython -I -m pip --isolated --disable-pip-version-check install --timeout 120 --retries 5 --no-cache-dir --only-binary=:all: --no-deps --index-url 'https://download.pytorch.org/whl/cpu' 'torch==2.8.0+cpu' --report (Join-Path $setupEvidence 'torch-install.json')
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed; partial environment and evidence are preserved.' }
& $venvPython -I -m pip --isolated --disable-pip-version-check install --timeout 120 --retries 5 --no-cache-dir --only-binary=:all: --index-url 'https://pypi.org/simple' --requirement $requirements --report (Join-Path $setupEvidence 'pypi-install.json')
if ($LASTEXITCODE -ne 0) { throw 'Pinned training dependency installation failed; partial state is preserved.' }
& $venvPython -I -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependency consistency check failed.' }

$verification = @'
import importlib.metadata as md
import json
import platform
import sys
import torch
expected = {'stable-baselines3':'2.9.0','gymnasium':'1.2.3','torch':'2.8.0+cpu','numpy':'2.2.6'}
for package, version in expected.items():
    assert md.version(package) == version, (package, md.version(package), version)
assert torch.version.cuda is None, 'The installed Torch wheel is not CPU-only'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert (torch.ones(2) + 1).tolist() == [2.0, 2.0]
packages = []
for dist in sorted(md.distributions(), key=lambda item: item.metadata['Name'].lower()):
    meta = dist.metadata
    packages.append({'name':meta['Name'],'version':dist.version,'license_expression':meta.get('License-Expression'),
        'license':meta.get('License'),'license_files':meta.get_all('License-File') or [],
        'license_classifiers':[value for value in meta.get_all('Classifier',[]) if value.startswith('License ::')]})
print(json.dumps({'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'platform':platform.platform(),
    'cpu_only':True,'torch_threads':torch.get_num_threads(),'packages':packages}, indent=2))
'@
$installed = & $venvPython -I -c $verification
if ($LASTEXITCODE -ne 0) { throw 'CPU training runtime verification failed.' }
$installed | Set-Content -LiteralPath (Join-Path $setupEvidence 'installed-packages.json') -Encoding utf8
& $venvPython -I -m pip --isolated --disable-pip-version-check freeze --all | Set-Content -LiteralPath (Join-Path $setupEvidence 'installed-freeze.txt') -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'Recording the installed versions failed.' }
Write-Output ('Training interpreter: ' + $venvPython)
Write-Output ('Setup evidence: ' + $setupEvidence)
