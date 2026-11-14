# Explicit opt-in. Creates ONLY an isolated Python 3.12 venv; does not install packages.
[CmdletBinding()]
param([string]$EnvironmentDir = (Join-Path $PSScriptRoot '..\.venv-gfx1032'))
$ErrorActionPreference = 'Stop'
if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') {
    throw 'This environment creation helper targets Windows.'
}
$adapters = @(Get-CimInstance -ClassName Win32_VideoController)
$matches = @($adapters | Where-Object { $_.Name -match 'RX\s*6650\s*XT' })
if ($matches.Count -ne 1) { throw 'Expected exactly one RX 6650 XT in Windows device inventory.' }
$launcher = Get-Command py.exe -ErrorAction SilentlyContinue
if (-not $launcher) { throw 'Python 3.12 launcher (py.exe) missing. Install Python 3.12 manually.' }
if (Test-Path -LiteralPath $EnvironmentDir) { throw "Environment exists: $EnvironmentDir. Refusing to overwrite." }
& $launcher.Source -3.12 -m venv $EnvironmentDir
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
$venvPython = Join-Path $EnvironmentDir 'Scripts\python.exe'
Write-Host "Created local venv: $EnvironmentDir"
Write-Host 'Install TheRock ROCm PyTorch manually in this environment only:'
Write-Host "  & `"$venvPython`" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'torch[device-gfx1032]'"
Write-Host 'Then test:'
Write-Host "  python .\scripts\run-tests.py --tier all --gpu-python `"$venvPython`""
Write-Host 'Check TheRock/RELEASES.md for latest release constraints before installing.'
