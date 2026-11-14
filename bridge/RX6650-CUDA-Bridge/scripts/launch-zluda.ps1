# Per-application ZLUDA launcher. Does not replace system DLLs.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ZludaExe,
    [Parameter(Mandatory=$true)][string]$AppExe,
    [string[]]$AppArgs = @()
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ZludaExe -PathType Leaf)) {
    throw "ZLUDA launcher not found: $ZludaExe"
}
if (-not (Test-Path -LiteralPath $AppExe -PathType Leaf)) {
    throw "Application not found: $AppExe"
}
$gpuName = @(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'RX\s*6650\s*XT' })
if ($gpuName.Count -eq 0) { throw 'RX 6650 XT not detected. Refusing to launch with this hardware-specific profile.' }
Write-Host 'Launching via ZLUDA. CUDA compatibility and GPU computation are NOT guaranteed.'
$zludaPath = (Resolve-Path -LiteralPath $ZludaExe).Path
$appPath = (Resolve-Path -LiteralPath $AppExe).Path
& $zludaPath -- $appPath @AppArgs
exit $LASTEXITCODE
