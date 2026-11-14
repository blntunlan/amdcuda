param(
    [string]$Prompt = 'A person walks forward.',
    [ValidateRange(1,10)][double]$DurationSec = 3,
    [ValidateRange(1,200)][int]$Steps = 20,
    [string]$Output = 'output\first_motion',
    [string]$Workspace = (Join-Path $HOME 'kimodo-rx6650'),
    [ValidateSet('gpu','cpu')][string]$Mode = 'gpu',
    [string]$TextEncoderUrl = '',
    [switch]$AllowLowRAM
)
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $Workspace '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Run install.ps1 before generation.' }
if ($Mode -eq 'gpu') {
    & $python (Join-Path $scriptRoot 'scripts\backend_probe.py') --report (Join-Path $Workspace 'reports\backend_probe.json')
    if ($LASTEXITCODE -ne 0) { throw 'gfx1032 GPU preflight failed; generation stopped. No silent CPU fallback.' }
}
$ramGiB = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)
if ([string]::IsNullOrWhiteSpace($TextEncoderUrl) -and $ramGiB -lt 20 -and -not $AllowLowRAM) {
    throw "Local Llama-8B encoder may need more than $ramGiB GiB RAM. Use -TextEncoderUrl <your encoder service URL> or explicitly -AllowLowRAM (risk of OOM)."
}
$env:TEXT_ENCODER_DEVICE='cpu'
if (-not [string]::IsNullOrWhiteSpace($TextEncoderUrl)) {
    $env:TEXT_ENCODER_URL=$TextEncoderUrl
    $env:TEXT_ENCODER_MODE='api'
} else { $env:TEXT_ENCODER_MODE='local' }
$repo = Join-Path $Workspace 'kimodo'
Push-Location $repo
try {
    $outputParent = Split-Path -Parent $Output
    if ($outputParent) { New-Item -ItemType Directory -Path $outputParent -Force | Out-Null }
    # Native MotionCorrection extension intentionally omitted in first pass.
    & $python -m kimodo.scripts.generate $Prompt --model Kimodo-SOMA-RP-v1.2 --duration "$DurationSec" --num_samples 1 --diffusion_steps $Steps --no-postprocess --bvh --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Kimodo generation failed, exit code $LASTEXITCODE; see terminal log." }
    Write-Host "Generated files under $repo\$Output (.npz and .bvh)"
} finally {
    Pop-Location
    Remove-Item Env:\TEXT_ENCODER_DEVICE -ErrorAction SilentlyContinue
    Remove-Item Env:\TEXT_ENCODER_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:\TEXT_ENCODER_URL -ErrorAction SilentlyContinue
}
