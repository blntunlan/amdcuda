param(
    [ValidateSet('gpu','cpu')][string]$Mode = 'gpu',
    [switch]$WithDemo,
    [string]$Workspace = (Join-Path $HOME 'kimodo-rx6650')
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
function Fail($message) { throw $message }
function RunNative([string]$command, [string[]]$params) {
    & $command @params
    if ($LASTEXITCODE -ne 0) { Fail "$command failed with exit code $LASTEXITCODE" }
}
if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') { Fail 'This launcher is for native Windows.' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail 'Install Git for Windows first: https://git-scm.com/downloads/win' }
New-Item -ItemType Directory -Path $Workspace -Force | Out-Null
$repo = Join-Path $Workspace 'kimodo'
if (-not (Test-Path (Join-Path $repo 'pyproject.toml'))) {
    if (Test-Path $repo) { Fail "$repo exists but is not a Kimodo source tree. Inspect it manually." }
    RunNative git @('clone','--depth','1','https://github.com/nv-tlabs/kimodo.git',$repo)
}
$venv = Join-Path $Workspace '.venv'
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        RunNative py @('-3.11','-m','venv',$venv)
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        RunNative python @('-m','venv',$venv)
    } else { Fail 'Install Python 3.11 x64 first.' }
}
$python = Join-Path $venv 'Scripts\python.exe'
RunNative $python @('-m','pip','install','--upgrade','pip','setuptools','wheel')
if ($Mode -eq 'gpu') {
    Write-Host 'Installing experimental TheRock ROCm PyTorch for gfx1032. This may fail on your driver or Python build.'
    RunNative $python @('-m','pip','install','--index-url','https://stable.repo.amd.com/rocm/whl-next/','torch[device-gfx1032]')
} else {
    Write-Warning 'CPU-only mode: slow, and 16GB RAM may be insufficient for the text encoder.'
    RunNative $python @('-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch')
}
# Build-free first pass: optional motion correction native extension is omitted, hence --no-postprocess.
$env:SKIP_MOTION_CORRECTION_IN_SETUP = '1'
try {
    Push-Location $repo
    try {
        if ($WithDemo) { RunNative $python @('-m','pip','install','-e','.[all]') }
        else { RunNative $python @('-m','pip','install','-e','.[soma]') }
        RunNative $python @('-m','pip','install','huggingface_hub')
    } finally { Pop-Location }
} finally { Remove-Item Env:\SKIP_MOTION_CORRECTION_IN_SETUP -ErrorAction SilentlyContinue }
Write-Host "Installed in: $Workspace"
Write-Host 'Next: .\doctor.ps1 -Workspace <same path> ; then log in using .venv\Scripts\hf.exe auth login'
