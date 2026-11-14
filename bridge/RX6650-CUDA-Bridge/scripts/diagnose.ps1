# PowerShell 5.1+; read-only checks. Does not install drivers or change PATH.
[CmdletBinding()]
param(
    [string]$OutFile = (Join-Path $PSScriptRoot '..\reports\diagnostic.json')
)
$ErrorActionPreference = 'Stop'
$profilePath = Join-Path $PSScriptRoot '..\config\gfx1032.json'
$profile = Get-Content -Raw -LiteralPath $profilePath | ConvertFrom-Json
$os = Get-CimInstance -ClassName Win32_OperatingSystem
$gpus = @(Get-CimInstance -ClassName Win32_VideoController | ForEach-Object {
    [ordered]@{
        name = $_.Name
        driver_version = $_.DriverVersion
        pnp_device_id = $_.PNPDeviceID
        video_processor = $_.VideoProcessor
        adapter_ram_reported_bytes = $_.AdapterRAM
    }
})
$allowed = @($gpus | Where-Object { $_.name -match 'RX\s*6650\s*XT' })
$hipCmd = Get-Command hipconfig -ErrorAction SilentlyContinue
$compilerCmd = Get-Command hipcc -ErrorAction SilentlyContinue
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
$zludaCmd = Get-Command zluda.exe -ErrorAction SilentlyContinue
$hipPath = [Environment]::GetEnvironmentVariable('HIP_PATH', 'Process')
$paths = @(
    'C:\Program Files\AMD\ROCm\6.2',
    'C:\Program Files\AMD\ROCm\7.2'
)
$existingPaths = @($paths | Where-Object { Test-Path -LiteralPath $_ })
$report = [ordered]@{
    project = $profile.project
    profile_version = $profile.profile_version
    timestamp = (Get-Date).ToString('o')
    target_gpu = $profile.target_gpu
    target_arch = $profile.llvm_target
    target_detected = ($allowed.Count -gt 0)
    operating_system = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build = $os.BuildNumber
    }
    gpus = $gpus
    detected_tools = [ordered]@{
        hipconfig = if ($hipCmd) { $hipCmd.Source } else { $null }
        hipcc = if ($compilerCmd) { $compilerCmd.Source } else { $null }
        python = if ($pythonCmd) { $pythonCmd.Source } else { $null }
        zluda = if ($zludaCmd) { $zludaCmd.Source } else { $null }
    }
    hip_path_environment = $hipPath
    rocm_known_paths_found = $existingPaths
    caveats = @(
        'The Windows WMI AdapterRAM value may be truncated for 8GB cards.',
        'Detected HIP tools do not prove gfx1032 kernel or rocBLAS support.',
        'No GPU kernel was executed by this diagnostic.'
    )
}
$directory = Split-Path -Parent $OutFile
if (-not (Test-Path -LiteralPath $directory)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutFile -Encoding UTF8
Write-Host ("Target GPU detected: {0}" -f $report.target_detected)
Write-Host ("Diagnostic saved to: {0}" -f (Resolve-Path -LiteralPath $OutFile).Path)
if (-not $report.target_detected) {
    Write-Warning 'Expected Radeon RX 6650 XT was not found. Do not apply target-specific fixes yet.'
    exit 2
}
