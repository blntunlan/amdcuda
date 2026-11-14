param([string]$Workspace = (Join-Path $HOME 'kimodo-rx6650'))
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $Workspace '.venv\Scripts\python.exe'
$reportDir = Join-Path $Workspace 'reports'
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
Write-Host '=== Windows detected graphics adapters ==='
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | Format-Table -AutoSize
$ramGiB = [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)
Write-Host "System RAM: $ramGiB GiB"
if ($ramGiB -lt 20) { Write-Warning 'Llama 3 8B text encoding may exhaust RAM. A remote encoder or more RAM may be needed.' }
if (-not (Test-Path $venvPython)) { throw 'Virtual environment missing. Run install.ps1 first.' }
& $venvPython (Join-Path $scriptRoot 'scripts\backend_probe.py') --report (Join-Path $reportDir 'backend_probe.json')
if ($LASTEXITCODE -ne 0) { Write-Warning 'GPU validation not passed; do not interpret as working AMD acceleration.' }
Write-Host "Saved report: $(Join-Path $reportDir 'backend_probe.json')"
