# Kimodo Studio: gerekirse derler ve http://127.0.0.1:8787 adresinde başlatır.
param([switch]$Build)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $root 'server\target\release\kimodo-studio.exe'
$url = 'http://127.0.0.1:8787'

function Test-StudioReady {
    try {
        $r = Invoke-WebRequest -Uri "$url/api/status" -UseBasicParsing -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch { return $false }
}

if (Test-StudioReady) {
    Write-Host "Kimodo Studio zaten çalışıyor: $url"
    Start-Process $url
    exit 0
}

if ($Build -or -not (Test-Path $exe)) {
    Push-Location (Join-Path $root 'server')
    try {
        cargo build --release
        if ($LASTEXITCODE -ne 0) { throw 'cargo build başarısız' }
    } finally { Pop-Location }
}

$env:KIMODO_STUDIO_ROOT = $root
$openJob = Start-Job -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 40; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "$u/api/status" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { Start-Process $u; break }
        } catch {}
        Start-Sleep -Milliseconds 300
    }
} -ArgumentList $url

try {
    & $exe
} finally {
    Remove-Job $openJob -Force -ErrorAction SilentlyContinue
}
