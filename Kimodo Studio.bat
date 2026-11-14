@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0studio\start.ps1"
if errorlevel 1 (
  echo.
  echo Kimodo Studio baslatilamadi. Yukaridaki hatayi kontrol edin.
  pause
)
