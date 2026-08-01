@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project environment is missing.
  echo Run: py -3.12 -m venv --system-site-packages .venv
  pause
  exit /b 1
)
powershell -NoProfile -Command "try { $status = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/status' -TimeoutSec 2; if ($status.device) { Start-Process 'http://127.0.0.1:7860'; exit 0 } } catch {}; exit 1"
if not errorlevel 1 exit /b 0
".venv\Scripts\python.exe" -m vectorlearner.ui --open
if errorlevel 1 pause
