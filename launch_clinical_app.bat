@echo off
setlocal

cd /d "%~dp0"

set "PYCMD="
py -3.14 --version >nul 2>&1
if %errorlevel%==0 set "PYCMD=py -3.14"

if not defined PYCMD (
  where python3.14.exe >nul 2>&1
  if %errorlevel%==0 set "PYCMD=python3.14.exe"
)

if not defined PYCMD (
  echo [ERROR] Python 3.14 was not found.
  echo This project targets Python 3.14.3.
  echo.
  echo Run setup_python_3_14_3.bat first, then retry launch_clinical_app.bat.
  pause
  exit /b 1
)

for /f "tokens=2" %%V in ('%PYCMD% --version') do set "PYVER=%%V"
echo Using Python %PYVER%

set "BACKEND_DIR=%~dp0"
set "BACKEND_FILE=%BACKEND_DIR%clinical_blindspot.py"
set "FRONTEND_DIR=%~dp0frontend"
set "BACKEND_RUNNING=0"
set "FRONTEND_RUNNING=0"

if not exist "%BACKEND_FILE%" (
  echo [ERROR] Backend file not found: %BACKEND_FILE%
  pause
  exit /b 1
)

if not exist "%FRONTEND_DIR%\index.html" (
  echo [ERROR] Frontend index not found: %FRONTEND_DIR%\index.html
  pause
  exit /b 1
)

if not exist "%BACKEND_DIR%models\faster-whisper-base" (
  echo [INFO] Local Whisper model folder not found.
  echo [INFO] Backend will try online download on first run.
  echo [INFO] For offline PCs, run setup_whisper_model.bat on an online PC and copy models\faster-whisper-base.
)

echo Checking Python dependencies...
%PYCMD% -m pip install -r "%~dp0requirements.txt"
if not %errorlevel%==0 (
  echo [ERROR] Dependency installation failed.
  pause
  exit /b 1
)

for /f "tokens=*" %%A in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do set "BACKEND_RUNNING=1"
for /f "tokens=*" %%A in ('netstat -ano ^| findstr /R /C:":5500 .*LISTENING"') do set "FRONTEND_RUNNING=1"

if "%BACKEND_RUNNING%"=="1" (
  echo Backend already running on port 8000.
) else (
  echo Starting Clinical Blindspot backend on http://127.0.0.1:8000 ...
  start "Clinical Backend" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%BACKEND_DIR%'; & %PYCMD% clinical_blindspot.py --mode server --host 127.0.0.1 --port 8000"
)

if "%FRONTEND_RUNNING%"=="1" (
  echo Frontend server already running on port 5500.
) else (
  echo Starting frontend server on http://127.0.0.1:5500 ...
  start "Clinical Frontend" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location '%FRONTEND_DIR%'; & %PYCMD% -m http.server 5500"
)

echo Waiting for backend health check...
set "BACKEND_READY=0"
for /l %%I in (1,1,25) do (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-RestMethod -Method GET -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; if($r.status -eq 'ok'){ exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 (
    set "BACKEND_READY=1"
    goto :backend_ready
  )
  timeout /t 1 /nobreak >nul
)

:backend_ready
if not "%BACKEND_READY%"=="1" (
  echo.
  echo [ERROR] Backend did not become ready on http://127.0.0.1:8000
  echo Check the "Clinical Backend" terminal for startup errors.
  pause
  exit /b 1
)

echo Opening app in browser...
start "" "http://127.0.0.1:5500/index.html"

echo.
echo Clinical system launched.
echo Use stop_clinical_app.bat to stop both servers.
timeout /t 2 /nobreak >nul
exit /b 0
