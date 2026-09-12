@echo off
setlocal

echo Checking Python 3.14 installation...
py -3.14 --version >nul 2>&1
if %errorlevel%==0 (
  echo Python 3.14 detected.
  py -3.14 --version
  echo Ready. You can now run launch_clinical_app.bat
  pause
  exit /b 0
)

echo.
echo Python 3.14.3 is required and was not found.
echo.
echo Setup steps:
echo 1) Download Python 3.14.3 installer from python.org.
echo 2) Run installer and enable these options:
echo    - Add Python to PATH
echo    - Install launcher for all users (recommended)
echo 3) Restart terminal.
echo 4) Verify using: py -3.14 --version
echo.
start "" "https://www.python.org/downloads/release/python-3143/"
echo Opened download page in browser.
pause
exit /b 1
