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
  echo [ERROR] Python 3.14 is required for this setup script.
  pause
  exit /b 1
)

echo Downloading Whisper base model to local folder models\faster-whisper-base ...
%PYCMD% -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-base', local_dir='models/faster-whisper-base', local_dir_use_symlinks=False); print('Model download complete.')"
if not %errorlevel%==0 (
  echo [ERROR] Model download failed. Check internet connection and DNS.
  pause
  exit /b 1
)

echo Local Whisper model is ready.
pause
exit /b 0
