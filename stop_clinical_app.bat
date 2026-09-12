@echo off
setlocal

echo Stopping services on ports 8000 and 5500...

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
  echo Stopping process on 8000: PID %%P
  taskkill /PID %%P /F >nul 2>&1
)

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":5500 .*LISTENING"') do (
  echo Stopping process on 5500: PID %%P
  taskkill /PID %%P /F >nul 2>&1
)

echo Done.
timeout /t 1 /nobreak >nul
exit /b 0
