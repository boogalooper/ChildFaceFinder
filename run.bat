@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Child Face Finder launcher

echo [Child Face Finder] Starting...
echo [1/3] Checking private CPython 3.11.16 environment...

if not exist "venv\Scripts\python.exe" goto :not_installed
if not exist "venv\Scripts\pythonw.exe" goto :not_installed
if not exist "setup\repair_venv.ps1" goto :not_installed

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\repair_venv.ps1"
if errorlevel 1 goto :broken

echo [2/3] Local Python environment OK.
echo [3/3] Opening Child Face Finder...
start "Child Face Finder" "venv\Scripts\pythonw.exe" "app\app.py"
if errorlevel 1 goto :launch_failed
echo Interface launch requested. This window will close automatically.
timeout /t 2 /nobreak >nul
exit /b 0

:not_installed
echo.
echo Private Python environment is not installed yet.
echo Run install.bat once. It will download uv and its own CPython 3.11.16 x64.
echo System Python is NOT required.
pause
exit /b 1

:broken
echo.
echo The private Python environment could not be checked or repaired.
echo Run install.bat once to rebuild it. System Python is NOT required.
pause
exit /b 1

:launch_failed
echo.
echo Child Face Finder could not be started.
pause
exit /b 1
