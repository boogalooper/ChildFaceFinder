@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0.."
title Child Face Finder console launcher

echo [Child Face Finder] Console mode starting...
echo [1/2] Checking private CPython 3.11.16 environment...
if not exist "venv\Scripts\python.exe" goto :not_installed

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%CD%\setup\repair_venv.ps1"
if errorlevel 1 goto :broken

echo [2/2] Starting application with console diagnostics...
"venv\Scripts\python.exe" "app\app.py"
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%

:not_installed
echo Private Python environment is not installed. Run install.bat once.
echo System Python is NOT required.
pause
exit /b 1

:broken
echo Private Python environment could not be repaired. Run install.bat once.
pause
exit /b 1
