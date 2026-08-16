@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
    echo Environment not found. Run install.bat first.
    pause
    exit /b 1
)
start "Child Face Finder" "venv\Scripts\pythonw.exe" "app\app.py"
exit /b 0
