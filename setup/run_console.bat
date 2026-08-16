@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if not exist "venv\Scripts\python.exe" (
    echo Environment not found. Run install.bat first.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" "app\app.py"
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
