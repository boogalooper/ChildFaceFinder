@echo off
setlocal EnableExtensions
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\install.ps1"
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo Installation failed. See the messages above.
    pause
    exit /b %RC%
)

echo Installation completed successfully.
echo Start the application with run.bat.
pause
exit /b 0
