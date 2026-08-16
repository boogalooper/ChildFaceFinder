@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_portable.ps1"
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo Portable build failed. See the messages above.
    pause
    exit /b %RC%
)
echo Portable build completed successfully.
pause
exit /b 0
