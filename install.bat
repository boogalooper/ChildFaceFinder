@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Always prefer 64-bit Windows PowerShell on 64-bit Windows.
rem When install.bat is started from a 32-bit host (for example 32-bit
rem Total Commander), %%SystemRoot%%\System32 is redirected to SysWOW64.
rem Sysnative bypasses that redirection and lets the installer see the
rem 64-bit NVIDIA driver tools such as nvidia-smi.exe.
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 (
    if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" (
        set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
    )
)
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\install.ps1"
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
