@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Child Face Finder installer

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

echo [Child Face Finder] Installing private Python environment...
echo Project: %CD%
echo System Python is not required.
echo.

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\install.ps1"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" goto :failed

rem Never report success unless the exact files used by run.bat exist.
if not exist "tools\uv\uv.exe" (
    echo.
    echo INSTALLATION CHECK FAILED: tools\uv\uv.exe is missing.
    goto :postcheck_failed
)
if not exist "venv\Scripts\python.exe" (
    echo.
    echo INSTALLATION CHECK FAILED: venv\Scripts\python.exe is missing.
    goto :postcheck_failed
)
if not exist "venv\Scripts\pythonw.exe" (
    echo.
    echo INSTALLATION CHECK FAILED: venv\Scripts\pythonw.exe is missing.
    goto :postcheck_failed
)

set "APP_EXPECTED_VENV=%CD%\venv"
"venv\Scripts\python.exe" -c "import os,pathlib,struct,sys; expected=pathlib.Path(os.environ['APP_EXPECTED_VENV']).resolve(); assert sys.version_info[:3]==(3,11,16); assert struct.calcsize('P')==8; assert pathlib.Path(sys.prefix).resolve()==expected"
set "RC=%ERRORLEVEL%"
set "APP_EXPECTED_VENV="
if not "%RC%"=="0" (
    echo.
    echo INSTALLATION CHECK FAILED: the new venv exists but did not pass the final launcher check.
    goto :postcheck_failed
)

echo.
echo ============================================================
echo Installation completed successfully.
echo Verified: %CD%\venv\Scripts\python.exe
echo Start the application with run.bat.
echo Full log: %CD%\childfacefinder_install.log
echo ============================================================
pause
exit /b 0

:postcheck_failed
echo The installer did NOT complete successfully.
echo Full log: %CD%\childfacefinder_install.log
echo Please send that log if the problem repeats.
pause
exit /b 1

:failed
echo.
echo Installation failed with exit code %RC%.
echo Full log: %CD%\childfacefinder_install.log
echo Please send that log if the cause is not clear above.
pause
exit /b %RC%
