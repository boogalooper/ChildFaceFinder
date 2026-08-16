$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$SetupDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $SetupDir
$AppDir = Join-Path $ProjectDir 'app'
$ModelsDir = Join-Path $ProjectDir 'models'
$VenvPython = Join-Path $ProjectDir 'venv\Scripts\python.exe'
$ManagedPythonRoot = Join-Path $ProjectDir 'tools\python'
$OutputRoot = Join-Path $ProjectDir 'portable'
$PortableDir = Join-Path $OutputRoot 'ChildFaceFinder_Portable'
$PortableZip = Join-Path $OutputRoot 'ChildFaceFinder_Portable.zip'

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Write-AsciiCrlf {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $Normalized = ($Text -replace "`r?`n", "`r`n")
    [System.IO.File]::WriteAllText($Path, $Normalized, [System.Text.Encoding]::ASCII)
}

function Copy-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
    }
}

Write-Host '============================================================'
Write-Host 'Child Face Finder - portable builder'
Write-Host 'Creates a self-contained Windows x64 folder and ZIP.'
Write-Host 'Python/CUDA Toolkit are not required on the target PC.'
Write-Host 'A compatible NVIDIA driver is still required for GPU mode.'
Write-Host '============================================================'
Write-Host

try {
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw 'Installed environment not found. Run install.bat successfully before building portable.'
    }
    if (-not (Test-Path -LiteralPath $ModelsDir -PathType Container)) {
        throw 'Models folder not found. Run install.bat successfully before building portable.'
    }

    Write-Host '[1/7] Locating managed CPython and installed packages...'
    $BasePythonExe = [string]((& $VenvPython -c 'import sys; print(sys._base_executable)' 2>$null | Select-Object -First 1))
    $BasePythonExe = $BasePythonExe.Trim()
    if (-not $BasePythonExe -or -not (Test-Path -LiteralPath $BasePythonExe -PathType Leaf)) {
        throw 'Could not locate the managed base Python used by the venv.'
    }
    $BasePythonDir = Split-Path -Parent $BasePythonExe

    if (-not (Test-Path -LiteralPath $ManagedPythonRoot -PathType Container)) {
        throw 'Managed Python folder tools\python was not found.'
    }
    $ManagedRootResolved = (Resolve-Path -LiteralPath $ManagedPythonRoot).Path.TrimEnd('\')
    $BaseDirResolved = (Resolve-Path -LiteralPath $BasePythonDir).Path.TrimEnd('\')
    if (-not $BaseDirResolved.StartsWith($ManagedRootResolved + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "The venv base Python is outside tools\python and will not be packaged: $BaseDirResolved"
    }

    $VenvSitePackages = [string]((& $VenvPython -c 'import site; print(site.getsitepackages()[0])' 2>$null | Select-Object -First 1))
    $VenvSitePackages = $VenvSitePackages.Trim()
    if (-not $VenvSitePackages -or -not (Test-Path -LiteralPath $VenvSitePackages -PathType Container)) {
        throw 'Could not locate venv site-packages.'
    }

    Write-Host "Managed Python: $BasePythonDir"
    Write-Host "Packages:       $VenvSitePackages"

    Write-Host '[2/7] Preparing portable folder...'
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    if (Test-Path -LiteralPath $PortableDir) {
        Remove-Item -LiteralPath $PortableDir -Recurse -Force
    }
    if (Test-Path -LiteralPath $PortableZip) {
        Remove-Item -LiteralPath $PortableZip -Force
    }
    New-Item -ItemType Directory -Path $PortableDir -Force | Out-Null

    Write-Host '[3/7] Copying application, models and managed Python...'
    Copy-DirectoryContents -Source $AppDir -Destination (Join-Path $PortableDir 'app')
    Copy-DirectoryContents -Source $ModelsDir -Destination (Join-Path $PortableDir 'models')
    Copy-DirectoryContents -Source $BasePythonDir -Destination (Join-Path $PortableDir 'python')

    $PortablePython = Join-Path $PortableDir 'python\python.exe'
    $PortablePythonW = Join-Path $PortableDir 'python\pythonw.exe'
    if (-not (Test-Path -LiteralPath $PortablePython -PathType Leaf)) {
        throw 'Portable python.exe was not copied as expected.'
    }
    if (-not (Test-Path -LiteralPath $PortablePythonW -PathType Leaf)) {
        throw 'Portable pythonw.exe was not copied as expected.'
    }

    Write-Host '[4/7] Copying installed third-party packages...'
    $PortableRuntime = Join-Path $PortableDir 'runtime'
    $PortableSitePackages = Join-Path $PortableRuntime 'site-packages'
    Copy-DirectoryContents -Source $VenvSitePackages -Destination $PortableSitePackages

    # Keep the portable folder clean and avoid bytecode containing old venv paths.
    foreach ($CleanRoot in @((Join-Path $PortableDir 'app'), $PortableSitePackages)) {
        Get-ChildItem -LiteralPath $CleanRoot -Directory -Filter '__pycache__' -Recurse -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -LiteralPath $CleanRoot -File -Filter '*.pyc' -Recurse -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
    }

    Write-Host '[5/7] Creating portable launchers and documentation...'
    $RunBat = @'
@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "python\pythonw.exe" (
    echo Portable Python runtime not found.
    pause
    exit /b 1
)
start "Child Face Finder" "python\pythonw.exe" "app\app.py"
exit /b 0
'@
    Write-AsciiCrlf -Path (Join-Path $PortableDir 'run.bat') -Text $RunBat

    $RunConsoleBat = @'
@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "python\python.exe" (
    echo Portable Python runtime not found.
    pause
    exit /b 1
)
"python\python.exe" "app\app.py"
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
'@
    Write-AsciiCrlf -Path (Join-Path $PortableDir 'run_console.bat') -Text $RunConsoleBat

    $PortableReadme = @'
# Child Face Finder Portable

Эта папка является самодостаточной portable-версией для 64-битной Windows.

Запуск: `run.bat`. Для диагностики используйте `run_console.bat` — консоль останется открытой и покажет ошибки/журнал запуска.

На целевом компьютере НЕ требуется отдельно устанавливать Python, uv, venv, CUDA Toolkit или cuDNN. Управляемый Python и нужные CUDA/cuDNN user-mode библиотеки уже находятся внутри portable.

Для ускорения на NVIDIA всё равно нужны совместимая видеокарта NVIDIA и достаточно свежий драйвер NVIDIA. Драйвер в portable не включается. Если в интерфейсе включено «Требовать NVIDIA CUDA», а CUDAExecutionProvider не запускается, программа выдаст ошибку вместо скрытого перехода на CPU.

Переносите папку `ChildFaceFinder_Portable` целиком. Нельзя отделять `run.bat` от папок `app`, `python`, `runtime` и `models`.

Временные изображения при необходимости создаются только в системной папке TEMP операционной системы. Используется та же логика штатной очистки и удаления остатков предыдущих аварийных запусков, что и в обычной версии.
'@
    [System.IO.File]::WriteAllText(
        (Join-Path $PortableDir 'README_PORTABLE.md'),
        ($PortableReadme -replace "`r?`n", "`r`n"),
        (New-Object System.Text.UTF8Encoding($true))
    )

    Write-Host '[6/7] Validating portable runtime and GPU...'
    $OldNoUserSite = $env:PYTHONNOUSERSITE
    $OldNoBytecode = $env:PYTHONDONTWRITEBYTECODE
    $OldPortableSite = $env:CFF_PORTABLE_SITE
    try {
        $env:PYTHONNOUSERSITE = '1'
        $env:PYTHONDONTWRITEBYTECODE = '1'
        $env:CFF_PORTABLE_SITE = $PortableSitePackages
        $Bootstrap = 'import os,site,runpy,sys; site.addsitedir(os.environ["CFF_PORTABLE_SITE"]); sys.path.insert(0, os.path.dirname(sys.argv[1])); runpy.run_path(sys.argv[1], run_name="__main__")'
        Invoke-Native $PortablePython '-c' $Bootstrap (Join-Path $PortableDir 'app\check_install.py')
        Invoke-Native $PortablePython '-c' $Bootstrap (Join-Path $PortableDir 'app\smoke_gpu.py')
    }
    finally {
        $env:PYTHONNOUSERSITE = $OldNoUserSite
        $env:PYTHONDONTWRITEBYTECODE = $OldNoBytecode
        $env:CFF_PORTABLE_SITE = $OldPortableSite
    }

    Write-Host '[7/7] Creating ZIP archive...'
    Compress-Archive -LiteralPath $PortableDir -DestinationPath $PortableZip -CompressionLevel Optimal -Force

    $FolderSize = (Get-ChildItem -LiteralPath $PortableDir -File -Recurse | Measure-Object -Property Length -Sum).Sum
    $ZipSize = (Get-Item -LiteralPath $PortableZip).Length
    Write-Host
    Write-Host '============================================================'
    Write-Host 'Portable build completed successfully.' -ForegroundColor Green
    Write-Host "Folder: $PortableDir"
    Write-Host "ZIP:    $PortableZip"
    Write-Host ('Folder size: {0:N1} MB' -f ($FolderSize / 1MB))
    Write-Host ('ZIP size:    {0:N1} MB' -f ($ZipSize / 1MB))
    Write-Host 'Copy/extract the entire portable folder on the target PC.'
    Write-Host 'Only a compatible NVIDIA driver is external to the package.'
    Write-Host '============================================================'
    exit 0
}
catch {
    Write-Host
    Write-Host '============================================================' -ForegroundColor Red
    Write-Host 'PORTABLE BUILD FAILED' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'The normal installed version was not modified.' -ForegroundColor Yellow
    Write-Host '============================================================' -ForegroundColor Red
    exit 1
}
