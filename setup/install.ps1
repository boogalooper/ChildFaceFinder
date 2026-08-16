$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$SetupDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $SetupDir
$AppDir = Join-Path $ProjectDir 'app'
Set-Location -LiteralPath $ProjectDir

$UvVersion = '0.12.5'
$PythonVersion = '3.11.16'
$ToolsDir = Join-Path $ProjectDir 'tools'
$UvDir = Join-Path $ToolsDir 'uv'
$UvExe = Join-Path $UvDir 'uv.exe'
$PythonInstallDir = Join-Path $ToolsDir 'python'
$VenvDir = Join-Path $ProjectDir 'venv'
$PythonExe = Join-Path $VenvDir 'Scripts\python.exe'
$Requirements = Join-Path $SetupDir 'requirements.txt'
$UvUrl = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvSha256 = '4c4d49d8738847d9b71ba319e49a5688c93eac0fe6204b1df24e98528dddf39a'

# Keep uv-managed Python inside the project. Do not modify the system Python.
$env:UV_PYTHON_INSTALL_DIR = $PythonInstallDir
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_NO_CONFIG = '1'

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

function Test-Uv {
    if (-not (Test-Path -LiteralPath $UvExe -PathType Leaf)) {
        return $false
    }
    try {
        $versionText = (& $UvExe --version 2>$null) -join "`n"
        return $versionText -match ([regex]::Escape("uv $UvVersion"))
    }
    catch {
        return $false
    }
}

Write-Host '============================================================'
Write-Host 'Child Face Finder - installer'
Write-Host "Managed tools: uv $UvVersion, CPython $PythonVersion"
Write-Host 'System Python is not used or modified.'
Write-Host 'Temporary installer files use the Windows TEMP directory.'
Write-Host '============================================================'
Write-Host

$ExitCode = 0
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ChildFaceFinder_install_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

try {
    Write-Host '[1/7] Preparing local uv...'
    if (-not (Test-Uv)) {
        if (Test-Path -LiteralPath $UvDir) {
            Remove-Item -LiteralPath $UvDir -Recurse -Force
        }
        New-Item -ItemType Directory -Path $UvDir -Force | Out-Null

        $UvZip = Join-Path $TempRoot 'uv.zip'
        Write-Host "Downloading $UvUrl"
        Invoke-WebRequest -UseBasicParsing -Uri $UvUrl -OutFile $UvZip
        $ActualHash = (Get-FileHash -LiteralPath $UvZip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $UvSha256) {
            throw "uv archive SHA256 mismatch: $ActualHash"
        }
        Expand-Archive -LiteralPath $UvZip -DestinationPath $UvDir -Force

        if (-not (Test-Uv)) {
            throw "uv $UvVersion was downloaded but uv.exe could not be validated."
        }
    }
    Invoke-Native $UvExe '--version'

    Write-Host '[2/7] Installing managed CPython...'
    Invoke-Native $UvExe 'python' 'install' $PythonVersion

    Write-Host '[3/7] Creating venv...'
    if (Test-Path -LiteralPath $VenvDir) {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
    Invoke-Native $UvExe 'venv' $VenvDir '--python' $PythonVersion

    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw 'venv was created but python.exe was not found.'
    }

    Write-Host '[4/7] Installing runtime dependencies...'
    Invoke-Native $UvExe 'pip' 'install' '--python' $PythonExe '-r' $Requirements

    # InsightFace is installed separately to avoid replacing our selected
    # GPU/headless runtime packages with CPU/GUI variants from transitive deps.
    Invoke-Native $UvExe 'pip' 'install' '--python' $PythonExe '--no-deps' 'insightface==1.0.1'

    Write-Host '[5/7] Checking Python, Tkinter and CUDA provider...'
    Invoke-Native $PythonExe (Join-Path $AppDir 'check_install.py')

    Write-Host '[6/7] Downloading/verifying InsightFace antelopev2...'
    Invoke-Native $PythonExe (Join-Path $AppDir 'model_setup.py') '--download'

    Write-Host '[7/7] Running GPU/model smoke test...'
    Invoke-Native $PythonExe (Join-Path $AppDir 'smoke_gpu.py')

    Write-Host
    Write-Host '============================================================'
    Write-Host 'Installation completed successfully.'
    Write-Host 'Start the application with run.bat.'
    Write-Host '============================================================'
}
catch {
    Write-Host
    Write-Host '============================================================' -ForegroundColor Red
    Write-Host 'INSTALLATION FAILED' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host 'Review the messages above for the failing step.' -ForegroundColor Red
    Write-Host 'For RTX 5090, install a current NVIDIA driver.' -ForegroundColor Yellow
    Write-Host 'A separate CUDA Toolkit install is not required by this project.' -ForegroundColor Yellow
    Write-Host '============================================================' -ForegroundColor Red
    $ExitCode = 1
}
finally {
    # Always remove installer downloads/extraction from the system TEMP folder.
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($ExitCode -eq 0) {
    Write-Host
    $PortableAnswer = Read-Host 'Собрать portable-версию сейчас? [y/N]'
    if ($PortableAnswer -match '^(?i:y|yes|д|да)$') {
        Write-Host
        & (Join-Path $SetupDir 'build_portable.ps1')
        if ($LASTEXITCODE -ne 0) {
            Write-Host
            Write-Host 'Portable build failed, but the normal installation is valid.' -ForegroundColor Yellow
            Write-Host 'Можно повторить сборку позже через setup\build_portable.bat.' -ForegroundColor Yellow
        }
    }
    else {
        Write-Host 'Portable-сборка пропущена. Позже её можно создать через setup\build_portable.bat.'
    }
}

exit $ExitCode
