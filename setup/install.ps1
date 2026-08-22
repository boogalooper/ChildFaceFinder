$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

# If this script was started from a 32-bit host on 64-bit Windows, relaunch
# it in native 64-bit Windows PowerShell before touching System32, NVIDIA
# tools or the x64 Python environment.  This avoids WOW64 filesystem
# redirection (System32 -> SysWOW64).
if ([System.Environment]::Is64BitOperatingSystem -and -not [System.Environment]::Is64BitProcess) {
    $nativePowerShell = Join-Path $env:SystemRoot 'Sysnative\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $nativePowerShell -PathType Leaf)) {
        throw "64-bit Windows PowerShell could not be located through Sysnative: $nativePowerShell"
    }
    Write-Host '32-bit PowerShell detected; restarting installer in 64-bit PowerShell...' -ForegroundColor Yellow
    & $nativePowerShell -NoLogo -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath
    exit $LASTEXITCODE
}

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
$VenvTransactionFile = Join-Path $SetupDir 'venv_transaction.txt'
$UvUrl = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
$UvSha256 = '4c4d49d8738847d9b71ba319e49a5688c93eac0fe6204b1df24e98528dddf39a'

# Windows PowerShell 5.1 on older systems can otherwise negotiate an obsolete TLS
# version. Keep the current flags and explicitly add TLS 1.2.
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
}
catch {
    # Modern Windows already uses a suitable TLS stack.
}

# Keep uv-managed Python inside the project. Do not modify the system Python.
$env:UV_PYTHON_INSTALL_DIR = $PythonInstallDir
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_NO_CONFIG = '1'
# Use the Windows certificate store too. This is more robust behind corporate
# HTTPS proxies while retaining normal certificate verification.
$env:UV_SYSTEM_CERTS = 'true'
# uv already retries HTTP requests, but the defaults are deliberately raised for
# this large one-shot installer.
$env:UV_HTTP_RETRIES = '5'
$env:UV_HTTP_TIMEOUT = '60'
# Freeze the resolver horizon so transitive dependencies cannot silently change
# on a later reinstall. All versions used by this release were published before
# this cutoff.
$env:UV_EXCLUDE_NEWER = '2026-08-23'

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Invoke-WebDownloadWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [int]$Attempts = 4
    )
    $lastError = $null
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $OutFile -TimeoutSec 120
            if (-not (Test-Path -LiteralPath $OutFile -PathType Leaf)) {
                throw 'download finished without creating the destination file'
            }
            return
        }
        catch {
            $lastError = $_.Exception
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            if ($attempt -lt $Attempts) {
                Write-Warning "Download failed ($($lastError.Message)). Retry $attempt/$Attempts..."
                Start-Sleep -Seconds ([Math]::Min(8, 2 * $attempt))
            }
        }
    }
    throw "Download failed after $Attempts attempts: $($lastError.Message)"
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

function Assert-DirectoryWritable {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    try {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        $probe = Join-Path $Path (".cff_write_test_" + [guid]::NewGuid().ToString('N') + '.tmp')
        [System.IO.File]::WriteAllText($probe, 'ok')
        Remove-Item -LiteralPath $probe -Force
    }
    catch {
        throw "$Label is not writable: $Path. Move the project to a writable local folder and retry. Details: $($_.Exception.Message)"
    }
}

function Get-FreeSpaceGB {
    param([Parameter(Mandatory = $true)][string]$Path)
    try {
        $full = [System.IO.Path]::GetFullPath($Path)
        $root = [System.IO.Path]::GetPathRoot($full)
        if ($root -notmatch '^[A-Za-z]:\\$') {
            return $null
        }
        $drive = New-Object System.IO.DriveInfo($root)
        return [Math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    }
    catch {
        return $null
    }
}

function Write-VenvTransactionMarker {
    param([string]$BackupPath)
    $value = if ($BackupPath) { $BackupPath } else { '__NONE__' }
    [System.IO.File]::WriteAllText($VenvTransactionFile, $value, [System.Text.Encoding]::UTF8)
}

function Remove-VenvTransactionMarker {
    Remove-Item -LiteralPath $VenvTransactionFile -Force -ErrorAction SilentlyContinue
}

function Recover-InterruptedVenvTransaction {
    if (-not (Test-Path -LiteralPath $VenvTransactionFile -PathType Leaf)) {
        return
    }

    Write-Warning 'An interrupted venv replacement from a previous installer run was detected.'
    $marker = (Get-Content -LiteralPath $VenvTransactionFile -Raw).Trim()
    if ($marker -eq '__NONE__') {
        if (Test-Path -LiteralPath $VenvDir) {
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
        Remove-VenvTransactionMarker
        Write-Host 'Removed the incomplete venv from the interrupted first installation.' -ForegroundColor Yellow
        return
    }

    $backupFull = [System.IO.Path]::GetFullPath($marker)
    $projectFull = [System.IO.Path]::GetFullPath($ProjectDir).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
    $backupParent = [System.IO.Path]::GetDirectoryName($backupFull).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
    $backupName = [System.IO.Path]::GetFileName($backupFull)
    if ($backupParent -ne $projectFull -or -not $backupName.StartsWith('venv.previous.', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Invalid venv transaction marker: $marker. Remove $VenvTransactionFile manually after checking the project folder."
    }

    if (Test-Path -LiteralPath $backupFull) {
        if (Test-Path -LiteralPath $VenvDir) {
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
        Move-Item -LiteralPath $backupFull -Destination $VenvDir
        Write-Host 'Previous working venv restored after the interrupted installation.' -ForegroundColor Yellow
    }
    else {
        Write-Warning 'The transaction marker remained, but its backup no longer exists. Keeping the current venv.'
    }
    Remove-VenvTransactionMarker
}

Write-Host '============================================================'
Write-Host 'Child Face Finder - installer'
Write-Host "Managed tools: uv $UvVersion, CPython $PythonVersion"
Write-Host 'System Python is not used or modified.'
Write-Host 'Temporary installer/cache files use the Windows TEMP directory.'
Write-Host '============================================================'
Write-Host

$ExitCode = 0
$InstallSucceeded = $false
$VenvBackupDir = $null
$VenvReplacementStarted = $false
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("ChildFaceFinder_install_" + [guid]::NewGuid().ToString('N'))
# Prevent uv from leaving multi-gigabyte package caches in the user profile.
$env:UV_CACHE_DIR = Join-Path $TempRoot 'uv-cache'

try {
    New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null

    Write-Host '[0/7] Preflight checks...'
    Write-Host "PowerShell process: $([IntPtr]::Size * 8)-bit"
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        throw "PowerShell 5.1 or newer is required. Detected: $($PSVersionTable.PSVersion)"
    }
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        throw "Windows 10/11 x64 is required. Detected OS version: $osVersion"
    }

    $nativeArch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    if ($nativeArch -ne 'AMD64') {
        throw "This installer requires 64-bit x86 Windows (AMD64). Detected architecture: $nativeArch"
    }

    Assert-DirectoryWritable -Path $ProjectDir -Label 'Project directory'
    Assert-DirectoryWritable -Path $TempRoot -Label 'Windows TEMP directory'

    $requiredFiles = @(
        $Requirements,
        (Join-Path $AppDir 'check_install.py'),
        (Join-Path $AppDir 'model_setup.py'),
        (Join-Path $AppDir 'smoke_gpu.py')
    )
    foreach ($requiredFile in $requiredFiles) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "Installation file is missing: $requiredFile. Re-extract the complete ChildFaceFinder archive and retry."
        }
    }

    Recover-InterruptedVenvTransaction

    if ($ProjectDir.Length -gt 120) {
        Write-Warning 'The project path is very long. If Windows long paths are disabled, package extraction can fail. Prefer a short path such as C:\ChildFaceFinder.'
    }

    $projectFree = Get-FreeSpaceGB -Path $ProjectDir
    $tempFree = Get-FreeSpaceGB -Path $TempRoot
    $projectRoot = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($ProjectDir))
    $tempRootDrive = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($TempRoot))
    if ($projectFree -ne $null -and $tempFree -ne $null) {
        if ($projectRoot -eq $tempRootDrive) {
            Write-Host "Free disk space: $projectFree GB"
            if ($projectFree -lt 10) {
                throw 'Less than 10 GB is free. Installation temporarily keeps the previous venv while downloading/extracting large CUDA/cuDNN wheels and the model; free disk space before retrying.'
            }
            elseif ($projectFree -lt 14) {
                Write-Warning 'Less than 14 GB is free. Installation should normally fit, but a reinstall temporarily keeps the previous venv and CUDA/cuDNN wheels are large.'
            }
        }
        else {
            Write-Host "Free disk space: project $projectFree GB, TEMP $tempFree GB"
            if ($projectFree -lt 7) {
                throw 'Less than 7 GB is free on the project drive. A reinstall temporarily keeps the previous venv while the new CUDA environment is built; free disk space before retrying.'
            }
            elseif ($projectFree -lt 9) {
                Write-Warning 'Less than 9 GB is free on the project drive. Installation should normally fit, but the old and new venv can coexist temporarily.'
            }
            if ($tempFree -lt 5) {
                throw 'Less than 5 GB is free on the TEMP drive used for package downloads/cache and model staging. Free disk space before installation.'
            }
            elseif ($tempFree -lt 7) {
                Write-Warning 'Less than 7 GB is free on the TEMP drive. CUDA/cuDNN wheels and model staging are large.'
            }
        }
    }

    # Resolve nvidia-smi robustly.  A 32-bit launcher on 64-bit Windows can
    # redirect System32 to SysWOW64, which makes a perfectly valid NVIDIA
    # installation look missing.  install.bat already prefers 64-bit
    # PowerShell, but keep these fallbacks for direct install.ps1 launches.
    $nvidiaSmiPath = $null
    $nvidiaSmiCommand = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($nvidiaSmiCommand) {
        $nvidiaSmiPath = $nvidiaSmiCommand.Source
    }

    if (-not $nvidiaSmiPath) {
        $nvidiaCandidates = @(
            (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
            (Join-Path $env:SystemRoot 'Sysnative\nvidia-smi.exe')
        )
        if ($env:ProgramW6432) {
            $nvidiaCandidates += Join-Path $env:ProgramW6432 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
        }
        if ($env:ProgramFiles) {
            $nvidiaCandidates += Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
        }

        foreach ($candidate in $nvidiaCandidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $nvidiaSmiPath = $candidate
                break
            }
        }
    }

    if (-not $nvidiaSmiPath) {
        # Some DCH/newer driver packages keep nvidia-smi in DriverStore instead
        # of exposing it through PATH. Search NVIDIA driver package folders only.
        $driverStoreRoots = @(
            (Join-Path $env:SystemRoot 'System32\DriverStore\FileRepository'),
            (Join-Path $env:SystemRoot 'Sysnative\DriverStore\FileRepository')
        )
        foreach ($driverStoreRoot in $driverStoreRoots) {
            if (-not (Test-Path -LiteralPath $driverStoreRoot -PathType Container)) {
                continue
            }
            try {
                $nvidiaPackages = Get-ChildItem -LiteralPath $driverStoreRoot -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like 'nv*' }
                foreach ($package in $nvidiaPackages) {
                    $found = Get-ChildItem -LiteralPath $package.FullName -Filter 'nvidia-smi.exe' -File -Recurse -ErrorAction SilentlyContinue |
                        Select-Object -First 1
                    if ($found) {
                        $nvidiaSmiPath = $found.FullName
                        break
                    }
                }
            }
            catch {
                # The final error below is clearer than a DriverStore access error.
            }
            if ($nvidiaSmiPath) { break }
        }
    }
    if ($nvidiaSmiPath) {
        try {
            Write-Host "nvidia-smi: $nvidiaSmiPath"
            Write-Host 'NVIDIA GPU/driver:'
            $gpuInfo = @(& $nvidiaSmiPath --query-gpu=name,driver_version --format=csv,noheader)
            if ($LASTEXITCODE -ne 0) {
                throw "nvidia-smi exited with code $LASTEXITCODE"
            }
            $gpuInfo | ForEach-Object { Write-Host $_ }
            if (-not $gpuInfo -or [string]::IsNullOrWhiteSpace(($gpuInfo -join ''))) {
                throw 'nvidia-smi returned no GPU information'
            }

            # CUDA 12.x minor-version compatibility on Windows requires >=528.33.
            # Reject an incompatible driver before downloading several gigabytes.
            $driverVersions = @()
            foreach ($line in $gpuInfo) {
                $parts = $line -split ','
                if ($parts.Count -ge 2) {
                    $text = $parts[$parts.Count - 1].Trim()
                    try { $driverVersions += [version]$text } catch { }
                }
            }
            if ($driverVersions.Count -eq 0) {
                Write-Warning 'Could not parse the NVIDIA driver version; the final CUDA smoke-test will verify compatibility.'
            }
            else {
                $oldestDriver = ($driverVersions | Sort-Object)[0]
                if ($oldestDriver -lt [version]'528.33') {
                    throw "NVIDIA driver $oldestDriver is too old for CUDA 12.x on Windows (minimum 528.33). Update the NVIDIA driver before installation."
                }
                if ($oldestDriver -lt [version]'572.61') {
                    Write-Warning "NVIDIA driver $oldestDriver meets CUDA 12.x minor-compatibility minimum, but is older than the CUDA 12.8 Update 1 toolkit driver (572.61). If the final GPU smoke-test fails, update the driver."
                }
            }
        }
        catch {
            throw "NVIDIA driver/GPU check failed: $($_.Exception.Message). Update the NVIDIA driver and retry."
        }
    }
    else {
        throw 'nvidia-smi was not found. A supported NVIDIA GPU with a working NVIDIA driver is required.'
    }

    $vcRuntimeCandidates = @(
        (Join-Path $env:SystemRoot 'System32\vcruntime140_1.dll'),
        (Join-Path $env:SystemRoot 'Sysnative\vcruntime140_1.dll')
    )
    $vcRuntimeFound = $false
    foreach ($vcRuntime in $vcRuntimeCandidates) {
        if (Test-Path -LiteralPath $vcRuntime -PathType Leaf) {
            $vcRuntimeFound = $true
            break
        }
    }
    if (-not $vcRuntimeFound) {
        throw 'Microsoft Visual C++ 2015-2022 x64 Redistributable was not detected (vcruntime140_1.dll is missing). Install/update it before running install.bat.'
    }

    Write-Host '[1/7] Preparing local uv...'
    if (-not (Test-Uv)) {
        if (Test-Path -LiteralPath $UvDir) {
            Remove-Item -LiteralPath $UvDir -Recurse -Force
        }
        New-Item -ItemType Directory -Path $UvDir -Force | Out-Null

        $UvZip = Join-Path $TempRoot 'uv.zip'
        Write-Host "Downloading $UvUrl"
        Invoke-WebDownloadWithRetry -Uri $UvUrl -OutFile $UvZip
        $ActualHash = (Get-FileHash -LiteralPath $UvZip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $UvSha256) {
            throw "uv archive SHA256 mismatch: $ActualHash"
        }
        Expand-Archive -LiteralPath $UvZip -DestinationPath $UvDir -Force

        if (-not (Test-Uv)) {
            throw "uv $UvVersion was downloaded but uv.exe could not be validated. Antivirus/security software may have blocked the executable."
        }
    }
    Invoke-Native -FilePath $UvExe -Arguments @('--version')

    Write-Host '[2/7] Installing managed CPython...'
    Invoke-Native -FilePath $UvExe -Arguments @('python', 'install', $PythonVersion)

    Write-Host '[3/7] Creating venv...'
    if (Test-Path -LiteralPath $VenvDir) {
        $VenvBackupDir = Join-Path $ProjectDir ("venv.previous." + [guid]::NewGuid().ToString('N'))
        Write-VenvTransactionMarker -BackupPath $VenvBackupDir
        try {
            Move-Item -LiteralPath $VenvDir -Destination $VenvBackupDir
            $VenvReplacementStarted = $true
            Write-Host 'Previous venv saved temporarily until the new installation passes all checks.'
        }
        catch {
            $VenvBackupDir = $null
            Remove-VenvTransactionMarker
            throw "Could not preserve the existing venv. Close Child Face Finder and any process using its venv, then retry. Details: $($_.Exception.Message)"
        }
    }
    else {
        Write-VenvTransactionMarker -BackupPath $null
        $VenvReplacementStarted = $true
    }

    Invoke-Native -FilePath $UvExe -Arguments @('venv', $VenvDir, '--python', $PythonVersion)

    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw 'venv was created but python.exe was not found.'
    }

    Write-Host '[4/7] Installing runtime dependencies...'
    Invoke-Native -FilePath $UvExe -Arguments @('pip', 'install', '--python', $PythonExe, '--only-binary=:all:', '-r', $Requirements)

    # InsightFace is installed separately to avoid replacing our selected
    # GPU/headless runtime packages with CPU/GUI variants from transitive deps.
    Invoke-Native -FilePath $UvExe -Arguments @('pip', 'install', '--python', $PythonExe, '--only-binary=:all:', '--no-deps', 'insightface==1.0.1')

    # Package installation is finished; free the large wheel cache before the
    # model archive is downloaded/extracted in TEMP.
    if (Test-Path -LiteralPath $env:UV_CACHE_DIR) {
        Remove-Item -LiteralPath $env:UV_CACHE_DIR -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host '[5/7] Checking Python, Tkinter and CUDA provider...'
    Invoke-Native -FilePath $PythonExe -Arguments @((Join-Path $AppDir 'check_install.py'))

    Write-Host '[6/7] Downloading/verifying InsightFace antelopev2...'
    Invoke-Native -FilePath $PythonExe -Arguments @((Join-Path $AppDir 'model_setup.py'))

    Write-Host '[7/7] Running GPU/model smoke test...'
    Invoke-Native -FilePath $PythonExe -Arguments @((Join-Path $AppDir 'smoke_gpu.py'))

    $InstallSucceeded = $true
    if ($VenvBackupDir -and (Test-Path -LiteralPath $VenvBackupDir)) {
        try {
            Remove-Item -LiteralPath $VenvBackupDir -Recurse -Force
            $VenvBackupDir = $null
        }
        catch {
            Write-Warning "New installation is valid, but the previous venv backup could not be deleted: $VenvBackupDir"
        }
    }
    # The new environment already passed every check. Even if an obsolete backup
    # could not be deleted, a future installer must not roll back to it.
    Remove-VenvTransactionMarker

    Write-Host
    Write-Host '============================================================'
    Write-Host 'Installation completed successfully.'
    Write-Host 'Start the application with run.bat.'
    Write-Host '============================================================'
}
catch {
    $InstallError = $_.Exception

    if (-not $InstallSucceeded -and $VenvReplacementStarted) {
        # Remove only the new/incomplete environment. The flag stays false for
        # failures that happen before the old venv is moved aside.
        $rollbackComplete = $true
        if (Test-Path -LiteralPath $VenvDir) {
            try {
                Remove-Item -LiteralPath $VenvDir -Recurse -Force
            }
            catch {
                $rollbackComplete = $false
                Write-Warning "Could not completely remove the failed venv: $($_.Exception.Message)"
            }
        }

        if ($VenvBackupDir -and (Test-Path -LiteralPath $VenvBackupDir)) {
            Write-Host
            Write-Host 'Restoring the previous working venv...' -ForegroundColor Yellow
            try {
                if (Test-Path -LiteralPath $VenvDir) {
                    throw "Failed venv still exists and blocks restore: $VenvDir"
                }
                Move-Item -LiteralPath $VenvBackupDir -Destination $VenvDir
                $VenvBackupDir = $null
                Write-Host 'Previous venv restored.' -ForegroundColor Yellow
            }
            catch {
                $rollbackComplete = $false
                Write-Warning "Automatic venv restore failed. The previous environment is still at: $VenvBackupDir"
                Write-Warning "Restore error: $($_.Exception.Message)"
            }
        }
        if ($rollbackComplete) {
            Remove-VenvTransactionMarker
        }
    }

    Write-Host
    Write-Host '============================================================' -ForegroundColor Red
    Write-Host 'INSTALLATION FAILED' -ForegroundColor Red
    Write-Host $InstallError.Message -ForegroundColor Red
    Write-Host 'Review the messages above for the failing step.' -ForegroundColor Red
    Write-Host 'If a download/package step failed, retry install.bat after checking access to GitHub, PyPI and files.pythonhosted.org; the previous working venv is restored automatically when possible.' -ForegroundColor Yellow
    Write-Host 'If ONNX Runtime reports a missing DLL, install/update the Microsoft Visual C++ 2015-2022 x64 Redistributable.' -ForegroundColor Yellow
    Write-Host 'For RTX 50-series, use a current NVIDIA driver. CUDA 12.x requires at least driver 528.33 on Windows; a current driver is strongly recommended.' -ForegroundColor Yellow
    Write-Host 'A separate CUDA Toolkit install is not required by this project.' -ForegroundColor Yellow
    Write-Host '============================================================' -ForegroundColor Red
    $ExitCode = 1
}
finally {
    # Always remove installer downloads and uv package cache from system TEMP.
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

exit $ExitCode
