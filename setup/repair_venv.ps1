$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$ProjectDir = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectDir 'venv'
$ConfigPath = Join-Path $VenvDir 'pyvenv.cfg'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$ManagedPythonRoot = Join-Path $ProjectDir 'tools\python'

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Host 'Local venv was not found.'
    exit 2
}

$Candidates = @(Get-ChildItem -LiteralPath $ManagedPythonRoot -Filter 'python.exe' -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -like 'cpython-3.11.16-windows-x86_64-none' })
if ($Candidates.Count -ne 1) {
    Write-Host 'Private CPython 3.11.16 was not found. Run install.bat once.'
    exit 3
}
$BasePython = $Candidates[0].FullName
$BaseHome = $Candidates[0].Directory.FullName
$ConfigText = [System.IO.File]::ReadAllText($ConfigPath)
if ($ConfigText -notmatch '(?im)^home\s*=.*$') {
    Write-Host 'pyvenv.cfg has no home entry. Run install.bat once.'
    exit 4
}
$CurrentHome = ([regex]::Match($ConfigText, '(?im)^home\s*=.*$').Value -replace '(?i)^home\s*=\s*', '').Trim()
if (-not [string]::Equals($CurrentHome, $BaseHome, [System.StringComparison]::OrdinalIgnoreCase)) {
    $HomeLine = 'home = ' + $BaseHome
    $ConfigText = [regex]::Replace($ConfigText, '(?im)^home\s*=.*$', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $HomeLine })
    $ExecutableLine = 'executable = ' + $BasePython
    if ($ConfigText -match '(?im)^executable\s*=.*$') {
        $ConfigText = [regex]::Replace($ConfigText, '(?im)^executable\s*=.*$', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $ExecutableLine })
    }
    $CommandLine = 'command = ' + $BasePython + ' -m venv ' + $VenvDir
    if ($ConfigText -match '(?im)^command\s*=.*$') {
        $ConfigText = [regex]::Replace($ConfigText, '(?im)^command\s*=.*$', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $CommandLine })
    }
    $tmp = $ConfigPath + '.relocate.' + [guid]::NewGuid().ToString('N') + '.tmp'
    try {
        [System.IO.File]::WriteAllText($tmp, $ConfigText, (New-Object System.Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $tmp -Destination $ConfigPath -Force
        Write-Host 'Program folder move detected; local Python path repaired.'
    } finally { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
}

$env:APP_EXPECTED_VENV = $VenvDir
try {
    & $VenvPython -c "import os,pathlib,struct,sys; expected=pathlib.Path(os.environ['APP_EXPECTED_VENV']).resolve(); assert sys.version_info[:3]==(3,11,16); assert struct.calcsize('P')==8; assert pathlib.Path(sys.prefix).resolve()==expected" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Local Python environment validation failed. Run install.bat once.'
        exit 5
    }
}
finally { Remove-Item Env:APP_EXPECTED_VENV -ErrorAction SilentlyContinue }
exit 0
