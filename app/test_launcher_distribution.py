from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launchers_have_no_bom_and_use_crlf():
    for rel in ("run.bat", "install.bat", "setup/run_console.bat"):
        data = (ROOT / rel).read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf")
        assert data.startswith(b"@echo off\r\n")
        assert data.count(b"\n") == data.count(b"\r\n")


def test_childfacefinder_is_private_python_install():
    install = (ROOT / "setup" / "install.ps1").read_text(encoding="utf-8-sig")
    assert "$UvVersion = '0.12.5'" in install
    assert "$PythonVersion = '3.11.16'" in install
    assert "UV_PYTHON_INSTALL_DIR" in install


def test_launcher_repair_is_self_locating():
    run = (ROOT / "run.bat").read_text(encoding="utf-8")
    console = (ROOT / "setup" / "run_console.bat").read_text(encoding="utf-8")
    repair = (ROOT / "setup" / "repair_venv.ps1").read_text(encoding="utf-8")
    assert "-ProjectDir" not in run
    assert "-ProjectDir" not in console
    assert not repair.lstrip().lower().startswith("param(")
    assert "$PSScriptRoot" in repair


def test_installer_inline_python_survives_windows_powershell_quoting():
    install = (ROOT / "setup" / "install.ps1").read_text(encoding="utf-8-sig")
    # Windows PowerShell strips nested double quotes when forwarding native
    # executable arguments. Python string literals inside -c therefore use
    # single quotes, which are ordinary characters to cmd/PowerShell.
    assert 'struct.calcsize("P")' not in install
    assert "struct.calcsize('P')" in install
    assert 'os.environ["APP_EXPECTED_VENV"]' not in install
    assert "os.environ['APP_EXPECTED_VENV']" in install


def test_venv_is_created_and_repaired_as_relocatable():
    install = (ROOT / "setup" / "install.ps1").read_text(encoding="utf-8-sig")
    repair = (ROOT / "setup" / "repair_venv.ps1").read_text(encoding="utf-8-sig")
    assert "--relocatable" in install
    assert "--relocatable" in repair
    assert "--allow-existing" in repair
    assert "--no-python-downloads" in repair
    assert "NativeCommandError" in repair
    assert "Move-Item -LiteralPath $tmp" not in repair
