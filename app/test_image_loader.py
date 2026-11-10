from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np


def _import_image_loader(monkeypatch):
    fake_heif = types.ModuleType("pillow_heif")
    fake_heif.register_heif_opener = lambda: None
    fake_rawpy = types.ModuleType("rawpy")
    monkeypatch.setitem(sys.modules, "pillow_heif", fake_heif)
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)
    sys.modules.pop("image_loader", None)
    return importlib.import_module("image_loader")


def test_heic_decodes_directly_without_temp_jpeg(monkeypatch, tmp_path: Path) -> None:
    image_loader = _import_image_loader(monkeypatch)
    expected = np.zeros((4, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(image_loader, "_pil_to_bgr", lambda path: expected)

    result = image_loader.decode_image(Path("photo.heic"), tmp_path)

    assert result is expected
    assert list(tmp_path.iterdir()) == []


def test_windows_process_access_denied_is_treated_as_alive(monkeypatch):
    loader = _import_image_loader(monkeypatch)
    class Kernel:
        pass
    kernel = Kernel()
    kernel.OpenProcess = lambda *args: 0
    kernel.GetExitCodeProcess = lambda *args: 0
    kernel.CloseHandle = lambda *args: 1
    kernel.GetLastError = lambda: 5
    assert loader._windows_pid_is_running(1234, kernel)
    kernel.GetLastError = lambda: 87
    assert not loader._windows_pid_is_running(1234, kernel)


def test_windows_large_process_handle_is_preserved(monkeypatch):
    import ctypes
    loader = _import_image_loader(monkeypatch)
    handle = 0x123456789
    seen = []
    def query(value, output):
        assert value == handle
        output._obj.value = 259
        return 1
    kernel = types.SimpleNamespace(OpenProcess=lambda *args: handle, GetExitCodeProcess=query,
                                   CloseHandle=lambda value: seen.append(value), GetLastError=lambda: 0)
    assert loader._windows_pid_is_running(1234, kernel)
    assert kernel.OpenProcess.restype == ctypes.c_void_p
    assert seen == [handle]
