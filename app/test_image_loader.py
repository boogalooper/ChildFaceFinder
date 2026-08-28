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
