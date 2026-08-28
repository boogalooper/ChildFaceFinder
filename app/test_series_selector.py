from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path

import series_selector


def test_raw_capture_time_uses_rawpy_metadata_before_mtime(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "IMG_0001.CR3"
    path.write_bytes(b"not-a-real-raw")
    expected = datetime(2026, 8, 28, 10, 11, 12)

    class FakeRaw:
        other = types.SimpleNamespace(timestamp=expected)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_rawpy = types.SimpleNamespace(imread=lambda value: FakeRaw())
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    info = series_selector.read_photo_order_info(path)

    assert info.capture_time == expected
    assert info.time_is_metadata is True
    assert info.sequence_number == 1


def test_non_raw_without_exif_keeps_mtime_fallback(tmp_path: Path) -> None:
    path = tmp_path / "plain.bin"
    path.write_bytes(b"data")
    info = series_selector.read_photo_order_info(path)
    assert info.time_is_metadata is False
    assert info.capture_time != datetime.min
