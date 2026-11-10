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


def test_series_reads_metadata_once_per_photo(tmp_path, monkeypatch):
    path = tmp_path/'IMG_0001.jpg'
    calls = []
    def read(p):
        calls.append(p)
        return series_selector.PhotoOrderInfo(p, datetime(2026, 1, 1), True, 1, ('folder', 'img'))
    monkeypatch.setattr(series_selector, 'read_photo_order_info', read)
    candidates = [series_selector.BestFrameCandidate(person, path, path.name, path.stem,
                  .1, 'full', .9, 100., .2, .2, (0, 0, 0)) for person in ['А', 'Б', 'В']]
    result = series_selector.select_best_series_frames(candidates, match_threshold=.45, eye_open_threshold=.1)
    assert len(result) == 3
    assert calls == [path]


def test_backwards_exif_jump_splits_series(tmp_path):
    from datetime import timedelta
    a, b = tmp_path/'IMG_0001.jpg', tmp_path/'IMG_0002.jpg'
    candidates = [series_selector.BestFrameCandidate('А', p, p.name, p.stem,
                  .1, 'full', .9, 100., .2, .2, (0, 0, 0)) for p in (a, b)]
    now = datetime(2026, 1, 1, 12)
    cache = {a: series_selector.PhotoOrderInfo(a, now, True, 1, ('folder', 'img')),
             b: series_selector.PhotoOrderInfo(b, now-timedelta(hours=1), True, 2, ('folder', 'img'))}
    assert len(series_selector.split_target_series(candidates, info_cache=cache)) == 2
