from __future__ import annotations

import csv
from pathlib import Path

from collector import collect_from_csv


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle, delimiter=";", lineterminator="\n").writerows(rows)


def test_incomplete_row_with_source_column_is_skipped_not_crashed(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "out"
    source_root.mkdir()
    photo = source_root / "IMG_0001.jpg"
    photo.write_bytes(b"photo")

    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            ["Идентификатор цели", "номер фото с этой целью", "Исходный файл"],
            ["Ребёнок 1", "IMG_0001", "IMG_0001.jpg"],
            ["Ребёнок 2", "IMG_0002"],
        ],
    )

    summary = collect_from_csv(csv_path, source_root, destination)

    assert summary.copied == 1
    assert summary.skipped == 1
    assert summary.errors == 0
    report = summary.report_csv.read_text(encoding="utf-8-sig")
    assert "неполная строка CSV: отсутствует исходный файл" in report


def test_legacy_csv_without_source_column_uses_unique_stem(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "out"
    source_root.mkdir()
    (source_root / "IMG_0042.CR3").write_bytes(b"raw")

    csv_path = tmp_path / "legacy.csv"
    _write_csv(
        csv_path,
        [
            ["Идентификатор цели", "номер фото с этой целью"],
            ["Ребёнок", "IMG_0042"],
        ],
    )

    summary = collect_from_csv(csv_path, source_root, destination)

    assert summary.copied == 1
    assert summary.skipped == 0
    assert (destination / "Ребёнок" / "IMG_0042.CR3").is_file()


def test_copy_failure_never_deletes_existing_target(tmp_path):
    import pytest
    from collector import _copy_file_cancellable
    source = tmp_path/'missing.jpg'
    target = tmp_path/'original.jpg'
    target.write_bytes(b'valuable original')
    with pytest.raises(FileNotFoundError):
        _copy_file_cancellable(source, target)
    assert target.read_bytes() == b'valuable original'
    source.write_bytes(b'new image')
    with pytest.raises(FileExistsError):
        _copy_file_cancellable(source, target)
    assert target.read_bytes() == b'valuable original'


def test_cancel_copy_removes_only_its_partial_file(tmp_path):
    import pytest
    from collector import _copy_file_cancellable, CollectorCancelled
    source = tmp_path/'source.jpg'
    target = tmp_path/'copy.jpg'
    source.write_bytes(b'x' * 100)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 3
    with pytest.raises(CollectorCancelled):
        _copy_file_cancellable(source, target, cancel_check=cancel, chunk_size=10)
    assert not target.exists()
    assert source.read_bytes() == b'x' * 100
