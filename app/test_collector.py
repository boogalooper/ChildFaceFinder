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
