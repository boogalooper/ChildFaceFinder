from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(slots=True)
class CollectorSummary:
    copied: int
    skipped: int
    errors: int
    target_count: int
    report_csv: Path
    cancelled: bool = False


class CollectorCancelled(Exception):
    pass


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise CollectorCancelled("Сбор файлов отменён пользователем")


def _clean_folder_name(name: str) -> str:
    value = INVALID_WINDOWS_CHARS.sub("_", " ".join(str(name).split())).strip(" .")
    if not value:
        value = "Без имени"
    if value.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        value = f"_{value}"
    return value[:120].rstrip(" .") or "Без имени"


def _unique_target_folder_names(person_names: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: dict[str, int] = {}
    for person in person_names:
        base = _clean_folder_name(person)
        key = base.casefold()
        number = used.get(key, 0) + 1
        used[key] = number
        candidate = base if number == 1 else f"{base} ({number})"
        while candidate.casefold() in {value.casefold() for value in result.values()}:
            number += 1
            used[key] = number
            candidate = f"{base} ({number})"
        result[person] = candidate
    return result


def _read_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV пуст")
    header = [cell.strip() for cell in rows[0]]
    data = [[cell.strip() for cell in row] for row in rows[1:] if any(cell.strip() for cell in row)]
    return header, data


def _find_column(header: list[str], candidates: tuple[str, ...], fallback: int | None) -> int | None:
    normalized = [" ".join(value.casefold().split()) for value in header]
    for candidate in candidates:
        key = " ".join(candidate.casefold().split())
        if key in normalized:
            return normalized.index(key)
    if fallback is not None and fallback < len(header):
        return fallback
    return None


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _source_from_relative(source_root: Path, relative_text: str) -> Path | None:
    if not relative_text:
        return None
    normalized = relative_text.replace("\\", os.sep).replace("/", os.sep)
    candidate = (source_root / normalized).resolve()
    root = source_root.resolve()
    if not _is_inside(candidate, root):
        return None
    return candidate if candidate.is_file() else None


def _build_stem_index(
    source_root: Path,
    destination_root: Path,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, list[Path]]:
    root = source_root.resolve()
    destination = destination_root.resolve()
    result: dict[str, list[Path]] = {}
    scanned_files = 0
    if progress is not None:
        progress(0, 0, "Индексация исходных фотографий…")
    for path in root.rglob("*"):
        _raise_if_cancelled(cancel_check)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not _is_inside(resolved, root):
            continue
        if _is_inside(resolved, destination):
            continue
        if not path.is_file() or path.name.startswith("."):
            continue
        result.setdefault(path.stem.casefold(), []).append(path)
        scanned_files += 1
        if progress is not None and (scanned_files == 1 or scanned_files % 250 == 0):
            progress(scanned_files, 0, f"Индексация: просмотрено файлов {scanned_files}")
    if progress is not None:
        progress(scanned_files, 0, f"Индекс готов: файлов {scanned_files}")
    return result


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem} ({counter}){suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _copy_file_cancellable(
    source: Path,
    target: Path,
    cancel_check: Callable[[], bool] | None = None,
    chunk_size: int = 1024 * 1024,
) -> None:
    _raise_if_cancelled(cancel_check)
    try:
        with source.open("rb") as src, target.open("wb") as dst:
            while True:
                _raise_if_cancelled(cancel_check)
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
        _raise_if_cancelled(cancel_check)
        shutil.copystat(source, target)
    except BaseException:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def collect_from_csv(
    csv_path: Path,
    source_root: Path,
    destination_root: Path,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> CollectorSummary:
    csv_path = Path(csv_path).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    destination_root = Path(destination_root).expanduser().resolve()
    if not csv_path.is_file():
        raise ValueError("Укажите существующий CSV")
    if not source_root.is_dir():
        raise ValueError("Укажите существующую папку исходных фотографий")
    destination_root.mkdir(parents=True, exist_ok=True)

    header, rows = _read_rows(csv_path)
    person_col = _find_column(
        header,
        ("Идентификатор цели", "Идентификатор человека", "Идентификатор ребенка", "Идентификатор ребёнка"),
        0,
    )
    photo_col = _find_column(
        header,
        ("номер фото с этой целью", "номер фото с этим человеком", "номер фото с этим ребенком", "номер фото с этим ребёнком", "номер фото"),
        1,
    )
    source_col = _find_column(header, ("Исходный файл", "Относительный путь", "Source file"), None)
    if person_col is None or photo_col is None:
        raise ValueError("Не удалось определить столбцы цели и фотографии в CSV")

    people = sorted({row[person_col] for row in rows if len(row) > person_col and row[person_col]}, key=str.casefold)
    folder_names = _unique_target_folder_names(people)
    report_rows: list[list[str]] = []
    copied = skipped = errors = 0
    touched_people: set[str] = set()
    total_rows = len(rows)
    cancelled = False
    stem_index: dict[str, list[Path]] | None = None

    try:
        _raise_if_cancelled(cancel_check)
        if source_col is None:
            stem_index = _build_stem_index(
                source_root,
                destination_root,
                progress=progress,
                cancel_check=cancel_check,
            )

        if progress is not None:
            progress(0, total_rows, "Подготовка…")

        for item_index, (row_number, row) in enumerate(enumerate(rows, start=2), start=1):
            _raise_if_cancelled(cancel_check)
            if len(row) <= max(person_col, photo_col):
                skipped += 1
                report_rows.append([str(row_number), "", "", "пропущено", "неполная строка CSV"])
                if progress is not None:
                    progress(item_index, total_rows, f"Строка {row_number}: пропущена")
                continue
            person = row[person_col].strip()
            photo_id = row[photo_col].strip()
            if not person or not photo_id:
                skipped += 1
                report_rows.append([str(row_number), person, photo_id, "пропущено", "пустая цель или номер фото"])
                if progress is not None:
                    progress(item_index, total_rows, f"Строка {row_number}: пропущена")
                continue

            source: Path | None = None
            reason = ""
            if source_col is not None:
                if len(row) <= source_col:
                    reason = "неполная строка CSV: отсутствует исходный файл"
                else:
                    source_text = row[source_col].strip()
                    source = _source_from_relative(source_root, source_text)
                    if source is None:
                        reason = "исходный путь отсутствует или выходит за выбранную папку"
            else:
                # Индекс по stem строится только для старых CSV без столбца
                # «Исходный файл». Если столбец есть, неполная строка должна
                # быть пропущена, а не переключать весь сборщик на другой режим.
                if stem_index is None:
                    reason = "внутренняя ошибка индекса исходных файлов"
                else:
                    candidates = stem_index.get(Path(photo_id).stem.casefold(), [])
                    if len(candidates) == 1:
                        source = candidates[0]
                    elif not candidates:
                        reason = "файл не найден по номеру фото"
                    else:
                        reason = "неоднозначное имя: найдено несколько файлов с таким stem"

            if source is None:
                skipped += 1
                report_rows.append([str(row_number), person, photo_id, "пропущено", reason])
                if progress is not None:
                    progress(item_index, total_rows, f"{person}: {photo_id} — пропущен")
                continue

            target: Path | None = None
            try:
                _raise_if_cancelled(cancel_check)
                target_dir = destination_root / folder_names[person]
                target_dir.mkdir(parents=True, exist_ok=True)
                target = _unique_destination(target_dir / source.name)
                _copy_file_cancellable(source, target, cancel_check=cancel_check)
                copied += 1
                touched_people.add(person)
                report_rows.append([str(row_number), person, str(source), "скопировано", str(target)])
            except CollectorCancelled:
                report_rows.append([
                    str(row_number), person, str(source), "отменено",
                    f"копирование прервано пользователем{f'; неполный файл удалён: {target}' if target is not None else ''}",
                ])
                raise
            except Exception as exc:
                errors += 1
                report_rows.append([str(row_number), person, str(source), "ошибка", str(exc)])
            finally:
                if progress is not None:
                    progress(item_index, total_rows, f"{person}: {source.name if source is not None else photo_id}")

    except CollectorCancelled:
        cancelled = True
        if not report_rows or report_rows[-1][3] != "отменено":
            report_rows.append(["", "", "", "отменено", "сбор файлов отменён пользователем"])

    report_csv = destination_root / "collector_report.csv"
    with report_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(["Строка CSV", "Цель", "Исходный файл/номер", "Статус", "Результат/причина"])
        writer.writerows(report_rows)

    return CollectorSummary(
        copied=copied,
        skipped=skipped,
        errors=errors,
        target_count=len(touched_people),
        report_csv=report_csv,
        cancelled=cancelled,
    )
