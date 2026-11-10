"""Export recognized photographs and sidecars without overwriting user files."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from collector import _unique_target_folder_names
from metadata_writer import XmpMetadataWriter
from runtime_safety import atomic_path


@dataclass
class FileOutputSummary:
    copies: int = 0
    moved: int = 0
    failed: int = 0
    skipped: int = 0
    journal: Path | None = None


def _signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def _digest(path, cancel):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            cancel()
            block = stream.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def _copy_verified(source, target, cancel):
    before = _signature(source)
    created = False
    digest = hashlib.sha256()
    try:
        with source.open('rb') as src, target.open('xb') as dst:
            created = True
            while True:
                cancel()
                block = src.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                dst.write(block)
            dst.flush()
            import os
            os.fsync(dst.fileno())
        if _signature(source) != before or _digest(target, cancel) != digest.hexdigest():
            raise RuntimeError('Файл изменился или проверка копии не пройдена')
        shutil.copystat(source, target)
        return before, digest.hexdigest()
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise


def _target_path(folder, source):
    # Reserve the whole basename, including a possible future sidecar.
    existing = {p.stem.casefold() for p in folder.iterdir()}
    stem = source.stem
    index = 1
    while stem.casefold() in existing:
        index += 1
        stem = f'{source.stem} ({index})'
    return folder / (stem + source.suffix)


def export_people(people_by_photo, source_root, destination, *, move=False,
                  all_photos=None, blocked=None, cancel=lambda: None, log=lambda text: None,
                  progress=lambda done, total, name: None):
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise ValueError('Папки съёмки и результата должны находиться отдельно друг от друга')
    total = len(people_by_photo)
    stage = "Перемещение" if move else "Копирование"
    progress(0, total, f"{stage}: подготовка вывода")
    destination.mkdir(parents=True, exist_ok=True)
    people = sorted({p for names in people_by_photo.values() for p in names}, key=str.casefold)
    folders = _unique_target_folder_names(people)
    siblings = {}
    for image in all_photos or people_by_photo:
        siblings.setdefault((str(image.parent.resolve()).casefold(), image.stem.casefold()), set()).add(image)
    summary = FileOutputSummary(journal=destination / f'ChildFaceFinder_transfer_{uuid.uuid4().hex[:10]}.json')
    journal = {'mode': 'move' if move else 'copy', 'source': str(source_root), 'items': []}
    def save():
        with atomic_path(summary.journal) as temporary:
            temporary.write_text(json.dumps(journal, ensure_ascii=False, indent=2), encoding='utf-8')
    for done, (image, names) in enumerate(sorted(people_by_photo.items(), key=lambda item: str(item[0]).casefold()), 1):
        cancel()
        image = Path(image)
        progress(done - 1, total, f"{stage}: {image.name}")
        item = {'source': str(image), 'copies': [], 'status': 'preparing'}
        journal['items'].append(item)
        try:
            if image.resolve() == source_root or source_root not in image.resolve().parents or image.is_symlink():
                raise ValueError('Исходный файл находится вне папки съёмки или является ссылкой')
            sidecar = XmpMetadataWriter._sidecar_path(image)
            has_xmp = sidecar.exists()
            if sidecar.is_symlink():
                raise ValueError('XMP является ссылкой; перенос пропущен')
            if move and image in (blocked or set()):
                raise ValueError('XMP не был успешно обновлён: исходник не перемещён')
            if move and has_xmp and len(siblings.get((str(image.parent.resolve()).casefold(), image.stem.casefold()), ())) > 1:
                raise ValueError('Общий XMP у нескольких изображений: перемещение пропущено')
            originals = [image] + ([sidecar] if has_xmp else [])
            versions = {p: _signature(p) for p in originals}
            digests = {}
            for person in sorted(names, key=str.casefold):
                cancel()
                folder = destination / folders[person]
                folder.mkdir(exist_ok=True)
                if folder.resolve().parent != destination:
                    raise ValueError('Папка человека является внешней ссылкой')
                target = _target_path(folder, image)
                targets = [target] + ([target.with_suffix('.xmp')] if has_xmp else [])
                record = {'person': person, 'files': [str(p) for p in targets], 'status': 'pending'}
                item['copies'].append(record)
                save()  # durable recovery plan before any source deletion
                made = []
                try:
                    for src, dst in zip(originals, targets):
                        signature, digest = _copy_verified(src, dst, cancel)
                        made.append(dst)
                        if signature != versions[src] or (src in digests and digests[src] != digest):
                            raise RuntimeError('Исходник изменился между копиями')
                        digests[src] = digest
                except BaseException:
                    for path in made:
                        path.unlink(missing_ok=True)
                    record['status'] = 'failed'
                    raise
                record['status'] = 'verified'
                summary.copies += 1
            # The entire image+XMP pair has been verified in EVERY person's folder.
            cancel()
            if has_xmp != sidecar.exists():
                raise RuntimeError('Состав пары фото/XMP изменился во время копирования')
            for src in originals:
                if _signature(src) != versions[src] or (move and _digest(src, cancel) != digests[src]):
                    raise RuntimeError('Исходник изменился; удаление запрещено')
            if move and names:
                item['status'] = 'verified_before_delete'
                save()
                cancel()
                # Finish this pair before observing the next cancellation request.
                for src in originals:
                    src.unlink()
                summary.moved += 1
                item['status'] = 'moved'
            else:
                item['status'] = 'copied'
        except Exception as exc:
            # Cancellation callbacks raise; re-check outside error accounting.
            try:
                cancel()
            except BaseException:
                item['status'] = 'cancelled'
                save()
                raise
            item['status'] = 'error'
            item['error'] = str(exc)
            summary.failed += 1
            log(f'Файлы: {image.name}: {exc}')
        except BaseException:
            item['status'] = 'cancelled'
            save()
            raise
        save()
        progress(done, total, f"{stage}: {image.name}")
    if not summary.journal.exists():
        save()
    return summary
