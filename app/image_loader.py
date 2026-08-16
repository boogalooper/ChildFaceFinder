from __future__ import annotations

import ctypes
import os
import re
import shutil
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
import rawpy

# pillow-heif 1.5 exposes the HEIF/HEIC Pillow plugin through this opener.
# AVIF is handled by modern Pillow when supported by its Windows wheel.
register_heif_opener()

RAW_EXTENSIONS = {
    ".3fr", ".ari", ".arw", ".bay", ".braw", ".cap", ".cr2", ".cr3",
    ".crw", ".dcr", ".dcs", ".dng", ".drf", ".eip", ".erf", ".fff",
    ".gpr", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos", ".mrw",
    ".nef", ".nrw", ".orf", ".pef", ".ptx", ".pxn", ".r3d", ".raf",
    ".raw", ".rw2", ".rwl", ".rwz", ".sr2", ".srf", ".srw", ".x3f",
}
HEIF_EXTENSIONS = {".heic", ".heif", ".avif"}

# RAW не проявляется в полном разрешении без необходимости. Сначала используем
# встроенный JPEG-preview камеры; если он слишком мал/отсутствует — быстрый
# half-size postprocess. Итоговую рабочую копию ограничиваем по длинной стороне.
RAW_WORKING_MAX_SIDE = 4096
RAW_EMBEDDED_PREVIEW_MIN_SIDE = 2400

# Файлы этих типов читаются напрямую. RAW использует embedded preview/half-size
# в памяти; HEIF и некоторые неизвестные форматы сохраняют TEMP-нормализацию.
DIRECT_EXTENSIONS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".dib", ".tif", ".tiff",
    ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".pnm", ".ico",
}


SKIP_EXTENSIONS = {
    ".csv", ".txt", ".json", ".xml", ".yaml", ".yml", ".ini", ".log",
    ".md", ".py", ".pyw", ".bat", ".cmd", ".ps1", ".exe", ".dll",
    ".zip", ".7z", ".rar", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".xmp",
}

# Runtime TEMP directories carry the creating PID. This lets a later run remove
# leftovers after a crash without touching a concurrently running instance.
RUNTIME_TEMP_PREFIX = "ChildFaceFinder_run_"
RUNTIME_TEMP_RE = re.compile(r"^ChildFaceFinder_run_(\d+)_")
LEGACY_TEMP_PREFIX = "ChildFaceFinder_"
LEGACY_TEMP_MIN_AGE_SECONDS = 6 * 60 * 60
AUX_TEMP_MIN_AGE_SECONDS = 24 * 60 * 60


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True

    if os.name == "nt":
        # OpenProcess + GetExitCodeProcess avoids extra dependencies such as psutil.
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                # If Windows refuses the query after opening the process, be safe
                # and consider it active rather than deleting its TEMP directory.
                return True
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def cleanup_stale_temp_dirs() -> tuple[list[Path], list[tuple[Path, str]]]:
    """
    Removes stale Child Face Finder directories from the OS TEMP directory.

    New runtime directories embed the creator PID and are removed as soon as
    that process no longer exists. Legacy runtime folders (older versions had
    no PID) are removed only when older than 6 hours. Installer/model download
    folders are treated even more conservatively and removed after 24 hours.
    Active runtime folders are never deliberately removed.
    """
    temp_root = Path(tempfile.gettempdir())
    now = time.time()
    removed: list[Path] = []
    failed: list[tuple[Path, str]] = []

    try:
        entries = list(temp_root.iterdir())
    except OSError as exc:
        return removed, [(temp_root, str(exc))]

    for entry in entries:
        name = entry.name
        if not name.startswith(LEGACY_TEMP_PREFIX):
            continue
        try:
            if entry.is_symlink() or not entry.is_dir():
                continue
        except OSError:
            continue

        should_remove = False
        match = RUNTIME_TEMP_RE.match(name)
        if match:
            owner_pid = int(match.group(1))
            should_remove = not _pid_is_running(owner_pid)
        else:
            # Previous versions used ChildFaceFinder_<random>. Do not delete a
            # recent directory because an older build may still be running.
            # Model/install temp folders can contain large downloads, but use a
            # longer grace period for safety.
            try:
                age = max(0.0, now - entry.stat().st_mtime)
            except OSError:
                continue
            if name.startswith(("ChildFaceFinder_model_", "ChildFaceFinder_install_")):
                should_remove = age >= AUX_TEMP_MIN_AGE_SECONDS
            else:
                should_remove = age >= LEGACY_TEMP_MIN_AGE_SECONDS

        if not should_remove:
            continue
        try:
            shutil.rmtree(entry)
            removed.append(entry)
        except FileNotFoundError:
            pass
        except OSError as exc:
            failed.append((entry, str(exc)))

    return removed, failed


def iter_candidate_files(folder: Path, recursive: bool = True) -> list[Path]:
    """
    Возвращает вероятные изображения. Известные служебные/документные форматы
    пропускаются сразу; неизвестное расширение оставляем декодеру, чтобы не
    потерять редкий графический/RAW-формат.
    """
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        (
            p for p in iterator
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.casefold() not in SKIP_EXTENSIONS
        ),
        key=lambda p: str(p).casefold(),
    )


def _pil_to_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        # Для анимированных форматов используем первый кадр.
        try:
            img.seek(0)
        except Exception:
            pass
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _resize_max_side(image_bgr: np.ndarray, max_side: int = RAW_WORKING_MAX_SIDE) -> np.ndarray:
    """Уменьшает рабочую копию без увеличения маленьких изображений."""
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image_bgr
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def _embedded_raw_preview(path: Path) -> np.ndarray | None:
    """
    Извлекает готовый preview/thumbnail из RAW без проявки сенсорных данных.
    rawpy/LibRaw возвращает наибольший доступный embedded preview.
    """
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
    except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
        return None

    if thumb.format == rawpy.ThumbFormat.JPEG:
        # Через Pillow, чтобы корректно применить EXIF Orientation embedded JPEG.
        with Image.open(BytesIO(thumb.data)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif thumb.format == rawpy.ThumbFormat.BITMAP:
        data = np.asarray(thumb.data)
        if data.ndim != 3 or data.shape[2] < 3:
            return None
        image = cv2.cvtColor(data[:, :, :3].astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
    else:
        return None

    if image.size == 0:
        return None
    return image


def _raw_half_size_to_bgr(path: Path) -> np.ndarray:
    """Быстрый fallback: проявляет RAW сразу в половинном разрешении."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            half_size=True,
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=False,
            output_bps=8,
            gamma=(2.222, 4.5),
        )
    if rgb is None or rgb.size == 0:
        raise ValueError("LibRaw вернул пустое изображение")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _raw_to_bgr(path: Path) -> np.ndarray:
    preview = _embedded_raw_preview(path)
    if preview is not None and max(preview.shape[:2]) >= RAW_EMBEDDED_PREVIEW_MIN_SIDE:
        return _resize_max_side(preview)

    # Небольшой embedded thumbnail может быть слишком мал для групповых фото.
    # В этом случае используем half-size RAW — медленнее preview, но гораздо
    # быстрее полной проявки и достаточно детально для tiled face detection.
    return _resize_max_side(_raw_half_size_to_bgr(path))


def _roundtrip_temp_jpeg(image_bgr: np.ndarray, temp_dir: Path) -> np.ndarray:
    """
    Нормализация через JPEG во временном каталоге ОС.
    Файл удаляется в finally сразу после декодирования обратно.
    """
    temp_path = temp_dir / f"norm_{uuid.uuid4().hex}.jpg"
    try:
        ok, encoded = cv2.imencode(
            ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96]
        )
        if not ok:
            raise ValueError("Не удалось закодировать временный JPEG")
        temp_path.write_bytes(encoded.tobytes())
        data = np.fromfile(str(temp_path), dtype=np.uint8)
        normalized = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if normalized is None:
            raise ValueError("Не удалось прочитать временный JPEG")
        return normalized
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def decode_image(path: Path, temp_dir: Path) -> np.ndarray:
    """
    Декодирует изображение в BGR uint8.

    RAW обрабатывается по быстрому пути: embedded JPEG-preview, затем при
    необходимости half-size LibRaw. Рабочая копия ограничивается 4096 px по
    длинной стороне и не требует промежуточного JPEG на диске.

    HEIF/HEIC/AVIF и неизвестные форматы сохраняют прежнюю нормализацию через
    системную TEMP-папку. Неизвестные расширения сначала пробуются Pillow,
    затем LibRaw.
    """
    suffix = path.suffix.casefold()

    if suffix in RAW_EXTENSIONS:
        return _raw_to_bgr(path)

    try:
        image = _pil_to_bgr(path)
        if suffix in HEIF_EXTENSIONS or suffix not in DIRECT_EXTENSIONS:
            image = _roundtrip_temp_jpeg(image, temp_dir)
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        # Некоторые RAW имеют необычные расширения. Последняя попытка — LibRaw.
        try:
            return _raw_to_bgr(path)
        except Exception as raw_error:
            raise ValueError(f"Неподдерживаемый/повреждённый файл: {raw_error}") from raw_error


def make_system_temp_dir() -> tempfile.TemporaryDirectory[str]:
    """Creates a PID-tagged runtime directory in the OS TEMP/TMP folder."""
    return tempfile.TemporaryDirectory(prefix=f"{RUNTIME_TEMP_PREFIX}{os.getpid()}_")
