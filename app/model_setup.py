from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
INSIGHTFACE_ROOT = PROJECT_DIR / "models" / "insightface"
MODEL_NAME = "antelopev2"
MODEL_DIR = INSIGHTFACE_ROOT / "models" / MODEL_NAME
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"


def model_is_valid(path: Path = MODEL_DIR) -> bool:
    if not path.is_dir():
        return False
    onnx_files = list(path.glob("*.onnx"))
    # antelopev2 содержит detection, recognition, 2d106, 3d68, gender/age.
    return len(onnx_files) >= 5 and all(p.stat().st_size > 100_000 for p in onnx_files)


def _download(url: str, destination: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ChildFaceFinder/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response, destination.open("wb") as out:
                total = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                last_print = 0.0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= 0.5:
                        if total:
                            pct = downloaded * 100.0 / total
                            print(f"\rЗагрузка antelopev2: {pct:5.1f}%", end="", flush=True)
                        else:
                            print(f"\rЗагрузка antelopev2: {downloaded / 1048576:.0f} MB", end="", flush=True)
                        last_print = now
                print()
            actual_size = destination.stat().st_size
            if actual_size < 10_000_000:
                raise RuntimeError("загруженный архив слишком мал")
            if total and actual_size != total:
                raise RuntimeError(
                    f"неполная загрузка модели: получено {actual_size} байт из {total}"
                )

            # Проверяем ZIP до выхода из retry-цикла. Иначе оборванная загрузка,
            # которая закончилась обычным EOF и осталась >10 MB, падала бы уже
            # на распаковке без повторной попытки скачать архив.
            try:
                with zipfile.ZipFile(destination) as zf:
                    broken_member = zf.testzip()
                    if broken_member is not None:
                        raise RuntimeError(
                            f"повреждён файл внутри архива модели: {broken_member}"
                        )
            except zipfile.BadZipFile as exc:
                raise RuntimeError("загруженный файл модели не является корректным ZIP") from exc
            return
        except Exception as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt < retries:
                print(f"Ошибка загрузки ({exc}), повтор {attempt}/{retries}...")
                time.sleep(2 * attempt)
    raise RuntimeError(f"Не удалось загрузить модель: {last_error}")


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError("Некорректный путь внутри архива модели")
        zf.extractall(dest)


def _recover_interrupted_model_swap() -> None:
    """Restore a valid previous model if Windows/power interrupted the swap."""
    parent = MODEL_DIR.parent
    if not parent.exists():
        return

    # Incomplete candidates were never made active and are safe to remove.
    for candidate in parent.glob(f".{MODEL_NAME}.new.*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)

    backups = sorted(
        (p for p in parent.glob(f".{MODEL_NAME}.previous.*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if MODEL_DIR.exists():
        # A valid active model wins. Old backup leftovers are non-fatal and may
        # simply have been locked by antivirus during cleanup.
        return

    for backup in backups:
        if model_is_valid(backup):
            backup.rename(MODEL_DIR)
            print(f"Восстановлена модель после прерванной предыдущей установки: {MODEL_DIR}")
            return



def ensure_antelopev2(force: bool = False) -> Path:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_model_swap()

    if model_is_valid() and not force:
        print(f"Модель уже установлена: {MODEL_DIR}")
        return MODEL_DIR

    # Все промежуточные файлы находятся в системном TEMP и удаляются контекстом.
    with tempfile.TemporaryDirectory(prefix="ChildFaceFinder_model_") as tmp_name:
        tmp = Path(tmp_name)
        archive = tmp / "antelopev2.zip"
        extracted = tmp / "extracted"
        _download(MODEL_URL, archive)
        print("Распаковка модели...")
        _safe_extract(archive, extracted)

        candidates = [p for p in extracted.rglob("*") if p.is_dir() and len(list(p.glob("*.onnx"))) >= 5]
        if not candidates:
            raise RuntimeError("В архиве antelopev2 не найден ожидаемый набор ONNX-моделей")
        source = max(candidates, key=lambda p: len(list(p.glob("*.onnx"))))

        staged = tmp / MODEL_NAME
        staged.mkdir()
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, staged / item.name)

        if not model_is_valid(staged):
            raise RuntimeError("Проверка распакованной модели не пройдена")

        # Install transactionally. First copy a complete verified candidate next
        # to the final model directory; only then swap it into place. This keeps
        # an existing working model intact if copying is interrupted.
        install_candidate = MODEL_DIR.parent / f".{MODEL_NAME}.new.{uuid.uuid4().hex}"
        backup = MODEL_DIR.parent / f".{MODEL_NAME}.previous.{uuid.uuid4().hex}"
        try:
            shutil.copytree(staged, install_candidate)
            if not model_is_valid(install_candidate):
                raise RuntimeError("Проверка подготовленной копии модели не пройдена")

            had_old_model = MODEL_DIR.exists()
            if had_old_model:
                MODEL_DIR.rename(backup)
            try:
                install_candidate.rename(MODEL_DIR)
            except Exception:
                if had_old_model and backup.exists() and not MODEL_DIR.exists():
                    backup.rename(MODEL_DIR)
                raise

            if backup.exists():
                try:
                    shutil.rmtree(backup)
                except OSError as exc:
                    print(f"Предупреждение: новая модель установлена, но старую резервную копию не удалось удалить: {backup} ({exc})")
        finally:
            if install_candidate.exists():
                shutil.rmtree(install_candidate, ignore_errors=True)
            # A backup should only remain if restoring failed. Do not delete it:
            # it may be the last intact copy and the next run can report/repair it.

    if not model_is_valid():
        raise RuntimeError("Модель была скопирована, но итоговая проверка не пройдена")
    print(f"Модель установлена: {MODEL_DIR}")
    return MODEL_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        ensure_antelopev2(force=args.force)
        return 0
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
