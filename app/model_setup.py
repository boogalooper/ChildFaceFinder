from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
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
            if destination.stat().st_size < 10_000_000:
                raise RuntimeError("загруженный архив слишком мал")
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


def ensure_antelopev2(force: bool = False, allow_download: bool = False) -> Path:
    if model_is_valid() and not force:
        print(f"Модель уже установлена: {MODEL_DIR}")
        return MODEL_DIR

    if not allow_download:
        raise RuntimeError(
            "Модель antelopev2 не найдена или повреждена. Обычный запуск работает полностью офлайн "
            "и не загружает модели автоматически. Восстановите папку models\\insightface\\models\\antelopev2 "
            "из резервной/portable-копии либо запустите install.bat при наличии интернета."
        )

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

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

        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        shutil.copytree(staged, MODEL_DIR)

    if not model_is_valid():
        raise RuntimeError("Модель была скопирована, но итоговая проверка не пройдена")
    print(f"Модель установлена: {MODEL_DIR}")
    return MODEL_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="явно разрешить загрузку antelopev2 из интернета",
    )
    args = parser.parse_args()
    try:
        ensure_antelopev2(force=args.force, allow_download=args.download or args.force)
        return 0
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
