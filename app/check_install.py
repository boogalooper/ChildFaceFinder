from __future__ import annotations

import struct
import sys
from importlib import metadata


EXPECTED = {
    "insightface": "1.0.1",
    "onnxruntime-gpu": "1.26.0",
    "nvidia-cudnn-cu12": "9.7.1.26",
    "numpy": "1.26.4",
    "onnx": "1.18.0",
    "opencv-python-headless": "4.11.0.86",
    "scipy": "1.14.1",
    "scikit-image": "0.25.2",
    "tqdm": "4.67.1",
    "requests": "2.32.3",
    "Pillow": "12.3.0",
    "pillow-heif": "1.5.0",
    "rawpy": "0.27.0",
}


def _require_version(distribution: str, expected: str) -> None:
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Пакет {distribution} не установлен") from exc
    if actual != expected:
        raise RuntimeError(f"Ожидается {distribution} {expected}, установлен {actual}")


def _require_absent(distribution: str, reason: str) -> None:
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return
    raise RuntimeError(f"Лишний пакет {distribution} {actual}: {reason}")


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    if sys.platform != "win32" or struct.calcsize("P") * 8 != 64:
        raise RuntimeError("Ожидается 64-битная Windows")
    if sys.version_info[:3] != (3, 11, 16):
        raise RuntimeError("Ожидается Python 3.11.16")

    for distribution, version in EXPECTED.items():
        _require_version(distribution, version)

    # Эти две distribution нельзя ставить параллельно с выбранными вариантами:
    # они кладут файлы в те же import-пакеты onnxruntime/cv2.
    _require_absent(
        "onnxruntime",
        "должен быть установлен только onnxruntime-gpu",
    )
    _require_absent(
        "opencv-python",
        "должен быть установлен только opencv-python-headless",
    )

    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.destroy()
    print("Tkinter: OK")

    import cv2
    import numpy
    import onnx
    import rawpy
    import PIL
    from PIL import features as pil_features
    import pillow_heif
    import scipy
    import skimage
    import insightface
    import onnxruntime as ort

    # ORT умеет загрузить CUDA/cuDNN DLL из nvidia-* wheel-пакетов,
    # CUDA-библиотеки ставятся extra [cuda], а cuDNN закреплён отдельным пакетом.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    providers = ort.get_available_providers()
    print("ONNX Runtime:", ort.__version__)
    print("cuDNN wheel:", metadata.version("nvidia-cudnn-cu12"))
    print("Providers:", providers)
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider не найден. Обновите драйвер NVIDIA и повторите install.bat."
        )

    print("InsightFace:", getattr(insightface, "__version__", metadata.version("insightface")))
    print("OpenCV:", cv2.__version__)
    print("NumPy:", numpy.__version__)
    print("ONNX:", onnx.__version__)
    print("SciPy:", scipy.__version__)
    print("scikit-image:", skimage.__version__)
    print("Pillow:", PIL.__version__)
    if not pil_features.check_module("avif"):
        raise RuntimeError("Windows wheel Pillow не содержит поддержку AVIF")
    print("Pillow AVIF: OK")
    print("rawpy:", getattr(rawpy, "__version__", "OK"))
    print("pillow-heif:", getattr(pillow_heif, "__version__", "OK"))
    print("GPU-зависимости: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
