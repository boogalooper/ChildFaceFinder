from __future__ import annotations

import sys

from engine import FaceEngine, Settings


def main() -> int:
    try:
        print("Финальная проверка InsightFace на GPU...")
        engine = FaceEngine(Settings(require_gpu=True), print)
        engine.initialize()
        if engine.provider != "CUDAExecutionProvider":
            raise RuntimeError(f"Ожидался CUDAExecutionProvider, получен: {engine.provider}")
        print("InsightFace + antelopev2 + CUDA: OK")
        return 0
    except Exception as exc:
        print(f"ОШИБКА GPU smoke-test: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
