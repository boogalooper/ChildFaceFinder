from __future__ import annotations

import copy
import csv
import math
import os
import queue
import re
import threading
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from image_loader import cleanup_stale_temp_dirs, decode_image, iter_candidate_files, make_system_temp_dir
from model_setup import INSIGHTFACE_ROOT, MODEL_NAME, ensure_antelopev2
from series_selector import BestFrameCandidate, select_best_series_frames

cv2.setNumThreads(1)

RIGHT_EYE_106 = slice(33, 43)
LEFT_EYE_106 = slice(87, 97)
LEADING_REFERENCE_NUMBER = re.compile(r"^\s*\d+\s+(.+?)\s*$")
LEADING_REFERENCE_LETTER = re.compile(r"^\s*[A-Z]\s+(.+?)\s*$")


@dataclass(slots=True)
class Settings:
    match_threshold: float = 0.45
    ambiguity_margin: float = 0.04
    detector_threshold: float = 0.18
    rescue_detector_threshold: float = 0.10
    rescue_max_candidates: int = 80
    recognition_batch_size: int = 64

    min_face_px: int = 45
    max_yaw: float = 45.0
    max_pitch: float = 35.0
    max_roll: float = 35.0
    eye_ratio_min: float = 0.105
    blur_threshold: float = 45.0
    max_clipped_fraction: float = 0.55

    reject_closed_eyes: bool = True
    reject_blur: bool = True
    reject_pose: bool = True
    reject_small_face: bool = True
    reject_exposure: bool = False
    reject_head_clipping: bool = True
    head_top_margin_ratio: float = 0.18
    head_side_margin_ratio: float = 0.07
    chin_margin_ratio: float = 0.03

    # Дополнительная композиционная зона у реальной границы кадра.
    # Она масштабируется по доле кадра, занимаемой bbox лица: маленькому лицу
    # требуется больший относительный отступ, крупному портрету — меньший.
    edge_guard_base_ratio: float = 0.020
    edge_guard_min_ratio: float = 0.008
    edge_guard_max_ratio: float = 0.050
    edge_guard_reference_face_area: float = 0.015
    edge_guard_size_exponent: float = 0.50

    tile_enabled: bool = True
    tile_size: int = 2200
    tile_overlap: float = 0.18
    tile_trigger_px: int = 1800
    nms_iou_threshold: float = 0.35

    decode_workers: int = max(2, min(8, os.cpu_count() or 4))
    inference_workers: int = 2
    recursive: bool = True
    require_gpu: bool = True
    verbose_diagnostics: bool = True

    select_best_series: bool = False
    series_max_gap_seconds: float = 12.0
    series_max_filename_gap: int = 5

    reference_min_det_score: float = 0.45
    reference_consistency_warn: float = 0.55


@dataclass(slots=True)
class FaceCandidate:
    bbox: tuple[int, int, int, int]
    raw_bbox: tuple[float, float, float, float]
    kps: np.ndarray
    det_score: float
    source: str = "full"


@dataclass(slots=True)
class FaceRecord:
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]
    raw_bbox: tuple[float, float, float, float]
    det_score: float
    quality_reasons: list[str] = field(default_factory=list)
    blur_score: float | None = None
    eye_left: float | None = None
    eye_right: float | None = None
    pose: tuple[float, float, float] | None = None
    source: str = "full"


@dataclass(slots=True)
class ImageAnalysis:
    path: Path
    faces: list[FaceRecord]
    error: str | None = None
    full_detected: int = 0
    tile_detected: int = 0
    tiles_used: int = 0


@dataclass(slots=True)
class ReferenceSet:
    person_ids: list[str]
    matrix: np.ndarray
    ref_person_indices: np.ndarray
    file_count: int


@dataclass(slots=True)
class RunSummary:
    reference_ids: int
    reference_files: int
    photo_count: int
    matched_pairs: int
    rejected_pairs: int
    review_rows: int
    output_csv: Path
    rejected_csv: Path
    review_csv: Path
    best_csv: Path | None = None
    best_series_count: int = 0


class CancelledError(RuntimeError):
    pass


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("пустой embedding")
    return arr / norm


def _eye_openness(points: np.ndarray) -> float | None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (10, 2) or not np.isfinite(pts).all():
        return None
    centered = pts - pts.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(s) < 2 or s[0] <= 1e-6:
        return None
    return float(s[1] / s[0])


def _safe_crop(image: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.08) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    return image[y1:y2, x1:x2]


def _blur_score(face_crop: np.ndarray) -> float | None:
    if face_crop.size == 0:
        return None
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > 256:
        scale = 256.0 / max(h, w)
        gray = cv2.resize(
            gray,
            (max(8, int(w * scale)), max(8, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_bad(face_crop: np.ndarray, max_clipped_fraction: float) -> tuple[bool, float, float]:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return True, 1.0, 0.0
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    mean = float(gray.mean())
    bad = clipped > max_clipped_fraction or mean < 22.0 or mean > 238.0
    return bad, clipped, mean


def _normalized_identity_key(name: str) -> str:
    text = unicodedata.normalize("NFKC", name)
    text = " ".join(text.split())
    return text.casefold().replace("ё", "е")


def parse_reference_identifier(path: Path) -> str:
    """
    Возвращает имя из basename, не учитывая каталог и необязательный
    ведущий числовой или односимвольный латинский индекс:

    '712 Котов Никита.jpg' -> 'Котов Никита'
    'A Котов Никита.jpg'   -> 'Котов Никита'
    'Котов Никита.jpg'     -> 'Котов Никита'
    'Smith John.jpg'       -> 'Smith John'
    'портрет/Котов.jpg'    -> 'Котов'

    Латиница в обычной части имени сохраняется полностью. Удаляется только
    отдельная ведущая заглавная латинская буква A-Z, используемая как индекс.
    Подкаталоги никогда не входят в ID цели.
    """
    stem = unicodedata.normalize("NFKC", path.stem).strip()
    match = LEADING_REFERENCE_NUMBER.match(stem)
    if not match:
        match = LEADING_REFERENCE_LETTER.match(stem)
    if match:
        stem = match.group(1).strip()
    return " ".join(stem.split())


def _identity_tokens(name: str) -> tuple[list[str], list[str]]:
    """Отображаемые и нормализованные токены имени, разделённые пробелами."""
    display = " ".join(unicodedata.normalize("NFKC", name).split())
    shown = display.split()
    normalized = [_normalized_identity_key(token) for token in shown]
    return shown, normalized


def _longest_common_token_run(a: str, b: str) -> tuple[int, int, int]:
    """
    Возвращает (длина, start_a, start_b) самой длинной общей непрерывной
    последовательности слов. Сравнение нечувствительно к регистру и е/ё.
    """
    _, ta = _identity_tokens(a)
    _, tb = _identity_tokens(b)
    best_len = best_a = best_b = 0
    for ia in range(len(ta)):
        for ib in range(len(tb)):
            run = 0
            while ia + run < len(ta) and ib + run < len(tb) and ta[ia + run] == tb[ib + run]:
                run += 1
            if run > best_len:
                best_len, best_a, best_b = run, ia, ib
    return best_len, best_a, best_b


def _common_reference_identity(a: str, b: str) -> tuple[str, tuple[int, float]] | None:
    """
    Находит безопасную общую часть двух имён эталонов.

    Разрешаем:
    - точное совпадение нормализованных имён;
    - общую последовательность из двух и более слов;
    - одно общее слово, только если одно из имён целиком состоит из этого слова
      (например, 'Котов' + 'Котов Никита').

    Возвращает отображаемый ID и score для разрешения неоднозначностей.
    """
    if _normalized_identity_key(a) == _normalized_identity_key(b):
        words = a.split()
        return a, (len(words), 1.0)

    shown_a, norm_a = _identity_tokens(a)
    shown_b, norm_b = _identity_tokens(b)
    common_len, start_a, _ = _longest_common_token_run(a, b)
    if common_len <= 0:
        return None
    if common_len == 1 and len(norm_a) != 1 and len(norm_b) != 1:
        return None

    common_display = " ".join(shown_a[start_a:start_a + common_len])
    coverage = common_len / max(1, min(len(norm_a), len(norm_b)))
    return common_display, (common_len, coverage)


def _contains_identity_sequence(name: str, sequence_key: str) -> bool:
    """Проверяет, содержит ли имя заданную нормализованную последовательность слов."""
    _, tokens = _identity_tokens(name)
    sequence = sequence_key.split()
    if not sequence or len(sequence) > len(tokens):
        return False
    for start in range(len(tokens) - len(sequence) + 1):
        if tokens[start:start + len(sequence)] == sequence:
            return True
    return False


def group_reference_paths(paths: list[Path]) -> list[tuple[str, list[Path]]]:
    """
    Группирует любое количество эталонов одного цели.

    1. Все точные нормализованные basename объединяются без ограничения количества.
    2. Различающиеся имена могут объединяться по однозначной общей части из двух
       и более слов (например, «Котов Никита портрет/стоя/улица»).
    3. Однословная общая часть разрешается только для однозначной пары, например
       «Котов» + «Котов Никита». Если «Котов» подходит сразу к нескольким людям,
       программа не угадывает.
    """
    exact: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path in paths:
        display = parse_reference_identifier(path)
        if not display:
            raise RuntimeError(f"Не удалось получить ID цели из имени: {path.name}")
        exact[_normalized_identity_key(display)].append((path, display))

    # Каждый точный нормализованный ID становится узлом. Внутри узла уже может
    # быть сколько угодно файлов из разных подпапок.
    nodes: list[tuple[str, list[Path]]] = []
    for items in exact.values():
        display = items[0][1]
        nodes.append((display, [path for path, _ in items]))
    nodes.sort(key=lambda item: item[0].casefold())

    # Собираем возможные общие идентификаторы между РАЗНЫМИ точными именами.
    # Затем расширяем каждый кандидат на все узлы, содержащие эту же
    # непрерывную последовательность слов.
    candidate_display: dict[str, str] = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            common = _common_reference_identity(nodes[i][0], nodes[j][0])
            if common is None:
                continue
            display, _score = common
            key = _normalized_identity_key(display)
            candidate_display.setdefault(key, display)

    candidate_members: dict[str, set[int]] = {}
    invalid_single_word_candidates: set[str] = set()
    for key, display in candidate_display.items():
        members = {idx for idx, (name, _) in enumerate(nodes) if _contains_identity_sequence(name, key)}
        if len(members) < 2:
            continue
        token_count = len(key.split())
        # Одно слово безопасно только для одной пары. Три и более разных имён
        # («Котов», «Котов Никита», «Котов Сергей») уже неоднозначны.
        if token_count == 1 and len(members) > 2:
            invalid_single_word_candidates.add(key)
            continue
        candidate_members[key] = members

    if invalid_single_word_candidates:
        names: set[str] = set()
        for key in invalid_single_word_candidates:
            for idx, (name, _) in enumerate(nodes):
                if _contains_identity_sequence(name, key):
                    names.add(name)
        raise RuntimeError(
            "Неоднозначные имена эталонов: короткая общая часть подходит сразу к нескольким людям. "
            "Уточните имена файлов: " + ", ".join(sorted(names, key=str.casefold))
        )

    # Для каждого узла выбираем только самый длинный общий ID. Если существует
    # несколько разных кандидатов одинаковой максимальной длины — не угадываем.
    choices: dict[int, str] = {}
    ambiguous_names: set[str] = set()
    for idx, (name, _) in enumerate(nodes):
        candidates = [key for key, members in candidate_members.items() if idx in members]
        if not candidates:
            continue
        max_tokens = max(len(key.split()) for key in candidates)
        best = sorted({key for key in candidates if len(key.split()) == max_tokens})
        if len(best) == 1:
            choices[idx] = best[0]
        else:
            ambiguous_names.add(name)

    if ambiguous_names:
        raise RuntimeError(
            "Неоднозначные имена эталонов: найдено несколько одинаково подходящих общих частей. "
            "Уточните имена файлов: " + ", ".join(sorted(ambiguous_names, key=str.casefold))
        )

    final_groups: list[tuple[str, list[Path]]] = []
    used: set[int] = set()
    for key, members in sorted(candidate_members.items(), key=lambda item: (-len(item[0].split()), item[0])):
        # Группа принимается только если каждый её участник сам выбрал именно
        # этот ID как лучший. Это не позволяет широкой общей части поглотить
        # более точную группу.
        selected = {idx for idx in members if choices.get(idx) == key}
        if len(selected) < 2 or selected != members or selected & used:
            continue
        group_paths: list[Path] = []
        for idx in sorted(selected):
            group_paths.extend(nodes[idx][1])
        final_groups.append((candidate_display[key], group_paths))
        used.update(selected)

    # Узлы без безопасного общего кандидата остаются самостоятельными группами.
    for idx, (display, group_paths) in enumerate(nodes):
        if idx not in used:
            final_groups.append((display, group_paths))

    final_groups.sort(key=lambda item: item[0].casefold())
    return final_groups


def _tile_positions(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    last = length - tile_size
    positions = list(range(0, last + 1, step))
    if positions[-1] != last:
        positions.append(last)
    return positions


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return float(inter / (area_a + area_b - inter))


def _nms_records(records: list[FaceRecord | FaceCandidate], threshold: float) -> list[FaceRecord | FaceCandidate]:
    if len(records) <= 1:
        return records
    # Сначала оставляем наиболее уверенный bbox. При равном score отдаём
    # предпочтение более крупному bbox: у него обычно меньше обрезаны края лица.
    ordered = sorted(
        records,
        key=lambda r: (r.det_score, (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])),
        reverse=True,
    )
    kept: list[FaceRecord | FaceCandidate] = []
    for record in ordered:
        if all(_bbox_iou(record.bbox, old.bbox) < threshold for old in kept):
            kept.append(record)
    kept.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    return kept


class FaceEngine:
    def __init__(self, settings: Settings, log: Callable[[str], None] | None = None):
        self.settings = settings
        self.log = log or (lambda _: None)
        self.analyzer = None
        self.detector = None
        self.recognition_model = None
        self.landmark_models: list[object] = []
        self.provider = ""
        self._thread_state = threading.local()

    def initialize(self) -> None:
        self.log("Проверка модели antelopev2...")
        ensure_antelopev2()

        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
            except Exception as exc:
                self.log(f"Предзагрузка CUDA DLL: {exc}")

        available = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in available
        if self.settings.require_gpu and not has_cuda:
            raise RuntimeError(
                "CUDAExecutionProvider недоступен. Обновите драйвер NVIDIA и повторно запустите install.bat. "
                f"Доступные провайдеры: {available}"
            )

        providers = ["CUDAExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]
        self.provider = providers[0]
        self.log(f"ONNX Runtime: {self.provider}")

        from insightface.app import FaceAnalysis

        allowed = ["detection", "recognition", "landmark_2d_106", "landmark_3d_68"]
        self.analyzer = FaceAnalysis(
            name=MODEL_NAME,
            root=str(INSIGHTFACE_ROOT),
            allowed_modules=allowed,
            providers=providers,
        )
        # SCRFD запускается с более низким внутренним rescue-порогом. Обычный
        # пользовательский detector_threshold применяется уже к кандидатам.
        # Это позволяет сохранить слабые маленькие лица без второго прохода GPU.
        internal_threshold = min(self.settings.detector_threshold, self.settings.rescue_detector_threshold)
        self.analyzer.prepare(
            ctx_id=0 if has_cuda else -1,
            det_thresh=internal_threshold,
            det_size=(640, 640),
        )

        self.detector = getattr(self.analyzer, "det_model", None)
        models = getattr(self.analyzer, "models", {})
        self.recognition_model = models.get("recognition")
        self.landmark_models = [
            model for name, model in models.items()
            if name in {"landmark_2d_106", "landmark_3d_68"}
        ]
        if self.detector is None or self.recognition_model is None:
            raise RuntimeError("В antelopev2 не найдены обязательные detection/recognition модели")

        if has_cuda:
            bad_sessions: list[str] = []
            for name, model in models.items():
                session = getattr(model, "session", None)
                if session is not None and "CUDAExecutionProvider" not in session.get_providers():
                    bad_sessions.append(name)
            if bad_sessions:
                raise RuntimeError("Часть моделей InsightFace не запустилась на CUDA: " + ", ".join(bad_sessions))

        self.log("Прогрев моделей...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        # Один полный вызов при инициализации допустим: он прогревает все сессии.
        # В рабочем цикле FaceAnalysis.get() больше не используется на тайлах.
        self.analyzer.get(dummy)
        self.log("Модели готовы: detection -> NMS -> batch recognition -> landmarks.")

    def _get_thread_detector(self):
        if self.detector is None:
            raise RuntimeError("FaceEngine не инициализирован")
        detector = getattr(self._thread_state, "detector", None)
        if detector is None:
            # SCRFD хранит изменяемый center_cache в Python-wrapper. Делаем
            # отдельный wrapper на inference-поток, но оставляем общую ORT Session.
            detector = copy.copy(self.detector)
            detector.center_cache = dict(getattr(self.detector, "center_cache", {}))
            detector.det_thresh = min(self.settings.detector_threshold, self.settings.rescue_detector_threshold)
            self._thread_state.detector = detector
        return detector

    @staticmethod
    def _clip_bbox(
        raw_bbox: tuple[float, float, float, float], image_w: int, image_h: int
    ) -> tuple[int, int, int, int] | None:
        rx1, ry1, rx2, ry2 = raw_bbox
        if not np.isfinite([rx1, ry1, rx2, ry2]).all():
            return None
        x1 = max(0, min(image_w - 1, int(round(rx1))))
        y1 = max(0, min(image_h - 1, int(round(ry1))))
        x2 = max(x1 + 1, min(image_w, int(round(rx2))))
        y2 = max(y1 + 1, min(image_h, int(round(ry2))))
        return x1, y1, x2, y2

    def _detect_candidates(
        self,
        image: np.ndarray,
        source: str,
        image_w: int,
        image_h: int,
        xoff: int = 0,
        yoff: int = 0,
        reject_internal_tile_edges: bool = False,
    ) -> list[FaceCandidate]:
        detector = self._get_thread_detector()
        bboxes, kpss = detector.detect(image, input_size=(640, 640), max_num=0)
        if bboxes is None or len(bboxes) == 0 or kpss is None:
            return []

        local_h, local_w = image.shape[:2]
        results: list[FaceCandidate] = []
        rescue_threshold = min(self.settings.detector_threshold, self.settings.rescue_detector_threshold)
        for index, row in enumerate(np.asarray(bboxes)):
            if row.size < 5:
                continue
            lx1, ly1, lx2, ly2, score = map(float, row[:5])
            if score < rescue_threshold:
                continue
            fw = max(1.0, lx2 - lx1)
            fh = max(1.0, ly2 - ly1)
            if reject_internal_tile_edges:
                edge_margin = max(6.0, min(28.0, min(fw, fh) * 0.10))
                if xoff > 0 and lx1 <= edge_margin:
                    continue
                if yoff > 0 and ly1 <= edge_margin:
                    continue
                if xoff + local_w < image_w and lx2 >= local_w - edge_margin:
                    continue
                if yoff + local_h < image_h and ly2 >= local_h - edge_margin:
                    continue

            raw_bbox = (lx1 + xoff, ly1 + yoff, lx2 + xoff, ly2 + yoff)
            bbox = self._clip_bbox(raw_bbox, image_w, image_h)
            if bbox is None:
                continue
            kps = np.asarray(kpss[index], dtype=np.float32).copy()
            if kps.shape != (5, 2) or not np.isfinite(kps).all():
                continue
            kps[:, 0] += float(xoff)
            kps[:, 1] += float(yoff)
            actual_source = source if score >= self.settings.detector_threshold else f"rescue-{source}"
            results.append(
                FaceCandidate(
                    bbox=bbox,
                    raw_bbox=raw_bbox,
                    kps=kps,
                    det_score=score,
                    source=actual_source,
                )
            )
        return results

    def _head_clipping_reasons(
        self,
        raw_bbox: tuple[float, float, float, float],
        image_w: int,
        image_h: int,
    ) -> list[str]:
        if not self.settings.reject_head_clipping:
            return []
        x1, y1, x2, y2 = raw_bbox
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)
        reasons: list[str] = []

        # Если сам bbox вышел за реальную границу фото, это сильный сигнал.
        hard_tol_x = max(2.0, fw * 0.015)
        hard_tol_y = max(2.0, fh * 0.015)
        if y1 < -hard_tol_y:
            reasons.append("лицо/голова фактически обрезаны верхним краем кадра")
        if x1 < -hard_tol_x or x2 > image_w + hard_tol_x:
            reasons.append("лицо/голова фактически обрезаны боковым краем кадра")
        if y2 > image_h + hard_tol_y:
            reasons.append("подбородок фактически обрезан нижним краем кадра")

        # SCRFD bbox описывает лицо, а не всю голову. Для макушки и боков нужен
        # дополнительный анатомический запас. Эта проверка ловит кадры, где bbox
        # ещё помещается, но волосы/макушка уже срезаны границей фотографии.
        top_clearance = max(0.0, y1)
        left_clearance = max(0.0, x1)
        right_clearance = max(0.0, image_w - x2)
        bottom_clearance = max(0.0, image_h - y2)
        if top_clearance < fh * self.settings.head_top_margin_ratio:
            reasons.append("слишком мало места над лицом: вероятно обрезана макушка")
        if min(left_clearance, right_clearance) < fw * self.settings.head_side_margin_ratio:
            reasons.append("слишком мало места сбоку от лица: вероятно обрезана голова")
        if bottom_clearance < fh * self.settings.chin_margin_ratio:
            reasons.append("слишком мало места под лицом: вероятно обрезан подбородок")

        # Если уже есть прямой признак физической обрезки головы, дополнительная
        # композиционная причина только засорит rejected.csv.
        if reasons:
            return reasons

        # Отдельная композиционная защита от случайно попавших в край кадра людей.
        # Фиксированный процент здесь работает плохо: для группового кадра лицо может
        # занимать 0.05% площади, а для тесного портрета — 10% и более. Поэтому
        # требуемый отступ плавно увеличивается при уменьшении лица.
        frame_area = max(1.0, float(image_w) * float(image_h))
        face_area_ratio = max(1.0e-8, min(1.0, (fw * fh) / frame_area))
        ref_area = max(1.0e-8, self.settings.edge_guard_reference_face_area)
        size_scale = (ref_area / face_area_ratio) ** self.settings.edge_guard_size_exponent
        # Сам guard ограничен min/max ratio, поэтому экстремально маленький bbox
        # не сможет потребовать половину кадра свободного пространства.
        guard_ratio = self.settings.edge_guard_base_ratio * size_scale
        guard_ratio = max(self.settings.edge_guard_min_ratio, min(self.settings.edge_guard_max_ratio, guard_ratio))

        # Сначала расширяем bbox до минимально ожидаемой области головы, затем
        # требуем ещё композиционный отступ от реальной границы фотографии.
        projected_left = x1 - fw * self.settings.head_side_margin_ratio
        projected_right = x2 + fw * self.settings.head_side_margin_ratio
        projected_top = y1 - fh * self.settings.head_top_margin_ratio
        projected_bottom = y2 + fh * self.settings.chin_margin_ratio

        required_x = float(image_w) * guard_ratio
        required_y = float(image_h) * guard_ratio
        side_gap = min(projected_left, float(image_w) - projected_right)
        top_gap = projected_top
        bottom_gap = float(image_h) - projected_bottom
        face_area_pct = face_area_ratio * 100.0

        if side_gap < required_x:
            reasons.append(
                f"голова слишком близко к боковому краю кадра "
                f"(лицо {face_area_pct:.2f}% кадра, нужен отступ ~{guard_ratio * 100:.1f}%)"
            )
        if top_gap < required_y:
            reasons.append(
                f"голова слишком близко к верхнему краю кадра "
                f"(лицо {face_area_pct:.2f}% кадра, нужен отступ ~{guard_ratio * 100:.1f}%)"
            )
        if bottom_gap < required_y:
            reasons.append(
                f"голова слишком близко к нижнему краю кадра "
                f"(лицо {face_area_pct:.2f}% кадра, нужен отступ ~{guard_ratio * 100:.1f}%)"
            )
        return reasons

    def _record_from_face(
        self,
        quality_image: np.ndarray,
        face,
        candidate: FaceCandidate,
        *,
        apply_head_clipping: bool = True,
    ) -> FaceRecord:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            raise ValueError("InsightFace не вернул embedding")
        embedding = _l2_normalize(emb)

        reasons: list[str] = []
        x1, y1, x2, y2 = candidate.bbox
        fw, fh = x2 - x1, y2 - y1
        if self.settings.reject_small_face and min(fw, fh) < self.settings.min_face_px:
            reasons.append(f"слишком маленькое лицо ({min(fw, fh)} px)")

        if apply_head_clipping:
            reasons.extend(self._head_clipping_reasons(candidate.raw_bbox, quality_image.shape[1], quality_image.shape[0]))

        crop = _safe_crop(quality_image, candidate.bbox)
        blur = _blur_score(crop)
        if self.settings.reject_blur and blur is not None and blur < self.settings.blur_threshold:
            reasons.append(f"нерезкое лицо ({blur:.1f})")

        if self.settings.reject_exposure and crop.size:
            bad_exp, clipped, mean = _exposure_bad(crop, self.settings.max_clipped_fraction)
            if bad_exp:
                reasons.append(f"плохая экспозиция (clip {clipped:.0%}, mean {mean:.0f})")

        pose_tuple: tuple[float, float, float] | None = None
        pose = getattr(face, "pose", None)
        if pose is not None and len(pose) >= 3 and np.isfinite(pose[:3]).all():
            pitch, yaw, roll = (float(pose[0]), float(pose[1]), float(pose[2]))
            pose_tuple = (pitch, yaw, roll)
            if self.settings.reject_pose:
                if abs(yaw) > self.settings.max_yaw:
                    reasons.append(f"сильный поворот головы yaw={yaw:.0f}°")
                if abs(pitch) > self.settings.max_pitch:
                    reasons.append(f"сильный наклон головы pitch={pitch:.0f}°")
                if abs(roll) > self.settings.max_roll:
                    reasons.append(f"сильный наклон кадра/головы roll={roll:.0f}°")

        left_eye = right_eye = None
        lmk = getattr(face, "landmark_2d_106", None)
        if lmk is not None:
            pts = np.asarray(lmk, dtype=np.float32)
            if pts.shape == (106, 2):
                right_eye = _eye_openness(pts[RIGHT_EYE_106])
                left_eye = _eye_openness(pts[LEFT_EYE_106])
                if self.settings.reject_closed_eyes:
                    vals = [v for v in (left_eye, right_eye) if v is not None]
                    if len(vals) == 2 and min(vals) < self.settings.eye_ratio_min:
                        reasons.append(f"закрыт/сильно прищурен глаз ({left_eye:.3f}/{right_eye:.3f})")

        return FaceRecord(
            embedding=embedding,
            bbox=candidate.bbox,
            raw_bbox=candidate.raw_bbox,
            det_score=candidate.det_score,
            quality_reasons=reasons,
            blur_score=blur,
            eye_left=left_eye,
            eye_right=right_eye,
            pose=pose_tuple,
            source=candidate.source,
        )

    def _analyze_candidates(
        self,
        image: np.ndarray,
        candidates: list[FaceCandidate],
        *,
        reference: bool = False,
    ) -> list[FaceRecord]:
        if not candidates:
            return []
        if self.recognition_model is None:
            raise RuntimeError("Recognition model не инициализирован")

        from insightface.app.common import Face
        from insightface.utils import face_align

        faces = [
            Face(
                bbox=np.asarray(candidate.bbox, dtype=np.float32),
                kps=np.asarray(candidate.kps, dtype=np.float32),
                det_score=float(candidate.det_score),
            )
            for candidate in candidates
        ]

        # Главное ускорение новой архитектуры: после общего NMS каждое лицо
        # проходит ArcFace только один раз; нормализованные crops отправляются
        # в модель батчами вместо recognition для каждого overlap-тайла.
        crops: list[np.ndarray] = []
        input_size = int(getattr(self.recognition_model, "input_size", (112, 112))[0])
        for face in faces:
            crops.append(face_align.norm_crop(image, landmark=face.kps, image_size=input_size))

        batch_size = max(1, int(self.settings.recognition_batch_size))
        try:
            embeddings: list[np.ndarray] = []
            for start in range(0, len(crops), batch_size):
                chunk = crops[start:start + batch_size]
                features = np.asarray(self.recognition_model.get_feat(chunk), dtype=np.float32)
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                embeddings.extend(features)
            if len(embeddings) != len(faces):
                raise RuntimeError("ArcFace вернул неверное число embeddings")
            for face, embedding in zip(faces, embeddings):
                face.embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except Exception as batch_error:
            self.log(f"Batch ArcFace недоступен ({batch_error}); fallback на одиночный режим")
            for face in faces:
                self.recognition_model.get(image, face)

        # Landmarks запускаются только для уже дедуплицированных лиц.
        for face in faces:
            for model in self.landmark_models:
                model.get(image, face)

        records: list[FaceRecord] = []
        for face, candidate in zip(faces, candidates):
            try:
                records.append(
                    self._record_from_face(
                        image,
                        face,
                        candidate,
                        apply_head_clipping=not reference,
                    )
                )
            except Exception as exc:
                self.log(f"Лицо {candidate.source} пропущено: {exc}")
        return records

    def analyze_reference(self, image: np.ndarray, path: Path) -> ImageAnalysis:
        h, w = image.shape[:2]
        candidates = self._detect_candidates(image, "reference", w, h)
        strong = [candidate for candidate in candidates if candidate.det_score >= self.settings.reference_min_det_score]
        strong = _nms_records(strong, self.settings.nms_iou_threshold)
        if len(strong) != 1:
            return ImageAnalysis(
                path=path,
                faces=[],
                error=f"на эталоне требуется ровно 1 уверенно найденное лицо; найдено {len(strong)}",
                full_detected=len(candidates),
            )
        records = self._analyze_candidates(image, strong, reference=True)
        if len(records) != 1:
            return ImageAnalysis(path=path, faces=[], error="не удалось проанализировать лицо эталона")
        return ImageAnalysis(path=path, faces=records, full_detected=len(candidates))

    def _tile_candidates(self, image: np.ndarray) -> tuple[list[FaceCandidate], int]:
        if not self.settings.tile_enabled:
            return [], 0
        h, w = image.shape[:2]
        if max(h, w) <= self.settings.tile_trigger_px or (h <= self.settings.tile_size and w <= self.settings.tile_size):
            return [], 0

        tile_size = max(800, int(self.settings.tile_size))
        overlap = min(0.45, max(0.0, float(self.settings.tile_overlap)))
        xs = _tile_positions(w, tile_size, overlap)
        ys = _tile_positions(h, tile_size, overlap)
        candidates: list[FaceCandidate] = []
        tiles_used = 0

        for y0 in ys:
            for x0 in xs:
                x1 = min(w, x0 + tile_size)
                y1 = min(h, y0 + tile_size)
                if x0 == 0 and y0 == 0 and x1 == w and y1 == h:
                    continue
                tile = image[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                tiles_used += 1
                candidates.extend(
                    self._detect_candidates(
                        tile,
                        "tile",
                        w,
                        h,
                        xoff=x0,
                        yoff=y0,
                        reject_internal_tile_edges=True,
                    )
                )
        return candidates, tiles_used

    def analyze(self, image: np.ndarray, path: Path) -> ImageAnalysis:
        h, w = image.shape[:2]
        full_candidates = self._detect_candidates(image, "full", w, h)
        tile_candidates, tiles_used = self._tile_candidates(image)
        all_candidates = _nms_records(full_candidates + tile_candidates, self.settings.nms_iou_threshold)

        # Слабые rescue-кандидаты полезны, но на аномальном кадре не должны
        # породить тысячи ArcFace crops. Сильные кандидаты не ограничиваются.
        strong = [c for c in all_candidates if c.det_score >= self.settings.detector_threshold]
        weak = [c for c in all_candidates if c.det_score < self.settings.detector_threshold]
        weak.sort(
            key=lambda c: (c.det_score, (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1])),
            reverse=True,
        )
        candidates = strong + weak[: max(0, int(self.settings.rescue_max_candidates))]
        candidates.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
        records = self._analyze_candidates(image, candidates)
        return ImageAnalysis(
            path=path,
            faces=records,
            full_detected=len(full_candidates),
            tile_detected=len(tile_candidates),
            tiles_used=tiles_used,
        )

class ChildFaceFinder:
    def __init__(
        self,
        settings: Settings,
        log: Callable[[str], None] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ):
        self.settings = settings
        self.log = log or (lambda _: None)
        self.progress = progress or (lambda _a, _b, _c: None)
        self.cancel_event = threading.Event()
        self.engine = FaceEngine(settings, self.log)

    def cancel(self) -> None:
        self.cancel_event.set()

    def _pipeline(self, paths: list[Path], temp_dir: Path, references: bool = False) -> Iterable[ImageAnalysis]:
        if not paths:
            return

        path_q: queue.Queue[Path | None] = queue.Queue()
        max_images_in_flight = max(2, self.settings.inference_workers * 2)
        decoded_q: queue.Queue[tuple[Path, np.ndarray | None, str | None] | None] = queue.Queue(maxsize=max_images_in_flight)
        image_slots = threading.Semaphore(max_images_in_flight)
        result_q: queue.Queue[ImageAnalysis] = queue.Queue()

        for path in paths:
            path_q.put(path)
        for _ in range(self.settings.decode_workers):
            path_q.put(None)

        decoders_left = self.settings.decode_workers
        decoder_lock = threading.Lock()

        def decoder_worker() -> None:
            nonlocal decoders_left
            try:
                while True:
                    path = path_q.get()
                    if path is None:
                        break
                    if self.cancel_event.is_set():
                        decoded_q.put((path, None, "отменено"))
                        continue
                    image_slots.acquire()
                    if self.cancel_event.is_set():
                        image_slots.release()
                        decoded_q.put((path, None, "отменено"))
                        continue
                    try:
                        img = decode_image(path, temp_dir)
                        decoded_q.put((path, img, None))
                    except Exception as exc:
                        image_slots.release()
                        decoded_q.put((path, None, str(exc)))
            finally:
                with decoder_lock:
                    decoders_left -= 1
                    if decoders_left == 0:
                        for _ in range(self.settings.inference_workers):
                            decoded_q.put(None)

        def inference_worker() -> None:
            while True:
                item = decoded_q.get()
                if item is None:
                    return
                path, image, error = item
                if error is not None or image is None:
                    result_q.put(ImageAnalysis(path=path, faces=[], error=error or "ошибка декодирования"))
                    continue
                try:
                    if self.cancel_event.is_set():
                        result_q.put(ImageAnalysis(path=path, faces=[], error="отменено"))
                        continue
                    try:
                        analysis = self.engine.analyze_reference(image, path) if references else self.engine.analyze(image, path)
                        result_q.put(analysis)
                    except Exception as exc:
                        result_q.put(ImageAnalysis(path=path, faces=[], error=str(exc)))
                finally:
                    del image
                    image_slots.release()

        total_workers = self.settings.decode_workers + self.settings.inference_workers
        with ThreadPoolExecutor(max_workers=total_workers, thread_name_prefix="cff") as executor:
            decoder_futures = [executor.submit(decoder_worker) for _ in range(self.settings.decode_workers)]
            inference_futures = [executor.submit(inference_worker) for _ in range(self.settings.inference_workers)]
            for _ in range(len(paths)):
                yield result_q.get()
            for future in decoder_futures + inference_futures:
                future.result()

    def _build_references(self, folder: Path, temp_dir: Path) -> ReferenceSet:
        # Эталонная папка всегда сканируется рекурсивно: поза/тип кадра может
        # задаваться подкаталогом (например, portrait/ и standing/), но каталог
        # никогда не является частью ID цели.
        paths = iter_candidate_files(folder, recursive=True)
        if not paths:
            raise RuntimeError("В папке эталонов нет файлов")

        reference_groups = group_reference_paths(paths)
        path_meta: dict[Path, tuple[str, str]] = {}
        display_by_key: dict[str, str] = {}
        grouped_paths: dict[str, list[Path]] = {}
        for group_index, (display, group_paths) in enumerate(reference_groups):
            key = f"reference-group-{group_index:06d}"
            display_by_key[key] = display
            grouped_paths[key] = group_paths
            for path in group_paths:
                path_meta[path] = (key, display)
            if len(group_paths) > 1 and self.settings.verbose_diagnostics:
                rels = [str(p.relative_to(folder)) for p in group_paths]
                self.log(f"Эталоны «{display}»: объединено файлов {len(group_paths)}: " + " | ".join(rels))

        embeddings: dict[str, list[tuple[np.ndarray, Path]]] = defaultdict(list)
        done = 0
        for analysis in self._pipeline(paths, temp_dir, references=True):
            done += 1
            self.progress(done, len(paths), f"Эталоны: {analysis.path.name}")
            key, display = path_meta[analysis.path]
            if analysis.error:
                if analysis.error != "отменено":
                    self.log(f"Эталон {analysis.path.name}: {analysis.error} — пропущен")
                continue
            if len(analysis.faces) != 1:
                self.log(f"Эталон {analysis.path.name}: не удалось получить ровно одно лицо — пропущен")
                continue
            face = analysis.faces[0]
            if face.quality_reasons:
                self.log(f"Эталон {analysis.path.name}: низкое качество ({'; '.join(face.quality_reasons)}) — пропущен")
                continue
            embeddings[key].append((face.embedding, analysis.path))

        if self.cancel_event.is_set():
            raise CancelledError("Обработка отменена")
        if not embeddings:
            raise RuntimeError("Не удалось получить ни одного качественного эталонного лица")

        valid_keys = sorted(embeddings.keys(), key=lambda k: display_by_key[k].casefold())
        person_ids = [display_by_key[k] for k in valid_keys]
        person_index_by_key = {key: idx for idx, key in enumerate(valid_keys)}
        vectors: list[np.ndarray] = []
        ref_person_indices: list[int] = []

        for key in valid_keys:
            refs = embeddings[key]
            if len(refs) > 1:
                ref_vectors = np.stack([vector for vector, _ in refs], axis=0).astype(np.float32, copy=False)
                distances = 1.0 - np.matmul(ref_vectors, ref_vectors.T)
                upper = np.triu_indices(len(refs), k=1)
                pair_distances = distances[upper]
                worst_pos = int(np.argmax(pair_distances))
                worst_distance = float(pair_distances[worst_pos])
                i = int(upper[0][worst_pos])
                j = int(upper[1][worst_pos])
                if worst_distance > self.settings.reference_consistency_warn:
                    self.log(
                        f"ВНИМАНИЕ: эталоны «{display_by_key[key]}» содержат слабо совпадающую пару "
                        f"{refs[i][1].name} ↔ {refs[j][1].name} (distance={worst_distance:.3f}). "
                        "Проверьте имена файлов. Все эталоны оставлены."
                    )
                elif self.settings.verbose_diagnostics:
                    self.log(
                        f"Эталоны «{display_by_key[key]}»: файлов {len(refs)}, "
                        f"максимальная взаимная distance={worst_distance:.3f}"
                    )
            for vector, _path in refs:
                vectors.append(vector)
                ref_person_indices.append(person_index_by_key[key])

        matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
        index_array = np.asarray(ref_person_indices, dtype=np.int32)
        self.log(f"Готово эталонов: {len(person_ids)} целей из {len(vectors)} файлов")
        return ReferenceSet(person_ids=person_ids, matrix=matrix, ref_person_indices=index_array, file_count=len(vectors))

    def _match_face(
        self,
        face: FaceRecord,
        references: ReferenceSet,
    ) -> tuple[str | None, str, float, str | None, float | None, str | None]:
        ref_distances = 1.0 - np.matmul(references.matrix, face.embedding)
        person_distances = np.full(len(references.person_ids), np.inf, dtype=np.float32)
        np.minimum.at(person_distances, references.ref_person_indices, ref_distances)
        order = np.argsort(person_distances)
        best_idx = int(order[0])
        best_id = references.person_ids[best_idx]
        best_dist = float(person_distances[best_idx])
        second_id: str | None = None
        second_dist: float | None = None
        if len(order) > 1:
            second_idx = int(order[1])
            second_id = references.person_ids[second_idx]
            second_dist = float(person_distances[second_idx])

        if best_dist > self.settings.match_threshold:
            return None, best_id, best_dist, second_id, second_dist, "дистанция выше порога"
        if second_dist is not None and (second_dist - best_dist) < self.settings.ambiguity_margin:
            return None, best_id, best_dist, second_id, second_dist, "слишком близкая вторая цель"
        return best_id, best_id, best_dist, second_id, second_dist, None

    @staticmethod
    def _write_csv_atomic(path: Path, header: list[str], rows: list[list[object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        try:
            with temp.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f, delimiter=";", lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def run(self, references_dir: Path, photos_dir: Path, output_csv: Path) -> RunSummary:
        self.cancel_event.clear()

        removed_temp, failed_temp = cleanup_stale_temp_dirs()
        if removed_temp:
            self.log(f"TEMP: удалено оставшихся каталогов предыдущих запусков: {len(removed_temp)}")
        for path, error in failed_temp:
            self.log(f"TEMP: не удалось удалить {path}: {error}")

        self.engine.initialize()

        with make_system_temp_dir() as temp_name:
            temp_dir = Path(temp_name)
            self.log(f"Временный каталог ОС: {temp_dir}")
            references = self._build_references(references_dir, temp_dir)

            photo_paths = iter_candidate_files(photos_dir, self.settings.recursive)
            if not photo_paths:
                raise RuntimeError("В папке фотографий нет файлов")

            analyses: list[ImageAnalysis] = []
            done = 0
            for analysis in self._pipeline(photo_paths, temp_dir, references=False):
                done += 1
                self.progress(done, len(photo_paths), analysis.path.name)
                if analysis.error:
                    if analysis.error != "отменено":
                        self.log(f"{analysis.path.name}: {analysis.error}")
                    continue
                analyses.append(analysis)
                if self.settings.verbose_diagnostics:
                    rescue_count = sum(face.source.startswith("rescue-") for face in analysis.faces)
                    self.log(
                        f"{analysis.path.name}: full={analysis.full_detected}; "
                        f"tiles=+{analysis.tile_detected} ({analysis.tiles_used} проходов SCRFD); "
                        f"после NMS/ArcFace={len(analysis.faces)}; rescue={rescue_count}"
                    )

            if self.cancel_event.is_set():
                raise CancelledError("Обработка отменена")

            # Первый проход matching: только исходные пользовательские эталоны.
            base_matches: dict[int, list[tuple[FaceRecord, tuple[str | None, str, float, str | None, float | None, str | None]]]] = {}
            anchor_candidates: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
            anchor_limit_distance = min(0.34, max(0.18, self.settings.match_threshold - 0.06))

            for analysis_index, analysis in enumerate(analyses):
                rows = []
                for face in analysis.faces:
                    info = self._match_face(face, references)
                    rows.append((face, info))
                    person_id, _best_id, dist, _second_id, _second_dist, _problem = info
                    if (
                        person_id is not None
                        and not face.quality_reasons
                        and not face.source.startswith("rescue-")
                        and face.det_score >= max(0.40, self.settings.detector_threshold)
                        and dist <= anchor_limit_distance
                    ):
                        anchor_candidates[person_id].append((dist, face.embedding))
                base_matches[analysis_index] = rows

            # Session anchors: до 8 самых уверенных лиц на цель. Они создаются
            # только из первого прохода и никогда из восстановленных совпадений,
            # поэтому второй проход не может сам себя "раскачать".
            extra_vectors: list[np.ndarray] = []
            extra_indices: list[int] = []
            person_index = {person: index for index, person in enumerate(references.person_ids)}
            for person, values in anchor_candidates.items():
                values.sort(key=lambda item: item[0])
                for _dist, vector in values[:8]:
                    extra_vectors.append(vector)
                    extra_indices.append(person_index[person])

            expanded_references = references
            if extra_vectors:
                expanded_references = ReferenceSet(
                    person_ids=references.person_ids,
                    matrix=np.concatenate(
                        [references.matrix, np.stack(extra_vectors).astype(np.float32, copy=False)], axis=0
                    ),
                    ref_person_indices=np.concatenate(
                        [references.ref_person_indices, np.asarray(extra_indices, dtype=np.int32)], axis=0
                    ),
                    file_count=references.file_count,
                )
                self.log(
                    f"Session anchors: добавлено {len(extra_vectors)} уверенных временных эталонов "
                    f"для {len(anchor_candidates)} целей."
                )

            matches: dict[tuple[str, str], tuple[float, str]] = {}
            match_details: dict[tuple[str, str], tuple[float, str, FaceRecord, Path]] = {}
            rejects: dict[tuple[str, str], tuple[float, str, str]] = {}
            reviews: list[list[object]] = []
            recovered = 0

            for analysis_index, analysis in enumerate(analyses):
                try:
                    relative_path = analysis.path.relative_to(photos_dir).as_posix()
                except ValueError:
                    relative_path = analysis.path.name
                photo_id = analysis.path.stem
                per_photo_accept: dict[str, tuple[float, str, FaceRecord]] = {}
                per_photo_reject: dict[str, tuple[float, str, str]] = {}
                review_for_photo: list[list[object]] = []

                for face, base_info in base_matches[analysis_index]:
                    person_id, best_id, dist, second_id, second_dist, match_problem = base_info
                    used_info = base_info
                    recovered_by_anchor = False

                    # Только пограничные лица рассматриваются повторно. Требуем,
                    # чтобы исходно лучшей целью оставалась та же цель, а расширенная
                    # галерея дала реальное улучшение и прошла обычный margin.
                    if (
                        person_id is None
                        and extra_vectors
                        and dist <= self.settings.match_threshold + 0.10
                    ):
                        expanded_info = self._match_face(face, expanded_references)
                        expanded_person, expanded_best, expanded_dist, expanded_second, expanded_second_dist, expanded_problem = expanded_info
                        if (
                            expanded_person is not None
                            and expanded_best == best_id
                            and expanded_dist <= dist - 0.015
                        ):
                            person_id = expanded_person
                            used_info = expanded_info
                            best_id = expanded_best
                            dist = expanded_dist
                            second_id = expanded_second
                            second_dist = expanded_second_dist
                            match_problem = expanded_problem
                            recovered_by_anchor = True
                            recovered += 1

                    if person_id is None:
                        if dist <= self.settings.match_threshold + 0.08:
                            review_for_photo.append([
                                photo_id,
                                relative_path,
                                best_id,
                                f"{dist:.4f}",
                                second_id or "",
                                "" if second_dist is None else f"{second_dist:.4f}",
                                match_problem or "неуверенное совпадение",
                                face.source,
                            ])
                        continue

                    match_source = "session-anchor" if recovered_by_anchor else face.source
                    if face.quality_reasons:
                        reason = "; ".join(face.quality_reasons)
                        old = per_photo_reject.get(person_id)
                        if old is None or dist < old[0]:
                            per_photo_reject[person_id] = (dist, reason, match_source)
                    else:
                        old = per_photo_accept.get(person_id)
                        if old is None or dist < old[0]:
                            per_photo_accept[person_id] = (dist, match_source, face)

                for person_id, (dist, source, face) in per_photo_accept.items():
                    key = (person_id, relative_path)
                    old = matches.get(key)
                    if old is None or dist < old[0]:
                        matches[key] = (dist, source)
                        match_details[key] = (dist, source, face, analysis.path)
                    rejects.pop(key, None)

                for person_id, (dist, reason, source) in per_photo_reject.items():
                    key = (person_id, relative_path)
                    if key not in matches:
                        old = rejects.get(key)
                        if old is None or dist < old[0]:
                            rejects[key] = (dist, reason, source)

                reviews.extend(review_for_photo)
                if self.settings.verbose_diagnostics:
                    self.log(
                        f"{analysis.path.name}: результат={len(per_photo_accept)}; "
                        f"отбраковано={len(per_photo_reject)}; review={len(review_for_photo)}"
                    )

            if recovered:
                self.log(f"Session anchors: восстановлено пограничных совпадений: {recovered}")

            match_rows = [
                [person_id, Path(relative_path).stem, relative_path]
                for (person_id, relative_path), (_dist, _source) in sorted(
                    matches.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
                )
            ]
            rejected_rows = [
                [person_id, Path(relative_path).stem, relative_path, reason, f"{dist:.4f}", source]
                for (person_id, relative_path), (dist, reason, source) in sorted(
                    rejects.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
                )
            ]

            rejected_csv = output_csv.with_name(output_csv.stem + "_rejected.csv")
            review_csv = output_csv.with_name(output_csv.stem + "_review.csv")
            self._write_csv_atomic(
                output_csv,
                ["Идентификатор цели", "номер фото с этой целью", "Исходный файл"],
                match_rows,
            )
            self._write_csv_atomic(
                rejected_csv,
                ["Идентификатор цели", "номер фото", "Исходный файл", "Причина", "Cosine distance", "Источник"],
                rejected_rows,
            )
            self._write_csv_atomic(
                review_csv,
                ["номер фото", "Исходный файл", "Лучший кандидат", "Distance", "Второй кандидат", "Distance 2", "Причина", "Источник"],
                reviews,
            )

            best_csv: Path | None = None
            best_series_count = 0
            if self.settings.select_best_series:
                self.log("Серии: определение последовательных серий отдельно для каждой цели…")
                best_candidates = [
                    BestFrameCandidate(
                        person_id=person_id,
                        path=path,
                        relative_path=relative_path,
                        photo_id=path.stem,
                        distance=dist,
                        source=source,
                        det_score=face.det_score,
                        blur_score=face.blur_score,
                        eye_left=face.eye_left,
                        eye_right=face.eye_right,
                        pose=face.pose,
                    )
                    for (person_id, relative_path), (dist, source, face, path) in match_details.items()
                ]
                selections = select_best_series_frames(
                    best_candidates,
                    match_threshold=self.settings.match_threshold,
                    eye_open_threshold=self.settings.eye_ratio_min,
                    max_gap_seconds=self.settings.series_max_gap_seconds,
                    max_filename_gap=self.settings.series_max_filename_gap,
                )
                best_rows = []
                for selection in selections:
                    candidate = selection.candidate
                    eye_text = "" if selection.eye_score is None else f"{selection.eye_score:.4f}"
                    blur_text = "" if candidate.blur_score is None else f"{candidate.blur_score:.2f}"
                    best_rows.append([
                        selection.person_id,
                        candidate.photo_id,
                        candidate.relative_path,
                        selection.series_index,
                        selection.series_size,
                        f"{selection.score:.4f}",
                        f"{candidate.distance:.4f}",
                        blur_text,
                        eye_text,
                        f"{selection.pose_score:.4f}",
                        candidate.source,
                    ])
                best_csv = output_csv.with_name(output_csv.stem + "_best.csv")
                self._write_csv_atomic(
                    best_csv,
                    [
                        "Идентификатор цели", "номер фото с этой целью", "Исходный файл",
                        "Серия", "Кадров в серии", "Best score", "Cosine distance",
                        "Резкость лица", "Открытость глаз", "Pose score", "Источник",
                    ],
                    best_rows,
                )
                best_series_count = len(best_rows)
                self.log(
                    f"Серии: выбрано лучших кадров: {best_series_count}; CSV: {best_csv}"
                )

        return RunSummary(
            reference_ids=len(references.person_ids),
            reference_files=references.file_count,
            photo_count=len(photo_paths),
            matched_pairs=len(match_rows),
            rejected_pairs=len(rejected_rows),
            review_rows=len(reviews),
            output_csv=output_csv,
            rejected_csv=rejected_csv,
            review_csv=review_csv,
            best_csv=best_csv,
            best_series_count=best_series_count,
        )
