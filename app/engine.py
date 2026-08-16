from __future__ import annotations

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

    reference_min_det_score: float = 0.45
    reference_consistency_warn: float = 0.55


@dataclass(slots=True)
class FaceRecord:
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]
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
    child_ids: list[str]
    matrix: np.ndarray
    ref_child_indices: np.ndarray
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
    Подкаталоги никогда не входят в ID ребёнка.
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
    Группирует любое количество эталонов одного ребёнка.

    1. Все точные нормализованные basename объединяются без ограничения количества.
    2. Различающиеся имена могут объединяться по однозначной общей части из двух
       и более слов (например, «Котов Никита портрет/стоя/улица»).
    3. Однословная общая часть разрешается только для однозначной пары, например
       «Котов» + «Котов Никита». Если «Котов» подходит сразу к нескольким детям,
       программа не угадывает.
    """
    exact: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path in paths:
        display = parse_reference_identifier(path)
        if not display:
            raise RuntimeError(f"Не удалось получить ID ребёнка из имени: {path.name}")
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
            "Неоднозначные имена эталонов: короткая общая часть подходит сразу к нескольким детям. "
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


def _nms_records(records: list[FaceRecord], threshold: float) -> list[FaceRecord]:
    if len(records) <= 1:
        return records
    # Сначала оставляем наиболее уверенный bbox. При равном score отдаём
    # предпочтение более крупному bbox: у него обычно меньше обрезаны края лица.
    ordered = sorted(
        records,
        key=lambda r: (r.det_score, (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])),
        reverse=True,
    )
    kept: list[FaceRecord] = []
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
        self.provider = ""

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
        self.analyzer.prepare(
            ctx_id=0 if has_cuda else -1,
            det_thresh=self.settings.detector_threshold,
            det_size=(640, 640),
        )

        if has_cuda:
            bad_sessions: list[str] = []
            for name, model in getattr(self.analyzer, "models", {}).items():
                session = getattr(model, "session", None)
                if session is not None and "CUDAExecutionProvider" not in session.get_providers():
                    bad_sessions.append(name)
            if bad_sessions:
                raise RuntimeError("Часть моделей InsightFace не запустилась на CUDA: " + ", ".join(bad_sessions))

        self.log("Прогрев моделей...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.analyzer.get(dummy)
        for name, model in getattr(self.analyzer, "models", {}).items():
            if name == "detection":
                continue
            session = getattr(model, "session", None)
            if session is None:
                continue
            feed: dict[str, np.ndarray] = {}
            for inp in session.get_inputs():
                safe_shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
                type_name = str(getattr(inp, "type", "tensor(float)"))
                if "float16" in type_name:
                    dtype = np.float16
                elif "double" in type_name:
                    dtype = np.float64
                elif "int64" in type_name:
                    dtype = np.int64
                elif "int32" in type_name:
                    dtype = np.int32
                elif "uint8" in type_name:
                    dtype = np.uint8
                else:
                    dtype = np.float32
                feed[inp.name] = np.zeros(safe_shape, dtype=dtype)
            try:
                session.run(None, feed)
            except Exception as exc:
                raise RuntimeError(f"Не удалось прогреть ONNX-модель {name}: {exc}") from exc
        self.log("Модели готовы.")

    def _record_from_face(
        self,
        quality_image: np.ndarray,
        face,
        bbox: tuple[int, int, int, int],
        source: str,
    ) -> FaceRecord:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            raise ValueError("InsightFace не вернул embedding")
        embedding = _l2_normalize(emb)

        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        reasons: list[str] = []
        x1, y1, x2, y2 = bbox
        fw, fh = x2 - x1, y2 - y1
        if self.settings.reject_small_face and min(fw, fh) < self.settings.min_face_px:
            reasons.append(f"слишком маленькое лицо ({min(fw, fh)} px)")

        crop = _safe_crop(quality_image, bbox)
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
            bbox=bbox,
            det_score=det_score,
            quality_reasons=reasons,
            blur_score=blur,
            eye_left=left_eye,
            eye_right=right_eye,
            pose=pose_tuple,
            source=source,
        )

    @staticmethod
    def _mapped_bbox(face, image_w: int, image_h: int, xoff: int = 0, yoff: int = 0) -> tuple[int, int, int, int] | None:
        bbox_arr = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
        if bbox_arr.size < 4:
            return None
        x1 = max(0, min(image_w - 1, int(round(float(bbox_arr[0]))) + xoff))
        y1 = max(0, min(image_h - 1, int(round(float(bbox_arr[1]))) + yoff))
        x2 = max(x1 + 1, min(image_w, int(round(float(bbox_arr[2]))) + xoff))
        y2 = max(y1 + 1, min(image_h, int(round(float(bbox_arr[3]))) + yoff))
        return x1, y1, x2, y2

    def analyze_reference(self, image: np.ndarray, path: Path) -> ImageAnalysis:
        if self.analyzer is None:
            raise RuntimeError("FaceEngine не инициализирован")
        faces = self.analyzer.get(image)
        strong = [
            f for f in faces
            if float(getattr(f, "det_score", 0.0) or 0.0) >= self.settings.reference_min_det_score
        ]
        if len(strong) != 1:
            return ImageAnalysis(
                path=path,
                faces=[],
                error=f"на эталоне требуется ровно 1 уверенно найденное лицо; найдено {len(strong)}",
                full_detected=len(faces),
            )
        h, w = image.shape[:2]
        bbox = self._mapped_bbox(strong[0], w, h)
        if bbox is None:
            return ImageAnalysis(path=path, faces=[], error="InsightFace не вернул bbox эталона")
        try:
            record = self._record_from_face(image, strong[0], bbox, "reference")
        except Exception as exc:
            return ImageAnalysis(path=path, faces=[], error=str(exc), full_detected=len(faces))
        return ImageAnalysis(path=path, faces=[record], full_detected=len(faces))

    def _tile_records(self, image: np.ndarray) -> tuple[list[FaceRecord], int]:
        if not self.settings.tile_enabled:
            return [], 0
        h, w = image.shape[:2]
        if max(h, w) <= self.settings.tile_trigger_px or (h <= self.settings.tile_size and w <= self.settings.tile_size):
            return [], 0

        tile_size = max(640, int(self.settings.tile_size))
        overlap = min(0.45, max(0.0, float(self.settings.tile_overlap)))
        xs = _tile_positions(w, tile_size, overlap)
        ys = _tile_positions(h, tile_size, overlap)
        records: list[FaceRecord] = []
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
                faces = self.analyzer.get(tile)
                th, tw = tile.shape[:2]
                for face in faces:
                    local = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
                    if local.size < 4:
                        continue
                    lx1, ly1, lx2, ly2 = map(float, local[:4])
                    fw = max(1.0, lx2 - lx1)
                    fh = max(1.0, ly2 - ly1)
                    edge_margin = max(6.0, min(28.0, min(fw, fh) * 0.10))
                    # Не используем лицо, обрезанное внутренней границей тайла.
                    # Благодаря перекрытию оно должно целиком попасть в соседний тайл.
                    if x0 > 0 and lx1 <= edge_margin:
                        continue
                    if y0 > 0 and ly1 <= edge_margin:
                        continue
                    if x0 + tw < w and lx2 >= tw - edge_margin:
                        continue
                    if y0 + th < h and ly2 >= th - edge_margin:
                        continue

                    bbox = self._mapped_bbox(face, w, h, x0, y0)
                    if bbox is None:
                        continue
                    try:
                        records.append(self._record_from_face(image, face, bbox, "tile"))
                    except Exception:
                        continue
        return records, tiles_used

    def analyze(self, image: np.ndarray, path: Path) -> ImageAnalysis:
        if self.analyzer is None:
            raise RuntimeError("FaceEngine не инициализирован")
        h, w = image.shape[:2]
        full_faces = self.analyzer.get(image)
        records: list[FaceRecord] = []
        for face in full_faces:
            bbox = self._mapped_bbox(face, w, h)
            if bbox is None:
                continue
            try:
                records.append(self._record_from_face(image, face, bbox, "full"))
            except Exception as exc:
                self.log(f"{path.name}: лицо полного кадра пропущено: {exc}")

        tile_records, tiles_used = self._tile_records(image)
        all_records = records + tile_records
        deduped = _nms_records(all_records, self.settings.nms_iou_threshold)
        return ImageAnalysis(
            path=path,
            faces=deduped,
            full_detected=len(records),
            tile_detected=len(tile_records),
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
        # никогда не является частью ID ребёнка.
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
        child_ids = [display_by_key[k] for k in valid_keys]
        child_index_by_key = {key: idx for idx, key in enumerate(valid_keys)}
        vectors: list[np.ndarray] = []
        ref_child_indices: list[int] = []

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
                ref_child_indices.append(child_index_by_key[key])

        matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
        index_array = np.asarray(ref_child_indices, dtype=np.int32)
        self.log(f"Готово эталонов: {len(child_ids)} детей из {len(vectors)} файлов")
        return ReferenceSet(child_ids=child_ids, matrix=matrix, ref_child_indices=index_array, file_count=len(vectors))

    def _match_face(
        self,
        face: FaceRecord,
        references: ReferenceSet,
    ) -> tuple[str | None, str, float, str | None, float | None, str | None]:
        ref_distances = 1.0 - np.matmul(references.matrix, face.embedding)
        child_distances = np.full(len(references.child_ids), np.inf, dtype=np.float32)
        np.minimum.at(child_distances, references.ref_child_indices, ref_distances)
        order = np.argsort(child_distances)
        best_idx = int(order[0])
        best_id = references.child_ids[best_idx]
        best_dist = float(child_distances[best_idx])
        second_id: str | None = None
        second_dist: float | None = None
        if len(order) > 1:
            second_idx = int(order[1])
            second_id = references.child_ids[second_idx]
            second_dist = float(child_distances[second_idx])

        if best_dist > self.settings.match_threshold:
            return None, best_id, best_dist, second_id, second_dist, "дистанция выше порога"
        if second_dist is not None and (second_dist - best_dist) < self.settings.ambiguity_margin:
            return None, best_id, best_dist, second_id, second_dist, "слишком близкий второй ребёнок"
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

            matches: dict[tuple[str, str], float] = {}
            rejects: dict[tuple[str, str], tuple[float, str]] = {}
            reviews: list[list[object]] = []

            done = 0
            for analysis in self._pipeline(photo_paths, temp_dir, references=False):
                done += 1
                self.progress(done, len(photo_paths), analysis.path.name)
                if analysis.error:
                    if analysis.error != "отменено":
                        self.log(f"{analysis.path.name}: {analysis.error}")
                    continue

                photo_id = analysis.path.stem
                per_photo_accept: dict[str, float] = {}
                per_photo_reject: dict[str, tuple[float, str]] = {}
                review_count_before = len(reviews)

                for face in analysis.faces:
                    child_id, best_id, dist, second_id, second_dist, match_problem = self._match_face(face, references)
                    if child_id is None:
                        if dist <= self.settings.match_threshold + 0.08:
                            reviews.append([
                                photo_id,
                                best_id,
                                f"{dist:.4f}",
                                second_id or "",
                                "" if second_dist is None else f"{second_dist:.4f}",
                                match_problem or "неуверенное совпадение",
                            ])
                        continue

                    if face.quality_reasons:
                        reason = "; ".join(face.quality_reasons)
                        old = per_photo_reject.get(child_id)
                        if old is None or dist < old[0]:
                            per_photo_reject[child_id] = (dist, reason)
                    else:
                        old = per_photo_accept.get(child_id)
                        if old is None or dist < old:
                            per_photo_accept[child_id] = dist

                for child_id, dist in per_photo_accept.items():
                    key = (child_id, photo_id)
                    old = matches.get(key)
                    if old is None or dist < old:
                        matches[key] = dist
                    rejects.pop(key, None)

                for child_id, (dist, reason) in per_photo_reject.items():
                    key = (child_id, photo_id)
                    if key not in matches:
                        old = rejects.get(key)
                        if old is None or dist < old[0]:
                            rejects[key] = (dist, reason)

                if self.settings.verbose_diagnostics:
                    self.log(
                        f"{analysis.path.name}: full={analysis.full_detected}; "
                        f"tiles=+{analysis.tile_detected} ({analysis.tiles_used} проходов); "
                        f"после NMS={len(analysis.faces)}; результат={len(per_photo_accept)}; "
                        f"отбраковано={len(per_photo_reject)}; review={len(reviews) - review_count_before}"
                    )

            if self.cancel_event.is_set():
                raise CancelledError("Обработка отменена")

            match_rows = [
                [child_id, photo_id]
                for (child_id, photo_id), _ in sorted(
                    matches.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
                )
            ]
            rejected_rows = [
                [child_id, photo_id, reason, f"{dist:.4f}"]
                for (child_id, photo_id), (dist, reason) in sorted(
                    rejects.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
                )
            ]

            rejected_csv = output_csv.with_name(output_csv.stem + "_rejected.csv")
            review_csv = output_csv.with_name(output_csv.stem + "_review.csv")
            self._write_csv_atomic(
                output_csv,
                ["Идентификатор ребенка", "номер фото с этим ребенком"],
                match_rows,
            )
            self._write_csv_atomic(
                rejected_csv,
                ["Идентификатор ребенка", "номер фото", "Причина", "Cosine distance"],
                rejected_rows,
            )
            self._write_csv_atomic(
                review_csv,
                ["номер фото", "Лучший кандидат", "Distance", "Второй кандидат", "Distance 2", "Причина"],
                reviews,
            )

        return RunSummary(
            reference_ids=len(references.child_ids),
            reference_files=references.file_count,
            photo_count=len(photo_paths),
            matched_pairs=len(match_rows),
            rejected_pairs=len(rejected_rows),
            review_rows=len(reviews),
            output_csv=output_csv,
            rejected_csv=rejected_csv,
            review_csv=review_csv,
        )
