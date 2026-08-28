from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image

_SEQUENCE_RE = re.compile(r"(\d+)(?!.*\d)")
_DATE_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")

# Formats for which LibRaw/rawpy is a useful metadata fallback when Pillow
# cannot expose DateTimeOriginal. This list is deliberately conservative:
# uncommon/unknown extensions still use the existing filename/mtime fallback.
_RAW_METADATA_EXTENSIONS = {
    ".3fr", ".arw", ".cr2", ".cr3", ".dng", ".erf", ".fff", ".iiq",
    ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
}


@dataclass(slots=True)
class BestFrameCandidate:
    person_id: str
    path: Path
    relative_path: str
    photo_id: str
    distance: float
    source: str
    det_score: float
    blur_score: float | None
    eye_left: float | None
    eye_right: float | None
    pose: tuple[float, float, float] | None


@dataclass(slots=True)
class PhotoOrderInfo:
    path: Path
    capture_time: datetime
    time_is_metadata: bool
    sequence_number: int | None
    sequence_source: tuple[str, str]


@dataclass(slots=True)
class BestSeriesSelection:
    person_id: str
    series_index: int
    series_size: int
    candidate: BestFrameCandidate
    score: float
    eye_score: float | None
    sharpness_rank: float
    pose_score: float


def _sequence_number(path: Path) -> int | None:
    match = _SEQUENCE_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _sequence_source(path: Path) -> tuple[str, str]:
    """Directory + filename prefix, matching photo_select_ai's conservative rule."""
    stem = path.stem
    cut = len(stem)
    while cut > 0 and stem[cut - 1].isdigit():
        cut -= 1
    return (str(path.parent).casefold(), stem[:cut].casefold())


def _parse_date(value: object) -> datetime | None:
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            pass
    return None


def _read_raw_capture_time(path: Path) -> datetime | None:
    if path.suffix.casefold() not in _RAW_METADATA_EXTENSIONS:
        return None
    try:
        # Lazy import keeps this module lightweight in tests/tools that only use
        # raster files. rawpy is already a required runtime dependency.
        import rawpy

        with rawpy.imread(str(path)) as raw:
            timestamp = raw.other.timestamp
        if isinstance(timestamp, datetime) and timestamp.year > 1971:
            return timestamp
    except Exception:
        pass
    return None


def read_photo_order_info(path: Path) -> PhotoOrderInfo:
    capture_time: datetime | None = None
    metadata_time = False
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag in (36867, 36868, 306):  # DateTimeOriginal, Digitized, Image DateTime
                if tag in exif:
                    capture_time = _parse_date(exif.get(tag))
                    if capture_time is not None:
                        metadata_time = True
                        break
    except Exception:
        pass

    if capture_time is None:
        capture_time = _read_raw_capture_time(path)
        metadata_time = capture_time is not None

    if capture_time is None:
        try:
            capture_time = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            capture_time = datetime.min

    return PhotoOrderInfo(
        path=path,
        capture_time=capture_time,
        time_is_metadata=metadata_time,
        sequence_number=_sequence_number(path),
        sequence_source=_sequence_source(path),
    )


def _ordered_candidates(
    candidates: list[BestFrameCandidate],
    info_cache: dict[Path, PhotoOrderInfo],
) -> list[BestFrameCandidate]:
    if len(candidates) <= 1:
        return list(candidates)

    infos = [info_cache[c.path] for c in candidates]
    with_sequence = [info for info in infos if info.sequence_number is not None]
    sources = {info.sequence_source for info in with_sequence}
    sequence_span = (
        max(info.sequence_number for info in with_sequence if info.sequence_number is not None)
        - min(info.sequence_number for info in with_sequence if info.sequence_number is not None)
        if with_sequence else 0
    )

    # Same rule used by the other script for repairing unreliable EXIF order:
    # when >=80% of frames share one camera filename sequence, that sequence is
    # more trustworthy than copied/coarse timestamps.
    if (
        len(with_sequence) / len(infos) >= 0.80
        and len(sources) == 1
        and sequence_span <= 5000
    ):
        return sorted(
            candidates,
            key=lambda c: (
                info_cache[c.path].sequence_number
                if info_cache[c.path].sequence_number is not None
                else 10**15,
                info_cache[c.path].capture_time,
                c.path.name.casefold(),
            ),
        )

    return sorted(
        candidates,
        key=lambda c: (
            info_cache[c.path].capture_time,
            info_cache[c.path].sequence_number
            if info_cache[c.path].sequence_number is not None
            else 10**15,
            str(c.path).casefold(),
        ),
    )




def _confirmed_foreign_target_between(
    person_id: str,
    previous: BestFrameCandidate,
    current: BestFrameCandidate,
    targets_by_path: dict[Path, set[str]],
    info_cache: dict[Path, PhotoOrderInfo],
    *,
    confirm_frames: int = 2,
) -> bool:
    """Conservative A/B/A boundary guard using already-confirmed reference IDs."""
    if confirm_frames <= 0 or previous.path.parent != current.path.parent:
        return False
    prev_info = info_cache[previous.path]
    cur_info = info_cache[current.path]
    between: list[Path] = []

    if (
        prev_info.sequence_number is not None
        and cur_info.sequence_number is not None
        and prev_info.sequence_source == cur_info.sequence_source
        and prev_info.sequence_number < cur_info.sequence_number
    ):
        for path in targets_by_path:
            if path in (previous.path, current.path):
                continue
            info = info_cache.get(path)
            if info is None or info.sequence_source != prev_info.sequence_source or info.sequence_number is None:
                continue
            if prev_info.sequence_number < info.sequence_number < cur_info.sequence_number:
                between.append(path)
    else:
        start = min(prev_info.capture_time, cur_info.capture_time)
        end = max(prev_info.capture_time, cur_info.capture_time)
        for path in targets_by_path:
            if path in (previous.path, current.path) or path.parent != previous.path.parent:
                continue
            info = info_cache.get(path)
            if info is not None and start < info.capture_time < end:
                between.append(path)

    foreign_counts: dict[str, int] = {}
    for path in set(between):
        targets = targets_by_path.get(path, set())
        if person_id in targets:
            continue
        for target in targets:
            if target == person_id:
                continue
            foreign_counts[target] = foreign_counts.get(target, 0) + 1
    return any(count >= confirm_frames for count in foreign_counts.values())


def split_target_series(
    candidates: list[BestFrameCandidate],
    *,
    max_gap_seconds: float = 12.0,
    max_filename_gap: int = 5,
    info_cache: dict[Path, PhotoOrderInfo] | None = None,
    targets_by_path: dict[Path, set[str]] | None = None,
    foreign_break_confirm_frames: int = 2,
) -> list[list[BestFrameCandidate]]:
    """Split already-identified frames into chronological shooting runs.

    Identity is intentionally not inferred here: every input row already belongs
    to one reference target. We only reproduce the hard temporal/filename logic
    from photo_select_ai's sequential portrait mode.
    """
    if not candidates:
        return []
    cache = info_cache if info_cache is not None else {}
    for candidate in candidates:
        cache.setdefault(candidate.path, read_photo_order_info(candidate.path))

    ordered = _ordered_candidates(candidates, cache)
    groups: list[list[BestFrameCandidate]] = [[ordered[0]]]

    for previous, current in zip(ordered, ordered[1:]):
        prev_info = cache[previous.path]
        cur_info = cache[current.path]

        same_directory = previous.path.parent == current.path.parent
        same_sequence_source = prev_info.sequence_source == cur_info.sequence_source
        seq_ok = True
        if prev_info.sequence_number is not None and cur_info.sequence_number is not None:
            delta = cur_info.sequence_number - prev_info.sequence_number
            seq_ok = same_sequence_source and 0 < delta <= max_filename_gap
        elif not same_directory:
            seq_ok = False

        # EXIF capture time is authoritative when both frames have it. If time is
        # only filesystem mtime and a camera sequence exists, rely on the camera
        # counter instead because copied files often share arbitrary mtimes.
        time_ok = True
        delta_seconds = max(0.0, (cur_info.capture_time - prev_info.capture_time).total_seconds())
        if prev_info.time_is_metadata and cur_info.time_is_metadata:
            time_ok = delta_seconds <= max_gap_seconds
        elif prev_info.sequence_number is None or cur_info.sequence_number is None:
            time_ok = delta_seconds <= max_gap_seconds

        foreign_boundary = False
        if targets_by_path is not None:
            foreign_boundary = _confirmed_foreign_target_between(
                previous.person_id,
                previous,
                current,
                targets_by_path,
                cache,
                confirm_frames=foreign_break_confirm_frames,
            )

        if same_directory and seq_ok and time_ok and not foreign_boundary:
            groups[-1].append(current)
        else:
            groups.append([current])

    return groups


def _rank(values: list[float], value: float) -> float:
    if not values:
        return 0.5
    ordered = sorted(values)
    if len(ordered) == 1 or ordered[-1] - ordered[0] <= 1e-9:
        return 0.5
    # Average rank for ties; values are tiny lists so the simple version is fine.
    less = sum(1 for item in ordered if item < value)
    equal = sum(1 for item in ordered if abs(item - value) <= 1e-9)
    index = less + max(0, equal - 1) / 2.0
    return index / max(1, len(ordered) - 1)


def _eye_value(candidate: BestFrameCandidate) -> float | None:
    if candidate.eye_left is None or candidate.eye_right is None:
        return None
    if not math.isfinite(candidate.eye_left) or not math.isfinite(candidate.eye_right):
        return None
    return min(float(candidate.eye_left), float(candidate.eye_right))


def _pose_score(candidate: BestFrameCandidate) -> float:
    if candidate.pose is None:
        return 0.65
    pitch, yaw, roll = candidate.pose
    penalty = max(abs(yaw) / 45.0, abs(pitch) / 35.0, abs(roll) / 35.0)
    return max(0.0, min(1.0, 1.0 - penalty))


def select_best_from_series(
    series: list[BestFrameCandidate],
    *,
    match_threshold: float,
    eye_open_threshold: float,
) -> tuple[BestFrameCandidate, float, float | None, float, float]:
    if not series:
        raise ValueError("Пустая серия")

    # Mirror photo_select_ai: if at least one frame has reliably open eyes, do
    # not let an otherwise sharp closed-eye frame win the series.
    reliably_open = [
        candidate
        for candidate in series
        if (_eye_value(candidate) is not None and _eye_value(candidate) >= eye_open_threshold)
    ]
    pool = reliably_open or list(series)

    blur_values = [
        math.log1p(max(0.0, float(candidate.blur_score)))
        for candidate in pool
        if candidate.blur_score is not None and math.isfinite(candidate.blur_score)
    ]
    eye_values = [value for candidate in pool if (value := _eye_value(candidate)) is not None]

    best_candidate = pool[0]
    best_score = -1.0
    best_eye: float | None = None
    best_blur_rank = 0.5
    best_pose = 0.65

    for candidate in pool:
        eye = _eye_value(candidate)
        if eye is None:
            eye_component = 0.35 if eye_values else 0.50
        else:
            eye_component = _rank(eye_values, eye) if len(eye_values) > 1 else 0.75

        if candidate.blur_score is None or not math.isfinite(candidate.blur_score):
            blur_rank = 0.35 if blur_values else 0.50
        else:
            blur_rank = _rank(blur_values, math.log1p(max(0.0, float(candidate.blur_score))))

        threshold = max(1e-6, float(match_threshold))
        identity = max(0.0, min(1.0, 1.0 - float(candidate.distance) / threshold))
        pose = _pose_score(candidate)
        det = max(0.0, min(1.0, float(candidate.det_score)))

        score = (
            0.40 * blur_rank
            + 0.32 * eye_component
            + 0.13 * identity
            + 0.10 * pose
            + 0.05 * det
        )

        # Deterministic tie-breaks favor better identity, then sharper face,
        # then filename order rather than whichever dict happened to iterate first.
        candidate_blur = float(candidate.blur_score) if candidate.blur_score is not None else -1.0
        best_blur = float(best_candidate.blur_score) if best_candidate.blur_score is not None else -1.0
        is_better = score > best_score + 1e-12
        if not is_better and abs(score - best_score) <= 1e-12:
            if candidate.distance < best_candidate.distance - 1e-12:
                is_better = True
            elif abs(candidate.distance - best_candidate.distance) <= 1e-12:
                if candidate_blur > best_blur + 1e-12:
                    is_better = True
                elif abs(candidate_blur - best_blur) <= 1e-12:
                    is_better = str(candidate.path).casefold() < str(best_candidate.path).casefold()
        if is_better:
            best_candidate = candidate
            best_score = score
            best_eye = eye
            best_blur_rank = blur_rank
            best_pose = pose

    return best_candidate, best_score, best_eye, best_blur_rank, best_pose


def select_best_series_frames(
    candidates: Iterable[BestFrameCandidate],
    *,
    match_threshold: float,
    eye_open_threshold: float,
    max_gap_seconds: float = 12.0,
    max_filename_gap: int = 5,
) -> list[BestSeriesSelection]:
    by_person: dict[str, list[BestFrameCandidate]] = {}
    all_candidates = list(candidates)
    for candidate in all_candidates:
        by_person.setdefault(candidate.person_id, []).append(candidate)

    info_cache: dict[Path, PhotoOrderInfo] = {}
    targets_by_path: dict[Path, set[str]] = {}
    for candidate in all_candidates:
        info_cache.setdefault(candidate.path, read_photo_order_info(candidate.path))
        targets_by_path.setdefault(candidate.path, set()).add(candidate.person_id)

    selections: list[BestSeriesSelection] = []
    for person_id in sorted(by_person, key=str.casefold):
        groups = split_target_series(
            by_person[person_id],
            max_gap_seconds=max_gap_seconds,
            max_filename_gap=max_filename_gap,
            info_cache=info_cache,
            targets_by_path=targets_by_path,
            foreign_break_confirm_frames=2,
        )
        for series_index, group in enumerate(groups, start=1):
            candidate, score, eye, sharp_rank, pose = select_best_from_series(
                group,
                match_threshold=match_threshold,
                eye_open_threshold=eye_open_threshold,
            )
            selections.append(
                BestSeriesSelection(
                    person_id=person_id,
                    series_index=series_index,
                    series_size=len(group),
                    candidate=candidate,
                    score=score,
                    eye_score=eye,
                    sharpness_rank=sharp_rank,
                    pose_score=pose,
                )
            )
    return selections
