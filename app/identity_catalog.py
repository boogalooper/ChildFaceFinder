"""Two-pass, constrained face grouping; no image/analysis cache."""
import json
import re
from pathlib import Path

import numpy as np
from runtime_safety import atomic_path, file_snapshot


def recognition_issues(face, *, anchor=False, min_face_px=45, blur_threshold=45):
    """Identity reliability, intentionally independent of eyes/composition filters."""
    issues = []
    try:
        box = np.asarray(face.bbox, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            return ['некорректные координаты лица']
        x1, y1, x2, y2 = box
        score = float(face.det_score)
    except (TypeError, ValueError):
        return ['некорректные данные детекции']
    size = min(x2 - x1, y2 - y1)
    if size < (max(60, min_face_px) if anchor else max(24, min_face_px * .5)):
        issues.append('недостаточный размер лица для распознавания')
    if not np.isfinite(score) or not 0 <= score <= 1 or score < (.6 if anchor else .25):
        issues.append('низкая уверенность детектора')
    blur = getattr(face, 'blur_score', None)
    if blur is not None and (not np.isfinite(blur) or blur < (max(30, blur_threshold) if anchor else max(10, blur_threshold * .25))):
        issues.append('недостаточная резкость для распознавания')
    pose = getattr(face, 'pose', None)
    if pose is not None:
        pose = np.asarray(pose)
        if pose.shape != (3,):
            return issues + ['некорректный ракурс лица']
        pitch, yaw, roll = pose
        limits = (35, 45, 45) if anchor else (65, 80, 85)
        if not np.isfinite(pose).all() or any(abs(v) > limit for v, limit in zip(pose, limits)):
            issues.append('слишком сложный ракурс для распознавания')
    if anchor and getattr(face, 'source', '').startswith('rescue-'):
        issues.append('восстановленная детекция не используется как образец')
    vector = np.asarray(face.embedding, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-8:
        issues.append('некорректный embedding')
    return issues


def _gallery(vectors, limit=8):
    """Deterministic farthest-point sampling retains different confident views."""
    matrix = np.stack(vectors)
    selected = [0]
    distances = 1 - matrix @ matrix[0]
    while len(selected) < min(limit, len(matrix)):
        distances[selected] = -1
        index = int(np.argmax(distances))
        if distances[index] < 1e-6:
            break
        selected.append(index)
        distances = np.minimum(distances, 1 - matrix @ matrix[index])
    return matrix[selected]


def assign_identities(analyses, folder, threshold, margin, cancel, *, details=None,
                      min_face_px=45, blur_threshold=45, log=None):
    """Group reliable faces globally, then match weaker faces to frozen galleries.

    details holds the same match tuples as the reference matcher. Unresolved
    faces are omitted from assignments and never create permanent identities.
    """
    if not np.isfinite([threshold, margin, min_face_px, blur_threshold]).all() or not 0 < threshold <= 1 or not 0 <= margin <= 1 or min_face_px <= 0 or blur_threshold < 0:
        raise ValueError('Некорректные параметры группировки')
    details = details if details is not None else {}
    log = log or (lambda text: None)
    path = Path(folder) / 'ChildFaceFinder_people.json'
    cancel()
    records = []
    before = file_snapshot(path)
    if before is not None:
        data = json.loads(before.decode('utf-8-sig'))
        if not isinstance(data, dict) or data.get('version') not in (1, 2):
            raise ValueError('Неизвестная версия каталога ID')
        records = data.get('people')
        if not isinstance(records, list):
            raise ValueError('Повреждён каталог ID: people должен быть списком')
    entries = []
    for analysis in sorted(analyses, key=lambda a: (str(a.path).casefold(), str(a.path))):
        for face in sorted(analysis.faces, key=lambda f: (f.bbox, -f.det_score)):
            entries.append((analysis.path, face))
    groups = []
    names_seen = set()
    dimension = None
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not record['id'].strip():
            raise ValueError('Повреждён каталог ID: отсутствует имя')
        name = re.sub(r'^Ребёнок_(\d+)$', r'Лицо_\1', record['id'])
        if name.casefold() in {n.casefold() for n in names_seen}:
            raise ValueError('Повторяющиеся ID в каталоге')
        names_seen.add(name)
        vectors = np.asarray(record.get('embeddings', [record.get('embedding')]), dtype=np.float64)
        if (vectors.ndim != 2 or not 1 <= len(vectors) <= 8 or not np.isfinite(vectors).all()
                or np.any(abs(np.linalg.norm(vectors, axis=1) - 1) > .01)):
            raise ValueError('Повреждён embedding в каталоге ID')
        if dimension is not None and dimension != vectors.shape[1]:
            raise ValueError('Несовместимые размеры embeddings каталога')
        dimension = vectors.shape[1]
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        groups.append(dict(name=name, vectors=list(vectors.astype(np.float32)), photos=set(), members=[]))
    options = dict(min_face_px=min_face_px, blur_threshold=blur_threshold)
    for index, (photo, face) in enumerate(entries):
        cancel()
        if recognition_issues(face, anchor=True, **options):
            continue
        vector = np.asarray(face.embedding, dtype=np.float64)
        vector = (vector / np.linalg.norm(vector)).astype(np.float32)
        if dimension is not None and dimension != len(vector):
            raise ValueError('Каталог ID создан другой моделью: размер embeddings отличается')
        dimension = len(vector)
        groups.append(dict(name=None, vectors=[vector], photos={photo}, members=[index]))
    # Build bounded nearest-neighbour edges in blocks; never allocate an N x N matrix.
    parents = list(range(len(groups)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    strict = min(.32, max(.02, threshold - max(.04, margin)))
    log(f'Группировка: надёжных лиц {sum(len(g["members"]) for g in groups)}; строгий порог {strict:.3f}')
    edges = []
    deferred = set()
    if groups:
        owners = np.array([i for i, g in enumerate(groups) for _ in g['vectors']])
        matrix = np.stack([v for g in groups for v in g['vectors']])
        for start in range(0, len(matrix), 128):
            cancel()
            distances = 1 - matrix[start:start + 128] @ matrix.T
            for offset, row in enumerate(distances):
                i = start + offset
                row[owners == owners[i]] = np.inf
                k = min(16, len(row))
                candidates = np.argpartition(row, k - 1)[:k]
                # If equally plausible neighbours are different faces from the
                # same photograph, do not use this face to join either group.
                nearest = float(np.min(row[candidates]))
                close = [int(owners[j]) for j in candidates
                         if row[j] <= strict and row[j] - nearest < max(margin, 1e-6)]
                if groups[int(owners[i])]['name'] is None:
                    if len({groups[o]['name'] for o in close if groups[o]['name'] is not None}) > 1:
                        deferred.add(int(owners[i]))
                    competing_photos = set()
                    for owner in sorted(set(close)):
                        photos = groups[owner]['photos']
                        if photos & competing_photos:
                            deferred.add(int(owners[i]))
                            break
                        competing_photos.update(photos)
                for j in candidates:
                    if row[j] <= strict:
                        a, b = sorted((int(owners[i]), int(owners[j])))
                        edges.append((float(row[j]), a, b))
            if start % 1024 == 0:
                log(f'Группировка: сравнение образцов {min(start + 128, len(matrix))}/{len(matrix)}')
    # Complete-link prevents a chain of lookalikes gradually joining two people.
    for distance, a, b in sorted(set(edges)):
        cancel()
        if a in deferred or b in deferred:
            continue
        a, b = root(a), root(b)
        if a == b:
            continue
        ga, gb = groups[a], groups[b]
        if ga['photos'] & gb['photos'] or (ga['name'] and gb['name']):
            continue
        va, vb = np.stack(ga['vectors']), np.stack(gb['vectors'])
        compatible = True
        for start in range(0, len(va), 128):
            cancel()
            if np.any(1 - va[start:start + 128] @ vb.T > strict + 1e-6):
                compatible = False
                break
        if not compatible:
            continue
        parents[b] = a
        ga['name'] = ga['name'] or gb['name']
        ga['vectors'].extend(gb['vectors'])
        ga['photos'].update(gb['photos'])
        ga['members'].extend(gb['members'])
        gb['vectors'] = []
        gb['photos'] = set()
        gb['members'] = []
    groups = [g for i, g in enumerate(groups) if root(i) == i and i not in deferred]
    number = max([int(n.split('_')[1]) for n in names_seen if re.fullmatch(r'Лицо_\d+', n)] or [0]) + 1
    for group in groups:
        if group['name'] is None:
            while f'Лицо_{number:04d}' in names_seen:
                number += 1
            group['name'] = f'Лицо_{number:04d}'
            names_seen.add(group['name'])
            number += 1
        group['gallery'] = _gallery(group['vectors'])
    names = [g['name'] for g in groups]
    assignments = {}
    used = {}
    for group in groups:
        for index in group['members']:
            photo, face = entries[index]
            assignments[id(face)] = group['name']
            details[id(face)] = (group['name'], group['name'], 0., None, None, None)
            used.setdefault(photo, set()).add(group['name'])
    # Freeze samples here: second-pass matches never become new anchors.
    proposals = []
    for index, (photo, face) in enumerate(entries):
        cancel()
        if id(face) in assignments:
            continue
        issues = recognition_issues(face, **options)
        if issues or not groups:
            details[id(face)] = (None, '', float('inf'), None, None,
                                 '; '.join(issues) if issues else 'нет надёжных образцов')
            continue
        vector = np.asarray(face.embedding, dtype=np.float64)
        vector = (vector / np.linalg.norm(vector)).astype(np.float32)
        if len(vector) != dimension:
            raise ValueError('Несовместимый размер embedding лица')
        distances = [float(np.min(1 - g['gallery'] @ vector)) for g in groups]
        order = np.argsort(distances, kind='stable')
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else None
        dist = distances[best]
        second_dist = distances[second] if second is not None else None
        problem = None
        if dist > threshold:
            problem = 'дистанция выше порога; отдельный ID не создан без надёжного образца'
        elif second_dist is not None and second_dist - dist < max(margin, 1e-6):
            problem = 'слишком близкая вторая цель'
        info = (None, names[best], dist, names[second] if second is not None else None, second_dist, problem)
        details[id(face)] = info
        if problem is None:
            proposals.append((dist, index, names[best]))
    # Competing weak detections cannot silently steal one identity in one photo.
    competing = {}
    for dist, index, name in proposals:
        competing.setdefault((entries[index][0], name), []).append((dist, index))
    for dist, index, name in sorted(proposals):
        cancel()
        photo, face = entries[index]
        competitors = [d for d, j in competing[(photo, name)] if j != index]
        if name in used.get(photo, set()):
            problem = 'этот человек уже найден в другом лице того же кадра'
        elif competitors and min(competitors) - dist < max(margin, 1e-6):
            problem = 'два лица одного кадра претендуют на один ID'
        else:
            problem = None
            assignments[id(face)] = name
            used.setdefault(photo, set()).add(name)
        old = details[id(face)]
        details[id(face)] = (name if problem is None else None, *old[1:5], problem)
    # Flag separate but similar groups for review, without forcing a merge.
    warnings = {}
    for a, ga in enumerate(groups):
        cancel()
        for gb in groups[a + 1:]:
            if ga['photos'] & gb['photos']:
                continue
            distance = float(np.min(1 - ga['gallery'] @ gb['gallery'].T))
            if distance <= threshold:
                warnings.setdefault(ga['name'], []).append(gb['name'])
                warnings.setdefault(gb['name'], []).append(ga['name'])
    for _photo, face in entries:
        name = assignments.get(id(face))
        if name in warnings:
            old = details[id(face)]
            details[id(face)] = (*old[:5], 'похожие отдельные группы: ' + ', '.join(warnings[name]))
    cancel()
    payload = {'version': 2, 'people': [
        {'id': g['name'], 'embeddings': g['gallery'].tolist()} for g in groups
    ]}
    with atomic_path(path, expected=before, cancel=cancel) as temp:
        temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    log(f'Группировка: людей в текущей папке {len(set(assignments.values()))}; '
        f'лиц определено {len(assignments)}/{len(entries)}')
    return names, [g['gallery'][0] for g in groups], assignments


def unique_photo_matches(rows, margin):
    """At most one face per identity; close competitors remain unresolved."""
    result = list(rows)
    candidates = {}
    for index, (_face, info) in enumerate(rows):
        if info[0] is not None:
            candidates.setdefault(info[0], []).append((info[2], index))
    for values in candidates.values():
        if len(values) < 2:
            continue
        values.sort()
        winner = values[0][1] if values[1][0] - values[0][0] >= max(margin, 1e-6) else None
        for _distance, index in values:
            if index == winner:
                continue
            face, info = result[index]
            result[index] = (face, (None, *info[1:5], 'два лица одного кадра претендуют на одного человека'))
    return result
