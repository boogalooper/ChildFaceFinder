"""Persistent, conservative identity assignment without reference photographs."""
import json
import os
import re
from pathlib import Path

import numpy as np


def assign_identities(analyses, folder, threshold, margin, cancel):
    path = Path(folder) / 'ChildFaceFinder_people.json'
    records = []
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') != 1:
            raise ValueError('Неизвестная версия каталога ID')
        records = data['people']
    names = [re.sub(r'^Ребёнок_(\d+)$', r'Лицо_\1', r['id']) for r in records]
    vectors = [np.asarray(r['embedding'], dtype=np.float32) for r in records]
    if len(set(names)) != len(names):
        raise ValueError('Повторяющиеся ID в каталоге')
    for vector in vectors:
        if vector.ndim != 1 or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > .01:
            raise ValueError('Повреждён embedding в каталоге ID')
    assignments = {}
    for analysis in sorted(analyses, key=lambda a: str(a.path).casefold()):
        used = set()
        # Strong detections get first choice; two faces in one photo cannot share an ID.
        for face in sorted(analysis.faces, key=lambda f: (-f.det_score, f.bbox)):
            cancel()
            vector = np.asarray(face.embedding, dtype=np.float32)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            chosen = None
            if vectors:
                distances = 1 - np.stack(vectors) @ vector
                order = np.argsort(distances)
                best = int(order[0])
                if (names[best] not in used and distances[best] <= threshold
                        and (len(order) == 1 or distances[order[1]] - distances[best] >= margin)):
                    chosen = best
            if chosen is None:
                number = len(names) + 1
                while f'Лицо_{number:04d}' in names:
                    number += 1
                names.append(f'Лицо_{number:04d}')
                vectors.append(vector)
                chosen = len(names) - 1
            used.add(names[chosen])
            assignments[id(face)] = names[chosen]
    cancel()
    payload = {'version': 1, 'people': [
        {'id': name, 'embedding': vector.tolist()} for name, vector in zip(names, vectors)
    ]}
    temp = path.with_suffix('.json.tmp')
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return names, vectors, assignments
