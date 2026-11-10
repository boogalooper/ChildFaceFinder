from types import SimpleNamespace
import numpy as np
from identity_catalog import assign_identities


def face(v):
    return SimpleNamespace(embedding=np.array(v, dtype=np.float32), det_score=.9, bbox=(0, 0, 100, 100))


def test_persistent_ids_and_same_photo_constraint(tmp_path):
    a, b, c = face([1, 0]), face([0, 1]), face([1, 0])
    photos = [SimpleNamespace(path=tmp_path/'a.cr2', faces=[a, b]),
              SimpleNamespace(path=tmp_path/'b.cr2', faces=[c])]
    names, _, ids = assign_identities(photos, tmp_path, .4, .04, lambda: None)
    assert ids[id(a)] == ids[id(c)] != ids[id(b)]
    d, e = face([1, 0]), face([1, 0])
    _, _, repeated = assign_identities([SimpleNamespace(path=tmp_path/'c.cr2', faces=[d, e])], tmp_path, .4, .04, lambda: None)
    assert repeated[id(d)] == ids[id(a)]
    assert repeated[id(d)] != repeated[id(e)]


def test_cancel_does_not_overwrite_catalog(tmp_path):
    import pytest
    def cancel():
        raise RuntimeError('cancel')
    with pytest.raises(RuntimeError):
        assign_identities([], tmp_path, .4, .04, cancel)
    assert not (tmp_path/'ChildFaceFinder_people.json').exists()


def test_legacy_catalog_migration(tmp_path):
    import json
    path = tmp_path / 'ChildFaceFinder_people.json'
    path.write_text(json.dumps({'version': 1, 'people': [
        {'id': 'Ребёнок_0007', 'embedding': [1., 0.]}
    ]}), encoding='utf-8')
    a = face([1, 0])
    names, _, ids = assign_identities([SimpleNamespace(path=tmp_path/'a.cr2', faces=[a])],
                                      tmp_path, .4, .04, lambda: None)
    assert names == ['Лицо_0007']
    assert ids[id(a)] == 'Лицо_0007'
    assert json.loads(path.read_text())['people'][0]['id'] == 'Лицо_0007'


def test_legacy_xmp_migration(tmp_path):
    from metadata_writer import XmpMetadataWriter
    image = tmp_path/'a.cr2'
    writer = XmpMetadataWriter()
    writer.export({image: {'Ребёнок_0007', 'Семья'}})
    writer.export({image: {'Лицо_0007'}}, rename_people={'Ребёнок_0007': 'Лицо_0007'})
    text = image.with_suffix('.xmp').read_text()
    assert 'Ребёнок_0007' not in text
    assert 'Лицо_0007' in text
    assert 'People|Лицо_0007' in text
    assert 'Семья' in text


def angled(degrees, *, detector=.9, blur=100):
    radians = np.deg2rad(degrees)
    result = face([np.cos(radians), np.sin(radians)])
    result.det_score = detector
    result.blur_score = blur
    result.quality_reasons = []
    return result


def run_faces(tmp_path, faces, *, threshold=.45, margin=.04):
    analyses = [SimpleNamespace(path=tmp_path/f'{i}.cr2', faces=[f]) for i, f in enumerate(faces)]
    details = {}
    result = assign_identities(analyses, tmp_path, threshold, margin, lambda: None, details=details)
    return result, details


def test_multiple_views_help_second_pass(tmp_path):
    import json
    front, turned, difficult = angled(0), angled(40), angled(75, detector=.5)
    (names, _, ids), details = run_faces(tmp_path, [front, turned, difficult])
    assert len(names) == 1
    assert len(set(ids.values())) == 1 and len(ids) == 3
    assert details[id(difficult)][2] > 0
    saved = json.loads((tmp_path/'ChildFaceFinder_people.json').read_text())
    assert saved['version'] == 2
    assert len(saved['people'][0]['embeddings']) == 2


def test_second_pass_cannot_propagate_a_chain(tmp_path):
    anchor, weak, distant = angled(0), angled(40, detector=.5), angled(80, detector=.5)
    (names, _, ids), details = run_faces(tmp_path, [anchor, weak, distant])
    assert len(names) == 1
    assert ids[id(anchor)] == ids[id(weak)]
    assert id(distant) not in ids
    assert 'дистанция' in details[id(distant)][5]


def test_complete_link_blocks_chain_in_first_pass(tmp_path):
    a, b, c = angled(0), angled(35), angled(70)
    (names, _, ids), details = run_faces(tmp_path, [a, b, c])
    assert len(names) == 2
    assert ids[id(a)] != ids[id(c)]
    assert any('похожие отдельные группы' in (info[5] or '') for info in details.values())


def test_input_order_does_not_change_grouping(tmp_path):
    a, b, c = angled(0), angled(35), angled(140)
    analyses = [SimpleNamespace(path=tmp_path/f'{i}.cr2', faces=[f]) for i, f in enumerate([a, b, c])]
    first = assign_identities(analyses, tmp_path, .45, .04, lambda: None)[2]
    (tmp_path/'ChildFaceFinder_people.json').unlink()
    second = assign_identities(list(reversed(analyses)), tmp_path, .45, .04, lambda: None)[2]
    assert first == second


def test_closed_eyes_are_not_an_identity_quality_failure(tmp_path):
    a = angled(0)
    a.quality_reasons = ['закрытые глаза', 'обрезана макушка']
    (_, _, ids), _ = run_faces(tmp_path, [a])
    assert id(a) in ids


def test_bad_faces_do_not_create_ids(tmp_path):
    a, b = angled(0, blur=5), angled(90)
    b.bbox = (0, 0, 15, 15)
    (names, _, ids), details = run_faces(tmp_path, [a, b])
    assert names == [] and ids == {}
    assert all(info[0] is None and info[5] for info in details.values())


def test_ambiguous_weak_face_is_unresolved(tmp_path):
    a, b, c = angled(-50), angled(50), angled(0, detector=.5)
    (names, _, ids), details = run_faces(tmp_path, [a, b, c])
    assert len(names) == 2 and id(c) not in ids
    assert 'вторая цель' in details[id(c)][5]


def test_cooccurrence_survives_transitive_merges(tmp_path):
    a, b, c = angled(0), angled(10), angled(5)
    photos = [SimpleNamespace(path=tmp_path/'group.cr2', faces=[a, b]),
              SimpleNamespace(path=tmp_path/'single.cr2', faces=[c])]
    _, _, ids = assign_identities(photos, tmp_path, .45, .04, lambda: None)
    assert ids[id(a)] != ids[id(b)]
    assert len(set(ids.values())) == 2


def test_two_weak_faces_cannot_take_same_identity(tmp_path):
    a, b, c = angled(0), angled(20, detector=.5), angled(20, detector=.5)
    photos = [SimpleNamespace(path=tmp_path/'single.cr2', faces=[a]),
              SimpleNamespace(path=tmp_path/'group.cr2', faces=[b, c])]
    details = {}
    _, _, ids = assign_identities(photos, tmp_path, .45, .04, lambda: None, details=details)
    assert id(b) not in ids and id(c) not in ids
    assert 'одного кадра' in details[id(b)][5]


def test_reference_conflict_does_not_select_arbitrarily():
    from identity_catalog import unique_photo_matches
    a, b = angled(0), angled(0)
    info = ('Иван', 'Иван', .1, None, None, None)
    result = unique_photo_matches([(a, info), (b, info)], .04)
    assert all(row[1][0] is None for row in result)
    weaker = ('Иван', 'Иван', .2, None, None, None)
    result = unique_photo_matches([(a, info), (b, weaker)], .04)
    assert result[0][1][0] == 'Иван' and result[1][1][0] is None


def test_saved_multiple_views_remain_usable(tmp_path):
    a, b = angled(0), angled(40)
    (_, _, ids), _ = run_faces(tmp_path, [a, b])
    c = angled(75, detector=.5)
    (_, _, repeated), _ = run_faces(tmp_path, [c])
    assert repeated[id(c)] == ids[id(a)]


def test_ambiguous_strong_bridge_is_deferred(tmp_path):
    a, b, c = angled(-40), angled(40), angled(0)
    photos = [SimpleNamespace(path=tmp_path/'group.cr2', faces=[a, b]),
              SimpleNamespace(path=tmp_path/'single.cr2', faces=[c])]
    details = {}
    names, _, ids = assign_identities(photos, tmp_path, .45, .04, lambda: None, details=details)
    assert len(names) == 2
    assert id(c) not in ids
    assert 'вторая цель' in details[id(c)][5]


def test_cancel_during_comparisons_preserves_catalog(tmp_path):
    import pytest
    a = angled(0)
    run_faces(tmp_path, [a])
    path = tmp_path/'ChildFaceFinder_people.json'
    before = path.read_bytes()
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError('cancel')
    with pytest.raises(RuntimeError, match='cancel'):
        assign_identities([SimpleNamespace(path=tmp_path/'new.cr2', faces=[angled(5), angled(90)])],
                          tmp_path, .45, .04, cancel)
    assert path.read_bytes() == before


def test_nan_detector_cannot_create_identity(tmp_path):
    a = angled(0, detector=float('nan'))
    (names, _, ids), details = run_faces(tmp_path, [a])
    assert not names and not ids
    assert 'детектора' in details[id(a)][5]


def test_zero_margin_still_rejects_exact_tie(tmp_path):
    a, b, c = angled(-50), angled(50), angled(0, detector=.5)
    (_, _, ids), details = run_faces(tmp_path, [a, b, c], margin=0)
    assert id(c) not in ids
    assert 'вторая цель' in details[id(c)][5]


def test_ambiguous_saved_identities_are_not_chosen_arbitrarily(tmp_path):
    import json
    a, b, c = angled(-40), angled(40), angled(0)
    catalog = {'version': 2, 'people': [
        {'id': 'Лицо_0001', 'embeddings': [a.embedding.tolist()]},
        {'id': 'Лицо_0002', 'embeddings': [b.embedding.tolist()]}
    ]}
    (tmp_path/'ChildFaceFinder_people.json').write_text(json.dumps(catalog), encoding='utf-8')
    (names, _, ids), details = run_faces(tmp_path, [c])
    assert len(names) == 2 and id(c) not in ids
    assert 'вторая цель' in details[id(c)][5]


def test_corrupted_catalog_is_not_overwritten(tmp_path):
    import json
    import pytest
    for payload in [[], {'version': 2, 'people': None}, {'version': 2, 'people': [{'id': 'X', 'embeddings': [[0, 0]]}]}]:
        path = tmp_path/'ChildFaceFinder_people.json'
        before = json.dumps(payload).encode()
        path.write_bytes(before)
        with pytest.raises(ValueError):
            run_faces(tmp_path, [angled(0)])
        assert path.read_bytes() == before
