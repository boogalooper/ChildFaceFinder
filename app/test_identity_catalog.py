from types import SimpleNamespace
import numpy as np
from identity_catalog import assign_identities


def face(v):
    return SimpleNamespace(embedding=np.array(v, dtype=np.float32), det_score=.9, bbox=(0, 0, 10, 10))


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
