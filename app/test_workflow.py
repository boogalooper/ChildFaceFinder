import csv
import json
import xml.etree.ElementTree as ET
import pytest
import numpy as np
from workflow import WorkflowOptions
from metadata_writer import XmpMetadataWriter, RDF_NS, DC_NS, LR_NS, XMP_NS, IPTC_NS
from file_output import export_people


def values(path, ns, field):
    root = ET.parse(path).getroot()
    return [n.text for n in root.findall(f'.//{{{ns}}}{field}/{{{RDF_NS}}}Bag/{{{RDF_NS}}}li')]


@pytest.mark.parametrize('scenario', ['references', 'discover'])
@pytest.mark.parametrize('output', ['csv', 'folders', 'analysis'])
@pytest.mark.parametrize('people,best', [(False, False), (True, False), (False, True), (True, True)])
def test_workflow_matrix(engine_module, monkeypatch, tmp_path, scenario, output, people, best):
    e = engine_module
    source, refs, dest = tmp_path/'source', tmp_path/'refs', tmp_path/'result'
    source.mkdir(); refs.mkdir()
    image = source/'IMG_0001.CR2'
    image.write_bytes(b'original')
    wf = WorkflowOptions(scenario=scenario, output=output, destination=dest,
                         xmp_enabled=people or best, write_people=people, best_frames=best, best_label='MyRed')
    runner = e.ChildFaceFinder(e.Settings(workflow=wf))
    runner.engine.initialize = lambda: None
    monkeypatch.setattr(e, 'iter_candidate_files', lambda *args: [image])
    monkeypatch.setattr(e, 'cleanup_stale_temp_dirs', lambda: ([], []))
    face = e.FaceRecord(np.array([1., 0.], dtype=np.float32), (0, 0, 100, 100), (0, 0, 100, 100),
                        .95, quality_reasons=['закрытые глаза'], blur_score=100., eye_left=.01, eye_right=.01, pose=(0, 0, 0))
    runner._pipeline = lambda *args, **kwargs: [e.ImageAnalysis(image, [face])]
    runner._build_references = lambda *args: e.ReferenceSet(['Иван'], np.array([[1., 0.]], dtype=np.float32), np.array([0]), 1)
    ref_arg = refs if scenario == 'references' else None
    csv_path = tmp_path/'list.csv' if output == 'csv' else None
    if output == 'analysis' and not (people or best):
        with pytest.raises(ValueError, match='Только анализ'):
            runner.run(ref_arg, source, csv_path)
        return
    result = runner.run(ref_arg, source, csv_path)
    expected = 'Иван' if scenario == 'references' else 'Лицо_0001'
    assert result.found_people == 1
    assert result.identified_faces == 1
    assert image.read_bytes() == b'original'
    if output == 'csv':
        assert list(tmp_path.rglob('*.csv')) == [csv_path]
        rows = list(csv.reader(csv_path.open(encoding='utf-8-sig'), delimiter=';'))
        assert rows[1][0] == expected
        assert rows[1][3] == ('да' if best else '')
    else:
        assert not list(tmp_path.rglob('*.csv'))
        assert result.output_csv is None
    xmp = image.with_suffix('.xmp')
    assert xmp.exists() == (people or best)
    if people or best:
        tree = ET.parse(xmp).getroot()
        assert values(xmp, DC_NS, 'subject') == ([expected] if people else [])
        label = tree.find(f'.//{{{XMP_NS}}}Label')
        assert (label.text if label is not None else '') == ('MyRed' if best else '')
        assert 'ChildFaceFinder|Best' not in values(xmp, LR_NS, 'hierarchicalSubject')
    if output == 'folders':
        assert (dest/expected/image.name).read_bytes() == b'original'
        assert (dest/expected/xmp.name).exists() == (people or best)
        assert result.file_copies == 1


def test_people_replacement_preserves_label_rating_and_camera_raw(tmp_path):
    image = tmp_path/'photo.CR2'
    path = image.with_suffix('.xmp')
    writer = XmpMetadataWriter()
    writer.export({image: {'Старый'}})
    tree = ET.parse(path)
    desc = tree.getroot().find(f'.//{{{RDF_NS}}}Description')
    desc.set(f'{{{XMP_NS}}}Label', 'BlueCustom')
    desc.set(f'{{{XMP_NS}}}Rating', '4')
    desc.set('{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012', '1.25')
    bag = desc.find(f'{{{DC_NS}}}subject/{{{RDF_NS}}}Bag')
    ET.SubElement(bag, f'{{{RDF_NS}}}li').text = 'Школа'
    tree.write(path, encoding='utf-8')
    kwargs = dict(sync_people=True, legacy_best=False)
    writer.export({image: {'Новый'}}, **kwargs)
    assert values(path, DC_NS, 'subject') == ['Школа', 'Новый']
    assert values(path, IPTC_NS, 'PersonInImage') == ['Новый']
    desc = ET.parse(path).getroot().find(f'.//{{{RDF_NS}}}Description')
    assert desc.get(f'{{{XMP_NS}}}Label') == 'BlueCustom'
    assert desc.get(f'{{{XMP_NS}}}Rating') == '4'
    assert desc.get('{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012') == '1.25'
    before = path.read_bytes()
    assert writer.export({image: {'Новый'}}, **kwargs).unchanged_files == 1
    assert path.read_bytes() == before
    writer.export({}, all_photos=[image], **kwargs)
    assert values(path, DC_NS, 'subject') == ['Школа']
    assert not values(path, IPTC_NS, 'PersonInImage')


def test_color_only_changes_label_not_people(tmp_path):
    a, b = tmp_path/'a.CR2', tmp_path/'b.CR2'
    writer = XmpMetadataWriter()
    writer.export({a: {'Иван'}, b: {'Анна'}})
    kwargs = dict(write_people=False, sync_labels=True, legacy_best=False, best_label='Select')
    writer.export({}, {a}, all_photos=[a, b], **kwargs)
    writer.export({}, {b}, all_photos=[a, b], **kwargs)
    assert ET.parse(a.with_suffix('.xmp')).getroot().find(f'.//{{{XMP_NS}}}Label') is None
    assert ET.parse(b.with_suffix('.xmp')).getroot().find(f'.//{{{XMP_NS}}}Label').text == 'Select'
    assert values(a.with_suffix('.xmp'), DC_NS, 'subject') == ['Иван']
    assert values(b.with_suffix('.xmp'), DC_NS, 'subject') == ['Анна']


def test_group_move_copies_image_and_xmp_to_every_person(tmp_path):
    source, dest = tmp_path/'source', tmp_path/'dest'
    source.mkdir()
    image = source/'photo.CR2'; image.write_bytes(b'original raw')
    sidecar = image.with_suffix('.XMP'); sidecar.write_bytes(b'original metadata')
    result = export_people({image: {'А', 'Б'}}, source, dest, move=True)
    assert result.moved == 1 and result.copies == 2
    assert not image.exists() and not sidecar.exists()
    for person in ['А', 'Б']:
        assert (dest/person/image.name).read_bytes() == b'original raw'
        assert (dest/person/'photo.xmp').read_bytes() == b'original metadata'
    assert json.loads(result.journal.read_text())['items'][0]['status'] == 'moved'


def test_failed_second_person_keeps_entire_source_pair(tmp_path, monkeypatch):
    import file_output
    source, dest = tmp_path/'source', tmp_path/'dest'
    source.mkdir()
    image = source/'photo.CR2'; image.write_bytes(b'raw')
    sidecar = image.with_suffix('.xmp'); sidecar.write_bytes(b'xmp')
    original = file_output._copy_verified
    def fail(src, target, cancel):
        if target.parent.name == 'Б':
            raise OSError('disk full')
        return original(src, target, cancel)
    monkeypatch.setattr(file_output, '_copy_verified', fail)
    result = export_people({image: {'А', 'Б'}}, source, dest, move=True)
    assert result.failed == 1 and result.moved == 0
    assert image.read_bytes() == b'raw' and sidecar.read_bytes() == b'xmp'
    assert (dest/'А'/'photo.CR2').read_bytes() == b'raw'


def test_copy_name_conflicts_keep_matching_sidecars(tmp_path):
    source, dest = tmp_path/'source', tmp_path/'dest'
    source.mkdir(); (source/'sub').mkdir()
    a, b = source/'photo.CR2', source/'sub'/'photo.CR2'
    for number, path in enumerate((a, b)):
        path.write_bytes(str(number).encode())
        path.with_suffix('.xmp').write_bytes(('x'+str(number)).encode())
    result = export_people({a: {'А'}, b: {'А'}}, source, dest)
    assert result.copies == 2
    for path in (dest/'А').glob('*.CR2'):
        assert path.with_suffix('.xmp').read_bytes() == b'x'+path.read_bytes()
    assert a.exists() and b.exists()


def test_move_never_deletes_shared_sidecar(tmp_path):
    source, dest = tmp_path/'source', tmp_path/'dest'
    source.mkdir()
    a, b = source/'photo.CR2', source/'photo.jpg'
    a.write_bytes(b'raw'); b.write_bytes(b'jpeg')
    sidecar = source/'photo.xmp'; sidecar.write_bytes(b'xmp')
    result = export_people({a: {'А'}}, source, dest, move=True, all_photos=[a, b])
    assert result.moved == 0 and result.failed == 1
    assert a.exists() and b.exists() and sidecar.exists()


def test_cancel_before_delete_keeps_originals(tmp_path, monkeypatch):
    import file_output
    source, dest = tmp_path/'source', tmp_path/'dest'
    source.mkdir(); image = source/'photo.CR2'; image.write_bytes(b'raw')
    class Stop(RuntimeError): pass
    cancelled = False
    original = file_output._copy_verified
    def copied(src, target, cancel):
        nonlocal cancelled
        result = original(src, target, cancel)
        cancelled = True
        return result
    monkeypatch.setattr(file_output, '_copy_verified', copied)
    def cancel():
        if cancelled: raise Stop()
    with pytest.raises(Stop):
        export_people({image: {'А'}}, source, dest, move=True, cancel=cancel)
    assert image.read_bytes() == b'raw'


def test_disallow_destination_in_source(tmp_path):
    wf = WorkflowOptions(scenario='discover', output='folders', destination=tmp_path/'sub')
    with pytest.raises(ValueError, match='отдельно'):
        wf.validate(None, tmp_path, None)


def test_master_off_keeps_existing_xmp_byte_exact(engine_module, monkeypatch, tmp_path):
    e = engine_module
    source = tmp_path/'source'; source.mkdir()
    refs = tmp_path/'refs'; refs.mkdir()
    image = source/'photo.CR2'; image.write_bytes(b'raw')
    sidecar = image.with_suffix('.xmp'); sidecar.write_bytes(b'original noncanonical XMP')
    wf = WorkflowOptions(output='folders', destination=tmp_path/'dest',
                         xmp_enabled=False, write_people=True, best_frames=True)
    runner = e.ChildFaceFinder(e.Settings(workflow=wf))
    runner.engine.initialize = lambda: None
    monkeypatch.setattr(e, 'iter_candidate_files', lambda *args: [image])
    monkeypatch.setattr(e, 'cleanup_stale_temp_dirs', lambda: ([], []))
    face = e.FaceRecord(np.array([1., 0.], dtype=np.float32), (0, 0, 100, 100), (0, 0, 100, 100), .95)
    runner._pipeline = lambda *args, **kwargs: [e.ImageAnalysis(image, [face])]
    runner._build_references = lambda *args: e.ReferenceSet(['Иван'], np.array([[1., 0.]], dtype=np.float32), np.array([0]), 1)
    runner.run(refs, source, None)
    assert sidecar.read_bytes() == b'original noncanonical XMP'
    assert (tmp_path/'dest'/'Иван'/'photo.xmp').read_bytes() == sidecar.read_bytes()


def test_bad_xmp_identifies_only_failed_photo(tmp_path):
    good, bad = tmp_path/'good.CR2', tmp_path/'bad.CR2'
    bad.with_suffix('.xmp').write_text('not XML')
    result = XmpMetadataWriter().export({good: {'Иван'}, bad: {'Анна'}},
                                      sync_people=True, legacy_best=False)
    assert result.failed_photos == {bad}
    assert values(good.with_suffix('.xmp'), DC_NS, 'subject') == ['Иван']


@pytest.mark.parametrize('move', [False, True])
def test_output_progress_resets_and_tracks_completed_photos(tmp_path, move):
    source = tmp_path/'source'; source.mkdir()
    a, b = source/'a.CR2', source/'b.CR2'
    for p in (a, b):
        p.write_bytes(b'photo')
        p.with_suffix('.xmp').write_bytes(b'xmp')
    events = []
    def progress(done, total, name):
        if done:
            # A photo advances progress only after all person copies and XMP exist.
            previous = a if done == 1 else b
            for person in ('A', 'B'):
                assert (tmp_path/'dest'/person/previous.name).exists()
                assert (tmp_path/'dest'/person/previous.with_suffix('.xmp').name).exists()
        events.append((done, total, name))
    export_people({a: {'A', 'B'}, b: {'A', 'B'}}, source, tmp_path/'dest',
                  move=move, progress=progress)
    assert events[0][:2] == (0, 2)
    assert events[-1][:2] == (2, 2)
    assert [v[0] for v in events] == [0, 0, 1, 1, 2]
    assert all(('Перемещение' if move else 'Копирование') in v[2] for v in events)
