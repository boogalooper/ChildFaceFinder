from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from metadata_writer import DC_NS, LR_NS, RDF_NS, XmpMetadataWriter


def _values(path: Path, tag: str) -> list[str]:
    root = ET.parse(path).getroot()
    node = root.find(f".//{tag}/{{{RDF_NS}}}Bag")
    assert node is not None
    return [(li.text or "") for li in node.findall(f"{{{RDF_NS}}}li")]


def test_creates_sidecar_and_is_idempotent(tmp_path: Path) -> None:
    image = tmp_path / "IMG_0001.CR3"
    image.write_bytes(b"not modified")
    writer = XmpMetadataWriter()

    first = writer.export({image: {"Котов Никита", "Иванова Мария"}}, {image})
    assert first.written_files == 1
    assert image.read_bytes() == b"not modified"
    xmp = tmp_path / "IMG_0001.xmp"
    assert xmp.exists()
    assert sorted(_values(xmp, f"{{{DC_NS}}}subject")) == ["Иванова Мария", "Котов Никита"]
    assert "People|Котов Никита" in _values(xmp, f"{{{LR_NS}}}hierarchicalSubject")
    assert "ChildFaceFinder|Best" in _values(xmp, f"{{{LR_NS}}}hierarchicalSubject")

    before = xmp.read_bytes()
    second = writer.export({image: {"Котов Никита", "Иванова Мария"}}, {image})
    assert second.unchanged_files == 1
    assert xmp.read_bytes() == before


def test_merges_existing_unknown_metadata(tmp_path: Path) -> None:
    image = tmp_path / "IMG_0002.CR2"
    image.write_bytes(b"jpeg")
    xmp = tmp_path / "IMG_0002.xmp"
    xmp.write_text(
        '''<?xml version="1.0" encoding="utf-8"?>\n'''
        '''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'''
        '''<rdf:RDF><rdf:Description crs:Exposure2012="1.25"><dc:subject><rdf:Bag><rdf:li>Школа</rdf:li></rdf:Bag></dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>''',
        encoding="utf-8",
    )
    writer = XmpMetadataWriter()
    result = writer.export({image: {"Котов Никита"}})
    assert result.written_files == 1
    root = ET.parse(xmp).getroot()
    desc = root.find(f".//{{{RDF_NS}}}Description")
    assert desc is not None
    assert desc.attrib["{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012"] == "1.25"
    assert _values(xmp, f"{{{DC_NS}}}subject") == ["Школа", "Котов Никита"]


def test_broken_existing_xmp_is_not_overwritten(tmp_path: Path) -> None:
    image = tmp_path / "IMG_0003.NEF"
    image.write_bytes(b"raw")
    xmp = tmp_path / "IMG_0003.xmp"
    broken = b"<not-valid"
    xmp.write_bytes(broken)
    writer = XmpMetadataWriter()
    result = writer.export({image: {"Котов Никита"}})
    assert result.failed_files == 1
    assert xmp.read_bytes() == broken


def test_best_marker_is_synchronized_without_removing_people(tmp_path: Path) -> None:
    best_old = tmp_path / "old.CR3"
    best_new = tmp_path / "new.CR3"
    best_old.write_bytes(b"old")
    best_new.write_bytes(b"new")
    writer = XmpMetadataWriter()

    writer.export(
        {best_old: {"Котов Никита"}, best_new: {"Котов Никита"}},
        {best_old},
        all_photos=[best_old, best_new],
    )
    result = writer.export(
        {best_new: {"Котов Никита"}},
        {best_new},
        all_photos=[best_old, best_new],
    )
    assert result.written_files == 2
    old_h = _values(tmp_path / "old.xmp", f"{{{LR_NS}}}hierarchicalSubject")
    new_h = _values(tmp_path / "new.xmp", f"{{{LR_NS}}}hierarchicalSubject")
    assert "ChildFaceFinder|Best" not in old_h
    assert "People|Котов Никита" in old_h  # add-only person metadata is preserved
    assert "ChildFaceFinder|Best" in new_h


def test_uppercase_sidecar_is_reused(tmp_path):
    image = tmp_path/'photo.CR2'
    writer = XmpMetadataWriter()
    writer.export({image: {'Первый'}})
    lower = image.with_suffix('.xmp')
    upper = image.with_suffix('.XMP')
    lower.rename(upper)
    assert writer.export({image: {'Второй'}}).written_files == 1
    assert set(_values(upper, f'{{{DC_NS}}}subject')) == {'Первый', 'Второй'}
    assert len(list(tmp_path.iterdir())) == 1


def test_shared_sidecar_is_not_arbitrarily_changed(tmp_path):
    raw = tmp_path/'photo.CR2'
    jpeg = tmp_path/'photo.NEF'
    writer = XmpMetadataWriter()
    writer.export({raw: {'До'}}, {raw})
    before = raw.with_suffix('.xmp').read_bytes()
    result = writer.export({raw: {'Один'}, jpeg: {'Другой'}}, {raw}, [raw, jpeg])
    assert result.failed_files == 1
    assert raw.with_suffix('.xmp').read_bytes() == before


def test_preserve_failed_photo_metadata(tmp_path):
    image = tmp_path/'photo.CR2'
    writer = XmpMetadataWriter()
    writer.export({image: {'До'}}, {image})
    before = image.with_suffix('.xmp').read_bytes()
    result = writer.export({}, all_photos=[image], preserve_photos={image})
    assert result.unchanged_files == 1
    assert image.with_suffix('.xmp').read_bytes() == before


def test_separate_descriptions_keep_existing_keywords(tmp_path):
    image = tmp_path/'photo.CR2'
    xmp = image.with_suffix('.xmp')
    xmp.write_text(f'''<rdf:RDF xmlns:rdf="{RDF_NS}" xmlns:dc="{DC_NS}" xmlns:lr="{LR_NS}">
    <rdf:Description other="keep"/>
    <rdf:Description><dc:subject><rdf:Bag><rdf:li>Старое</rdf:li></rdf:Bag></dc:subject></rdf:Description>
    <rdf:Description><lr:hierarchicalSubject><rdf:Bag><rdf:li>ChildFaceFinder|Best</rdf:li></rdf:Bag></lr:hierarchicalSubject></rdf:Description>
    </rdf:RDF>''', encoding='utf-8')
    result = XmpMetadataWriter().export({image: {'Новое'}})
    assert result.written_files == 1
    root = ET.parse(xmp).getroot()
    assert len(root.findall(f'.//{{{DC_NS}}}subject')) == 1
    assert _values(xmp, f'{{{DC_NS}}}subject') == ['Старое', 'Новое']
    assert 'ChildFaceFinder|Best' not in _values(xmp, f'{{{LR_NS}}}hierarchicalSubject')
    assert root[0].get('other') == 'keep'


def test_valid_xml_without_rdf_is_not_overwritten(tmp_path):
    image = tmp_path/'photo.CR2'
    xmp = image.with_suffix('.xmp')
    before = b'<unrelated>keep</unrelated>'
    xmp.write_bytes(before)
    assert XmpMetadataWriter().export({image: {'Имя'}}).failed_files == 1
    assert xmp.read_bytes() == before


def test_external_xmp_edit_is_preserved(tmp_path, monkeypatch):
    image = tmp_path/'photo.CR2'
    writer = XmpMetadataWriter()
    original = writer._write_atomic
    def concurrent(tree, path, **kwargs):
        path.write_bytes(b'external edit')
        original(tree, path, **kwargs)
    monkeypatch.setattr(writer, '_write_atomic', concurrent)
    assert writer.export({image: {'Имя'}}).failed_files == 1
    assert image.with_suffix('.xmp').read_bytes() == b'external edit'


def test_export_cancellation_propagates(tmp_path):
    import pytest
    class Stop(Exception):
        pass
    def cancel():
        raise Stop()
    with pytest.raises(Stop):
        XmpMetadataWriter().export({tmp_path/'photo.CR2': {'Имя'}}, check_cancel=cancel)
    assert not list(tmp_path.glob('*.xmp'))
