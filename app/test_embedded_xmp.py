import pytest
from PIL import Image
import pyexiv2
from metadata_writer import XmpMetadataWriter
from file_output import export_people


@pytest.mark.parametrize('extension', ['jpg', 'tif', 'png', 'webp'])
def test_real_embedded_roundtrip_preserves_pixels_and_other_metadata(tmp_path, extension):
    path = tmp_path/('Иван.'+extension)
    Image.new('RGB', (32, 24), (12, 60, 110)).save(path)
    with pyexiv2.Image(str(path)) as image:
        image.modify_exif({'Exif.Image.Artist': 'Автор'})
        image.modify_xmp({'Xmp.xmp.Rating': '4', 'Xmp.crs.Exposure2012': '1.25', 'Xmp.dc.subject': ['Школа']})
    pixels = Image.open(path).tobytes()
    writer = XmpMetadataWriter()
    result = writer.export({path: {'Иван'}}, {path}, sync_people=True, sync_labels=True, legacy_best=False)
    assert result.written_files == 1 and not result.failed_files
    assert not path.with_suffix('.xmp').exists()
    assert Image.open(path).tobytes() == pixels
    with pyexiv2.Image(str(path)) as image:
        tags = image.read_xmp()
        assert image.read_exif()['Exif.Image.Artist'] == 'Автор'
        assert tags['Xmp.xmp.Rating'] == '4'
        assert tags['Xmp.crs.Exposure2012'] == '1.25'
        assert set(tags['Xmp.dc.subject']) == {'Школа', 'Иван'}
        assert tags['Xmp.xmp.Label'] == 'Select'
    writer.export({path: {'Анна'}}, sync_people=True, legacy_best=False)
    with pyexiv2.Image(str(path)) as image:
        assert set(image.read_xmp()['Xmp.dc.subject']) == {'Школа', 'Анна'}
        assert image.read_xmp()['Xmp.xmp.Label'] == 'Select'
    writer.export({}, all_photos=[path], write_people=False, sync_labels=True, legacy_best=False)
    with pyexiv2.Image(str(path)) as image:
        assert 'Xmp.xmp.Label' not in image.read_xmp()
        assert 'Анна' in image.read_xmp()['Xmp.dc.subject']


def test_corrupt_jpeg_does_not_fall_back_or_modify_source(tmp_path):
    path = tmp_path/'bad.jpg';path.write_bytes(b'corrupt')
    result = XmpMetadataWriter().export({path: {'Иван'}}, sync_people=True)
    assert result.failed_files == 1
    assert path.read_bytes() == b'corrupt'
    assert not path.with_suffix('.xmp').exists()


def test_raw_jpeg_metadata_independent_and_move_preserves_embedded(tmp_path):
    source = tmp_path/'source';source.mkdir()
    jpeg = source/'photo.jpg';Image.new('RGB',(10,10)).save(jpeg)
    raw = source/'photo.CR2';raw.write_bytes(b'raw')
    result = XmpMetadataWriter().export({raw: {'RAW'}, jpeg: {'JPEG'}}, sync_people=True, legacy_best=False)
    assert result.written_files == 2
    with pyexiv2.Image(str(jpeg)) as image:assert image.read_xmp()['Xmp.dc.subject']==['JPEG']
    # Without a shared sidecar, moving the updated JPEG keeps embedded data.
    solo = tmp_path/'solo';solo.mkdir();target=solo/'one.jpg';jpeg.rename(target)
    result=export_people({target:{'JPEG'}},solo,tmp_path/'output',move=True)
    assert result.moved==1
    with pyexiv2.Image(str(tmp_path/'output'/'JPEG'/'one.jpg')) as image:
        assert image.read_xmp()['Xmp.dc.subject']==['JPEG']


def test_unicode_paths_never_reach_native_filename_api(tmp_path, monkeypatch):
    import embedded_xmp
    path = tmp_path/'Кузнечики'/'Портрет Алёна.jpg'
    path.parent.mkdir()
    Image.new('RGB', (16, 16)).save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Native filename API must not be called')
    monkeypatch.setattr(pyexiv2, 'Image', forbidden)
    result = XmpMetadataWriter().export({path: {'Алёна'}}, sync_people=True, legacy_best=False)
    assert result.written_files == 1 and not result.failed_files
    assert 'Алёна' in embedded_xmp.read(path).decode('utf-8')
    embedded_xmp.self_test()
