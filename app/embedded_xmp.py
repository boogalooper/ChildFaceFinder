"""Embedded XMP without re-encoding pixels; writes staged copies only."""
import hashlib
import shutil
from pathlib import Path
from runtime_safety import atomic_path

# Camera RAW uses Adobe sidecars; DNG is a writable document format.
EMBEDDED_EXTENSIONS = {'.jpg', '.jpeg', '.jpe', '.tif', '.tiff', '.dng',
                       '.png', '.webp', '.psd', '.jp2', '.jpf', '.jpx'}
MAX_WRITE_SIZE = 1024 ** 3


def supports_embedded(path):
    return Path(path).suffix.lower() in EMBEDDED_EXTENSIONS


def backend():
    try:
        import pyexiv2
    except ImportError as exc:
        raise RuntimeError('Для встроенного XMP требуется pyexiv2: запустите install.bat') from exc
    return pyexiv2


def fingerprint(path, cancel=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            if cancel:
                cancel()
            data = stream.read(1024 * 1024)
            if not data:
                break
            digest.update(data)
    return digest.digest()


def read(path):
    # Python handles Unicode/long Windows paths; do not pass filenames to Exiv2.
    with backend().ImageData(path.read_bytes()) as image:
        return image.read_raw_xmp().encode('utf-8') or None


def write(path, packet, expected_digest, cancel=None):
    with atomic_path(path, cancel=cancel) as temp:
        shutil.copy2(path, temp)
        if fingerprint(temp, cancel) != expected_digest:
            raise RuntimeError('Фотография изменилась после чтения XMP')
        with backend().ImageData(temp.read_bytes()) as image:
            image.modify_raw_xmp(packet.decode('utf-8'))
            expected_tags = image.read_xmp()
            updated = image.get_bytes()
        temp.write_bytes(updated)
        del updated
        with backend().ImageData(temp.read_bytes()) as image:
            if image.read_xmp() != expected_tags:
                raise RuntimeError('Проверка встроенного XMP не пройдена')
        if fingerprint(path, cancel) != expected_digest:
            raise RuntimeError('Фотография изменена другим процессом; запись отменена')


def self_test():
    """Exercise the real metadata backend on a disposable Cyrillic filename."""
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='ChildFaceFinder_XMP_') as folder:
        path = Path(folder) / 'Проверка метаданных.jpg'
        Image.new('RGB', (8, 8), (20, 40, 60)).save(path)
        packet = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Label="Select"/></rdf:RDF></x:xmpmeta>'
        write(path, packet, fingerprint(path))
        if b'Select' not in (read(path) or b''):
            raise RuntimeError('Проверка записи XMP не пройдена')
