"""Small filesystem primitives shared by runtime writers and installers."""
from contextlib import contextmanager, ExitStack
from functools import wraps
import os
import math
from pathlib import Path
import tempfile

_UNCHECKED = object()


def check_output_directory(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.cff-write-check-', dir=folder)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(b'ChildFaceFinder output check')
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        Path(name).unlink()


def file_snapshot(path):
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return None


@contextmanager
def atomic_path(target, *, expected=_UNCHECKED, cancel=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent)
    os.close(handle)
    temp = Path(name)
    try:
        yield temp
        with temp.open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if cancel:
            cancel()
        if expected is not _UNCHECKED and file_snapshot(target) != expected:
            raise RuntimeError(f'Файл изменён другим процессом; запись пропущена: {target}')
        os.replace(temp, target)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def exclusive_lock(path):
    """OS-owned advisory lock, released even after process termination.

    Keep the small lock file: unlinking it creates an inode race on POSIX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f'Уже выполняется другая операция: {path.parent}') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f'Уже выполняется другая операция: {path.parent}') from exc
        locked = True
        yield
    finally:
        try:
            if locked:
                if os.name == 'nt':
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def guarded_run(function):
    @wraps(function)
    def wrapped(self, references_dir, photos_dir, output_csv):
        photos_dir = Path(photos_dir).expanduser().resolve()
        references_dir = Path(references_dir).expanduser().resolve() if references_dir is not None else None
        workflow = getattr(self.settings, 'workflow', None)
        if workflow is not None and workflow.output != 'csv':
            output_csv = None
        else:
            output_csv = Path(output_csv).expanduser().resolve() if output_csv is not None else None
        if not photos_dir.is_dir():
            raise ValueError('Папка фотографий не существует')
        if references_dir is not None and not references_dir.is_dir():
            raise ValueError('Папка эталонов не существует')
        if workflow is not None:
            workflow.validate(references_dir, photos_dir, output_csv)
        elif output_csv is None or output_csv.suffix.casefold() != '.csv':
            raise ValueError('Файл результата должен иметь расширение .csv')
        if self.cancel_event.is_set():
            # Use the application's cancellation exception without circular imports.
            self._check_cancel()
        for name, low, high in [('match_threshold', .01, 1), ('ambiguity_margin', 0, 1),
                                ('detector_threshold', .01, 1)]:
            value = getattr(self.settings, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Некорректная настройка {name}: {value}')
        for name in ('decode_workers', 'inference_workers'):
            value = getattr(self.settings, name, 1)
            if not isinstance(value, int) or not 1 <= value <= 32:
                raise ValueError(f'Некорректное число потоков {name}: {value}')
        outputs = [output_csv] + [output_csv.with_name(output_csv.stem + suffix + '.csv')
                                 for suffix in ('_rejected', '_review', '_best')] if output_csv is not None else []
        with ExitStack() as stack:
            stack.enter_context(exclusive_lock(photos_dir / '.ChildFaceFinder.lock'))
            if workflow is not None and workflow.output == 'folders':
                check_output_directory(workflow.destination)
                stack.enter_context(exclusive_lock(Path(workflow.destination) / '.ChildFaceFinder.lock'))
            for folder in {p.parent for p in outputs}:
                check_output_directory(folder)
            for path in sorted(outputs, key=str):
                stack.enter_context(exclusive_lock(path.with_name('.' + path.name + '.cff.lock')))
            return function(self, references_dir, photos_dir, output_csv)
    return wrapped
