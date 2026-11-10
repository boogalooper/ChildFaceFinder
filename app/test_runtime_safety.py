from types import SimpleNamespace as NS
import threading
import pytest
import numpy as np
from runtime_safety import atomic_path, exclusive_lock


def test_atomic_failure_and_concurrent_change_preserve_original(tmp_path):
    path = tmp_path/'result.csv'
    path.write_bytes(b'old')
    with pytest.raises(RuntimeError):
        with atomic_path(path) as temp:
            temp.write_bytes(b'partial')
            raise RuntimeError('disk failure')
    assert path.read_bytes() == b'old'
    with pytest.raises(RuntimeError, match='другим процессом'):
        with atomic_path(path, expected=b'old') as temp:
            temp.write_bytes(b'new')
            path.write_bytes(b'external')
    assert path.read_bytes() == b'external'
    assert not list(tmp_path.glob('*.tmp'))


def test_os_lock_releases_after_error(tmp_path):
    lock = tmp_path/'test.lock'
    with pytest.raises(ValueError):
        with exclusive_lock(lock):
            with pytest.raises(RuntimeError):
                with exclusive_lock(lock):
                    pass
            raise ValueError('failed')
    with exclusive_lock(lock):
        pass


def test_pipeline_handles_empty_decodes_and_worker_errors(engine_module, monkeypatch, tmp_path):
    e = engine_module
    runner = e.ChildFaceFinder(e.Settings(decode_workers=3, inference_workers=2))
    paths = [tmp_path/f'{i}.jpg' for i in range(25)]
    def decode(path, _temp):
        number = int(path.stem)
        if number % 4 == 0:
            return None
        if number % 4 == 1:
            raise SystemExit('unexpected worker exit')
        if number % 4 == 2:
            raise ValueError('broken image')
        return np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(e, 'decode_image', decode)
    runner.engine.analyze = lambda image, path: e.ImageAnalysis(path, [])
    results = list(runner._pipeline(paths, tmp_path))
    assert len(results) == len(paths)
    assert len({r.path for r in results}) == len(paths)
    assert sum(r.error is None for r in results) == 6
    assert not any(t.name.startswith(('cff-decode', 'cff-infer')) for t in threading.enumerate())


def test_pipeline_generator_close_and_cancel_join_workers(engine_module, monkeypatch, tmp_path):
    e = engine_module
    runner = e.ChildFaceFinder(e.Settings(decode_workers=3, inference_workers=2))
    monkeypatch.setattr(e, 'decode_image', lambda *args: np.zeros((2, 2, 3), dtype=np.uint8))
    runner.engine.analyze = lambda image, path: e.ImageAnalysis(path, [])
    stream = runner._pipeline([tmp_path/f'{i}.jpg' for i in range(100)], tmp_path)
    next(stream)
    stream.close()
    assert not any(t.name.startswith(('cff-decode', 'cff-infer')) for t in threading.enumerate())
    def analyze(image, path):
        runner.cancel()
        return e.ImageAnalysis(path, [])
    runner.engine.analyze = analyze
    with pytest.raises(e.CancelledError):
        list(runner._pipeline([tmp_path/f'{i}.jpg' for i in range(100)], tmp_path))
    assert not any(t.name.startswith(('cff-decode', 'cff-infer')) for t in threading.enumerate())


def test_cancel_before_run_is_not_reset(engine_module, tmp_path):
    e = engine_module
    runner = e.ChildFaceFinder(e.Settings(write_xmp_tags=True))
    runner.engine.initialize = lambda: pytest.fail('should not initialize')
    runner.cancel()
    with pytest.raises(e.CancelledError):
        runner.run(None, tmp_path, tmp_path/'out.csv')
    assert not (tmp_path/'out.csv').exists()


def test_total_decode_failure_preserves_previous_results(engine_module, monkeypatch, tmp_path):
    e = engine_module
    runner = e.ChildFaceFinder(e.Settings(write_xmp_tags=True))
    runner.engine.initialize = lambda: None
    monkeypatch.setattr(e, 'iter_candidate_files', lambda *args: [tmp_path/'broken.cr2'])
    monkeypatch.setattr(e, 'cleanup_stale_temp_dirs', lambda: ([], []))
    runner._pipeline = lambda *args, **kwargs: [e.ImageAnalysis(tmp_path/'broken.cr2', [], error='bad RAW')]
    output = tmp_path/'out.csv'
    output.write_text('previous', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Ни одну'):
        runner.run(None, tmp_path, output)
    assert output.read_text() == 'previous'
    assert not (tmp_path/'ChildFaceFinder_people.json').exists()


def test_same_folder_cannot_run_twice(engine_module, tmp_path):
    e = engine_module
    runner = e.ChildFaceFinder(e.Settings(write_xmp_tags=True))
    runner.engine.initialize = lambda: pytest.fail('should not initialize')
    with exclusive_lock(tmp_path/'.ChildFaceFinder.lock'):
        with pytest.raises(RuntimeError, match='другая операция'):
            runner.run(None, tmp_path, tmp_path/'out.csv')


def test_zero_margin_does_not_accept_equal_reference_matches(engine_module):
    e = engine_module
    runner = e.ChildFaceFinder(e.Settings(ambiguity_margin=0))
    refs = e.ReferenceSet(['А', 'Б'], np.array([[1., 0.], [1., 0.]]), np.array([0, 1]), 2)
    assert runner._match_face(NS(embedding=np.array([1., 0.])), refs)[0] is None
    empty = e.ReferenceSet([], np.empty((0, 2)), np.array([], dtype=int), 0)
    assert runner._match_face(NS(embedding=np.array([1., 0.])), empty)[0] is None


def test_full_reference_run_is_independent_of_pipeline_order(engine_module, monkeypatch, tmp_path):
    e = engine_module
    paths = [tmp_path/f'IMG_{i:04}.CR2' for i in range(1, 13)]
    for path in paths:
        path.write_bytes(b'fake image')
    monkeypatch.setattr(e, 'iter_candidate_files', lambda *args: paths)
    monkeypatch.setattr(e, 'cleanup_stale_temp_dirs', lambda: ([], []))
    analyses = [e.ImageAnalysis(p, [e.FaceRecord(
        embedding=np.array([1., 0.], dtype=np.float32), bbox=(0, 0, 100, 100),
        raw_bbox=(0, 0, 100, 100), det_score=.9, blur_score=100., eye_left=.2, eye_right=.2, pose=(0, 0, 0)
    )]) for p in paths]
    output = tmp_path/'result.csv'
    reports = [output, tmp_path/'result_best.csv', tmp_path/'result_rejected.csv', tmp_path/'result_review.csv']
    snapshots = []
    for ordered in (analyses, list(reversed(analyses))):
        runner = e.ChildFaceFinder(e.Settings(write_xmp_tags=True, select_best_series=True, xmp_mark_best=True))
        runner.engine.initialize = lambda: None
        runner._build_references = lambda *args: e.ReferenceSet(['Имя'], np.array([[1., 0.]], dtype=np.float32), np.array([0]), 1)
        runner._pipeline = lambda *args, **kwargs: ordered
        result = runner.run(tmp_path, tmp_path, output)
        assert result.identified_faces == 12 and result.found_people == 1
        assert result.best_series_count == 1
        snapshots.append([p.read_bytes() for p in reports + [p.with_suffix('.xmp') for p in paths]])
    assert snapshots[0] == snapshots[1]
