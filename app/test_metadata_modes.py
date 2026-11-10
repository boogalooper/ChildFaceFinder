"""Exercise the actual run method without loading GPU/model dependencies."""
import csv
import threading
from types import SimpleNamespace as NS

import pytest
import numpy as np
from metadata_writer import DC_NS, RDF_NS
import xml.etree.ElementTree as ET


@pytest.mark.parametrize("scenario", ["normal", "unknown_only", "no_faces"])
def test_both_metadata_modes_include_quality_rejected_faces(tmp_path, scenario, engine_module, monkeypatch):
    engine = engine_module
    monkeypatch.setattr(engine, 'iter_candidate_files', lambda *args: [tmp_path/'photo.cr2'])
    monkeypatch.setattr(engine, 'cleanup_stale_temp_dirs', lambda: ([], []))
    for with_refs, expected in [(False, 'Лицо_0001'), (True, 'Котов Никита')]:
        sidecar = tmp_path/'photo.xmp'
        sidecar.unlink(missing_ok=True)
        runner = engine.ChildFaceFinder(engine.Settings())
        runner.cancel_event = threading.Event()
        runner.engine = NS(initialize=lambda: None)
        runner.settings = NS(write_xmp_tags=True, recursive=False, verbose_diagnostics=False,
                             match_threshold=.4, ambiguity_margin=.04, detector_threshold=.4,
                             min_face_px=45, blur_threshold=45,
                             select_best_series=False, xmp_mark_best=False)
        runner.log = lambda text: None
        runner.progress = lambda *args: None
        face = NS(embedding=np.array([1., 0.], dtype=np.float32), bbox=(0, 0, 100, 100),
                  quality_reasons=['закрытые глаза'], source='full', det_score=.9)
        unknown = NS(embedding=np.array([0., 1.], dtype=np.float32), bbox=(110, 0, 120, 10),
                     quality_reasons=['маленькое лицо'], source='full', det_score=.9)
        faces = [face, unknown] if scenario == 'normal' else ([unknown] if scenario == 'unknown_only' else [])
        runner._pipeline = lambda *args, **kwargs: [NS(path=tmp_path/'photo.cr2', faces=faces, error=None)]
        runner._build_references = lambda *args: NS(person_ids=['Котов Никита'], matrix=np.array([[1., 0.]]),
                                                   ref_person_indices=np.array([0]), file_count=1)
        result = runner.run(tmp_path if with_refs else None, tmp_path, tmp_path/'result.csv')
        known_count = int(scenario == 'normal')
        unknown_count = int(scenario != 'no_faces')
        assert result.detected_faces == len(faces)
        assert result.identified_faces == known_count
        assert result.unidentified_faces == unknown_count
        assert result.found_people == known_count
        assert result.review_photos == unknown_count
        report = list(csv.reader(result.review_csv.open(encoding='utf-8-sig'), delimiter=';'))
        assert len(report) == 1 + unknown_count
        assert all(len(row) == 10 for row in report)
        if unknown_count:
            assert report[1][-1] == 'не определено'
        assert result.matched_pairs == 0
        assert result.rejected_pairs == known_count
        if known_count:
            values = ET.parse(sidecar).findall(f'.//{{{DC_NS}}}subject/{{{RDF_NS}}}Bag/{{{RDF_NS}}}li')
            assert [v.text for v in values] == [expected]
        else:
            assert not sidecar.exists()
