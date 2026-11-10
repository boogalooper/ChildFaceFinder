"""Exercise the actual run method without loading GPU/model dependencies."""
import ast
import csv
import os
import threading
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

import numpy as np
from identity_catalog import assign_identities
from metadata_writer import XmpMetadataWriter, DC_NS, RDF_NS
import xml.etree.ElementTree as ET


def test_both_metadata_modes_include_quality_rejected_faces(tmp_path):
    tree = ast.parse(Path(__file__).with_name('engine.py').read_text())
    methods = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            methods.extend(n for n in cls.body if isinstance(n, ast.FunctionDef)
                           and n.name in ('run', '_match_face', '_write_csv_atomic'))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             ast.ClassDef(name='Runner', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ns = dict(np=np, Path=Path, csv=csv, os=os, defaultdict=defaultdict,
              cleanup_stale_temp_dirs=lambda: ([], []), make_system_temp_dir=TemporaryDirectory,
              iter_candidate_files=lambda *args: [tmp_path/'photo.cr2'],
              ReferenceSet=NS, RunSummary=NS, CancelledError=RuntimeError,
              assign_identities=assign_identities, XmpMetadataWriter=XmpMetadataWriter)
    exec(compile(ast.fix_missing_locations(module), '<engine run>', 'exec'), ns)
    for with_refs, expected in [(False, 'Лицо_0001'), (True, 'Котов Никита')]:
        sidecar = tmp_path/'photo.xmp'
        sidecar.unlink(missing_ok=True)
        runner = ns['Runner']()
        runner.cancel_event = threading.Event()
        runner.engine = NS(initialize=lambda: None)
        runner.settings = NS(write_xmp_tags=True, recursive=False, verbose_diagnostics=False,
                             match_threshold=.4, ambiguity_margin=.04, detector_threshold=.4,
                             select_best_series=False, xmp_mark_best=False)
        runner.log = lambda text: None
        runner.progress = lambda *args: None
        face = NS(embedding=np.array([1., 0.], dtype=np.float32), bbox=(0, 0, 10, 10),
                  quality_reasons=['закрытые глаза'], source='full', det_score=.9)
        runner._pipeline = lambda *args, **kwargs: [NS(path=tmp_path/'photo.cr2', faces=[face], error=None)]
        runner._build_references = lambda *args: NS(person_ids=['Котов Никита'], matrix=np.array([[1., 0.]]),
                                                   ref_person_indices=np.array([0]), file_count=1)
        result = runner.run(tmp_path if with_refs else None, tmp_path, tmp_path/'result.csv')
        assert result.matched_pairs == 0
        assert result.rejected_pairs == 1
        values = ET.parse(sidecar).findall(f'.//{{{DC_NS}}}subject/{{{RDF_NS}}}Bag/{{{RDF_NS}}}li')
        assert [v.text for v in values] == [expected]
