from stage_progress import track
from metadata_writer import XmpMetadataWriter


def test_stage_progress_counts_skipped_items():
    events=[]
    for item in track([1,2,3],lambda *v:events.append(v),'Stage'):
        if item==2:continue
    assert events[0][:2]==(0,3)
    assert events[-1][:2]==(3,3)


def test_xmp_progress_reaches_end_with_error(tmp_path):
    a,b=tmp_path/'a.CR2',tmp_path/'b.CR2'
    b.with_suffix('.xmp').write_text('bad XML')
    events=[]
    result=XmpMetadataWriter().export({a:{'A'},b:{'B'}},progress=lambda *v:events.append(v))
    assert result.written_files==1 and result.failed_files==1
    assert events[0][:2]==(0,2)
    assert events[-1][:2]==(2,2)
    assert events[-1][2].startswith('Запись XMP')
