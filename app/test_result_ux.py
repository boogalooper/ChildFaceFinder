from types import SimpleNamespace
import pytest
from result_status import result_title
from runtime_safety import check_output_directory
from file_output import export_people


@pytest.mark.parametrize('changes,expected', [
    ({}, 'Готово'),
    ({'xmp_failed':4},'Результат не сохранён'),
    ({'xmp_failed':4,'xmp_written':1},'Выполнено частично'),
    ({'file_errors':1,'file_copies':1,'destination':'out'},'Выполнено частично'),
    ({'destination':'out'},'нет фотографий'),
])
def test_result_classification(changes,expected):
    values=dict(unreadable_photos=0,xmp_failed=0,file_errors=0,output_csv=None,
                file_copies=0,xmp_written=0,xmp_unchanged=0,destination=None,completed_photos=0)
    values.update(changes)
    assert expected in result_title(SimpleNamespace(**values))


def test_output_probe_leaves_no_files(tmp_path):
    folder=tmp_path/'output'
    check_output_directory(folder)
    assert list(folder.iterdir())==[]


def test_photo_and_copy_counts_and_blocked_reason(tmp_path):
    source=tmp_path/'source';source.mkdir()
    a,b=source/'a.jpg',source/'b.jpg'
    a.write_bytes(b'a');b.write_bytes(b'b')
    result=export_people({a:{'A','B'},b:{'B'}},source,tmp_path/'output',move=True,blocked={b:'XMP error'})
    assert result.completed_photos==1
    assert result.copies==2 and result.moved==1
    assert result.metadata_blocked==1 and result.transfer_errors==0
    assert b.exists() and not a.exists()
