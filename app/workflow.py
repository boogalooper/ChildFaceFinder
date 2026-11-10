"""User-facing workflow choices, independent of detection settings."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkflowOptions:
    scenario: str = 'references'
    output: str = 'csv'
    destination: Path | None = None
    transfer: str = 'copy'
    xmp_enabled: bool = False
    write_people: bool = True
    best_frames: bool = False
    best_label: str = 'Select'
    filter_quality: bool = False
    detailed_csv: bool = False

    @property
    def people_enabled(self):
        return self.xmp_enabled and self.write_people

    @property
    def labels_enabled(self):
        return self.xmp_enabled and self.best_frames

    def validate(self, references, photos, csv_path):
        if self.scenario not in ('references', 'discover') or self.output not in ('csv', 'folders', 'analysis'):
            raise ValueError('Выберите сценарий и результат обработки')
        if self.scenario == 'references' and references is None:
            raise ValueError('Для поиска по эталонам укажите папку эталонных фотографий')
        if self.scenario == 'discover' and references is not None:
            raise ValueError('В режиме поиска уникальных лиц папка эталонов не используется')
        if references is not None and Path(references).resolve() == Path(photos).resolve():
            raise ValueError('Папки эталонов и съёмки должны различаться')
        if self.xmp_enabled and not (self.write_people or self.best_frames):
            raise ValueError('Выберите действие с XMP: люди и/или цветовые метки лучших кадров')
        if self.output == 'analysis' and not (self.people_enabled or self.labels_enabled):
            raise ValueError('«Только анализ» требует включить запись людей или цветовые метки в XMP')
        if self.labels_enabled and not self.best_label.strip():
            raise ValueError('Укажите название цветовой метки, совпадающее с настройкой Bridge')
        if self.output == 'csv' and (csv_path is None or Path(csv_path).suffix.casefold() != '.csv'):
            raise ValueError('Укажите файл результата .csv')
        if self.output == 'folders':
            if self.transfer not in ('copy', 'move') or self.destination is None:
                raise ValueError('Укажите папку результата и способ: копировать или переместить')
            dest = Path(self.destination).expanduser().resolve()
            roots = [Path(photos).resolve()] + ([Path(references).resolve()] if references is not None else [])
            for root in roots:
                if dest == root or dest in root.parents or root in dest.parents:
                    raise ValueError('Папка результата должна находиться отдельно от папок съёмки и эталонов')
