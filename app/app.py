from __future__ import annotations

from workflow import WorkflowOptions

import os
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from collector import CollectorSummary, collect_from_csv
from engine import CancelledError, ChildFaceFinder, Settings
from image_loader import cleanup_stale_temp_dirs
from user_settings import load_user_settings, save_user_settings

APP_TITLE = "Child Face Finder — поиск людей на фотографиях"

QUALITY_PRESETS = {
    "Мягкий": dict(
        min_face_px=35,
        max_yaw=55.0,
        max_pitch=45.0,
        max_roll=45.0,
        eye_ratio_min=0.080,
        blur_threshold=25.0,
        max_clipped_fraction=0.70,
        head_top_margin_ratio=0.12,
        head_side_margin_ratio=0.05,
        chin_margin_ratio=0.02,
        edge_guard_base_ratio=0.012,
        edge_guard_min_ratio=0.006,
        edge_guard_max_ratio=0.035,
    ),
    "Нормальный": dict(
        min_face_px=45,
        max_yaw=45.0,
        max_pitch=35.0,
        max_roll=35.0,
        eye_ratio_min=0.105,
        blur_threshold=45.0,
        max_clipped_fraction=0.55,
        head_top_margin_ratio=0.18,
        head_side_margin_ratio=0.07,
        chin_margin_ratio=0.03,
        edge_guard_base_ratio=0.020,
        edge_guard_min_ratio=0.008,
        edge_guard_max_ratio=0.050,
    ),
    "Строгий": dict(
        min_face_px=65,
        max_yaw=30.0,
        max_pitch=25.0,
        max_roll=25.0,
        eye_ratio_min=0.125,
        blur_threshold=75.0,
        max_clipped_fraction=0.35,
        head_top_margin_ratio=0.24,
        head_side_margin_ratio=0.10,
        chin_margin_ratio=0.05,
        edge_guard_base_ratio=0.030,
        edge_guard_min_ratio=0.012,
        edge_guard_max_ratio=0.065,
    ),
}

SEARCH_MODES = {
    "Быстрый": dict(tile_enabled=False, tile_size=1400, tile_overlap=0.18, tile_trigger_px=1600, rescue_detector_threshold=0.12, rescue_max_candidates=40),
    "Улучшенный": dict(tile_enabled=True, tile_size=1400, tile_overlap=0.20, tile_trigger_px=1600, rescue_detector_threshold=0.10, rescue_max_candidates=80),
    "Максимальный": dict(tile_enabled=True, tile_size=1000, tile_overlap=0.24, tile_trigger_px=1200, rescue_detector_threshold=0.08, rescue_max_candidates=120),
}


def _saved_int(values: dict, key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(values.get(key, default))
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(low, min(high, value))


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 350) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.after_id: str | None = None
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self.after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _show(self) -> None:
        self.after_id = None
        if self.window is not None or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        top = tk.Toplevel(self.widget)
        top.wm_overrideredirect(True)
        top.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            top,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=430,
        )
        label.pack()
        self.window = top

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self.window is not None:
            try:
                self.window.destroy()
            finally:
                self.window = None


def add_tooltip(widget: tk.Widget, text: str) -> tk.Widget:
    ToolTip(widget, text)
    return widget


def hint(parent: tk.Widget, text: str) -> ttk.Label:
    label = ttk.Label(parent, text="(?)", cursor="question_arrow")
    add_tooltip(label, text)
    return label


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x850")
        self.minsize(900, 720)

        self.events: queue.Queue[tuple] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.finder: ChildFaceFinder | None = None
        self.collector_cancel_event: threading.Event | None = None
        self.closing_after_cancel = False

        saved = load_user_settings()
        self.ref_var = tk.StringVar(value=str(saved.get("reference_folder", "")))
        self.photos_var = tk.StringVar(value=str(saved.get("photo_folder", "")))
        self.output_var = tk.StringVar(value=str(saved.get("output_csv", "")))

        self.search_mode_var = tk.StringVar(value=str(saved.get("search_mode", "Улучшенный")))
        self.detector_threshold_var = tk.StringVar(value=str(saved.get("detector_threshold", "0.18")))
        self.quality_preset_var = tk.StringVar(value=str(saved.get("quality_preset", "Нормальный")))
        self.threshold_var = tk.StringVar(value=str(saved.get("match_threshold", "0.45")))
        self.margin_var = tk.StringVar(value=str(saved.get("ambiguity_margin", "0.04")))

        self.closed_eyes_var = tk.BooleanVar(value=bool(saved.get("reject_closed_eyes", True)))
        self.blur_var = tk.BooleanVar(value=bool(saved.get("reject_blur", True)))
        self.pose_var = tk.BooleanVar(value=bool(saved.get("reject_pose", True)))
        self.small_face_var = tk.BooleanVar(value=bool(saved.get("reject_small_face", True)))
        self.exposure_var = tk.BooleanVar(value=bool(saved.get("reject_exposure", False)))
        self.head_clip_var = tk.BooleanVar(value=bool(saved.get("reject_head_clipping", True)))

        self.decode_workers_var = tk.IntVar(value=_saved_int(saved, "decode_workers", max(2, min(8, os.cpu_count() or 4)), 1, 32))
        self.inference_workers_var = tk.IntVar(value=_saved_int(saved, "gpu_sessions", 2, 1, 8))
        self.recursive_var = tk.BooleanVar(value=bool(saved.get("recursive", True)))
        self.require_gpu_var = tk.BooleanVar(value=bool(saved.get("require_gpu", True)))
        self.verbose_var = tk.BooleanVar(value=bool(saved.get("verbose", True)))
        self.best_series_var = tk.BooleanVar(value=bool(saved.get("select_best_series", False)))
        self.write_xmp_var = tk.BooleanVar(value=bool(saved.get("write_xmp_tags", False)))
        self.xmp_mark_best_var = tk.BooleanVar(value=bool(saved.get("xmp_mark_best", False)))

        if self.search_mode_var.get() not in SEARCH_MODES:
            self.search_mode_var.set("Улучшенный")
        if self.quality_preset_var.get() not in QUALITY_PRESETS:
            self.quality_preset_var.set("Нормальный")

        self.scenario_var = tk.StringVar(value=saved.get('scenario', 'references'))
        self.output_mode_var = tk.StringVar(value=saved.get('output_mode', 'csv'))
        self.destination_var = tk.StringVar(value=saved.get('destination', ''))
        self.transfer_var = tk.StringVar(value=saved.get('transfer', 'copy'))
        self.xmp_enabled_var = tk.BooleanVar(value=bool(saved.get('xmp_enabled', saved.get('write_xmp_tags', False))))
        self.write_xmp_var.set(bool(saved.get('write_people', True)))
        self.best_label_var = tk.StringVar(value=saved.get('best_label', 'Select'))
        self.filter_quality_var = tk.BooleanVar(value=bool(saved.get('filter_quality', self.scenario_var.get() == 'references')))
        self.detailed_csv_var = tk.BooleanVar(value=bool(saved.get('detailed_csv', False)))
        for variable, allowed, default in ((self.scenario_var, ('references', 'discover'), 'references'),
                (self.output_mode_var, ('csv', 'folders', 'analysis'), 'csv'),
                (self.transfer_var, ('copy', 'move'), 'copy')):
            if variable.get() not in allowed:
                variable.set(default)
        self._starting = False
        self.status_var = tk.StringVar(value="Готово к запуску")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.collector_indeterminate = False

        self._build_ui()
        removed_temp, failed_temp = cleanup_stale_temp_dirs()
        if removed_temp:
            self._append_log(f"TEMP: удалено оставшихся каталогов предыдущих запусков: {len(removed_temp)}")
        for path, error in failed_temp:
            self._append_log(f"TEMP: не удалось удалить {path}: {error}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)

    def _build_ui(self) -> None:
        self.title('Child Face Finder — поиск людей на фотографиях')
        self.geometry('920x790')
        self.minsize(840, 720)
        self._main_controls = []
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill='both', expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(7, weight=1)
        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        ttk.Label(top, text='Люди на фотографиях', font=('Segoe UI', 15, 'bold')).pack(side='left')
        self.settings_btn = ttk.Button(top, text='Настройки…', command=self._open_settings)
        self.settings_btn.pack(side='right')
        self._main_controls.append(self.settings_btn)

        scenario = ttk.LabelFrame(outer, text='1. Что найти', padding=8)
        scenario.grid(row=1, column=0, sticky='ew', pady=4)
        for label, value in [('Людей по эталонным фото', 'references'), ('Уникальные лица в съёмке', 'discover')]:
            button = ttk.Radiobutton(scenario, text=label, variable=self.scenario_var, value=value,
                                    command=self._scenario_changed)
            button.pack(side='left', padx=(0, 24))
            self._main_controls.append(button)

        inputs = ttk.LabelFrame(outer, text='2. Исходные фотографии', padding=8)
        inputs.grid(row=2, column=0, sticky='ew', pady=4)
        inputs.columnconfigure(0, weight=1)
        def path_row(parent, label, variable, command):
            row = ttk.Frame(parent)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=label, width=17).grid(row=0, column=0, sticky='w')
            entry = ttk.Entry(row, textvariable=variable)
            entry.grid(row=0, column=1, sticky='ew', padx=6)
            button = ttk.Button(row, text='Обзор…', command=command, width=10)
            button.grid(row=0, column=2)
            self._main_controls.extend((entry, button))
            return row
        self.ref_row = path_row(inputs, 'Эталонные фото', self.ref_var, self._browse_refs)
        self.ref_row.grid(row=0, column=0, sticky='ew', pady=3)
        path_row(inputs, 'Фото съёмки', self.photos_var, self._browse_photos).grid(row=1, column=0, sticky='ew', pady=3)
        recursive = ttk.Checkbutton(inputs, text='Включая подпапки съёмки', variable=self.recursive_var)
        recursive.grid(row=2, column=0, sticky='w', pady=(4, 0))
        self._main_controls.append(recursive)
        self.scenario_hint = ttk.Label(inputs, wraplength=840, foreground='#555555')
        self.scenario_hint.grid(row=3, column=0, sticky='w', pady=(4, 0))

        output = ttk.LabelFrame(outer, text='3. Как сохранить результат', padding=8)
        output.grid(row=3, column=0, sticky='ew', pady=4)
        output.columnconfigure(0, weight=1)
        choices = ttk.Frame(output)
        choices.grid(row=0, column=0, sticky='w', pady=(0, 5))
        for label, value in [('Список CSV', 'csv'), ('По папкам людей', 'folders'), ('Только анализ и метки', 'analysis')]:
            button = ttk.Radiobutton(choices, text=label, variable=self.output_mode_var, value=value)
            button.pack(side='left', padx=(0, 24))
            self._main_controls.append(button)
        self.csv_row = path_row(output, 'Файл CSV', self.output_var, self._browse_output)
        self.csv_row.grid(row=1, column=0, sticky='ew')
        self.folder_rows = ttk.Frame(output)
        self.folder_rows.columnconfigure(0, weight=1)
        self.folder_rows.grid(row=2, column=0, sticky='ew')
        path_row(self.folder_rows, 'Папка результата', self.destination_var, self._browse_destination).grid(row=0, column=0, sticky='ew')
        transfer = ttk.Frame(self.folder_rows)
        transfer.grid(row=1, column=0, sticky='w', pady=(6, 0))
        for label, value in [('Копировать', 'copy'), ('Переместить', 'move')]:
            button = ttk.Radiobutton(transfer, text=label, variable=self.transfer_var, value=value)
            button.pack(side='left', padx=(0, 20))
            self._main_controls.append(button)
        self.transfer_hint = ttk.Label(self.folder_rows, wraplength=830, foreground='#555555')
        self.transfer_hint.grid(row=2, column=0, sticky='w', pady=(4, 0))

        metadata = ttk.LabelFrame(outer, text='4. Метаданные — необязательно', padding=8)
        metadata.grid(row=4, column=0, sticky='ew', pady=4)
        master = ttk.Checkbutton(metadata, text='Записывать метаданные XMP', variable=self.xmp_enabled_var)
        master.grid(row=0, column=0, sticky='w')
        self._main_controls.append(master)
        self.people_check = ttk.Checkbutton(metadata, text='Перезаписать список найденных людей', variable=self.write_xmp_var)
        self.people_check.grid(row=1, column=0, sticky='w', padx=20, pady=2)
        self.best_check = ttk.Checkbutton(metadata, text='Определить серии и отметить лучшие кадры цветом', variable=self.best_series_var)
        self.best_check.grid(row=2, column=0, sticky='w', padx=20, pady=2)
        self._main_controls.extend((self.people_check, self.best_check))
        self.metadata_hint = ttk.Label(metadata, wraplength=840, foreground='#555555')
        self.metadata_hint.grid(row=3, column=0, sticky='w', pady=(4, 0))

        actions = ttk.Frame(outer)
        actions.grid(row=5, column=0, sticky='ew', pady=(8, 3))
        self.start_btn = ttk.Button(actions, text='Начать обработку', command=self._start)
        self.start_btn.pack(side='left')
        self.cancel_btn = ttk.Button(actions, text='Отмена', command=self._cancel, state='disabled')
        self.cancel_btn.pack(side='left', padx=8)
        self.collect_btn = ttk.Button(self, command=self._collect_files)  # legacy entry point, not a main-screen action
        self.launch_hint = ttk.Label(actions, wraplength=590, foreground='#555555')
        self.launch_hint.pack(side='left', padx=8)
        self.progress = ttk.Progressbar(outer, variable=self.progress_var, maximum=100)
        self.progress.grid(row=6, column=0, sticky='ew', pady=(3, 5))
        log = ttk.LabelFrame(outer, text='Ход обработки', padding=5)
        log.grid(row=7, column=0, sticky='nsew')
        log.rowconfigure(0, weight=1)
        log.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log, height=6, wrap='word', state='disabled')
        self.log_text.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(log, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.log_text.configure(yscrollcommand=scrollbar.set)
        ttk.Label(outer, textvariable=self.status_var).grid(row=8, column=0, sticky='w', pady=(5, 0))
        for variable in (self.scenario_var, self.output_mode_var, self.transfer_var, self.xmp_enabled_var,
                         self.write_xmp_var, self.best_series_var, self.ref_var, self.photos_var,
                         self.output_var, self.destination_var, self.best_label_var, self.filter_quality_var):
            variable.trace_add('write', lambda *_: self._refresh_workflow())
        self._refresh_workflow()

    def _scenario_changed(self):
        self.filter_quality_var.set(self.scenario_var.get() == 'references')
        self._refresh_workflow()

    def _workflow(self):
        return WorkflowOptions(scenario=self.scenario_var.get(), output=self.output_mode_var.get(),
            destination=Path(self.destination_var.get()).expanduser().resolve() if self.destination_var.get().strip() else None,
            transfer=self.transfer_var.get(), xmp_enabled=self.xmp_enabled_var.get(),
            write_people=self.write_xmp_var.get(), best_frames=self.best_series_var.get(),
            best_label=self.best_label_var.get(), filter_quality=self.filter_quality_var.get(),
            detailed_csv=self.detailed_csv_var.get())

    def _refresh_workflow(self):
        if not hasattr(self, 'start_btn'):
            return
        reference = self.scenario_var.get() == 'references'
        self.ref_row.grid() if reference else self.ref_row.grid_remove()
        self.scenario_hint.configure(text='Имена людей берутся из имён эталонных файлов. Эталоны ищутся во всех подпапках.' if reference
            else 'Эталоны не нужны. Найденные люди получат обозначения Лицо_0001, Лицо_0002…')
        mode = self.output_mode_var.get()
        self.csv_row.grid() if mode == 'csv' else self.csv_row.grid_remove()
        self.folder_rows.grid() if mode == 'folders' else self.folder_rows.grid_remove()
        moving = self.transfer_var.get() == 'move'
        self.transfer_hint.configure(text=('Фото и XMP удаляются из съёмки только после проверки всех копий. ' if moving else 'Исходники остаются на месте; XMP копируются вместе с фото. ')
            + 'Групповой кадр попадёт в папку каждого найденного человека.')
        enabled = self.xmp_enabled_var.get()
        active = enabled and (self.write_xmp_var.get() or self.best_series_var.get())
        if enabled and self.best_series_var.get():
            note = f'Метка лучшего кадра: «{self.best_label_var.get()}» (меняется в настройках). Прежние цветовые метки будут заменены.'
        elif enabled:
            note = 'Список людей заменяется. Другие ключевые слова, цветовые метки, рейтинг и настройки Camera Raw сохраняются.'
        else:
            note = 'Существующие XMP не изменяются. При раскладке по папкам они переносятся вместе с фото.'
        self.metadata_hint.configure(text=note)
        problem = ''
        if not self.photos_var.get().strip():
            problem = 'Укажите папку фотографий съёмки.'
        elif reference and not self.ref_var.get().strip():
            problem = 'Укажите папку эталонных фото.'
        elif mode == 'csv' and not self.output_var.get().strip():
            problem = 'Укажите файл CSV.'
        elif mode == 'folders' and not self.destination_var.get().strip():
            problem = 'Выберите отдельную папку результата.'
        elif enabled and not active:
            problem = 'Выберите, что записывать в XMP.'
        elif mode == 'analysis' and not active:
            problem = 'Для «Только анализ» включите людей и/или цветовые метки XMP.'
        elif enabled and self.best_series_var.get() and not self.best_label_var.get().strip():
            problem = 'Укажите название цветовой метки в настройках.'
        busy = bool(self.worker and self.worker.is_alive()) or getattr(self, '_starting', False)
        for widget in self._main_controls:
            widget.state(['disabled'] if busy else ['!disabled'])
        self.people_check.state(['!disabled'] if enabled and not busy else ['disabled'])
        self.best_check.state(['!disabled'] if enabled and not busy else ['disabled'])
        self.start_btn.state(['disabled'] if problem or busy else ['!disabled'])
        self.launch_hint.configure(text=problem or ('Перемещение: проверьте выбранную папку результата.' if mode == 'folders' and moving else ('Готово. Неудачные кадры исключаются из результата.' if self.filter_quality_var.get() else 'Готово. В результат попадут все уверенно распознанные кадры.')))

    def _browse_destination(self):
        value = filedialog.askdirectory(title='Папка результата — отдельно от исходников')
        if value:
            self.destination_var.set(value)

    def _open_settings(self):
        if self.worker and self.worker.is_alive():
            return
        names = ('search_mode_var', 'detector_threshold_var', 'quality_preset_var', 'threshold_var', 'margin_var',
                 'closed_eyes_var', 'blur_var', 'pose_var', 'small_face_var', 'exposure_var', 'head_clip_var',
                 'decode_workers_var', 'inference_workers_var', 'require_gpu_var', 'verbose_var',
                 'filter_quality_var', 'detailed_csv_var', 'best_label_var')
        before = {name: getattr(self, name).get() for name in names}
        dialog = tk.Toplevel(self)
        dialog.withdraw()
        dialog.title('Настройки обработки')
        dialog.geometry('760x510')
        dialog.resizable(False, False)
        dialog.transient(self)
        tabs = ttk.Notebook(dialog)
        tabs.pack(fill='both', expand=True, padx=12, pady=12)
        pages = {}
        for name in ('Распознавание', 'Отбор кадров', 'Производительность', 'Вывод и XMP'):
            page = ttk.Frame(tabs, padding=14)
            page.columnconfigure(1, weight=1)
            tabs.add(page, text=name)
            pages[name] = page
        def field(page, row, label, variable, values=None):
            ttk.Label(page, text=label).grid(row=row, column=0, sticky='w', padx=(0, 15), pady=7)
            widget = ttk.Combobox(page, textvariable=variable, values=values, state='readonly', width=24) if values else ttk.Entry(page, textvariable=variable, width=26)
            widget.grid(row=row, column=1, sticky='w', pady=7)
        page = pages['Распознавание']
        field(page, 0, 'Поиск небольших лиц', self.search_mode_var, list(SEARCH_MODES))
        field(page, 1, 'Порог обнаружения (0.05–0.80)', self.detector_threshold_var)
        field(page, 2, 'Порог совпадения (0.05–1.00)', self.threshold_var)
        field(page, 3, 'Отрыв от другого человека (0–0.50)', self.margin_var)
        ttk.Label(page, text='Меньше порог совпадения — строже распознавание.\nСомнительные лица не получают чужое имя. Рекомендуемые значения подходят для первого запуска.', wraplength=670).grid(row=4, column=0, columnspan=2, sticky='w', pady=14)
        page = pages['Отбор кадров']
        field(page, 0, 'Строгость оценки качества', self.quality_preset_var, list(QUALITY_PRESETS))
        ttk.Checkbutton(page, text='Исключать неудачные кадры из CSV и папок людей', variable=self.filter_quality_var).grid(row=1, column=0, columnspan=2, sticky='w', pady=7)
        for row, (label, variable) in enumerate([
            ('Закрытые глаза', self.closed_eyes_var), ('Нерезкое лицо', self.blur_var),
            ('Сильный поворот головы', self.pose_var), ('Слишком маленькое лицо', self.small_face_var),
            ('Плохая экспозиция', self.exposure_var), ('Голова обрезана или близко к краю', self.head_clip_var)], start=2):
            ttk.Checkbutton(page, text=label, variable=variable).grid(row=row, column=0, columnspan=2, sticky='w', pady=2)
        ttk.Label(page, text='Этот отбор не влияет на список людей в XMP.\nПри включённом отборе неудачные кадры также не участвуют в выборе лучших.', wraplength=670).grid(row=8, column=0, columnspan=2, sticky='w', pady=12)
        page = pages['Производительность']
        field(page, 0, 'Потоки декодирования (1–32)', self.decode_workers_var)
        field(page, 1, 'Параллельные GPU-сеансы (1–8)', self.inference_workers_var)
        ttk.Checkbutton(page, text='Требовать NVIDIA CUDA', variable=self.require_gpu_var).grid(row=2, column=0, columnspan=2, sticky='w', pady=10)
        ttk.Checkbutton(page, text='Подробный журнал по каждому фото', variable=self.verbose_var).grid(row=3, column=0, columnspan=2, sticky='w', pady=10)
        ttk.Label(page, text='Для RTX 5090 начните с 2 GPU-сеансов. Увеличение числа потоков повышает расход памяти.', wraplength=660).grid(row=4, column=0, columnspan=2, sticky='w', pady=15)
        page = pages['Вывод и XMP']
        field(page, 0, 'Название метки лучшего кадра в Bridge', self.best_label_var)
        def label_from_xmp():
            path = filedialog.askopenfilename(parent=dialog, title='XMP снимка с нужной цветовой меткой', filetypes=[('XMP', '*.xmp *.XMP')])
            if not path:
                return
            try:
                import xml.etree.ElementTree as ET
                from metadata_writer import XMP_NS
                root = ET.parse(path).getroot()
                values = [node.attrib.get(f'{{{XMP_NS}}}Label') for node in root.iter()]
                values += [node.text for node in root.iter(f'{{{XMP_NS}}}Label')]
                label = next((v for v in values if v and v.strip()), None)
                if not label:
                    raise ValueError('В выбранном XMP нет цветовой метки')
                self.best_label_var.set(label)
            except Exception as exc:
                messagebox.showerror('Цветовая метка', str(exc), parent=dialog)
        ttk.Button(page, text='Взять название из готового XMP…', command=label_from_xmp).grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Label(page, text='Bridge определяет цвет по названию метки в своих установках.\nSelect — лишь исходное значение; можно взять точное название из вашего XMP.\n\nЦвет относится ко всему файлу: групповой кадр отмечается, если он лучший хотя бы для одного человека.', wraplength=670).grid(row=2, column=0, columnspan=2, sticky='w', pady=12)
        ttk.Checkbutton(page, text='Дополнительные CSV: брак, сомнения и лучшие кадры', variable=self.detailed_csv_var).grid(row=3, column=0, columnspan=2, sticky='w', pady=10)
        ttk.Label(page, text='Дополнительные отчёты создаются только при результате «Список CSV».\nПри копировании/перемещении создаётся журнал операций JSON.', wraplength=670).grid(row=4, column=0, columnspan=2, sticky='w')
        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill='x')
        def close(apply):
            if apply:
                try:
                    self._make_settings()
                except Exception as exc:
                    messagebox.showerror('Настройки', str(exc), parent=dialog)
                    return
                self._save_settings()
            else:
                for name, value in before.items():
                    getattr(self, name).set(value)
            dialog.grab_release()
            dialog.destroy()
            self._refresh_workflow()
        ttk.Button(buttons, text='Рекомендуемые значения', command=self._reset_recommended).pack(side='left')
        ttk.Button(buttons, text='Применить', command=lambda: close(True)).pack(side='right')
        ttk.Button(buttons, text='Отмена', command=lambda: close(False)).pack(side='right', padx=8)
        dialog.protocol('WM_DELETE_WINDOW', lambda: close(False))
        dialog.bind('<Escape>', lambda event: close(False))
        self.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - 760) // 2
        y = self.winfo_rooty() + (self.winfo_height() - 510) // 2
        dialog.geometry(f'760x510+{x}+{y}')
        dialog.deiconify()
        dialog.grab_set()

    def _browse_refs(self) -> None:
        value = filedialog.askdirectory(title="Папка с эталонами")
        if value:
            self.ref_var.set(value)

    def _browse_photos(self) -> None:
        value = filedialog.askdirectory(title="Папка с фотографиями")
        if value:
            self.photos_var.set(value)
            if not self.output_var.get().strip():
                self.output_var.set(str(Path(value) / "result.csv"))
            if not self.destination_var.get().strip():
                folder = Path(value)
                self.destination_var.set(str(folder.parent / (folder.name + '_по_людям')))

    def _browse_output(self) -> None:
        value = filedialog.asksaveasfilename(
            title="CSV результата",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="result.csv",
        )
        if value:
            self.output_var.set(value)

    def _collect_files(self) -> None:
        if self.worker and self.worker.is_alive():
            return

        initial_csv_dir = ""
        initial_csv_file = ""
        current_output = self.output_var.get().strip()
        if current_output:
            candidate = Path(current_output).expanduser()
            initial_csv_dir = str(candidate.parent)
            initial_csv_file = candidate.name
        csv_value = filedialog.askopenfilename(
            title="CSV с найденными целями",
            filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")],
            initialdir=initial_csv_dir or None,
            initialfile=initial_csv_file or None,
        )
        if not csv_value:
            return

        source_value = self.photos_var.get().strip()
        if not source_value:
            source_value = filedialog.askdirectory(
                title="Корневая папка исходных фотографий",
            )
            if not source_value:
                return
        else:
            source_candidate = Path(source_value).expanduser()
            if not source_candidate.is_dir():
                messagebox.showerror(
                    APP_TITLE,
                    "Папка фотографий, указанная в основном интерфейсе, больше не существует.\n"
                    "Исправьте поле «Папка фотографий» или очистите его, чтобы выбрать папку при сборке.",
                    parent=self,
                )
                return

        destination_value = filedialog.askdirectory(
            title="Каталог, где создать подпапки целей",
            initialdir=str(Path(csv_value).parent),
        )
        if not destination_value:
            return

        csv_path = Path(csv_value)
        source_root = Path(source_value)
        destination_root = Path(destination_value)
        self.start_btn.configure(state="disabled")
        self.collect_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.collector_cancel_event = threading.Event()
        cancel_event = self.collector_cancel_event
        self.status_var.set("Сбор файлов…")
        self.progress_var.set(0.0)
        self._append_log(f"Сборщик: CSV={csv_path}")
        self._append_log(f"Сборщик: исходники={source_root}")
        self._append_log(f"Сборщик: назначение={destination_root}")

        def collector_progress(done: int, total: int, name: str) -> None:
            self.events.put(("collector_progress", done, total, name))

        def task() -> None:
            try:
                summary = collect_from_csv(
                    csv_path,
                    source_root,
                    destination_root,
                    progress=collector_progress,
                    cancel_check=cancel_event.is_set,
                )
                self.events.put(("collector_done", summary))
            except Exception as exc:
                self.events.put(("collector_error", str(exc), traceback.format_exc()))

        self.worker = threading.Thread(target=task, name="cff-collector", daemon=True)
        self.worker.start()

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _reset_recommended(self) -> None:
        self.search_mode_var.set("Улучшенный")
        self.detector_threshold_var.set("0.18")
        self.quality_preset_var.set("Нормальный")
        self.threshold_var.set("0.45")
        self.margin_var.set("0.04")
        self.closed_eyes_var.set(True)
        self.blur_var.set(True)
        self.pose_var.set(True)
        self.small_face_var.set(True)
        self.exposure_var.set(False)
        self.head_clip_var.set(True)
        self.decode_workers_var.set(max(2, min(8, os.cpu_count() or 4)))
        self.inference_workers_var.set(2)
        self.require_gpu_var.set(True)
        self.verbose_var.set(True)
        self.filter_quality_var.set(self.scenario_var.get() == 'references')
        self.detailed_csv_var.set(False)
        self.best_label_var.set('Select')
        self.status_var.set("Восстановлены рекомендуемые значения")

    def _settings_snapshot(self) -> dict[str, object]:
        return {
            'scenario': self.scenario_var.get(),
            'output_mode': self.output_mode_var.get(),
            'destination': self.destination_var.get().strip(),
            'transfer': self.transfer_var.get(),
            'xmp_enabled': self.xmp_enabled_var.get(),
            'write_people': self.write_xmp_var.get(),
            'best_label': self.best_label_var.get(),
            'filter_quality': self.filter_quality_var.get(),
            'detailed_csv': self.detailed_csv_var.get(),
            "reference_folder": self.ref_var.get().strip(),
            "photo_folder": self.photos_var.get().strip(),
            "output_csv": self.output_var.get().strip(),
            "search_mode": self.search_mode_var.get(),
            "detector_threshold": self.detector_threshold_var.get(),
            "quality_preset": self.quality_preset_var.get(),
            "match_threshold": self.threshold_var.get(),
            "ambiguity_margin": self.margin_var.get(),
            "reject_closed_eyes": self.closed_eyes_var.get(),
            "reject_blur": self.blur_var.get(),
            "reject_pose": self.pose_var.get(),
            "reject_small_face": self.small_face_var.get(),
            "reject_exposure": self.exposure_var.get(),
            "reject_head_clipping": self.head_clip_var.get(),
            "decode_workers": int(self.decode_workers_var.get()),
            "gpu_sessions": int(self.inference_workers_var.get()),
            "recursive": self.recursive_var.get(),
            "require_gpu": self.require_gpu_var.get(),
            "verbose": self.verbose_var.get(),
            "select_best_series": self.best_series_var.get(),
            "write_xmp_tags": self.write_xmp_var.get(),
            "xmp_mark_best": self.xmp_mark_best_var.get(),
        }

    def _save_settings(self) -> None:
        try:
            save_user_settings(self._settings_snapshot())
        except (OSError, ValueError, tk.TclError) as exc:
            self._append_log(f"Настройки: не удалось сохранить: {exc}")

    @staticmethod
    def _parse_float(value: str, label: str, low: float, high: float) -> float:
        try:
            number = float(value.replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"{label}: требуется число") from exc
        if not low <= number <= high:
            raise ValueError(f"{label}: допустимо {low}…{high}")
        return number

    def _make_settings(self) -> Settings:
        quality = QUALITY_PRESETS[self.quality_preset_var.get()]
        search = SEARCH_MODES[self.search_mode_var.get()]
        return Settings(
            workflow=self._workflow(),
            match_threshold=self._parse_float(self.threshold_var.get(), "Порог совпадения", 0.05, 1.0),
            ambiguity_margin=self._parse_float(self.margin_var.get(), "Отрыв от второй цели", 0.0, 0.5),
            detector_threshold=self._parse_float(self.detector_threshold_var.get(), "Порог детектора", 0.05, 0.8),
            rescue_detector_threshold=min(self._parse_float(self.detector_threshold_var.get(), "Порог детектора", 0.05, 0.8), search["rescue_detector_threshold"]),
            rescue_max_candidates=search["rescue_max_candidates"],
            min_face_px=quality["min_face_px"],
            max_yaw=quality["max_yaw"],
            max_pitch=quality["max_pitch"],
            max_roll=quality["max_roll"],
            eye_ratio_min=quality["eye_ratio_min"],
            blur_threshold=quality["blur_threshold"],
            max_clipped_fraction=quality["max_clipped_fraction"],
            reject_closed_eyes=self.closed_eyes_var.get(),
            reject_blur=self.blur_var.get(),
            reject_pose=self.pose_var.get(),
            reject_small_face=self.small_face_var.get(),
            reject_exposure=self.exposure_var.get(),
            reject_head_clipping=self.head_clip_var.get(),
            head_top_margin_ratio=quality["head_top_margin_ratio"],
            head_side_margin_ratio=quality["head_side_margin_ratio"],
            chin_margin_ratio=quality["chin_margin_ratio"],
            edge_guard_base_ratio=quality["edge_guard_base_ratio"],
            edge_guard_min_ratio=quality["edge_guard_min_ratio"],
            edge_guard_max_ratio=quality["edge_guard_max_ratio"],
            tile_enabled=search["tile_enabled"],
            tile_size=search["tile_size"],
            tile_overlap=search["tile_overlap"],
            tile_trigger_px=search["tile_trigger_px"],
            decode_workers=max(1, min(32, int(self.decode_workers_var.get()))),
            inference_workers=max(1, min(8, int(self.inference_workers_var.get()))),
            recursive=self.recursive_var.get(),
            require_gpu=self.require_gpu_var.get(),
            verbose_diagnostics=self.verbose_var.get(),
            select_best_series=self.best_series_var.get(),
            write_xmp_tags=self.write_xmp_var.get(),
            xmp_mark_best=self.xmp_mark_best_var.get(),
        )

    def _validate_paths(self):
        photos_text = self.photos_var.get().strip()
        if not photos_text:
            raise ValueError('Укажите папку фотографий')
        photos = Path(photos_text).expanduser().resolve()
        refs = None
        if self.scenario_var.get() == 'references':
            if not self.ref_var.get().strip():
                raise ValueError('Укажите папку эталонных фото')
            refs = Path(self.ref_var.get().strip()).expanduser().resolve()
            if not refs.is_dir():
                raise ValueError('Папка эталонов не существует')
        if not photos.is_dir():
            raise ValueError('Папка фотографий не существует')
        output = None
        if self.output_mode_var.get() == 'csv':
            if not self.output_var.get().strip():
                raise ValueError('Укажите файл CSV')
            output = Path(self.output_var.get().strip()).expanduser().resolve()
            if output.suffix.casefold() != '.csv':
                output = output.with_suffix('.csv')
                self.output_var.set(str(output))
        self._workflow().validate(refs, photos, output)
        return refs, photos, output

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            refs, photos, output = self._validate_paths()
            settings = self._make_settings()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=self)
            return

        self._save_settings()

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Инициализация…")
        self.start_btn.configure(state="disabled")
        self.collect_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        def log(text: str) -> None:
            self.events.put(("log", text))

        def progress(done: int, total: int, name: str) -> None:
            self.events.put(("progress", done, total, name))

        self.finder = ChildFaceFinder(settings, log=log, progress=progress)

        def task() -> None:
            try:
                summary = self.finder.run(refs, photos, output)
                self.events.put(("done", summary))
            except CancelledError as exc:
                self.events.put(("cancelled", str(exc)))
            except Exception as exc:
                self.events.put(("error", str(exc), traceback.format_exc()))

        self.worker = threading.Thread(target=task, name="cff-main", daemon=True)
        self.worker.start()
        self._refresh_workflow()

    def _cancel(self) -> None:
        if self.finder is not None:
            self.finder.cancel()
            self.cancel_btn.configure(state="disabled")
            self.status_var.set("Отмена…")
            self._append_log("Запрошена отмена. Завершается текущая операция…")
            return
        if self.collector_cancel_event is not None and self.worker and self.worker.is_alive():
            self.collector_cancel_event.set()
            self.cancel_btn.configure(state="disabled")
            self.status_var.set("Отмена сборки…")
            self._append_log("Сборщик: запрошена отмена. Текущее копирование будет прервано, неполный файл удалён.")

    def _finish_ui(self) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.start_btn.configure(state="normal")
        self.collect_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.finder = None
        self.worker = None
        self._refresh_workflow()
        if self.closing_after_cancel:
            self.destroy()

    def _poll_events(self) -> None:
        try:
            for _ in range(200):
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._append_log(event[1])
                elif kind == "progress":
                    _, done, total, name = event
                    if total <= 0:
                        if str(self.progress.cget('mode')) != 'indeterminate':
                            self.progress.configure(mode='indeterminate')
                            self.progress.start(12)
                        self.status_var.set(name)
                    else:
                        self.progress.stop()
                        self.progress.configure(mode='determinate')
                        self.progress_var.set(done * 100.0 / total)
                        self.status_var.set(f"{done}/{total}: {name}")
                elif kind == "done":
                    summary = event[1]
                    self.progress_var.set(100)
                    self.status_var.set('Готово')
                    lines = [f'Найдено людей: {summary.found_people}',
                             f'Лиц определено: {summary.identified_faces}; не определено: {summary.unidentified_faces}',
                             f'Фотографий для проверки: {summary.review_photos}']
                    if summary.output_csv is not None:
                        lines += ['', f'Список CSV: {summary.output_csv}']
                    if summary.destination is not None:
                        lines += ['', f'Папка результата: {summary.destination}',
                                  f'Копий: {summary.file_copies}; исходников перемещено: {summary.file_moves}; ошибок: {summary.file_errors}',
                                  f'Журнал операций: {summary.transfer_journal}']
                    if summary.best_series_count:
                        lines.append(f'Лучших кадров для людей/серий: {summary.best_series_count}')
                    if summary.xmp_enabled:
                        lines.append(f'XMP: обновлено {summary.xmp_written}; без изменений {summary.xmp_unchanged}; ошибок {summary.xmp_failed}')
                    if summary.review_csv is not None:
                        lines.append(f'Отчёт проверки: {summary.review_csv}')
                    result = '\n'.join(lines)
                    self._append_log(result)
                    self._finish_ui()
                    if self.closing_after_cancel:
                        return
                    messagebox.showinfo('Обработка завершена', result, parent=self)
                elif kind == "collector_progress":
                    _, done, total, name = event
                    if total > 0:
                        if self.collector_indeterminate:
                            self.progress.stop()
                            self.progress.configure(mode="determinate")
                            self.collector_indeterminate = False
                        self.progress_var.set(done * 100.0 / total)
                        self.status_var.set(f"Сбор файлов: {done}/{total}: {name}")
                    else:
                        if not self.collector_indeterminate:
                            self.progress.configure(mode="indeterminate")
                            self.progress.start(12)
                            self.collector_indeterminate = True
                        self.status_var.set(f"Сбор файлов: {name}")
                elif kind == "collector_done":
                    summary: CollectorSummary = event[1]
                    if self.collector_indeterminate:
                        self.progress.stop()
                        self.progress.configure(mode="determinate")
                        self.collector_indeterminate = False
                    self.start_btn.configure(state="normal")
                    self.collect_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.worker = None
                    self.collector_cancel_event = None
                    if summary.cancelled:
                        self.status_var.set("Сбор файлов отменён")
                        self._append_log(
                            f"Сборщик отменён. Успело скопироваться={summary.copied}; пропущено={summary.skipped}; "
                            f"ошибок={summary.errors}; папок целей={summary.target_count}. "
                            f"Отчёт: {summary.report_csv}"
                        )
                        if self.closing_after_cancel:
                            self.destroy()
                            return
                        messagebox.showinfo(
                            APP_TITLE,
                            "Сбор файлов отменён.\n\n"
                            f"Успело скопироваться: {summary.copied}\n"
                            f"Пропущено: {summary.skipped}\n"
                            f"Ошибок: {summary.errors}\n\n"
                            "Недокопированный файл удалён. Уже скопированные файлы сохранены.\n\n"
                            f"Отчёт: {summary.report_csv}",
                            parent=self,
                        )
                    else:
                        self.progress_var.set(100.0)
                        self.status_var.set("Сбор файлов завершён")
                        self._append_log(
                            f"Сборщик: скопировано={summary.copied}; пропущено={summary.skipped}; "
                            f"ошибок={summary.errors}; папок целей={summary.target_count}."
                        )
                        if self.closing_after_cancel:
                            self.destroy()
                            return
                        messagebox.showinfo(
                            APP_TITLE,
                            "Сбор файлов завершён.\n\n"
                            f"Скопировано: {summary.copied}\n"
                            f"Пропущено: {summary.skipped}\n"
                            f"Ошибок: {summary.errors}\n\n"
                            f"Отчёт: {summary.report_csv}",
                            parent=self,
                        )
                elif kind == "collector_error":
                    _, text, details = event
                    if self.collector_indeterminate:
                        self.progress.stop()
                        self.progress.configure(mode="determinate")
                        self.collector_indeterminate = False
                    self.status_var.set("Ошибка сборщика")
                    self._append_log(details)
                    self.start_btn.configure(state="normal")
                    self.collect_btn.configure(state="normal")
                    self.cancel_btn.configure(state="disabled")
                    self.worker = None
                    self.collector_cancel_event = None
                    if self.closing_after_cancel:
                        self.destroy()
                        return
                    if not self.closing_after_cancel:
                        messagebox.showerror(APP_TITLE, text, parent=self)
                elif kind == "cancelled":
                    self.status_var.set("Отменено")
                    self._append_log(event[1])
                    self._finish_ui()
                    if self.closing_after_cancel:
                        return
                elif kind == "error":
                    _, text, details = event
                    self.status_var.set("Ошибка")
                    self._append_log(details)
                    self._finish_ui()
                    if self.closing_after_cancel:
                        return
                    if not self.closing_after_cancel:
                        messagebox.showerror(APP_TITLE, text, parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll_events)

    def _on_close(self) -> None:
        self._save_settings()
        if self.worker and self.worker.is_alive():
            self.closing_after_cancel = True
            self._cancel()
            self.status_var.set("Отмена перед закрытием…")
            return
        self.destroy()


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
