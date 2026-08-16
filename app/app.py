from __future__ import annotations

import os
import queue
import site
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Portable builds keep third-party packages outside the bundled CPython tree.
# addsitedir() also processes any .pth files, unlike a plain PYTHONPATH entry.
_PROJECT_DIR = Path(__file__).resolve().parent.parent
_PORTABLE_SITE_PACKAGES = _PROJECT_DIR / "runtime" / "site-packages"
if _PORTABLE_SITE_PACKAGES.is_dir():
    site.addsitedir(str(_PORTABLE_SITE_PACKAGES))

from engine import CancelledError, ChildFaceFinder, RunSummary, Settings
from image_loader import cleanup_stale_temp_dirs

APP_TITLE = "Child Face Finder"

QUALITY_PRESETS = {
    "Мягкий": dict(
        min_face_px=35,
        max_yaw=55.0,
        max_pitch=45.0,
        max_roll=45.0,
        eye_ratio_min=0.080,
        blur_threshold=25.0,
        max_clipped_fraction=0.70,
    ),
    "Нормальный": dict(
        min_face_px=45,
        max_yaw=45.0,
        max_pitch=35.0,
        max_roll=35.0,
        eye_ratio_min=0.105,
        blur_threshold=45.0,
        max_clipped_fraction=0.55,
    ),
    "Строгий": dict(
        min_face_px=65,
        max_yaw=30.0,
        max_pitch=25.0,
        max_roll=25.0,
        eye_ratio_min=0.125,
        blur_threshold=75.0,
        max_clipped_fraction=0.35,
    ),
}

SEARCH_MODES = {
    "Быстрый": dict(tile_enabled=False, tile_size=2200, tile_overlap=0.18, tile_trigger_px=1800),
    "Улучшенный": dict(tile_enabled=True, tile_size=2200, tile_overlap=0.18, tile_trigger_px=1800),
    "Максимальный": dict(tile_enabled=True, tile_size=1600, tile_overlap=0.22, tile_trigger_px=1400),
}


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
        self.closing_after_cancel = False

        self.ref_var = tk.StringVar()
        self.photos_var = tk.StringVar()
        self.output_var = tk.StringVar()

        self.search_mode_var = tk.StringVar(value="Улучшенный")
        self.detector_threshold_var = tk.StringVar(value="0.18")
        self.quality_preset_var = tk.StringVar(value="Нормальный")
        self.threshold_var = tk.StringVar(value="0.45")
        self.margin_var = tk.StringVar(value="0.04")

        self.closed_eyes_var = tk.BooleanVar(value=True)
        self.blur_var = tk.BooleanVar(value=True)
        self.pose_var = tk.BooleanVar(value=True)
        self.small_face_var = tk.BooleanVar(value=True)
        self.exposure_var = tk.BooleanVar(value=False)

        self.decode_workers_var = tk.IntVar(value=max(2, min(8, os.cpu_count() or 4)))
        self.inference_workers_var = tk.IntVar(value=2)
        self.recursive_var = tk.BooleanVar(value=True)
        self.require_gpu_var = tk.BooleanVar(value=True)
        self.verbose_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Готово к запуску")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        removed_temp, failed_temp = cleanup_stale_temp_dirs()
        if removed_temp:
            self._append_log(f"TEMP: удалено оставшихся каталогов предыдущих запусков: {len(removed_temp)}")
        for path, error in failed_temp:
            self._append_log(f"TEMP: не удалось удалить {path}: {error}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(9, weight=1)

        ttk.Label(outer, text="Эталонные изображения детей:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.ref_var).grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(outer, text="Обзор…", command=self._browse_refs).grid(row=0, column=2, pady=4)

        ttk.Label(outer, text="Папка фотографий:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.photos_var).grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(outer, text="Обзор…", command=self._browse_photos).grid(row=1, column=2, pady=4)

        ttk.Label(outer, text="CSV результата:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(outer, textvariable=self.output_var).grid(row=2, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(outer, text="Обзор…", command=self._browse_output).grid(row=2, column=2, pady=4)

        ref_note = (
            "Эталоны ищутся во всех подпапках. ID берётся только из имени файла: "
            "«портрет\\Котов Никита.jpg», «стоя\\Котов Никита.jpg» и другие файлы с тем же именем — эталоны «Котов Никита». "
            "Также поддерживаются ведущие номера («712 Котов Никита.jpg») и одинарные имена («Котов.jpg»)."
        )
        ref_label = ttk.Label(outer, text=ref_note, wraplength=950)
        ref_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(1, 8))
        add_tooltip(
            ref_label,
            "Подкаталог (например «портрет» или «стоя») не входит в ID. Сначала объединяются одинаковые нормализованные имена. "
            "Если имена отличаются, несколько эталонов могут быть объединены по однозначной общей части, например «Котов» + «Котов Никита» или "
            "«Котов Никита портрет» + «Котов Никита стоя» + «Котов Никита улица». При неоднозначности программа не угадывает, а сообщает ошибку. "
            "Регистр, лишние пробелы и Е/Ё не мешают. Количество эталонов не ограничено; их embeddings не усредняются.",
        )

        recognition = ttk.LabelFrame(outer, text="Поиск и распознавание лиц", padding=10)
        recognition.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(3, 6))
        for col in range(8):
            recognition.columnconfigure(col, weight=1 if col in (1, 4, 7) else 0)

        ttk.Label(recognition, text="Поиск маленьких лиц:").grid(row=0, column=0, sticky="w", pady=3)
        search_combo = ttk.Combobox(
            recognition,
            textvariable=self.search_mode_var,
            values=list(SEARCH_MODES),
            state="readonly",
            width=15,
        )
        search_combo.grid(row=0, column=1, sticky="w", pady=3)
        hint(
            recognition,
            "Определяет глубину поиска маленьких лиц; пользователь выбирает один из трёх режимов.\n"
            "• Быстрый — только полный кадр, без тайлов. Самый быстрый, но чаще пропускает маленькие лица.\n"
            "• Улучшенный — полный кадр + тайлы 2200 px с перекрытием 18%; рекомендуемый режим.\n"
            "• Максимальный — полный кадр + тайлы 1600 px с перекрытием 22%; лучше для очень маленьких лиц, но медленнее.",
        ).grid(row=0, column=2, sticky="w", padx=(3, 12))

        ttk.Label(recognition, text="Порог детектора:").grid(row=0, column=3, sticky="e", padx=(8, 5), pady=3)
        ttk.Entry(recognition, textvariable=self.detector_threshold_var, width=8).grid(row=0, column=4, sticky="w", pady=3)
        hint(
            recognition,
            "Минимальная уверенность SCRFD при первоначальном поиске лица. Допустимый диапазон ввода: 0.05…0.80.\n"
            "Меньше значение → ищется больше слабых и маленьких лиц, но увеличивается число ложных кандидатов.\n"
            "Больше значение → меньше ложных срабатываний, но выше риск потерять мелкие лица.\n"
            "Практически полезный диапазон: 0.15…0.30. Значение по умолчанию: 0.18. Это НЕ критерий качества фотографии.",
        ).grid(row=0, column=5, sticky="w", padx=(3, 12))

        ttk.Label(recognition, text="Фильтр качества:").grid(row=0, column=6, sticky="e", padx=(8, 5), pady=3)
        quality_combo = ttk.Combobox(
            recognition,
            textvariable=self.quality_preset_var,
            values=list(QUALITY_PRESETS),
            state="readonly",
            width=13,
        )
        quality_combo.grid(row=0, column=7, sticky="w", pady=3)
        add_tooltip(
            quality_combo,
            "Меняет числовую строгость критериев качества. Это не числовой ввод, а выбор из 3 пресетов.\n"
            "• Мягкий: лицо от 35 px, yaw до 55°, pitch/roll до 45°, глаза от 0.080, blur от 25, clipped до 0.70.\n"
            "• Нормальный: лицо от 45 px, yaw до 45°, pitch/roll до 35°, глаза от 0.105, blur от 45, clipped до 0.55.\n"
            "• Строгий: лицо от 65 px, yaw до 30°, pitch/roll до 25°, глаза от 0.125, blur от 75, clipped до 0.35.\n"
            "Сами критерии включаются и выключаются отдельными галочками ниже.",
        )

        ttk.Label(recognition, text="Порог совпадения:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(recognition, textvariable=self.threshold_var, width=8).grid(row=1, column=1, sticky="w", pady=3)
        hint(
            recognition,
            "Максимальная cosine distance до эталона. Допустимый диапазон ввода: 0.05…1.00.\n"
            "Меньше значение → совпадение строже, ложных совпадений меньше, но часть правильных лиц может потеряться.\n"
            "Больше значение → полнота выше, но возрастает риск ложных совпадений.\n"
            "Практически полезный диапазон: 0.35…0.60. Значение по умолчанию: 0.45.",
        ).grid(row=1, column=2, sticky="w", padx=(3, 12))

        ttk.Label(recognition, text="Отрыв от 2-го ребёнка:").grid(row=1, column=3, sticky="e", padx=(8, 5), pady=3)
        ttk.Entry(recognition, textvariable=self.margin_var, width=8).grid(row=1, column=4, sticky="w", pady=3)
        hint(
            recognition,
            "Минимальный отрыв между лучшим ребёнком и следующим ДРУГИМ ребёнком. Допустимый диапазон ввода: 0.00…0.50.\n"
            "Если фактическая разница меньше этого значения, совпадение не назначается автоматически и уходит в review.csv.\n"
            "0.00 — отключает этот дополнительный фильтр.\n"
            "Практически полезный диапазон: 0.02…0.08. Значение по умолчанию: 0.04. Два эталона одного ребёнка здесь не конкурируют друг с другом.",
        ).grid(row=1, column=5, sticky="w", padx=(3, 12))

        quality = ttk.LabelFrame(outer, text="Что считать неудачным фото", padding=10)
        quality.grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        for col in range(3):
            quality.columnconfigure(col, weight=1)

        q1 = ttk.Checkbutton(quality, text="Отбраковывать закрытые/сильно прищуренные глаза", variable=self.closed_eyes_var)
        q1.grid(row=0, column=0, sticky="w", padx=(0, 12), pady=3)
        add_tooltip(q1, "Если включено, распознанный ребёнок будет исключён из result.csv при подозрении на закрытые или сильно прищуренные глаза и попадёт в rejected.csv. Числовой порог задаётся выбранным пресетом качества: 0.080 / 0.105 / 0.125 для Мягкий / Нормальный / Строгий. Проверка эвристическая.")

        q2 = ttk.Checkbutton(quality, text="Отбраковывать нерезкие лица", variable=self.blur_var)
        q2.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=3)
        add_tooltip(q2, "Если включено, нерезкие лица исключаются из result.csv и записываются в rejected.csv. Проверяется резкость области лица по variance of Laplacian. Порог зависит от пресета качества: 25 / 45 / 75 для Мягкий / Нормальный / Строгий. Чем выше порог, тем строже отбраковка.")

        q3 = ttk.Checkbutton(quality, text="Отбраковывать сильный поворот/наклон", variable=self.pose_var)
        q3.grid(row=0, column=2, sticky="w", pady=3)
        add_tooltip(q3, "Если включено, лица с сильным поворотом или наклоном исключаются из result.csv и записываются в rejected.csv. Используются yaw / pitch / roll из InsightFace. Пороги по пресетам: yaw до 55° / 45° / 30°, pitch и roll до 45° / 35° / 25° для Мягкий / Нормальный / Строгий. Отключите, если нужны и выраженные профили.")

        q4 = ttk.Checkbutton(quality, text="Отбраковывать слишком маленькие лица", variable=self.small_face_var)
        q4.grid(row=1, column=0, sticky="w", padx=(0, 12), pady=3)
        add_tooltip(q4, "Если включено, распознанные, но слишком маленькие лица исключаются из result.csv и записываются в rejected.csv. Минимальный размер лица зависит от пресета качества: 35 / 45 / 65 px для Мягкий / Нормальный / Строгий. Это не мешает детектору сначала попытаться найти лицо.")

        q5 = ttk.Checkbutton(quality, text="Отбраковывать плохую экспозицию", variable=self.exposure_var)
        q5.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=3)
        add_tooltip(q5, "Если включено, лица с сильным пересветом или провалом в тени исключаются из result.csv и записываются в rejected.csv. Используется доля пересвеченных/проваленных пикселей в области лица. Порог по пресетам: до 0.70 / 0.55 / 0.35 для Мягкий / Нормальный / Строгий. По умолчанию выключено, потому что критерий заметно зависит от конкретной съёмки.")

        quality_note = ttk.Label(
            quality,
            text="Важно: включённый критерий качества исключает распознанного ребёнка из основного CSV для этого фото и записывает причину в *_rejected.csv.",
            wraplength=930,
        )
        quality_note.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        performance = ttk.LabelFrame(outer, text="Производительность и диагностика", padding=10)
        performance.grid(row=6, column=0, columnspan=3, sticky="ew", pady=6)
        for col in range(8):
            performance.columnconfigure(col, weight=1 if col in (1, 4) else 0)

        ttk.Label(performance, text="Потоки декодирования:").grid(row=0, column=0, sticky="w", pady=3)
        decode_spin = ttk.Spinbox(performance, from_=1, to=32, textvariable=self.decode_workers_var, width=7)
        decode_spin.grid(row=0, column=1, sticky="w", pady=3)
        hint(
            performance,
            "Количество CPU-потоков для чтения JPEG/HEIC/RAW. Допустимый диапазон ввода: 1…32.\n"
            "Они работают параллельно с GPU. Больше потоков ускоряет подготовку файлов, но увеличивает расход RAM и нагрузку на диск.\n"
            "Практически полезный диапазон: 4…8 для большинства ПК. Значение по умолчанию подбирается автоматически.",
        ).grid(row=0, column=2, sticky="w", padx=(3, 15))

        ttk.Label(performance, text="GPU inference-потоки:").grid(row=0, column=3, sticky="e", padx=(8, 5), pady=3)
        inference_spin = ttk.Spinbox(performance, from_=1, to=8, textvariable=self.inference_workers_var, width=7)
        inference_spin.grid(row=0, column=4, sticky="w", pady=3)
        hint(
            performance,
            "Количество фотографий, одновременно проходящих InsightFace на GPU. Допустимый диапазон ввода: 1…8.\n"
            "Больше потоков повышает загрузку GPU, но может увеличить расход VRAM и снизить стабильность на слабых картах.\n"
            "Для RTX 5090 разумный старт: 2. Полезный диапазон для тестов: 1…4. Обычно имеет смысл увеличивать только если GPU недогружен.",
        ).grid(row=0, column=5, sticky="w", padx=(3, 15))

        c1 = ttk.Checkbutton(performance, text="Обрабатывать подпапки с фотографиями", variable=self.recursive_var)
        c1.grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        add_tooltip(c1, "Искать целевые фотографии рекурсивно во всех подпапках. Папка эталонов всегда сканируется рекурсивно независимо от этой галочки.")

        c2 = ttk.Checkbutton(performance, text="Требовать NVIDIA CUDA", variable=self.require_gpu_var)
        c2.grid(row=1, column=2, columnspan=2, sticky="w", pady=3)
        add_tooltip(c2, "Если включено, программа остановится, если CUDAExecutionProvider недоступен, вместо скрытого перехода на CPU.")

        c3 = ttk.Checkbutton(performance, text="Подробная диагностика по каждому фото", variable=self.verbose_var)
        c3.grid(row=1, column=4, columnspan=4, sticky="w", pady=3)
        add_tooltip(c3, "Показывает в журнале: сколько лиц найдено полным кадром, сколько добавили тайлы, сколько осталось после NMS, сколько принято/отбраковано/отправлено на проверку.")

        buttons = ttk.Frame(outer)
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew", pady=4)
        self.start_btn = ttk.Button(buttons, text="Начать", command=self._start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(buttons, text="Отмена", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=8)
        ttk.Label(buttons, textvariable=self.status_var).pack(side="left", padx=12)

        self.progress = ttk.Progressbar(outer, variable=self.progress_var, maximum=100.0)
        self.progress.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(6, 8))

        log_frame = ttk.LabelFrame(outer, text="Журнал", padding=5)
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=sb.set)

        ttk.Label(
            outer,
            text="Предобученные модели InsightFace имеют отдельные лицензионные ограничения; см. README.md.",
            wraplength=950,
        ).grid(row=10, column=0, columnspan=3, sticky="w", pady=(7, 0))

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

    def _browse_output(self) -> None:
        value = filedialog.asksaveasfilename(
            title="CSV результата",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="result.csv",
        )
        if value:
            self.output_var.set(value)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

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
            match_threshold=self._parse_float(self.threshold_var.get(), "Порог совпадения", 0.05, 1.0),
            ambiguity_margin=self._parse_float(self.margin_var.get(), "Отрыв от второго ребёнка", 0.0, 0.5),
            detector_threshold=self._parse_float(self.detector_threshold_var.get(), "Порог детектора", 0.05, 0.8),
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
            tile_enabled=search["tile_enabled"],
            tile_size=search["tile_size"],
            tile_overlap=search["tile_overlap"],
            tile_trigger_px=search["tile_trigger_px"],
            decode_workers=max(1, min(32, int(self.decode_workers_var.get()))),
            inference_workers=max(1, min(8, int(self.inference_workers_var.get()))),
            recursive=self.recursive_var.get(),
            require_gpu=self.require_gpu_var.get(),
            verbose_diagnostics=self.verbose_var.get(),
        )

    def _validate_paths(self) -> tuple[Path, Path, Path]:
        ref_text = self.ref_var.get().strip()
        photos_text = self.photos_var.get().strip()
        output_text = self.output_var.get().strip()
        if not ref_text:
            raise ValueError("Укажите папку эталонов")
        if not photos_text:
            raise ValueError("Укажите папку фотографий")
        if not output_text:
            raise ValueError("Укажите CSV результата")
        refs = Path(ref_text).expanduser()
        photos = Path(photos_text).expanduser()
        output = Path(output_text).expanduser()
        if not refs.is_dir():
            raise ValueError("Укажите существующую папку эталонов")
        if not photos.is_dir():
            raise ValueError("Укажите существующую папку фотографий")
        if output.suffix.casefold() != ".csv":
            output = output.with_suffix(".csv")
            self.output_var.set(str(output))
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

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Инициализация…")
        self.start_btn.configure(state="disabled")
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

    def _cancel(self) -> None:
        if self.finder is not None:
            self.finder.cancel()
            self.cancel_btn.configure(state="disabled")
            self.status_var.set("Отмена…")
            self._append_log("Запрошена отмена. Завершается текущая операция…")

    def _finish_ui(self) -> None:
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.finder = None
        self.worker = None
        if self.closing_after_cancel:
            self.destroy()

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._append_log(event[1])
                elif kind == "progress":
                    _, done, total, name = event
                    self.progress_var.set(0.0 if total <= 0 else done * 100.0 / total)
                    self.status_var.set(f"{done}/{total}: {name}")
                elif kind == "done":
                    summary: RunSummary = event[1]
                    self.progress_var.set(100.0)
                    self.status_var.set("Готово")
                    self._append_log(
                        f"Готово. Детей: {summary.reference_ids}; эталонных файлов: {summary.reference_files}; "
                        f"фото: {summary.photo_count}; совпадений: {summary.matched_pairs}; "
                        f"отбраковано: {summary.rejected_pairs}; на проверку: {summary.review_rows}."
                    )
                    self._finish_ui()
                    if not self.closing_after_cancel:
                        messagebox.showinfo(
                            APP_TITLE,
                            "Обработка завершена.\n\n"
                            f"Основной CSV:\n{summary.output_csv}\n\n"
                            f"Отбракованные:\n{summary.rejected_csv}\n\n"
                            f"Сомнительные совпадения:\n{summary.review_csv}",
                            parent=self,
                        )
                elif kind == "cancelled":
                    self.status_var.set("Отменено")
                    self._append_log(event[1])
                    self._finish_ui()
                elif kind == "error":
                    _, text, details = event
                    self.status_var.set("Ошибка")
                    self._append_log(details)
                    self._finish_ui()
                    if not self.closing_after_cancel:
                        messagebox.showerror(APP_TITLE, text, parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll_events)

    def _on_close(self) -> None:
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
