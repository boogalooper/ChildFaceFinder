"""Light two-column desktop UI; recognition and file operations live in App."""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import tkinter.font as fonts

BG = '#eef2f7'
WHITE = '#ffffff'
INK = '#202c40'
MUTED = '#718096'
BLUE = '#2864dc'


def build(app):
    app.title('Child Face Finder — новый интерфейс · тестовая версия')
    app.geometry('1100x760')
    app.minsize(940, 720)
    app.configure(background=BG)
    family = 'Segoe UI' if 'Segoe UI' in fonts.families(app) else 'DejaVu Sans'
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont'):
        fonts.nametofont(name).configure(family=family, size=10)
    style = ttk.Style(app)
    style.theme_use('clam')
    style.configure('.', font=(family, 10), foreground=INK, background=WHITE)
    style.configure('TFrame', background=WHITE)
    style.configure('Background.TFrame', background=BG)
    style.configure('TLabel', background=WHITE)
    style.configure('Muted.TLabel', foreground=MUTED)
    style.configure('Heading.TLabel', font=(family, 12, 'bold'))
    style.configure('TButton', padding=(10, 5), background=WHITE, bordercolor='#dce4ef', relief='flat')
    style.map('TButton', background=[('active', '#eaf1fd')])
    style.configure('Primary.TButton', background=BLUE, foreground=WHITE, padding=(15, 12), font=(family, 10, 'bold'))
    style.map('Primary.TButton', background=[('disabled', '#bdcbe3'), ('active', '#1d52be')], foreground=[('disabled', '#f7f9fc')])
    style.configure('TEntry', fieldbackground='#f9fbfe', bordercolor='#dce4ef', padding=5)
    style.configure('TCheckbutton', padding=(0, 1), background=WHITE)
    style.map('TCheckbutton', background=[('active', WHITE)])
    style.configure('TRadiobutton', background=WHITE, padding=(0, 1))
    style.configure('Card.TRadiobutton', padding=(8, 7), font=(family, 10, 'bold'), relief='solid', borderwidth=1)
    style.map('Card.TRadiobutton', background=[('selected', '#eaf2ff'), ('active', '#f4f7fc')], foreground=[('selected', BLUE)])
    style.configure('Horizontal.TProgressbar', background=BLUE, troughcolor='#e8edf5', borderwidth=0, thickness=8)
    style.configure('TNotebook', background=BG, borderwidth=0)
    style.configure('TNotebook.Tab', padding=(10, 8))
    style.map('TNotebook.Tab', background=[('selected', WHITE)], foreground=[('selected', BLUE)])
    app._main_controls = []
    outer = ttk.Frame(app, style='Background.TFrame', padding=16)
    outer.pack(fill='both', expand=True)
    header = ttk.Frame(outer, style='Background.TFrame')
    header.pack(fill='x', pady=(0, 12))
    ttk.Label(header, text='Child Face Finder', background=BG, font=(family, 18, 'bold')).pack(anchor='w')
    ttk.Label(header, text='Фотографии, собранные по людям', background=BG, foreground=MUTED).pack(anchor='w', pady=(3, 0))
    from theme_manager import MODES
    theme = ttk.Combobox(header, textvariable=app.theme_var, values=MODES, state='readonly', width=17)
    theme.place(relx=1, rely=.5, anchor='e')
    body = ttk.Frame(outer, style='Background.TFrame')
    body.pack(fill='both', expand=True)
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=0, minsize=300)
    body.rowconfigure(0, weight=1)
    left = ttk.Frame(body)
    left.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
    right = ttk.Frame(body, width=290)
    right.grid(row=0, column=1, sticky='nsew')
    def panel(parent, title):
        card = ttk.Frame(parent, padding=10)
        card.pack(fill='x', pady=(0, 8))
        ttk.Label(card, text=title, style='Heading.TLabel').pack(anchor='w', pady=(0, 8))
        return card
    def path_row(parent, title, var, command):
        row = ttk.Frame(parent)
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text=title, style='Muted.TLabel', width=17).grid(row=0, column=0, sticky='w', padx=(0, 8))
        entry = ttk.Entry(row, textvariable=var)
        entry.grid(row=0, column=1, sticky='ew', padx=(0, 8))
        button = ttk.Button(row, text='Обзор…', command=command)
        button.grid(row=0, column=2)
        app._main_controls.extend((entry, button))
        return row
    first = panel(left, '01   Что нужно найти')
    choices = ttk.Frame(first)
    choices.pack(fill='x')
    for i, (title, value) in enumerate((('Людей по эталонам', 'references'), ('Уникальные лица', 'discover'))):
        choices.columnconfigure(i, weight=1)
        b = ttk.Radiobutton(choices, text=title, variable=app.scenario_var, value=value, command=app._scenario_changed, style='Card.TRadiobutton')
        b.grid(row=0, column=i, sticky='ew', padx=(0, 8) if i == 0 else 0)
        app._main_controls.append(b)
    paths = ttk.Frame(first)
    paths.pack(fill='x', pady=(14, 0))
    paths.columnconfigure(0, weight=1)
    app.ref_row = path_row(paths, 'Эталонные фото', app.ref_var, app._browse_refs)
    app.ref_row.grid(row=0, column=0, sticky='ew', pady=(0, 8))
    path_row(paths, 'Фото съёмки', app.photos_var, app._browse_photos).grid(row=1, column=0, sticky='ew', pady=(0, 8))
    recursive = ttk.Checkbutton(first, text='Включая подпапки съёмки', variable=app.recursive_var)
    recursive.pack(anchor='w')
    app._main_controls.append(recursive)
    app.scenario_hint = ttk.Label(first, style='Muted.TLabel', wraplength=560)
    app.scenario_hint.pack(anchor='w', pady=(8, 0))
    second = panel(left, '02   Как сохранить результат')
    second_inner = ttk.Frame(second)
    second_inner.pack(fill='x')
    second_inner.columnconfigure(0, weight=1)
    outputs = ttk.Frame(second_inner)
    outputs.grid(row=0, column=0, sticky='ew', pady=(0, 12))
    for i, (title, value) in enumerate((('По папкам', 'folders'), ('Список CSV', 'csv'), ('Только метаданные', 'analysis'))):
        outputs.columnconfigure(i, weight=1)
        b = ttk.Radiobutton(outputs, text=title, variable=app.output_mode_var, value=value, style='Card.TRadiobutton')
        b.grid(row=0, column=i, sticky='ew')
        app._main_controls.append(b)
    app.csv_row = path_row(second_inner, 'Файл CSV', app.output_var, app._browse_output)
    app.csv_row.grid(row=1, column=0, sticky='ew')
    app.folder_rows = ttk.Frame(second_inner)
    app.folder_rows.columnconfigure(0, weight=1)
    app.folder_rows.grid(row=2, column=0, sticky='ew')
    path_row(app.folder_rows, 'Папка результата', app.destination_var, app._browse_destination).grid(row=0, column=0, sticky='ew')
    transfer = ttk.Frame(app.folder_rows)
    transfer.grid(row=1, column=0, sticky='w', pady=(6, 0))
    for title, value in (('Копировать', 'copy'), ('Переместить', 'move')):
        b = ttk.Radiobutton(transfer, text=title, variable=app.transfer_var, value=value)
        b.pack(side='left', padx=(0, 20))
        app._main_controls.append(b)
    app.transfer_hint = ttk.Label(app.folder_rows, wraplength=560, style='Muted.TLabel')
    app.transfer_hint.grid(row=2, column=0, sticky='w', pady=(8, 0))
    third = panel(left, '03   Метаданные XMP')
    master = ttk.Checkbutton(third, text='Записывать метаданные XMP', variable=app.xmp_enabled_var)
    master.pack(anchor='w')
    app.people_check = ttk.Checkbutton(third, text='Перезаписать список найденных людей', variable=app.write_xmp_var)
    app.people_check.pack(anchor='w', pady=(6, 0))
    app.best_check = ttk.Checkbutton(third, text='Отметить лучшие кадры в сериях цветом', variable=app.best_series_var)
    app.best_check.pack(anchor='w', pady=(4, 0))
    app._main_controls.extend((master, app.people_check, app.best_check))
    app.metadata_hint = ttk.Label(third, wraplength=560, style='Muted.TLabel')
    app.metadata_hint.pack(anchor='w', pady=(6, 0))
    third.pack_configure(fill='both', expand=True, pady=0)
    summary = panel(right, 'Ваше задание')
    summary.pack_configure(fill='both', expand=True, pady=0)
    app.parameter_panel, app.task_panel = left, right
    app.summary_vars = {}
    for key, title in (('scenario', 'Поиск'), ('output', 'Результат'), ('metadata', 'XMP'), ('quality', 'Отбор качества')):
        row = ttk.Frame(summary)
        row.pack(fill='x', pady=3)
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text=title, style='Muted.TLabel').grid(row=0, column=0, sticky='nw', padx=(0, 12))
        var = tk.StringVar()
        app.summary_vars[key] = var
        ttk.Label(row, textvariable=var, wraplength=145, anchor='e', justify='right', font=(family, 10, 'bold')).grid(row=0, column=1, sticky='ne')
    ttk.Separator(summary).pack(fill='x', pady=12)
    app.settings_btn = ttk.Button(summary, text='Настройки обработки', command=app._open_settings)
    app.settings_btn.pack(fill='x')
    app._main_controls.append(app.settings_btn)
    app.launch_hint = ttk.Label(summary, wraplength=250, style='Muted.TLabel')
    app.launch_hint.pack(fill='x', pady=10)
    app.start_btn = ttk.Button(summary, text='Начать обработку', command=app._start, style='Primary.TButton')
    app.start_btn.pack(fill='x')
    app.cancel_btn = ttk.Button(summary, text='Отмена', command=app._cancel, state='disabled')
    app.cancel_btn.pack(fill='x', pady=(8, 0))
    app.collect_btn = ttk.Button(app, command=app._collect_files)
    ttk.Frame(summary).pack(fill='both', expand=True)
    progress = panel(summary, 'Ход обработки')
    progress.pack_configure(pady=0)
    ttk.Label(progress, textvariable=app.status_var, wraplength=250).pack(fill='x')
    app.progress = ttk.Progressbar(progress, variable=app.progress_var, maximum=100)
    app.progress.pack(fill='x', pady=10)
    def open_log():
        from user_settings import default_settings_path
        path = getattr(app, 'run_log_path', None)
        if path is None:
            path = default_settings_path().parent / 'logs'
            path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except Exception as exc:
            messagebox.showerror('Журнал', str(exc), parent=app)
    ttk.Button(progress, text='Открыть лог-файл', command=open_log).pack(fill='x')
    def wrap(event):
        width = max(280, event.width - 48)
        for label in (app.scenario_hint, app.transfer_hint, app.metadata_hint):
            label.configure(wraplength=width)
    left.bind('<Configure>', wrap, add='+')
    for variable in (app.scenario_var, app.output_mode_var, app.transfer_var, app.xmp_enabled_var,
                     app.write_xmp_var, app.best_series_var, app.ref_var, app.photos_var,
                     app.output_var, app.destination_var, app.best_label_var, app.filter_quality_var):
        variable.trace_add('write', lambda *_: app._refresh_workflow())
    app._refresh_workflow()
