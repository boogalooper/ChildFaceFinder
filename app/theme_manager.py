"""Application theme follows the Windows apps preference without changing it."""
import tkinter as tk
from tkinter import ttk

MODES = ('Как в Windows', 'Светлая', 'Тёмная')


def windows_dark():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize') as key:
            return winreg.QueryValueEx(key, 'AppsUseLightTheme')[0] == 0
    except (ImportError, OSError):
        return False


def is_dark(mode):
    return mode == 'Тёмная' or (mode == 'Как в Windows' and windows_dark())


class ThemeManager:
    def __init__(self, app):
        self.app = app
        self.dark = None
        app.theme_var.trace_add('write', self.changed)
        self.apply()
        app.after(2000, self.poll)

    def changed(self, *_):
        self.apply()
        self.app._save_settings()

    def poll(self):
        if is_dark(self.app.theme_var.get()) != self.dark:
            self.apply()
        self.app.after(2000, self.poll)

    def apply(self):
        self.dark = is_dark(self.app.theme_var.get())
        bg, panel, text, muted, field, border, selected, accent = (
            ('#151b25','#202936','#e5edf7','#9eacc0','#18212d','#3a485c','#263f65','#83b3ff') if self.dark else
            ('#eef2f7','#ffffff','#202c40','#718096','#f9fbfe','#dce4ef','#eaf2ff','#2864dc'))
        s = ttk.Style(self.app)
        s.configure('.', background=panel, foreground=text, bordercolor=border, lightcolor=border, darkcolor=border)
        for cls in ('TFrame','TLabel','TCheckbutton','TRadiobutton','TButton','TLabelframe','TLabelframe.Label'):
            s.configure(cls, background=panel, foreground=text)
        s.configure('Background.TFrame', background=bg)
        s.configure('Muted.TLabel', foreground=muted)
        s.configure('TSeparator', background=border)
        for cls in ('TButton','TCheckbutton','TRadiobutton','Card.TRadiobutton'):
            s.map(cls, background=[('disabled',panel),('selected',selected),('active',selected)], foreground=[('disabled',muted),('selected',accent)])
        for cls in ('TEntry','TCombobox','TSpinbox'):
            s.configure(cls, fieldbackground=field, background=panel, foreground=text, insertcolor=text, arrowcolor=text, selectbackground='#2864dc', selectforeground='white')
            s.map(cls, fieldbackground=[('disabled',panel),('readonly',field)], foreground=[('disabled',muted),('readonly',text)])
        s.configure('Primary.TButton', background='#2864dc', foreground='white')
        s.map('Primary.TButton', background=[('disabled',border),('active','#3676eb')],foreground=[('disabled',muted),('!disabled','white')])
        s.configure('Horizontal.TProgressbar', background='#4d8af0', troughcolor=field)
        s.configure('TNotebook', background=bg)
        s.configure('TNotebook.Tab', background=bg, foreground=muted)
        s.map('TNotebook.Tab', background=[('selected',panel),('active',selected)],foreground=[('selected',accent)])
        s.configure('Vertical.TScrollbar', background=border, troughcolor=field, arrowcolor=text)
        self.app.option_add('*TCombobox*Listbox.background', field)
        self.app.option_add('*TCombobox*Listbox.foreground', text)
        self.app.option_add('*TCombobox*Listbox.selectBackground', '#2864dc')
        self.app.option_add('*TCombobox*Listbox.selectForeground', 'white')
        colors = {'#eef2f7':bg,'#ffffff':panel,'#202c40':text,'#718096':muted,'#555555':muted,'#2864dc':accent}
        def visit(widget):
            if isinstance(widget, (tk.Tk,tk.Toplevel)):
                widget.configure(background=bg)
            elif isinstance(widget,tk.Text):
                widget.configure(background=field,foreground=text,insertbackground=text,selectbackground='#2864dc',selectforeground='white')
            elif isinstance(widget,ttk.Label):
                if not hasattr(widget,'_theme_colors'):
                    widget._theme_colors={k:str(widget.cget(k)) for k in ('background','foreground')}
                for key,value in widget._theme_colors.items():
                    if value:
                        widget.configure(**{key:colors.get(value,value)})
            for child in widget.winfo_children():
                visit(child)
        visit(self.app)
