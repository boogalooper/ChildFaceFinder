from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SETTINGS_VERSION = 1


def default_settings_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return base / "ChildFaceFinder" / "settings.json"


def load_user_settings(path: Path | None = None) -> dict[str, Any]:
    target = path or default_settings_path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != SETTINGS_VERSION:
        return {}
    values = payload.get("values")
    return values if isinstance(values, dict) else {}


def save_user_settings(values: dict[str, Any], path: Path | None = None) -> None:
    target = path or default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": SETTINGS_VERSION, "values": values}
    handle, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
