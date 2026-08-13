from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Display X Studio"


def get_app_data_dir() -> str:
    """Return a per-user directory for writable application state."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        path = Path(base) / APP_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        base = os.environ.get("XDG_STATE_HOME")
        if base:
            path = Path(base) / APP_NAME
        else:
            path = Path.home() / ".local" / "state" / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_app_data_path(filename: str) -> str:
    return str(Path(get_app_data_dir()) / filename)


def migrate_legacy_file(new_path: str, legacy_path: str) -> str:
    """Copy a legacy source-tree file into app data if the new file is missing."""
    new_p = Path(new_path)
    legacy_p = Path(legacy_path)
    if new_p.exists():
        return str(new_p)
    if legacy_p.exists():
        new_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil
            shutil.copy2(legacy_p, new_p)
        except Exception:
            pass
    return str(new_p)
