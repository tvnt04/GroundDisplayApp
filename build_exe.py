#!/usr/bin/env python3
"""Build a distributable executable for DisplayGroundx using PyInstaller.

Usage:
  python3 build_exe.py

Output:
  dist/DisplayGroundx/ (or dist/DisplayGroundx.exe on Windows when using --onefile)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "DisplayGroundx"
ENTRY_SCRIPT = "main.py"


def _pyinstaller_cmd() -> list[str]:
    # Prefer executable on PATH; fallback to module invocation.
    if shutil.which("pyinstaller"):
        return ["pyinstaller"]
    return [sys.executable, "-m", "PyInstaller"]


def _add_data_arg(path: Path) -> list[str]:
    sep = ";" if os.name == "nt" else ":"
    return ["--add-data", f"{path}{sep}."]


def build() -> int:
    root = Path(__file__).resolve().parent
    entry = root / ENTRY_SCRIPT
    if not entry.exists():
        print(f"Error: entry file not found: {entry}")
        return 1

    pyinstaller_cmd = _pyinstaller_cmd()
    if pyinstaller_cmd[:3] == [sys.executable, "-m", "PyInstaller"]:
        try:
            __import__("PyInstaller")
        except Exception:
            print("PyInstaller is not installed in this Python environment.")
            print(f"Install it with: {sys.executable} -m pip install pyinstaller")
            return 1

    hidden_imports = [
        "pyqtgraph",
        "cv2",
        "OpenGL",
    ]
    # Platform-specific OpenGL backends.
    if os.name == "nt":
        hidden_imports.extend(["OpenGL.platform.win32"])
    else:
        hidden_imports.extend([
            "OpenGL.platform.egl",
            "OpenGL.platform.glx",
            "OpenGL.platform.osmesa",
            "OpenGL.platform.x11",
        ])

    cmd = pyinstaller_cmd + [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name",
        APP_NAME,
        "--collect-submodules",
        "cv2",
        "--collect-submodules",
        "OpenGL",
        "--collect-data",
        "pyqtgraph",
    ]

    for mod in hidden_imports:
        cmd.extend(["--hidden-import", mod])

    # Optional runtime files used by the app.
    optional_files = ["recent.json", "last_session.json", "Xdlinx_Cam"]
    if os.name == "nt":
        optional_files.insert(0, "Xdlinx_Cam.exe")
    for optional_file in optional_files:
        p = root / optional_file
        if p.exists():
            cmd.extend(_add_data_arg(p))

    cmd.append(str(entry))

    print("Building executable with command:")
    print(" ".join(cmd))
    print()

    try:
        subprocess.run(cmd, cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Build failed with exit code {exc.returncode}")
        return exc.returncode
    except FileNotFoundError:
        print("PyInstaller is not installed.")
        print(f"Install it with: {sys.executable} -m pip install pyinstaller")
        return 1

    dist_path = root / "dist" / APP_NAME
    print("\nBuild complete.")
    print(f"Executable folder: {dist_path}")
    if os.name == "nt":
        print(f"Main executable: {dist_path / (APP_NAME + '.exe')}")
    else:
        print(f"Main executable: {dist_path / APP_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
