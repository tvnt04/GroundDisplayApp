from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQ_FILE = ROOT / "requirement.txt"
VENV_DIR = ROOT / ".venv"


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT))


def venv_python() -> Path:
    if platform.system().lower() == "windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> Path:
    py = venv_python()
    if py.exists():
        return py

    print("Creating virtual environment...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    return py


def main() -> int:
    if not REQ_FILE.exists():
        print(f"Missing dependency file: {REQ_FILE}")
        return 1

    py = ensure_venv()

    print("Upgrading pip...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing application dependencies...")
    run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])

    print()
    print("Installation complete.")
    print(f"Virtual environment: {VENV_DIR}")
    print("Run the app with:")
    print(f"  {py} main.py")
    print()
    print("If you are on Linux/macOS and need system packages for Qt/OpenCV,")
    print("install those with your OS package manager before launching the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
