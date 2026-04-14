#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

python3 -m pip install --upgrade pyinstaller
python3 build_exe.py

echo "Linux build complete: $ROOT_DIR/dist/DisplayGroundx/DisplayGroundx"
