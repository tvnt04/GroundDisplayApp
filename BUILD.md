# Build Guide (Linux + Windows)

PyInstaller builds are platform-specific.

- Build Linux executable on Linux.
- Build Windows executable on Windows.

## Linux

```bash
./build_linux.sh
```

Output:

- `dist/DisplayGroundx/DisplayGroundx`

## Windows (PowerShell)

```powershell
./build_windows.ps1
```

Output:

- `dist\DisplayGroundx\DisplayGroundx.exe`

## Notes

- `build_exe.py` auto-selects platform-specific OpenGL imports.
- Live mode camera helper binary is platform-aware:
  - Linux: `Xdlinx_Cam`
  - Windows: `Xdlinx_Cam.exe`
- Include the correct camera helper binary in the project root before building.
