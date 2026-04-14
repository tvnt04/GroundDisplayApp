# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

datas = [('/home/xdlinx/Downloads/DisplayGoundAD/recent.json', '.'), ('/home/xdlinx/Downloads/DisplayGoundAD/last_session.json', '.')]
hiddenimports = ['pyqtgraph', 'cv2', 'OpenGL', 'OpenGL.platform.egl', 'OpenGL.platform.glx', 'OpenGL.platform.osmesa', 'OpenGL.platform.x11']
datas += collect_data_files('pyqtgraph')
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('OpenGL')


a = Analysis(
    ['/home/xdlinx/Downloads/DisplayGoundAD/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DisplayGroundx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DisplayGroundx',
)
