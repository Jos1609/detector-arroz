# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['web_python.py'],
    pathex=[],
    binaries=[],
    datas=[('rice_app\\templates', 'rice_app\\templates'), ('rice_app\\static', 'rice_app\\static'), ('modelo_arroz_detect.pt', '.'), ('yolov8n.pt', '.')],
    hiddenimports=[],
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
    name='DetectorArroz',
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
    name='DetectorArroz',
)
