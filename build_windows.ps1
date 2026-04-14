$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

py -m pip install --upgrade pyinstaller
py build_exe.py

Write-Host "Windows build complete: $PSScriptRoot\dist\DisplayGroundx\DisplayGroundx.exe"
