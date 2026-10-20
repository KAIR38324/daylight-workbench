# Run on Windows x64 with Python 3.12: powershell -ExecutionPolicy Bypass -File .\build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 x64 is required." }
}
$Python = ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $Python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed" }
& $Python -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $Python -m daylight_workbench.interactive_plot --prepare-assets
if ($LASTEXITCODE -ne 0) { throw "Offline plot asset preparation failed" }
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; packaging stopped" }
& $Python scripts/smoke.py --gui
if ($LASTEXITCODE -ne 0) { throw "GUI smoke failed; packaging stopped" }
& $Python qa/gui_smoke.py
if ($LASTEXITCODE -ne 0) { throw "Async GUI integration failed; packaging stopped" }
& $Python qa/v02_gui_smoke.py
if ($LASTEXITCODE -ne 0) { throw "v0.2 multi-model GUI integration failed; packaging stopped" }
& $Python -m PyInstaller --noconfirm --clean --onedir --windowed --name DaylightWorkbench --icon "daylight_workbench/assets/app-icon.ico" --collect-all pymoo --collect-all torch --collect-binaries xgboost --collect-data xgboost --collect-submodules numpy --collect-submodules sklearn --hidden-import numpy._core._exceptions --hidden-import PySide6.QtWebEngineWidgets --hidden-import PySide6.QtWebChannel --add-data "daylight_workbench/assets;daylight_workbench/assets" --add-data "daylight_workbench/plot_assets;daylight_workbench/plot_assets" --hidden-import matplotlib.backends.backend_qtagg --add-data "demo;demo" --add-data "README.zh-CN.md;." main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller packaging failed" }
$env:QT_QPA_PLATFORM = "offscreen"
$Report = Join-Path $env:TEMP "daylight-workbench-selftest.json"
if (Test-Path $Report) { Remove-Item $Report }
$Process = Start-Process -FilePath "dist\DaylightWorkbench\DaylightWorkbench.exe" -ArgumentList "--self-test" -Wait -PassThru
if (Test-Path $Report) {
    New-Item -ItemType Directory -Force "qa\evidence" | Out-Null
    Copy-Item $Report "qa\evidence\packaged-selftest.json"
    Get-Content -Raw -Encoding UTF8 $Report | Write-Host
} else {
    Write-Host "Packaged executable did not write a self-test report."
}
if ($Process.ExitCode -ne 0) { throw "Packaged EXE self-test failed: $($Process.ExitCode)" }
Write-Host "Built dist\DaylightWorkbench\DaylightWorkbench.exe. Distribute the ENTIRE folder, not the exe alone."
Write-Host "Manual Windows acceptance is still required: import -> train -> optimize -> filter -> export -> save/reopen -> cancel."
