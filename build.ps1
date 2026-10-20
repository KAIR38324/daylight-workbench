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
& $Python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; packaging stopped" }
& $Python scripts/smoke.py --gui
if ($LASTEXITCODE -ne 0) { throw "GUI smoke failed; packaging stopped" }
& $Python qa/gui_smoke.py
if ($LASTEXITCODE -ne 0) { throw "Async GUI integration failed; packaging stopped" }
& $Python -m PyInstaller --noconfirm --clean --onedir --windowed --name DaylightWorkbench --collect-all pymoo --collect-all torch --collect-submodules sklearn --hidden-import matplotlib.backends.backend_qtagg --add-data "demo;demo" --add-data "README.zh-CN.md;." main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller packaging failed" }
$env:QT_QPA_PLATFORM = "offscreen"
$Process = Start-Process -FilePath "dist\DaylightWorkbench\DaylightWorkbench.exe" -ArgumentList "--self-test" -Wait -PassThru
if ($Process.ExitCode -ne 0) { throw "Packaged EXE self-test failed: $($Process.ExitCode)" }
Write-Host "Built dist\DaylightWorkbench\DaylightWorkbench.exe. Distribute the ENTIRE folder, not the exe alone."
Write-Host "Manual Windows acceptance is still required: import -> train -> optimize -> filter -> export -> save/reopen -> cancel."
