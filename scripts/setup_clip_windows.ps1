$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Run scripts\setup_windows.ps1 first to create the virtual environment.' }
$env:HF_HOME = Join-Path $Root 'models\huggingface'
New-Item -ItemType Directory -Force $env:HF_HOME | Out-Null
& $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU-only PyTorch installation failed.' }
& $Python -m pip install -r requirements-vision.txt
if ($LASTEXITCODE -ne 0) { throw 'Vision dependency installation failed.' }
& $Python scripts\create_sample_motor_image.py
& $Python scripts\cache_clip_model.py
if ($LASTEXITCODE -ne 0) { throw 'CLIP CPU load/inference verification failed.' }
Write-Host 'CLIP is cached locally and one CPU inference pass succeeded.'
