param([switch]$WithClip)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$PythonExe = $env:EDGEFIX_PYTHON
$PythonArgs = @()
if (-not $PythonExe) {
    if (Get-Command py -ErrorAction SilentlyContinue) { $PythonExe = 'py'; $PythonArgs = @('-3') }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $PythonExe = (Get-Command python).Source }
    else { throw 'Python 3.11+ is required. Install Python or set EDGEFIX_PYTHON to its python.exe path.' }
}
& $PythonExe @PythonArgs -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or newer is required.' }
& $PythonExe @PythonArgs -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data\manuals,data\processed,data\sample_images | Out-Null
& .\.venv\Scripts\python.exe scripts\generate_sample_manuals.py
& .\.venv\Scripts\python.exe scripts\ingest_documents.py
& .\.venv\Scripts\python.exe scripts\health_check.py
if ($WithClip) {
    & .\scripts\setup_clip_windows.ps1
    if ($LASTEXITCODE -ne 0) { throw 'Optional CLIP setup failed.' }
}
Write-Host 'Setup complete. Start the app with: .\.venv\Scripts\python.exe -m streamlit run app.py'
