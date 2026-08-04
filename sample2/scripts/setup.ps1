# DischargeFlow — Windows setup (creates .venv locally; never committed to git)
# Run in PowerShell from the sample2 folder:
#   .\scripts\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "DischargeFlow setup (Windows)" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python not found. Install Python 3.11+ and add it to PATH."
}

if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment in .venv ..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists — skipping creation."
}

Write-Host "Activating .venv and installing dependencies ..."
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example — edit it with your AWS and LangFuse keys."
} else {
    Write-Host ".env already exists — not overwritten."
}

# Materialise Mock EHR JSON if missing
python -m mock_ehr.export_json

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "  Run all:   python run.py"
Write-Host "  Tests:     pytest tests/ -q"
Write-Host ""
Write-Host "The .venv folder (including Scripts/) stays on your machine only — it is git-ignored."
