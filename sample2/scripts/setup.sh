#!/usr/bin/env bash
# DischargeFlow — Linux/Mac setup (creates .venv locally; never committed to git)
# Run from the sample2 folder:
#   bash scripts/setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "DischargeFlow setup (Linux/Mac)"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+ first." >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    echo "Creating virtual environment in .venv ..."
    python3 -m venv .venv
else
    echo ".venv already exists — skipping creation."
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it with your AWS and LangFuse keys."
else
    echo ".env already exists — not overwritten."
fi

python -m mock_ehr.export_json

echo ""
echo "Setup complete."
echo "  Activate:  source .venv/bin/activate"
echo "  Run all:   python run.py"
echo "  Tests:     pytest tests/ -q"
echo ""
echo "The .venv folder (including bin/) stays on your machine only — it is git-ignored."
