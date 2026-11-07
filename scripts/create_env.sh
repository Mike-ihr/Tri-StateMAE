#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-Tri_State_MAE}"

python -m venv "${ENV_NAME}"
source "${ENV_NAME}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Environment ready: source ${ENV_NAME}/bin/activate"
