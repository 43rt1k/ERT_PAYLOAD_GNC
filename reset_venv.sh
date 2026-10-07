#!/bin/bash
set -euo pipefail



VENV_NAME=".RTvenv"
PYTHON_VERSION="3.11"
PYTHON="/opt/homebrew/bin/python${PYTHON_VERSION}"


# rm -r "$VENV_NAME" 

"$PYTHON" -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -e .
python -m pip check

echo "Virtual environment reset and dependencies installed successfully."