#!/usr/bin/env bash
set -euo pipefail

PY_CMD=${PY_CMD:-python3.12}
VENV_PATH=${VENV_PATH:-.venv}
REQ_FILE=${REQ_FILE:-requirements.txt}

if ! command -v "$PY_CMD" >/dev/null 2>&1; then
  echo "$PY_CMD is required but not found. Install Python 3.12 and retry." >&2
  exit 1
fi

if [ ! -d "$VENV_PATH" ]; then
  "$PY_CMD" -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo "✅ Installed dependencies into $VENV_PATH using $PY_CMD. Activate with: source $VENV_PATH/bin/activate"
