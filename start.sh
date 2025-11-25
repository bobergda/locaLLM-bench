#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=${VENV_PATH:-.venv}
PY_BIN="$VENV_PATH/bin/python"
INSTALL_SCRIPT=${INSTALL_SCRIPT:-install.sh}

run_install() {
  if [ ! -x "$INSTALL_SCRIPT" ]; then
    echo "Install script '$INSTALL_SCRIPT' not found or not executable." >&2
    exit 1
  fi
  bash "$INSTALL_SCRIPT"
}

ensure_venv() {
  if [ ! -x "$PY_BIN" ]; then
    echo "Virtualenv not found at $VENV_PATH. Run install first (uses python3.12 by default)." >&2
    exit 1
  fi
}

run_runner() {
  ensure_venv
  "$PY_BIN" runner.py
}

run_report() {
  ensure_venv
  "$PY_BIN" report.py
}

echo "Select action:"
echo "1) Install dependencies (python3.12 venv)"
echo "2) Run benchmark (runner.py)"
echo "3) Generate report (report.py)"
echo "q) Quit"
read -rp "> " choice

case "$choice" in
  1) run_install ;;
  2) run_runner ;;
  3) run_report ;;
  q|Q) echo "Bye"; exit 0 ;;
  *) echo "Unknown option"; exit 1 ;;
esac
