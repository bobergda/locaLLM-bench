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
  local results_path=""

  echo ""
  echo "Select which results.json to report:"

  local options=()
  local paths=()

  if [ -f "results/latest/results.json" ]; then
    options+=("latest (results/latest/results.json)")
    paths+=("results/latest/results.json")
  fi
  if [ -d "results/runs" ]; then
    mapfile -t run_paths < <(find results/runs -maxdepth 2 -type f -name results.json -print 2>/dev/null | sort -r)
    local limit=25
    local count=0
    for p in "${run_paths[@]}"; do
      options+=("run $(basename "$(dirname "$p")") ($p)")
      paths+=("$p")
      count=$((count + 1))
      if [ "$count" -ge "$limit" ]; then
        break
      fi
    done
  fi
  options+=("custom path…")
  paths+=("__CUSTOM__")

  local i=1
  for opt in "${options[@]}"; do
    echo "$i) $opt"
    i=$((i + 1))
  done
  read -rp "> " choice
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#paths[@]}" ]; then
    echo "Invalid selection." >&2
    exit 1
  fi
  results_path="${paths[$((choice - 1))]}"
  if [ "$results_path" = "__CUSTOM__" ]; then
    read -rp "Enter path to results.json: " results_path
  fi
  if [ ! -f "$results_path" ]; then
    echo "results.json not found: $results_path" >&2
    exit 1
  fi

  "$PY_BIN" report.py --results "$results_path"
}

run_report_all() {
  ensure_venv
  if [ ! -d "results/runs" ]; then
    echo "No runs found (results/runs does not exist)." >&2
    exit 1
  fi

  mapfile -t run_paths < <(find results/runs -maxdepth 2 -type f -name results.json -print 2>/dev/null | sort)
  if [ "${#run_paths[@]}" -eq 0 ]; then
    echo "No results.json files found under results/runs/." >&2
    exit 1
  fi

  echo "Generating report for ${#run_paths[@]} run(s)..."
  for p in "${run_paths[@]}"; do
    echo "- $p"
    "$PY_BIN" report.py --results "$p" >/dev/null
  done
  echo "Done."
}

echo "Select action:"
echo "1) Run benchmark (runner.py)"
echo "2) Generate report (report.py)"
echo "3) Generate reports for all runs"
echo "4) Install dependencies (python3.12 venv)"
echo "q) Quit"
read -rp "> " choice

case "$choice" in
  4) run_install ;;
  1) run_runner ;;
  2) run_report ;;
  3) run_report_all ;;
  q|Q) echo "Bye"; exit 0 ;;
  *) echo "Unknown option"; exit 1 ;;
esac
