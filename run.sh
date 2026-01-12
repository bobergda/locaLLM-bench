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
  echo ""
  echo "Generate reports for recent runs:"
  echo "1) Last 5"
  echo "2) Last 10"
  echo "3) Last 20"
  echo "4) All"
  echo "5) Custom path..."
  read -rp "> " choice

  local limit=""
  case "$choice" in
    1) limit=5 ;;
    2) limit=10 ;;
    3) limit=20 ;;
    4) limit="all" ;;
    5)
      read -rp "Enter path to results.json: " results_path
      if [ ! -f "$results_path" ]; then
        echo "results.json not found: $results_path" >&2
        exit 1
      fi
      "$PY_BIN" report.py --results "$results_path"
      return
      ;;
    *)
      echo "Invalid selection." >&2
      exit 1
      ;;
  esac

  local run_paths=()
  if [ -d "results/runs" ]; then
    mapfile -t run_paths < <(find results/runs -maxdepth 2 -type f -name results.json -print 2>/dev/null | sort -r)
  fi
  if [ "${#run_paths[@]}" -eq 0 ] && [ -f "results/latest/results.json" ]; then
    run_paths=("results/latest/results.json")
  fi
  if [ "${#run_paths[@]}" -eq 0 ]; then
    echo "No results.json files found." >&2
    exit 1
  fi

  if [ "$limit" != "all" ] && [ "${#run_paths[@]}" -gt "$limit" ]; then
    run_paths=("${run_paths[@]:0:$limit}")
  fi

  echo "Generating report for ${#run_paths[@]} run(s)..."
  for p in "${run_paths[@]}"; do
    echo "- $p"
    "$PY_BIN" report.py --results "$p" >/dev/null
  done
  echo "Done."
}

clean_runs_without_results() {
  if [ ! -d "results/runs" ]; then
    echo "No runs found (results/runs does not exist)." >&2
    exit 1
  fi

  local removed=0
  while IFS= read -r -d '' dir; do
    if ! find "$dir" -maxdepth 2 -type f -name results.json -print -quit | grep -q .; then
      rm -rf "$dir"
      echo "Removed $dir"
      removed=$((removed + 1))
    fi
  done < <(find results/runs -mindepth 1 -maxdepth 1 -type d -print0)

  if [ "$removed" -eq 0 ]; then
    echo "No runs without results.json found."
  fi
}

echo "Select action:"
echo "1) Run benchmark (runner.py)"
echo "2) Generate reports for recent runs (5/10/20/all)"
echo "3) Clean runs without results.json"
echo "i) Install dependencies (python3.12 venv)"
echo "q) Quit"
read -rp "> " choice

case "$choice" in

  1) run_runner ;;
  2) run_report ;;
  3) clean_runs_without_results ;;
  i|I) run_install ;;
  q|Q) echo "Bye"; exit 0 ;;
  *) echo "Unknown option"; exit 1 ;;
esac
