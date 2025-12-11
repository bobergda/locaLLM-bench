# LocaLLM Bench – Ops Note

## Goal
Minimal benchmark of local LLMs via Ollama: compare quality and response time on simple task sets (instructions, code with auto-tests, logic, PL), save raw results, and generate a report with accuracy and helper metrics.

## Project Status
- Runtime: Python 3.12.
- Connection: `ollama` library (`client.generate(stream=False)`), host set in `config.yaml`.
- Key files: `runner.py`, `metrics.py`, `report.py`, `config.yaml`, `tests/*.json`.
- Start: `start.sh` (menu: install, benchmark, report) or directly `.venv/bin/python runner.py`.
- Debug supported via config (`debug`, `debug_models`, `debug_categories`, `debug_task_limit`); logs go to stdout and `runner.log`.

## Workflow
1) Install: `bash install.sh` (creates `.venv`, installs from `requirements.txt`).
2) Run: `bash start.sh` → option 2 (benchmark) or 3 (report). You can also run `.venv/bin/python runner.py`.
3) Debug: set `debug: true` in `config.yaml` and optionally `debug_models`/`debug_categories`/`debug_task_limit`; runner logs model responses for quick inspection.
4) Report: after `runner.py` run `report.py` – it generates `report.md` (Markdown tables), computes accuracy, the `contains_all` metric, code auto-test stats, and saves extracted code snippets to `artifacts/`.

## Configuration (`config.yaml`)
- `ollama_host`: URL to the Ollama instance.
- `models`: list of models for full runs.
- `tests_dir`: directory with JSON sets (`instruction.json`, `code.json`, `polish.json`, ...).
- Debug: `debug` (bool), `debug_models`, `debug_categories`, `debug_task_limit` (number of tasks per category).

## Input Tests (`tests/*.json`)
Each entry: `prompt` plus scoring keys:
- `expected` (exact), `contains_all`, `contains_any`, `asserts` (alias for contains_all), `code_tests` (list of tests: `call`, `expected`; run on the generated code).

## Scoring and Report
- `metrics.py`: `score_task` combines criteria (exact/contains_all/contains_any/asserts) and can run simple `code_tests` on extracted code (pulled from ```python blocks).
- `report.py`: reads `results.json`, computes accuracy per model/category and overall, summarizes `contains_all` and `code_tests`, saves code to `artifacts/*.py`, and prints network/runtime errors.
