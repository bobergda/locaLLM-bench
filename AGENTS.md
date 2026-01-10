# AGENTS.md — agent instructions (short)

Goal: fast, repeatable benchmarking of Ollama- or LM Studio-served models with stable scoring/reporting.

## Quick commands
- Install: `bash install.sh`
- Activate venv: `source .venv/bin/activate`
- Benchmark: `python runner.py`
- Report: `python report.py`
- Required: use the `.venv` Python for all runs.
- Never install packages unless the venv is active.

## Key files
- `config.yaml` — Ollama host, models, test sets, debug options.
- `runner.py` — runs tasks, writes `results.json`.
- `metrics.py` — scoring + optional `code_tests`.
- `report.py` — aggregates into `report.md`.
- `tests/*.json` — prompts + scoring keys.

## Security (code execution)
`metrics.py` and `report.py` `exec()`/`eval()` model code.
Do not run on untrusted `results.json` or prompts.

## Validate
- `python -m py_compile runner.py metrics.py report.py`
- Quick smoke: set `task_limit: 1`, run `python runner.py`, then `python report.py`
