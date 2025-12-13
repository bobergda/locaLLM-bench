# AGENTS.md — agent instructions

This file defines how coding agents (e.g., Codex/LLMs) should work in **LocaLLM Bench**. The goal is fast, repeatable benchmarking of Ollama-served models, plus stable scoring and reporting.

## What this project is
- A minimal harness to benchmark local LLMs via Ollama: compares output quality and response time on simple task sets.
- Saves raw results (`results.json`) and generates a report (`report.md`) with accuracy and helper metrics.

## Quick start (commands)
- Install: `bash install.sh` (creates `.venv`, installs from `requirements.txt`)
- Benchmark: `bash start.sh` → option 2, or `python3 runner.py` / `.venv/bin/python runner.py`
- Report: `python3 report.py` / `.venv/bin/python report.py`

## Repo map (key files)
- `config.yaml` — Ollama host, model list, tests directory, debug options.
- `runner.py` — runs tasks, calls `ollama` (`client.generate(stream=False)`), writes `results.json` and logs.
- `metrics.py` — scoring (`expected`, `contains_all`, `contains_any`, `asserts`) + optional `code_tests`.
- `report.py` — aggregates `results.json` into `report.md`, saves extracted snippets into `artifacts/`.
- `tests/*.json` — task definitions and scoring keys.

## Test format (`tests/*.json`)
Each entry is an object with `prompt` plus optional scoring keys:
- `expected` — exact match (after `strip()`).
- `contains_all` / `asserts` — all phrases must be present (case-insensitive).
- `contains_any` — any phrase is sufficient.
- `code_tests` — tests executed against code extracted from the model output (prefer ```python fenced blocks).
  - Format: `{"assert": "expr"}` or legacy `{"call": "expr", "expected": ...}`.

## Critical security (code execution)
- `metrics.py:run_code_tests()` and `report.py:run_code_tests_verbose()` use `exec()` and `eval()` on model-generated code.
- Do not run benchmarks/reports on **untrusted** `results.json` or prompts that may induce malicious code.
- If you change `code_tests`, do not expand the attack surface (e.g., avoid adding “convenience” system imports, file/network access, or shell execution).

## Change rules (best practices)
- Keep changes minimal and intentional; avoid drive-by refactors.
- Preserve the `results.json` schema (any format change must update `report.py` and be documented in `README.md`).
- Don’t add dependencies without a clear reason; if you do, update `requirements.txt` and keep Python 3.12 compatibility.
- Keep metrics/reporting deterministic for the same `results.json`.
- Treat `tests/*.json` as an API: keep backward compatibility or migrate all files consistently.

## Debugging and cost control
- Use `config.yaml`: `debug`, `debug_models`, `debug_categories`, `debug_task_limit`.
- Prefer small, fast runs (task limits) during iteration.

## How to validate changes
- Minimum: `python3 -m py_compile runner.py metrics.py report.py`
- Quick smoke test: set `debug_task_limit: 1`, run `python3 runner.py`, then `python3 report.py`
