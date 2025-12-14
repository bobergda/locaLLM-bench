# locaLLM-bench

## Overview
A minimal harness to benchmark small local LLMs served by Ollama. It loads test prompts, queries each model, and records raw responses and timings for later scoring and reporting.

## Features
- Loads configurable model list and test sets from disk.
- Calls Ollama `/api/generate` without streaming to simplify timing.
- Captures response text, eval counts (if returned), and wall-clock duration.
- Emits JSON results ready for scoring and reporting.

## Requirements
- Python 3.10+
- Ollama running locally at `http://localhost:11434`
- Python packages: `ollama`, `PyYAML`

## Installation
```bash
./install.sh
```

## Usage
1. Adjust `config.yaml` (see below).
2. Add test files under `tests/` (examples in `tests/instruction.json`).
3. Run:
```bash
./run.sh
```

Security note: code-based scoring (`code_tests`) executes model-produced code. Enable it only if you trust `results.json`:
- set `unsafe_code_exec: true` in `config.yaml`

## Agent instructions
- See `AGENTS.md` for the prompt/instructions used by a coding agent to work in this repo.

## Configuration
Example `config.yaml`:
```yaml
ollama_host: "http://localhost:11434"
models:
  - "llama3"
  - "mistral"
tests_dir: "tests"
include_test_sets: [ "code" ] # optional, empty = all
task_start_id: 0 # optional, 0-based start index per test set
task_limit: 4 # optional, how many tasks to run from task_start_id (0/null = to the end)
```
`ollama_host` points to your Ollama instance, `models` lists model names to benchmark, and `tests_dir` sets the directory with JSON test files.
