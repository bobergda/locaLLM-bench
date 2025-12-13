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
- Python packages: `ollama`, `PyYAML`, `psutil`

## Installation
```bash
./install.sh
```

## Usage
1. Adjust `config.yaml` (see below).
2. Add test files under `tests/` (examples in `tests/instruction.json`).
3. Run:
```bash
./start.sh
```
4. Generate a report:
```bash
.venv/bin/python report.py
```

Security note: `report.py --unsafe-code-exec` will execute model-produced code for `code_tests`. Only enable this if you trust the results file.

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
```
`ollama_host` points to your Ollama instance, `models` lists model names to benchmark, and `tests_dir` sets the directory with JSON test files.
