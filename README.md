# ollabench

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
- Python packages: `requests`, `pyyaml`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install requests pyyaml
```

## Usage
1. Adjust `config.yaml` (see below).
2. Add test files under `tests/` (examples in `tests/instruction.json`).
3. Run:
```bash
python runner.py
```
4. Review `results.json` for collected outputs.

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
