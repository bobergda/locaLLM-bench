# locaLLM-bench

## Overview
A minimal harness to benchmark small local LLMs served by Ollama or LM Studio. It loads test prompts, queries each model, and records raw responses and timings for later scoring and reporting.

## Features
- Loads configurable model list and test sets from disk.
- Calls Ollama or LM Studio (OpenAI-compatible) endpoints.
- Captures response text, eval counts (if returned), and wall-clock duration.
- Emits JSON results ready for scoring and reporting.

## Requirements
- Python 3.10+
- Ollama running locally at `http://localhost:11434` or LM Studio at `http://localhost:1234/v1`
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
provider: "ollama" # ollama|lmstudio
ollama_host: "http://localhost:11434"
lmstudio_host: "http://localhost:1234/v1"
models:
  - "llama3"
  - "mistral"
tests_dir: "tests"
include_test_sets: [ "code" ] # optional, empty = all
task_start_id: 0 # optional, 0-based start index per test set
task_limit: 4 # optional, how many tasks to run from task_start_id (0/null = to the end)
```
`provider` selects the backend, `ollama_host`/`lmstudio_host` point to the server, `models` lists model names to benchmark, and `tests_dir` sets the directory with JSON test files.
