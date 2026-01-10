import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, List

import lmstudio
import ollama
import yaml


# ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def print_streaming_chunk(chunk: str, is_first: bool = False) -> None:
    """Print a streaming response chunk."""
    if is_first:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[Response]{Colors.RESET} ", end="", flush=True)
    print(chunk, end="", flush=True)


def save_results_atomic(results: List[Dict[str, Any]], path: Path) -> None:
    """
    Rewrite the entire results file after each response.
    This keeps results.json always valid JSON even if the run is interrupted.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def setup_logger(log_path: Path, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ollabench")
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    logger.setLevel(level)

    # Only log to file - console will show pretty messages via print()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 10_000):
        candidate = Path(f"{path}_{i:04d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to create unique output directory near: {path}")


def make_run_dir(cfg: Dict[str, Any]) -> Path:
    """
    Creates and returns a per-run output directory under a shared root.
    Default: results/runs/YYYYmmdd-HHMMSSZ/
    """
    root = Path(str(cfg.get("output_root_dir") or "results/runs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = _ensure_unique_dir(root / stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_latest_pointer(run_dir: Path) -> None:
    """
    Best-effort "latest run" pointer:
    - results/latest -> symlink to run dir (if possible)
    - results/latest.txt with run dir path (fallback)
    """
    base = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
    latest_link = base / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_dir() and not latest_link.is_symlink():
                rmtree(latest_link)
            else:
                latest_link.unlink()
        latest_link.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception:
        (base / "latest.txt").write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tests(tests_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    test_sets: Dict[str, List[Dict[str, Any]]] = {}
    for json_file in sorted(tests_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            test_sets[json_file.stem] = json.load(f)
    return test_sets


def call_ollama(
    model: str,
    prompt: str,
    client: ollama.Client,
    logger: logging.Logger,
    gen_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True, "think": True}
    if gen_options:
        payload["options"] = gen_options

    start = time.perf_counter()
    logger.info("Calling model=%s (streaming=True, show_thinking=True)", model)
    logger.debug("Request payload: %s", payload)

    # Streaming mode: collect chunks and display live
    response_chunks = []
    thinking_chunks = []
    eval_count = None
    is_first_chunk = True
    in_thinking = False

    logger.info("Starting streaming response...")
    print(f"\n{Colors.YELLOW}[Streaming from {model}...]{Colors.RESET}")

    try:
        for chunk in client.generate(**payload):
            # Check for native thinking support from API
            thinking_text = (
                chunk.get("thinking")
                or chunk.get("message", {}).get("thinking")
            )

            if thinking_text:
                thinking_chunks.append(thinking_text)
                if not in_thinking:
                    print(f"\n{Colors.CYAN}{Colors.BOLD}🧠 Model thinking...{Colors.RESET}")
                    in_thinking = True
                print(f"{Colors.DIM}{thinking_text}{Colors.RESET}", end="", flush=True)
                continue

            if in_thinking and chunk.get("response"):
                print()  # Newline after thinking
                in_thinking = False

            # Handle regular response text
            chunk_text = chunk.get("response", "")
            if chunk_text:
                response_chunks.append(chunk_text)
                print_streaming_chunk(chunk_text, is_first=is_first_chunk)
                is_first_chunk = False

            # Capture eval_count from final chunk
            if chunk.get("done", False):
                eval_count = chunk.get("eval_count")

        if in_thinking:
            print()  # Newline if we ended in thinking mode
        print()  # Newline after streaming

    except Exception as e:
        logger.error("Streaming error: %s", e)
        raise

    elapsed = time.perf_counter() - start
    response_text = "".join(response_chunks)
    native_thinking = "".join(thinking_chunks) if thinking_chunks else None

    # Only use native thinking from API
    thinking_blocks = []
    if native_thinking:
        thinking_blocks = [native_thinking]
        logger.info("Captured native thinking from API (%d chars)", len(native_thinking))

    logger.info("Streaming completed model=%s in %.2fs (tokens=%s)", model, elapsed, eval_count or "N/A")

    return {
        "response": response_text,
        "thinking_blocks": thinking_blocks if thinking_blocks else None,
        "eval_count": eval_count,
        "total_duration_s": elapsed,
    }


def _normalize_lmstudio_host(host: str | None) -> str | None:
    raw = (host or "").strip()
    if not raw:
        return None
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme:
        return parsed.netloc or None
    return raw


def _build_lmstudio_prediction_config(gen_options: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not gen_options:
        return None
    config: Dict[str, Any] = {}
    if "temperature" in gen_options:
        config["temperature"] = gen_options["temperature"]
    if "top_p" in gen_options:
        config["top_p_sampling"] = gen_options["top_p"]
    if "top_k" in gen_options:
        config["top_k_sampling"] = gen_options["top_k"]
    if "repeat_penalty" in gen_options:
        config["repeat_penalty"] = gen_options["repeat_penalty"]
    if "min_p" in gen_options:
        config["min_p_sampling"] = gen_options["min_p"]
    if "stop" in gen_options:
        stop_val = gen_options["stop"]
        config["stop_strings"] = stop_val if isinstance(stop_val, list) else [str(stop_val)]
    if "max_tokens" in gen_options:
        config["max_tokens"] = gen_options["max_tokens"]
    elif "num_predict" in gen_options:
        config["max_tokens"] = gen_options["num_predict"]
    if "reasoning_parsing" in gen_options:
        config["reasoning_parsing"] = gen_options["reasoning_parsing"]
    return config or None


def call_lmstudio(
    model: str,
    prompt: str,
    client: lmstudio.Client,
    logger: logging.Logger,
    gen_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    def _get_value(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _build_openai_options(options: Dict[str, Any] | None) -> Dict[str, Any]:
        if not options:
            return {}
        payload: Dict[str, Any] = {}
        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
        if "max_tokens" in options:
            payload["max_tokens"] = options["max_tokens"]
        elif "num_predict" in options:
            payload["max_tokens"] = options["num_predict"]
        if "stop" in options:
            payload["stop"] = options["stop"]
        if "seed" in options:
            payload["seed"] = options["seed"]
        return payload

    def _get_delta(choice: Any) -> Any:
        delta = _get_value(choice, "delta")
        if delta is None:
            delta = _get_value(choice, "message")
        return delta or {}

    def _get_delta_text(delta: Any) -> str:
        if not delta:
            return ""
        value = _get_value(delta, "content")
        if value is None:
            value = _get_value(delta, "text")
        if value is None:
            return ""
        return str(value)

    def _get_delta_reasoning(delta: Any) -> str:
        if not delta:
            return ""
        value = _get_value(delta, "reasoning_content")
        if value is None:
            value = _get_value(delta, "reasoningContent")
        if value is None:
            return ""
        return str(value)

    def _iter_sse_events(response: Any) -> Any:
        for raw_line in response:
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8", errors="replace")
            except Exception:
                continue
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event

    def _get_stream_url() -> str:
        host = None
        try:
            host = client.api_host
        except Exception:
            host = getattr(client, "_api_host", None)
        raw = str(host or "127.0.0.1:1234").strip()
        if "://" not in raw:
            raw = f"http://{raw}"
        parsed = urllib.parse.urlparse(raw)
        base = raw.rstrip("/")
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    payload.update(_build_openai_options(gen_options))

    start = time.perf_counter()
    logger.info("Calling LM Studio model=%s (streaming=True)", model)
    logger.debug("LM Studio request payload: %s", payload)

    response_chunks: list[str] = []
    thinking_chunks: list[str] = []
    is_first_chunk = True
    in_thinking = False
    eval_count: int | None = None

    print(f"\n{Colors.YELLOW}[Streaming from {model}...]{Colors.RESET}")

    request = urllib.request.Request(
        _get_stream_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for event in _iter_sse_events(response):
                usage = event.get("usage") if isinstance(event, dict) else None
                if isinstance(usage, dict):
                    eval_count = usage.get("completion_tokens") or usage.get("total_tokens") or eval_count

                for choice in event.get("choices", []) if isinstance(event, dict) else []:
                    delta = _get_delta(choice)
                    reasoning_text = _get_delta_reasoning(delta)
                    if reasoning_text:
                        thinking_chunks.append(reasoning_text)
                        if not in_thinking:
                            print(f"\n{Colors.CYAN}{Colors.BOLD}🧠 Model thinking...{Colors.RESET}")
                            in_thinking = True
                        print(f"{Colors.DIM}{reasoning_text}{Colors.RESET}", end="", flush=True)

                    content_text = _get_delta_text(delta)
                    if content_text:
                        if in_thinking:
                            print()
                            in_thinking = False
                        response_chunks.append(content_text)
                        print_streaming_chunk(content_text, is_first=is_first_chunk)
                        is_first_chunk = False
    except Exception as exc:
        logger.error("LM Studio streaming error: %s", exc)

    if in_thinking:
        print()
    print()

    elapsed = time.perf_counter() - start
    response_text = "".join(response_chunks)

    logger.info("LM Studio completed model=%s in %.2fs (tokens=%s)", model, elapsed, eval_count or "N/A")

    return {
        "response": response_text,
        "thinking_blocks": ["".join(thinking_chunks)] if thinking_chunks else None,
        "eval_count": eval_count,
        "total_duration_s": elapsed,
    }


def run_benchmark(config_path: Path, stream_path: Path | None = None) -> List[Dict[str, Any]]:
    logger = logging.getLogger("ollabench")
    cfg = load_config(config_path)
    tests_dir = Path(cfg.get("tests_dir", "tests"))
    provider = str(cfg.get("provider", "ollama")).strip().lower()
    if provider not in {"ollama", "lmstudio"}:
        raise ValueError("config.yaml: 'provider' must be 'ollama' or 'lmstudio'")
    host = (
        cfg.get("ollama_host", "http://localhost:11434")
        if provider == "ollama"
        else cfg.get("lmstudio_host", "http://localhost:1234")
    )
    lmstudio_reasoning_parsing = cfg.get("lmstudio_reasoning_parsing")
    models = cfg.get("models") or []
    if not isinstance(models, list):
        raise TypeError("config.yaml: 'models' must be a list of model names")
    models_to_run = models
    include_test_sets = cfg.get("include_test_sets") or cfg.get("test_sets") or []
    task_start_id = cfg.get("task_start_id")
    task_limit = cfg.get("task_limit")
    gen_options = cfg.get("generate_options") or {}
    if provider == "lmstudio" and lmstudio_reasoning_parsing and isinstance(gen_options, dict):
        if "reasoning_parsing" not in gen_options:
            gen_options = {**gen_options, "reasoning_parsing": lmstudio_reasoning_parsing}
    schema_version = int(cfg.get("results_schema_version", 1))
    run_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = run_started_at
    logger.info(
        "Loaded config: provider=%s, host=%s, models=%s, tests_dir=%s",
        provider,
        host,
        models_to_run,
        tests_dir,
    )

    test_sets = load_tests(tests_dir)
    logger.info("Loaded test sets: %s", list(test_sets.keys()))

    # Display benchmark start information
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}   BENCHMARK STARTED{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}Provider:{Colors.RESET} {provider}")
    print(f"{Colors.BOLD}Models:{Colors.RESET} {', '.join(models_to_run)}")
    print(f"{Colors.BOLD}Test sets:{Colors.RESET} {', '.join(test_sets.keys())}")
    print(f"{Colors.BOLD}Streaming:{Colors.RESET} Enabled")
    print(f"{Colors.BOLD}Thinking:{Colors.RESET} {'Enabled' if provider == 'ollama' else 'Disabled'}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'═' * 70}{Colors.RESET}\n")

    client = None
    if provider == "ollama":
        client = ollama.Client(host=host)
    else:
        lmstudio_host = _normalize_lmstudio_host(host)
        client = lmstudio.Client(api_host=lmstudio_host)

    results: List[Dict[str, Any]] = []
    for model in models_to_run:
        logger.info("Starting model: %s", model)
        print(f"\n{Colors.YELLOW}{Colors.BOLD}{'─' * 70}{Colors.RESET}")
        print(f"{Colors.YELLOW}{Colors.BOLD}Processing model: {model}{Colors.RESET}")
        print(f"{Colors.YELLOW}{Colors.BOLD}{'─' * 70}{Colors.RESET}")

        for test_set, tasks in test_sets.items():
            if include_test_sets and test_set not in include_test_sets:
                logger.info("Skipping test_set %s (not selected)", test_set)
                print(f"{Colors.DIM}Skipping test set: {test_set} (not selected){Colors.RESET}")
                continue
            logger.info("  Test set: %s (%d tasks)", test_set, len(tasks))
            print(f"\n{Colors.CYAN}Test set: {test_set} ({len(tasks)} tasks){Colors.RESET}")

            try:
                start_id = int(task_start_id) if task_start_id is not None else 0
            except Exception as exc:
                raise TypeError("config.yaml: 'task_start_id' must be an integer or null") from exc
            if start_id < 0 or start_id > len(tasks):
                raise ValueError(
                    f"config.yaml: task_start_id={start_id} out of range for test_set={test_set} (0..{len(tasks)})"
                )

            end_id = len(tasks)
            if task_limit:
                end_id = min(len(tasks), start_id + int(task_limit))

            if start_id:
                logger.info("Starting from task %d in %s", start_id, test_set)
                print(f"{Colors.DIM}Starting from task {start_id}{Colors.RESET}")
            if task_limit:
                logger.info("Limiting to %s task(s) in %s", int(task_limit), test_set)
                print(f"{Colors.DIM}Limiting to {int(task_limit)} task(s){Colors.RESET}")

            for task_id in range(start_id, end_id):
                task = tasks[task_id]
                prompt = task["prompt"]
                logger.debug("Prompt: %s", prompt)

                # Display task header for visibility
                print(f"\n{Colors.BOLD}{'═' * 70}{Colors.RESET}")
                print(f"{Colors.YELLOW}{Colors.BOLD}Task {task_id + 1}/{len(tasks)}{Colors.RESET} "
                      f"[{test_set}] - Model: {Colors.CYAN}{model}{Colors.RESET}")
                print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")
                print(f"{Colors.DIM}Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}{Colors.RESET}")
                print(f"{Colors.BOLD}{'─' * 70}{Colors.RESET}")

                meta = {k: v for k, v in task.items() if k != "prompt"}
                try:
                    if provider == "ollama":
                        result = call_ollama(
                            model,
                            prompt,
                            client,
                            logger,
                            gen_options=gen_options,
                        )
                    else:
                        result = call_lmstudio(
                            model,
                            prompt,
                            client,
                            logger,
                            gen_options=gen_options,
                        )
                except Exception as exc:
                    logger.error("Request failed for model=%s test_set=%s task=%d: %s", model, test_set, task_id, exc)
                    print(f"\n{Colors.RED}⚠ Error occurred: {str(exc)}{Colors.RESET}")
                    if provider == "lmstudio":
                        print(f"{Colors.DIM}Hint: LM Studio must be running with Local Server enabled and `lmstudio_host` reachable.{Colors.RESET}")
                    else:
                        print(f"{Colors.DIM}Hint: Ollama must be running and `ollama_host` reachable.{Colors.RESET}")
                    result = {"error": str(exc)}
                if "response" in result:
                    text = (result.get("response") or "").strip()
                    logger.debug("Response [model=%s, test_set=%s, task=%d]:\n%s", model, test_set, task_id, text)
                    # Show summary
                    thinking_count = len(result.get("thinking_blocks") or [])
                    if thinking_count > 0:
                        print(f"\n{Colors.GREEN}✓ Task completed with {thinking_count} thinking block(s){Colors.RESET}")
                    else:
                        print(f"\n{Colors.GREEN}✓ Task completed{Colors.RESET}")
                row = {
                    "results_schema_version": schema_version,
                    "run_id": run_id,
                    "run_started_at": run_started_at,
                    "provider": provider,
                    "provider_host": host,
                    "ollama_host": host if provider == "ollama" else None,
                    "lmstudio_host": host if provider == "lmstudio" else None,
                    "generate_options": gen_options,
                    "model": model,
                    "test_set": test_set,
                    "task_id": task_id,
                    "prompt": prompt,
                    **meta,
                    **result,
                }
                results.append(row)
                if stream_path:
                    save_results_atomic(results, stream_path)
    return results


if __name__ == "__main__":
    config_file = Path("config.yaml")
    cfg = load_config(config_file)
    run_dir = make_run_dir(cfg)
    log_level = str(cfg.get("log_level", "INFO"))
    logger = setup_logger(run_dir / "runner.log", log_level=log_level)
    logger.info("Starting benchmark run (log_level=%s)", log_level.upper())

    print(f"\n{Colors.GREEN}{Colors.BOLD}Starting benchmark...{Colors.RESET}")
    print(f"{Colors.DIM}Run directory: {run_dir}{Colors.RESET}")
    print(f"{Colors.DIM}Detailed logs: {run_dir / 'runner.log'}{Colors.RESET}\n")

    output_path = run_dir / "results.json"
    results = run_benchmark(config_file, stream_path=output_path)
    save_results_atomic(results, output_path)
    logger.info("Saved %d results to %s", len(results), output_path)
    write_latest_pointer(run_dir)
    logger.info("Run directory: %s", run_dir)

    error_rows = [row for row in results if row.get("error")]
    if error_rows:
        print(f"\n{Colors.RED}{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}   ERRORS DETECTED{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}Failed tasks:{Colors.RESET} {len(error_rows)}")
        last_error = str(error_rows[-1].get("error") or "").strip()
        if last_error:
            print(f"{Colors.BOLD}Last error:{Colors.RESET} {last_error}")
        provider = str(cfg.get("provider", "ollama")).strip().lower()
        if provider == "lmstudio":
            print(f"{Colors.DIM}Check LM Studio is running and `lmstudio_host` is reachable.{Colors.RESET}")
        else:
            print(f"{Colors.DIM}Check Ollama is running and `ollama_host` is reachable.{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}{'═' * 70}{Colors.RESET}\n")

    print(f"\n{Colors.GREEN}{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.GREEN}{Colors.BOLD}   BENCHMARK COMPLETED{Colors.RESET}")
    print(f"{Colors.GREEN}{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}Results:{Colors.RESET} {len(results)} tasks completed")
    print(f"{Colors.BOLD}Output:{Colors.RESET} {output_path}")
    print(f"{Colors.BOLD}Logs:{Colors.RESET} {run_dir / 'runner.log'}")
    print(f"{Colors.GREEN}{Colors.BOLD}{'═' * 70}{Colors.RESET}\n")
