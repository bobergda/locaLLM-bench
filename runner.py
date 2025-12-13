import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import ollama
import yaml


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


def setup_logger(log_path: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("ollabench")
    if logger.handlers:
        # Update level if already configured
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return logger
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if gen_options:
        payload["options"] = gen_options
    start = time.perf_counter()
    logger.info("Calling model=%s", model)
    logger.debug("Request payload: %s", payload)
    data = client.generate(**payload)
    elapsed = time.perf_counter() - start
    logger.info("Completed model=%s in %.2fs", model, elapsed)
    #logger.debug("Raw response: %s", data)
    return {
        "response": data.get("response", ""),
        "eval_count": data.get("eval_count"),
        "total_duration_s": elapsed,
    }


def run_benchmark(config_path: Path, debug: bool = False, stream_path: Path | None = None) -> List[Dict[str, Any]]:
    logger = logging.getLogger("ollabench")
    cfg = load_config(config_path)
    tests_dir = Path(cfg.get("tests_dir", "tests"))
    host = cfg.get("ollama_host", "http://localhost:11434")
    debug_mode = bool(cfg.get("debug", debug))
    models = cfg.get("models", [])
    debug_models = cfg.get("debug_models") or []
    models_to_run = debug_models if debug_mode and debug_models else models
    debug_categories = cfg.get("debug_categories") or []
    debug_task_limit = cfg.get("debug_task_limit")
    gen_options = cfg.get("generate_options") or {}
    schema_version = int(cfg.get("results_schema_version", 1))
    run_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = run_started_at
    logger.info(
        "Loaded config: host=%s, models=%s, tests_dir=%s, debug=%s",
        host,
        models_to_run,
        tests_dir,
        debug_mode,
    )

    test_sets = load_tests(tests_dir)
    logger.info("Loaded test sets: %s", list(test_sets.keys()))
    client = ollama.Client(host=host)

    results: List[Dict[str, Any]] = []
    if stream_path:
        save_results_atomic(results, stream_path)
    for model in models_to_run:
        logger.info("Starting model: %s", model)
        for category, tasks in test_sets.items():
            if debug_mode and debug_categories and category not in debug_categories:
                logger.debug("Debug mode: skipping category %s", category)
                continue
            logger.info("  Category: %s (%d tasks)", category, len(tasks))
            tasks_iter = tasks
            if debug_mode and debug_task_limit:
                tasks_iter = tasks[: int(debug_task_limit)]
                logger.debug("Debug mode: limiting to first %s task(s) in %s", debug_task_limit, category)
            for idx, task in enumerate(tasks_iter):
                prompt = task["prompt"]
                logger.debug("Prompt: %s", prompt)
                meta = {k: v for k, v in task.items() if k != "prompt"}
                try:
                    result = call_ollama(model, prompt, client, logger, gen_options=gen_options)
                except Exception as exc:
                    logger.error("Request failed for model=%s category=%s task=%d: %s", model, category, idx, exc)
                    result = {"error": str(exc)}
                if debug_mode and "response" in result:
                    text = (result.get("response") or "").strip()
                    logger.debug("Response [model=%s, cat=%s, task=%d]:\n%s", model, category, idx, text)
                row = {
                    "results_schema_version": schema_version,
                    "run_id": run_id,
                    "run_started_at": run_started_at,
                    "ollama_host": host,
                    "generate_options": gen_options,
                    "model": model,
                    "category": category,
                    "task_id": idx,
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
    debug_flag = bool(cfg.get("debug", False))
    logger = setup_logger(Path("runner.log"), debug=debug_flag)
    logger.info("Starting benchmark run%s", " (debug)" if debug_flag else "")
    output_path = Path("results.json")
    results = run_benchmark(config_file, debug=debug_flag, stream_path=output_path)
    save_results_atomic(results, output_path)
    logger.info("Saved %d results to %s", len(results), output_path)
