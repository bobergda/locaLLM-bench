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


def setup_logger(log_path: Path, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ollabench")
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    logger.setLevel(level)

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


def run_benchmark(config_path: Path, stream_path: Path | None = None) -> List[Dict[str, Any]]:
    logger = logging.getLogger("ollabench")
    cfg = load_config(config_path)
    tests_dir = Path(cfg.get("tests_dir", "tests"))
    host = cfg.get("ollama_host", "http://localhost:11434")
    models = cfg.get("models") or []
    if not isinstance(models, list):
        raise TypeError("config.yaml: 'models' must be a list of model names")
    models_to_run = models
    include_test_sets = cfg.get("include_test_sets") or cfg.get("test_sets") or []
    task_limit = cfg.get("task_limit")
    gen_options = cfg.get("generate_options") or {}
    schema_version = int(cfg.get("results_schema_version", 1))
    run_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = run_started_at
    logger.info(
        "Loaded config: host=%s, models=%s, tests_dir=%s",
        host,
        models_to_run,
        tests_dir,
    )

    test_sets = load_tests(tests_dir)
    logger.info("Loaded test sets: %s", list(test_sets.keys()))
    client = ollama.Client(host=host)

    results: List[Dict[str, Any]] = []
    for model in models_to_run:
        logger.info("Starting model: %s", model)
        for test_set, tasks in test_sets.items():
            if include_test_sets and test_set not in include_test_sets:
                logger.info("Skipping test_set %s (not selected)", test_set)
                continue
            logger.info("  Test set: %s (%d tasks)", test_set, len(tasks))
            tasks_iter = tasks
            if task_limit:
                tasks_iter = tasks[: int(task_limit)]
                logger.info("Limiting to first %s task(s) in %s", task_limit, test_set)
            for idx, task in enumerate(tasks_iter):
                prompt = task["prompt"]
                logger.debug("Prompt: %s", prompt)
                meta = {k: v for k, v in task.items() if k != "prompt"}
                try:
                    result = call_ollama(model, prompt, client, logger, gen_options=gen_options)
                except Exception as exc:
                    logger.error("Request failed for model=%s test_set=%s task=%d: %s", model, test_set, idx, exc)
                    result = {"error": str(exc)}
                if "response" in result:
                    text = (result.get("response") or "").strip()
                    logger.debug("Response [model=%s, test_set=%s, task=%d]:\n%s", model, test_set, idx, text)
                row = {
                    "results_schema_version": schema_version,
                    "run_id": run_id,
                    "run_started_at": run_started_at,
                    "ollama_host": host,
                    "generate_options": gen_options,
                    "model": model,
                    "test_set": test_set,
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
    log_level = str(cfg.get("log_level", "INFO"))
    logger = setup_logger(Path("runner.log"), log_level=log_level)
    logger.info("Starting benchmark run (log_level=%s)", log_level.upper())
    output_path = Path("results.json")
    results = run_benchmark(config_file, stream_path=output_path)
    save_results_atomic(results, output_path)
    logger.info("Saved %d results to %s", len(results), output_path)
