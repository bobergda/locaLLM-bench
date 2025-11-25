import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ollabench")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
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


def call_ollama(model: str, prompt: str, host: str, logger: logging.Logger) -> Dict[str, Any]:
    payload = {"model": model, "prompt": prompt, "stream": False}
    start = time.perf_counter()
    logger.info("Calling model=%s", model)
    response = requests.post(f"{host}/api/generate", json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    logger.info("Completed model=%s in %.2fs", model, elapsed)
    return {
        "response": data.get("response", ""),
        "eval_count": data.get("eval_count"),
        "total_duration_s": elapsed,
    }


def run_benchmark(config_path: Path) -> List[Dict[str, Any]]:
    logger = logging.getLogger("ollabench")
    cfg = load_config(config_path)
    tests_dir = Path(cfg.get("tests_dir", "tests"))
    host = cfg.get("ollama_host", "http://localhost:11434")
    models = cfg.get("models", [])
    logger.info("Loaded config: host=%s, models=%s, tests_dir=%s", host, models, tests_dir)

    test_sets = load_tests(tests_dir)
    logger.info("Loaded test sets: %s", list(test_sets.keys()))

    results: List[Dict[str, Any]] = []
    for model in models:
        logger.info("Starting model: %s", model)
        for category, tasks in test_sets.items():
            logger.info("  Category: %s (%d tasks)", category, len(tasks))
            for idx, task in enumerate(tasks):
                prompt = task["prompt"]
                meta = {k: v for k, v in task.items() if k != "prompt"}
                try:
                    result = call_ollama(model, prompt, host, logger)
                except requests.RequestException as exc:
                    logger.error("Request failed for model=%s category=%s task=%d: %s", model, category, idx, exc)
                    result = {"error": str(exc)}
                results.append(
                    {
                        "model": model,
                        "category": category,
                        "task_id": idx,
                        "prompt": prompt,
                        **meta,
                        **result,
                    }
                )
    return results


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logger = setup_logger(Path("runner.log"))
    config_file = Path("config.yaml")
    logger.info("Starting benchmark run")
    results = run_benchmark(config_file)
    output_path = Path("results.json")
    save_results(results, output_path)
    logger.info("Saved %d results to %s", len(results), output_path)
