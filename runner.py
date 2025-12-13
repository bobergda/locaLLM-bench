import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import ollama
import yaml


class ResultStreamer:
    """Stream results into a JSON array so progress is saved during the run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("w", encoding="utf-8")
        self._fh.write("[\n")
        self._first = True

    def write(self, row: Dict[str, Any]) -> None:
        if not self._first:
            self._fh.write(",\n")
        json.dump(row, self._fh, ensure_ascii=False)
        self._fh.flush()
        self._first = False

    def close(self) -> None:
        self._fh.write("\n]\n")
        self._fh.flush()
        self._fh.close()


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


def call_ollama(model: str, prompt: str, client: ollama.Client, logger: logging.Logger) -> Dict[str, Any]:
    payload = {"model": model, "prompt": prompt, "stream": False}
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
    streamer = ResultStreamer(stream_path) if stream_path else None
    try:
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
                        result = call_ollama(model, prompt, client, logger)
                    except Exception as exc:
                        logger.error("Request failed for model=%s category=%s task=%d: %s", model, category, idx, exc)
                        result = {"error": str(exc)}
                    if debug_mode and "response" in result:
                        text = (result.get("response") or "").strip()
                        logger.debug("Response [model=%s, cat=%s, task=%d]:\n%s", model, category, idx, text)
                    row = {
                        "model": model,
                        "category": category,
                        "task_id": idx,
                        "prompt": prompt,
                        **meta,
                        **result,
                    }
                    results.append(row)
                    if streamer:
                        streamer.write(row)
    finally:
        if streamer:
            streamer.close()
    return results


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    config_file = Path("config.yaml")
    cfg = load_config(config_file)
    debug_flag = bool(cfg.get("debug", False))
    logger = setup_logger(Path("runner.log"), debug=debug_flag)
    logger.info("Starting benchmark run%s", " (debug)" if debug_flag else "")
    output_path = Path("results.json")
    results = run_benchmark(config_file, debug=debug_flag, stream_path=output_path)
    logger.info("Saved %d results to %s", len(results), output_path)
