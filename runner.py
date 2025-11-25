import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tests(tests_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    test_sets: Dict[str, List[Dict[str, Any]]] = {}
    for json_file in sorted(tests_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            test_sets[json_file.stem] = json.load(f)
    return test_sets


def call_ollama(model: str, prompt: str, host: str) -> Dict[str, Any]:
    payload = {"model": model, "prompt": prompt, "stream": False}
    start = time.perf_counter()
    response = requests.post(f"{host}/api/generate", json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    return {
        "response": data.get("response", ""),
        "eval_count": data.get("eval_count"),
        "total_duration_s": elapsed,
    }


def run_benchmark(config_path: Path) -> List[Dict[str, Any]]:
    cfg = load_config(config_path)
    tests_dir = Path(cfg.get("tests_dir", "tests"))
    host = cfg.get("ollama_host", "http://localhost:11434")
    models = cfg.get("models", [])
    test_sets = load_tests(tests_dir)

    results: List[Dict[str, Any]] = []
    for model in models:
        for category, tasks in test_sets.items():
            for idx, task in enumerate(tasks):
                prompt = task["prompt"]
                meta = {k: v for k, v in task.items() if k != "prompt"}
                try:
                    result = call_ollama(model, prompt, host)
                except requests.RequestException as exc:
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
    config_file = Path("config.yaml")
    results = run_benchmark(config_file)
    output_path = Path("results.json")
    save_results(results, output_path)
    print(f"Saved {len(results)} results to {output_path}")
