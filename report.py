import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from metrics import compute_accuracy, extract_python_code, score_contains_all, score_task


def load_results(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_markdown_table(accuracies: Dict[str, Dict[str, float]], overall: Dict[str, float]) -> str:
    categories = sorted({cat for model_data in accuracies.values() for cat in model_data})
    header = ["Model"] + categories + ["Overall"]
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for model, cats in accuracies.items():
        row = [model] + [f"{cats.get(cat, 0.0):.2f}" for cat in categories] + [f"{overall.get(model, 0.0):.2f}"]
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def format_contains_all_table(stats: Dict[str, Dict[str, Tuple[int, int]]]) -> str:
    categories = sorted({cat for model_data in stats.values() for cat in model_data})
    header = ["Model"] + categories
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for model, cats in stats.items():
        row_vals = []
        for cat in categories:
            passed, total = cats.get(cat, (0, 0))
            ratio = 0 if total == 0 else passed / total
            row_vals.append(f"{passed}/{total} ({ratio:.2f})")
        lines.append("|" + "|".join([model] + row_vals) + "|")
    return "\n".join(lines)


def compute_contains_all_stats(results: List[Dict]) -> Dict[str, Dict[str, Tuple[int, int]]]:
    stats: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for row in results:
        if row.get("error"):
            continue
        parts = []
        if row.get("contains_all"):
            parts.extend(row["contains_all"])
        if row.get("asserts"):
            parts.extend(row["asserts"])
        if not parts:
            continue
        model = row["model"]
        category = row["category"]
        output = row.get("response", "")
        is_ok = score_contains_all(output, parts)
        if model not in stats:
            stats[model] = {}
        passed, total = stats[model].get(category, (0, 0))
        stats[model][category] = (passed + (1 if is_ok else 0), total + 1)
    return stats


def run_code_tests_verbose(output: str, tests: List[Dict[str, Any]]) -> bool:
    code = extract_python_code(output)
    if not code:
        print("  no code extracted (FAIL)")
        return False
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as exc:
        print(f"  exec failed with {exc} (FAIL)")
        return False
    all_ok = True
    for test in tests:
        if "assert" in test:
            expr = test.get("assert")
            if not expr:
                print("  missing assert expression (FAIL)")
                all_ok = False
                continue
            try:
                result = bool(eval(expr, namespace))
                print(f"  assert {expr} ({'OK' if result else 'FAIL'})")
                if not result:
                    all_ok = False
            except Exception as exc:
                print(f"  {expr} raised {exc} (FAIL)")
                all_ok = False
            continue
        expr = test.get("call")
        expected = test.get("expected")
        if not expr:
            print("  missing call expression (FAIL)")
            all_ok = False
            continue
        try:
            value = eval(expr, namespace)
            print(f"  {expr} -> {value} (expected {expected}) ({'OK' if value == expected else 'FAIL'})")
            if value != expected:
                all_ok = False
        except Exception as exc:
            print(f"  {expr} raised {exc} (FAIL)")
            all_ok = False
    return all_ok


def save_code_artifacts(results: List[Dict], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for row in results:
        resp = row.get("response")
        if not resp:
            continue
        category = row.get("category", "")
        if category != "code" and "```" not in resp:
            continue
        model = row.get("model", "unknown")
        task_id = row.get("task_id", 0)
        code = extract_python_code(resp)
        if not code:
            continue
        filename = f"{model}_{category}_task{task_id}.py".replace("/", "_").replace(":", "_")
        (output_dir / filename).write_text(code, encoding="utf-8")
        saved += 1
    return saved


def compute_overall_accuracy(results: List[Dict]) -> Dict[str, float]:
    totals: Dict[str, int] = {}
    correct: Dict[str, int] = {}
    for row in results:
        if row.get("error"):
            continue
        model = row["model"]
        output = row.get("response", "")
        task = {
            "expected": row.get("expected"),
            "contains_all": row.get("contains_all"),
            "contains_any": row.get("contains_any"),
        }
        task = {k: v for k, v in task.items() if v is not None}
        ok = score_task(task, output)
        totals[model] = totals.get(model, 0) + 1
        if ok:
            correct[model] = correct.get(model, 0) + 1
    return {m: (correct.get(m, 0) / totals[m]) if totals.get(m, 0) else 0.0 for m in totals}


def collect_errors(results: List[Dict], limit: int = 5) -> List[str]:
    errors = []
    for row in results:
        if row.get("error"):
            errors.append(f"{row.get('model','?')} / {row.get('category','?')} / task {row.get('task_id','?')}: {row.get('error')}")
            if len(errors) >= limit:
                break
    return errors


def compute_code_test_stats(results: List[Dict], limit_failures: int = 5) -> Tuple[Dict[str, Dict[str, Tuple[int, int]]], List[str]]:
    stats: Dict[str, Dict[str, Tuple[int, int]]] = {}
    failures: List[str] = []
    for row in results:
        tests = row.get("code_tests")
        if not tests or row.get("error"):
            continue
        model = row["model"]
        category = row["category"]
        prompt_preview = (row.get("prompt", "")[:80] + "...") if row.get("prompt") else ""
        print(f"Running test ({model}): {prompt_preview}")
        ok = run_code_tests_verbose(row.get("response", ""), tests)
        if not ok and len(failures) < limit_failures:
            failures.append(f"{model}/{category}/task{row.get('task_id','?')} failed code tests")
        if model not in stats:
            stats[model] = {}
        passed, total = stats[model].get(category, (0, 0))
        stats[model][category] = (passed + (1 if ok else 0), total + 1)
    return stats, failures


def main() -> None:
    results_path = Path("results.json")
    if not results_path.exists():
        raise FileNotFoundError("results.json not found. Run runner.py first.")

    results = load_results(results_path)
    accuracies = compute_accuracy(results)
    overall_acc = compute_overall_accuracy(results)
    contains_all_stats = compute_contains_all_stats(results)
    code_test_stats, code_test_failures = compute_code_test_stats(results)
    saved_codes = save_code_artifacts(results, Path("artifacts"))
    errors = collect_errors(results)
    models = sorted({r["model"] for r in results})
    categories = sorted({r["category"] for r in results})
    total = len(results)
    error_count = sum(1 for r in results if r.get("error"))

    report_path = Path("report.md")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Models: {', '.join(models)}\n")
        f.write(f"- Categories: {', '.join(categories)}\n")
        f.write(f"- Total results: {total} (errors: {error_count})\n")
        f.write(f"- Extracted Python snippets: {saved_codes} file(s) in artifacts/\n\n")
        f.write("## Accuracy\n\n")
        f.write(format_markdown_table(accuracies, overall_acc))
        f.write("\n\n")
        if contains_all_stats:
            f.write("## Contains_all checks\n\n")
            f.write(format_contains_all_table(contains_all_stats))
            f.write("\n\n")
        if code_test_stats:
            f.write("## Code tests\n\n")
            f.write(format_contains_all_table(code_test_stats))
            f.write("\n\n")
        if code_test_failures:
            f.write("## Code test failures (first few)\n\n")
            for fail in code_test_failures:
                f.write(f"- {fail}\n")
            f.write("\n")
        if errors:
            f.write("## Errors (first few)\n\n")
            for err in errors:
                f.write(f"- {err}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
