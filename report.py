import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

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


def save_csv(accuracies: Dict[str, Dict[str, float]], path: Path) -> None:
    categories = sorted({cat for model_data in accuracies.values() for cat in model_data})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *categories, "overall"])
        for model, cats in accuracies.items():
            overall = sum(cats.values()) / len(categories) if categories else 0.0
            writer.writerow([model, *[f"{cats.get(cat, 0.0):.4f}" for cat in categories], f"{overall:.4f}"])


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
        filename = f"{model}_{category}_task{task_id}.py".replace("/", "_")
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


def main() -> None:
    results_path = Path("results.json")
    if not results_path.exists():
        raise FileNotFoundError("results.json not found. Run runner.py first.")

    results = load_results(results_path)
    accuracies = compute_accuracy(results)
    overall_acc = compute_overall_accuracy(results)
    contains_all_stats = compute_contains_all_stats(results)
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
        if errors:
            f.write("## Errors (first few)\n\n")
            for err in errors:
                f.write(f"- {err}\n")
            f.write("\n")

    save_csv(accuracies, Path("report.csv"))
    print(f"Wrote {report_path} and report.csv; saved {saved_codes} code artifact(s) to artifacts/")


if __name__ == "__main__":
    main()
