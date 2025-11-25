import csv
import json
from pathlib import Path
from typing import Dict, List

from metrics import compute_accuracy


def load_results(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_markdown_table(accuracies: Dict[str, Dict[str, float]]) -> str:
    # Collect all categories to build header
    categories = sorted({cat for model_data in accuracies.values() for cat in model_data})
    header = ["Model"] + categories
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for model, cats in accuracies.items():
        row = [model] + [f"{cats.get(cat, 0.0):.2f}" for cat in categories]
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def save_csv(accuracies: Dict[str, Dict[str, float]], path: Path) -> None:
    categories = sorted({cat for model_data in accuracies.values() for cat in model_data})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *categories])
        for model, cats in accuracies.items():
            writer.writerow([model, *[f"{cats.get(cat, 0.0):.4f}" for cat in categories]])


def main() -> None:
    results_path = Path("results.json")
    if not results_path.exists():
        raise FileNotFoundError("results.json not found. Run runner.py first.")

    results = load_results(results_path)
    accuracies = compute_accuracy(results)

    md_table = format_markdown_table(accuracies)
    report_path = Path("report.md")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Report\n\n")
        f.write("## Accuracy\n\n")
        f.write(md_table)
        f.write("\n")

    save_csv(accuracies, Path("report.csv"))
    print(f"Wrote {report_path} and report.csv")


if __name__ == "__main__":
    main()
