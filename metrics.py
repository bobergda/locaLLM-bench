from collections import defaultdict
from typing import Any, Dict, Iterable, List


def score_exact(output: str, expected: str) -> bool:
    return output.strip() == expected.strip()


def score_contains_all(output: str, expected_parts: Iterable[str]) -> bool:
    text = output.lower()
    return all(part.lower() in text for part in expected_parts)


def score_contains_any(output: str, expected_parts: Iterable[str]) -> bool:
    text = output.lower()
    return any(part.lower() in text for part in expected_parts)


def score_task(task: Dict[str, Any], output: str) -> bool:
    if "expected" in task:
        return score_exact(output, task["expected"])
    if "contains_all" in task:
        return score_contains_all(output, task["contains_all"])
    if "contains_any" in task:
        return score_contains_any(output, task["contains_any"])
    return False


def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in results:
        model = row["model"]
        category = row["category"]
        output = row.get("response", "")
        if row.get("error"):
            totals[model][category] += 1
            continue
        task = {
            "expected": row.get("expected"),
            "contains_all": row.get("contains_all"),
            "contains_any": row.get("contains_any"),
        }
        # Remove None entries for cleaner logic
        task = {k: v for k, v in task.items() if v is not None}
        is_correct = score_task(task, output)
        totals[model][category] += 1
        if is_correct:
            correct[model][category] += 1

    accuracies: Dict[str, Dict[str, float]] = {}
    for model, categories in totals.items():
        accuracies[model] = {}
        for category, total in categories.items():
            if total == 0:
                accuracies[model][category] = 0.0
            else:
                accuracies[model][category] = correct[model][category] / total
    return accuracies
