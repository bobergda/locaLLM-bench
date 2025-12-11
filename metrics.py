import re
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


def extract_python_code(text: str) -> str:
    fenced = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    generic = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if generic:
        return generic[0].strip()
    return text.strip()


def _eval_assert(namespace: Dict[str, Any], expr: str) -> bool:
    try:
        return bool(eval(expr, namespace))
    except Exception:
        return False


def run_code_tests(output: str, tests: List[Dict[str, Any]]) -> bool:
    """
    Execute model-produced code and run simple assert-style checks.
    Tests can be provided as {"assert": "expr"} or legacy {"call": "...", "expected": value}.
    """
    code = extract_python_code(output)
    if not code:
        return False
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception:
        return False
    for test in tests:
        if "assert" in test:
            expr = test.get("assert")
            if not expr or not _eval_assert(namespace, expr):
                return False
            continue
        expr = test.get("call")
        expected = test.get("expected")
        if not expr:
            return False
        try:
            value = eval(expr, namespace)
        except Exception:
            return False
        if value != expected:
            return False
    return True


def score_task(task: Dict[str, Any], output: str) -> bool:
    """
    Evaluate a task against an output. Supports combining multiple criteria.
    Returns True only if all specified criteria pass.
    """
    checks = []
    if "expected" in task:
        checks.append(score_exact(output, task["expected"]))
    if "contains_all" in task:
        checks.append(score_contains_all(output, task["contains_all"]))
    if "contains_any" in task:
        checks.append(score_contains_any(output, task["contains_any"]))
    if "asserts" in task:
        checks.append(score_contains_all(output, task["asserts"]))
    if "code_tests" in task:
        checks.append(run_code_tests(output, task["code_tests"]))
    if not checks:
        return False
    return all(checks)


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
            "asserts": row.get("asserts"),
            "code_tests": row.get("code_tests"),
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
