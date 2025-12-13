import multiprocessing as mp
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List


DEFAULT_CODE_TEST_TIMEOUT_S = 2.0


def _get_mp_context() -> Any:
    """
    Prefer `fork` on Unix for speed; fall back to `spawn` when unavailable.
    """
    try:
        return mp.get_context("fork")
    except Exception:
        return mp.get_context("spawn")


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


def _run_code_tests_worker(output: str, tests: List[Dict[str, Any]], queue: Any) -> None:
    code = extract_python_code(output)
    if not code:
        queue.put(False)
        return
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception:
        queue.put(False)
        return
    for test in tests:
        if "assert" in test:
            expr = test.get("assert")
            if not expr or not _eval_assert(namespace, expr):
                queue.put(False)
                return
            continue
        expr = test.get("call")
        expected = test.get("expected")
        if not expr:
            queue.put(False)
            return
        try:
            value = eval(expr, namespace)
        except Exception:
            queue.put(False)
            return
        if value != expected:
            queue.put(False)
            return
    queue.put(True)


def _run_code_tests_detailed_worker(output: str, tests: List[Dict[str, Any]], queue: Any) -> None:
    code = extract_python_code(output)
    if not code:
        queue.put({"ok": False, "lines": ["no code extracted (FAIL)"]})
        return
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as exc:
        queue.put({"ok": False, "lines": [f"exec failed with {exc} (FAIL)"]})
        return

    lines: List[str] = []
    ok_all = True
    for test in tests:
        if "assert" in test:
            expr = test.get("assert")
            if not expr:
                lines.append("missing assert expression (FAIL)")
                ok_all = False
                continue
            try:
                result = bool(eval(expr, namespace))
                lines.append(f"assert {expr} ({'OK' if result else 'FAIL'})")
                if not result:
                    ok_all = False
            except Exception as exc:
                lines.append(f"{expr} raised {exc} (FAIL)")
                ok_all = False
            continue

        expr = test.get("call")
        expected = test.get("expected")
        if not expr:
            lines.append("missing call expression (FAIL)")
            ok_all = False
            continue
        try:
            value = eval(expr, namespace)
            is_ok = value == expected
            lines.append(f"{expr} -> {value} (expected {expected}) ({'OK' if is_ok else 'FAIL'})")
            if not is_ok:
                ok_all = False
        except Exception as exc:
            lines.append(f"{expr} raised {exc} (FAIL)")
            ok_all = False

    queue.put({"ok": ok_all, "lines": lines})


def run_code_tests_inline(output: str, tests: List[Dict[str, Any]]) -> bool:
    """
    Fast path: executes code in-process (no timeout). Dangerous on untrusted outputs.
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


def run_code_tests_detailed_inline(output: str, tests: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """
    Fast path: executes code in-process (no timeout). Dangerous on untrusted outputs.
    """
    code = extract_python_code(output)
    if not code:
        return False, ["no code extracted (FAIL)"]
    namespace: Dict[str, Any] = {}
    try:
        exec(code, namespace)
    except Exception as exc:
        return False, [f"exec failed with {exc} (FAIL)"]

    lines: List[str] = []
    ok_all = True
    for test in tests:
        if "assert" in test:
            expr = test.get("assert")
            if not expr:
                lines.append("missing assert expression (FAIL)")
                ok_all = False
                continue
            try:
                result = bool(eval(expr, namespace))
                lines.append(f"assert {expr} ({'OK' if result else 'FAIL'})")
                if not result:
                    ok_all = False
            except Exception as exc:
                lines.append(f"{expr} raised {exc} (FAIL)")
                ok_all = False
            continue

        expr = test.get("call")
        expected = test.get("expected")
        if not expr:
            lines.append("missing call expression (FAIL)")
            ok_all = False
            continue
        try:
            value = eval(expr, namespace)
            is_ok = value == expected
            lines.append(f"{expr} -> {value} (expected {expected}) ({'OK' if is_ok else 'FAIL'})")
            if not is_ok:
                ok_all = False
        except Exception as exc:
            lines.append(f"{expr} raised {exc} (FAIL)")
            ok_all = False
    return ok_all, lines


def run_code_tests_detailed(
    output: str,
    tests: List[Dict[str, Any]],
    timeout_s: float = DEFAULT_CODE_TEST_TIMEOUT_S,
) -> tuple[bool, List[str]]:
    """
    Like `run_code_tests`, but returns per-test lines suitable for printing.
    """
    ctx = _get_mp_context()
    queue: Any = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_code_tests_detailed_worker, args=(output, tests, queue))
    proc.daemon = True
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        return False, [f"timeout after {timeout_s:.2f}s (FAIL)"]
    try:
        msg = queue.get_nowait()
        ok = bool(msg.get("ok"))
        lines = msg.get("lines") or []
        return ok, [str(x) for x in lines]
    except Exception:
        return False, ["no result from worker (FAIL)"]


def run_code_tests(output: str, tests: List[Dict[str, Any]], timeout_s: float = DEFAULT_CODE_TEST_TIMEOUT_S) -> bool:
    """
    Execute model-produced code and run simple assert-style checks.
    Tests can be provided as {"assert": "expr"} or legacy {"call": "...", "expected": value}.
    """
    ctx = _get_mp_context()
    queue: Any = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_code_tests_worker, args=(output, tests, queue))
    proc.daemon = True
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        return False
    try:
        return bool(queue.get_nowait())
    except Exception:
        return False


def score_task(
    task: Dict[str, Any],
    output: str,
    *,
    allow_code_exec: bool = False,
    code_test_timeout_s: float = DEFAULT_CODE_TEST_TIMEOUT_S,
) -> bool:
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
        if allow_code_exec:
            checks.append(run_code_tests(output, task["code_tests"], timeout_s=code_test_timeout_s))
        else:
            checks.append(False)
    if not checks:
        return False
    return all(checks)


def compute_accuracy(
    results: List[Dict[str, Any]],
    *,
    allow_code_exec: bool = False,
    code_test_timeout_s: float = DEFAULT_CODE_TEST_TIMEOUT_S,
) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correct: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in results:
        model = row["model"]
        test_set = row["test_set"]
        output = row.get("response", "")
        if row.get("error"):
            totals[model][test_set] += 1
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
        is_correct = score_task(
            task,
            output,
            allow_code_exec=allow_code_exec,
            code_test_timeout_s=code_test_timeout_s,
        )
        totals[model][test_set] += 1
        if is_correct:
            correct[model][test_set] += 1

    accuracies: Dict[str, Dict[str, float]] = {}
    for model, test_sets in totals.items():
        accuracies[model] = {}
        for test_set, total in test_sets.items():
            if total == 0:
                accuracies[model][test_set] = 0.0
            else:
                accuracies[model][test_set] = correct[model][test_set] / total
    return accuracies
