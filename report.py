import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from metrics import (
    DEFAULT_CODE_TEST_TIMEOUT_S,
    compute_accuracy,
    extract_python_code,
    run_code_tests,
    run_code_tests_detailed,
    score_contains_all,
    score_task,
)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_results(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def compute_timing_stats(results: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns stats per model/test_set: count, avg_s, p50_s, p95_s.
    """
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for row in results:
        model = row.get("model")
        test_set = row.get("test_set")
        if not model or not test_set:
            continue
        val = row.get("total_duration_s")
        if val is None:
            continue
        try:
            dur = float(val)
        except Exception:
            continue
        buckets.setdefault(model, {}).setdefault(test_set, []).append(dur)

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in sorted(buckets):
        stats[model] = {}
        for test_set in sorted(buckets[model]):
            vals = sorted(buckets[model][test_set])
            avg = sum(vals) / len(vals) if vals else 0.0
            stats[model][test_set] = {
                "count": float(len(vals)),
                "avg_s": avg,
                "p50_s": _percentile(vals, 50),
                "p95_s": _percentile(vals, 95),
            }
    return stats


def format_timing_table(timing: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    test_sets = sorted({name for model_data in timing.values() for name in model_data})
    header = ["Model/Test set"] + test_sets
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for model in sorted(timing):
        for metric in ("avg_s", "p95_s"):
            row = [f"{model} {metric}"]
            for test_set in test_sets:
                val = timing.get(model, {}).get(test_set, {}).get(metric)
                row.append(f"{val:.2f}" if val is not None else "")
            lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def _failure_reasons(
    row: Dict[str, Any],
    *,
    allow_code_exec: bool,
) -> List[str]:
    output = row.get("response", "") or ""
    reasons: List[str] = []
    expected = row.get("expected")
    if expected is not None and output.strip() != str(expected).strip():
        reasons.append("expected")
    contains_all = row.get("contains_all")
    if contains_all and not score_contains_all(output, contains_all):
        reasons.append("contains_all")
    contains_any = row.get("contains_any")
    if contains_any:
        text = output.lower()
        if not any(str(part).lower() in text for part in contains_any):
            reasons.append("contains_any")
    asserts = row.get("asserts")
    if asserts and not score_contains_all(output, asserts):
        reasons.append("asserts")
    if row.get("code_tests") is not None:
        reasons.append("code_tests" if allow_code_exec else "code_tests(skipped)")
    return reasons


def collect_failure_samples(
    results: List[Dict],
    *,
    allow_code_exec: bool,
    code_test_timeout_s: float,
    per_group_limit: int = 3,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Returns a dict[model][test_set] -> list of sample failure rows (deterministic order).
    """
    samples: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in results:
        model = row.get("model")
        test_set = row.get("test_set")
        if not model or not test_set:
            continue
        grouped.setdefault((model, test_set), []).append(row)

    for (model, test_set), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        for row in sorted(rows, key=lambda r: int(r.get("task_id", 0))):
            if len(samples.get(model, {}).get(test_set, [])) >= per_group_limit:
                continue
            if row.get("error"):
                ok = False
            else:
                task = {
                    "expected": row.get("expected"),
                    "contains_all": row.get("contains_all"),
                    "contains_any": row.get("contains_any"),
                    "asserts": row.get("asserts"),
                    "code_tests": row.get("code_tests"),
                }
                task = {k: v for k, v in task.items() if v is not None}
                ok = score_task(
                    task,
                    row.get("response", "") or "",
                    allow_code_exec=allow_code_exec,
                    code_test_timeout_s=code_test_timeout_s,
                )
            if ok:
                continue
            lst = samples.setdefault(model, {}).setdefault(test_set, [])
            prompt = (row.get("prompt", "") or "").strip().replace("\n", " ")
            resp = (row.get("response", "") or "").strip().replace("\n", " ")
            item = {
                "task_id": row.get("task_id"),
                "reasons": _failure_reasons(row, allow_code_exec=allow_code_exec),
                "prompt": (prompt[:160] + "...") if len(prompt) > 160 else prompt,
                "response": (resp[:160] + "...") if len(resp) > 160 else resp,
                "error": row.get("error"),
            }
            lst.append(item)
    return samples


def format_markdown_table(accuracies: Dict[str, Dict[str, float]], overall: Dict[str, float]) -> str:
    test_sets = sorted({name for model_data in accuracies.values() for name in model_data})
    header = ["Model"] + test_sets + ["Overall"]
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for model, cats in accuracies.items():
        row = [model] + [f"{cats.get(name, 0.0):.2f}" for name in test_sets] + [f"{overall.get(model, 0.0):.2f}"]
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def format_contains_all_table(stats: Dict[str, Dict[str, Tuple[int, int]]]) -> str:
    test_sets = sorted({name for model_data in stats.values() for name in model_data})
    header = ["Model"] + test_sets
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for model, cats in stats.items():
        row_vals = []
        for name in test_sets:
            passed, total = cats.get(name, (0, 0))
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
        test_set = row["test_set"]
        output = row.get("response", "")
        is_ok = score_contains_all(output, parts)
        if model not in stats:
            stats[model] = {}
        passed, total = stats[model].get(test_set, (0, 0))
        stats[model][test_set] = (passed + (1 if is_ok else 0), total + 1)
    return stats


def save_code_artifacts(results: List[Dict], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for row in results:
        resp = row.get("response")
        if not resp:
            continue
        test_set = row.get("test_set", "")
        if test_set != "code" and "```" not in resp:
            continue
        model = row.get("model", "unknown")
        task_id = row.get("task_id", 0)
        code = extract_python_code(resp)
        if not code:
            continue
        filename = f"{model}_{test_set}_task{task_id}.py".replace("/", "_").replace(":", "_")
        (output_dir / filename).write_text(code, encoding="utf-8")
        saved += 1
    return saved


def compute_overall_accuracy(
    results: List[Dict],
    *,
    allow_code_exec: bool = False,
    code_test_timeout_s: float = DEFAULT_CODE_TEST_TIMEOUT_S,
) -> Dict[str, float]:
    totals: Dict[str, int] = {}
    correct: Dict[str, int] = {}
    for row in results:
        model = row["model"]
        totals[model] = totals.get(model, 0) + 1
        if row.get("error"):
            continue
        output = row.get("response", "")
        task = {
            "expected": row.get("expected"),
            "contains_all": row.get("contains_all"),
            "contains_any": row.get("contains_any"),
            "asserts": row.get("asserts"),
            "code_tests": row.get("code_tests"),
        }
        task = {k: v for k, v in task.items() if v is not None}
        ok = score_task(
            task,
            output,
            allow_code_exec=allow_code_exec,
            code_test_timeout_s=code_test_timeout_s,
        )
        if ok:
            correct[model] = correct.get(model, 0) + 1
    return {m: (correct.get(m, 0) / totals[m]) if totals.get(m, 0) else 0.0 for m in totals}


def collect_errors(results: List[Dict], limit: int = 5) -> List[str]:
    errors = []
    for row in results:
        if row.get("error"):
            test_set = row.get("test_set", "?")
            errors.append(f"{row.get('model','?')} / {test_set} / task {row.get('task_id','?')}: {row.get('error')}")
            if len(errors) >= limit:
                break
    return errors


def compute_code_test_stats(
    results: List[Dict],
    *,
    code_test_timeout_s: float = DEFAULT_CODE_TEST_TIMEOUT_S,
    limit_failures: int = 5,
) -> Tuple[Dict[str, Dict[str, Tuple[int, int]]], List[str]]:
    stats: Dict[str, Dict[str, Tuple[int, int]]] = {}
    failures: List[str] = []
    for row in results:
        tests = row.get("code_tests")
        if not tests or row.get("error"):
            continue
        model = row["model"]
        test_set = row["test_set"]
        ok = run_code_tests(row.get("response", ""), tests, timeout_s=code_test_timeout_s)
        if not ok and len(failures) < limit_failures:
            failures.append(f"{model}/{test_set}/task{row.get('task_id','?')} failed code tests")
        if model not in stats:
            stats[model] = {}
        passed, total = stats[model].get(test_set, (0, 0))
        stats[model][test_set] = (passed + (1 if ok else 0), total + 1)
    return stats, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report from results.json")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    parser.add_argument(
        "--unsafe-code-exec",
        action="store_true",
        help="Allow executing model-produced code for code_tests (overrides config).",
    )
    parser.add_argument(
        "--verbose-code-tests",
        action="store_true",
        help="Print each code_tests assertion result to stdout (requires unsafe code exec).",
    )
    parser.add_argument(
        "--code-test-timeout-s",
        type=float,
        default=None,
        help="Timeout for code_tests execution (seconds); overrides config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    allow_code_exec = bool(cfg.get("unsafe_code_exec", False)) or bool(args.unsafe_code_exec)
    verbose_code_tests = bool(cfg.get("report_verbose_code_tests", False)) or bool(args.verbose_code_tests)
    code_test_timeout_s = (
        float(args.code_test_timeout_s)
        if args.code_test_timeout_s is not None
        else float(cfg.get("code_test_timeout_s", DEFAULT_CODE_TEST_TIMEOUT_S))
    )

    results_path = Path("results.json")
    if not results_path.exists():
        raise FileNotFoundError("results.json not found. Run runner.py first.")

    results = load_results(results_path)
    accuracies = compute_accuracy(results, allow_code_exec=allow_code_exec, code_test_timeout_s=code_test_timeout_s)
    overall_acc = compute_overall_accuracy(results, allow_code_exec=allow_code_exec, code_test_timeout_s=code_test_timeout_s)
    contains_all_stats = compute_contains_all_stats(results)
    code_test_stats: Dict[str, Dict[str, Tuple[int, int]]] = {}
    code_test_failures: List[str] = []
    if allow_code_exec:
        code_test_stats, code_test_failures = compute_code_test_stats(results, code_test_timeout_s=code_test_timeout_s)
    saved_codes = save_code_artifacts(results, Path("artifacts"))
    errors = collect_errors(results)
    models = sorted({r["model"] for r in results})
    test_sets = sorted({r.get("test_set") for r in results if r.get("test_set")})
    total = len(results)
    error_count = sum(1 for r in results if r.get("error"))
    timing = compute_timing_stats(results)
    failure_samples = collect_failure_samples(
        results,
        allow_code_exec=allow_code_exec,
        code_test_timeout_s=code_test_timeout_s,
        per_group_limit=3,
    )

    print(f"Models: {', '.join(models)}")
    print(f"Test sets: {', '.join(test_sets)}")
    print(f"Total results: {total} (errors: {error_count})")
    print(f"Code execution for code_tests: {'ENABLED' if allow_code_exec else 'DISABLED'}")
    print(f"Code test timeout: {code_test_timeout_s:.2f}s")
    if timing:
        print("Timing stats: available (avg_s/p95_s per model/test_set)")
    if failure_samples:
        groups = sum(len(v) for v in failure_samples.values())
        print(f"Failure samples: collected for {groups} model/test_set group(s)")

    if verbose_code_tests and not allow_code_exec:
        print("Verbose code tests requested but unsafe code execution is DISABLED; skipping per-test output.")

    if verbose_code_tests and allow_code_exec:
        rows_with_tests = [r for r in results if r.get("code_tests") and not r.get("error")]
        rows_with_tests.sort(key=lambda r: (r.get("model", ""), r.get("test_set", ""), int(r.get("task_id", 0))))
        for row in rows_with_tests:
            model = row.get("model", "?")
            test_set = row.get("test_set", "?")
            task_id = row.get("task_id", "?")
            prompt_preview = (row.get("prompt", "") or "").strip().replace("\n", " ")
            if len(prompt_preview) > 120:
                prompt_preview = prompt_preview[:120] + "..."
            print(f"[code_tests] model={model} test_set={test_set} task={task_id} prompt=\"{prompt_preview}\"")
            ok, lines = run_code_tests_detailed(row.get("response", "") or "", row["code_tests"], timeout_s=code_test_timeout_s)
            for line in lines:
                print(f"  - {line}")
            if not ok:
                print("  => FAIL")

    report_path = Path("report.md")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Models: {', '.join(models)}\n")
        f.write(f"- Test sets: {', '.join(test_sets)}\n")
        f.write(f"- Total results: {total} (errors: {error_count})\n")
        f.write(f"- Extracted Python snippets: {saved_codes} file(s) in artifacts/\n\n")
        f.write(f"- Code execution for code_tests: {'ENABLED' if allow_code_exec else 'DISABLED'}\n")
        f.write(f"- Code test timeout: {code_test_timeout_s:.2f}s\n\n")
        f.write("## Accuracy\n\n")
        f.write(format_markdown_table(accuracies, overall_acc))
        f.write("\n\n")
        if contains_all_stats:
            f.write("## Contains_all checks\n\n")
            f.write(format_contains_all_table(contains_all_stats))
            f.write("\n\n")
        if allow_code_exec and code_test_stats:
            f.write("## Code tests\n\n")
            f.write(format_contains_all_table(code_test_stats))
            f.write("\n\n")
        if allow_code_exec and code_test_failures:
            f.write("## Code test failures (first few)\n\n")
            for fail in code_test_failures:
                f.write(f"- {fail}\n")
            f.write("\n")
        if errors:
            f.write("## Errors (first few)\n\n")
            for err in errors:
                f.write(f"- {err}\n")
            f.write("\n")
        f.write("## Details\n\n")
        if timing:
            f.write("### Timing (avg_s and p95_s)\n\n")
            f.write(format_timing_table(timing))
            f.write("\n\n")
        if failure_samples:
            f.write("### Failure samples (first few per model/test_set)\n\n")
            for model in sorted(failure_samples):
                for test_set in sorted(failure_samples[model]):
                    items = failure_samples[model][test_set]
                    if not items:
                        continue
                    f.write(f"- {model} / {test_set}\n")
                    for item in items:
                        reasons = ",".join(item.get("reasons", []))
                        task_id = item.get("task_id")
                        err = item.get("error")
                        if err:
                            f.write(f"  - task {task_id}: error={err}\n")
                        else:
                            f.write(
                                f"  - task {task_id}: reasons={reasons} prompt=\"{item.get('prompt','')}\" response=\"{item.get('response','')}\"\n"
                            )
            f.write("\n")


if __name__ == "__main__":
    main()
