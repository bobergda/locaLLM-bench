import json
import os
import re
import html
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from metrics import (
    DEFAULT_CODE_TEST_TIMEOUT_S,
    extract_python_code,
    run_code_tests_detailed,
    run_code_tests_detailed_inline,
    score_contains_all,
    score_contains_any,
)

_MARK_START = "__LOCALLLM_MARK_START__"
_MARK_END = "__LOCALLLM_MARK_END__"


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


def compute_tokens_per_sec_stats(results: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns throughput stats per model/test_set based on eval_count / total_duration_s:
    count, avg_tps, p50_tps, p95_tps.
    """
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for row in results:
        model = row.get("model")
        test_set = row.get("test_set")
        if not model or not test_set:
            continue

        eval_count = row.get("eval_count")
        total_duration_s = row.get("total_duration_s")
        if eval_count is None or total_duration_s is None:
            continue
        try:
            tokens = float(eval_count)
            dur = float(total_duration_s)
        except Exception:
            continue
        if tokens <= 0 or dur <= 0:
            continue
        buckets.setdefault(model, {}).setdefault(test_set, []).append(tokens / dur)

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in sorted(buckets):
        stats[model] = {}
        for test_set in sorted(buckets[model]):
            vals = sorted(buckets[model][test_set])
            avg = sum(vals) / len(vals) if vals else 0.0
            stats[model][test_set] = {
                "count": float(len(vals)),
                "avg_tps": avg,
                "p50_tps": _percentile(vals, 50),
                "p95_tps": _percentile(vals, 95),
            }
    return stats


def format_tokens_per_sec_table(throughput: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    test_sets = sorted({name for model_data in throughput.values() for name in model_data})
    header = ["Model/Test set"] + test_sets
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    for model in sorted(throughput):
        for metric in ("avg_tps", "p95_tps"):
            row = [f"{model} {metric}"]
            for test_set in test_sets:
                val = throughput.get(model, {}).get(test_set, {}).get(metric)
                row.append(f"{val:.1f}" if val is not None else "")
            lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def _truncate_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return text[: max_chars - 1] + "…", True


def _unique_nonempty_phrases(phrases: List[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for p in phrases:
        if p is None:
            continue
        s = str(p).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _highlight_html(text: str, phrases: List[str]) -> str:
    """
    Returns HTML-escaped text with <mark> wrapping any occurrences of phrases (case-insensitive).
    """
    if not text:
        return ""
    phrases = _unique_nonempty_phrases(phrases)
    if not phrases:
        return html.escape(text)

    phrases_sorted = sorted(phrases, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(p) for p in phrases_sorted), flags=re.IGNORECASE)
    marked = pattern.sub(lambda m: f"{_MARK_START}{m.group(0)}{_MARK_END}", text)
    escaped = html.escape(marked)
    return escaped.replace(_MARK_START, "<mark>").replace(_MARK_END, "</mark>")


def _format_response_pre_block(response: str, *, highlight_phrases: List[str], max_chars: int) -> tuple[str, bool]:
    clipped, was_truncated = _truncate_text(response or "", max_chars=max_chars)
    body = _highlight_html(clipped, highlight_phrases)
    return f"<pre>{body}</pre>", was_truncated


def _row_tokens_per_sec(row: Dict[str, Any]) -> float | None:
    eval_count = row.get("eval_count")
    total_duration_s = row.get("total_duration_s")
    if eval_count is None or total_duration_s is None:
        return None
    try:
        tokens = float(eval_count)
        dur = float(total_duration_s)
    except Exception:
        return None
    if tokens <= 0 or dur <= 0:
        return None
    return tokens / dur


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
    code_test_ok: Dict[tuple[str, str, int], bool],
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
            ok = score_row(row, allow_code_exec=allow_code_exec, code_test_ok=code_test_ok)
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


def _score_row_text_only(row: Dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    output = row.get("response", "") or ""
    checks: List[bool] = []

    expected = row.get("expected")
    if expected is not None:
        checks.append(output.strip() == str(expected).strip())
    if row.get("contains_all") is not None:
        checks.append(score_contains_all(output, row["contains_all"]))
    if row.get("contains_any") is not None:
        checks.append(score_contains_any(output, row["contains_any"]))
    if row.get("asserts") is not None:
        checks.append(score_contains_all(output, row["asserts"]))

    return bool(checks) and all(checks)


def collect_text_check_samples(
    results: List[Dict],
) -> List[Dict[str, Any]]:
    """
    Returns a list of rows with *text* checks only (expected/contains_all/contains_any/asserts),
    preserving the original order from results.json.
    """
    out: List[Dict[str, Any]] = []
    for row in results:
        model = row.get("model")
        test_set = row.get("test_set")
        if not model or not test_set:
            continue
        has_text_checks = any(row.get(k) is not None for k in ("expected", "contains_all", "contains_any", "asserts"))
        if not has_text_checks:
            continue
        ok = bool(_score_row_text_only(row))
        out.append(
            {
                "model": model,
                "test_set": test_set,
                "task_id": row.get("task_id"),
                "prompt": row.get("prompt", "") or "",
                "response": row.get("response", "") or "",
                "total_duration_s": row.get("total_duration_s"),
                "eval_count": row.get("eval_count"),
                "ok": ok,
                "expected": row.get("expected"),
                "contains_all": row.get("contains_all") or [],
                "contains_any": row.get("contains_any") or [],
                "asserts": row.get("asserts") or [],
                "error": row.get("error"),
            }
        )
    return out


def _missing_phrases(text: str, phrases: List[str]) -> List[str]:
    lowered = text.lower()
    return [p for p in phrases if p.lower() not in lowered]


def format_text_check_samples_md(
    samples: List[Dict[str, Any]],
    *,
    max_prompt_chars: int = 240,
    max_response_chars: int = 1600,
) -> str:
    if not samples:
        return "(no text checks to display)\n"

    lines: List[str] = []
    last_group: tuple[str, str] | None = None
    for row in samples:
        model = str(row.get("model", "?"))
        test_set = str(row.get("test_set", "?"))
        group = (model, test_set)
        if group != last_group:
            lines.append(f"#### {model} / {test_set}\n")
            last_group = group

        task_id = row.get("task_id", "?")
        prompt = (row.get("prompt") or "").strip()
        response = row.get("response") or ""
        ok_overall = bool(row.get("ok", False)) and not bool(row.get("error"))

        prompt_preview, prompt_trunc = _truncate_text(prompt.replace("\n", " "), max_chars=max_prompt_chars)
        if prompt_trunc:
            prompt_preview += " (truncated)"

        expected = row.get("expected")
        contains_all = _unique_nonempty_phrases(list(row.get("contains_all") or []))
        contains_any = _unique_nonempty_phrases(list(row.get("contains_any") or []))
        asserts = _unique_nonempty_phrases(list(row.get("asserts") or []))

        missing_all = _missing_phrases(response, contains_all) if contains_all else []
        missing_asserts = _missing_phrases(response, asserts) if asserts else []
        matched_any = [p for p in contains_any if p.lower() in response.lower()] if contains_any else []

        tps = _row_tokens_per_sec(row)
        tps_txt = f"{tps:.1f}" if tps is not None else "n/a"
        summary = (
            f"{model} / {test_set} / task {task_id} — {'OK' if ok_overall else 'FAIL'} — "
            f"tps={tps_txt} — "
            f"{html.escape(prompt_preview)}"
        )
        lines.append("<details>")
        lines.append(f"<summary>{summary}</summary>")
        lines.append("")
        lines.append(f"- prompt: {prompt_preview}")
        if expected is not None:
            ok = response.strip() == str(expected).strip()
            lines.append(f"- expected: {'OK' if ok else 'FAIL'} (`{str(expected).strip()}`)")
        if contains_all:
            lines.append(f"- contains_all: {'OK' if not missing_all else 'FAIL'} ({', '.join(contains_all)})")
            if missing_all:
                lines.append(f"  - missing: {', '.join(missing_all)}")
        if contains_any:
            lines.append(f"- contains_any: {'OK' if matched_any else 'FAIL'} ({', '.join(contains_any)})")
            if matched_any:
                lines.append(f"  - matched: {', '.join(matched_any)}")
            else:
                lines.append(f"  - missing any-of: {', '.join(contains_any)}")
        if asserts:
            lines.append(f"- asserts: {'OK' if not missing_asserts else 'FAIL'} ({', '.join(asserts)})")
            if missing_asserts:
                lines.append(f"  - missing: {', '.join(missing_asserts)}")

        highlight_phrases = []
        highlight_phrases.extend([p for p in contains_all if p.lower() in response.lower()])
        highlight_phrases.extend(matched_any)
        highlight_phrases.extend([p for p in asserts if p.lower() in response.lower()])

        pre, resp_trunc = _format_response_pre_block(
            response,
            highlight_phrases=highlight_phrases,
            max_chars=max_response_chars,
        )
        lines.append("- response:")
        if resp_trunc:
            lines.append(f"  - note: truncated to {max_response_chars} chars")
        lines.append("")
        lines.append(pre)
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_per_request_stats_md(
    results: List[Dict[str, Any]],
    *,
    max_prompt_chars: int = 240,
) -> str:
    """
    Per-row stats in the same order as results.json.
    """
    lines: List[str] = []
    any_rows = False
    for i, row in enumerate(results):
        model = str(row.get("model", "?"))
        test_set = str(row.get("test_set", "?"))
        task_id = row.get("task_id", "?")
        prompt = (row.get("prompt") or "").strip()
        prompt_preview, prompt_trunc = _truncate_text(prompt.replace("\n", " "), max_chars=max_prompt_chars)
        if prompt_trunc:
            prompt_preview += " (truncated)"

        dur = row.get("total_duration_s")
        eval_count = row.get("eval_count")
        tps = _row_tokens_per_sec(row)
        any_rows = True

        dur_txt = f"{float(dur):.2f}s" if dur is not None else "n/a"
        eval_txt = str(eval_count) if eval_count is not None else "n/a"
        tps_txt = f"{tps:.1f}" if tps is not None else "n/a"
        ok_txt = "ERROR" if row.get("error") else "OK"

        summary = (
            f"{i:04d} — {model} / {test_set} / task {task_id} — {ok_txt} — "
            f"tps={tps_txt}, tokens={eval_txt}, dur={dur_txt} — {html.escape(prompt_preview)}"
        )
        lines.append("<details>")
        lines.append(f"<summary>{summary}</summary>")
        lines.append("")
        lines.append(f"- model: `{model}`")
        lines.append(f"- test_set: `{test_set}`")
        lines.append(f"- task_id: `{task_id}`")
        lines.append(f"- total_duration_s: `{dur}`")
        lines.append(f"- eval_count: `{eval_count}`")
        lines.append(f"- tokens_per_sec: `{tps}`")
        if row.get("error"):
            lines.append(f"- error: `{row.get('error')}`")
        lines.append(f"- prompt: {prompt_preview}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if not any_rows:
        return "(no results to display)\n"
    return "\n".join(lines).rstrip() + "\n"


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


def _row_key(row: Dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("model", "")),
        str(row.get("test_set", "")),
        int(row.get("task_id", 0)),
    )


def score_row(
    row: Dict[str, Any],
    *,
    allow_code_exec: bool,
    code_test_ok: Dict[tuple[str, str, int], bool],
) -> bool:
    if row.get("error"):
        return False
    output = row.get("response", "") or ""
    checks: List[bool] = []

    expected = row.get("expected")
    if expected is not None:
        checks.append(output.strip() == str(expected).strip())
    if row.get("contains_all") is not None:
        checks.append(score_contains_all(output, row["contains_all"]))
    if row.get("contains_any") is not None:
        checks.append(score_contains_any(output, row["contains_any"]))
    if row.get("asserts") is not None:
        checks.append(score_contains_all(output, row["asserts"]))
    if row.get("code_tests") is not None:
        if allow_code_exec:
            checks.append(bool(code_test_ok.get(_row_key(row), False)))
        else:
            checks.append(False)

    return bool(checks) and all(checks)


def compute_accuracy_and_overall(
    results: List[Dict],
    *,
    allow_code_exec: bool,
    code_test_ok: Dict[tuple[str, str, int], bool],
) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    totals: Dict[str, Dict[str, int]] = {}
    correct: Dict[str, Dict[str, int]] = {}
    totals_overall: Dict[str, int] = {}
    correct_overall: Dict[str, int] = {}

    for row in results:
        model = row.get("model")
        test_set = row.get("test_set")
        if not model or not test_set:
            continue
        totals.setdefault(model, {})
        correct.setdefault(model, {})
        totals[model][test_set] = totals[model].get(test_set, 0) + 1
        totals_overall[model] = totals_overall.get(model, 0) + 1

        ok = score_row(row, allow_code_exec=allow_code_exec, code_test_ok=code_test_ok)
        if ok:
            correct[model][test_set] = correct[model].get(test_set, 0) + 1
            correct_overall[model] = correct_overall.get(model, 0) + 1

    accuracies: Dict[str, Dict[str, float]] = {}
    for model, test_sets in totals.items():
        accuracies[model] = {}
        for test_set, total in test_sets.items():
            accuracies[model][test_set] = (correct.get(model, {}).get(test_set, 0) / total) if total else 0.0

    overall: Dict[str, float] = {}
    for model, total in totals_overall.items():
        overall[model] = (correct_overall.get(model, 0) / total) if total else 0.0
    return accuracies, overall


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
    limit_failures: int = 5,
    code_test_ok: Dict[tuple[str, str, int], bool],
) -> Tuple[Dict[str, Dict[str, Tuple[int, int]]], List[str]]:
    stats: Dict[str, Dict[str, Tuple[int, int]]] = {}
    failures: List[str] = []
    for row in results:
        tests = row.get("code_tests")
        if not tests or row.get("error"):
            continue
        model = row["model"]
        test_set = row["test_set"]
        ok = bool(code_test_ok.get(_row_key(row), False))
        if not ok and len(failures) < limit_failures:
            failures.append(f"{model}/{test_set}/task{row.get('task_id','?')} failed code tests")
        if model not in stats:
            stats[model] = {}
        passed, total = stats[model].get(test_set, (0, 0))
        stats[model][test_set] = (passed + (1 if ok else 0), total + 1)
    return stats, failures


def _load_report_settings(cfg: Dict[str, Any]) -> tuple[bool, str, float]:
    allow_code_exec = bool(cfg.get("unsafe_code_exec", False))
    code_test_mode = str(cfg.get("code_test_mode", "safe")).strip().lower()
    if code_test_mode not in ("safe", "fast"):
        raise ValueError("config.yaml: code_test_mode must be 'safe' or 'fast'")
    code_test_timeout_s = float(cfg.get("code_test_timeout_s", DEFAULT_CODE_TEST_TIMEOUT_S))
    return allow_code_exec, code_test_mode, code_test_timeout_s


def _require_results(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return load_results(path)


def _sorted_rows_with_code_tests(results: List[Dict]) -> List[Dict]:
    rows = [r for r in results if r.get("code_tests") and not r.get("error")]
    rows.sort(key=lambda r: (r.get("model", ""), r.get("test_set", ""), int(r.get("task_id", 0))))
    return rows


def _find_latest_run_dir() -> Path | None:
    """
    Returns the latest run directory, if available.
    Prefers results/latest -> symlink created by runner.py.
    """
    latest_link = Path("results") / "latest"
    if latest_link.exists() and latest_link.is_dir():
        return latest_link.resolve()

    root = Path("results") / "runs"
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.name)
    return dirs[-1]


def _default_results_path() -> Path:
    """
    Default results.json resolution:
    1) LOCALLLM_RESULTS env
    2) results/latest/results.json
    3) latest dir under results/runs/*/results.json
    4) ./results.json (legacy)
    5) ./results/results.json (legacy)
    """
    if (env_path := os.environ.get("LOCALLLM_RESULTS")):
        return Path(env_path)

    if (latest_dir := _find_latest_run_dir()) is not None:
        candidate = latest_dir / "results.json"
        if candidate.exists():
            return candidate

    for candidate in (Path("results.json"), Path("results") / "results.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No results.json found (run runner.py first).")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate report.md from results.json")
    p.add_argument(
        "--results",
        default=None,
        help="Path to results.json (overrides auto-detected latest and LOCALLLM_RESULTS).",
    )
    return p.parse_args()


def _run_code_tests(
    rows_with_tests: List[Dict],
    *,
    allow_code_exec: bool,
    code_test_mode: str,
    code_test_timeout_s: float,
) -> tuple[Dict[tuple[str, str, int], bool], Dict[tuple[str, str, int], List[str]]]:
    code_test_ok: Dict[tuple[str, str, int], bool] = {}
    code_test_lines: Dict[tuple[str, str, int], List[str]] = {}
    if not allow_code_exec:
        return code_test_ok, code_test_lines

    for row in rows_with_tests:
        key = _row_key(row)
        tests = row["code_tests"]
        output = row.get("response", "") or ""
        if code_test_mode == "fast":
            ok, lines = run_code_tests_detailed_inline(output, tests)
        else:
            ok, lines = run_code_tests_detailed(output, tests, timeout_s=code_test_timeout_s)
        code_test_lines[key] = lines
        code_test_ok[key] = bool(ok)

    return code_test_ok, code_test_lines


def _print_console_summary(
    *,
    models: List[str],
    test_sets: List[str],
    total: int,
    error_count: int,
    allow_code_exec: bool,
    code_test_mode: str,
    code_test_timeout_s: float,
    timing_available: bool,
    failure_samples_groups: int,
) -> None:
    print(f"Models: {', '.join(models)}")
    print(f"Test sets: {', '.join(test_sets)}")
    print(f"Total results: {total} (errors: {error_count})")
    print(f"Code execution for code_tests: {'ENABLED' if allow_code_exec else 'DISABLED'}")
    if allow_code_exec:
        print(f"Code test mode: {code_test_mode}")
    if allow_code_exec and code_test_mode != "fast":
        print(f"Code test timeout: {code_test_timeout_s:.2f}s")
    if timing_available:
        print("Timing stats: available (avg_s/p95_s per model/test_set)")
    if failure_samples_groups:
        print(f"Failure samples: collected for {failure_samples_groups} model/test_set group(s)")


def _print_code_test_details(
    rows_with_tests: List[Dict],
    *,
    code_test_ok: Dict[tuple[str, str, int], bool],
    code_test_lines: Dict[tuple[str, str, int], List[str]],
) -> None:
    text = _format_code_test_details_text(
        rows_with_tests,
        code_test_ok=code_test_ok,
        code_test_lines=code_test_lines,
    )
    if text:
        print(text)


def _format_code_test_details_text(
    rows_with_tests: List[Dict],
    *,
    code_test_ok: Dict[tuple[str, str, int], bool],
    code_test_lines: Dict[tuple[str, str, int], List[str]],
) -> str:
    lines: List[str] = []
    for row in rows_with_tests:
        model = row.get("model", "?")
        test_set = row.get("test_set", "?")
        task_id = row.get("task_id", "?")
        prompt_preview = (row.get("prompt", "") or "").strip().replace("\n", " ")
        if len(prompt_preview) > 120:
            prompt_preview = prompt_preview[:120] + "..."
        lines.append(f"[code_tests] model={model} test_set={test_set} task={task_id} prompt=\"{prompt_preview}\"")
        key = _row_key(row)
        for line in code_test_lines.get(key, []):
            lines.append(f"  - {line}")
        if not code_test_ok.get(key, False):
            lines.append("  => FAIL")
    return "\n".join(lines)


def _artifact_filename(model: str, test_set: str, task_id: Any) -> str:
    safe_model = str(model).replace("/", "_").replace(":", "_")
    safe_set = str(test_set).replace("/", "_").replace(":", "_")
    return f"{safe_model}_{safe_set}_task{int(task_id)}.py"


def _format_code_test_lines_pre(lines_in: List[str]) -> str:
    if not lines_in:
        return "<pre>(no code test output)</pre>"
    rendered: List[str] = []
    for line in lines_in:
        s = str(line)
        esc = html.escape(s)
        if "(FAIL)" in s or " FAIL" in s:
            rendered.append(f"<mark>{esc}</mark>")
        else:
            rendered.append(esc)
    return "<pre>" + "\n".join(rendered) + "</pre>"


def format_code_test_details_md(
    results: List[Dict[str, Any]],
    *,
    allow_code_exec: bool,
    code_test_ok: Dict[tuple[str, str, int], bool],
    code_test_lines: Dict[tuple[str, str, int], List[str]],
    max_prompt_chars: int = 240,
    max_code_chars: int = 1600,
) -> str:
    if not allow_code_exec:
        return "(skipped: unsafe_code_exec=false)\n"

    lines: List[str] = []
    any_rows = False
    last_group: tuple[str, str] | None = None
    for row in results:
        if not row.get("code_tests"):
            continue
        model = str(row.get("model", "?"))
        test_set = str(row.get("test_set", "?"))
        task_id = row.get("task_id", "?")
        group = (model, test_set)
        if group != last_group:
            lines.append(f"#### {model} / {test_set}\n")
            last_group = group
        prompt = (row.get("prompt") or "").strip()
        prompt_preview, prompt_trunc = _truncate_text(prompt.replace("\n", " "), max_chars=max_prompt_chars)
        if prompt_trunc:
            prompt_preview += " (truncated)"

        key = _row_key(row)
        ok = bool(code_test_ok.get(key, False))
        status = "OK" if ok else "FAIL"

        tps = _row_tokens_per_sec(row)
        tps_txt = f"{tps:.1f}" if tps is not None else "n/a"

        artifact = _artifact_filename(model, test_set, task_id)
        any_rows = True

        summary = f"task {task_id} — {status} — tps={tps_txt} — {html.escape(prompt_preview)}"
        lines.append("<details>")
        lines.append(f"<summary>{summary}</summary>")
        lines.append("")
        lines.append(f"- prompt: {prompt_preview}")
        lines.append(f"- artifact: `artifacts/{artifact}`")

        code = extract_python_code(row.get("response", "") or "")
        code_clip, code_trunc = _truncate_text(code, max_chars=max_code_chars)
        lines.append("- code:")
        if code_trunc:
            lines.append(f"  - note: truncated to {max_code_chars} chars")
        lines.append("")
        lines.append("<pre>" + html.escape(code_clip) + "</pre>")
        lines.append("")

        lines.append("- code_tests:")
        lines.append("")
        lines.append(_format_code_test_lines_pre(code_test_lines.get(key, [])))
        lines.append("")
        lines.append("</details>")
        lines.append("")

    if not any_rows:
        return "(no code_tests to display)\n"
    return "\n".join(lines).rstrip() + "\n"


def _write_report_md(
    report_path: Path,
    *,
    models: List[str],
    test_sets: List[str],
    total: int,
    error_count: int,
    saved_codes: int,
    allow_code_exec: bool,
    code_test_mode: str,
    code_test_timeout_s: float,
    code_test_details_text: str,
    accuracies: Dict[str, Dict[str, float]],
    overall_acc: Dict[str, float],
    contains_all_stats: Dict[str, Dict[str, Tuple[int, int]]],
    code_test_stats: Dict[str, Dict[str, Tuple[int, int]]],
    code_test_failures: List[str],
    errors: List[str],
    timing: Dict[str, Dict[str, Dict[str, float]]],
    throughput: Dict[str, Dict[str, Dict[str, float]]],
    failure_samples: Dict[str, Dict[str, List[Dict[str, Any]]]],
    text_check_samples_md: str,
    per_request_stats_md: str,
    code_test_details_md: str,
) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Models: {', '.join(models)}\n")
        f.write(f"- Test sets: {', '.join(test_sets)}\n")
        f.write(f"- Total results: {total} (errors: {error_count})\n")
        f.write(f"- Extracted Python snippets: {saved_codes} file(s) in artifacts/\n\n")
        f.write(f"- Code execution for code_tests: {'ENABLED' if allow_code_exec else 'DISABLED'}\n")
        if allow_code_exec:
            f.write(f"- Code test mode: {code_test_mode}\n")
            if code_test_mode != "fast":
                f.write(f"- Code test timeout: {code_test_timeout_s:.2f}s\n\n")
            else:
                f.write("\n")
        else:
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
        if throughput:
            f.write("### Throughput (tokens_per_sec avg and p95)\n\n")
            f.write(format_tokens_per_sec_table(throughput))
            f.write("\n\n")
        f.write("### Per-request stats\n\n")
        f.write(per_request_stats_md)
        f.write("\n")
        f.write("### Text check details\n\n")
        f.write(text_check_samples_md)
        f.write("\n")
        f.write("### Code test details\n\n")
        f.write(code_test_details_md)
        f.write("\n")


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path("config.yaml"))
    allow_code_exec, code_test_mode, code_test_timeout_s = _load_report_settings(cfg)

    # CLI args override env vars; env vars remain supported for backward compatibility.
    results_path = Path(args.results) if args.results else _default_results_path()
    results = _require_results(results_path)

    output_dir = results_path.parent
    rows_with_tests = _sorted_rows_with_code_tests(results)
    code_test_ok, code_test_lines = _run_code_tests(
        rows_with_tests,
        allow_code_exec=allow_code_exec,
        code_test_mode=code_test_mode,
        code_test_timeout_s=code_test_timeout_s,
    )
    code_test_details_text = (
        _format_code_test_details_text(
            rows_with_tests,
            code_test_ok=code_test_ok,
            code_test_lines=code_test_lines,
        )
        if allow_code_exec
        else ""
    )
    code_test_details_md = format_code_test_details_md(
        results,
        allow_code_exec=allow_code_exec,
        code_test_ok=code_test_ok,
        code_test_lines=code_test_lines,
    )

    accuracies, overall_acc = compute_accuracy_and_overall(
        results,
        allow_code_exec=allow_code_exec,
        code_test_ok=code_test_ok,
    )
    contains_all_stats = compute_contains_all_stats(results)
    code_test_stats: Dict[str, Dict[str, Tuple[int, int]]] = {}
    code_test_failures: List[str] = []
    if allow_code_exec:
        code_test_stats, code_test_failures = compute_code_test_stats(results, code_test_ok=code_test_ok)
    saved_codes = save_code_artifacts(results, output_dir / "artifacts")
    errors = collect_errors(results)
    models = sorted({r["model"] for r in results})
    test_sets = sorted({str(test_set) for r in results if (test_set := r.get("test_set"))})
    total = len(results)
    error_count = sum(1 for r in results if r.get("error"))
    timing = compute_timing_stats(results)
    throughput = compute_tokens_per_sec_stats(results)
    per_request_stats_md = format_per_request_stats_md(results)
    failure_samples = collect_failure_samples(
        results,
        allow_code_exec=allow_code_exec,
        code_test_ok=code_test_ok,
        per_group_limit=3,
    )
    text_check_samples_md = format_text_check_samples_md(collect_text_check_samples(results))

    failure_samples_groups = sum(len(v) for v in failure_samples.values()) if failure_samples else 0
    _print_console_summary(
        models=models,
        test_sets=test_sets,
        total=total,
        error_count=error_count,
        allow_code_exec=allow_code_exec,
        code_test_mode=code_test_mode,
        code_test_timeout_s=code_test_timeout_s,
        timing_available=bool(timing),
        failure_samples_groups=failure_samples_groups,
    )

    if allow_code_exec:
        _print_code_test_details(
            rows_with_tests,
            code_test_ok=code_test_ok,
            code_test_lines=code_test_lines,
        )

    _write_report_md(
        output_dir / "report.md",
        models=models,
        test_sets=test_sets,
        total=total,
        error_count=error_count,
        saved_codes=saved_codes,
        allow_code_exec=allow_code_exec,
        code_test_mode=code_test_mode,
        code_test_timeout_s=code_test_timeout_s,
        code_test_details_text=code_test_details_text,
        accuracies=accuracies,
        overall_acc=overall_acc,
        contains_all_stats=contains_all_stats,
        code_test_stats=code_test_stats,
        code_test_failures=code_test_failures,
        errors=errors,
        timing=timing,
        throughput=throughput,
        failure_samples=failure_samples,
        text_check_samples_md=text_check_samples_md,
        per_request_stats_md=per_request_stats_md,
        code_test_details_md=code_test_details_md,
    )


if __name__ == "__main__":
    main()
