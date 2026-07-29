"""RAGAS-based evaluation harness for kb-rag.

Runs offline evaluation of the kb-rag query API against a JSONL testset using
RAGAS metrics (faithfulness, answer_relevancy, context_precision). Falls back
to lightweight heuristics when the ``ragas`` package is unavailable, marking
the report as ``heuristic`` mode.

Usage::

    python eval/rag_eval.py --testset eval/testset.jsonl \
        --api http://localhost:8000 --output eval/report.md

Flow:
    1. Read ``testset.jsonl`` (one ``{"question","ground_truth"}`` per line).
    2. Call ``POST /api/v1/query`` for each question to fetch answer + refs.
    3. Build a RAGAS dataset (question / answer / contexts / ground_truth).
    4. Run ``ragas.evaluate`` with the three metrics.
    5. Render a Markdown report (overall means + per-row detail table).

The ``requests`` library is imported lazily so ``--help`` works without it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

QUERY_PATH = "/api/v1/query"
TIMEOUT = 30.0
TEMPLATE_PATH = Path(__file__).parent / "report_template.md"

DEFAULT_TEMPLATE = """\
# kb-rag 评测报告

> 评测模式: {{MODE}}

## 整体指标

| 指标 | 数值 |
|------|------|
| Faithfulness | {{OVERALL_FAITHFULNESS}} |
| Answer Relevancy | {{OVERALL_ANSWER_RELEVANCY}} |
| Context Precision | {{OVERALL_CONTEXT_PRECISION}} |

## 逐条详情

{{DETAILS_TABLE}}
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate kb-rag with RAGAS metrics (or heuristic fallback).",
    )
    parser.add_argument(
        "--testset",
        default="eval/testset.jsonl",
        help="Path to JSONL testset (default: eval/testset.jsonl).",
    )
    parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="Base URL of the kb-rag API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--output",
        default="eval/report.md",
        help="Path to write the markdown report (default: eval/report.md).",
    )
    return parser.parse_args(argv)


def load_testset(path: str) -> list[dict[str, str]]:
    """Load a JSONL testset, one ``{"question","ground_truth"}`` object per line."""
    items: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj or "ground_truth" not in obj:
                raise ValueError(
                    f"testset line {lineno}: missing question/ground_truth"
                )
            items.append(obj)
    return items


def call_query(api_base: str, question: str, session: Any) -> dict[str, Any]:
    """POST a question to the query API and return the parsed JSON response."""
    url = api_base.rstrip("/") + QUERY_PATH
    resp = session.post(url, json={"question": question}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def build_contexts(result: dict[str, Any]) -> list[str]:
    """Extract non-empty reference snippets from a query response."""
    refs = result.get("references") or []
    contexts: list[str] = []
    for ref in refs:
        snippet = (
            (ref.get("snippet") or "").strip() if isinstance(ref, dict) else str(ref)
        )
        if snippet:
            contexts.append(snippet)
    return contexts


# --------------------------------------------------------------------------- #
# RAGAS path
# --------------------------------------------------------------------------- #


def _run_ragas(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]], bool]:
    """Attempt RAGAS evaluation. Returns ``(overall, per_row, ok)``.

    On any import or evaluation failure, returns ``({}, [], False)`` so the
    caller can fall back to the heuristic path.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except Exception as exc:  # noqa: BLE001 - any import issue triggers fallback
        print(f"[eval] ragas unavailable ({exc}); falling back to heuristic", file=sys.stderr)
        return {}, [], False

    rows = [
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ]
    try:
        ds = Dataset.from_list(rows)
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
    except Exception as exc:  # noqa: BLE001 - evaluation failure triggers fallback
        print(
            f"[eval] ragas evaluate() failed ({exc}); falling back to heuristic",
            file=sys.stderr,
        )
        return {}, [], False

    df = result.to_pandas() if hasattr(result, "to_pandas") else None
    per_row: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        row_metrics: dict[str, float] = {}
        if df is not None and i < len(df):
            for col in ("faithfulness", "answer_relevancy", "context_precision"):
                if col in df.columns:
                    val = df.iloc[i].get(col)
                    try:
                        fval = float(val)
                        if fval != fval:  # NaN check (NaN != NaN)
                            fval = 0.0
                    except (TypeError, ValueError):
                        fval = 0.0
                    row_metrics[col] = fval
        per_row.append({**s, **row_metrics})

    overall: dict[str, float] = {}
    for col in ("faithfulness", "answer_relevancy", "context_precision"):
        vals = [r.get(col, 0.0) for r in per_row if r.get(col) is not None]
        overall[col] = sum(vals) / len(vals) if vals else 0.0
    return overall, per_row, True


# --------------------------------------------------------------------------- #
# Heuristic fallback
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", (text or "").lower()) if t]


def _heuristic_score(sample: dict[str, Any]) -> dict[str, float]:
    """Lightweight token-overlap heuristics used when ragas is unavailable."""
    answer = sample.get("answer", "") or ""
    contexts = sample.get("contexts", []) or []
    ground_truth = sample.get("ground_truth", "") or ""

    gt_tokens = set(_tokenize(ground_truth))
    ans_tokens = _tokenize(answer)
    ans_token_set = set(ans_tokens)
    ctx_tokens: set[str] = set()
    for c in contexts:
        ctx_tokens.update(_tokenize(c))

    # faithfulness: fraction of answer tokens supported by retrieved context
    if ans_token_set:
        faithfulness = len(ans_token_set & ctx_tokens) / len(ans_token_set)
    else:
        faithfulness = 0.0

    # context_precision: fraction of ground-truth tokens covered by context
    if gt_tokens:
        context_precision = len(gt_tokens & ctx_tokens) / len(gt_tokens)
    else:
        context_precision = 0.0

    # answer_relevancy: overlap between answer and ground_truth
    if gt_tokens and ans_token_set:
        answer_relevancy = len(ans_token_set & gt_tokens) / max(len(gt_tokens), 1)
    else:
        answer_relevancy = 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(context_precision, 4),
    }


def _run_heuristic(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    per_row = [{**s, **_heuristic_score(s)} for s in samples]
    overall: dict[str, float] = {}
    for col in ("faithfulness", "answer_relevancy", "context_precision"):
        vals = [r.get(col, 0.0) for r in per_row]
        overall[col] = sum(vals) / len(vals) if vals else 0.0
    return overall, per_row


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #


def render_report(
    overall: dict[str, float],
    per_row: list[dict[str, Any]],
    mode: str,
) -> str:
    """Fill the report template (or the built-in default) with results."""
    if TEMPLATE_PATH.exists():
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        template = DEFAULT_TEMPLATE

    detail_lines = [
        "| # | Question | Faithfulness | Answer Relevancy | Context Precision |",
        "|---|----------|--------------|------------------|-------------------|",
    ]
    for i, row in enumerate(per_row, 1):
        q = row.get("question", "").replace("|", "\\|")
        if len(q) > 80:
            q = q[:77] + "..."
        detail_lines.append(
            f"| {i} | {q} | {row.get('faithfulness', 0.0):.4f} | "
            f"{row.get('answer_relevancy', 0.0):.4f} | "
            f"{row.get('context_precision', 0.0):.4f} |"
        )
    details_table = "\n".join(detail_lines)

    return (
        template.replace("{{MODE}}", mode)
        .replace("{{OVERALL_FAITHFULNESS}}", f"{overall.get('faithfulness', 0.0):.4f}")
        .replace("{{OVERALL_ANSWER_RELEVANCY}}", f"{overall.get('answer_relevancy', 0.0):.4f}")
        .replace("{{OVERALL_CONTEXT_PRECISION}}", f"{overall.get('context_precision', 0.0):.4f}")
        .replace("{{DETAILS_TABLE}}", details_table)
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: load testset, query API, evaluate, write report."""
    args = parse_args(argv)

    try:
        testset = load_testset(args.testset)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[eval] failed to load testset: {exc}", file=sys.stderr)
        return 2
    if not testset:
        print(f"[eval] testset is empty: {args.testset}", file=sys.stderr)
        return 2
    print(f"[eval] loaded {len(testset)} questions from {args.testset}")

    try:
        import requests
    except ImportError:
        print(
            "error: the 'requests' package is required. "
            "Install with: pip install requests",
            file=sys.stderr,
        )
        return 2

    session = requests.Session()
    samples: list[dict[str, Any]] = []
    for i, item in enumerate(testset, 1):
        question = item["question"]
        try:
            result = call_query(args.api, question, session)
        except Exception as exc:  # noqa: BLE001 - keep evaluating remaining items
            print(f"[eval] Q{i} API call failed: {exc}", file=sys.stderr)
            result = {"answer": "", "references": []}
        answer = result.get("answer", "") or ""
        contexts = build_contexts(result)
        samples.append(
            {
                "question": question,
                "ground_truth": item["ground_truth"],
                "answer": answer,
                "contexts": contexts,
            }
        )
        print(
            f"[eval] Q{i} answered (answer_len={len(answer)} ctx_count={len(contexts)})"
        )

    overall, per_row, ragas_ok = _run_ragas(samples)
    if ragas_ok:
        mode = "ragas"
    else:
        overall, per_row = _run_heuristic(samples)
        mode = "heuristic"

    report = render_report(overall, per_row, mode)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[eval] report written to {out_path} (mode={mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
