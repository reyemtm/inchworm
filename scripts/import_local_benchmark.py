#!/usr/bin/env python3
"""Import a finished local HumanEval run into models.json.

This script reads an evalplus-style *_eval_results.json file, computes
HumanEval+/HumanEval pass@1 with the same rule used by local_eval.py, and
updates the canonical models.json entry. If the model is currently only in
pending[], it can be promoted into benchmarked[] automatically.

Usage:
  python3 scripts/import_local_benchmark.py
  python3 scripts/import_local_benchmark.py --result .eval/out/humaneval/granite-4.2-3b_chatml_greedy_eval_results.json
  python3 scripts/import_local_benchmark.py --model granite-4.2-3b --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_JSON = os.path.join(ROOT, "models.json")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_page as bp
import local_eval as le


DISPLAY_NAMES = {
    "qwen3-8b": "Qwen3-8B",
    "cogito-8b": "Cogito-8B",
    "granite-3.3-8b": "Granite-3.3-8B-Instruct",
    "granite-4.2-3b": "Granite-4.2-3B",
    "ministral-8b": "Ministral-8B-Instruct-2410",
    "ministral-3-14b": "Ministral-3-14B",
    "ministral-3-8b": "Ministral-3-8B",
    "ministral-3-3b": "Ministral-3-3B",
    "ds-r1-distill-llama-8b": "DeepSeek-R1-Distill-Llama-8B",
    "opencoder-1.5b": "OpenCoder-1.5B-Instruct",
    "qwen2.5-coder-1.5b": "Qwen2.5-Coder-1.5B-Instruct",
    "qwen3-1.7b": "Qwen3-1.7B",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _parse_params_b(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(value).lower())
    return float(m.group(1)) if m else None


def _memory_estimate(params_b: Optional[float]) -> str:
    if params_b is None:
        return "~unknown @ 4-bit"
    gb = max(1.0, round(params_b * 0.6, 1))
    if gb.is_integer():
        return f"~{int(gb)} GB @ 4-bit"
    return f"~{gb} GB @ 4-bit"


def _newest_result_path() -> Optional[str]:
    cands = []
    for pat in ("**/eval_results.json", "**/*_eval_results.json", "**/humaneval*/eval_results.json"):
        cands.extend(glob.glob(os.path.join(le.OUT, pat), recursive=True))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _compute_scores(result_path: str) -> Tuple[float, float]:
    data = json.load(open(result_path, "r", encoding="utf-8"))
    tasks = list((data.get("eval") or {}).values())
    if not tasks:
        raise SystemExit(f"[error] no task eval data in {result_path}")

    def pct(key: str, both: Optional[str] = None) -> float:
        vals = []
        for t in tasks:
            n = len(t) or 1
            ok = sum(1 for r in t if r.get(key) == "pass" and (both is None or r.get(both) == "pass"))
            vals.append(ok / n)
        return 100.0 * sum(vals) / len(vals) if vals else 0.0

    plus = round(pct("plus_status", both="base_status"), 1)
    base = round(pct("base_status"), 1)
    return plus, base


def _infer_model_key(args_model: Optional[str], result_path: str) -> Optional[str]:
    if args_model:
        return args_model
    m = re.search(r"/([^/]+)_chatml_greedy_eval_results\.json$", result_path)
    if not m:
        return None
    slug = m.group(1)
    for key, (_hf, mslug, _targets) in le.MODELS.items():
        if mslug == slug:
            return key
    return None


def _find_row(data: Dict[str, Any], name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
    ns = _slug(name)
    for section in ("benchmarked", "pending"):
        rows = data.get(section, [])
        for i, row in enumerate(rows):
            if _slug(str(row.get("name", ""))) == ns:
                return row, section, i
    return None, None, None


def _build_benchmarked_row_from_pending(row: Dict[str, Any], plus: float, base: float) -> Dict[str, Any]:
    params_b = _parse_params_b(row.get("params"))
    fit = row.get("fit") or ("max" if (params_b or 0) >= 12 else "mid")
    return {
        "name": row.get("name"),
        "org": row.get("org"),
        "released": row.get("released"),
        "params": row.get("params"),
        "plus": plus,
        "base": base,
        "fit": fit,
        "memory": _memory_estimate(params_b),
        "local": plus,
        "plus_source": "self-eval",
        "best": row.get("best") or "Fresh local eval result.",
        "note": row.get("note") or "HumanEval scores from local self-evaluation run.",
        "url": row.get("url"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default=None, help="path to *_eval_results.json (default: newest under .eval/out)")
    ap.add_argument("--model", default=None, help="local_eval.py model key (optional; inferred from result filename)")
    ap.add_argument("--name", default=None, help="explicit models.json display name override")
    ap.add_argument("--models", default=MODELS_JSON, help="path to models.json")
    ap.add_argument("--promote-pending", action="store_true", default=True,
                    help="promote pending[] row to benchmarked[] when needed (default: on)")
    ap.add_argument("--no-promote-pending", action="store_false", dest="promote_pending")
    ap.add_argument("--overwrite-plus-base", action="store_true",
                    help="overwrite existing plus/base with local values instead of only setting local")
    ap.add_argument("--dry-run", action="store_true", help="show planned changes only")
    args = ap.parse_args()

    result_path = args.result or _newest_result_path()
    if not result_path:
        raise SystemExit("[error] no eval result file found under .eval/out")
    if not os.path.isfile(result_path):
        raise SystemExit(f"[error] result file not found: {result_path}")

    plus, base = _compute_scores(result_path)
    model_key = _infer_model_key(args.model, result_path)
    if model_key and model_key not in le.MODELS:
        raise SystemExit(f"[error] unknown model key: {model_key}")

    if args.name:
        display_name = args.name
    elif model_key:
        display_name = DISPLAY_NAMES.get(model_key)
        if not display_name:
            hf_id = le.MODELS[model_key][0]
            display_name = hf_id.split("/")[-1]
    else:
        raise SystemExit("[error] could not infer model name; pass --model or --name")

    data = json.load(open(args.models, "r", encoding="utf-8"))
    row, section, idx = _find_row(data, display_name)
    if row is None:
        raise SystemExit(f"[error] model '{display_name}' not found in benchmarked[] or pending[]")

    action = "update"
    if section == "pending":
        if not args.promote_pending:
            raise SystemExit("[error] model is in pending[]; rerun with --promote-pending")
        bench_row = _build_benchmarked_row_from_pending(row, plus, base)
        data["pending"].pop(idx)
        data.setdefault("benchmarked", []).append(bench_row)
        action = "promote"
    else:
        row["local"] = plus
        if row.get("plus") is None or args.overwrite_plus_base:
            row["plus"] = plus
            row["plus_source"] = "self-eval"
        if row.get("base") is None or args.overwrite_plus_base:
            row["base"] = base

    print(f"[import] result: {os.path.relpath(result_path, ROOT)}")
    print(f"[import] model: {display_name}")
    print(f"[import] computed HumanEval+={plus:.1f} HumanEval={base:.1f}")
    print(f"[import] action: {action} ({section} -> benchmarked)")

    if args.dry_run:
        print("[dry-run] no files changed")
        return 0

    with open(args.models, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    bp.rebuild(json_path=args.models)
    print(f"[ok] updated {args.models}")
    print("[ok] rebuilt index.html from models.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())