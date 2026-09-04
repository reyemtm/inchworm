#!/usr/bin/env python3
"""Discover small coding-relevant models from Hugging Face benchmark boards.

This is a read-only discovery aid. It does not edit models.json or index.html;
review its output before adding candidates to models.json's pending[] list.

The HF leaderboard API is a registry of benchmark datasets, not a uniform
HumanEval+/MBPP+ score feed. Results from different boards must not be merged
or compared numerically. This script therefore prints each result with its
benchmark, verification flag, source URL, and parameter count for editorial
review.

Usage:
    python3 scripts/discover_candidates.py
    python3 scripts/discover_candidates.py --max-params 14 --min-score 0
    python3 scripts/discover_candidates.py --benchmark SWE-bench/SWE-bench_Verified

No packages are required. The script uses the public endpoints:
  /api/datasets?filter=benchmark:official
  /api/datasets/{dataset_id}/leaderboard
"""

import argparse
import json
import urllib.parse
import urllib.request

DATASETS_URL = "https://huggingface.co/api/datasets?filter=benchmark:official&limit=200"
LEADERBOARD_URL = "https://huggingface.co/api/datasets/{}/leaderboard"
USER_AGENT = "inchworm-candidate-discovery/1.0"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def params_in_billions(entry):
    value = entry.get("num_parameters")
    if isinstance(value, (int, float)):
        return float(value) / 1e9
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-params", type=float, default=14,
                    help="maximum parameter count in billions (default: 14)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="optional board-local minimum score")
    ap.add_argument("--benchmark", action="append", default=[],
                    help="only inspect this dataset id; may be repeated")
    args = ap.parse_args()

    datasets = get_json(DATASETS_URL)
    ids = args.benchmark or [d.get("id") for d in datasets if d.get("id")]
    found = []
    for dataset_id in ids:
        try:
            rows = get_json(LEADERBOARD_URL.format(urllib.parse.quote(dataset_id, safe="/")))
        except Exception as exc:  # one broken board should not stop discovery
            print(f"[skip] {dataset_id}: {exc}")
            continue
        for row in rows:
            params = params_in_billions(row)
            score = row.get("value")
            if params is None or params > args.max_params:
                continue
            if args.min_score is not None and not isinstance(score, (int, float)):
                continue
            if args.min_score is not None and score < args.min_score:
                continue
            source = row.get("source") or {}
            found.append((dataset_id, params, score, row.get("verified"),
                          row.get("modelId") or row.get("model_id"),
                          source.get("url", "")))

    found.sort(key=lambda item: (item[2] is None, -(item[2] or 0), item[0], item[4] or ""))
    if not found:
        print("No entries with parameter counts at or below", args.max_params, "B")
        return 0

    print("benchmark | params | score | verified | model | source")
    print("-" * 120)
    for benchmark, params, score, verified, model, source in found:
        print(f"{benchmark} | {params:.2f}B | {score!s} | {verified!s:<5} | "
              f"{model or '?'} | {source}")
    print(f"\n{len(found)} qualifying board entries.")
    print("Review manually; scores across different benchmarks are not comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
