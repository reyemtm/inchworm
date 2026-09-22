#!/usr/bin/env python3
"""Discover small coding-relevant models from external rosters and benchmark boards.

This is a read-only discovery aid. It does not edit models.json or index.html;
review its output before adding candidates to models.json's pending[] list.

Sources:
  hf       Hugging Face benchmark boards (datasets tagged benchmark:official).
           The HF leaderboard API is a registry of benchmark datasets, not a
           uniform HumanEval+/MBPP+ score feed. Results from different boards
           must not be merged or compared numerically, so each result is printed
           with its benchmark, verification flag, source URL and parameter count.
  modelfit ModelFit's hardware-compatibility dataset, CC BY 4.0
           (https://modelfit.io/api/dataset/, discovered via /llms.txt):
           141 models with params, quantization, estimatedLoadGb, minRamGb,
           kvKbPerToken and an `ollama run` command. It carries NO scores of any
           kind - every speed figure on the site is a bandwidth estimate, not a
           measurement - so use it to find roster gaps and to fill in the
           `ollama` tag a model needs before tinymark_batch.py will run it.

Usage:
    python3 scripts/discover_candidates.py
    python3 scripts/discover_candidates.py --source modelfit --min-params 1
    python3 scripts/discover_candidates.py --source modelfit --coding-only
    python3 scripts/discover_candidates.py --source modelfit --modelfit-json mf.json
    python3 scripts/discover_candidates.py --source hf --min-score 0
    python3 scripts/discover_candidates.py --source hf --benchmark SWE-bench/SWE-bench_Verified

No packages are required. Public endpoints used:
  https://huggingface.co/api/datasets?filter=benchmark:official
  https://huggingface.co/api/datasets/{dataset_id}/leaderboard
  https://modelfit.io/api/dataset/
"""

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_URL = "https://huggingface.co/api/datasets?filter=benchmark:official&limit=200"
LEADERBOARD_URL = "https://huggingface.co/api/datasets/{}/leaderboard"
MODELFIT_URL = "https://modelfit.io/api/dataset/"
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


# --------------------------------------------------------------------------
# Roster matching (shared: both sources are only useful deduped against us)
# --------------------------------------------------------------------------

def norm_name(value):
    """`Qwen3.5-9B` and `Qwen3.5 9B Instruct` both reduce to comparable tokens."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def base_tag(tag):
    """Drop a quant suffix so `qwen3:14b-q4_K_M` aligns with `qwen3:14b`."""
    return re.sub(r"-(?:i?q\d[\w.]*|f16|bf16)$", "", (tag or "").strip(),
                  flags=re.IGNORECASE).lower()


def load_roster():
    """Our own entries, keyed for matching by tag and by normalized name."""
    with open(os.path.join(ROOT, "models.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    entries = []
    for section in ("benchmarked", "pending"):
        for model in data.get(section, []):
            tags = {model[k].lower() for k in ("ollama", "tinymark_model") if model.get(k)}
            entries.append({
                "name": model.get("name", "?"),
                "section": section,
                "tags": tags,
                "base_tags": {base_tag(t) for t in tags},
                "norm": norm_name(model.get("name")),
                "params": model.get("params"),
                "has_ollama": bool(model.get("ollama")),
            })
    return entries


def params_agree(ours, theirs, tolerance=1.35):
    """Same model implies broadly the same size; 1.35x absorbs 3B-vs-3.7B naming.

    Without this, a normalized-name substring test matches `Phi-4 Mini 3.8B`
    against our 14B `Phi-4` ("phi4" is a prefix of "phi4mini38b"), and short
    names like that are exactly where substring matching is unsafe.
    """
    if not isinstance(ours, (int, float)) or not isinstance(theirs, (int, float)):
        return True
    return ours <= theirs * tolerance and theirs <= ours * tolerance


def match_roster(roster, label, tag, params=None):
    """Return (entry, how) for the roster model this row refers to, else (None, '')."""
    tag = (tag or "").lower()
    if tag:
        for entry in roster:
            if tag in entry["tags"]:
                return entry, "tag"
        stripped = base_tag(tag)
        if stripped:
            for entry in roster:
                if stripped in entry["base_tags"]:
                    return entry, "tag"
    name = norm_name(label)
    if name:
        for entry in roster:
            if entry["norm"] and (entry["norm"] in name or name in entry["norm"]):
                if params_agree(entry.get("params"), params):
                    return entry, "name"
    return None, ""


# --------------------------------------------------------------------------
# Source: Hugging Face benchmark boards
# --------------------------------------------------------------------------

def sweep_hf(args):
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


# --------------------------------------------------------------------------
# Source: ModelFit hardware-compatibility dataset
# --------------------------------------------------------------------------

# ModelFit lists every quantization as its own row ("Gemma 4 12B (Q8)"); the
# base row already carries the default quant, so collapse them by default.
QUANT_VARIANT_RE = re.compile(r"\((?:Q\d[\w.]*|[^)]*bit[^)]*)\)\s*$", re.IGNORECASE)


def tag_of_command(command):
    match = re.match(r"\s*ollama\s+run\s+(\S+)", command or "")
    return match.group(1) if match else None


def modelfit_rows(args, data, roster):
    rows = []
    for model in data.get("models", []):
        if not model.get("runsLocally"):
            continue
        params = model.get("params")
        if not isinstance(params, (int, float)):
            continue
        if not args.min_params <= params <= args.max_params:
            continue
        label = model.get("model", "")
        if not args.include_quants and QUANT_VARIANT_RE.search(label):
            continue
        if args.coding_only and "oding" not in (model.get("bestFor") or ""):
            continue
        tag = tag_of_command(model.get("ollamaCommand"))
        entry, how = match_roster(roster, label, tag, params)
        if entry is None:
            status = "NEW"
        elif not entry["has_ollama"]:
            status = "no ollama tag"
        else:
            status = "in roster"
        rows.append({
            "params": params, "model": label, "family": model.get("family", ""),
            "best_for": model.get("bestFor", ""),
            "load": model.get("estimatedLoadGb"), "ram": model.get("minRamGb"),
            "kv": model.get("kvKbPerToken"), "tag": tag or "-",
            "status": status, "how": how, "entry": entry,
        })
    rows.sort(key=lambda row: (row["params"], row["model"]))
    return rows


def sweep_modelfit(args, roster):
    if args.modelfit_json:
        with open(args.modelfit_json, encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = get_json(MODELFIT_URL)

    rows = modelfit_rows(args, data, roster)
    print(f"ModelFit {data.get('source', '')} "
          f"| updated {data.get('updated', '?')} "
          f"| {data.get('license', '?').split(' (')[0]} "
          f"| {data.get('counts', {}).get('local', '?')} local models "
          f"| NO scores (sizing + tags only)")
    print()
    if not rows:
        print(f"No local models in the {args.min_params}-{args.max_params}B band.")
        return 0

    print("params | model | family | best for | load GB | min RAM | ollama tag | status")
    print("-" * 130)
    for row in rows:
        print(f"{row['params']:>5}B | {row['model'][:34]:<34} | {row['family'][:9]:<9} | "
              f"{row['best_for'][:26]:<26} | {str(row['load']):>7} | {str(row['ram']):>3} GB | "
              f"{row['tag']:<30} | {row['status']}")

    new = [row for row in rows if row["status"] == "NEW"]
    untagged = [row for row in rows if row["status"] == "no ollama tag"]
    print(f"\n{len(rows)} local models in band; {len(new)} new to models.json, "
          f"{len(untagged)} already ours but missing an `ollama` tag.")
    if new:
        print("\nNot in models.json - review, then add to pending[] or benchmarked[]:")
        for row in new:
            print(f"  {row['params']:>5}B  {row['model']}  ({row['best_for']})")
    if untagged:
        print("\nAdd these tags to make tinymark_batch.py pick the model up:")
        for row in untagged:
            print(f"  {row['tag']:<32} -> {row['entry']['name']} "
                  f"[{row['entry']['section']}] (matched by {row['how']})")
    print("\nEstimates, not measurements; no benchmark scores in this source.")
    return 0


SOURCES = {"hf": sweep_hf, "modelfit": sweep_modelfit}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", default=[], choices=sorted(SOURCES),
                    help="discovery source(s) to run; repeatable (default: all)")
    ap.add_argument("--max-params", type=float, default=14,
                    help="maximum parameter count in billions (default: 14)")
    ap.add_argument("--min-params", type=float, default=0,
                    help="minimum parameter count in billions (ModelFit; default: 0)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="optional board-local minimum score (hf only)")
    ap.add_argument("--benchmark", action="append", default=[],
                    help="only inspect this dataset id; may be repeated (hf only)")
    ap.add_argument("--coding-only", action="store_true",
                    help="ModelFit rows whose bestFor mentions coding")
    ap.add_argument("--include-quants", action="store_true",
                    help="keep per-quantization ModelFit rows (default: collapse them)")
    ap.add_argument("--modelfit-json", default="",
                    help="read a saved copy of the dataset instead of fetching it")
    args = ap.parse_args()

    sources = args.source or sorted(SOURCES)
    roster = load_roster()
    status = 0
    for name in sources:
        if len(sources) > 1:
            print(f"\n{'=' * 130}\nSOURCE: {name}\n{'=' * 130}")
        status |= SOURCES[name](args, roster) if name == "modelfit" else SOURCES[name](args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
