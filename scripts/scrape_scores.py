#!/usr/bin/env python3
"""scrape_scores.py — refresh benchmark scores in models.json, then rebuild the page.

models.json is the single source of truth for the roster. This scraper only
touches numeric score fields there — `plus`/`base` from EvalPlus and
`bcb`/`bcb_hard`/`bcb_mode` from BigCodeBench
then calls scripts/build_page.py to regenerate index.html's data consts from
the updated models.json. Snapshot values are kept for any model a source does
not cover, so the page is never left broken.

Sources (license-safe, machine-readable):

  EvalPlus leaderboard data            (HumanEval+ `plus` + HumanEval `base`)
    https://raw.githubusercontent.com/evalplus/evalplus.github.io/main/results.json
    (Apache-2.0 licensed repo; canonical pass@1 for both humaneval and humaneval+)

  BigCodeBench results datasets        (BigCodeBench Full `bcb` + Hard `bcb_hard`)
    https://datasets-server.huggingface.co/rows?dataset=bigcode/bigcodebench-results...
    https://datasets-server.huggingface.co/rows?dataset=bigcode/bigcodebench-hard-results...
    (Apache-2.0; official leaderboard data behind https://bigcode-bench.github.io/.
     `instruct` (brief NL instruction -> code, the leaderboard's "vibe check") is
     preferred; base models without an instruct run fall back to `complete`.)

Usage:
    python3 scripts/scrape_scores.py [--json path/to/models.json] [--dry-run]
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_page as bp  # noqa: E402  (rebuild index.html from models.json)

EVALPLUS_RESULTS = (
    "https://raw.githubusercontent.com/evalplus/evalplus.github.io/main/results.json"
)
BCB_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset={dataset}&config=default&split=train&offset={offset}&length=100"
)
BCB_RESULTS_DATASET = "bigcode/bigcodebench-results"
BCB_HARD_DATASET = "bigcode/bigcodebench-hard-results"
USER_AGENT = "laptop-llm-leaderboard-scraper/1.0 (nightly snapshot refresh)"

# models.json model name -> BigCodeBench dataset name (repo-style casing/aliases).
BCB_ALIASES = {
    "CodeGemma-7B-it": "CodeGemma-7B-Instruct",
    "Phi-3.5-mini-instruct": "Phi-3.5-Mini-Instruct",
}

# models.json model name -> EvalPlus results.json key (EvalPlus uses repo-style keys).
EP_ALIASES = {
    "Llama-3.1-8B-Instruct": "Llama3.1-8B-instruct",
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def load_evalplus():
    """Return the results.json dict, or None (caller falls back to snapshot)."""
    try:
        data = json.loads(fetch(EVALPLUS_RESULTS))
        if not isinstance(data, dict):
            raise ValueError("results.json is not an object")
        return data
    except Exception as exc:  # noqa: BLE001 - never fail the run over a fetch
        print(f"[warn] could not fetch EvalPlus results.json: {exc}", file=sys.stderr)
        return None


def _bcb_fetch_all(dataset: str) -> dict:
    """Fetch every row of a BigCodeBench results dataset, keyed by model name."""
    rows, offset = {}, 0
    while True:
        page = json.loads(fetch(BCB_ROWS.format(dataset=dataset, offset=offset)))
        page_rows = page.get("rows", [])
        for item in page_rows:
            row = item.get("row", {})
            if row.get("model"):
                rows[row["model"]] = row
        offset += 100
        if len(page_rows) < 100:
            break
    return rows


def load_bigcodebench():
    """Return (full, hard) dicts keyed by model name, or (None, None) on failure."""
    try:
        full = _bcb_fetch_all(BCB_RESULTS_DATASET)
        hard = _bcb_fetch_all(BCB_HARD_DATASET)
        return full, hard
    except Exception as exc:  # noqa: BLE001 - never fail the run over a fetch
        print(f"[warn] could not fetch BigCodeBench results: {exc}", file=sys.stderr)
        return None, None


def lookup(evalplus, name: str, params):
    """Match a roster model name to an EvalPlus results.json entry, or None."""
    if evalplus is None:
        return None
    key = EP_ALIASES.get(name, name)
    row = evalplus.get(key)
    if row is None:
        for k, value in evalplus.items():
            if k.lower() == key.lower():
                row = value
                break
    if row is None:
        return None
    if row.get("prompted") is not True:
        return None
    size = row.get("size")
    if params is not None and isinstance(size, (int, float)) and abs(size - params) > 2:
        return None
    return row


def bcb_lookup(full, hard, name: str, params):
    """Match a roster model name to BigCodeBench results.

    Returns (full_v, hard_v, mode) where mode is 'instruct' when the model has
    an instruct-run score (preferred) and 'complete' for base models, or
    (None, None, None) when there is no safe match. Size-guarded like EvalPlus.
    """
    if full is None:
        return None, None, None
    key = BCB_ALIASES.get(name, name)
    row = full.get(key)
    if row is None:
        for k, v in full.items():
            if k.lower() == key.lower():
                row = v
                break
    if row is None:
        return None, None, None
    size = row.get("size")
    if params is not None and isinstance(size, (int, float)) and abs(size - params) > 2:
        return None, None, None

    def pick(r):
        if not r:
            return None, None
        v = r.get("instruct")
        mode = "instruct"
        if v is None:
            v = r.get("complete")
            mode = "complete"
        if not isinstance(v, (int, float)):
            return None, None
        return float(v), mode

    full_v, mode = pick(row)
    hard_row = None
    if hard is not None:
        hard_row = hard.get(key)
        if hard_row is None:
            for k, v in hard.items():
                if k.lower() == key.lower():
                    hard_row = v
                    break
    hard_v, _ = pick(hard_row)
    return full_v, hard_v, mode


def derive_org(link):
    """Best-effort org label from a model link (BCB rows carry no org field)."""
    if not link:
        return "?"
    if "huggingface.co/" in link:
        org = link.split("huggingface.co/", 1)[1].split("/", 1)[0]
        return {"01-ai": "01.AI", "deepseek-ai": "DeepSeek",
                "meta-llama": "Meta", "bigcode": "BigCode"}.get(org, org)
    for frag, org in (("openai.com", "OpenAI"), ("anthropic.com", "Anthropic"),
                      ("x.com", "xAI"), ("blog.google", "Google"),
                      ("google.com", "Google")):
        if frag in link:
            return org
    return link.split("//", 1)[-1].split("/", 1)[0]


def _round1(value):
    return round(float(value), 1) if value is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=bp.DEFAULT_JSON, help="path to models.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="report changes without writing either file")
    args = parser.parse_args()

    data = json.load(open(args.json, encoding="utf-8"))

    evalplus = load_evalplus()
    bcb_full, bcb_hard = load_bigcodebench()

    updated, bcb_updated, missing = [], [], []
    for row in data["benchmarked"]:
        name = row.get("name")
        if not name:
            continue
        params = row.get("params")
        ep_row = lookup(evalplus, name, params)
        if ep_row and ep_row.get("pass@1"):
            p = ep_row["pass@1"].get("humaneval+")
            b = ep_row["pass@1"].get("humaneval")
            new_plus = _round1(p) if isinstance(p, (int, float)) else None
            new_base = _round1(b) if isinstance(b, (int, float)) else None
        else:
            new_plus = new_base = None
        if new_plus is not None or new_base is not None:
            updated.append((name, new_plus, new_base))
            row["plus"] = new_plus
            row["base"] = new_base
        else:
            missing.append(name)

        fv, hv, mode = bcb_lookup(bcb_full, bcb_hard, name, params)
        if fv is not None:
            bcb_updated.append((name, fv, hv, mode))
            row["bcb"] = _round1(fv)
            row["bcb_hard"] = _round1(hv)
            row["bcb_mode"] = mode

    today = datetime.now(timezone.utc).date().isoformat()
    data["_meta"]["snapshot"] = today

    print(f"EvalPlus source: {EVALPLUS_RESULTS}")
    print(f"BigCodeBench source: {BCB_RESULTS_DATASET} + {BCB_HARD_DATASET} "
          f"(datasets-server)")
    print(f"models: {len(data['benchmarked'])} | evalplus updated: {len(updated)} | "
          f"bigcodebench updated: {len(bcb_updated)} | "
          f"no source (snapshot kept): {len(missing)}")
    for name, p, b in updated:
        print(f"  ~ {name}: plus={p} base={b}")
    for name, fv, hv, mode in bcb_updated:
        print(f"  ~ {name}: bcb={fv} bcb_hard={hv} ({mode})")
    for name in missing:
        print(f"  = {name}: no EvalPlus row; snapshot kept")
    print(f"models.json snapshot date -> {today}")

    if args.dry_run:
        print("[dry-run] no changes written")
        return 0

    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    bp._atomic_write(args.json, text)
    print(f"wrote {args.json}")
    bp.rebuild(json_path=args.json, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
