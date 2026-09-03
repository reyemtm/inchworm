#!/usr/bin/env python3
"""scrape_scores.py — nightly refresh of the laptop LLM leaderboard snapshot.

Pulls fresh scores into the `models` array of index.html from license-safe,
machine-readable sources:

  EvalPlus leaderboard data            (HumanEval+ `plus` + HumanEval `base`)
    https://raw.githubusercontent.com/evalplus/evalplus.github.io/main/results.json
    (Apache-2.0 licensed repo; canonical pass@1 for both humaneval and humaneval+)

  BigCodeBench results datasets        (BigCodeBench Full `bcb` + Hard `bcb_hard`)
    https://datasets-server.huggingface.co/rows?dataset=bigcode/bigcodebench-results...
    https://datasets-server.huggingface.co/rows?dataset=bigcode/bigcodebench-hard-results...
    (Apache-2.0; official leaderboard data behind https://bigcode-bench.github.io/.
     `instruct` (brief NL instruction -> code, the leaderboard's "vibe check") is
     preferred; base models without an instruct run fall back to `complete`.)

For any model a source does not cover, the existing snapshot value is kept.
Only the numeric score fields (plus/base/bcb/bcb_hard/bcb_mode) and the snapshot
date are ever touched; editorial fields (best/note/memory/fit/url) live in the
page and stay untouched. The page is never left broken: on any fetch/parse
failure prior values remain.

Usage:
    python3 scripts/scrape_scores.py [--html path/to/index.html] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone

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

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MODELS_RE = re.compile(r"const models = \[(.*?)\];", re.DOTALL)
ENTRY_RE = re.compile(r"\{[^{}]*\}")  # each model entry is a flat object
LEADER_RE = re.compile(r"const leader = \{.*?\};", re.DOTALL)
NAME_RE = re.compile(r"name:'([^']*)'")
PLUS_RE = re.compile(r"plus:([\d.]+)")
BASE_RE = re.compile(r"base:([\d.]+)")
PARAMS_RE = re.compile(r"params:([\d.]+)")
# A model's existing bcb run (bcb_hard is optional — some models have no
# Hard-set run). Replaced wholesale each night; see the rewrite below.
BCB_TAIL_RE = re.compile(r", bcb:[\d.]+(?:, bcb_hard:[\d.]+)?, bcb_mode:'[^']*'")
SNAPSHOT_RE = re.compile(r"(<strong>snapshot</strong>\s*·\s*)\d{1,2} [A-Z][a-z]{2} \d{4}")

# Page model name -> BigCodeBench dataset name (repo-style casing/aliases).
BCB_ALIASES = {
    "CodeGemma-7B-it": "CodeGemma-7B-Instruct",
    "Phi-3.5-mini-instruct": "Phi-3.5-Mini-Instruct",
}

# Page model name -> EvalPlus results.json key (EvalPlus uses repo-style keys).
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
    """Match a page model name to an EvalPlus results.json entry, or None."""
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
    """Match a page model name to BigCodeBench results.

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


def fmt_date(dt: datetime) -> str:
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year}"


def bcb_block(full_v, hard_v, mode) -> str:
    block = f"bcb:{full_v:.1f}"
    if hard_v is not None:
        block += f", bcb_hard:{hard_v:.1f}"
    return block + f", bcb_mode:'{mode}'"


def overall_leader(full):
    """Top model across all sizes under the page's column lens (instruct when
    available, else complete). Returns (row, value) or None."""
    best = None
    for row in full.values():
        v = row.get("instruct")
        if v is None:
            v = row.get("complete")
        if isinstance(v, (int, float)) and (best is None or float(v) > best[1]):
            best = (row, float(v))
    return best


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default="index.html", help="path to index.html")
    parser.add_argument("--dry-run", action="store_true",
                        help="report changes without writing the file")
    args = parser.parse_args()

    with open(args.html, "r", encoding="utf-8") as fh:
        html = fh.read()

    m = MODELS_RE.search(html)
    if not m:
        print("[error] could not locate the models array in index.html", file=sys.stderr)
        sys.exit(1)
    array_text = m.group(1)

    evalplus = load_evalplus()
    bcb_full, bcb_hard = load_bigcodebench()
    entries = list(ENTRY_RE.finditer(array_text))
    if not entries:
        print("[error] could not parse any model entries", file=sys.stderr)
        sys.exit(1)

    updated, bcb_updated, missing = [], [], []
    for ent in entries:
        name_m = NAME_RE.search(ent.group(0))
        if not name_m:
            continue
        name = name_m.group(1)
        params_m = PARAMS_RE.search(ent.group(0))
        params = float(params_m.group(1)) if params_m else None

        row = lookup(evalplus, name, params)
        if row and row.get("pass@1"):
            p = row["pass@1"].get("humaneval+")
            b = row["pass@1"].get("humaneval")
            new_plus = float(p) if isinstance(p, (int, float)) else None
            new_base = float(b) if isinstance(b, (int, float)) else None
        else:
            new_plus = new_base = None
        if new_plus is not None or new_base is not None:
            updated.append((name, new_plus, new_base))
        else:
            missing.append(name)

        fv, hv, mode = bcb_lookup(bcb_full, bcb_hard, name, params)
        if fv is not None:
            bcb_updated.append((name, fv, hv, mode))

    # Rebuild the array. Replace entries in reverse offset order so the
    # offsets of not-yet-processed matches stay valid.
    new_array = array_text
    for ent in reversed(entries):
        name_m = NAME_RE.search(ent.group(0))
        if not name_m:
            continue
        params_m = PARAMS_RE.search(ent.group(0))
        params = float(params_m.group(1)) if params_m else None
        row = lookup(evalplus, name_m.group(1), params)
        text = ent.group(0)
        if row and row.get("pass@1"):
            p = row["pass@1"].get("humaneval+")
            b = row["pass@1"].get("humaneval")
            if isinstance(p, (int, float)):
                text = PLUS_RE.sub(lambda mm, v=float(p): f"plus:{v:.1f}", text, count=1)
            if isinstance(b, (int, float)):
                text = BASE_RE.sub(lambda mm, v=float(b): f"base:{v:.1f}", text, count=1)

        fv, hv, mode = bcb_lookup(bcb_full, bcb_hard, name_m.group(1), params)
        if fv is not None:
            block = bcb_block(fv, hv, mode)
            # Drop any stale bcb run (with or without bcb_hard), then append the
            # fresh block right after the base value — or after params for
            # models with no EvalPlus HumanEval pair.
            text = BCB_TAIL_RE.sub("", text, count=1)
            anchor = re.search(r"base:[\d.]+", text) or re.search(r"params:[\d.]+", text)
            if anchor:
                text = text[:anchor.end()] + f", {block}" + text[anchor.end():]
        new_array = new_array[:ent.start()] + text + new_array[ent.end():]

    new_html = html[:m.start(1)] + new_array + html[m.end(1):]

    # Refresh the pinned overall-leader reference row (top model across all
    # sizes in the BCB dataset) so it tracks nightly updates too.
    leader_old_m = LEADER_RE.search(new_html)
    leader_text = None
    if bcb_full:
        lead = overall_leader(bcb_full)
        if lead:
            row, val = lead
            hard_row = bcb_hard.get(row["model"]) if bcb_hard else None
            hard_v = None
            if hard_row:
                hv = hard_row.get("instruct")
                if hv is None:
                    hv = hard_row.get("complete")
                if isinstance(hv, (int, float)):
                    hard_v = float(hv)
            mode = "instruct" if row.get("instruct") is not None else "complete"
            ep_row = lookup(evalplus, row["model"], None)
            p = b = None
            if ep_row and ep_row.get("pass@1"):
                pp = ep_row["pass@1"].get("humaneval+")
                bb = ep_row["pass@1"].get("humaneval")
                p = float(pp) if isinstance(pp, (int, float)) else None
                b = float(bb) if isinstance(bb, (int, float)) else None
            # Closed API leaders have no EvalPlus row; keep any snapshot values.
            old = leader_old_m.group(0) if leader_old_m else ""
            for cur, old_re in ((p, PLUS_RE), (b, BASE_RE)):
                if cur is None:
                    old_m = old_re.search(old)
                    if old_m:
                        try:
                            cur = float(old_m.group(1))
                        except ValueError:
                            pass
                if cur is None:
                    continue
                if old_re is PLUS_RE:
                    p = cur
                else:
                    b = cur
            size = row.get("size")
            params_txt = (f"{float(size):.1f}" if isinstance(size, (int, float))
                          else "null")
            fmt_or = lambda v: ("null" if v is None else f"{v:.1f}")
            hard_txt = fmt_or(hard_v)
            leader_text = (f"const leader = {{ name:'{row['model']}', "
                           f"org:'{derive_org(row.get('link'))}', params:{params_txt}, "
                           f"plus:{fmt_or(p)}, base:{fmt_or(b)}, bcb:{val:.1f}, "
                           f"bcb_hard:{hard_txt}, bcb_mode:'{mode}', "
                           f"url:'{row.get('link') or ''}' }};")
            if leader_old_m:
                new_html = (new_html[:leader_old_m.start()] + leader_text
                            + new_html[leader_old_m.end():])
            else:
                marker = "const models = ["
                idx = new_html.find(marker)
                new_html = new_html[:idx] + leader_text + "\n    " + new_html[idx:]

    today = datetime.now(timezone.utc)
    new_html, date_changed = SNAPSHOT_RE.subn(
        lambda mm: mm.group(1) + fmt_date(today), new_html, count=1)

    # Report.
    print(f"EvalPlus source: {EVALPLUS_RESULTS}")
    print(f"BigCodeBench source: {BCB_RESULTS_DATASET} + {BCB_HARD_DATASET} "
          f"(datasets-server)")
    print(f"models parsed: {len(entries)} | evalplus updated: {len(updated)} | "
          f"bigcodebench updated: {len(bcb_updated)} | "
          f"no source (snapshot kept): {len(missing)}")
    for name, p, b in updated:
        print(f"  ~ {name}: plus={p} base={b}")
    for name, fv, hv, mode in bcb_updated:
        print(f"  ~ {name}: bcb={fv} bcb_hard={hv} ({mode})")
    for name in missing:
        print(f"  = {name}: no EvalPlus row; snapshot kept")
    if leader_text:
        print(f"overall leader (pinned): {row['model']} bcb={val} ({mode})")
    if date_changed:
        print(f"snapshot date -> {fmt_date(today)}")

    if args.dry_run:
        print("[dry-run] no changes written")
        return

    # Write atomically: temp file in the same dir, then os.replace.
    out_dir = os.path.dirname(os.path.abspath(args.html)) or "."
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".index.html.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(new_html)
        os.replace(tmp, args.html)
        print(f"wrote {args.html}")
    except Exception:
        os.unlink(tmp)
        raise


if __name__ == "__main__":
    main()