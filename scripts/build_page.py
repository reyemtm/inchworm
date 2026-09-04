#!/usr/bin/env python3
"""build_page.py — regenerate index.html's data layer from models.json.

models.json is the single source of truth for the page's roster:

  benchmarked              the "Benchmarked" tab: models with score rows
  pending                  the "Pending Eval" tab: newer models awaiting a
                           comparable run (scores render as —)
  candidates_no_benchmarks editorial discovery pool (never rendered)

Every run regenerates the two inline data consts in index.html (`models`,
`pending`) from this file and stamps the "catalog · checked <date>"
badge from _meta.snapshot. Nothing else in the page is touched — styles,
markup, and app JS stay human-authored. The shipped page therefore remains a
single self-contained file with no runtime fetch; models.json is a dev-time
input, exactly like user_benchmarks.json.

The data consts use a flat single-quoted JS object format with one fixed
field order; the serializers here are the only place that format is produced.
Keep entries flat (no nested objects) so the regex tooling in
scripts/scrape_scores.py keeps working.

Usage:
    python3 scripts/build_page.py            # models.json -> index.html
    python3 scripts/build_page.py --adopt    # rescue: parse index.html's data
                                             # (benchmarked + the Pending
                                             # Eval table) back into models.json
                                             # first, then rebuild index.html
    python3 scripts/build_page.py --dry-run  # report only, write nothing
"""

import argparse
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HTML = os.path.join(ROOT, "index.html")
DEFAULT_JSON = os.path.join(ROOT, "models.json")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_LOOKUP = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}

MODELS_RE = re.compile(r"const models = \[(.*?)\];", re.DOTALL)
PENDING_RE = re.compile(r"const pending = \[(.*?)\];", re.DOTALL)
CATALOG_DATE_RE = re.compile(r"(<strong>catalog</strong>\s*·\s*checked\s*)"
                             r"\d{1,2} [A-Z][a-z]{2} \d{4}")
ENTRY_RE = re.compile(r"\{[^{}]*\}")
WATCHLIST_TBODY_RE = re.compile(
    r"<tbody>\s*(<tr><td class=\"rank\">.*?)</tbody>", re.DOTALL)
WATCHLIST_ROW_RE = re.compile(
    r"<tr><td class=\"rank\">\d+</td>"
    r"<td><span class=\"model-name\">(.*?)</span>"
    r"<span class=\"model-org\">(.*?)</span></td>"
    r"<td class=\"mono\">(.*?)</td>"
    r"<td class=\"mono\">(.*?)</td>"
    r"<td>.*?</td>"
    r"<td><span class=\"fit (\\w+)\">.*?</span></td></tr>", re.DOTALL)

_FIELD_VALUE = re.compile(r"(?:^|[,{]\s*)(\w+):(?:'((?:[^'\\]|\\.)*)'|([\d.]+)|null)")


# ----------------------------------------------------------------- js fmt ----
def js_str(value):
    """Single-quoted JS string literal, or 'null' for None."""
    if value is None:
        return "null"
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def js_num(value):
    """One-decimal JS number literal, or 'null' for None."""
    if value is None:
        return "null"
    return f"{float(value):.1f}"


def js_params(value):
    """JS literal for a param count: ints/floats stay bare, display strings
    (e.g. pending '≤14B') are quoted."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return js_str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def row_to_js(row):
    """Serialize one benchmarked model to the page's flat JS object format.

    Fixed field order is load-bearing: scripts/local_batch.py looks for
    `local:` immediately before `best:`, and scripts/scrape_scores.py replaces
    the bcb block in place.
    """
    parts = [f"name:{js_str(row.get('name'))}",
             f"org:{js_str(row.get('org'))}",
             f"released:{js_str(row.get('released'))}",
             f"params:{js_params(row.get('params'))}",
             f"plus:{js_num(row.get('plus'))}",
             f"base:{js_num(row.get('base'))}"]
    if row.get("bcb") is not None:
        parts.append(f"bcb:{js_num(row.get('bcb'))}")
        if row.get("bcb_hard") is not None:
            parts.append(f"bcb_hard:{js_num(row.get('bcb_hard'))}")
        parts.append(f"bcb_mode:{js_str(row.get('bcb_mode'))}")
    parts.append(f"fit:{js_str(row.get('fit'))}")
    parts.append(f"memory:{js_str(row.get('memory'))}")
    if row.get("local") is not None:
        parts.append(f"local:{js_num(row.get('local'))}")
    if row.get("plus_source") is not None:
        parts.append(f"plus_source:{js_str(row.get('plus_source'))}")
    parts.append(f"best:{js_str(row.get('best'))}")
    parts.append(f"note:{js_str(row.get('note'))}")
    parts.append(f"url:{js_str(row.get('url'))}")
    return "{ " + ", ".join(parts) + " }"


def pending_to_js(row):
    """Serialize one pending-eval model (params may be a display string).

    Optional flat fields: pub_bench/pub_score hold a card-reported
    benchmark (e.g. LiveCodeBench) the row publishes for reference.
    """
    parts = [f"name:{js_str(row.get('name'))}",
             f"org:{js_str(row.get('org'))}",
             f"released:{js_str(row.get('released'))}",
             f"params:{js_params(row.get('params'))}",
             f"fit:{js_str(row.get('fit'))}"]
    if row.get("pub_score") is not None:
        parts.append(f"pub_bench:{js_str(row.get('pub_bench'))}")
        parts.append(f"pub_score:{js_num(row.get('pub_score'))}")
    return "{ " + ", ".join(parts) + " }"


def _const_block(name, rows, fmt):
    """Full const text, e.g. '    const models = [\\n      {...},\\n    ];'."""
    body = "\n".join(f"      {fmt(r)}," for r in rows)
    if body:
        return f"    const {name} = [\n{body}\n    ];"
    return f"    const {name} = [\n    ];"


def _interior(block_text):
    """Text between the const's `[` and the closing `];`."""
    return block_text.split("[", 1)[1].rsplit("]", 1)[0]


# ------------------------------------------------------------- html rebuild ----
def rebuild(html_path=DEFAULT_HTML, json_path=DEFAULT_JSON, dry_run=False):
    """Regenerate the data consts in index.html from models.json.

    Idempotent: a run against an already-generated page changes nothing.
    Returns a list of human-readable change lines (empty when unchanged).
    """
    data = json.load(open(json_path, encoding="utf-8"))
    html = open(html_path, encoding="utf-8").read()
    changes = []

    # models
    block = _const_block("models", data.get("benchmarked", []), row_to_js)
    m = MODELS_RE.search(html)
    if m is None:
        raise SystemExit("[error] could not locate `const models` in index.html")
    if m.group(1) != _interior(block):
        html = html[:m.start(1)] + _interior(block) + html[m.end(1):]
        changes.append(f"models: {len(data.get('benchmarked', []))} entries")

    # pending
    pblock = _const_block("pending", data.get("pending", []), pending_to_js)
    p = PENDING_RE.search(html)
    if p is None:
        anchor = MODELS_RE.search(html)
        html = html[:anchor.end()] + "\n" + pblock + html[anchor.end():]
        changes.append(f"pending: inserted ({len(data.get('pending', []))} entries)")
    elif p.group(1) != _interior(pblock):
        html = html[:p.start(1)] + _interior(pblock) + html[p.end(1):]
        changes.append(f"pending: {len(data.get('pending', []))} entries")

    # catalog badge date from _meta.snapshot (ISO yyyy-mm-dd)
    snap = (data.get("_meta") or {}).get("snapshot")
    if snap:
        try:
            year, month, day = snap.split("-")
            stamp = f"{int(day)} {MONTHS[int(month) - 1]} {year}"
        except (ValueError, IndexError):
            stamp = None
        if stamp:
            dated = CATALOG_DATE_RE.sub(
                lambda mm: mm.group(1) + stamp, html, count=1)
            if dated != html:
                changes.append(f"catalog badge date -> {stamp}")
                html = dated

    if not changes:
        return changes
    if dry_run:
        print("[dry-run] would regenerate: " + "; ".join(changes))
        return changes
    _atomic_write(html_path, html)
    print("wrote " + html_path + " (" + "; ".join(changes) + ")")
    return changes


def _atomic_write(path, text):
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".tmp.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


# ------------------------------------------------------------------ adopt ----
def _js_unescape(text):
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _parse_entry_fields(text):
    row = {}
    for mtch in _FIELD_VALUE.finditer(text):
        field, sval, nval = mtch.group(1), mtch.group(2), mtch.group(3)
        if sval is not None:
            row[field] = _js_unescape(sval)
        elif nval is not None:
            row[field] = float(nval) if "." in nval else int(nval)
        else:
            row[field] = None
    return row


def adopt(html_path=DEFAULT_HTML, json_path=DEFAULT_JSON, dry_run=False):
    """Rescue: parse index.html's inline data back into models.json, then rebuild.

    Keeps any existing top-level sections it does not regenerate
    (e.g. candidates_no_benchmarks) and updates _meta.snapshot from the
    catalog badge date.
    """
    html = open(html_path, encoding="utf-8").read()
    old = {}
    if os.path.isfile(json_path):
        old = json.load(open(json_path, encoding="utf-8"))

    data = {"_meta": dict(old.get("_meta", {})),
            "benchmarked": [],
            "pending": [],
            "candidates_no_benchmarks": old.get("candidates_no_benchmarks", {})}

    # Snapshot date from the catalog badge ("checked 4 Sep 2026").
    date_m = re.search(r"checked\s+(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})", html)
    if date_m:
        day, mon, year = date_m.groups()
        data["_meta"]["snapshot"] = \
            f"{year}-{MONTHS_LOOKUP[mon.lower()]:02d}-{int(day):02d}"

    mm = MODELS_RE.search(html)
    if mm:
        data["benchmarked"] = [_parse_entry_fields(e.group(0))
                               for e in ENTRY_RE.finditer(mm.group(1))]

    pm = PENDING_RE.search(html)
    if pm:
        data["pending"] = [_parse_entry_fields(e.group(0))
                           for e in ENTRY_RE.finditer(pm.group(1))]
    else:
        wm = WATCHLIST_TBODY_RE.search(html)
        if wm:
            for rm in WATCHLIST_ROW_RE.finditer(wm.group(1)):
                name, org, params, released, fit = rm.groups()
                params = None if params in ("—", "–", "") else params
                released = None if released in ("—", "–", "") else released
                data["pending"].append({"name": name, "org": org,
                                        "released": released,
                                        "params": params, "fit": fit})

    data["benchmarked"] = [r for r in data["benchmarked"] if r.get("name")]
    data["pending"] = [r for r in data["pending"] if r.get("name")]

    if not data["benchmarked"]:
        raise SystemExit("[error] adopt parsed no benchmarked models — giving up")

    for row in data["benchmarked"]:
        for key, value in list(row.items()):
            if isinstance(value, float) and key != "params":
                row[key] = round(value, 1)
    for row in data["pending"]:
        for key, value in list(row.items()):
            if isinstance(value, float) and key != "params":
                row[key] = round(value, 1)

    data["_meta"]["description"] = (
        "Inchworm roster (canonical). Edit this file, then run "
        "`python3 scripts/build_page.py` to regenerate index.html's data layer. "
        "benchmarked = the 'Benchmarked' tab rows; pending = the 'Pending Eval' "
        "tab (newer models, no comparable scores yet); candidates_no_benchmarks "
        "is a discovery pool never rendered.")
    data["_meta"]["build"] = \
        "index.html data consts generated by scripts/build_page.py"

    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if dry_run:
        print(f"[dry-run] would write {json_path} "
              f"({len(data['benchmarked'])} benchmarked, "
              f"{len(data['pending'])} pending)")
        return
    _atomic_write(json_path, text)
    print(f"wrote {json_path} ({len(data['benchmarked'])} benchmarked, "
          f"{len(data['pending'])} pending)")
    rebuild(html_path, json_path, dry_run=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default=DEFAULT_HTML, help="path to index.html")
    ap.add_argument("--json", default=DEFAULT_JSON, help="path to models.json")
    ap.add_argument("--adopt", action="store_true",
                    help="rescue index.html's inline data back into models.json first")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, write nothing")
    args = ap.parse_args()

    if args.adopt:
        adopt(args.html, args.json, dry_run=args.dry_run)
    else:
        rebuild(args.html, args.json, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
