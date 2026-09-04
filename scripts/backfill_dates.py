#!/usr/bin/env python3
"""backfill_dates.py — fill missing `released` dates in models.json from Hugging Face.

models.json rows carry an optional `released` field ("YYYY-MM", or just
"YYYY"). This helper looks up any model that is missing one and records the
month its weights appeared publicly, from the HF model API's `createdAt`
timestamp on the linked model card — the same repo each `url` points at.

It only ever *fills* missing values; an existing `released` is never
overwritten, so editorial corrections (e.g. a repo that predates the public
launch) are safe. Models with no resolvable HF repo (non-HF sources, pending
rows without a card yet) are listed as skipped.

Sources:
  Hugging Face model API
    https://huggingface.co/api/models/{owner}/{name}   (public JSON; `createdAt`)

Usage:
    python3 scripts/backfill_dates.py [--json path/to/models.json] [--dry-run]
    python3 scripts/backfill_dates.py --repo "Qwen3-14B=Qwen/Qwen3-14B" [--dry-run]
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_page as bp  # noqa: E402  (atomic write + rebuild index.html)

API = "https://huggingface.co/api/models/{repo}"
USER_AGENT = "laptop-llm-leaderboard-backfill/1.0"

# Explicit name -> HF repo fallbacks for rows with no HF `url` (e.g. pending
# rows not yet benchmarked). Kept small; prefer `--repo` for one-off fills.
HINTS = {}


def fetch_repo_created_at(repo: str):
    """Return the repo's createdAt as 'YYYY-MM', or None on any failure."""
    req = urllib.request.Request(
        API.format(repo=urllib.parse.quote(repo, safe="/")),
        headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - report and skip
        print(f"  ! could not fetch {repo}: {exc}", file=sys.stderr)
        return None
    created = (meta.get("createdAt") or "")[:7]
    return created or None


def repo_from_url(url):
    """HF repo id ('owner/name') from a model-card url, else None."""
    if not url or "huggingface.co/" not in url:
        return None
    rest = url.split("huggingface.co/", 1)[1].split("?", 1)[0].strip("/")
    parts = rest.split("/")
    if len(parts) != 2 or parts[0] in ("datasets", "spaces", "models", "api"):
        return None
    return f"{parts[0]}/{parts[1]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=bp.DEFAULT_JSON, help="path to models.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be filled without writing anything")
    parser.add_argument("--repo", action="append", default=[], metavar="NAME=owner/repo",
                        help="explicit HF repo for a model that has no HF url "
                             "(repeatable); takes precedence over the url")
    args = parser.parse_args()

    hints = dict(HINTS)
    for item in args.repo:
        if "=" in item:
            name, _, repo = item.partition("=")
            hints[name.strip()] = repo.strip()

    data = json.load(open(args.json, encoding="utf-8"))
    rows = list(data.get("benchmarked", [])) + list(data.get("pending", []))

    filled, skipped = [], []
    for row in rows:
        name = row.get("name")
        if not name:
            continue
        if row.get("released"):
            continue  # never overwrite an existing date
        repo = hints.get(name) or repo_from_url(row.get("url"))
        if not repo:
            skipped.append((name, "no resolvable HF repo"))
            continue
        month = fetch_repo_created_at(repo)
        if not month:
            skipped.append((name, f"no createdAt for {repo}"))
            continue
        row["released"] = month
        filled.append((name, repo, month))

    print(f"models checked: {len(rows)} | would fill: {len(filled)} | "
          f"skipped: {len(skipped)}")
    for name, repo, month in filled:
        print(f"  ~ {name}: released -> {month} (from {repo})")
    for name, why in skipped:
        print(f"  = {name}: {why}")

    if args.dry_run or not filled:
        if args.dry_run and filled:
            print("[dry-run] no changes written")
        return 0

    bp._atomic_write(args.json, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.json}")
    bp.rebuild(json_path=args.json, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
