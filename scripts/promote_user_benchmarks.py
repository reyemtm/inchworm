#!/usr/bin/env python3
"""Promote crowd-verified local evals into the leaderboard.

Groups user_benchmarks.json entries by model; when a model has >= 3 entries
from distinct users, writes the median HumanEval+ into the `local` field of
models.json and rebuilds index.html from it (scripts/build_page.py), then
marks the entries as promoted. --dry-run reports what qualifies without
touching anything.

Usage:
    python3 scripts/promote_user_benchmarks.py [--dry-run] [--threshold N]
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import local_batch as batch  # reuse update_json / rebuild_page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threshold", type=int, default=None,
                    help="override PROMOTION_THRESHOLD")
    args = ap.parse_args()

    path = os.path.join(ROOT, "user_benchmarks.json")
    d = json.load(open(path))
    meta = d["_meta"]
    threshold = args.threshold or meta.get("PROMOTION_THRESHOLD", 3)

    groups = {}
    for e in meta["entries"]:
        groups.setdefault(e["model"], []).append(e)

    promoted, pending = [], []
    for model, entries in sorted(groups.items()):
        users = {e["user"] for e in entries}
        if len(users) >= threshold:
            plus = statistics.median(e["humaneval_plus"] for e in entries)
            promoted.append((model, plus, users, entries))
        else:
            pending.append((model, len(users), entries))

    print(f"{'model':<34} {'users':>5}  status")
    for model, plus, users, _ in promoted:
        print(f"{model:<34} {len(users):>5}  PROMOTE  HE+ median {plus:.1f}")
    for model, nusers, _ in pending:
        print(f"{model:<34} {nusers:>5}  waiting ({threshold - nusers} more user(s) needed)")

    if args.dry_run:
        print("\n[dry-run] no changes made")
        return 0

    changed = False
    for model, plus, users, entries in promoted:
        if batch.update_json(model, plus):
            changed = True
        else:
            print(f"[promote] WARNING: '{model}' not found in models.json benchmarked[] — skipped")
        # tag entries so they aren't double-counted as fresh signal later
        for e in entries:
            e["promoted"] = True
        print(f"[promote] {model}: local={plus:.1f} (median of {len(users)} users)")

    if changed:
        batch.rebuild_page()
        json.dump(d, open(path, "w"), indent=2)
        open(path, "a").write("\n")
        print("[promote] wrote user_benchmarks.json (entries flagged promoted=true) "
              "and rebuilt index.html from models.json")
    else:
        print("[promote] nothing to promote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
