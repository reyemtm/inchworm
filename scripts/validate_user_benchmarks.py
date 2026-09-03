#!/usr/bin/env python3
"""Validate user_benchmarks.json (stdlib-only; used by CI and pre-submit).

Checks structure, required/optional fields, types, value ranges, dates,
duplicate (user, model) entries, and that the PR author's username appears
among new-entry users when GITHUB_PR_ACTOR is set (set by the workflow).
Exit 0 = valid, 1 = invalid (errors printed).
"""
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "user_benchmarks.json")

REQUIRED = {"model", "hf_id", "user", "date", "hardware", "precision",
            "runtime", "humaneval_plus"}
OPTIONAL = {"humaneval", "notes"}
KNOWN_RUNTIMES = {"transformers", "llama.cpp", "ollama", "vllm", "mlx", "other"}


def fail(msg, errors):
    errors.append(msg)


def main():
    errors = []
    try:
        d = json.load(open(PATH))
    except Exception as e:
        print(f"[validate] user_benchmarks.json is not valid JSON: {e}")
        return 1

    meta = d.get("_meta") or {}
    entries = meta.get("entries")
    if not isinstance(entries, list):
        fail("`_meta.entries` must be a list", errors)
        entries = []

    seen = set()
    for i, e in enumerate(entries):
        ctx = f"entries[{i}]"
        if not isinstance(e, dict):
            fail(f"{ctx}: not an object", errors)
            continue
        missing = REQUIRED - set(e)
        if missing:
            fail(f"{ctx}: missing required fields: {sorted(missing)}", errors)
        unknown = set(e) - REQUIRED - OPTIONAL
        if unknown:
            fail(f"{ctx}: unknown fields: {sorted(unknown)}", errors)

        hp = e.get("humaneval_plus")
        if not isinstance(hp, (int, float)) or not (0 <= hp <= 100):
            fail(f"{ctx}: humaneval_plus must be a number in 0..100, got {hp!r}", errors)
        he = e.get("humaneval")
        if he is not None and not (isinstance(he, (int, float)) and 0 <= he <= 100):
            fail(f"{ctx}: humaneval must be a number in 0..100, got {he!r}", errors)
        if isinstance(hp, (int, float)) and isinstance(he, (int, float)) and he < hp:
            # passing fewer strict tests is harder, so HE >= HE+ always holds
            fail(f"{ctx}: humaneval ({he}) < humaneval_plus ({hp}) — impossible", errors)

        if e.get("runtime") not in KNOWN_RUNTIMES:
            fail(f"{ctx}: runtime must be one of {sorted(KNOWN_RUNTIMES)}", errors)

        try:
            datetime.date.fromisoformat(e.get("date", ""))
        except (TypeError, ValueError):
            fail(f"{ctx}: date must be ISO YYYY-MM-DD, got {e.get('date')!r}", errors)

        for f in ("model", "hf_id", "user", "hardware", "precision"):
            v = e.get(f)
            if not isinstance(v, str) or not v.strip():
                fail(f"{ctx}: {f} must be a non-empty string", errors)

        key = (e.get("user", "").lower(), e.get("model", ""))
        if key in seen:
            fail(f"{ctx}: duplicate (user, model) pair {key} — re-runs replace the earlier entry", errors)
        seen.add(key)

    # In a PR context, require the PR author to have authored something.
    actor = os.environ.get("GITHUB_PR_ACTOR")
    if actor and entries:
        users = {e.get("user", "").lower() for e in entries if isinstance(e, dict)}
        if actor.lower() not in users:
            fail(f"PR author '{actor}' must appear as `user` on at least one entry "
                 f"(found: {sorted(users)})", errors)

    if errors:
        print(f"[validate] {len(errors)} problem(s) in user_benchmarks.json:")
        for e in errors:
            print("  -", e)
        return 1
    print(f"[validate] OK — {len(entries)} entr{'y' if len(entries)==1 else 'ies'} valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
