#!/usr/bin/env python3
"""sweep_new_models.py — find recent small (≤14B) text-generation models for review.

Read-only discovery aid for the Pending Eval tab. Lists the newest
text-generation models on Hugging Face (optionally scoped to known model
orgs), resolves each one's real parameter count from the model API's
`safetensors.total`, keeps only entries at or below --max-params, drops
duplicate/quant variants of the same release, and prints a triage table for
editorial review. It never edits models.json or index.html.

Models already present in the roster (benchmarked[], pending[], or the
candidates pool) are skipped, and entries the frozen EvalPlus/BigCodeBench
feeds already cover are flagged — those are benchmark-qualified, so they are
candidates for benchmarked[] rather than the pending list.

Sources (public JSON, no packages):
  /api/models?author=...&pipeline_tag=text-generation&sort=createdAt&direction=-1
  /api/models/{id}                                   (safetensors.total)
  EvalPlus results.json + BigCodeBench rows          (via scrape_scores helpers)

Usage:
    python3 scripts/sweep_new_models.py                     # tracked orgs
    python3 scripts/sweep_new_models.py --orgs Qwen,google  # just these orgs
    python3 scripts/sweep_new_models.py --limit 40 --max-params 14
    python3 scripts/sweep_new_models.py --since 2025-04     # after feed freeze
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

MODELS_URL = ("https://huggingface.co/api/models?author={author}"
              "&pipeline_tag=text-generation&sort=createdAt&direction=-1&limit={limit}")
DETAIL_URL = "https://huggingface.co/api/models/{model_id}"
USER_AGENT = "inchworm-new-model-sweep/1.0"

# Raw safetensors.total runs above a model's nominal size: the embedding and
# output matrices count toward the total, so e.g. JanusCoder-14B measures
# 14.77B and Qwen2.5-Coder-14B sits similarly over 14. The page stores nominal
# params, so compare against the ceiling with this slack and flag boundary
# rows for an editorial card check instead of silently dropping the 14B tier.
PARAM_SLACK = 1.08

# Org accounts that actually ship open coding-relevant models. Override with --orgs.
# Kept current as accounts consolidate/rename (THUDM -> zai-org, Xiaomi ->
# XiaomiMiMo); verify an org's account id with the search API before adding.
DEFAULT_ORGS = [
    "Qwen", "google", "microsoft", "meta-llama", "deepseek-ai",
    "mistralai", "ibm-granite", "allenai", "internlm", "zai-org",
    "XiaomiMiMo", "01-ai", "infly", "openbmb", "agentica-org",
    "LiquidAI", "bigcode", "open-thoughts", "deepcogito",
]

# Name fragments that mark a variant of an already-listed release, quant
# formats, or non-target artifact — these are dropped from the shortlist.
DENY_FRAGMENTS = [
    "-gguf", "_gguf", "-awq", "-gptq", "-fp8", "-bnb", "-bitsandbytes",
    "-q4", "-q8", "-int4", "-int8", "-mlx", "-onnx", "-openvino",
    "-nemo", "-sft", "-dpo", "-rm", "-merged", "-merge", "-trl", "-vllm",
    "-diffusers", "-paligemma", "dpo-", "orpo", "-nemo-", "-kto",
    # research checkpoint families / non-causal text2text: not page targets
    "dayhoff", "t5gemma", "-autoawq", "-awq-",
]

ORG_NICK = {"Qwen": "Qwen", "google": "Google", "microsoft": "Microsoft",
            "meta-llama": "Meta", "deepseek-ai": "DeepSeek", "mistralai": "Mistral",
            "ibm-granite": "IBM", "allenai": "Ai2", "internlm": "InternLM",
            "zai-org": "Z.ai", "XiaomiMiMo": "Xiaomi", "01-ai": "01.AI",
            "infly": "Infly", "openbmb": "OpenBMB", "agentica-org": "Agentica",
            "LiquidAI": "Liquid AI", "bigcode": "BigCode",
            "open-thoughts": "Open Thoughts", "deepcogito": "Deep Cogito"}


def get_json(url, retries=3):
    """GET + parse JSON with polite 429 backoff (HF rate-limits burst calls)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def load_feeds():
    """Best-effort sets of model names present in the frozen feeds."""
    names = set()
    try:
        import scrape_scores as sc
        ep = sc.load_evalplus()
        if ep:
            names.update(str(k).lower() for k in ep)
        full, _ = sc.load_bigcodebench()
        if full:
            names.update(str(k).lower() for k in full)
    except Exception as exc:  # noqa: BLE001 - feeds are best-effort here
        print(f"[warn] could not load frozen feeds: {exc}", file=sys.stderr)
    return names


def resolve_params(model_id):
    """Real parameter count in billions from safetensors.total, else None."""
    try:
        meta = get_json(DETAIL_URL.format(model_id=urllib.parse.quote(model_id, safe="/")))
    except Exception:  # noqa: BLE001
        return None
    total = (meta.get("safetensors") or {}).get("total")
    if isinstance(total, (int, float)):
        return total / 1e9
    return None


def is_denied(model_id):
    low = model_id.lower()
    return any(frag in low for frag in DENY_FRAGMENTS)


def normalize(model_id):
    """Sort-friendly key ignoring hyphens/dots/underscores/case."""
    return re.sub(r"[\-._]", "", model_id).lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orgs", default=",".join(DEFAULT_ORGS),
                    help="comma-separated orgs to sweep (default: tracked orgs)")
    ap.add_argument("--limit", type=int, default=40,
                    help="newest models to fetch per org (default: 40)")
    ap.add_argument("--max-params", type=float, default=14,
                    help="parameter ceiling in billions (default: 14)")
    ap.add_argument("--min-params", type=float, default=0.5,
                    help="parameter floor in billions (default: 0.5)")
    ap.add_argument("--min-likes", type=int, default=5,
                    help="likes floor to cut zero-audience research checkpoints "
                         "(default: 5; 0 disables)")
    ap.add_argument("--since", default="2025-01",
                    help="only models created on/after this YYYY-MM (default: 2025-01)")
    args = ap.parse_args()

    orgs = [o.strip() for o in args.orgs.split(",") if o.strip()]

    roster = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))
    known = set()
    for section in ("benchmarked", "pending"):
        for m in roster.get(section, []):
            known.add(m.get("name") or "")
    for m in (roster.get("candidates_no_benchmarks") or {}).get("models", []):
        known.add(m.get("name") or "")
    # Match against basename only: roster stores display names without the
    # org prefix, while the API ids carry it (deepseek-ai/DeepSeek-R1-...).
    known_norm = {normalize(n) for n in known if n}

    feed_names = load_feeds()
    since_year, _, since_month = args.since.partition("-")

    def is_known(model_id):
        base = model_id.split("/", 1)[-1]
        cand = normalize(base)
        if cand in known_norm or normalize(model_id) in known_norm:
            return True
        # point releases of a tracked model (e.g. Qwen3-4B-Instruct-2507 vs
        # tracked Qwen3-4B) are refreshes, not new roster entries
        return any(cand.startswith(k) and len(k) >= 6 for k in known_norm)

    def base_of(model_id):
        return re.sub(r"-(instruct|chat|it|base)$", "", model_id, flags=re.I)

    seen_bases = set()
    candidates = []
    detail_calls = 0
    for org in orgs:
        try:
            rows = get_json(MODELS_URL.format(author=urllib.parse.quote(org),
                                              limit=args.limit))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {org}: {exc}")
            continue
        for row in rows:
            if row.get("private"):
                continue
            model_id = row.get("id")
            if not model_id or is_denied(model_id):
                continue
            created = (row.get("createdAt") or "")[:10]
            if created and created[:4] < since_year:
                continue
            if created and created[:4] == since_year and created[5:7] < since_month:
                continue
            if args.min_likes and (row.get("likes") or 0) < args.min_likes:
                continue
            # Cheap checks first — never spend a detail call on known models
            # or duplicate variants of the same release.
            if is_known(model_id):
                continue
            base_norm = normalize(base_of(model_id))
            if base_norm in seen_bases:
                continue
            # Resolve params (1 detail call per genuinely new-looking model).
            params = resolve_params(model_id)
            detail_calls += 1
            if params is None or params < 0.05:  # 0/None = unresolved safetensors
                continue
            if params < args.min_params or params > args.max_params * PARAM_SLACK:
                continue
            seen_bases.add(base_norm)
            in_feeds = (normalize(model_id) in feed_names
                        or normalize(model_id.split("/", 1)[-1]) in feed_names)
            candidates.append({
                "id": model_id,
                "params": params,
                "created": created,
                "likes": row.get("likes", 0),
                "downloads": row.get("downloads", 0),
                "in_feeds": in_feeds,
                "over_ceiling": params > args.max_params,
            })
        time.sleep(0.3)

    # Only genuinely new releases; flag feed coverage; sort newest first.
    fresh = sorted(candidates, key=lambda c: (c["created"] or "", c["id"]),
                   reverse=True)

    print(f"swept orgs: {', '.join(orgs)} (newest {args.limit} text-generation "
          f"each, {args.min_params:.1f}-{args.max_params:.0f}B, created >= {args.since}, "
          f"likes >= {args.min_likes}); detail calls: {detail_calls}")
    print(f"new shortlist: {len(fresh)} (known/dup/denied filtered before param lookup)")
    print()
    if not fresh:
        print("No qualifying releases newer than", args.since)
        return 0
    print("id | params | created | likes | in frozen feeds? | url")
    print("-" * 140)
    for c in fresh:
        org_label = ORG_NICK.get(c["id"].split("/", 1)[0], c["id"].split("/", 1)[0])
        flag = "HAS FEED SCORE → benchmarked[] candidate" if c["in_feeds"] else "pending[] candidate"
        if c["over_ceiling"]:
            flag += f" · RAW {c['params']:.2f}B > {args.max_params:.0f}B — nominal? check card"
        print(f"{c['id']:52s} {c['params']:5.2f}B | {c['created'] or '?':10s} | "
              f"{c['likes']:5} | {flag} | https://huggingface.co/{c['id']}")
    print()
    print(f"{len(fresh)} candidates above the {args.since} cutoff for editorial "
          "review — none are written anywhere; triage into models.json manually "
          "after checking each model card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
