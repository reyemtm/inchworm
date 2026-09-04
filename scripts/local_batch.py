#!/usr/bin/env python3
"""Sequential local HumanEval+ benchmark driver.

Runs the pending 8B-class candidates one at a time. After each run completes
it updates `models.json` (adds/refreshes the `local:` field) and rebuilds
`index.html` from it via scripts/build_page.py, then asks whether to continue
to the next model. `--dry-run` shows what is pending and what would change
without touching anything.

Usage:
    python3 scripts/local_batch.py              # interactive run-all
    python3 scripts/local_batch.py --auto       # run-all unattended: eval -> record -> delete weights -> next
    python3 scripts/local_batch.py --only KEY   # just one specific model
    python3 scripts/local_batch.py --dry-run    # show pending work, change nothing

Prereqs: `.eval/venv` (create once with `python3 scripts/local_eval.py start
--model opencoder-1.5b`, or let the first run's setup do it). ~35-45 min per
model on an M2 Pro (MPS), mostly generation.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import local_eval as cli  # reuse registry, venv setup, launcher
import build_page as bp   # rebuild index.html from models.json

OUT = cli.OUT
JSON_PATH = os.path.join(ROOT, "models.json")

# The 8B-class candidate batch, in run order.
BATCH = ["qwen3-8b", "cogito-8b", "granite-3.3-8b", "ministral-8b", "ds-r1-distill-llama-8b"]


# ---------------------------------------------------------------- result ----
def read_result(hf_id, slug):
    """Return (plus, base) pass@1 percentages for a finished run, else None."""
    path = os.path.join(OUT, "humaneval", f"{slug}_chatml_greedy_eval_results.json")
    if not os.path.isfile(path):
        return None
    data = json.load(open(path))
    tasks = (data.get("eval") or {}).values()
    if not tasks:
        return None
    from collections import Counter

    def pct(key):
        c = Counter(r.get(key) for t in tasks for r in t)
        total = sum(c.values())
        return 100.0 * c.get("pass", 0) / total if total else 0.0

    return pct("plus_status"), pct("base_status")


def samples_complete(slug):
    n = sum(1 for _ in open(os.path.join(OUT, "humaneval", f"{slug}_chatml_greedy.jsonl"))) \
        if os.path.isfile(os.path.join(OUT, "humaneval", f"{slug}_chatml_greedy.jsonl")) else 0
    return n, n == 164


# ---------------------------------------------------------------- update ----
def update_json(name, plus):
    """Set the `local:` field on a model's row in models.json; returns True if changed."""
    d = json.load(open(JSON_PATH))
    for row in d["benchmarked"]:
        if row["name"] == name:
            row["local"] = round(plus, 1)
            break
    else:
        print(f"[json] WARNING: '{name}' not in models.json benchmarked[] — skipped")
        return False
    bp._atomic_write(JSON_PATH, json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    return True


def rebuild_page():
    """Regenerate index.html's data consts from the updated models.json."""
    bp.rebuild(json_path=JSON_PATH)


# ------------------------------------------------------------------ run ----
def delete_weights(hf_id):
    """Remove a model's snapshot from the local Hugging Face cache (reclaim disk)."""
    repo_dir = os.path.expanduser(os.path.join(
        "~", ".cache", "huggingface", "hub", "models--" + hf_id.replace("/", "--")))
    if os.path.isdir(repo_dir):
        import shutil
        size = subprocess.run(["du", "-sh", repo_dir], capture_output=True, text=True).stdout.split()[0]
        shutil.rmtree(repo_dir)
        print(f"[batch] deleted HF cache for {hf_id} ({size} reclaimed)")
    else:
        print(f"[batch] no HF cache for {hf_id} — nothing to delete")


def run_model(key):
    hf_id, slug, targets = cli.MODELS[key]
    proc = subprocess.Popen(
        cli._eval_cmd(hf_id, slug, ""),
        stdout=open(cli.LOG, "a"), stderr=subprocess.STDOUT,
        cwd=cli.ROOT, start_new_session=True,
        env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1"),
    )
    open(cli.PID, "w").write(str(proc.pid))
    with open(cli.META, "w") as fh:
        json.dump({"model": key, "hf": hf_id, "slug": slug, "targets": targets}, fh)
    print(f"[run] {key} launched (pid {proc.pid}) — log: {cli.LOG}")
    print(f"[run] watching for completion (log tail: python3 scripts/local_eval.py logs) ...")
    while proc.poll() is None:
        time.sleep(30)
    print(f"[run] {key} process exited with code {proc.returncode}")
    return proc.returncode


def model_display_name(key):
    """Map a registry key to the display name used in index.html/models.json."""
    return DISPLAY_NAMES.get(key, key)


DISPLAY_NAMES = {
    "qwen3-8b": "Qwen3-8B",
    "cogito-8b": "Cogito-v1-preview-llama-8B",
    "granite-3.3-8b": "Granite-3.3-8B-Instruct",
    "ministral-8b": "Ministral-8B-Instruct-2410",
    "ds-r1-distill-llama-8b": "DeepSeek-R1-Distill-Llama-8B",
}


def prompt_continue(key, next_key):
    while True:
        ans = input(f"\n[batch] {key} done. Run next model '{next_key}'? [y/N/q] ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        if ans in ("q", "quit"):
            print("[batch] quitting — remaining models stay pending")
            sys.exit(0)


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run just this model key and exit")
    ap.add_argument("--auto", action="store_true",
                    help="run all pending models back-to-back: eval -> record scores -> "
                         "delete weights -> next, with no prompts")
    ap.add_argument("--dry-run", action="store_true",
                    help="show pending/completed runs and what would change; touch nothing")
    args = ap.parse_args()

    queue = [args.only] if args.only else BATCH

    print("=" * 64)
    print("LOCAL HumanEval+ BATCH — 8B-class candidates")
    print("=" * 64)
    status = []
    for key in queue:
        hf_id, slug, targets = cli.MODELS[key]
        res = read_result(hf_id, slug)
        nsamp, complete = samples_complete(slug)
        if res:
            status.append((key, "done", f"HE+ {res[0]:.1f} / HE {res[1]:.1f}"))
        elif complete:
            status.append((key, "generated, not evaluated", f"{nsamp}/164 samples"))
        else:
            status.append((key, "pending", f"{nsamp}/164 samples"))
    w = max(len(k) for k, *_ in status)
    for key, st, detail in status:
        print(f"  {key:<{w}}  {st:<24} {detail}")

    if args.dry_run:
        print("\n[dry-run] no changes made. What a full run would do:")
        for key, st, detail in status:
            hf_id, slug, targets = cli.MODELS[key]
            if st == "done":
                print(f"  {key}: already done ({detail}) — would skip")
            else:
                print(f"  {key}: would generate + evaluate ~35-45 min, "
                      f"then set local: in models.json and rebuild index.html")
        return 0

    for i, key in enumerate(queue):
        hf_id, slug, targets = cli.MODELS[key]
        res = read_result(hf_id, slug)
        if res:
            print(f"\n[batch] {key} already done ({res[0]:.1f}/{res[1]:.1f}) — skipping")
            continue
        rc = run_model(key)
        res = read_result(hf_id, slug)
        if not res:
            print(f"[batch] {key} finished but no eval results found (rc={rc}) — "
                  f"check `python3 scripts/local_eval.py logs`")
            if args.auto:
                continue
            if i < len(queue) - 1 and not prompt_continue(key, queue[i + 1]):
                break
            continue
        plus, base = res
        print(f"\n[batch] {key} RESULTS: HumanEval+ {plus:.1f} · HumanEval {base:.1f}")
        display_name = model_display_name(key)
        j = update_json(display_name, plus)
        if j:
            rebuild_page()
        print(f"[batch] updated: models.json local field "
              f"({'ok' if j else 'skipped'}) — index.html rebuilt from it")
        if args.auto:
            delete_weights(hf_id)
            continue
        if i < len(queue) - 1 and not prompt_continue(key, queue[i + 1]):
            print(f"[batch] stopping — {len(queue) - i - 1} model(s) remain pending")
            break
    print("\n[batch] all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
