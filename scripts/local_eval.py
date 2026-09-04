#!/usr/bin/env python3
"""Dev CLI to run a local HumanEval+ eval of a small open model and compare
with the page's snapshot values. Everything is stdlib-only. Runtime artifacts
(venv, logs, pid, results) live in `.eval/` at the repo root, gitignored.

Usage:
    python3 scripts/local_eval.py start --model opencoder-1.5b
    python3 scripts/local_eval.py status
    python3 scripts/local_eval.py kill
    python3 scripts/local_eval.py logs [-n 50]
    python3 scripts/local_eval.py result

Models (name -> HF id + page snapshot values for the summary):
    opencoder-1.5b      infly/OpenCoder-1.5B-Instruct      HE+ 67.7 / HE 72.5 / BCB 34.9
    qwen2.5-coder-1.5b  Qwen/Qwen2.5-Coder-1.5B-Instruct  HE+ 56.8 / HE 72.1 / BCB 27.0
    qwen3-1.7b          Qwen/Qwen3-1.7B                  HE+ 73.6 claimed (SOTA2, suspect)
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, ".eval")
VENV = os.path.join(DIR, "venv")
PY = os.path.join(VENV, "bin", "python")
LOG = os.path.join(DIR, "run.log")
SETUP_LOG = os.path.join(DIR, "setup.log")
PID = os.path.join(DIR, "run.pid")
META = os.path.join(DIR, "run.meta.json")
OUT = os.path.join(DIR, "out")

DATASET = "humaneval"

# Model registry: name -> (HF id, slug for filenames, page snapshot values)
MODELS = {
    "opencoder-1.5b": (
        "infly/OpenCoder-1.5B-Instruct", "opencoder-1.5b-instruct",
        dict(plus=67.7, base=72.5, bcb=34.9),
    ),
    "qwen2.5-coder-1.5b": (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", "qwen2.5-coder-1.5b-instruct",
        dict(plus=56.8, base=72.1, bcb=27.0),
    ),
    # --- 8B-class candidates (no verified pass@1 coding numbers anywhere) ---
    "qwen3-8b": (
        "Qwen/Qwen3-8B", "qwen3-8b",
        dict(plus=None, base=None, bcb=None),
    ),
    # --- tiny third-party claims that deserve our own check ---
    "qwen3-1.7b": (
        "Qwen/Qwen3-1.7B", "qwen3-1.7b",
        dict(plus=73.6, base=None, bcb=None),  # 73.6 HE+ is the SOTA2 claim under test
    ),
    "cogito-8b": (
        "deepcogito/cogito-v1-preview-llama-8B", "cogito-v1-preview-llama-8b",
        dict(plus=None, base=None, bcb=None),
    ),
    "granite-3.3-8b": (
        "ibm-granite/granite-3.3-8b-instruct", "granite-3.3-8b-instruct",
        dict(plus=None, base=None, bcb=None),
    ),
    "ministral-8b": (
        "mistralai/Ministral-8B-Instruct-2410", "ministral-8b-instruct-2410",
        dict(plus=None, base=None, bcb=None),
    ),
    "ds-r1-distill-llama-8b": (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "ds-r1-distill-llama-8b",
        dict(plus=None, base=None, bcb=None),
    ),
}

def _model(args):
    """Resolve the selected model into (hf_id, slug, targets dict)."""
    key = getattr(args, "model", None) or "opencoder-1.5b"
    if key not in MODELS:
        print(f"[start] unknown model '{key}'. Known: {', '.join(MODELS)}")
        sys.exit(2)
    hf_id, slug, targets = MODELS[key]
    return key, hf_id, slug, targets


def _ensure_dir():
    os.makedirs(DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)


def _base_interp():
    """Pick a modern Python for the venv; skip the EOL system python 3.9."""
    import shutil
    for cand in ("python3.12", "python3.13", "python3.11"):
        p = shutil.which(cand)
        if p:
            return p
    return sys.executable


def _patch_evalplus():
    """Idempotently patch the installed evalplus HF provider so that:
    1. it passes trust_remote_code to AutoTokenizer (needed by OpenCoder), and
    2. it uses MPS on Apple Silicon instead of silently falling back to CPU.
    The venv lives in gitignored .eval/, so patching it is safe.
    """
    import glob
    sp = glob.glob(os.path.join(VENV, "lib", "python*", "site-packages"))
    hf_path = os.path.join(sp[0], "evalplus", "provider", "hf.py")
    src = open(hf_path).read()
    orig = src
    src = src.replace(
        'self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        'self.device = torch.device("mps" if torch.backends.mps.is_available() '
        'else "cuda" if torch.cuda.is_available() else "cpu")',
    )
    src = src.replace(
        "self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)",
        "self.tokenizer = AutoTokenizer.from_pretrained("
        "name, use_fast=False, trust_remote_code=self.trust_remote_code)",
    )
    if src != orig:
        open(hf_path, "w").write(src)
        print("[setup] patched evalplus HF provider (MPS + tokenizer trust_remote_code)")


def _setup():
    """Create venv + install deps if needed. Returns True if anything ran."""
    if not os.path.exists(PY):
        _ensure_dir()
        interp = _base_interp()
        print(f"[setup] creating venv at {VENV} (base: {interp}) ...")
        subprocess.check_call([interp, "-m", "venv", VENV])
        with open(SETUP_LOG, "w") as fh:
            fh.write("=== pip install ===\n")
        deps = ["torch", "transformers", "accelerate", "evalplus", "sentencepiece"]
        print(f"[setup] installing {', '.join(deps)} (this takes a few minutes) ...")
        subprocess.check_call(
            [PY, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=open(SETUP_LOG, "a"), stderr=subprocess.STDOUT,
        )
        subprocess.check_call(
            [PY, "-m", "pip", "install"] + deps,
            stdout=open(SETUP_LOG, "a"), stderr=subprocess.STDOUT,
        )
        _patch_evalplus()
        print("[setup] done. Setup log:", SETUP_LOG)
        return True
    # venv exists: make sure the patch is applied even after re-creating it
    _patch_evalplus()
    return False


def _eval_cmd(hf_id, slug, extra_args):
    # NOTE: we generate with our own script (evalplus 0.3.1's HF backend stops
    # at its own stop texts inside the prompt) and evaluate samples officially.
    return [
        PY, os.path.join(ROOT, "scripts", "local_run_gen.py"),
        "--model", hf_id, "--slug", slug,
    ] + (extra_args.split() if extra_args else [])


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _read_pid():
    try:
        with open(PID) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def cmd_start(args):
    if _read_pid() and _pid_alive(_read_pid()):
        print(f"[start] already running (pid {_read_pid()}). Stop it first: "
              f"python3 scripts/local_eval.py kill")
        return 1
    key, hf_id, slug, targets = _model(args)
    _setup()
    cmd = _eval_cmd(hf_id, slug, args.evalplus_args)
    with open(LOG, "w") as fh:
        fh.write(f"=== run {time.strftime('%Y-%m-%d %H:%M:%S')} · {key} ===\n")
        fh.write("$ " + " ".join(cmd) + "\n\n")
    proc = subprocess.Popen(
        cmd, stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
        cwd=ROOT, start_new_session=True,
        env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1"),
    )
    with open(PID, "w") as fh:
        fh.write(str(proc.pid))
    with open(META, "w") as fh:
        json.dump({"model": key, "hf": hf_id, "slug": slug, "targets": targets}, fh)
    print(f"[start] launched pid {proc.pid}")
    print(f"[start] model {hf_id} · dataset {DATASET} · greedy pass@1")
    print(f"[start] targets from page: HumanEval+ {targets['plus']} / HumanEval {targets['base']} / BigCodeBench {targets['bcb']}")
    print(f"[start] log: {LOG}  (tail it with: python3 scripts/local_eval.py logs)")
    print("[start] to stop early:        python3 scripts/local_eval.py kill")
    return 0


def cmd_status(args):
    pid = _read_pid()
    if pid and _pid_alive(pid):
        elapsed = time.time() - os.path.getmtime(PID)
        print(f"[status] RUNNING pid {pid}, elapsed {int(elapsed)}s")
        if os.path.exists(LOG):
            lines = open(LOG).read().splitlines()
            print("[status] last log lines:")
            for ln in lines[-5:]:
                print("   ", ln)
    else:
        print("[status] not running"
              + (f" (stale pid {pid})" if pid else ""))
    if os.path.exists(SETUP_LOG):
        print(f"[status] setup log: {SETUP_LOG} ({os.path.getsize(SETUP_LOG)} bytes)")
    return 0


def cmd_kill(args):
    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        print("[kill] nothing running")
        return 0
    print(f"[kill] sending SIGTERM to process group of {pid} ...")
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # also terminate the evalplus python process group leader's children
    for _ in range(10):
        if not _pid_alive(pid):
            break
        time.sleep(0.5)
    if _pid_alive(pid):
        print("[kill] SIGKILL fallback ...")
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    print("[kill] stopped. Partial results (if any):")
    print(f"        {OUT}")
    return 0


def cmd_logs(args):
    if not os.path.exists(LOG):
        print("[logs] no run log yet")
        return 0
    lines = open(LOG).read().splitlines()
    for ln in lines[-args.n:]:
        print(ln)
    return 0


def cmd_result(args):
    """Print eval_results.json from the newest evalplus output dir, if any."""
    if not os.path.isdir(OUT):
        print("[result] no output dir yet"); return 0
    import glob
    import json
    cands = []
    for pat in ("**/eval_results.json", "**/*_eval_results.json",
                "**/humaneval*/eval_results.json"):
        cands += glob.glob(os.path.join(OUT, pat), recursive=True)
    if not cands:
        print("[result] no eval_results.json yet — run still in progress?")
        return 0
    newest = max(cands, key=os.path.getmtime)
    data = json.load(open(newest))
    print(f"[result] {newest}")
    # evalplus 0.3.1 stores per-task statuses under data["eval"] rather than
    # a "stats" dict, so compute pass@1 from the base/plus status counts.
    tasks = (data.get("eval") or {}).values()
    if not tasks:
        stats = data.get("stats") or {}
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v}")
        return 0
    from collections import Counter
    def _pct(key):
        c = Counter(r.get(key) for t in tasks for r in t)
        ok = c.get("pass", 0); total = sum(c.values())
        return (100.0 * ok / total) if total else 0.0
    base = _pct("base_status")
    plus = _pct("plus_status")
    t = dict(plus=None, base=None, bcb=None)
    if os.path.isfile(META):
        t.update(json.load(open(META)).get("targets", {}))
    print()
    print(f"  local  HumanEval pass@1   : {base:.1f}%")
    print(f"  local  HumanEval+ pass@1  : {plus:.1f}%")
    print(f"  page   HumanEval          : {t['base']}")
    print(f"  page   HumanEval+         : {t['plus']}")
    print(f"  page   BigCodeBench       : {t['bcb']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_start = sub.add_parser("start")
    p_start.add_argument("--model", default="opencoder-1.5b",
                         help="model key (see docstring; default opencoder-1.5b)")
    p_start.add_argument("--evalplus-args", default="",
                         help="extra args passed through to evalplus.evaluate")
    p_start.set_defaults(fn=cmd_start)
    p_status = sub.add_parser("status"); p_status.set_defaults(fn=cmd_status)
    p_kill = sub.add_parser("kill"); p_kill.set_defaults(fn=cmd_kill)
    p_logs = sub.add_parser("logs")
    p_logs.add_argument("-n", type=int, default=50); p_logs.set_defaults(fn=cmd_logs)
    p_res = sub.add_parser("result"); p_res.set_defaults(fn=cmd_result)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
