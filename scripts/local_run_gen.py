#!/usr/bin/env python3
"""Greedy HumanEval+ generation for a small instruct model (MPS), then run
the official evalplus evaluator on the produced samples.

Run inside the .eval venv (has torch/evalplus/transformers):
    .eval/venv/bin/python scripts/local_run_gen.py

Why not `evalplus.evaluate --backend hf` directly? evalplus 0.3.1's HF path
stops generation at its own stop texts (e.g. "\\n```\\n") which also appear in
the *prompt*, so outputs truncate to a token or two. Generating here with a
plain chatml prompt (no fences) fixes it while evaluation stays official.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".eval", "out", "humaneval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", default=None, help="filename slug (defaults to model name)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--parallel", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize

    os.makedirs(OUT, exist_ok=True)
    slug = args.slug or args.model.replace("/", "-").lower()
    samples_path = os.path.join(OUT, f"{slug}_chatml_greedy.jsonl")

    problems = get_human_eval_plus()
    print(f"[gen] {len(problems)} problems; model {args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False,
                                        trust_remote_code=True)
    device = "mps"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="mps", trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="eager",
        )
    except Exception as e:  # OOM / MPS unsupported ops on some machines
        print(f"[gen] MPS load failed ({type(e).__name__}: {e}); falling back to CPU", flush=True)
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="cpu", trust_remote_code=True,
            torch_dtype=torch.float32, attn_implementation="eager",
        )
    model.eval()
    print("[gen] model loaded on", model.device, flush=True)

    # resume support
    done = set()
    if os.path.isfile(samples_path):
        with open(samples_path) as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["task_id"])
        print(f"[gen] resuming: {len(done)} already done", flush=True)

    def generate(prompt_text: str, max_new: int) -> str:
        ids = tok(prompt_text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                max_new_tokens=max_new, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)

    with open(samples_path, "a") as fh:
        for i, (task_id, task) in enumerate(problems.items()):
            if task_id in done:
                continue
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": task["prompt"].strip()}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # no-op for non-thinking models (Qwen3, R1 distills)
            )
            raw = generate(prompt, args.max_new_tokens)
            # evalplus's sanitize runs code_extract internally, which finds the
            # longest valid Python span amid prose/fences and then extracts the
            # target function — feed it the full raw output, not a pre-cut.
            solution = sanitize(raw, entrypoint=task["entry_point"]) or raw
            if not solution.strip():
                solution = raw
            fh.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"[gen] {i}/{len(problems)} ({task_id})", flush=True)

    print(f"[gen] done writing {samples_path}", flush=True)
    del model
    import gc
    gc.collect()

    # ---- official evaluation phase (execution of HumanEval+ tests) ----
    print("[eval] running official evalplus evaluator ...", flush=True)
    env = dict(os.environ)
    # Disable the memory rlimit guard, which crashes every worker on macOS
    # (resource.setrlimit RLIMIT_AS/DATA raises ValueError). Generation is
    # already done; we only need test execution, so no guard is required.
    env["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"
    cmd = [sys.executable, "-m", "evalplus.evaluate",
           "--dataset", "humaneval", "--samples", samples_path,
           "--parallel", str(args.parallel or max(1, os.cpu_count() // 2))]
    sys.exit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
