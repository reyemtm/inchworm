# Tiny ML Benchmark — design notes

## Goal
Tiny, fast, chat-based (and maybe HumanEval-style standalone) coding benchmark for <14B models on native Silicon Mac. Python or Node, native runtime (MLX, Ollama, etc.).

## Important principle
- **Durable, meaningful, and fast is non-negotiable.**
- Prefer tasks that look like a **hard real bug in a very small amount of code** over broad toy prompts.
- Every benchmark task should maximize signal-per-minute:
  - **Durable:** deterministic setup, stable prompts, fixed grading, reproducible runs.
  - **Meaningful:** evaluates real debugging/correction ability, not just surface completion.
  - **Fast:** tiny code scope, strict per-task timeout, quick feedback loop.

## Runtime targets (hard budgets)
Assume local inference on a typical native Silicon laptop with an 8B-9B model.

- **Small run:** under 1-5 minutes total.
- **Medium run:** 5-10 minutes total.
- **Large run (complete):** 20 minutes max total.

If a configuration consistently exceeds its budget on the reference machine, it should not be the default profile.

## Run profiles
- **Small:** quick confidence check before model iteration; minimal task set.
- **Medium:** balanced signal and speed for normal comparisons.
- **Large:** highest confidence local run, still capped to 20 minutes.

## Context from live benchmarks
- **HumanEval / HumanEval+** — small, static, runnable standalone (no chat harness needed). Can be run quickly with greedy decoding. Our page already uses this.
- **BigCodeBench** — harder, library-based, longer runtime but more realistic. Still static tasks, but may exceed time target for small models.
- **LiveCodeBench** — dynamic, multi-turn, often longer; probably too heavy for our speed target unless we subsample.
- **MBPP / MBPP+** — also short, static, good complement to HumanEval.

## Proposed design
- Small curated set of **~20–40 short coding problems** with:
  - problem statement (chat-style prompt optional)
  - reference solution / unit tests
  - key facts / summary tag per problem (e.g. "list comprehension", "regex parsing", "file I/O")
- Grading: pass@1 with greedy; optional speed tracking.
- Two modes:
  1. **Standalone HumanEval-style**: problem + test, model completes function, run tests.
  2. **Chat-based**: model receives problem as chat message, returns code block, extract & run tests.

## Speed strategy
- Keep problems tiny (one function, few tests).
- Allow timeout per problem (~10–30s).
- Run sequentially, report per-problem time.
- Stratify into simple/medium/slow buckets by expected runtime.

## Data format
- JSON of problems: id, prompt, test_code, summary, difficulty, expected_time.
- Results JSON per model run: per-problem pass/fail, time, notes.

## Baseline alignment protocol
Use known public baselines to validate whether this tiny benchmark tracks real coding ability.

- Start with **6 anchor models** total:
  - **3 models** with strong/known HumanEval+ baselines.
  - **3 models** with strong/known LiveCodeBench baselines.
- If LiveCodeBench coverage is better for newer models, prefer **5 LiveCodeBench anchors** and keep HumanEval+ anchors at 3.
- Keep anchors in the same size class when possible (focus on sub-14B, especially 8B-9B) so runtime and capability comparisons stay fair.

For each anchor model, report side-by-side deltas:

- `our_score`
- `humaneval_plus_score` (if available)
- `livecodebench_score` (if available)
- `delta_vs_humaneval_plus = our_score - humaneval_plus_score`
- `delta_vs_livecodebench = our_score - livecodebench_score`

Output should include both per-model rows and aggregate stats:

- Mean absolute delta per benchmark source.
- Rank correlation trend (does our ordering roughly match public benchmark ordering).
- Notes for outliers (large positive/negative deltas).

Interpretation goal: the tiny benchmark does not need to numerically match HumanEval+/LiveCodeBench, but it should preserve meaningful relative ordering and expose major capability gaps quickly.

## External data ingestion mappings

### 1) LLMCheck JSON -> our schema
LLMCheck is primarily a local inference speed/fit dataset (not a coding correctness benchmark), but it is useful as a hardware/runtime signal.

Expected source fields include:
- `model`
- `params`
- `quant`
- `chip`
- `ram`
- `engine`
- `tps`
- `ttft`
- `date`
- `provenance` (measured/sourced/community/estimated)

Map into our schema as:
- `source`: `"llmcheck"`
- `model_name`: `model`
- `params_b`: parsed numeric value from `params`
- `quant`: `quant`
- `hardware`: `{ chip, ram_gb: ram }`
- `runtime`: `engine`
- `perf_decode_tps`: `tps`
- `perf_ttft_s`: `ttft`
- `measured_at`: `date`
- `source_confidence`: from `provenance`
  - `measured` -> high
  - `community`/`sourced` -> medium
  - `estimated` -> low

Important: do not treat LLMCheck as a coding score baseline; treat it as feasibility and speed metadata.

### 2) GPT-Laboratory results -> our schema
GPT-Laboratory publishes code benchmark results and a runnable harness (BigCode-based).

Expected source fields include:
- `model`
- `benchmark_result.tasks.<task>.result.pass@1` (and other pass@k fields)
- `benchmark_result.tasks.<task>.total_elapsed_time_sec`
- `benchmark_result.tasks.<task>.average_time_per_generation_sec`
- `benchmark_result.tasks.<task>.max_vram_usage_mb_per_gpu`

Map into our schema as:
- `source`: `"gpt-laboratory"`
- `model_name`: `model`
- `task_name`: `<task>`
- `score_pass_at_1`: `result.pass@1` (normalize to percent if needed)
- `score_pass_at_k`: other available pass@k values
- `run_time_sec`: `total_elapsed_time_sec`
- `time_per_generation_sec`: `average_time_per_generation_sec`
- `vram_peak_mb`: `max_vram_usage_mb_per_gpu`
- `source_confidence`: medium-high (reproducible, but hardware/harness-dependent)

## Benchmark coverage check (for these two sources)
- **LLMCheck:** no HumanEval+/LiveCodeBench correctness results in the core dataset. It provides speed/fit metrics.
- **GPT-Laboratory:** includes **HumanEval** results (plus MBPP, CodeXGLUE, Mercury, HumanEvalPack). LiveCodeBench coverage is not the primary reported suite.

For coding baselines:
- Use GPT-Laboratory HumanEval-family results as secondary references.
- Use LiveCodeBench numbers from dedicated LiveCodeBench/public model-card sources when available.

## Use as pending-eval model signals
Yes, both can be used as signals for `pending[]` triage, with guardrails.

- Use **LLMCheck** to prioritize models likely to run well on target hardware (fit + tps + ttft).
- Use **GPT-Laboratory** to prioritize models with promising coding correctness on HumanEval/MBPP-style tasks.
- Do not auto-promote any model into benchmarked status from these feeds alone.
- Keep a per-model signal tag in triage notes, e.g.:
  - `signal_runtime`: high/medium/low (from LLMCheck provenance + speed)
  - `signal_coding`: high/medium/low (from GPT-Lab task scores)
  - `needs_local_eval`: always true until run in our own harness

## Possible next steps
- Curate problem set from HumanEval+/MBPP public domain problems.
- Build runner in Python using MLX or subprocess calling Ollama API.
- Optionally add chat harness for models that need conversation.

## Open questions
- Which models/backends to support first?
- Standalone vs chat-first?
- Public vs private problem set?
