# INCHWORM — Tiny LLM Benchmarks

A laptop-level leaderboard for open coding models at **≤14B parameters**.

> A sortable, filterable leaderboard comparing open coding LLMs you can actually run on a laptop. Every score is a pass@1 percentage compiled from public sources — EvalPlus, BigCodeBench, and official model cards — and embedded in the page as a local snapshot.

![INCHWORM leaderboard](docs/screenshot.png)

## Features

- **BigCodeBench** — the primary lens (Full-set pass@1, instruct lens; `complete` for base models). The ★ row pins the overall leader across all sizes for scale.
- **HumanEval+** / **Delta** — toggleable HumanEval+ view and its delta vs. plain HumanEval.
- **Local eval** — HumanEval+ measured from our own self-evaluated runs on this machine, a directional check on vendor numbers.
- **Local fit** — `easy` / `mid` / `heavy` / `max spec (14B, 24GB+)` tiers plus quantized memory estimates.
- **Dark/light themes**, live search, and per-column sorting — all in vanilla JS.

## Sources & methodology

Scores are compiled from public model reports, the EvalPlus leaderboard, and the BigCodeBench results datasets (both frozen as of Apr 2025). New numbers come from our own local runs — see the footer of the page for the full source list and methodology.

## Add your local benchmark

Ran a model yourself? Submit your HumanEval+ result for the **Local eval** column. Append one entry to `entries` in `user_benchmarks.json` and open a PR — CI validates the schema and that your GitHub username is on the entry.

```json
{
  "model": "OpenCoder-1.5B-Instruct",
  "hf_id": "infly/OpenCoder-1.5B-Instruct",
  "user": "your-gh-username",
  "date": "2026-09-03",
  "hardware": "M2 Pro 32GB (MPS)",
  "precision": "bf16",
  "runtime": "transformers",
  "humaneval_plus": 58.5,
  "humaneval": 59.8,
  "notes": "plain chatml, greedy; official EvalPlus evaluator"
}
```

`model` must match the name exactly as it appears on the leaderboard, and `humaneval_plus` (0–100) must come from the official EvalPlus evaluator or an equivalent harness. Re-runs by the same user replace their earlier entry. Once a model has entries from ≥3 distinct users, maintainers promote the median into the `local` field on the page.

## Pending local benchmarks

We're running the current **8B-class batch** of HumanEval+ self-evals locally (plain chatml, greedy, official EvalPlus evaluator). Scores land in the **Local eval** column as each run finishes.

| Status | Model | HumanEval+ (local) |
| --- | --- | --- |
| ✔ Done | OpenCoder-1.5B-Instruct | 58.5 |
| ✔ Done | Qwen2.5-Coder-1.5B-Instruct | 59.8 |
| 🟢 Running | Qwen3-8B | — |
| ⏳ Queued | Cogito-v1-preview-llama-8B | — |
| ⏳ Queued | Granite-3.3-8B-Instruct | — |
| ⏳ Queued | Ministral-8B-Instruct-2410 | — |
| ⏳ Queued | DeepSeek-R1-Distill-Llama-8B | — |

<!-- Each model takes ~35–45 min on an M2 Pro (MPS). -->