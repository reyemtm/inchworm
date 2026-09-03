# AGENTS.md

Guidance for AI coding agents working in this repository. Read this before making changes.

## Project overview

A single-page, dependency-free HTML page: a **laptop-level LLM leaderboard** for open coding models at **≤14B parameters** (≤10B by default, plus a clearly-flagged 14B "max spec" tier). Scores (HumanEval+ and HumanEval pass@1) are compiled from public online sources — the EvalPlus leaderboard, model cards on Hugging Face, and official model reports — and embedded in the page as a local snapshot. It is meant to run on a modern laptop with no server, no build step, and no installed packages.

## Constraints (do not break)

- **Vanilla only.** Plain HTML/CSS/JS. No frameworks, no build tooling, no npm/package.json, no external CDN dependencies.
- **Single file.** Everything lives in `index.html`: styles in the `<style>` block, data and logic in the inline `<script>`. Do not split into separate files.
- **Scope.** Models up to 14B parameters. Entries ≤10B are the default promise (everything runs on 16–24GB laptops); 14B entries must be flagged `fit:'max'` and get the red "Max spec" badge — they assume 24GB+ unified memory or VRAM. Flag anything above 14B.
- **GitHub styling.** Design tokens are CSS variables in `:root` / `[data-theme="dark"]` (canvas, surface, border, ink, muted, green, blue, etc.). Keep the GitHub-like palette; light/dark mode is driven by `data-theme` on `<html>` and persisted under `localStorage['patchwork-theme']`.

## File layout

- `index.html` — the entire app (vanilla HTML/CSS/JS; the "single file" constraint applies to the app itself).
  - `<style>`: design tokens + all component styles.
  - Top of `<body>`: slim header (`INCHWORM` brand lockup — inline green inchworm SVG mark + wordmark — and a borderless theme toggle), then the badges row (`snapshot` / `scope` tags) aligned above the leaderboard.
  - Root assets next to `index.html`: `og.png` (1200×630 social card, referenced by the OpenGraph meta tags at `https://reyemtm.github.io/inchworm/og.png`) and `favicon.png` (dark rounded tile with the green inchworm, referenced via `<link rel="icon">`). Regenerate both with headless Chrome if the design changes.
  - `<section id="leaderboard">`: search input, laptop-fit filter, HumanEval+/Delta segmented toggle, the sortable table (including the sortable BigCodeBench column and the sortable Local eval column), a pinned "overall leader" reference row (★ — the top model across all sizes in the BigCodeBench data, kept for scale and never sorted/ranked), and the selected-model detail panel (which also shows the BigCodeBench Hard score).
  - Footer: methodology + source links.
  - `<script>`: the data layer, state, and rendering.
- Deploy tooling (outside the app, stdlib-only Python):
  - `scripts/scrape_scores.py` — nightly scraper: pulls fresh `plus`/`base` scores from the EvalPlus leaderboard data (`https://raw.githubusercontent.com/evalplus/evalplus.github.io/main/results.json`, Apache-2.0) and fresh `bcb`/`bcb_hard`/`bcb_mode` from the BigCodeBench results datasets (`bigcode/bigcodebench-results` + `bigcode/bigcodebench-hard-results` via the public `datasets-server.huggingface.co` rows API, Apache-2.0), then rewrites the `models` array + snapshot date in `index.html`. Keeps existing values for any model a source doesn't cover; never leaves the page broken. Run manually with `python3 scripts/scrape_scores.py [--dry-run]`. The EvalPlus feed has CORS `access-control-allow-origin: *` (client-side lookup possible); the datasets-server rows API does not, so BigCodeBench refresh is server-side only.
  - `.github/workflows/nightly.yml` — scraper wrapper kept for manual runs (`workflow_dispatch` from the Actions tab). The nightly cron is **disabled**: both upstream feeds are frozen (BigCodeBench results stop 2025-04-14, EvalPlus earlier), so scheduled runs were guaranteed no-ops. Re-enable the cron in the workflow if a feed resumes. The workflow commits a refreshed `index.html` back to the default branch; requires `contents: write`. Serve the page with GitHub Pages (Settings → Pages → Deploy from a branch → `main` root) so pushes publish automatically. New numbers now come from local self-evals (`scripts/local_batch.py` / `scripts/local_eval.py`), not the feeds.
  - `.github/workflows/validate-pr.yml` + `scripts/validate_user_benchmarks.py` — **community submissions path**: PRs may only modify `user_benchmarks.json`; the workflow enforces that guard, validates schema/ranges, and requires the PR author's GitHub username on at least one entry. Nothing here writes to the leaderboard automatically.
  - `scripts/promote_user_benchmarks.py` — maintainer-only aggregator: when a model has entries from ≥ 3 distinct users, writes the median HumanEval+ into the `local` field in `index.html` + `models.json` and flags the entries as promoted. Run by hand (`--dry-run` to preview). Our own runs stay one-off entries in the same file, never special-cased.

**Adding a model:** still an editorial act — add the object to `models` by hand (or extend the scraper's matching), then run the scraper to backfill scores from EvalPlus.

## Data layer

The only data source is the `models` array in the inline script, plus a small
`leader` reference object (the pinned ★ row, below). Each entry:

```js
{
  name:   'Qwen2.5-Coder-7B-Instruct', // model card name
  org:    'Qwen',                      // organization
  params: 7,                           // parameter count in billions (≤ 10)
  plus:   73.8,                        // HumanEval+ pass@1, percent
  base:   88.4,                        // HumanEval pass@1, percent
  local:  58.5,                        // OPTIONAL: self-evaluated HumanEval+ on this machine
  bcb:    40.4,                        // BigCodeBench Full pass@1, percent
  bcb_hard: 20.3,                      // BigCodeBench Hard pass@1, percent
  bcb_mode: 'instruct',                // 'instruct' | 'complete' (base models)
  fit:    'mid',                       // 'easy' | 'mid' | 'heavy'
  memory: '~5 GB @ 4-bit',             // estimated quantized memory
  best:   '...',                       // one-line "best for" pitch
  note:   '...',                       // short qualitative note
  url:    'https://huggingface.co/...' // source / model card link
}
```

The `leader` object sits right above the array and mirrors the same shape
(`name`/`org`/`params`/`plus`/`base`/`bcb`/`bcb_hard`/`bcb_mode`/`url`), with
`null` in fields the sources don't cover (e.g. `params` and the HumanEval pair
for closed API models). The scraper recomputes it each night as the top row of
the BigCodeBench results dataset (instruct lens, complete fallback) and also
refreshes the roster's scores. The pinned row is excluded from sorting, ranks,
and the fit filter (shown only when the fit filter is "Any laptop fit").

### Adding or updating a model

1. Add/update an object in `models` (array order is editorial — display always sorts by the active column, default BigCodeBench). Then run the scraper to backfill scores: for models BigCodeBench covers it will insert the `bcb`/`bcb_hard`/`bcb_mode` fields automatically.
2. **Always attach a source.** `url` must point to the public page the scores come from (model card, EvalPlus leaderboard row, BigCodeBench leaderboard row, or official report). Do not add an entry without a verifiable source. The nightly scraper refreshes `plus`/`base` from the EvalPlus `results.json` feed and `bcb`/`bcb_hard`/`bcb_mode` from the BigCodeBench results datasets when it can match the model name; unmatched models keep their snapshot values.
3. Record where the score was scraped from in the commit message or a comment; the footer methodology states scores are "compiled from public model reports, EvalPlus results, and BigCodeBench leaderboard data where available."
4. The BigCodeBench column shows the **Full set** (1,140 tasks) pass@1, preferring the `instruct` lens (brief NL instruction → code) and falling back to `complete` for base models; the **Hard set** (~150 tasks) appears in the detail panel. `bcb_mode` records which lens `bcb` uses. Models missing from BigCodeBench (e.g. Qwen2.5-Coder-3B, Phi-3-mini-4k, Mistral-7B-v0.2) show `—` in that column; models EvalPlus never ran (e.g. Phi-4, DeepCoder-14B-Preview) show `—` in the HumanEval+ column but keep card-sourced HumanEval pairs when documented in `url`.
5. Expanding the roster is data-driven: mine the `bigcode/bigcodebench-results` rows for open models with `size ≤ 14` that the page lacks, and prefer candidates EvalPlus also covers (so every column fills). The scraper backfills both score sets on the next run; unmatched models keep whatever snapshot values they ship with.
   - **Feed recency:** both machine feeds are historical — EvalPlus's `results.json` stops around early 2025 and the BigCodeBench results dataset around April 2025. Models released after that (Qwen3-Coder ≥30B, Qwen3 small sizes, Gemma 3/4, Kimi/MiniMax, etc.) are **not in the feeds**, so they can only be added editorially with snapshot values from official model cards (HumanEval pairs from cards are fine — the render tolerates `null` `plus`/`base` and shows `—`).
6. Keep `fit` consistent with `params`: `easy` ≈ ≤4B, `mid` ≈ 5–7B, `heavy` = needs more RAM (≈9–10B), `max` = 14B tier that needs 24GB+ memory. Update `memory` to match typical 4-bit loading (~0.6 GB per billion params plus context overhead).

## Behavior & interactions (all vanilla JS)

- BigCodeBench is the **primary lens**: default `state.sort` is `'bcb'` desc, and the active sort header shows a ▼/▲ `sortmark`. The HumanEval+/Delta segmented toggle still controls which column is emphasized and the delta subscore text.
- The **Local eval** column (`local`, sortable) holds HumanEval+ pass@1 from our own self-evaluated runs on this machine (plain chatml, greedy, official EvalPlus evaluator). It is a directional single-run sanity check on vendor numbers, not a nightly source — most models show `—` until run locally. Only set `local` for models we have actually self-evaluated (see `scripts/local_eval.py`).
- The pinned ★ `leader` row is rendered first (via `leaderRow()`), never sorted, never selectable, and hidden when a fit filter or a non-matching search query is active.
- Single `state` object drives everything: `query`, `fit`, `score`, `selected`, `sort`, `direction`.
- `render()` rebuilds the `<tbody>` from `filtered()` and repaints the detail panel; `filtered()` applies search + fit filter, then sorts.
- **Sorting:** table headers carry `data-sort="params|plus|base|bcb"`. Clicking a header sorts by that field (first click descending, subsequent clicks toggle); there must be exactly **one** `applySort` definition and **one** listener attachment — duplicates silently double-toggle. The comparator multiplies by `1` (desc) or `-1` (asc); `direction` is the string `'desc'`/`'asc'`, never used in arithmetic directly. Models missing a score (e.g. `bcb` null) sort last in either direction.
- Clicking a row selects it and fills the detail panel; `state.selected` holds the index into `models` (not the filtered position).
- Segmented toggle switches the score lens: `plus`, `base`, or `delta` (delta = `plus - base`).
- No client-side refresh affordance: the snapshot in `models` is the single source of truth. The scraper is manual-only (feeds frozen); new scores arrive via local self-evals or editorial updates.
- Theme toggle swaps `data-theme` on `<html>` and persists to localStorage.

## Running & verifying

- No install or build. Open `index.html` directly, or serve statically, e.g.:
  `python3 -m http.server 4173`
- There is no test framework. Verify changes by loading the page and exercising the interactions (search, fit filter, score toggle, header sort, row selection, theme toggle) in the browser preview.
- If the leaderboard renders empty, the `models` array or `render()` broke — check the script for syntax errors first.

## Gotchas

- The page has no external dependencies, so there is no lockfile or package manager to respect.
- Keep accessibility: search input has an `aria-label`, the detail panel is `aria-live="polite"`, headers/buttons remain focusable, and dark/light themes must both be legible.
- Text encoding: the file uses entities like `↗` and `−` directly; keep the charset `utf-8` meta tag.
