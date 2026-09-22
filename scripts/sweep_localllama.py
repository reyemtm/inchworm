#!/usr/bin/env python3
"""sweep_localllama.py — read-only discovery sweep of r/LocalLLaMA for new small models.

Editorial discovery aid for the Pending Eval tab and the
candidates_no_benchmarks pool. Reddit's post-JSON API blocks anonymous
callers (403), but each subreddit's Atom feed is public, so this script reads
`/r/<sub>/new/.rss` (or a search feed), pulls the newest posts, and prints a
triage table: titles mentioning a ≤14B model, posts that claim a coding
benchmark, posts that mention a runnable distribution (GGUF/Ollama/MLX), and
whether the model is already in models.json.

It never edits models.json or index.html — a Reddit post is a *signal*, not a
source, so nothing here can become a score. Use it to find models to check
against the HF model API (`scripts/sweep_new_models.py`) and, for models the
feeds cover, `scripts/scrape_scores.py`.

Rate limits: Reddit throttles aggressively by IP (429), so the fetched feed is
cached under temp/ (gitignored) and reused for --cache-ttl seconds; pass
--refresh to force a fetch. One request per run, descriptive User-Agent.

Usage:
    python3 scripts/sweep_localllama.py                     # newest posts, flagged rows only
    python3 scripts/sweep_localllama.py --all               # every post in the feed
    python3 scripts/sweep_localllama.py --query "9B"        # search feed instead of /new
    python3 scripts/sweep_localllama.py --max-params 10     # tighten the ≤N B filter
    python3 scripts/sweep_localllama.py --feed-file f.xml   # parse a saved feed (offline)
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "temp")
USER_AGENT = ("inchworm-localllama-sweep/1.0 "
              "(read-only discovery; contact: repo maintainer)")

ATOM = {"a": "http://www.w3.org/2005/Atom"}

# "Qwen3-8B", "gemma3:12b", "granite4.2:3b", "Ministral-3-14B" — a name glued
# to a parameter count. Heuristic on purpose: the table is for a human to read.
NAME_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9.]*(?:[-_.:][A-Za-z0-9.]+)*\s?\d+(?:\.\d+)?\s?[Bb])\b")
OLLAMA_TAG_RE = re.compile(r"\b([a-z][a-z0-9.\-]*:[a-z0-9._\-]+)\b")
BENCH_RE = re.compile(
    r"humaneval\+?|bigcodebench|livecodebench|swe-?bench|mbpp\+?|"
    r"codeforces|aider|terminal-?bench", re.I)
RUNNABLE_RE = re.compile(r"\bgguf\b|\bollama\b|\bmlx\b|llama\.cpp|lm ?studio", re.I)
NOISE_WORDS = {"q4", "q8", "q5", "q6", "int4", "int8", "fp8", "fp16", "bf16",
               "f16", "f32", "gptq", "awq", "kv", "kv8", "cache", "8k", "16k",
               "32k", "128k", "256k", "512k", "1k", "2k", "4k", "64k", "tok"}

# Prose words that survive the casing test but never start a model name.
STOP_STEMS = {"the", "a", "an", "this", "that", "these", "those", "my", "our",
              "your", "it", "its", "all", "some", "any", "one", "two", "no",
              "vs", "how", "what", "why", "which", "ran", "runs", "using"}


# --------------------------------------------------------------- fetching ----
def cache_path(url):
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"localllama_{digest}.xml")


def fetch_feed(url, ttl, refresh=False, retries=2):
    """Return the raw Atom XML for `url`, cached under temp/ for `ttl` seconds.

    Reddit 429s quickly and without warning, so a successful fetch is kept on
    disk and reused; a stale cache is still better than no data.
    """
    path = cache_path(url)
    if not refresh and os.path.isfile(path):
        age = time.time() - os.path.getmtime(path)
        if ttl < 0 or age <= ttl:
            print(f"[cache] {path} ({age / 60:.1f} min old, ttl {ttl}s)")
            return open(path, encoding="utf-8").read()

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", "replace")
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(raw)
            return raw
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"[warn] 429 from Reddit; retrying in {wait}s "
                      f"({attempt + 1}/{retries - 1})", file=sys.stderr)
                time.sleep(wait)
                continue
            if exc.code == 429:
                raise SystemExit(
                    "[error] Reddit is rate-limiting this IP (429). RSS reads are\n"
                    "        anonymous and throttled; wait a few minutes, then\n"
                    "        re-run, or point --feed-file at a saved feed.")
            raise SystemExit(f"[error] {url} -> HTTP {exc.code}")
        except urllib.error.URLError as exc:
            raise SystemExit(f"[error] could not reach {url}: {exc.reason}")

    # A parseable cache beats nothing when the live fetch is throttled.
    if os.path.isfile(path):
        print("[warn] live fetch failed; falling back to cache", file=sys.stderr)
        return open(path, encoding="utf-8").read()
    raise SystemExit("[error] no feed data available")


# ----------------------------------------------------------------- parsing ----
def strip_html(text):
    """Atom `content` is HTML-escaped twice; unwrap both layers, drop tags."""
    for _ in range(2):
        text = html.unescape(text)
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def parse_entries(raw):
    """[(published, title, link, author, body)] from an Atom feed, newest first."""
    root = ET.fromstring(raw)
    out = []
    for entry in root.findall("a:entry", ATOM):
        title = html.unescape(entry.findtext("a:title", default="", namespaces=ATOM))
        when = (entry.findtext("a:updated", default="", namespaces=ATOM)
                or entry.findtext("a:published", default="", namespaces=ATOM))
        link_el = entry.find("a:link", ATOM)
        link = link_el.get("href") if link_el is not None else ""
        author = entry.findtext("a:author/a:name", default="", namespaces=ATOM)
        body = strip_html(entry.findtext("a:content", default="", namespaces=ATOM))
        out.append((when, title.strip(), link, author, body))
    out.sort(key=lambda e: e[0], reverse=True)
    return out


# ---------------------------------------------------------------- triage ------
def norm_name(name):
    """Loose key so a post's spelling still matches models.json.

    "Gemma4 12B" and "gemma-4-12B" are the same model; "Qwen2.5-1.5B" and
    "Qwen2.5-Coder-1.5B" are not, and normalizing keeps them apart.
    """
    return re.sub(r"[^a-z0-9.]", "", name.lower())


def roster_index():
    """{loose model name: 'benchmarked'|'pending'|'candidates'}."""
    data = json.load(open(os.path.join(ROOT, "models.json"), encoding="utf-8"))
    index = {}
    for section in ("benchmarked", "pending"):
        for row in data.get(section, []):
            if row.get("name"):
                index[norm_name(row["name"])] = section
    pool = data.get("candidates_no_benchmarks") or {}
    for row in pool.get("models", []) if isinstance(pool, dict) else []:
        if row.get("name"):
            index.setdefault(norm_name(row["name"]), "candidates")
    return index


def sizes_in(text, max_params, min_params):
    """Parameter counts mentioned as "7B"/"0.9B"/"9 b", filtered to the scope."""
    found = set()
    for mtch in re.finditer(r"\b(\d+(?:\.\d+)?)\s?[bB]\b", text):
        value = float(mtch.group(1))
        if min_params <= value <= max_params:
            found.add(value)
    return sorted(found)


def names_in(text):
    """Model-name-ish tokens ("Qwen3-8B", "gemma3:12b") from a post title/body.

    Deliberately conservative: a bare size ("7B", "32k"), a noise unit, an
    ordinary lowercase word glued to a size ("even35B", "why14b"), or a
    one-character stem ("a7B" out of "a 7B model") is not a name. Sizes are
    already reported separately, so a miss here only costs a candidate line.
    """
    names = set()
    for mtch in NAME_RE.finditer(text):
        token = mtch.group(1).strip()
        if not token or token[0].isdigit():
            continue
        stem = re.sub(r"\s?\d+(?:\.\d+)?\s?[bB]$", "", token).strip("-_.: ")
        if stem.lower() in NOISE_WORDS or token.lower() in NOISE_WORDS:
            continue
        if len(stem) < 3 and not any(ch.isdigit() for ch in stem):
            continue
        if stem.lower() in STOP_STEMS:
            continue
        # A real model stem carries a digit (Qwen3, Granite-4.2) or interior
        # caps (Taleeq, MiniCPM); a plain lowercase English word glued to a
        # size is prose, not a name.
        if not any(ch.isdigit() for ch in stem) and stem == stem.lower():
            continue
        names.add(re.sub(r"\s+", " ", token))
    for mtch in OLLAMA_TAG_RE.finditer(text):
        tag = mtch.group(1)
        if ":" in tag and any(ch.isdigit() for ch in tag.split(":")[-1][:1] or "0"):
            names.add(tag)
    return names


def name_size_b(name):
    """Largest number in a name as its size proxy ("Gemma-4-26B-A4B" -> 26.0).

    Names carry versions as well as sizes (Granite-4.2-3B, Qwen2.5-Coder-1.5B),
    so the biggest number is the safest reading of "how big is this model" for
    the out-of-scope filter. Returns None when a name has no number at all.
    """
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", name)]
    return max(numbers) if numbers else None



def classify(entry, roster, max_params, min_params):
    """Flags + extracted names for one post."""
    when, title, link, author, body = entry
    text = title + " " + body
    sizes = sizes_in(text, max_params, min_params)
    names = names_in(title) or names_in(text)
    known = sorted({roster[norm_name(n)] for n in names if norm_name(n) in roster})
    flags = []
    if sizes:
        flags.append("small")
    if BENCH_RE.search(text):
        flags.append("claims")
    if RUNNABLE_RE.search(text):
        flags.append("runnable")
    if known:
        flags.append("in-roster:" + ",".join(known))
    return {"when": when, "title": title, "link": link, "author": author,
            "sizes": sizes, "names": names, "known": known, "flags": flags}


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(
        description="Read-only r/LocalLLaMA sweep for new small-model discussion.")
    ap.add_argument("--sub", default="LocalLLaMA", help="subreddit (default LocalLLaMA)")
    ap.add_argument("--query", default="", help="use the search feed for this query")
    ap.add_argument("--limit", default="100", help="posts to request (default 100)")
    ap.add_argument("--max-params", type=float, default=14.0,
                    help="upper parameter bound in billions (default 14)")
    ap.add_argument("--min-params", type=float, default=0.05,
                    help="lower parameter bound in billions (default 0.05)")
    ap.add_argument("--cache-ttl", type=int, default=3600,
                    help="seconds to reuse a cached feed; -1 = forever (default 3600)")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--all", action="store_true", help="print every post, not just flagged")
    ap.add_argument("--feed-file", default="", help="parse a saved Atom feed instead of fetching")
    args = ap.parse_args()

    if args.feed_file:
        raw = open(args.feed_file, encoding="utf-8").read()
        source = args.feed_file
    else:
        if args.query:
            url = (f"https://www.reddit.com/r/{args.sub}/search.rss"
                   f"?q={urllib.parse.quote(args.query)}&restrict_sr=1&sort=new"
                   f"&t=month&limit={args.limit}")
        else:
            url = f"https://www.reddit.com/r/{args.sub}/new/.rss?limit={args.limit}"
        source = url
        raw = fetch_feed(url, args.cache_ttl, args.refresh)

    entries = parse_entries(raw)
    if not entries:
        raise SystemExit("[error] feed parsed but contained no entries")

    roster = roster_index()
    rows = [classify(e, roster, args.max_params, args.min_params) for e in entries]
    shown = rows if args.all else [r for r in rows if r["flags"]]

    print(f"\nr/{args.sub} sweep — {len(entries)} posts, {len(shown)} flagged"
          f"  (≤{args.max_params:g}B scope)")
    print(f"source: {source}\n")
    print(f"{'when':<17} {'sizes':<20} {'flags':<26} title")
    print("-" * 118)
    for r in shown:
        print(f"{r['when'][:16]:<17} {str(r['sizes'])[:19]:<20} "
              f"{','.join(r['flags'])[:25]:<26} {r['title'][:58]}")

    # Actionable shortlist: models named in small-model posts that we don't have.
    fresh = {}
    seen_lower = set()
    for r in rows:
        if "small" not in r["flags"]:
            continue
        for name in sorted(r["names"]):
            key = norm_name(name)
            if key in roster or key in seen_lower:
                continue
            size = name_size_b(name)
            if size is not None and not (args.min_params <= size <= args.max_params):
                continue  # a 27B mention is out of scope for this page
            seen_lower.add(key)
            fresh[name] = r
    print("\n--- candidate names not in models.json ---")
    if not fresh:
        print("(none — everything named in the feed is already on the roster)")
    for name, r in sorted(fresh.items()):
        print(f"  {name:<28} {r['when'][:10]}  {r['link']}")
    print("\nReview: confirm param count and a card-sourced score, then add to "
          "models.json\n(pending[] or candidates_no_benchmarks) and run "
          "scripts/build_page.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
