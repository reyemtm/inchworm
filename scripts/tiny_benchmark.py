#!/usr/bin/env python3
"""
Tiny Python Benchmark — Python twin of scripts/tiny_benchmark.js.

Same problems, same tests, same CLI shape. Candidates write Python
functions; the grader runs them with `python3` (stdlib only, no pip).

Usage:
  python3 scripts/tiny_benchmark.py                              # default problems + Ollama
  python3 scripts/tiny_benchmark.py --mode=ollama
  python3 scripts/tiny_benchmark.py --mode=direct
  python3 scripts/tiny_benchmark.py --tests=tmb-atoi --models=ministral-3:14b --level=large --runs=1
  python3 scripts/tiny_benchmark.py --tests=tmb-csv-parse,tmb-atoi --models=ministral-3:14b --timeout-s=60

Configuration precedence: CLI flag > process env (TMB_*) > .env >
tmb_env.json > defaults. Python standard library only — no pip packages.
"""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "TMB_MODE": "ollama",
    "TMB_RUNS": "3",
    "TMB_MODELS": "qwen3.5:9b,granite4.2:3b",
    "OLLAMA_URL": "http://localhost:11434",
    # interactive-chat bar: strong models answer in 3-20s; 30s total per
    # problem across all self-repair attempts
    "TMB_TIMEOUT_S": "30",
    "TMB_NUM_PREDICT": "256",
    "TMB_ATTEMPTS": "3",   # self-repair: max code-generation attempts per problem
    "TMB_THINK": "false",
    "TMB_PYTHON": "python3",
    "TMB_RESULTS": "results/tmb_result.json",
    "TMB_DEBUG": "false",
    "TMB_WARMUP": "false",
}


# ---------------------------------------------------------------------------
# config plumbing (mirrors the JS harness)

def parse_env(text):
    values = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        value = m.group(2).strip()
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        else:
            value = re.sub(r"\s+#.*$", "", value)
        values[m.group(1)] = value
    return values


def parse_args(args):
    values = {}
    pending = None
    for arg in args:
        if pending is not None:
            values["TMB_" + pending] = arg
            pending = None
            continue
        m = re.match(r"^--([^=]+)=(.*)$", arg)
        if m:
            key = m.group(1).replace("-", "_").upper()
            values["TMB_" + key] = m.group(2)
            continue
        m = re.match(r"^--([^=]+)$", arg)
        if m:
            pending = m.group(1).replace("-", "_").upper()
    return values


def load_config():
    values = {}
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                values = parse_env(f.read())
        except OSError:
            values = {}
    if not values:
        legacy = os.path.join(ROOT, "tmb_env.json")
        if os.path.exists(legacy):
            try:
                with open(legacy, "r", encoding="utf-8") as f:
                    values = json.load(f)
            except (OSError, ValueError):
                values = {}
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in values.items() if v is not None})
    for key, val in os.environ.items():
        if key.startswith("TMB_") or key == "OLLAMA_URL":
            cfg[key] = val
    cfg.update(parse_args(sys.argv[1:]))
    return cfg


def as_bool(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def as_int(value, fallback=None):
    try:
        n = int(value)
        return n if n > 0 else (fallback if fallback is not None else 1)
    except (TypeError, ValueError):
        return fallback if fallback is not None else 1


def normalize_url(value):
    return re.sub(r"/api/?$", "", str(value)).rstrip("/")


def clean_temp_paths(text):
    return re.sub(r"/var/folders/[\\w/\\-]+/tiny-py-benchmark-[\\w-]+/candidate\\.py(?::\\d+)?", "candidate.py", text or "")


ERROR_RE = re.compile(
    r"(SyntaxError|NameError|TypeError|ValueError|IndexError|KeyError|"
    r"ZeroDivisionError|AttributeError|ImportError|IndentationError|"
    r"RecursionError|OverflowError|Error:|Traceback)", re.I)


def pick_error_headline(stderr):
    lines = [l.strip() for l in (stderr or "").splitlines() if l.strip()]
    if not lines:
        return ""
    for line in lines:
        if ERROR_RE.search(line):
            return clean_temp_paths(line)
    return clean_temp_paths(lines[-1])


def compact_error(result):
    if not result or result.get("pass"):
        return ""
    msg = result.get("error") or pick_error_headline(result.get("stderr") or "") or "failed"
    return clean_temp_paths(str(msg))[:80]


def strip_thinking(text):
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text or "", flags=re.I)
    text = re.sub(r"\s*thinking[\s\S]*?<\\/think>", "", text, flags=re.I)
    text = re.sub(r"\[thinking\][\s\S]*?\[/thinking\]", "", text, flags=re.I)
    text = re.sub(r"</?think>", "", text, flags=re.I)
    return text.strip()


def extract_code(response):
    """Pull a Python `fn` definition out of a model response."""
    text = strip_thinking(response)
    blocks = re.findall(r"```(?:python|py)?\s*\n?([\s\S]*?)```", text, flags=re.I)
    blocks = [b.strip() for b in blocks if re.search(r"\bdef\s+fn\s*\(", b)]
    if blocks:
        return blocks[0]
    m = re.search(r"\bdef\s+fn\s*\(", text)
    if m:
        prefix = [l for l in text[:m.start()].splitlines()
                  if re.match(r"^\s*(import|from|def)\s+", l)]
        return "\n".join(prefix + [text[m.start():]]).strip()
    return ""


# ---------------------------------------------------------------------------
# problem set — same ids/tests as the JS harness, Python prompts.

def _p(body):
    """Wrap a problem's Task/Rules body with the shared Python instruction tail."""
    return (
        body
        + "\n\nReturn ONLY a Python code block containing the fn function.\n"
        "Do not explain your answer. Do not hardcode only the visible example; "
        "implement the stated behavior.\n"
        "Print the result with print(json.dumps(fn(...))) when run."
    )


PROBLEMS = {
    "tmb-jwt": {
        "id": "tmb-jwt",
        "title": "decode JWT payload",
        "summary": "base64url decode, JSON parse, standard library only",
        "difficulty": "simple",
        "timeout_s": 30,
        "tests": [
            {"input": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwidGFnIjoiaGVsbG8iLCJpYXQiOjE0OTk5OTk5OTl9.dummy", "expected": {"sub": "user", "tag": "hello", "iat": 1499999999}},
            {"input": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhIjoxfQ.dummy", "expected": {"a": 1}},
            {"input": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiY2FmXHUwMGU5IiwibiI6MCwib2siOnRydWV9.dummy", "expected": {"name": "caf\u00e9", "n": 0, "ok": True}},
            {"input": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcnIiOlsxLDIsM119.dummy", "expected": {"arr": [1, 2, 3]}},
            {"input": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGljZSIsInJvbGUiOiJhZG1pbiJ9.dummy", "expected": {"sub": "alice", "role": "admin"}},
        ],
        "prompt": _p("""Task: given a JWT string, decode and return the decoded payload.

The JWT format is three base64url-encoded segments separated by '.': header.payload.signature.
Decode the second segment (payload) from base64url to UTF-8 text, then parse it as JSON.
Return the resulting dict.

Use only the Python standard library.

Example:
  fn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyfSwidGFnIjoiaGVsbG8iLCJpYXQiOjE0OTk5OTk5OTl9.used-for-harness-only")
  should return the object { "sub": "user", "tag": "hello", "iat": 1499999999 }."""),
    },

    "tmb-slug": {
        "id": "tmb-slug",
        "title": "slugify string",
        "summary": "unicode lowercasing, punctuation removal, whitespace collapse",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": "Hello, World! 123", "expected": "hello-world-123"},
            {"input": "  Foo  Bar  ", "expected": "foo-bar"},
            {"input": "UPPER lower", "expected": "upper-lower"},
            {"input": "a--b---c", "expected": "a-b-c"},
            {"input": "123", "expected": "123"},
        ],
        "prompt": _p("""Task: convert a string into a URL-friendly slug.

Rules:
- Lowercase the string.
- Replace any sequence of non-alphanumeric characters with a single hyphen.
- Strip leading and trailing hyphens.
- Use only the Python standard library.

Example:
  fn("Hello, World! 123") should return "hello-world-123"."""),
    },

    "tmb-kv-parse": {
        "id": "tmb-kv-parse",
        "title": "parse k=v string",
        "summary": "split on comma, split on first equals, trim whitespace",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": "a=1,b=2,c=3", "expected": {"a": "1", "b": "2", "c": "3"}},
            {"input": "x=10", "expected": {"x": "10"}},
            {"input": " key = value , a = 1 ", "expected": {"key": "value", "a": "1"}},
            {"input": "a=1=2,b=3", "expected": {"a": "1=2", "b": "3"}},
            {"input": "", "expected": {}},
        ],
        "prompt": _p("""Task: parse a compact key=value string into a dict.

Rules:
- The input is a string of key=value pairs separated by commas: "k1=v1,k2=v2,...".
- Split on commas, then split each pair on the FIRST '=' only.
- Trim whitespace around keys and values.
- Return a dict mapping each key to its value as a string.
- Use only the Python standard library.

Example:
  fn("a=1,b=2,c=3") should return {"a": "1", "b": "2", "c": "3"}."""),
    },

    "tmb-fix-index-delim": {
        "id": "tmb-fix-index-delim",
        "title": "fix buggy parser",
        "summary": "off-by-one, wrong delimiter, wrong split, wrong comparison — fix fn",
        "difficulty": "simple",
        "timeout_s": 25,
        "tests": [
            {"input": "a=1,b=2,c=3,d=4", "expected": {"a": "1", "b": "2", "c": "3", "d": "4"}},
            {"input": "x=5", "expected": {"x": "5"}},
            {"input": " a = 1 , b = 2 ", "expected": {"a": "1", "b": "2"}},
            {"input": "a=1=2", "expected": {"a": "1=2"}},
            {"input": "", "expected": {}},
        ],
        "prompt": _p("""The following Python function is buggy. Fix it so it parses the input correctly and returns the expected dict.

Rules:
- The fixed function must keep the same signature: fn(s) returns a dict.
- Use only the Python standard library.

Buggy function:

def fn(s):
    pairs = s.split("|")
    obj = {}
    for i in range(len(pairs) + 1):
        pair = pairs[i]
        if not pair:
            continue
        parts = pair.split(":")
        k = parts[0].strip()
        v = parts[1].strip()
        if k == v:
            continue
        obj[k] = v
    return obj

Expected behavior for fn("a=1,b=2,c=3,d=4"):
- Split the input on commas.
- For each piece, split on the FIRST '=' only.
- Trim whitespace around key and value.
- Build a dict mapping each key to its value as a string.
- Return the dict.

Expected result: {"a": "1", "b": "2", "c": "3", "d": "4"}."""),
    },

    "tmb-merge-ranges": {
        "id": "tmb-merge-ranges",
        "title": "merge overlapping ranges",
        "summary": "sort by start, merge overlaps, keep output deterministic",
        "difficulty": "medium",
        "timeout_s": 25,
        "tests": [
            {"input": [[1, 3], [2, 6], [8, 10], [15, 18], [17, 20]], "expected": [[1, 6], [8, 10], [15, 20]]},
            {"input": [[1, 4], [4, 5]], "expected": [[1, 5]]},
            {"input": [[1, 2], [3, 4]], "expected": [[1, 2], [3, 4]]},
            {"input": [[5, 5]], "expected": [[5, 5]]},
            {"input": [], "expected": []},
        ],
        "prompt": _p("""Task: merge overlapping numeric ranges.

Input:
- A list of 2-item lists, where each item is [start, end].
- start and end are integers with start <= end.

Rules:
- Sort ranges by start ascending.
- Merge ranges when they overlap (next.start <= current.end).
- Return merged ranges as a list of [start, end] pairs.
- Keep output order ascending by start.
- Use only the Python standard library.

Example:
  fn([[1,3],[2,6],[8,10],[15,18],[17,20]])
  should return [[1,6],[8,10],[15,20]]."""),
    },

    "tmb-rpn-eval": {
        "id": "tmb-rpn-eval",
        "title": "evaluate reverse polish notation",
        "summary": "stack-based expression evaluation with integer truncation",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"], "expected": 22},
            {"input": ["2", "1", "+", "3", "*"], "expected": 9},
            {"input": ["4", "13", "5", "/", "+"], "expected": 6},
            {"input": ["3", "4", "+", "2", "*", "1", "-"], "expected": 13},
            {"input": ["5"], "expected": 5},
        ],
        "prompt": _p("""Task: evaluate an arithmetic expression in Reverse Polish Notation (RPN).

Input:
- A list of string tokens.
- Each token is either an integer (possibly negative) or one of: "+", "-", "*", "/".

Rules:
- Use stack evaluation.
- Division truncates toward zero (like int(a / b)).
- Assume input is valid and contains no divide-by-zero.
- Return the final integer result.
- Use only the Python standard library.

Example:
  fn(["2","1","+","3","*"]) should return 9.
  fn(["4","13","5","/","+"]) should return 6."""),
    },

    "tmb-lru-ops": {
        "id": "tmb-lru-ops",
        "title": "simulate LRU cache operations",
        "summary": "ordered map updates, evictions, and deterministic get traces",
        "difficulty": "hard",
        "timeout_s": 35,
        "tests": [
            {"input": {"capacity": 2, "ops": [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3], ["get", 2], ["put", 4, 4], ["get", 1], ["get", 3], ["get", 4]]}, "expected": [1, -1, -1, 3, 4]},
            {"input": {"capacity": 1, "ops": [["put", 1, 1], ["put", 2, 2], ["get", 1], ["get", 2]]}, "expected": [-1, 2]},
            {"input": {"capacity": 2, "ops": [["get", 1]]}, "expected": [-1]},
            {"input": {"capacity": 3, "ops": [["put", 1, 1], ["put", 2, 2], ["put", 3, 3], ["get", 1], ["put", 4, 4], ["get", 2]]}, "expected": [1, -1]},
            {"input": {"capacity": 2, "ops": [["put", 1, 1], ["get", 1], ["put", 1, 2], ["get", 1]]}, "expected": [1, 2]},
        ],
        "prompt": _p("""Task: simulate an LRU (Least Recently Used) cache and return outputs for get operations.

Input:
- A dict: {"capacity": number, "ops": list}
- Each op is one of:
  - ["put", key, value]
  - ["get", key]

Rules:
- put inserts/updates key with value.
- get returns value if key exists, otherwise -1.
- Accessing/updating a key marks it as most recently used.
- If put exceeds capacity, evict the least recently used key.
- Return a list containing results of each get operation in order.
- Use only the Python standard library.

Example:
  fn({"capacity": 2, "ops": [["put",1,1],["put",2,2],["get",1],["put",3,3],["get",2]]})
  should return [1,-1]."""),
    },

    "tmb-reverse-words": {
        "id": "tmb-reverse-words",
        "title": "reverse word order",
        "summary": "split on whitespace, reverse list, rejoin with single spaces",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": "the sky is blue", "expected": "blue is sky the"},
            {"input": "a b c", "expected": "c b a"},
            {"input": "  hello   world  ", "expected": "world hello"},
            {"input": "single", "expected": "single"},
            {"input": "1 2 3", "expected": "3 2 1"},
        ],
        "prompt": _p("""Task: reverse the order of the words in a string.

Rules:
- Words are separated by one or more whitespace characters.
- Return the words in reverse order, joined by a single space each.
- Ignore leading/trailing whitespace.
- Use only the Python standard library.

Example:
  fn("the sky is blue") should return "blue is sky the"."""),
    },

    "tmb-fizzbuzz": {
        "id": "tmb-fizzbuzz",
        "title": "fizzbuzz",
        "summary": "modulo classification, string building, list output",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": 15, "expected": ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]},
            {"input": 1, "expected": ["1"]},
            {"input": 3, "expected": ["1", "2", "Fizz"]},
            {"input": 5, "expected": ["1", "2", "Fizz", "4", "Buzz"]},
            {"input": 16, "expected": ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz", "16"]},
        ],
        "prompt": _p("""Task: classic FizzBuzz.

Rules:
- fn(n) returns a list of strings for the numbers 1 through n inclusive.
- For multiples of 3, append "Fizz".
- For multiples of 5, append "Buzz".
- For multiples of both 3 and 5, append "FizzBuzz".
- Otherwise append the number as a string.
- Use only the Python standard library.

Example:
  fn(15) should return ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]."""),
    },

    "tmb-palindrome": {
        "id": "tmb-palindrome",
        "title": "valid palindrome",
        "summary": "normalize case, strip non-alphanumerics, compare reversed",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": "A man, a plan, a canal: Panama", "expected": True},
            {"input": "race a car", "expected": False},
            {"input": "abba", "expected": True},
            {"input": "", "expected": True},
            {"input": "0P", "expected": False},
        ],
        "prompt": _p("""Task: determine if a string is a palindrome.

Rules:
- A string is a palindrome if it reads the same forward and backward after:
  - converting all letters to lowercase, and
  - removing all non-alphanumeric characters (spaces, punctuation, etc.).
- Return True or False.
- Use only the Python standard library.

Example:
  fn("A man, a plan, a canal: Panama") should return True."""),
    },

    "tmb-valid-parentheses": {
        "id": "tmb-valid-parentheses",
        "title": "valid parentheses",
        "summary": "stack matching of (), [], {} with correct nesting",
        "difficulty": "medium",
        "timeout_s": 25,
        "tests": [
            {"input": "([)]", "expected": False},
            {"input": "()[]{}", "expected": True},
            {"input": "{[]}", "expected": True},
            {"input": "(", "expected": False},
            {"input": "", "expected": True},
        ],
        "prompt": _p("""Task: determine if an input string has valid, correctly nested parentheses.

Rules:
- The string contains only the characters ( ) [ ] { }.
- Open brackets must be closed by the same type of bracket.
- Open brackets must be closed in the correct order (proper nesting).
- Return True if valid, False otherwise.
- Use only the Python standard library.

Examples:
  fn("()[]{}") should return True.
  fn("([)]") should return False (crossed, not nested).
  fn("{[]}") should return True."""),
    },

    "tmb-roman-to-int": {
        "id": "tmb-roman-to-int",
        "title": "roman numeral to integer",
        "summary": "symbol value mapping, subtractive rule handling",
        "difficulty": "medium",
        "timeout_s": 25,
        "tests": [
            {"input": "MCMXCIV", "expected": 1994},
            {"input": "III", "expected": 3},
            {"input": "LVIII", "expected": 58},
            {"input": "IV", "expected": 4},
            {"input": "M", "expected": 1000},
        ],
        "prompt": _p("""Task: convert a Roman numeral string to an integer.

Rules:
- Symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000.
- A smaller symbol before a larger one is subtracted (e.g. IV=4, IX=9, CM=900).
- Otherwise symbols are added.
- Input is a valid Roman numeral.
- Use only the Python standard library.

Example:
  fn("MCMXCIV") should return 1994."""),
    },

    "tmb-int-to-roman": {
        "id": "tmb-int-to-roman",
        "title": "integer to roman numeral",
        "summary": "greedy symbol table, subtractive forms, string building",
        "difficulty": "medium",
        "timeout_s": 25,
        "tests": [
            {"input": 1994, "expected": "MCMXCIV"},
            {"input": 3, "expected": "III"},
            {"input": 58, "expected": "LVIII"},
            {"input": 4, "expected": "IV"},
            {"input": 1000, "expected": "M"},
        ],
        "prompt": _p("""Task: convert an integer to a Roman numeral string.

Rules:
- Symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000.
- Use subtractive forms where appropriate: 4=IV, 9=IX, 40=XL, 90=XC, 400=CD, 900=CM.
- Input is a positive integer.
- Use only the Python standard library.

Example:
  fn(1994) should return "MCMXCIV"."""),
    },

    "tmb-first-unique-char": {
        "id": "tmb-first-unique-char",
        "title": "first unique character",
        "summary": "frequency count, then scan for first occurrence count of 1",
        "difficulty": "medium",
        "timeout_s": 25,
        "tests": [
            {"input": "loveleetcode", "expected": 2},
            {"input": "leetcode", "expected": 0},
            {"input": "aabb", "expected": -1},
            {"input": "a", "expected": 0},
            {"input": "", "expected": -1},
        ],
        "prompt": _p("""Task: return the index of the first non-repeating character in a string.

Rules:
- Count occurrences of each character.
- Return the index of the first character that appears exactly once.
- Return -1 if every character repeats.
- Use only the Python standard library.

Example:
  fn("loveleetcode") should return 2 (the 'v' is the first unique character)."""),
    },

    "tmb-longest-substr": {
        "id": "tmb-longest-substr",
        "title": "longest substring without repeating chars",
        "summary": "sliding window with last-seen index map",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": "abcabcbb", "expected": 3},
            {"input": "bbbbb", "expected": 1},
            {"input": "pwwkew", "expected": 3},
            {"input": "", "expected": 0},
            {"input": "au", "expected": 2},
        ],
        "prompt": _p("""Task: return the length of the longest substring without repeating characters.

Rules:
- A substring is a contiguous sequence of characters.
- No character may repeat within the substring.
- Return the maximum length.
- Use only the Python standard library.

Example:
  fn("abcabcbb") should return 3 (the substring "abc")."""),
    },

    "tmb-max-subarray": {
        "id": "tmb-max-subarray",
        "title": "maximum subarray sum",
        "summary": "Kadane's algorithm, contiguous subarray maximum",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [-2, 1, -3, 4, -1, 2, 1, -5, 4], "expected": 6},
            {"input": [1], "expected": 1},
            {"input": [-1], "expected": -1},
            {"input": [5, 4, -1, 7, 8], "expected": 23},
            {"input": [-2, -1], "expected": -1},
        ],
        "prompt": _p("""Task: return the sum of the contiguous subarray with the largest sum.

Rules:
- Input is a list of integers (may include negatives).
- Find the contiguous subarray (one or more adjacent elements) with the maximum sum.
- Return that maximum sum.
- Use only the Python standard library.

Example:
  fn([-2,1,-3,4,-1,2,1,-5,4]) should return 6 (subarray [4,-1,2,1])."""),
    },

    "tmb-trap-rain-water": {
        "id": "tmb-trap-rain-water",
        "title": "trapping rain water",
        "summary": "two-pointer, running max heights, accumulate trapped water",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], "expected": 6},
            {"input": [4, 2, 0, 3, 2, 5], "expected": 9},
            {"input": [1, 2, 3, 4], "expected": 0},
            {"input": [], "expected": 0},
            {"input": [0], "expected": 0},
        ],
        "prompt": _p("""Task: compute how much water can be trapped after it rains.

Rules:
- Input is a list of non-negative integers representing elevation heights.
- Each element is a bar of width 1.
- Water can be trapped on top of bars between higher bars on both sides.
- Return the total units of water trapped.
- Use only the Python standard library.

Example:
  fn([0,1,0,2,1,0,1,3,2,1,2,1]) should return 6."""),
    },

    "tmb-sum-multiples": {
        "id": "tmb-sum-multiples",
        "title": "sum of multiples",
        "summary": "sum all multiples of 3 or 5 below n",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": 10, "expected": 23},
            {"input": 20, "expected": 78},
            {"input": 1, "expected": 0},
            {"input": 0, "expected": 0},
            {"input": 15, "expected": 45},
        ],
        "prompt": _p("""Task: return the sum of all multiples of 3 or 5 below a given number n.

Rules:
- Consider all integers from 1 up to (but not including) n.
- Sum those divisible by 3 or by 5.
- Numbers divisible by both 3 and 5 are counted once.
- Use only the Python standard library.

Example:
  fn(10) should return 23 (3 + 5 + 6 + 9)."""),
    },

    "tmb-bounding-box": {
        "id": "tmb-bounding-box",
        "title": "bounding box",
        "summary": "min/max x and y over a set of 2D points",
        "difficulty": "medium",
        "timeout_s": 25,
        "tests": [
            {"input": [[1, 2], [3, 4], [0, 5], [-2, 3]], "expected": [-2, 2, 3, 5]},
            {"input": [[0, 0], [0, 0]], "expected": [0, 0, 0, 0]},
            {"input": [[5, 5]], "expected": [5, 5, 5, 5]},
            {"input": [[-1, -1], [1, 1]], "expected": [-1, -1, 1, 1]},
            {"input": [[3, 1], [3, 2], [3, 3]], "expected": [3, 1, 3, 3]},
        ],
        "prompt": _p("""Task: compute the axis-aligned bounding box of a set of 2D points.

Rules:
- Input is a list of [x, y] point pairs.
- Return [minX, minY, maxX, maxY] covering all points.
- Use only the Python standard library.

Example:
  fn([[1,2],[3,4],[0,5],[-2,3]]) should return [-2, 2, 3, 5]."""),
    },

    "tmb-polygon-area": {
        "id": "tmb-polygon-area",
        "title": "polygon area",
        "summary": "shoelace formula, absolute area of a simple polygon",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [[0, 0], [4, 0], [4, 3], [0, 3]], "expected": 12},
            {"input": [[0, 0], [1, 0], [1, 1], [0, 1]], "expected": 1},
            {"input": [[0, 0], [2, 0], [1, 1]], "expected": 1},
            {"input": [[0, 0], [0, 0], [0, 0]], "expected": 0},
            {"input": [[0, 0], [3, 0], [3, 4], [0, 4]], "expected": 12},
        ],
        "prompt": _p("""Task: compute the area of a simple polygon given its vertices in order.

Rules:
- Input is a list of [x, y] vertices listed in order around the polygon (clockwise or counter-clockwise).
- Use the shoelace formula.
- Return the absolute area (always non-negative).
- Use only the Python standard library.

Example:
  fn([[0,0],[4,0],[4,3],[0,3]]) should return 12 (a 4x3 rectangle)."""),
    },

    "tmb-longest-palindromic-substring": {
        "id": "tmb-longest-palindromic-substring",
        "title": "longest palindromic substring",
        "summary": "expand-around-center, longest palindromic substring",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": "cbbd", "expected": "bb"},
            {"input": "racecar", "expected": "racecar"},
            {"input": "a", "expected": "a"},
            {"input": "", "expected": ""},
            {"input": "forgeeksskeegfor", "expected": "geeksskeeg"},
        ],
        "prompt": _p("""Task: return the longest palindromic substring of a string.

Rules:
- A palindrome reads the same forward and backward.
- If there are multiple longest palindromes, return any one of them.
- Use only the Python standard library.

Example:
  fn("cbbd") should return "bb".
  fn("racecar") should return "racecar"."""),
    },

    "tmb-num-islands": {
        "id": "tmb-num-islands",
        "title": "number of islands",
        "summary": "grid DFS/BFS flood fill, count connected 1-regions",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]], "expected": 3},
            {"input": [["1"]], "expected": 1},
            {"input": [["0"]], "expected": 0},
            {"input": [], "expected": 0},
            {"input": [["1", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]], "expected": 5},
        ],
        "prompt": _p("""Task: count the number of islands in a 2D grid.

Rules:
- Input is a list of lists of strings, each cell "1" (land) or "0" (water).
- An island is a group of "1"s connected horizontally or vertically (not diagonally).
- Return the number of islands.
- Use only the Python standard library.

Example:
  fn([["1","1","0"],["1","0","0"],["0","0","1"]]) should return 2."""),
    },

    "tmb-lis": {
        "id": "tmb-lis",
        "title": "longest increasing subsequence",
        "summary": "dynamic programming, longest increasing subsequence length",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [10, 9, 2, 5, 3, 7, 101, 18], "expected": 4},
            {"input": [0, 1, 0, 3, 2, 3], "expected": 4},
            {"input": [7, 7, 7, 7], "expected": 1},
            {"input": [1], "expected": 1},
            {"input": [], "expected": 0},
        ],
        "prompt": _p("""Task: return the length of the longest strictly increasing subsequence.

Rules:
- A subsequence preserves relative order but may skip elements.
- Strictly increasing means each element is greater than the previous.
- Return the maximum length.
- Use only the Python standard library.

Example:
  fn([10,9,2,5,3,7,101,18]) should return 4 (e.g. [2,3,7,101])."""),
    },

    "tmb-edit-distance": {
        "id": "tmb-edit-distance",
        "title": "edit distance",
        "summary": "Levenshtein distance, dynamic programming on two strings",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": ["horse", "ros"], "expected": 3},
            {"input": ["intention", "execution"], "expected": 5},
            {"input": ["", "abc"], "expected": 3},
            {"input": ["a", "a"], "expected": 0},
            {"input": ["abc", ""], "expected": 3},
        ],
        "prompt": _p("""Task: return the Levenshtein edit distance between two strings.

Rules:
- Input is a list of exactly two strings: [a, b].
- Edit distance = minimum number of insertions, deletions, or substitutions to turn a into b.
- Return the distance.
- Use only the Python standard library.

Example:
  fn(["horse", "ros"]) should return 3."""),
    },

    "tmb-valid-sudoku": {
        "id": "tmb-valid-sudoku",
        "title": "valid sudoku",
        "summary": "validate rows, columns, and 3x3 boxes for duplicates",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [["5", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]], "expected": True},
            {"input": [["5", "5", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]], "expected": False},
            {"input": [["6", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]], "expected": False},
            {"input": [["5", "3", "9", ".", "7", ".", ".", ".", "."], ["6", "9", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]], "expected": False},
            {"input": [[".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", ".", "."]], "expected": True},
        ],
        "prompt": _p("""Task: determine if a 9x9 Sudoku board is valid.

Rules:
- Input is a 9x9 list of lists of strings: digits "1"-"9" or "." (empty).
- A board is valid if every row, every column, and every 3x3 sub-box has no repeated digit.
- Empty cells (".") are ignored.
- Return True or False.
- Use only the Python standard library."""),
    },

    "tmb-coin-change": {
        "id": "tmb-coin-change",
        "title": "coin change",
        "summary": "dynamic programming, fewest coins to reach amount",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [[1, 2, 5], 11], "expected": 3},
            {"input": [[2], 3], "expected": -1},
            {"input": [[1], 0], "expected": 0},
            {"input": [[1, 2, 5], 100], "expected": 20},
            {"input": [[3, 5], 7], "expected": -1},
        ],
        "prompt": _p("""Task: return the fewest number of coins needed to make up a given amount.

Rules:
- Input is a list of exactly two values: [coins, amount].
- coins is a list of coin denominations (positive integers, unlimited supply).
- amount is a non-negative integer.
- Return the minimum number of coins, or -1 if the amount cannot be made.
- Use only the Python standard library.

Example:
  fn([[1,2,5], 11]) should return 3 (5+5+1)."""),
    },

    "tmb-longest-valid-parentheses": {
        "id": "tmb-longest-valid-parentheses",
        "title": "longest valid parentheses",
        "summary": "stack/DP, length of longest valid parentheses substring",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": ")()())", "expected": 4},
            {"input": "(()", "expected": 2},
            {"input": "", "expected": 0},
            {"input": "()", "expected": 2},
            {"input": "((()))", "expected": 6},
        ],
        "prompt": _p("""Task: return the length of the longest valid (well-formed) parentheses substring.

Rules:
- The string contains only '(' and ')'.
- A substring is valid if parentheses are balanced and correctly nested.
- Return the maximum length.
- Use only the Python standard library.

Example:
  fn("(()") should return 2.
  fn(")()())") should return 4."""),
    },

    "tmb-median-two-sorted": {
        "id": "tmb-median-two-sorted",
        "title": "median of two sorted arrays",
        "summary": "median of two sorted lists",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [[1, 3], [2]], "expected": 2},
            {"input": [[1, 2], [3, 4]], "expected": 2.5},
            {"input": [[], [1]], "expected": 1},
            {"input": [[0, 0], [0, 0]], "expected": 0},
            {"input": [[], [2, 3]], "expected": 2.5},
        ],
        "prompt": _p("""Task: return the median of two sorted arrays.

Rules:
- Input is a list of exactly two sorted lists: [nums1, nums2].
- Return the median value (may be a float, e.g. 2.5).
- Use only the Python standard library.

Example:
  fn([[1,3],[2]]) should return 2.
  fn([[1,2],[3,4]]) should return 2.5."""),
    },

    "tmb-sliding-window-max": {
        "id": "tmb-sliding-window-max",
        "title": "sliding window maximum",
        "summary": "monotonic deque, max in each sliding window",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": [[1, 3, -1, -3, 5, 3, 6, 7], 3], "expected": [3, 3, 5, 5, 6, 7]},
            {"input": [[1], 1], "expected": [1]},
            {"input": [[1, -1], 1], "expected": [1, -1]},
            {"input": [[9, 11], 2], "expected": [11]},
            {"input": [[4, -2], 2], "expected": [4]},
        ],
        "prompt": _p("""Task: return the maximum value in each sliding window of size k.

Rules:
- Input is a list of exactly two values: [nums, k].
- Slide a window of size k from left to right over nums.
- Return a list of the maximum of each window.
- Use only the Python standard library.

Example:
  fn([[1,3,-1,-3,5,3,6,7], 3]) should return [3,3,5,5,6,7]."""),
    },

    "tmb-n-queens": {
        "id": "tmb-n-queens",
        "title": "n-queens count",
        "summary": "backtracking, count distinct n-queens solutions",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": 8, "expected": 92},
            {"input": 4, "expected": 2},
            {"input": 1, "expected": 1},
            {"input": 5, "expected": 10},
            {"input": 2, "expected": 0},
        ],
        "prompt": _p("""Task: return the number of distinct solutions to the n-queens puzzle.

Rules:
- Place n queens on an n x n board so no two queens attack each other (no same row, column, or diagonal).
- Input is an integer n.
- Return the number of distinct arrangements.
- Use only the Python standard library.

Example:
  fn(4) should return 2.
  fn(8) should return 92."""),
    },

    "tmb-atoi": {
        "id": "tmb-atoi",
        "title": "string to integer (atoi)",
        "summary": "whitespace, sign, digits, and 32-bit overflow clamping",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": "   -42", "expected": -42},
            {"input": "4193 with words", "expected": 4193},
            {"input": "words and 987", "expected": 0},
            {"input": "-91283472332", "expected": -2147483648},
            {"input": "  +42abc", "expected": 42},
            {"input": "2147483648", "expected": 2147483647},
            {"input": "  -0012a42", "expected": -12},
            {"input": "   +0 123", "expected": 0},
            {"input": "3.14159", "expected": 3},
            {"input": "", "expected": 0},
        ],
        "prompt": _p("""Task: implement the classic atoi — convert a string to a 32-bit signed integer.

Rules:
- fn(s) takes one string.
- Skip leading whitespace.
- An optional '+' or '-' sign may follow the whitespace.
- Read digits until a non-digit is reached or the string ends.
- If no digits are read, return 0.
- Clamp to the 32-bit signed integer range [-2147483648, 2147483647].
- Use only the Python standard library.

Examples:
  fn("   -42") should return -42.
  fn("4193 with words") should return 4193.
  fn("words and 987") should return 0.
  fn("-91283472332") should return -2147483648."""),
    },

    "tmb-csv-parse": {
        "id": "tmb-csv-parse",
        "title": "parse quoted CSV",
        "summary": "RFC-4180 style: quoted fields, escaped quotes, commas and newlines inside quotes",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": "a,b,c\n1,2,3", "expected": [["a", "b", "c"], ["1", "2", "3"]]},
            {"input": "\"a,b\",c", "expected": [["a,b", "c"]]},
            {"input": "\"he said \"\"hi\"\"\",x", "expected": [["he said \"hi\"", "x"]]},
            {"input": "a,\"b\nc\",d", "expected": [["a", "b\nc", "d"]]},
            {"input": "\"\"", "expected": [[""]]},
            {"input": "a,,c", "expected": [["a", "", "c"]]},
            {"input": "", "expected": []},
        ],
        "prompt": _p("""Task: parse a CSV string into a list of rows, where each row is a list of field strings.

Rules:
- fn(text) takes one string.
- Rows are separated by newline characters.
- Fields are separated by commas.
- A field may be wrapped in double quotes; inside quotes, a comma or newline is literal.
- Two consecutive double quotes inside a quoted field represent one literal double quote.
- Unquoted fields are kept exactly as-is.
- Return a list of rows; an empty input returns an empty list.
- Use only the Python standard library.

Examples:
  fn("a,b,c\\n1,2,3") should return [["a","b","c"],["1","2","3"]].
  fn("\\"a,b\\",c") should return [["a,b","c"]].
  fn("\\"he said \\"\\"hi\\"\\"\\",x") should return [["he said \\"hi\\"","x"]]."""),
    },

    "tmb-basic-calc": {
        "id": "tmb-basic-calc",
        "title": "basic calculator",
        "summary": "evaluate expression with + - * / and parentheses, division truncates toward zero",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": "1 + 2 * 3", "expected": 7},
            {"input": " 3+5 / 2 ", "expected": 5},
            {"input": "2*(3+4)", "expected": 14},
            {"input": "(1+(4+5+2)-3)+(6+8)", "expected": 23},
            {"input": "10/3", "expected": 3},
            {"input": "-3+4", "expected": 1},
            {"input": "-10/3", "expected": -3},
            {"input": " 2-1 + 2 ", "expected": 3},
            {"input": "14-3*2", "expected": 8},
        ],
        "prompt": _p("""Task: evaluate an arithmetic expression string and return the integer result.

Rules:
- fn(expr) takes one string containing non-negative integers, operators + - * /, and parentheses.
- Spaces may appear anywhere and must be ignored.
- Multiplication and division bind tighter than addition and subtraction.
- Division truncates toward zero (int(a / b)), e.g. 10/3 -> 3, -10/3 -> -3.
- Unary minus is allowed (e.g. "-3+4" -> 1).
- Assume input is valid.
- Use only the Python standard library.

Examples:
  fn("1 + 2 * 3") should return 7.
  fn(" 3+5 / 2 ") should return 5.
  fn("2*(3+4)") should return 14."""),
    },

    "tmb-text-justify": {
        "id": "tmb-text-justify",
        "title": "text justification",
        "summary": "greedy line packing with even space distribution and left-justified last line",
        "difficulty": "hard",
        "timeout_s": 35,
        "tests": [
            {"input": [["This", "is", "an", "example", "of", "text", "justification."], 16], "expected": ["This    is    an", "example  of text", "justification.  "]},
            {"input": [["What", "must", "be", "acknowledgment", "shall", "be"], 16], "expected": ["What   must   be", "acknowledgment  ", "shall be        "]},
            {"input": [["a"], 3], "expected": ["a  "]},
            {"input": [["a", "b", "c"], 1], "expected": ["a", "b", "c"]},
            {"input": [["hello", "world"], 10], "expected": ["hello     ", "world     "]},
        ],
        "prompt": _p("""Task: full text justification.

Rules:
- fn(args) receives [words, maxWidth]: a list of words and a line width.
- Pack words greedily: each line holds as many words as fit (at least one space between words).
- Every line except the last must be exactly maxWidth characters long.
- Distribute extra spaces between words as evenly as possible; when they cannot be split evenly, the LEFT gaps get more spaces.
- A single-word line is left-justified and padded with trailing spaces.
- The LAST line is left-justified with single spaces between words, padded with trailing spaces.
- Return a list of line strings.
- Use only the Python standard library.

Example:
  fn([["This","is","an","example","of","text","justification."], 16])
  should return ["This    is    an","example  of text","justification.  "]."""),
    },

    "tmb-wildcard-match": {
        "id": "tmb-wildcard-match",
        "title": "wildcard matching",
        "summary": "DP matching with '?' for single char and '*' for any sequence",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": ["aa", "a"], "expected": False},
            {"input": ["aa", "*"], "expected": True},
            {"input": ["cb", "?a"], "expected": False},
            {"input": ["adceb", "*a*b"], "expected": True},
            {"input": ["acdcb", "a*c?b"], "expected": False},
            {"input": ["", "*"], "expected": True},
            {"input": ["abc", "a*c"], "expected": True},
        ],
        "prompt": _p("""Task: wildcard pattern matching.

Rules:
- fn(args) receives [s, p]: the string and the pattern.
- '?' in the pattern matches exactly one character.
- '*' in the pattern matches any sequence of characters (including the empty sequence).
- The whole string must match the whole pattern.
- Return True or False.
- Use only the Python standard library.

Examples:
  fn(["aa","a"]) should return False.
  fn(["aa","*"]) should return True.
  fn(["adceb","*a*b"]) should return True.
  fn(["acdcb","a*c?b"]) should return False."""),
    },

    "tmb-valid-number": {
        "id": "tmb-valid-number",
        "title": "valid number",
        "summary": "validate integer/decimal with optional exponent, strict grammar",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": "0", "expected": True},
            {"input": " 0.1 ", "expected": False},
            {"input": "abc", "expected": False},
            {"input": "2e10", "expected": True},
            {"input": "-90E3", "expected": True},
            {"input": "1e", "expected": False},
            {"input": "e3", "expected": False},
            {"input": "99e2.5", "expected": False},
            {"input": "--6", "expected": False},
            {"input": "-+3", "expected": False},
            {"input": "95a54e53", "expected": False},
            {"input": ".", "expected": False},
            {"input": "-.9", "expected": True},
            {"input": "4.", "expected": True},
            {"input": "1.e+", "expected": False},
            {"input": "+.8", "expected": True},
        ],
        "prompt": _p("""Task: determine whether a string is a valid number in decimal or scientific notation.

Rules:
- fn(s) takes one string.
- A valid number is an integer or decimal, optionally followed by an exponent: mantissa [e|E] [+|-] exponent-digits.
- The mantissa is: [+|-] followed by either digits, digits with a decimal point, or a decimal point with digits (at least one digit must appear in the mantissa).
- The exponent part requires at least one digit after the optional sign.
- No surrounding whitespace is allowed; no letters other than e/E; no other symbols.
- Return True or False.
- Use only the Python standard library.

Examples:
  fn("2e10") should return True.
  fn("-.9") should return True.
  fn("4.") should return True.
  fn("1e") should return False.
  fn("99e2.5") should return False.
  fn(" 0.1 ") should return False."""),
    },

    "tmb-course-schedule": {
        "id": "tmb-course-schedule",
        "title": "course schedule",
        "summary": "detect cycles in a prerequisite graph (topological order feasibility)",
        "difficulty": "hard",
        "timeout_s": 35,
        "tests": [
            {"input": [2, [[1, 0]]], "expected": True},
            {"input": [2, [[1, 0], [0, 1]]], "expected": False},
            {"input": [5, [[1, 0], [2, 1], [3, 2], [4, 3]]], "expected": True},
            {"input": [3, [[0, 1], [0, 2], [1, 2]]], "expected": True},
            {"input": [3, [[0, 1], [1, 2], [2, 0]]], "expected": False},
            {"input": [1, []], "expected": True},
            {"input": [4, [[0, 1], [2, 3], [1, 2], [3, 1]]], "expected": False},
        ],
        "prompt": _p("""Task: determine whether all courses can be finished given prerequisites.

Rules:
- fn(args) receives [numCourses, prerequisites].
- numCourses is the number of courses labeled 0 through numCourses-1.
- prerequisites is a list of pairs [a, b] meaning course a depends on course b.
- Return True if it is possible to take all courses in some order (no cycle in the dependency graph), False otherwise.
- Use only the Python standard library.

Examples:
  fn([2, [[1,0]]]) should return True.
  fn([2, [[1,0],[0,1]]]) should return False."""),
    },

    "tmb-min-window": {
        "id": "tmb-min-window",
        "title": "minimum window substring",
        "summary": "sliding window, shortest substring containing all target characters",
        "difficulty": "hard",
        "timeout_s": 35,
        "tests": [
            {"input": ["ADOBECODEBANC", "ABC"], "expected": "BANC"},
            {"input": ["a", "a"], "expected": "a"},
            {"input": ["a", "aa"], "expected": ""},
            {"input": ["ab", "b"], "expected": "b"},
            {"input": ["aa", "aa"], "expected": "aa"},
            {"input": ["cabeca", "cae"], "expected": "eca"},
        ],
        "prompt": _p("""Task: find the minimum-length substring of s that contains every character of t.

Rules:
- fn(args) receives [s, t]: two strings.
- The window must contain each character of t at least as many times as it appears in t.
- Return the smallest such substring; if multiple tie, any of them is fine.
- Return an empty string if no window exists.
- Use only the Python standard library.

Examples:
  fn(["ADOBECODEBANC","ABC"]) should return "BANC".
  fn(["a","aa"]) should return ""."""),
    },

    "tmb-regex-match": {
        "id": "tmb-regex-match",
        "title": "regular expression matching",
        "summary": "DP matching with '.' for single char and '*' for zero-or-more of preceding",
        "difficulty": "hard",
        "timeout_s": 30,
        "tests": [
            {"input": ["aa", "a"], "expected": False},
            {"input": ["aa", "a*"], "expected": True},
            {"input": ["ab", ".*"], "expected": True},
            {"input": ["aab", "c*a*b"], "expected": True},
            {"input": ["mississippi", "mis*is*p*."], "expected": False},
            {"input": ["", "a*"], "expected": True},
            {"input": ["a", "ab*"], "expected": True},
        ],
        "prompt": _p("""Task: regular expression matching supporting '.' and '*'.

Rules:
- fn(args) receives [s, p]: the string and the pattern.
- '.' matches any single character.
- '*' matches zero or more of the preceding element (e.g. "a*" matches "", "a", "aa", ...).
- The match must cover the ENTIRE string.
- Return True or False.
- Use only the Python standard library.

Examples:
  fn(["aa","a"]) should return False.
  fn(["aa","a*"]) should return True.
  fn(["aab","c*a*b"]) should return True.
  fn(["mississippi","mis*is*p*."]) should return False."""),
    },

    "tmb-max-points-line": {
        "id": "tmb-max-points-line",
        "title": "max points on a line",
        "summary": "group points by reduced slope, count duplicates, find collinear maximum",
        "difficulty": "hard",
        "timeout_s": 35,
        "tests": [
            {"input": [[1, 1], [2, 2], [3, 3]], "expected": 3},
            {"input": [[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]], "expected": 4},
            {"input": [[0, 0]], "expected": 1},
            {"input": [[0, 0], [1, 1], [0, 0]], "expected": 3},
            {"input": [[1, 1], [2, 2], [3, 3], [3, 4]], "expected": 3},
            {"input": [[0, 0], [1, 0], [2, 0], [1, 1]], "expected": 3},
        ],
        "prompt": _p("""Task: return the maximum number of points that lie on the same straight line.

Rules:
- fn(points) takes one list of [x, y] integer pairs.
- A point may appear more than once; each occurrence counts.
- Points on a line share the same reduced slope from an anchor point.
- Return the maximum count of collinear points (a single point returns 1).
- Use only the Python standard library.

Examples:
  fn([[1,1],[2,2],[3,3]]) should return 3.
  fn([[0,0],[1,1],[0,0]]) should return 3."""),
    },
}


# ---------------------------------------------------------------------------
# trial problems — candidates for tightening the scored set. They are kept OUT
# of LEVELS (and so out of every recorded score) until a run shows they really
# separate models: run them one at a time with --tests=<id>, then promote the
# keepers by moving the entry up into PROBLEMS and deleting it here.
#
# Same ids, inputs and expected values as the JS harness. All five return a
# string or a flat list of ints, so grading is exact in both harnesses.
TRIAL_PROBLEMS = {
    "tmb-cents-split": {
        "id": "tmb-cents-split",
        "title": "split a total into n exact parts",
        "summary": "integer division, remainder distribution, parts must sum exactly",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": [1000, 3], "expected": [334, 333, 333]},
            {"input": [5, 2], "expected": [3, 2]},
            {"input": [1, 3], "expected": [1, 0, 0]},
            {"input": [0, 4], "expected": [0, 0, 0, 0]},
            {"input": [999, 7], "expected": [143, 143, 143, 143, 143, 142, 142]},
        ],
        "prompt": _p("""Task: split a total into n parts as evenly as possible, without losing a unit.

Input: fn(args) receives [total, n]: a non-negative integer total and a positive integer n.

Rules:
- Return a list of exactly n non-negative integers that sum EXACTLY to total.
- Split as evenly as possible: every part is either total // n or one more than that.
- When the total does not divide evenly, give the extra units to the EARLIEST parts (lowest indexes).
- Use only the Python standard library.

Examples:
  fn([1000, 3]) should return [334, 333, 333].
  fn([5, 2]) should return [3, 2].
  fn([1, 3]) should return [1, 0, 0]."""),
    },

    "tmb-truncate-utf8": {
        "id": "tmb-truncate-utf8",
        "title": "truncate a string to a byte budget",
        "summary": "UTF-8 byte length vs string length, never split a character",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": ["héllo", 2], "expected": "h"},
            {"input": ["héllo", 3], "expected": "hé"},
            {"input": ["日本語", 4], "expected": "日"},
            {"input": ["abc", 10], "expected": "abc"},
            {"input": ["😀x", 4], "expected": "😀"},
            {"input": ["héllo", 0], "expected": ""},
        ],
        "prompt": _p("""Task: truncate a string so that its UTF-8 encoding fits inside a byte budget.

Input: fn(args) receives [s, max_bytes]: a string and a non-negative integer byte budget.

Rules:
- Return the longest prefix of s whose UTF-8 encoding is at most max_bytes bytes long.
- Never split a character: if the next character does not fit, stop before it.
- Characters outside ASCII may take 2, 3, or 4 bytes in UTF-8.
- Use only the Python standard library.

Examples:
  fn(["héllo", 2]) should return "h".
  fn(["héllo", 3]) should return "hé".
  fn(["日本語", 4]) should return "日".
  fn(["😀x", 4]) should return "😀"."""),
    },

    "tmb-number-to-words": {
        "id": "tmb-number-to-words",
        "title": "integer to English words",
        "summary": "scale words, hyphenated tens, no 'and'",
        "difficulty": "simple",
        "timeout_s": 25,
        "tests": [
            {"input": 0, "expected": "zero"},
            {"input": 21, "expected": "twenty-one"},
            {"input": 101, "expected": "one hundred one"},
            {"input": 1000, "expected": "one thousand"},
            {"input": 1234, "expected": "one thousand two hundred thirty-four"},
            {"input": 100000, "expected": "one hundred thousand"},
            {"input": 1000000, "expected": "one million"},
        ],
        "prompt": _p("""Task: spell a non-negative integer in English words.

Input: fn(n) takes one integer from 0 to 999999999 inclusive.

Rules:
- Return lowercase words separated by single spaces.
- Write tens (21-99) with a hyphen: 21 -> "twenty-one".
- Never use the word "and": 101 -> "one hundred one", not "one hundred and one".
- Use scale words "thousand" and "million": 1000 -> "one thousand", 1000000 -> "one million".
- Drop empty scales: 100000 -> "one hundred thousand".
- Do not put a hyphen between a scale word and the rest.
- Use only the Python standard library.

Examples:
  fn(0) should return "zero".
  fn(21) should return "twenty-one".
  fn(101) should return "one hundred one".
  fn(1234) should return "one thousand two hundred thirty-four".
  fn(1000000) should return "one million"."""),
    },

    "tmb-path-normalize": {
        "id": "tmb-path-normalize",
        "title": "normalize a unix path",
        "summary": "collapse . and .., clamp at the root, relative vs absolute result",
        "difficulty": "simple",
        "timeout_s": 25,
        "tests": [
            {"input": "/a//b/./c/../d", "expected": "/a/b/d"},
            {"input": "a/b/../../c", "expected": "c"},
            {"input": "/../..", "expected": "/"},
            {"input": "/a/b/", "expected": "/a/b"},
            {"input": "./a", "expected": "a"},
            {"input": "a/..", "expected": "."},
            {"input": "../../x", "expected": "../../x"},
        ],
        "prompt": _p("""Task: normalize a Unix-style path string (like POSIX normpath).

Input: fn(path) takes one string.

Rules:
- Split on "/", then process the segments in order:
  - ignore empty segments (from "//") and "." segments;
  - ".." removes the previous segment if there is one to remove;
  - at the start of an ABSOLUTE path there is nothing to remove, so a leading ".." is dropped;
  - in a RELATIVE path, a ".." with no plain segment before it is kept (it cannot be resolved);
  - anything else is a normal segment.
- Rejoin the surviving segments with "/".
- A leading "/" in the input makes the result absolute; otherwise the result is relative.
- If the result has no segments: return "/" for an absolute input, and "." for a relative one.
- Ignore a trailing slash.
- Use only the Python standard library.

Examples:
  fn("/a//b/./c/../d") should return "/a/b/d".
  fn("a/b/../../c") should return "c".
  fn("/../..") should return "/".
  fn("a/..") should return ".".
  fn("../../x") should return "../../x"."""),
    },

    "tmb-kv-serialize": {
        "id": "tmb-kv-serialize",
        "title": "serialize k=v string",
        "summary": "escape %, comma and equals in keys and values, escape % first",
        "difficulty": "simple",
        "timeout_s": 20,
        "tests": [
            {"input": {"a": "1", "b": "2"}, "expected": "a=1,b=2"},
            {"input": {"a b": "c,d"}, "expected": "a b=c%2Cd"},
            {"input": {"x=y": "pct%"}, "expected": "x%3Dy=pct%25"},
            {"input": {"k": "a=b"}, "expected": "k=a%3Db"},
            {"input": {}, "expected": ""},
        ],
        "prompt": _p("""Task: serialize a dict into a compact "key=value" string — the inverse of parsing it back.

Input: fn(obj) takes one dict whose values are strings. Keys are strings too.

Rules:
- Produce "key=value" pairs joined by commas, in the dict's key order.
- In BOTH keys and values, escape these characters, and escape the percent sign FIRST:
  - "%" becomes "%25"
  - "," becomes "%2C"
  - "=" becomes "%3D"
- Escaping must not double-escape: a literal "%" always becomes exactly "%25".
- An empty dict returns an empty string.
- Use only the Python standard library.

Examples:
  fn({"a": "1", "b": "2"}) should return "a=1,b=2".
  fn({"a b": "c,d"}) should return "a b=c%2Cd".
  fn({"x=y": "pct%"}) should return "x%3Dy=pct%25"."""),
    },
}

# Every problem the harness can run: the scored set plus the trial candidates.
# LEVELS still resolves to PROBLEMS only, so trial problems never enter a
# recorded score until they are promoted.
ALL_PROBLEMS = {**PROBLEMS, **TRIAL_PROBLEMS}

# Run levels: which problems (and how many tests each) to run.
#   low   - the 10 hardest problems, 1 test each
#   med   - all problems, 1 test each (the default / "normal")
#   large - all problems, every edge-case test each
LEVELS = {
    "low": ["tmb-rpn-eval", "tmb-lru-ops", "tmb-longest-substr", "tmb-max-subarray", "tmb-trap-rain-water",
            "tmb-polygon-area", "tmb-merge-ranges", "tmb-valid-parentheses", "tmb-roman-to-int", "tmb-int-to-roman"],
    "med": list(PROBLEMS.keys()),
    "large": list(PROBLEMS.keys()),
}


# ---------------------------------------------------------------------------
# deterministic reference implementations, used by --mode=direct to validate
# the extraction/grading path end to end without a model.

FIXTURES = {
    # --- trial candidates ---------------------------------------------------
    "tmb-cents-split": '''def fn(args):
    total, n = args
    base = total // n
    extra = total - base * n
    return [base + (1 if i < extra else 0) for i in range(n)]
''',
    "tmb-truncate-utf8": '''def fn(args):
    s, max_bytes = args
    out = ""
    for ch in s:
        if len((out + ch).encode("utf-8")) > max_bytes:
            break
        out += ch
    return out
''',
    "tmb-number-to-words": '''ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def under1000(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        r = n % 10
        return TENS[n // 10] + ("-" + ONES[r] if r else "")
    r = n % 100
    return ONES[n // 100] + " hundred" + (" " + under1000(r) if r else "")

def fn(n):
    if n == 0:
        return "zero"
    parts = []
    rest = n
    for value, name in ((1000000, "million"), (1000, "thousand")):
        if rest >= value:
            parts.append(under1000(rest // value) + " " + name)
            rest %= value
    if rest:
        parts.append(under1000(rest))
    return " ".join(parts)
''',
    "tmb-path-normalize": '''def fn(path):
    absolute = path.startswith("/")
    out = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if out and out[-1] != "..":
                out.pop()
            elif not absolute:
                out.append("..")
            continue
        out.append(part)
    joined = "/".join(out)
    if absolute:
        return "/" + joined
    return joined or "."
''',
    "tmb-kv-serialize": '''def fn(obj):
    def esc(s):
        return str(s).replace("%", "%25").replace(",", "%2C").replace("=", "%3D")
    return ",".join(esc(k) + "=" + esc(v) for k, v in obj.items())
''',

    "tmb-jwt": '''import json
import base64

def fn(token):
    parts = token.split(".")
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    text = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
    return json.loads(text)
''',
    "tmb-slug": '''import re

def fn(s):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower())
    return s.strip("-")
''',
    "tmb-kv-parse": '''def fn(s):
    obj = {}
    for pair in s.split(","):
        eq = pair.find("=")
        if eq == -1:
            continue
        k = pair[:eq].strip()
        v = pair[eq + 1:].strip()
        if k:
            obj[k] = v
    return obj
''',
    "tmb-fix-index-delim": '''def fn(s):
    obj = {}
    for pair in s.split(","):
        eq = pair.find("=")
        if eq == -1:
            continue
        k = pair[:eq].strip()
        v = pair[eq + 1:].strip()
        obj[k] = v
    return obj
''',
    "tmb-merge-ranges": '''def fn(ranges):
    if not ranges:
        return []
    out = []
    for start, end in sorted(ranges):
        if not out or start > out[-1][1]:
            out.append([start, end])
        else:
            out[-1][1] = max(out[-1][1], end)
    return out
''',
    "tmb-rpn-eval": '''def fn(tokens):
    stack = []
    for token in tokens:
        if token in "+-*/":
            b = stack.pop()
            a = stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[-1]
''',
    "tmb-lru-ops": '''def fn(data):
    capacity = data["capacity"]
    cache = {}
    order = []
    out = []

    def touch(key):
        if key in order:
            order.remove(key)
        order.append(key)

    for op in data["ops"]:
        if op[0] == "get":
            key = op[1]
            if key not in cache:
                out.append(-1)
                continue
            touch(key)
            out.append(cache[key])
        else:
            key, value = op[1], op[2]
            if key not in cache and len(cache) >= capacity:
                oldest = order.pop(0)
                del cache[oldest]
            cache[key] = value
            touch(key)
    return out
''',
    "tmb-reverse-words": '''def fn(s):
    return " ".join(s.split()[::-1])
''',
    "tmb-fizzbuzz": '''def fn(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
''',
    "tmb-palindrome": '''import re

def fn(s):
    c = re.sub(r"[^a-z0-9]", "", s.lower())
    return c == c[::-1]
''',
    "tmb-valid-parentheses": '''def fn(s):
    stack = []
    close = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif not stack or stack.pop() != close[ch]:
            return False
    return len(stack) == 0
''',
    "tmb-roman-to-int": '''def fn(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        cur = v[ch]
        nxt = v[s[i + 1]] if i + 1 < len(s) else 0
        total += -cur if cur < nxt else cur
    return total
''',
    "tmb-int-to-roman": '''def fn(num):
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
             (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    result = ""
    for value, symbol in table:
        while num >= value:
            result += symbol
            num -= value
    return result
''',
    "tmb-first-unique-char": '''from collections import Counter

def fn(s):
    count = Counter(s)
    for i, ch in enumerate(s):
        if count[ch] == 1:
            return i
    return -1
''',
    "tmb-longest-substr": '''def fn(s):
    seen = {}
    start = 0
    best = 0
    for i, ch in enumerate(s):
        if ch in seen and seen[ch] >= start:
            start = seen[ch] + 1
        seen[ch] = i
        best = max(best, i - start + 1)
    return best
''',
    "tmb-max-subarray": '''def fn(nums):
    best = current = nums[0]
    for x in nums[1:]:
        current = max(x, current + x)
        best = max(best, current)
    return best
''',
    "tmb-trap-rain-water": '''def fn(height):
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            left_max = max(left_max, height[left])
            water += left_max - height[left]
            left += 1
        else:
            right_max = max(right_max, height[right])
            water += right_max - height[right]
            right -= 1
    return water
''',
    "tmb-sum-multiples": '''def fn(n):
    return sum(i for i in range(1, n) if i % 3 == 0 or i % 5 == 0)
''',
    "tmb-bounding-box": '''def fn(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]
''',
    "tmb-polygon-area": '''def fn(pts):
    area = 0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2
''',
    "tmb-longest-palindromic-substring": '''def fn(s):
    if not s:
        return ""
    start = 0
    max_len = 1
    for i in range(len(s)):
        for l, r in ((i, i), (i, i + 1)):
            a, b = l, r
            while a >= 0 and b < len(s) and s[a] == s[b]:
                a -= 1
                b += 1
            length = b - a - 1
            if length > max_len:
                max_len = length
                start = a + 1
    return s[start:start + max_len]
''',
    "tmb-num-islands": '''def fn(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and not seen[r][c]:
                count += 1
                stack = [(r, c)]
                seen[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1" and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
    return count
''',
    "tmb-lis": '''def fn(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
''',
    "tmb-edit-distance": '''def fn(args):
    a, b = args
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
''',
    "tmb-valid-sudoku": '''def fn(board):
    rows = set()
    cols = set()
    boxes = set()
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v == ".":
                continue
            b = r // 3 * 3 + c // 3
            rk = ("r", r, v)
            ck = ("c", c, v)
            bk = ("b", b, v)
            if rk in rows or ck in cols or bk in boxes:
                return False
            rows.add(rk)
            cols.add(ck)
            boxes.add(bk)
    return True
''',
    "tmb-coin-change": '''def fn(args):
    coins, amount = args
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return -1 if dp[amount] == float("inf") else dp[amount]
''',
    "tmb-longest-valid-parentheses": '''def fn(s):
    best = 0
    stack = [-1]
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                best = max(best, i - stack[-1])
    return best
''',
    "tmb-median-two-sorted": '''def fn(args):
    a, b = args
    merged = sorted(a + b)
    n = len(merged)
    if n % 2 == 1:
        return merged[n // 2]
    return (merged[n // 2 - 1] + merged[n // 2]) / 2
''',
    "tmb-sliding-window-max": '''def fn(args):
    nums, k = args
    out = []
    dq = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.pop(0)
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out
''',
    "tmb-n-queens": '''def fn(n):
    count = 0
    cols = set()
    diag = set()
    anti = set()

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for c in range(n):
            if c in cols or row - c in diag or row + c in anti:
                continue
            cols.add(c)
            diag.add(row - c)
            anti.add(row + c)
            backtrack(row + 1)
            cols.remove(c)
            diag.remove(row - c)
            anti.remove(row + c)

    backtrack(0)
    return count
''',
    "tmb-atoi": '''def fn(s):
    s = str(s)
    i = 0
    while i < len(s) and s[i] == " ":
        i += 1
    sign = 1
    if i < len(s) and s[i] in "+-":
        if s[i] == "-":
            sign = -1
        i += 1
    num = 0
    while i < len(s) and s[i].isdigit():
        num = num * 10 + int(s[i])
        if num > 2147483648:
            num = 2147483648
            break
        i += 1
    num *= sign
    return max(-2147483648, min(2147483647, num))
''',
    "tmb-csv-parse": '''def fn(text):
    rows = []
    row = []
    field = ""
    in_quotes = False
    field_quoted = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    field += '"'
                    i += 1
                else:
                    in_quotes = False
            else:
                field += ch
        elif ch == '"':
            in_quotes = True
            field_quoted = True
        elif ch == ",":
            row.append(field)
            field = ""
            field_quoted = False
        elif ch == "\\n":
            row.append(field)
            field = ""
            rows.append(row)
            row = []
            field_quoted = False
        else:
            field += ch
        i += 1
    row.append(field)
    if len(row) > 1 or row[0] != "" or field_quoted:
        rows.append(row)
    return rows
''',
    "tmb-basic-calc": '''import re

def fn(expr):
    s = re.sub(r"\\s+", "", expr)
    norm = ""
    for i, ch in enumerate(s):
        if ch in "+-" and (i == 0 or s[i - 1] in "(-+*/"):
            norm += "0"
        norm += ch
    s = norm
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "+-*/()":
            tokens.append(ch)
            i += 1
        else:
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            tokens.append(num)
    prec = {"+": 1, "-": 1, "*": 2, "/": 2}
    out = []
    ops = []
    for t in tokens:
        if t.isdigit():
            out.append(t)
        elif t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                out.append(ops.pop())
            ops.pop()
        else:
            while ops and ops[-1] != "(" and prec[ops[-1]] >= prec[t]:
                out.append(ops.pop())
            ops.append(t)
    while ops:
        out.append(ops.pop())
    stack = []
    for t in out:
        if t.isdigit():
            stack.append(int(t))
        else:
            b = stack.pop()
            a = stack.pop()
            if t == "+":
                stack.append(a + b)
            elif t == "-":
                stack.append(a - b)
            elif t == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))
    return stack[-1]
''',
    "tmb-text-justify": '''def fn(args):
    words, max_width = args
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur and cur_len + len(cur) + len(w) > max_width:
            lines.append((cur, cur_len))
            cur = []
            cur_len = 0
        cur.append(w)
        cur_len += len(w)
    if cur:
        lines.append((cur, cur_len))
    out = []
    for i, (ws, length) in enumerate(lines):
        is_last = i == len(lines) - 1
        gap_count = len(ws) - 1
        fill = max_width - length
        if is_last or gap_count == 0:
            out.append(" ".join(ws) + " " * (max_width - length - gap_count))
        else:
            base = fill // gap_count
            extra = fill % gap_count
            s = ""
            for j, w in enumerate(ws):
                if j > 0:
                    s += " " * (base + (1 if extra > 0 else 0))
                    if extra > 0:
                        extra -= 1
                s += w
            out.append(s)
    return out
''',
    "tmb-wildcard-match": '''def fn(args):
    s, p = args
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
            else:
                dp[i][j] = (p[j - 1] == "?" or p[j - 1] == s[i - 1]) and dp[i - 1][j - 1]
    return dp[m][n]
''',
    "tmb-valid-number": '''import re

def fn(s):
    s = str(s)
    if not s:
        return False
    return bool(re.match(r"^[+-]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][+-]?\\d+)?$", s))
''',
    "tmb-course-schedule": '''def fn(args):
    num_courses, prerequisites = args
    adj = [[] for _ in range(num_courses)]
    indeg = [0] * num_courses
    for a, b in prerequisites:
        adj[b].append(a)
        indeg[a] += 1
    queue = [i for i in range(num_courses) if indeg[i] == 0]
    done = 0
    while queue:
        c = queue.pop(0)
        done += 1
        for nxt in adj[c]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return done == num_courses
''',
    "tmb-min-window": '''def fn(args):
    s, t = args
    if not t:
        return ""
    need = {}
    for ch in t:
        need[ch] = need.get(ch, 0) + 1
    have = 0
    want = len(need)
    left = 0
    best = ""
    best_len = float("inf")
    win = {}
    for right, ch in enumerate(s):
        win[ch] = win.get(ch, 0) + 1
        if ch in need and win[ch] == need[ch]:
            have += 1
        while have == want:
            length = right - left + 1
            if length < best_len:
                best_len = length
                best = s[left:right + 1]
            lc = s[left]
            win[lc] -= 1
            if lc in need and win[lc] < need[lc]:
                have -= 1
            left += 1
    return best
''',
    "tmb-regex-match": '''def fn(args):
    s, p = args
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                dp[i][j] = dp[i][j - 2] or ((p[j - 2] == "." or p[j - 2] == s[i - 1]) and dp[i - 1][j])
            else:
                dp[i][j] = (p[j - 1] == "." or p[j - 1] == s[i - 1]) and dp[i - 1][j - 1]
    return dp[m][n]
''',
    "tmb-max-points-line": '''import math

def fn(points):
    if len(points) < 2:
        return len(points)
    best = 1
    for i in range(len(points)):
        slopes = {}
        same = 0
        for j in range(len(points)):
            if i == j:
                continue
            x1, y1 = points[i]
            x2, y2 = points[j]
            if x1 == x2 and y1 == y2:
                same += 1
                continue
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            if dx == 0:
                dy = abs(dy)
            key = (dx, dy)
            slopes[key] = slopes.get(key, 0) + 1
        mx = max(slopes.values()) if slopes else 0
        best = max(best, mx + same + 1)
    return best
''',
}


# ---------------------------------------------------------------------------
# grading

def problem_tests(problem):
    return problem.get("tests") or [{"input": problem["testInput"], "expected": problem["expected"]}]


def line_matches(expected, candidate):
    if not candidate:
        return False
    if candidate == json.dumps(expected):
        return True
    try:
        if json.loads(candidate) == expected:
            return True
    except (ValueError, TypeError):
        pass
    if isinstance(expected, dict):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return all(parsed.get(k) == v for k, v in expected.items())
        except (ValueError, TypeError):
            pass
    expected_text = str(expected)
    if expected_text and expected_text in candidate:
        return True
    return False


def grade_tests(expecteds, stdout):
    if not stdout:
        return False
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    if len(lines) < len(expecteds):
        return False
    answers = lines[-len(expecteds):]
    return all(line_matches(exp, ans) for exp, ans in zip(expecteds, answers))


def build_source(code, tests):
    # JSON.stringify is a JS global; Python needs an explicit import for the
    # harness's print(json.dumps(...)) calls (harmless if the code imports it too).
    #
    # ensure_ascii=False matters for non-BMP inputs (emoji): the default escapes
    # an astral character as a surrogate PAIR ("\ud83d\ude00"), which Python then
    # compiles into real lone surrogates, so a candidate that encodes its string
    # to UTF-8 dies with UnicodeEncodeError before any answer is produced.
    calls = "\n".join("print(json.dumps(fn(%s)))" % json.dumps(t["input"], ensure_ascii=False)
                      for t in tests)
    return "import json\n\n" + code + "\n\n# Harness calls\n" + calls + "\n"


def run_python_script(source, python, timeout_ms):
    with tempfile.TemporaryDirectory(prefix="tiny-py-benchmark-") as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        started = time.time()
        try:
            proc = subprocess.run([python, path], capture_output=True, text=True,
                                  timeout=timeout_ms / 1000.0, cwd=ROOT)
            return {
                "pass": False,
                "elapsed_ms": int((time.time() - started) * 1000),
                "stdout": (proc.stdout or "").strip(),
                "stderr": (proc.stderr or "").strip(),
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "pass": False,
                "elapsed_ms": int((time.time() - started) * 1000),
                "stdout": (e.stdout or "").strip(),
                "stderr": (e.stderr or "").strip(),
                "exit_code": -1,
                "error": "python timeout after %ds" % (timeout_ms / 1000),
            }
        except FileNotFoundError:
            return {
                "pass": False,
                "elapsed_ms": 0,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "error": "python interpreter not found: %s" % python,
            }


def classify_failure(result):
    if not result or result.get("pass"):
        return ""
    if result.get("error") and "timeout" in str(result["error"]).lower():
        return "timeout"
    hay = "%s\n%s" % (result.get("stderr") or "", result.get("error") or "")
    # Backend/transport failures are not the model's fault: an unloadable model
    # fails every problem identically, so scoring it would publish a fake 0%
    # for a model that never generated a token.
    if re.search(r"Ollama HTTP \d|error loading model|unknown model architecture"
                 r"|Internal Server Error|Connection refused|URL Error", hay):
        return "infra"
    if re.search(r"SyntaxError|IndentationError", hay):
        return "syntax"
    if re.search(r"NameError|TypeError|ValueError|KeyError|IndexError|AttributeError|Error:", hay):
        return "runtime"
    if result.get("error") and "no Python function named fn" in str(result["error"]):
        return "extract"
    if result.get("exit_code") not in (None, 0):
        return "py-exit"
    return "wrong-answer"


# ---------------------------------------------------------------------------
# backends

def direct_fixture(problem, cfg, use_all_tests):
    started = time.time()
    tests = problem_tests(problem)
    # Strict: a missing fixture used to silently fall back to the JWT fixture,
    # which would grade the wrong reference implementation against these tests.
    code = FIXTURES.get(problem["id"])
    if not code:
        raise SystemExit("No fixture for %s — direct mode cannot validate it." % problem["id"])
    # level=large validates EVERY expected value, the same way a model run is
    # graded; without it only the first test is checked (and a wrong expected
    # value in a later test would go unnoticed until a model failed it).
    tests = tests if use_all_tests else [tests[0]]
    source = build_source(code, tests)
    grade = run_python_script(source, cfg["TMB_PYTHON"], int(problem["timeout_s"]) * 1000)
    grade["pass"] = grade_tests([t["expected"] for t in tests], grade["stdout"])
    grade["generated_code"] = source
    grade["backend"] = "direct-fixture"
    grade["model"] = None
    grade["problem"] = problem["id"]
    grade["elapsed_ms"] = int((time.time() - started) * 1000)
    grade["response"] = "(deterministic fixture; validates extraction/grading path)"
    return grade


def ollama_chat(model, messages, cfg):
    url = normalize_url(cfg.get("OLLAMA_URL", DEFAULTS["OLLAMA_URL"])) + "/api/chat"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": as_bool(cfg.get("TMB_THINK")),
        "options": {"temperature": 0, "num_predict": as_int(cfg.get("TMB_NUM_PREDICT"), 256)},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=None) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip()
        # Surface the server's own message. A model the runner cannot load
        # reports it here (e.g. "unknown model architecture: 'k2-horizon'"),
        # and classify_failure turns that into an `infra` failure so it is
        # never scored as a wrong answer. `reason` alone says only
        # "Internal Server Error", which is not enough to tell infra from code.
        raise RuntimeError("Ollama HTTP %s: %s" % (e.code, detail[:400])) from e


def ollama(problem, cfg, use_all_tests):
    """One shared per-problem budget (signal alarm) across self-repair attempts."""
    started = time.time()
    timeout_s = as_int(cfg.get("TMB_TIMEOUT_S"), 30)
    model = cfg["TMB_MODEL"]
    response_text = ""

    def handler(signum, frame):
        raise TimeoutError("Ollama timeout after %ds" % timeout_s)

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_s)
    try:
        if as_bool(cfg.get("TMB_WARMUP")):
            print("warmup: greeting the model...")
            try:
                ollama_chat(model, [{"role": "user", "content": "Hello, let's begin."}],
                            {**cfg, "TMB_NUM_PREDICT": "32"})
                print("warmup: done.")
            except Exception as e:
                return error_result(model, problem["id"], started, response_text, "warmup failed: %s" % e)

        all_tests = problem_tests(problem)
        tests = all_tests if use_all_tests else [all_tests[0]]
        max_attempts = as_int(cfg.get("TMB_ATTEMPTS"), 3)
        messages = [{"role": "user", "content": model_prompt(problem)}]
        code = ""
        grade = None
        attempts_used = 0

        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            data = ollama_chat(model, messages, cfg)
            response_text = data.get("message", {}).get("content") or data.get("response") or ""
            code = extract_code(response_text)

            if not code:
                if attempt < max_attempts:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content":
                        "Your response did not contain a Python function named fn. "
                        "Return ONLY a code block defining function fn."})
                    continue
                return error_result(model, problem["id"], started, response_text,
                                    "no Python function named fn extracted")

            source = build_source(code, tests)
            grade = run_python_script(source, cfg["TMB_PYTHON"], int(problem["timeout_s"]) * 1000)
            grade["pass"] = grade_tests([t["expected"] for t in tests], grade["stdout"])

            if grade["pass"] or attempt >= max_attempts:
                break

            feedback = (
                "Your code did not pass all tests.\n"
                "stdout: %s%s%s\n"
                "Fix the fn function and return ONLY the corrected code block."
                % (fmt_stdout(grade.get("stdout")) or "(empty)",
                   "\nerror: %s" % grade["error"] if grade.get("error") else "",
                   "\nstderr: %s" % grade["stderr"][:300] if grade.get("stderr") else ""))
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": feedback})

        grade["generated_code"] = code
        grade["response"] = response_text[-4000:]
        grade["backend"] = "ollama"
        grade["model"] = model
        grade["problem"] = problem["id"]
        grade["attempts"] = attempts_used
        # run_python_script's elapsed_ms only measures the grading step;
        # report the whole per-problem wall time (generation + grading).
        grade["elapsed_ms"] = int((time.time() - started) * 1000)
        return grade
    except TimeoutError as e:
        return error_result(model, problem["id"], started, response_text, str(e))
    except Exception as e:
        return error_result(model, problem["id"], started, response_text,
                            str(getattr(e, "reason", e))[-1000:])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def error_result(model, problem_id, started, response_text, message):
    return {
        "backend": "ollama",
        "model": model,
        "pass": False,
        "elapsed_ms": int((time.time() - started) * 1000),
        "response": response_text[-4000:],
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "error": str(message)[-1000:],
        "problem": problem_id,
    }


def model_prompt(problem):
    return "\n".join([
        "You are completing one tiny coding benchmark task.",
        "Return ONLY a Python code block containing a function named fn.",
        "Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.",
        "Use only the Python standard library (no pip packages).",
        "",
        problem["prompt"],
    ])


def fmt_stdout(text):
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return (lines[-1] if lines else text)[:120]


def status_line(result):
    # Same icons, spacing and error marker as statusLine() in
    # tiny_benchmark.js, so a js and a py run read as one log.
    icon = "✅" if result.get("pass") else "❌"
    line = "%s %s %s" % (icon, result.get("problem"), result.get("backend"))
    if result.get("model"):
        line += "  %s" % result["model"]
    if result.get("elapsed_ms") is not None:
        line += "  %dms" % result["elapsed_ms"]
    if result.get("exit_code") not in (None, 0):
        line += "  exit %s" % result["exit_code"]
    print(line)
    if result.get("error"):
        print("   ⚠ %s" % compact_error(result))


def print_summary_table(results, level):
    """Short end-of-run report: one row per harness+model.

    Mirrors printSummaryTable() in tiny_benchmark.js — keep the two in step,
    since the whole point of the table is to read a js run and a py run side by
    side without scrolling the per-check lines.
    """
    order, groups = [], {}
    for r in results:
        key = (r.get("backend"), r.get("model") or "—")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    if not order:
        return

    headers = ["Harness", "Model", "Checks", "Pass", "Rate", "Timeouts", "Top failures"]
    rows = []
    for key in order:
        rs = groups[key]
        total = len(rs)
        passed = sum(1 for r in rs if r.get("pass"))
        timeouts = sum(1 for r in rs if r.get("failure_type") == "timeout")
        counts = {}
        for r in rs:
            if r.get("pass") or r.get("failure_type") == "timeout":
                continue
            t = r.get("failure_type") or "unknown"
            counts[t] = counts.get(t, 0) + 1
        top = ", ".join("%s %d" % kv
                         for kv in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        rows.append([key[0], key[1], str(total), str(passed),
                     "%.1f%%" % (100.0 * passed / (total or 1)), str(timeouts), top or "—"])

    widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    def line(cells):
        return "│ " + " │ ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " │"

    print("TinyMark report — py harness · level %s · %d checks" % (level, len(results)))
    print("┌" + "┬".join("─" * (w + 2) for w in widths) + "┐")
    print(line(headers))
    print("├" + "┼".join("─" * (w + 2) for w in widths) + "┤")
    for row in rows:
        print(line(row))
    print("└" + "┴".join("─" * (w + 2) for w in widths) + "┘")


# ---------------------------------------------------------------------------
# main

def main():
    # Line-buffer stdout so per-check lines stream when the harness is piped or
    # tee'd (a few hours ago a killed batch looked "stuck" because Python only
    # block-flushes to a pipe when the process exits).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    cfg = load_config()

    level = str(cfg.get("TMB_LEVEL") or "med").lower()
    use_all_tests = level == "large"
    test_sel = cfg.get("TMB_TESTS") or cfg.get("TMB_PROBLEMS")
    problem_ids = ([s.strip() for s in str(test_sel).split(",") if s.strip()]
                   if test_sel else LEVELS.get(level, LEVELS["med"]))

    if not problem_ids:
        raise SystemExit("No problems to run.")
    for pid in problem_ids:
        if pid not in ALL_PROBLEMS:
            raise SystemExit("Unknown problem %s. Known: %s" % (pid, ", ".join(ALL_PROBLEMS.keys())))

    model_ids = ([s.strip() for s in str(cfg.get("TMB_MODELS")).split(",") if s.strip()]
                 if cfg.get("TMB_MODELS") else [cfg.get("TMB_MODEL")])
    if not model_ids:
        raise SystemExit("No models to run.")

    mode = str(cfg.get("TMB_MODE")).lower()
    modes = ["direct", "ollama"] if mode == "both" else [mode]

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runs = as_int(cfg.get("TMB_RUNS"), 3)
    results = []

    print("problems: %s" % ", ".join(problem_ids))
    print("models:   %s" % ", ".join(model_ids))
    print("mode:     %s  level: %s  runs: %d  timeout_s: %s"
          % (mode, level, runs, cfg.get("TMB_TIMEOUT_S")))
    print("")

    for pid in problem_ids:
        problem = ALL_PROBLEMS[pid]
        for model in model_ids:
            model_cfg = dict(cfg)
            model_cfg["TMB_MODEL"] = model
            for m in modes:
                for run_index in range(1, runs + 1):
                    if m == "direct":
                        result = direct_fixture(problem, model_cfg, use_all_tests)
                    elif m == "ollama":
                        result = ollama(problem, model_cfg, use_all_tests)
                    else:
                        result = {"backend": m, "pass": False, "elapsed_ms": 0,
                                  "error": "unknown mode %s" % m, "problem": pid, "model": model}
                    result["run"] = run_index
                    result["problem"] = pid
                    result["model"] = model
                    result["failure_type"] = classify_failure(result)
                    result["error_compact"] = compact_error(result)
                    results.append(result)
                    status_line(result)

    print("")
    passed = sum(1 for r in results if r.get("pass"))
    print("passed: %d/%d" % (passed, len(results)))
    failures = [r for r in results if not r.get("pass")]
    if failures:
        print("failures:")
        for r in failures:
            print("  %-22s %-14s %s" % (r.get("problem"), r.get("failure_type"), r.get("error_compact")))
    print_summary_table(results, level)

    output = {
        "schema": "tiny-py-benchmark/v0",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problems": [{"id": pid, "title": ALL_PROBLEMS[pid]["title"],
                      "summary": ALL_PROBLEMS[pid]["summary"],
                      "difficulty": ALL_PROBLEMS[pid]["difficulty"]}
                     for pid in problem_ids],
        "models": model_ids,
        "config": {
            "mode": cfg.get("TMB_MODE"),
            "models": model_ids,
            "ollama_url": normalize_url(cfg.get("OLLAMA_URL", DEFAULTS["OLLAMA_URL"])),
            "warmup": as_bool(cfg.get("TMB_WARMUP")),
            "timeout_s": max(ALL_PROBLEMS[pid]["timeout_s"] for pid in problem_ids),
            "num_predict": as_int(cfg.get("TMB_NUM_PREDICT"), 256),
            "think": as_bool(cfg.get("TMB_THINK")),
            "level": level,
            "attempts": as_int(cfg.get("TMB_ATTEMPTS"), 3),
        },
        "results": results,
    }
    output_path = os.path.join(ROOT, cfg.get("TMB_RESULTS", DEFAULTS["TMB_RESULTS"]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(output, indent=2) + "\n")
    print("results: %s" % output_path)


if __name__ == "__main__":
    main()