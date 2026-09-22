#!/usr/bin/env node
/**
 * Tiny JS Benchmark: one problem, one chat turn, one executable grader.
 *
 * Usage:
 *   node scripts/tiny_benchmark.js                 # default problem + Ollama
 *   node scripts/tiny_benchmark.js --mode=ollama
 *   node scripts/tiny_benchmark.js --mode=direct
 *   node scripts/tiny_benchmark.js --problem=tmb-jwt --model=qwen3.5:4b
 *
 * Configuration precedence: CLI flag > process environment > .env >
 * tmb_env.json (legacy local config) > defaults. No packages required.
 */

import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const ROOT = join(import.meta.dirname, "..");

const DEFAULTS = {
  TMB_MODE: "ollama",
  TMB_RUNS: "3",
  TMB_MODELS: "qwen3.5:9b,granite4.2:3b",
  OLLAMA_URL: "http://localhost:11434",
  TMB_TIMEOUT_S: "30",  // interactive-chat bar: strong models answer in 3-20s; 30s total per problem across all self-repair attempts
  TMB_NUM_PREDICT: "256",
  TMB_ATTEMPTS: "3",  // self-repair: max code-generation attempts per problem
  TMB_THINK: "false",
  TMB_NODE: "node",
  TMB_RESULTS: "results/tmb_result.json",
  TMB_DEBUG: "false",
  TMB_WARMUP: "false",
};

function parseEnv(text) {
  const values = {};
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const match = trimmed.match(/^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!match) continue;
    let value = match[2].trim();
    if ((value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    } else {
      value = value.replace(/\s+#.*$/, "");
    }
    values[match[1]] = value;
  }
  return values;
}

async function loadLocalConfig() {
  let fileValues = {};
  try {
    fileValues = parseEnv(await readFile(join(ROOT, ".env"), "utf8"));
  } catch { }
  // Keep the prior tmb_env.json experiment useful, without requiring it.
  if (Object.keys(fileValues).length === 0) {
    try {
      fileValues = JSON.parse(await readFile(join(ROOT, "tmb_env.json"), "utf8"));
    } catch { }
  }
  return { ...DEFAULTS, ...fileValues, ...process.env, ...parseArgs(process.argv.slice(2)) };
}

function parseArgs(args) {
  const values = {};
  let pending = null;
  for (const arg of args) {
    if (pending) {
      values["TMB_" + pending] = arg;
      pending = null;
      continue;
    }
    const matchEq = arg.match(/^--([^=]+)=(.*)$/);
    if (matchEq) {
      const key = matchEq[1].replaceAll("-", "_").toUpperCase();
      values["TMB_" + key] = matchEq[2];
      continue;
    }
    const matchFlag = arg.match(/^--([^=]+)$/);
    if (matchFlag) {
      pending = matchFlag[1].replaceAll("-", "_").toUpperCase();
    }
  }
  return values;
}

function bool(value) {
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function int(value, fallback) {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : (fallback != null ? fallback : 1);
}

function normalizeUrl(value) {
  return String(value).replace(/\/api\/?$/, "").replace(/\/$/, "");
}

function stripAnsi(text) {
  return String(text || "").replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
}

function cleanTempPaths(text) {
  return String(text || "")
    .replace(/\/(?:private\/)?var\/folders\/[\w\/-]+\/tiny-js-benchmark-[\w-]+\/candidate\.js(?::\d+(?::\d+)?)?/g, "candidate.js")
    .replace(/\/tmp\/tiny-js-benchmark-[\w-]+\/candidate\.js(?::\d+(?::\d+)?)?/g, "candidate.js");
}

function pickErrorHeadline(stderr) {
  const lines = cleanTempPaths(stderr)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return "";

  const priority = /(SyntaxError|ReferenceError|TypeError|RangeError|URIError|EvalError|Error:|Unexpected)/i;
  const hit = lines.find((line) => priority.test(line));
  if (hit) return hit;

  // Skip bare path/location lines when possible.
  const notPath = lines.find((line) => !/^candidate\.js(?::\d+(?::\d+)?)?$/.test(line));
  return notPath || lines[0];
}

function debugErrorSnippet(stderr, maxLen = 220) {
  const lines = cleanTempPaths(stderr)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length) return "";

  const priority = /(SyntaxError|ReferenceError|TypeError|RangeError|URIError|EvalError|Error:|Unexpected)/i;
  const idx = lines.findIndex((line) => priority.test(line));
  const start = idx >= 0 ? idx : 0;
  return lines.slice(start, start + 2).join(" | ").slice(0, maxLen);
}

function stripThinking(text) {
  // Reasoning/thinking models (DeepSeek-R1, Qwen3, OpenThinker, Phi-4, ...)
  // wrap their chain-of-thought in model-specific tags. A real chat client
  // hides that trace and shows only the final answer, so strip it before we
  // look for the code block. Handles the common wrappers:
  //   <thinking>...</thinking>   (OpenThoughts / OpenThinker)
  //    thinking... response         (DeepSeek-R1)
  //   .../...        (Qwen3)
  //   [thinking]...[/thinking]   (some distills)
  return String(text || "")
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
    .replace(/ thinking[\s\S]*?<\/think>/gi, "")
    .replace(/\[thinking\][\s\S]*?\[\/thinking\]/gi, "")
    .replace(/<\/?think>/gi, "")
    .trim();
}

// Does this snippet actually *define* fn? Models overwhelmingly bind it with a
// const arrow (`const fn = (s) => ...`) rather than a declaration, and both are
// "a function named fn" as far as the prompt is concerned. Requiring `fn(`
// rejected the arrow form outright (and, worse, accepted a bare call `fn(x)`).
const FN_DECL_RE = /\b(?:async\s+)?function\s+fn\s*\(/;
const FN_BIND_RE = /\bfn\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[\w$]+\s*=>)/;
function definesFn(text) {
  return FN_DECL_RE.test(text) || FN_BIND_RE.test(text);
}

// Same test, anchored at the start of a line, so the no-code-fence fallback can
// slice from the `const`/`function` keyword instead of the bare `fn`.
const FN_DEF_LINE_RE = /^[ \t]*(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\s+fn\b|(?:const|let|var)\s+fn\s*=|fn\s*=)/m;

function extractCode(response) {
  let text = stripThinking(stripAnsi(response));
  const blocks = [...text.matchAll(/```(?:js|javascript|node)?\s*\n?([\s\S]*?)```/gi)]
    .map((match) => match[1].trim())
    .filter((block) => definesFn(block));
  if (blocks.length) return blocks[0];

  const decl = FN_DEF_LINE_RE.exec(text);
  if (decl) {
    const prefix = text.slice(0, decl.index).split("\n")
      .filter((line) => /^\s*(import|export|const|let|var|function)\s+/.test(line));
    return [...prefix, text.slice(decl.index)].join("\n").trim();
  }
  return "";
}

/*
 * One real JS problem for now. We're testing the harness, not curating a set.
 *
 * The task: given a JWT whose payload is base64url-encoded JSON, decode and
 * return the payload. The expected answer is the decoded payload object; the
 * harness checks that the expected payload appears in the script's stdout
 * (as JSON text or as a substring), not that the script prints a specific
 * format.
 */

const PROBLEMS = {
  "tmb-jwt": {
    id: "tmb-jwt",
    title: "decode JWT payload",
    summary: "base64url decode, JSON parse, no external packages",
    difficulty: "simple",
    timeout_s: 30,
    // A compact JWT whose payload we control. Not a real signature.
    // Header: {"alg":"HS256","typ":"JWT"}. Payload: {sub:"user",tag:"hello",iat:1499999999}.
    token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwidGFnIjoiaGVsbG8iLCJpYXQiOjE0OTk5OTk5OTl9.dummy-signature",    tests: [ { input:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIiwidGFnIjoiaGVsbG8iLCJpYXQiOjE0OTk5OTk5OTl9.dummy", expected:{"sub":"user","tag":"hello","iat":1499999999} }, { input:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhIjoxfQ.dummy", expected:{"a":1} }, { input:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiY2FmXHUwMGU5IiwibiI6MCwib2siOnRydWV9.dummy", expected:{"name":"café","n":0,"ok":true} }, { input:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcnIiOlsxLDIsM119.dummy", expected:{"arr":[1,2,3]} }, { input:"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGljZSIsInJvbGUiOiJhZG1pbiJ9.dummy", expected:{"sub":"alice","role":"admin"} } ],

    prompt: `Implement a JavaScript function named fn.

Task: given a JWT string, decode and return the decoded payload.

The JWT format is three base64url-encoded segments separated by '.': header.payload.signature.
Decode the second segment (payload) from base64url to UTF-8 text, then parse it as JSON.
Return the resulting object.

Do NOT use any external npm packages. Use only built-in Node.js (Buffer is allowed).

Example:
  fn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyfSwidGFnIjoiaGVsbG8iLCJpYXQiOjE0OTk5OTk5OTl9.used-for-harness-only")
  should return the object { sub: "user", tag: "hello", iat: 1499999999 }.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the decoded payload with console.log(JSON.stringify(fn(...))) when run.`,
    // The expected decoded payload. The harness checks that this appears in
    // stdout (as JSON text or a substring), not that stdout matches exactly.
    expected: {
      sub: "user",
      tag: "hello",
      iat: 1499999999,
    },
  },

  "tmb-slug": {
    id: "tmb-slug",
    title: "slugify string",
    summary: "unicode lowercasing, punctuation removal, whitespace collapse",
    difficulty: "simple",
    timeout_s: 20,
    input: "Hello, World! 123",
    testInput: "Hello, World! 123",
    expected: "hello-world-123",    tests: [ { input:"Hello, World! 123", expected:"hello-world-123" }, { input:"  Foo  Bar  ", expected:"foo-bar" }, { input:"UPPER lower", expected:"upper-lower" }, { input:"a--b---c", expected:"a-b-c" }, { input:"123", expected:"123" } ],

    prompt: `Implement a JavaScript function named fn.

Task: convert a string into a URL-friendly slug.

Rules:
- Lowercase the string.
- Replace any sequence of non-alphanumeric characters with a single hyphen.
- Strip leading and trailing hyphens.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("Hello, World! 123") should return "hello-world-123".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the slug with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-kv-parse": {
    id: "tmb-kv-parse",
    title: "parse k=v string",
    summary: "split on comma, split on first equals, trim whitespace",
    difficulty: "simple",
    timeout_s: 20,
    testInput: "a=1,b=2,c=3",
    expected: { a: "1", b: "2", c: "3" },    tests: [ { input:"a=1,b=2,c=3", expected:{"a":"1","b":"2","c":"3"} }, { input:"x=10", expected:{"x":"10"} }, { input:" key = value , a = 1 ", expected:{"key":"value","a":"1"} }, { input:"a=1=2,b=3", expected:{"a":"1=2","b":"3"} }, { input:"", expected:{} } ],

    prompt: `Implement a JavaScript function named fn.

Task: parse a compact key=value string into an object.

Rules:
- The input is a string of key=value pairs separated by commas: "k1=v1,k2=v2,...".
- Split on commas, then split each pair on the FIRST '=' only.
- Trim whitespace around keys and values.
- Return an object mapping each key to its value as a string.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("a=1,b=2,c=3") should return { a: "1", b: "2", c: "3" }.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result object with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-fix-index-delim": {
    id: "tmb-fix-index-delim",
    title: "fix buggy parser",
    summary: "off-by-one, wrong delimiter, wrong split, wrong comparison — fix fn",
    difficulty: "simple",
    timeout_s: 25,
    testInput: "a=1,b=2,c=3,d=4",
    expected: { a: "1", b: "2", c: "3", d: "4" },    tests: [ { input:"a=1,b=2,c=3,d=4", expected:{"a":"1","b":"2","c":"3","d":"4"} }, { input:"x=5", expected:{"x":"5"} }, { input:" a = 1 , b = 2 ", expected:{"a":"1","b":"2"} }, { input:"a=1=2", expected:{"a":"1=2"} }, { input:"", expected:{} } ],

    prompt: `The following JavaScript function is buggy. Fix it so it parses the input correctly and returns the expected object.

Rules:
- Return ONLY a JavaScript code block containing the corrected fn function.
- Do not explain your answer.
- Use only Node.js built-ins (no npm packages).
- The fixed function must keep the same signature: fn(s) returns an object.

Buggy function:

function fn(s) {
  const pairs = s.split("|");
  const obj = {};
  for (let i = 0; i <= pairs.length; i++) {
    const pair = pairs[i];
    if (!pair) continue;
    const parts = pair.split(":");
    const k = parts[0].trim();
    const v = parts[1].trim();
    if (k == v) continue;
    obj[k] = v;
  }
  return obj;
}

Expected behavior for fn("a=1,b=2,c=3,d=4"):
- Split the input on commas.
- For each piece, split on the FIRST '=' only.
- Trim whitespace around key and value.
- Build an object mapping each key to its value as a string.
- Return the object.

Expected result: { a: "1", b: "2", c: "3", d: "4" }

Print the result object with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-merge-ranges": {
    id: "tmb-merge-ranges",
    title: "merge overlapping ranges",
    summary: "sort by start, merge overlaps, keep output deterministic",
    difficulty: "medium",
    timeout_s: 25,
    testInput: [[1, 3], [2, 6], [8, 10], [15, 18], [17, 20]],
    expected: [[1, 6], [8, 10], [15, 20]],    tests: [ { input:[[1,3],[2,6],[8,10],[15,18],[17,20]], expected:[[1,6],[8,10],[15,20]] }, { input:[[1,4],[4,5]], expected:[[1,5]] }, { input:[[1,2],[3,4]], expected:[[1,2],[3,4]] }, { input:[[5,5]], expected:[[5,5]] }, { input:[], expected:[] } ],

    prompt: `Implement a JavaScript function named fn.

Task: merge overlapping numeric ranges.

Input:
- An array of 2-item arrays, where each item is [start, end].
- start and end are integers with start <= end.

Rules:
- Sort ranges by start ascending.
- Merge ranges when they overlap (next.start <= current.end).
- Return merged ranges as an array of [start, end] pairs.
- Keep output order ascending by start.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([[1,3],[2,6],[8,10],[15,18],[17,20]])
  should return [[1,6],[8,10],[15,20]].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the merged ranges with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-rpn-eval": {
    id: "tmb-rpn-eval",
    title: "evaluate reverse polish notation",
    summary: "stack-based expression evaluation with integer truncation",
    difficulty: "hard",
    timeout_s: 30,
    testInput: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"],
    expected: 22,    tests: [ { input:["10","6","9","3","+","-11","*","/","*","17","+","5","+"], expected:22 }, { input:["2","1","+","3","*"], expected:9 }, { input:["4","13","5","/","+"], expected:6 }, { input:["3","4","+","2","*","1","-"], expected:13 }, { input:["5"], expected:5 } ],

    prompt: `Implement a JavaScript function named fn.

Task: evaluate an arithmetic expression in Reverse Polish Notation (RPN).

Input:
- An array of string tokens.
- Each token is either an integer (possibly negative) or one of: "+", "-", "*", "/".

Rules:
- Use stack evaluation.
- Division truncates toward zero.
- Assume input is valid and contains no divide-by-zero.
- Return the final integer result.
- Use only Node.js built-ins (no npm packages).

Example:
  fn(["2","1","+","3","*"]) should return 9.
  fn(["4","13","5","/","+"]) should return 6.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the numeric result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-lru-ops": {
    id: "tmb-lru-ops",
    title: "simulate LRU cache operations",
    summary: "ordered map updates, evictions, and deterministic get traces",
    difficulty: "hard",
    timeout_s: 35,
    testInput: {
      capacity: 2,
      ops: [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3], ["get", 2], ["put", 4, 4], ["get", 1], ["get", 3], ["get", 4]],
    },
    expected: [1, -1, -1, 3, 4],    tests: [ { input:{"capacity":2,"ops":[["put",1,1],["put",2,2],["get",1],["put",3,3],["get",2],["put",4,4],["get",1],["get",3],["get",4]]}, expected:[1,-1,-1,3,4] }, { input:{"capacity":1,"ops":[["put",1,1],["put",2,2],["get",1],["get",2]]}, expected:[-1,2] }, { input:{"capacity":2,"ops":[["get",1]]}, expected:[-1] }, { input:{"capacity":3,"ops":[["put",1,1],["put",2,2],["put",3,3],["get",1],["put",4,4],["get",2]]}, expected:[1,-1] }, { input:{"capacity":2,"ops":[["put",1,1],["get",1],["put",1,2],["get",1]]}, expected:[1,2] } ],

    prompt: `Implement a JavaScript function named fn.

Task: simulate an LRU (Least Recently Used) cache and return outputs for get operations.

Input:
- An object: { capacity: number, ops: Array }
- Each op is one of:
  - ["put", key, value]
  - ["get", key]

Rules:
- put inserts/updates key with value.
- get returns value if key exists, otherwise -1.
- Accessing/updating a key marks it as most recently used.
- If put exceeds capacity, evict the least recently used key.
- Return an array containing results of each get operation in order.
- Use only Node.js built-ins (no npm packages).

Example:
  fn({ capacity: 2, ops: [["put",1,1],["put",2,2],["get",1],["put",3,3],["get",2]] })
  should return [1,-1].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the get-results array with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-reverse-words": {
    id: "tmb-reverse-words",
    title: "reverse word order",
    summary: "split on whitespace, reverse array, rejoin with single spaces",
    difficulty: "simple",
    timeout_s: 20,
    testInput: "the sky is blue",
    expected: "blue is sky the",    tests: [ { input:"the sky is blue", expected:"blue is sky the" }, { input:"a b c", expected:"c b a" }, { input:"  hello   world  ", expected:"world hello" }, { input:"single", expected:"single" }, { input:"1 2 3", expected:"3 2 1" } ],

    prompt: `Implement a JavaScript function named fn.

Task: reverse the order of the words in a string.

Rules:
- Words are separated by one or more whitespace characters.
- Return the words in reverse order, joined by a single space each.
- Ignore leading/trailing whitespace.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("the sky is blue") should return "blue is sky the".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result string with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-fizzbuzz": {
    id: "tmb-fizzbuzz",
    title: "fizzbuzz",
    summary: "modulo classification, string building, array output",
    difficulty: "simple",
    timeout_s: 20,
    testInput: 15,
    expected: ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"],    tests: [ { input:15, expected:["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"] }, { input:1, expected:["1"] }, { input:3, expected:["1","2","Fizz"] }, { input:5, expected:["1","2","Fizz","4","Buzz"] }, { input:16, expected:["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz","16"] } ],

    prompt: `Implement a JavaScript function named fn.

Task: classic FizzBuzz.

Rules:
- fn(n) returns an array of strings for the numbers 1 through n inclusive.
- For multiples of 3, push "Fizz".
- For multiples of 5, push "Buzz".
- For multiples of both 3 and 5, push "FizzBuzz".
- Otherwise push the number as a string.
- Use only Node.js built-ins (no npm packages).

Example:
  fn(15) should return ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result array with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-palindrome": {
    id: "tmb-palindrome",
    title: "valid palindrome",
    summary: "normalize case, strip non-alphanumerics, compare reversed",
    difficulty: "simple",
    timeout_s: 20,
    testInput: "A man, a plan, a canal: Panama",
    expected: true,    tests: [ { input:"A man, a plan, a canal: Panama", expected:true }, { input:"race a car", expected:false }, { input:"abba", expected:true }, { input:"", expected:true }, { input:"0P", expected:false } ],

    prompt: `Implement a JavaScript function named fn.

Task: determine if a string is a palindrome.

Rules:
- A string is a palindrome if it reads the same forward and backward after:
  - converting all letters to lowercase, and
  - removing all non-alphanumeric characters (spaces, punctuation, etc.).
- Return true or false.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("A man, a plan, a canal: Panama") should return true.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the boolean result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-valid-parentheses": {
    id: "tmb-valid-parentheses",
    title: "valid parentheses",
    summary: "stack matching of (), [], {} with correct nesting",
    difficulty: "medium",
    timeout_s: 25,
    testInput: "([)]",
    expected: false,    tests: [ { input:"([)]", expected:false }, { input:"()[]{}", expected:true }, { input:"{[]}", expected:true }, { input:"(", expected:false }, { input:"", expected:true } ],

    prompt: `Implement a JavaScript function named fn.

Task: determine if an input string has valid, correctly nested parentheses.

Rules:
- The string contains only the characters ( ) [ ] { }.
- Open brackets must be closed by the same type of bracket.
- Open brackets must be closed in the correct order (proper nesting).
- Return true if valid, false otherwise.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn("()[]{}") should return true.
  fn("([)]") should return false (crossed, not nested).
  fn("{[]}") should return true.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the boolean result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-roman-to-int": {
    id: "tmb-roman-to-int",
    title: "roman numeral to integer",
    summary: "symbol value mapping, subtractive rule handling",
    difficulty: "medium",
    timeout_s: 25,
    testInput: "MCMXCIV",
    expected: 1994,    tests: [ { input:"MCMXCIV", expected:1994 }, { input:"III", expected:3 }, { input:"LVIII", expected:58 }, { input:"IV", expected:4 }, { input:"M", expected:1000 } ],

    prompt: `Implement a JavaScript function named fn.

Task: convert a Roman numeral string to an integer.

Rules:
- Symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000.
- A smaller symbol before a larger one is subtracted (e.g. IV=4, IX=9, CM=900).
- Otherwise symbols are added.
- Input is a valid Roman numeral.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("MCMXCIV") should return 1994.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-int-to-roman": {
    id: "tmb-int-to-roman",
    title: "integer to roman numeral",
    summary: "greedy symbol table, subtractive forms, string building",
    difficulty: "medium",
    timeout_s: 25,
    testInput: 1994,
    expected: "MCMXCIV",    tests: [ { input:1994, expected:"MCMXCIV" }, { input:3, expected:"III" }, { input:58, expected:"LVIII" }, { input:4, expected:"IV" }, { input:1000, expected:"M" } ],

    prompt: `Implement a JavaScript function named fn.

Task: convert an integer to a Roman numeral string.

Rules:
- Symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000.
- Use subtractive forms where appropriate: 4=IV, 9=IX, 40=XL, 90=XC, 400=CD, 900=CM.
- Input is a positive integer.
- Use only Node.js built-ins (no npm packages).

Example:
  fn(1994) should return "MCMXCIV".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result string with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-first-unique-char": {
    id: "tmb-first-unique-char",
    title: "first unique character",
    summary: "frequency count, then scan for first occurrence count of 1",
    difficulty: "medium",
    timeout_s: 25,
    testInput: "loveleetcode",
    expected: 2,    tests: [ { input:"loveleetcode", expected:2 }, { input:"leetcode", expected:0 }, { input:"aabb", expected:-1 }, { input:"a", expected:0 }, { input:"", expected:-1 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the index of the first non-repeating character in a string.

Rules:
- Count occurrences of each character.
- Return the index of the first character that appears exactly once.
- Return -1 if every character repeats.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("loveleetcode") should return 2 (the 'v' is the first unique character).

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-longest-substr": {
    id: "tmb-longest-substr",
    title: "longest substring without repeating chars",
    summary: "sliding window with last-seen index map",
    difficulty: "hard",
    timeout_s: 30,
    testInput: "abcabcbb",
    expected: 3,    tests: [ { input:"abcabcbb", expected:3 }, { input:"bbbbb", expected:1 }, { input:"pwwkew", expected:3 }, { input:"", expected:0 }, { input:"au", expected:2 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the length of the longest substring without repeating characters.

Rules:
- A substring is a contiguous sequence of characters.
- No character may repeat within the substring.
- Return the maximum length.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("abcabcbb") should return 3 (the substring "abc").

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-max-subarray": {
    id: "tmb-max-subarray",
    title: "maximum subarray sum",
    summary: "Kadane's algorithm, contiguous subarray maximum",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [-2, 1, -3, 4, -1, 2, 1, -5, 4],
    expected: 6,    tests: [ { input:[-2,1,-3,4,-1,2,1,-5,4], expected:6 }, { input:[1], expected:1 }, { input:[-1], expected:-1 }, { input:[5,4,-1,7,8], expected:23 }, { input:[-2,-1], expected:-1 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the sum of the contiguous subarray with the largest sum.

Rules:
- Input is an array of integers (may include negatives).
- Find the contiguous subarray (one or more adjacent elements) with the maximum sum.
- Return that maximum sum.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([-2,1,-3,4,-1,2,1,-5,4]) should return 6 (subarray [4,-1,2,1]).

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-trap-rain-water": {
    id: "tmb-trap-rain-water",
    title: "trapping rain water",
    summary: "two-pointer, running max heights, accumulate trapped water",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
    expected: 6,    tests: [ { input:[0,1,0,2,1,0,1,3,2,1,2,1], expected:6 }, { input:[4,2,0,3,2,5], expected:9 }, { input:[1,2,3,4], expected:0 }, { input:[], expected:0 }, { input:[0], expected:0 } ],

    prompt: `Implement a JavaScript function named fn.

Task: compute how much water can be trapped after it rains.

Rules:
- Input is an array of non-negative integers representing elevation heights.
- Each element is a bar of width 1.
- Water can be trapped on top of bars between higher bars on both sides.
- Return the total units of water trapped.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([0,1,0,2,1,0,1,3,2,1,2,1]) should return 6.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-sum-multiples": {
    id: "tmb-sum-multiples",
    title: "sum of multiples",
    summary: "sum all multiples of 3 or 5 below n",
    difficulty: "simple",
    timeout_s: 20,
    testInput: 10,
    expected: 23,    tests: [ { input:10, expected:23 }, { input:20, expected:78 }, { input:1, expected:0 }, { input:0, expected:0 }, { input:15, expected:45 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the sum of all multiples of 3 or 5 below a given number n.

Rules:
- Consider all integers from 1 up to (but not including) n.
- Sum those divisible by 3 or by 5.
- Numbers divisible by both 3 and 5 are counted once.
- Use only Node.js built-ins (no npm packages).

Example:
  fn(10) should return 23 (3 + 5 + 6 + 9).

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-bounding-box": {
    id: "tmb-bounding-box",
    title: "bounding box",
    summary: "min/max x and y over a set of 2D points",
    difficulty: "medium",
    timeout_s: 25,
    testInput: [[1, 2], [3, 4], [0, 5], [-2, 3]],
    expected: [-2, 2, 3, 5],    tests: [ { input:[[1,2],[3,4],[0,5],[-2,3]], expected:[-2,2,3,5] }, { input:[[0,0],[0,0]], expected:[0,0,0,0] }, { input:[[5,5]], expected:[5,5,5,5] }, { input:[[-1,-1],[1,1]], expected:[-1,-1,1,1] }, { input:[[3,1],[3,2],[3,3]], expected:[3,1,3,3] } ],

    prompt: `Implement a JavaScript function named fn.

Task: compute the axis-aligned bounding box of a set of 2D points.

Rules:
- Input is an array of [x, y] point pairs.
- Return [minX, minY, maxX, maxY] covering all points.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([[1,2],[3,4],[0,5],[-2,3]]) should return [-2, 2, 3, 5].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result array with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-polygon-area": {
    id: "tmb-polygon-area",
    title: "polygon area",
    summary: "shoelace formula, absolute area of a simple polygon",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [[0, 0], [4, 0], [4, 3], [0, 3]],
    expected: 12,    tests: [ { input:[[0,0],[4,0],[4,3],[0,3]], expected:12 }, { input:[[0,0],[1,0],[1,1],[0,1]], expected:1 }, { input:[[0,0],[2,0],[1,1]], expected:1 }, { input:[[0,0],[0,0],[0,0]], expected:0 }, { input:[[0,0],[3,0],[3,4],[0,4]], expected:12 } ],

    prompt: `Implement a JavaScript function named fn.

Task: compute the area of a simple polygon given its vertices in order.

Rules:
- Input is an array of [x, y] vertices listed in order around the polygon (clockwise or counter-clockwise).
- Use the shoelace formula.
- Return the absolute area (always non-negative).
- Use only Node.js built-ins (no npm packages).

Example:
  fn([[0,0],[4,0],[4,3],[0,3]]) should return 12 (a 4x3 rectangle).

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the numeric result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-longest-palindromic-substring": {
    id: "tmb-longest-palindromic-substring",
    title: "longest palindromic substring",
    summary: "expand-around-center, longest palindromic substring",
    difficulty: "hard",
    timeout_s: 30,
    testInput: "cbbd",
    expected: "bb",    tests: [ { input:"cbbd", expected:"bb" }, { input:"racecar", expected:"racecar" }, { input:"a", expected:"a" }, { input:"", expected:"" }, { input:"forgeeksskeegfor", expected:"geeksskeeg" } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the longest palindromic substring of a string.

Rules:
- A palindrome reads the same forward and backward.
- If there are multiple longest palindromes, return any one of them.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("cbbd") should return "bb".
  fn("racecar") should return "racecar".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result string with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-num-islands": {
    id: "tmb-num-islands",
    title: "number of islands",
    summary: "grid DFS/BFS flood fill, count connected 1-regions",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]],
    expected: 3,    tests: [ { input:[["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]], expected:3 }, { input:[["1"]], expected:1 }, { input:[["0"]], expected:0 }, { input:[], expected:0 }, { input:[["1","0","1"],["0","1","0"],["1","0","1"]], expected:5 } ],

    prompt: `Implement a JavaScript function named fn.

Task: count the number of islands in a 2D grid.

Rules:
- Input is an array of arrays of strings, each cell "1" (land) or "0" (water).
- An island is a group of "1"s connected horizontally or vertically (not diagonally).
- Return the number of islands.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([["1","1","0"],["1","0","0"],["0","0","1"]]) should return 2.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-lis": {
    id: "tmb-lis",
    title: "longest increasing subsequence",
    summary: "dynamic programming, longest increasing subsequence length",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [10, 9, 2, 5, 3, 7, 101, 18],
    expected: 4,    tests: [ { input:[10,9,2,5,3,7,101,18], expected:4 }, { input:[0,1,0,3,2,3], expected:4 }, { input:[7,7,7,7], expected:1 }, { input:[1], expected:1 }, { input:[], expected:0 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the length of the longest strictly increasing subsequence.

Rules:
- A subsequence preserves relative order but may skip elements.
- Strictly increasing means each element is greater than the previous.
- Return the maximum length.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([10,9,2,5,3,7,101,18]) should return 4 (e.g. [2,3,7,101]).

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-edit-distance": {
    id: "tmb-edit-distance",
    title: "edit distance",
    summary: "Levenshtein distance, dynamic programming on two strings",
    difficulty: "hard",
    timeout_s: 30,
    testInput: ["horse", "ros"],
    expected: 3,    tests: [ { input:["horse","ros"], expected:3 }, { input:["intention","execution"], expected:5 }, { input:["","abc"], expected:3 }, { input:["a","a"], expected:0 }, { input:["abc",""], expected:3 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the Levenshtein edit distance between two strings.

Rules:
- Input is an array of exactly two strings: [a, b].
- Edit distance = minimum number of insertions, deletions, or substitutions to turn a into b.
- Return the distance.
- Use only Node.js built-ins (no npm packages).

Example:
  fn(["horse", "ros"]) should return 3.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-valid-sudoku": {
    id: "tmb-valid-sudoku",
    title: "valid sudoku",
    summary: "validate rows, columns, and 3x3 boxes for duplicates",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]],
    expected: true,    tests: [ { input:[["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]], expected:true }, { input:[["5","5",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]], expected:false }, { input:[["6","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]], expected:false }, { input:[["5","3","9",".","7",".",".",".","."],["6","9",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]], expected:false }, { input:[[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."]], expected:true } ],

    prompt: `Implement a JavaScript function named fn.

Task: determine if a 9x9 Sudoku board is valid.

Rules:
- Input is a 9x9 array of arrays of strings: digits "1"-"9" or "." (empty).
- A board is valid if every row, every column, and every 3x3 sub-box has no repeated digit.
- Empty cells (".") are ignored.
- Return true or false.
- Use only Node.js built-ins (no npm packages).

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the boolean result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-coin-change": {
    id: "tmb-coin-change",
    title: "coin change",
    summary: "dynamic programming, fewest coins to reach amount",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [[1, 2, 5], 11],
    expected: 3,    tests: [ { input:[[1,2,5],11], expected:3 }, { input:[[2],3], expected:-1 }, { input:[[1],0], expected:0 }, { input:[[1,2,5],100], expected:20 }, { input:[[3,5],7], expected:-1 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the fewest number of coins needed to make up a given amount.

Rules:
- Input is an array of exactly two values: [coins, amount].
- coins is an array of coin denominations (positive integers, unlimited supply).
- amount is a non-negative integer.
- Return the minimum number of coins, or -1 if the amount cannot be made.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([[1,2,5], 11]) should return 3 (5+5+1).

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-longest-valid-parentheses": {
    id: "tmb-longest-valid-parentheses",
    title: "longest valid parentheses",
    summary: "stack/DP, length of longest valid parentheses substring",
    difficulty: "hard",
    timeout_s: 30,
    testInput: ")()())",
    expected: 4,    tests: [ { input:")()())", expected:4 }, { input:"(()", expected:2 }, { input:"", expected:0 }, { input:"()", expected:2 }, { input:"((()))", expected:6 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the length of the longest valid (well-formed) parentheses substring.

Rules:
- The string contains only '(' and ')'.
- A substring is valid if parentheses are balanced and correctly nested.
- Return the maximum length.
- Use only Node.js built-ins (no npm packages).

Example:
  fn("(()") should return 2.
  fn(")()())") should return 4.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-median-two-sorted": {
    id: "tmb-median-two-sorted",
    title: "median of two sorted arrays",
    summary: "median of two sorted arrays, O(log(min)) merge",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [[1, 3], [2]],
    expected: 2,    tests: [ { input:[[1,3],[2]], expected:2 }, { input:[[1,2],[3,4]], expected:2.5 }, { input:[[],[1]], expected:1 }, { input:[[0,0],[0,0]], expected:0 }, { input:[[],[2,3]], expected:2.5 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the median of two sorted arrays.

Rules:
- Input is an array of exactly two sorted arrays: [nums1, nums2].
- Return the median value (may be a float, e.g. 2.5).
- Use only Node.js built-ins (no npm packages).

Example:
  fn([[1,3],[2]]) should return 2.
  fn([[1,2],[3,4]]) should return 2.5.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the numeric result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-sliding-window-max": {
    id: "tmb-sliding-window-max",
    title: "sliding window maximum",
    summary: "monotonic deque, max in each sliding window",
    difficulty: "hard",
    timeout_s: 30,
    testInput: [[1, 3, -1, -3, 5, 3, 6, 7], 3],
    expected: [3, 3, 5, 5, 6, 7],    tests: [ { input:[[1,3,-1,-3,5,3,6,7],3], expected:[3,3,5,5,6,7] }, { input:[[1],1], expected:[1] }, { input:[[1,-1],1], expected:[1,-1] }, { input:[[9,11],2], expected:[11] }, { input:[[4,-2],2], expected:[4] } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the maximum value in each sliding window of size k.

Rules:
- Input is an array of exactly two values: [nums, k].
- Slide a window of size k from left to right over nums.
- Return an array of the maximum of each window.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([[1,3,-1,-3,5,3,6,7], 3]) should return [3,3,5,5,6,7].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result array with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-n-queens": {
    id: "tmb-n-queens",
    title: "n-queens count",
    summary: "backtracking, count distinct n-queens solutions",
    difficulty: "hard",
    timeout_s: 30,
    testInput: 8,
    expected: 92,    tests: [ { input:8, expected:92 }, { input:4, expected:2 }, { input:1, expected:1 }, { input:5, expected:10 }, { input:2, expected:0 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the number of distinct solutions to the n-queens puzzle.

Rules:
- Place n queens on an n x n board so no two queens attack each other (no same row, column, or diagonal).
- Input is an integer n.
- Return the number of distinct arrangements.
- Use only Node.js built-ins (no npm packages).

Example:
  fn(4) should return 2.
  fn(8) should return 92.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-atoi": {
    id: "tmb-atoi",
    title: "string to integer (atoi)",
    summary: "whitespace, sign, digits, and 32-bit overflow clamping",
    difficulty: "hard",
    timeout_s: 30,
    tests: [ { input:"   -42", expected:-42 }, { input:"4193 with words", expected:4193 }, { input:"words and 987", expected:0 }, { input:"-91283472332", expected:-2147483648 }, { input:"  +42abc", expected:42 }, { input:"2147483648", expected:2147483647 }, { input:"  -0012a42", expected:-12 }, { input:"   +0 123", expected:0 }, { input:"3.14159", expected:3 }, { input:"", expected:0 } ],

    prompt: `Implement a JavaScript function named fn.

Task: implement the classic atoi — convert a string to a 32-bit signed integer.

Rules:
- fn(s) takes one string.
- Skip leading whitespace.
- An optional '+' or '-' sign may follow the whitespace.
- Read digits until a non-digit is reached or the string ends.
- If no digits are read, return 0.
- Clamp to the 32-bit signed integer range [-2147483648, 2147483647].
- Use only Node.js built-ins (no npm packages).

Examples:
  fn("   -42") should return -42.
  fn("4193 with words") should return 4193.
  fn("words and 987") should return 0.
  fn("-91283472332") should return -2147483648.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-csv-parse": {
    id: "tmb-csv-parse",
    title: "parse quoted CSV",
    summary: "RFC-4180 style: quoted fields, escaped quotes, commas and newlines inside quotes",
    difficulty: "hard",
    timeout_s: 30,
    tests: [ { input:"a,b,c\n1,2,3", expected:[["a","b","c"],["1","2","3"]] }, { input:"\"a,b\",c", expected:[["a,b","c"]] }, { input:"\"he said \"\"hi\"\"\",x", expected:[["he said \"hi\"","x"]] }, { input:"a,\"b\nc\",d", expected:[["a","b\nc","d"]] }, { input:"\"\"", expected:[[""]] }, { input:"a,,c", expected:[["a","","c"]] }, { input:"", expected:[] } ],

    prompt: `Implement a JavaScript function named fn.

Task: parse a CSV string into an array of rows, where each row is an array of field strings.

Rules:
- fn(text) takes one string.
- Rows are separated by newline characters.
- Fields are separated by commas.
- A field may be wrapped in double quotes; inside quotes, a comma or newline is literal.
- Two consecutive double quotes inside a quoted field represent one literal double quote.
- Unquoted fields are trimmed of nothing (keep exact content).
- Return an array of rows; an empty input returns an empty array.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn("a,b,c\n1,2,3") should return [["a","b","c"],["1","2","3"]].
  fn("\"a,b\",c") should return [["a,b","c"]].
  fn("\"he said \"\"hi\"\"\",x") should return [["he said \"hi\"","x"]].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result array with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-basic-calc": {
    id: "tmb-basic-calc",
    title: "basic calculator",
    summary: "evaluate expression with + - * / and parentheses, division truncates toward zero",
    difficulty: "hard",
    timeout_s: 30,
    tests: [ { input:"1 + 2 * 3", expected:7 }, { input:" 3+5 / 2 ", expected:5 }, { input:"2*(3+4)", expected:14 }, { input:"(1+(4+5+2)-3)+(6+8)", expected:23 }, { input:"10/3", expected:3 }, { input:"-3+4", expected:1 }, { input:"-10/3", expected:-3 }, { input:" 2-1 + 2 ", expected:3 }, { input:"14-3*2", expected:8 } ],

    prompt: `Implement a JavaScript function named fn.

Task: evaluate an arithmetic expression string and return the integer result.

Rules:
- fn(expr) takes one string containing non-negative integers, operators + - * /, and parentheses.
- Spaces may appear anywhere and must be ignored.
- Multiplication and division bind tighter than addition and subtraction.
- Division truncates toward zero (Math.trunc), e.g. 10/3 -> 3, -10/3 -> -3.
- Unary minus is allowed (e.g. "-3+4" -> 1).
- Assume input is valid.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn("1 + 2 * 3") should return 7.
  fn(" 3+5 / 2 ") should return 5.
  fn("2*(3+4)") should return 14.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-text-justify": {
    id: "tmb-text-justify",
    title: "text justification",
    summary: "greedy line packing with even space distribution and left-justified last line",
    difficulty: "hard",
    timeout_s: 35,
    tests: [ { input:[["This","is","an","example","of","text","justification."],16], expected:["This    is    an","example  of text","justification.  "] }, { input:[["What","must","be","acknowledgment","shall","be"],16], expected:["What   must   be","acknowledgment  ","shall be        "] }, { input:[["a"],3], expected:["a  "] }, { input:[["a","b","c"],1], expected:["a","b","c"] }, { input:[["hello","world"],10], expected:["hello     ","world     "] } ],

    prompt: `Implement a JavaScript function named fn.

Task: full text justification.

Rules:
- fn(args) receives [words, maxWidth]: an array of words and a line width.
- Pack words greedily: each line holds as many words as fit (at least one space between words).
- Every line except the last must be exactly maxWidth characters long.
- Distribute extra spaces between words as evenly as possible; when they cannot be split evenly, the LEFT gaps get more spaces.
- A single-word line is left-justified and padded with trailing spaces.
- The LAST line is left-justified with single spaces between words, padded with trailing spaces.
- Return an array of line strings.
- Use only Node.js built-ins (no npm packages).

Example:
  fn([["This","is","an","example","of","text","justification."], 16])
  should return ["This    is    an","example  of text","justification.  "].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result array with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-wildcard-match": {
    id: "tmb-wildcard-match",
    title: "wildcard matching",
    summary: "DP matching with '?' for single char and '*' for any sequence",
    difficulty: "hard",
    timeout_s: 30,
    tests: [ { input:["aa","a"], expected:false }, { input:["aa","*"], expected:true }, { input:["cb","?a"], expected:false }, { input:["adceb","*a*b"], expected:true }, { input:["acdcb","a*c?b"], expected:false }, { input:["","*"], expected:true }, { input:["abc","a*c"], expected:true } ],

    prompt: `Implement a JavaScript function named fn.

Task: wildcard pattern matching.

Rules:
- fn(args) receives [s, p]: the string and the pattern.
- '?' in the pattern matches exactly one character.
- '*' in the pattern matches any sequence of characters (including the empty sequence).
- The whole string must match the whole pattern.
- Return true or false.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn(["aa","a"]) should return false.
  fn(["aa","*"]) should return true.
  fn(["adceb","*a*b"]) should return true.
  fn(["acdcb","a*c?b"]) should return false.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the boolean result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-valid-number": {
    id: "tmb-valid-number",
    title: "valid number",
    summary: "validate integer/decimal with optional exponent, strict grammar",
    difficulty: "hard",
    timeout_s: 30,
    tests: [ { input:"0", expected:true }, { input:" 0.1 ", expected:false }, { input:"abc", expected:false }, { input:"2e10", expected:true }, { input:"-90E3", expected:true }, { input:"1e", expected:false }, { input:"e3", expected:false }, { input:"99e2.5", expected:false }, { input:"--6", expected:false }, { input:"-+3", expected:false }, { input:"95a54e53", expected:false }, { input:".", expected:false }, { input:"-.9", expected:true }, { input:"4.", expected:true }, { input:"1.e+", expected:false }, { input:"+.8", expected:true } ],

    prompt: `Implement a JavaScript function named fn.

Task: determine whether a string is a valid number in decimal or scientific notation.

Rules:
- fn(s) takes one string.
- A valid number is an integer or decimal, optionally followed by an exponent: mantissa [e|E] [+|-] exponent-digits.
- The mantissa is: [+|-] followed by either digits, digits with a decimal point, or a decimal point with digits (at least one digit must appear in the mantissa).
- The exponent part requires at least one digit after the optional sign.
- No surrounding whitespace is allowed; no letters other than e/E; no other symbols.
- Return true or false.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn("2e10") should return true.
  fn("-.9") should return true.
  fn("4.") should return true.
  fn("1e") should return false.
  fn("99e2.5") should return false.
  fn(" 0.1 ") should return false.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the boolean result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-course-schedule": {
    id: "tmb-course-schedule",
    title: "course schedule",
    summary: "detect cycles in a prerequisite graph (topological order feasibility)",
    difficulty: "hard",
    timeout_s: 35,
    tests: [ { input:[2,[[1,0]]], expected:true }, { input:[2,[[1,0],[0,1]]], expected:false }, { input:[5,[[1,0],[2,1],[3,2],[4,3]]], expected:true }, { input:[3,[[0,1],[0,2],[1,2]]], expected:true }, { input:[3,[[0,1],[1,2],[2,0]]], expected:false }, { input:[1,[]], expected:true }, { input:[4,[[0,1],[2,3],[1,2],[3,1]]], expected:false } ],

    prompt: `Implement a JavaScript function named fn.

Task: determine whether all courses can be finished given prerequisites.

Rules:
- fn(args) receives [numCourses, prerequisites].
- numCourses is the number of courses labeled 0 through numCourses-1.
- prerequisites is an array of pairs [a, b] meaning course a depends on course b.
- Return true if it is possible to take all courses in some order (no cycle in the dependency graph), false otherwise.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn([2, [[1,0]]]) should return true.
  fn([2, [[1,0],[0,1]]]) should return false.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the boolean result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-min-window": {
    id: "tmb-min-window",
    title: "minimum window substring",
    summary: "sliding window, shortest substring containing all target characters",
    difficulty: "hard",
    timeout_s: 35,
    tests: [ { input:["ADOBECODEBANC","ABC"], expected:"BANC" }, { input:["a","a"], expected:"a" }, { input:["a","aa"], expected:"" }, { input:["ab","b"], expected:"b" }, { input:["aa","aa"], expected:"aa" }, { input:["cabeca","cae"], expected:"eca" } ],

    prompt: `Implement a JavaScript function named fn.

Task: find the minimum-length substring of s that contains every character of t.

Rules:
- fn(args) receives [s, t]: two strings.
- The window must contain each character of t at least as many times as it appears in t.
- Return the smallest such substring; if multiple tie, any of them is fine.
- Return an empty string if no window exists.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn(["ADOBECODEBANC","ABC"]) should return "BANC".
  fn(["a","aa"]) should return "".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result string with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-regex-match": {
    id: "tmb-regex-match",
    title: "regular expression matching",
    summary: "DP matching with '.' for single char and '*' for zero-or-more of preceding",
    difficulty: "hard",
    timeout_s: 30,
    tests: [ { input:["aa","a"], expected:false }, { input:["aa","a*"], expected:true }, { input:["ab",".*"], expected:true }, { input:["aab","c*a*b"], expected:true }, { input:["mississippi","mis*is*p*."], expected:false }, { input:["","a*"], expected:true }, { input:["a","ab*"], expected:true } ],

    prompt: `Implement a JavaScript function named fn.

Task: regular expression matching supporting '.' and '*'.

Rules:
- fn(args) receives [s, p]: the string and the pattern.
- '.' matches any single character.
- '*' matches zero or more of the preceding element (e.g. "a*" matches "", "a", "aa", ...).
- The match must cover the ENTIRE string.
- Return true or false.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn(["aa","a"]) should return false.
  fn(["aa","a*"]) should return true.
  fn(["aab","c*a*b"]) should return true.
  fn(["mississippi","mis*is*p*."]) should return false.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the boolean result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-max-points-line": {
    id: "tmb-max-points-line",
    title: "max points on a line",
    summary: "group points by reduced slope, count duplicates, find collinear maximum",
    difficulty: "hard",
    timeout_s: 35,
    tests: [ { input:[[1,1],[2,2],[3,3]], expected:3 }, { input:[[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]], expected:4 }, { input:[[0,0]], expected:1 }, { input:[[0,0],[1,1],[0,0]], expected:3 }, { input:[[1,1],[2,2],[3,3],[3,4]], expected:3 }, { input:[[0,0],[1,0],[2,0],[1,1]], expected:3 } ],

    prompt: `Implement a JavaScript function named fn.

Task: return the maximum number of points that lie on the same straight line.

Rules:
- fn(points) takes one array of [x, y] integer pairs.
- A point may appear more than once; each occurrence counts.
- Points on a line share the same reduced slope from an anchor point.
- Return the maximum count of collinear points (a single point returns 1).
- Use only Node.js built-ins (no npm packages).

Examples:
  fn([[1,1],[2,2],[3,3]]) should return 3.
  fn([[0,0],[1,1],[0,0]]) should return 3.

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the integer result with console.log(JSON.stringify(fn(...))) when run.`,
  },
};

// Trial problems — candidates for tightening the scored set. They are kept OUT
// of LEVELS (and so out of every recorded score) until a run shows they really
// separate models: run them one at a time with --tests=<id>, then promote the
// keepers by moving the entry up into PROBLEMS and deleting it here.
//
// All five are deliberately simple to state and produce a string or a flat
// integer array, so grading is exact (no object-with-array expectations, which
// the JS grader's subset comparison cannot check).
const TRIAL_PROBLEMS = {
  "tmb-cents-split": {
    id: "tmb-cents-split",
    title: "split a total into n exact parts",
    summary: "integer division, remainder distribution, parts must sum exactly",
    difficulty: "simple",
    timeout_s: 20,
    tests: [
      { input: [1000, 3], expected: [334, 333, 333] },
      { input: [5, 2], expected: [3, 2] },
      { input: [1, 3], expected: [1, 0, 0] },
      { input: [0, 4], expected: [0, 0, 0, 0] },
      { input: [999, 7], expected: [143, 143, 143, 143, 143, 142, 142] },
    ],
    prompt: `Implement a JavaScript function named fn.

Task: split a total into n parts as evenly as possible, without losing a cent.

Input: fn(args) receives [total, n]: a non-negative integer total and a positive integer n.

Rules:
- Return an array of exactly n non-negative integers that sum EXACTLY to total.
- Split as evenly as possible: every part is either the floor of total/n or one more than that.
- When the total does not divide evenly, give the extra units to the EARLIEST parts (lowest indexes).
- Use only Node.js built-ins (no npm packages).

Examples:
  fn([1000, 3]) should return [334,333,333].
  fn([5, 2]) should return [3,2].
  fn([1, 3]) should return [1,0,0].

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-truncate-utf8": {
    id: "tmb-truncate-utf8",
    title: "truncate a string to a byte budget",
    summary: "UTF-8 byte length vs string length, never split a character",
    difficulty: "simple",
    timeout_s: 20,
    tests: [
      { input: ["héllo", 2], expected: "h" },
      { input: ["héllo", 3], expected: "hé" },
      { input: ["日本語", 4], expected: "日" },
      { input: ["abc", 10], expected: "abc" },
      { input: ["😀x", 4], expected: "😀" },
      { input: ["héllo", 0], expected: "" },
    ],
    prompt: `Implement a JavaScript function named fn.

Task: truncate a string so that its UTF-8 encoding fits inside a byte budget.

Input: fn(args) receives [s, maxBytes]: a string and a non-negative integer byte budget.

Rules:
- Return the longest prefix of s whose UTF-8 encoding is at most maxBytes bytes long.
- Never split a character: if the next character does not fit, stop before it.
- Characters outside ASCII may take 2, 3, or 4 bytes in UTF-8.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn(["héllo", 2]) should return "h".
  fn(["héllo", 3]) should return "hé".
  fn(["日本語", 4]) should return "日".
  fn(["😀x", 4]) should return "😀".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-number-to-words": {
    id: "tmb-number-to-words",
    title: "integer to English words",
    summary: "scale words, hyphenated tens, no 'and'",
    difficulty: "simple",
    timeout_s: 25,
    tests: [
      { input: 0, expected: "zero" },
      { input: 21, expected: "twenty-one" },
      { input: 101, expected: "one hundred one" },
      { input: 1000, expected: "one thousand" },
      { input: 1234, expected: "one thousand two hundred thirty-four" },
      { input: 100000, expected: "one hundred thousand" },
      { input: 1000000, expected: "one million" },
    ],
    prompt: `Implement a JavaScript function named fn.

Task: spell a non-negative integer in English words.

Input: fn(n) takes one integer from 0 to 999999999 inclusive.

Rules:
- Return lowercase words separated by single spaces.
- Write tens (21-99) with a hyphen: 21 -> "twenty-one".
- Never use the word "and": 101 -> "one hundred one", not "one hundred and one".
- Use scale words "thousand" and "million": 1000 -> "one thousand", 1000000 -> "one million".
- Drop empty scales: 100000 -> "one hundred thousand".
- Do not put a hyphen between a scale word and the rest.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn(0) should return "zero".
  fn(21) should return "twenty-one".
  fn(101) should return "one hundred one".
  fn(1234) should return "one thousand two hundred thirty-four".
  fn(1000000) should return "one million".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-path-normalize": {
    id: "tmb-path-normalize",
    title: "normalize a unix path",
    summary: "collapse . and .., clamp at the root, relative vs absolute result",
    difficulty: "simple",
    timeout_s: 25,
    tests: [
      { input: "/a//b/./c/../d", expected: "/a/b/d" },
      { input: "a/b/../../c", expected: "c" },
      { input: "/../..", expected: "/" },
      { input: "/a/b/", expected: "/a/b" },
      { input: "./a", expected: "a" },
      { input: "a/..", expected: "." },
      { input: "../../x", expected: "../../x" },
    ],
    prompt: `Implement a JavaScript function named fn.

Task: normalize a Unix-style path string (like POSIX normpath).

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
- Use only Node.js built-ins (no npm packages).

Examples:
  fn("/a//b/./c/../d") should return "/a/b/d".
  fn("a/b/../../c") should return "c".
  fn("/../..") should return "/".
  fn("a/..") should return ".".
  fn("../../x") should return "../../x".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result with console.log(JSON.stringify(fn(...))) when run.`,
  },

  "tmb-kv-serialize": {
    id: "tmb-kv-serialize",
    title: "serialize k=v string",
    summary: "escape %, comma and equals in keys and values, escape % first",
    difficulty: "simple",
    timeout_s: 20,
    tests: [
      { input: { a: "1", b: "2" }, expected: "a=1,b=2" },
      { input: { "a b": "c,d" }, expected: "a b=c%2Cd" },
      { input: { "x=y": "pct%" }, expected: "x%3Dy=pct%25" },
      { input: { k: "a=b" }, expected: "k=a%3Db" },
      { input: {}, expected: "" },
    ],
    prompt: `Implement a JavaScript function named fn.

Task: serialize a plain object into a compact "key=value" string — the inverse of parsing it back.

Input: fn(obj) takes one object whose values are strings. Keys are strings too.

Rules:
- Produce "key=value" pairs joined by commas, in the object's key order.
- In BOTH keys and values, escape these characters, and escape the percent sign FIRST:
  - "%" becomes "%25"
  - "," becomes "%2C"
  - "=" becomes "%3D"
- Escaping must not double-escape: a literal "%" always becomes exactly "%25".
- An empty object returns an empty string.
- Use only Node.js built-ins (no npm packages).

Examples:
  fn({a:"1",b:"2"}) should return "a=1,b=2".
  fn({"a b":"c,d"}) should return "a b=c%2Cd".
  fn({"x=y":"pct%"}) should return "x%3Dy=pct%25".

Return ONLY a JavaScript code block containing the fn function.
Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.
Print the result with console.log(JSON.stringify(fn(...))) when run.`,
  },
};

// Every problem the harness can run: the scored set plus the trial candidates.
// LEVELS still resolves to PROBLEMS only, so trial problems never enter a
// recorded score until they are promoted.
const ALL_PROBLEMS = { ...PROBLEMS, ...TRIAL_PROBLEMS };

// Run levels: which problems (and how many tests each) to run.
//   low   - the 10 hardest problems, 1 test each
//   med   - all problems, 1 test each (the default / "normal")
//   large - all problems, every edge-case test each
const LEVELS = {
  low: ["tmb-rpn-eval", "tmb-lru-ops", "tmb-longest-substr", "tmb-max-subarray", "tmb-trap-rain-water", "tmb-polygon-area", "tmb-merge-ranges", "tmb-valid-parentheses", "tmb-roman-to-int", "tmb-int-to-roman"],
  med: Object.keys(PROBLEMS),
  large: Object.keys(PROBLEMS),
};

function modelPrompt(problem) {
  return [
    "You are completing one tiny coding benchmark task.",
    "Return ONLY a JavaScript code block containing a function named fn.",
    "Do not explain your answer. Do not hardcode only the visible example; implement the stated behavior.",
    "Use only Node.js built-ins.",
    "",
    `Task: ${problem.prompt}`,
  ].join("\n");
}

async function runNodeScript(source, node, timeoutMs) {
  const dir = await mkdtemp(join(tmpdir(), "tiny-js-benchmark-"));
  const path = join(dir, "candidate.js");
  try {
    await writeFile(path, source, "utf8");
    const started = performance.now();
    const result = await execFileAsync(node, [path], {
      cwd: ROOT,
      timeout: timeoutMs,
      maxBuffer: 1024 * 1024,
    });
    const elapsedMs = Math.round(performance.now() - started);
    return {
      pass: false,
      elapsed_ms: elapsedMs,
      stdout: stripAnsi(result.stdout).trim(),
      stderr: stripAnsi(result.stderr).trim(),
      exit_code: result.status,
    };
  } catch (error) {
    const stderr = stripAnsi(String(error.stderr || "")).trim();
    const stderrFirst = pickErrorHeadline(stderr);
    const message = error.killed
      ? `node timeout after ${timeoutMs / 1000}s`
      : (stderrFirst || `node exited with code ${error.code || 1}`);
    return {
      pass: false,
      elapsed_ms: Math.round(performance.now() - (error.startedAt || performance.now())),
      stdout: stripAnsi(String(error.stdout || "")).trim(),
      stderr,
      exit_code: error.killed ? -1 : (error.code || 1),
      error: message.slice(0, 240),
    };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

function lineMatches(expected, candidate) {
  if (!candidate) return false;
  // 1. Exact JSON match.
  if (candidate === JSON.stringify(expected)) return true;
  // 2. For object expected, check key/value presence in the parsed candidate.
  if (typeof expected === "object" && expected !== null) {
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return Object.entries(expected).every(([k, v]) => parsed[k] === v);
      }
    } catch { }
  }
  // 3. Loose fallback for scalar expected (string/number/boolean).
  const expectedText = String(expected);
  if (expectedText && candidate.includes(expectedText)) return true;
  return false;
}

function expectedInOutput(expected, stdout) {
  if (!stdout) return false;
  const lines = stdout.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!lines.length) return false;
  return lineMatches(expected, lines[lines.length - 1]);
}

// Multi-test grading: the harness appends one console.log per test, so the
// last N non-empty stdout lines are the answers, checked in order.
function gradeTests(expecteds, stdout) {
  if (!stdout) return false;
  const lines = stdout.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.length < expecteds.length) return false;
  const answers = lines.slice(-expecteds.length);
  return expecteds.every((exp, i) => lineMatches(exp, answers[i]));
}

function problemTests(problem) {
  return (problem.tests && problem.tests.length) ? problem.tests : [{ input: problem.testInput, expected: problem.expected }];
}

function buildSource(code, tests) {
  const calls = tests.map(t => `console.log(JSON.stringify(fn(${JSON.stringify(t.input)})));`).join("\n");
  return `${code}\n\n// Harness calls\n${calls}\n`;
}

async function directFixture(problem, modelConfig, useAllTests) {
  const started = performance.now();
  // Deterministic reference implementations. Each prints the answer for the
  // problem's test input so the grading/extraction path is validated end-to-end.
  const FIXTURES = {
    "tmb-jwt": `function fn(token) {
  const parts = token.split(".");
  if (parts.length !== 3) throw new Error("bad jwt");
  const payloadB64 = parts[1];
  // base64url -> base64
  let base64 = payloadB64.replace(/-/g, "+").replace(/_/g, "/");
  // pad
  while (base64.length % 4) base64 += "=";
  const text = Buffer.from(base64, "base64").toString("utf8");
  return JSON.parse(text);
}

const token = process.argv[2] || "${problem.token}";
console.log(JSON.stringify(fn(token)));`,
    "tmb-slug": `function fn(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.input)})));`,
    "tmb-kv-parse": `function fn(s) {
  const obj = {};
  for (const pair of s.split(",")) {
    const eq = pair.indexOf("=");
    if (eq === -1) continue;
    const k = pair.slice(0, eq).trim();
    const v = pair.slice(eq + 1).trim();
    if (k) obj[k] = v;
  }
  return obj;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-fix-index-delim": `function fn(s) {
  const pairs = s.split(",");
  const obj = {};
  for (let i = 0; i < pairs.length; i++) {
    const pair = pairs[i];
    if (!pair) continue;
    const eq = pair.indexOf("=");
    if (eq === -1) continue;
    const k = pair.slice(0, eq).trim();
    const v = pair.slice(eq + 1).trim();
    obj[k] = v;
  }
  return obj;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-merge-ranges": `function fn(ranges) {
  if (!Array.isArray(ranges) || ranges.length === 0) return [];
  const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const pair of sorted) {
    const start = pair[0];
    const end = pair[1];
    if (!out.length || start > out[out.length - 1][1]) {
      out.push([start, end]);
    } else {
      out[out.length - 1][1] = Math.max(out[out.length - 1][1], end);
    }
  }
  return out;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-rpn-eval": `function fn(tokens) {
  const stack = [];
  for (const token of tokens) {
    if (token === "+" || token === "-" || token === "*" || token === "/") {
      const b = stack.pop();
      const a = stack.pop();
      if (token === "+") stack.push(a + b);
      else if (token === "-") stack.push(a - b);
      else if (token === "*") stack.push(a * b);
      else stack.push(Math.trunc(a / b));
    } else {
      stack.push(Number(token));
    }
  }
  return stack[stack.length - 1];
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-lru-ops": `function fn(input) {
  const capacity = input.capacity;
  const ops = input.ops;
  const cache = new Map();
  const out = [];

  for (const op of ops) {
    if (op[0] === "get") {
      const key = op[1];
      if (!cache.has(key)) {
        out.push(-1);
        continue;
      }
      const value = cache.get(key);
      cache.delete(key);
      cache.set(key, value);
      out.push(value);
      continue;
    }

    const key = op[1];
    const value = op[2];
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    if (cache.size > capacity) {
      const oldest = cache.keys().next().value;
      cache.delete(oldest);
    }
  }

  return out;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-reverse-words": `function fn(s) {
  return s.trim().split(/\\s+/).reverse().join(" ");
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-fizzbuzz": `function fn(n) {
  const out = [];
  for (let i = 1; i <= n; i++) {
    if (i % 15 === 0) out.push("FizzBuzz");
    else if (i % 3 === 0) out.push("Fizz");
    else if (i % 5 === 0) out.push("Buzz");
    else out.push(String(i));
  }
  return out;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-palindrome": `function fn(s) {
  const c = s.toLowerCase().replace(/[^a-z0-9]/g, "");
  return c === c.split("").reverse().join("");
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-valid-parentheses": `function fn(s) {
  const stack = [];
  const map = { ")": "(", "]": "[", "}": "{" };
  for (const ch of s) {
    if (ch === "(" || ch === "[" || ch === "{") stack.push(ch);
    else if (stack.pop() !== map[ch]) return false;
  }
  return stack.length === 0;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-roman-to-int": `function fn(s) {
  const v = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const cur = v[s[i]];
    const next = v[s[i + 1]] || 0;
    if (cur < next) total -= cur;
    else total += cur;
  }
  return total;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-int-to-roman": `function fn(num) {
  const map = [
    [1000, "M"], [900, "CM"], [500, "D"], [400, "CD"],
    [100, "C"], [90, "XC"], [50, "L"], [40, "XL"],
    [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"],
  ];
  let result = "";
  for (const [value, symbol] of map) {
    while (num >= value) {
      result += symbol;
      num -= value;
    }
  }
  return result;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-first-unique-char": `function fn(s) {
  const count = {};
  for (const ch of s) count[ch] = (count[ch] || 0) + 1;
  for (let i = 0; i < s.length; i++) {
    if (count[s[i]] === 1) return i;
  }
  return -1;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-longest-substr": `function fn(s) {
  const seen = new Map();
  let start = 0;
  let max = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (seen.has(ch) && seen.get(ch) >= start) start = seen.get(ch) + 1;
    seen.set(ch, i);
    max = Math.max(max, i - start + 1);
  }
  return max;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-max-subarray": `function fn(nums) {
  let best = nums[0];
  let current = nums[0];
  for (let i = 1; i < nums.length; i++) {
    current = Math.max(nums[i], current + nums[i]);
    best = Math.max(best, current);
  }
  return best;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-trap-rain-water": `function fn(height) {
  let left = 0, right = height.length - 1;
  let leftMax = 0, rightMax = 0, water = 0;
  while (left < right) {
    if (height[left] < height[right]) {
      leftMax = Math.max(leftMax, height[left]);
      water += leftMax - height[left];
      left++;
    } else {
      rightMax = Math.max(rightMax, height[right]);
      water += rightMax - height[right];
      right--;
    }
  }
  return water;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-sum-multiples": `function fn(n) {
  let s = 0;
  for (let i = 1; i < n; i++) {
    if (i % 3 === 0 || i % 5 === 0) s += i;
  }
  return s;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-bounding-box": `function fn(points) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
  }
  return [minX, minY, maxX, maxY];
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-polygon-area": `function fn(pts) {
  let area = 0;
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % n];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-longest-palindromic-substring": `function fn(s) {
  if (!s) return "";
  let start = 0, maxLen = 1;
  for (let i = 0; i < s.length; i++) {
    for (const [l, r] of [[i, i], [i, i + 1]]) {
      let a = l, b = r;
      while (a >= 0 && b < s.length && s[a] === s[b]) { a--; b++; }
      const len = b - a - 1;
      if (len > maxLen) { maxLen = len; start = a + 1; }
    }
  }
  return s.slice(start, start + maxLen);
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-num-islands": `function fn(grid) {
  if (!grid.length) return 0;
  const rows = grid.length, cols = grid[0].length;
  let count = 0;
  const seen = Array.from({ length: rows }, () => Array(cols).fill(false));
  const dirs = [[1,0],[-1,0],[0,1],[0,-1]];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === "1" && !seen[r][c]) {
        count++;
        const stack = [[r, c]];
        seen[r][c] = true;
        while (stack.length) {
          const [cr, cc] = stack.pop();
          for (const [dr, dc] of dirs) {
            const nr = cr + dr, nc = cc + dc;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] === "1" && !seen[nr][nc]) {
              seen[nr][nc] = true;
              stack.push([nr, nc]);
            }
          }
        }
      }
    }
  }
  return count;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-lis": `function fn(nums) {
  if (!nums.length) return 0;
  const dp = new Array(nums.length).fill(1);
  let best = 1;
  for (let i = 1; i < nums.length; i++) {
    for (let j = 0; j < i; j++) {
      if (nums[j] < nums[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
    }
    best = Math.max(best, dp[i]);
  }
  return best;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-edit-distance": `function fn(args) {
  const [a, b] = args;
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1];
      else dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-valid-sudoku": `function fn(board) {
  const rows = new Set(), cols = new Set(), boxes = new Set();
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const v = board[r][c];
      if (v === ".") continue;
      const b = Math.floor(r / 3) * 3 + Math.floor(c / 3);
      const rk = "r" + r + ":" + v, ck = "c" + c + ":" + v, bk = "b" + b + ":" + v;
      if (rows.has(rk) || cols.has(ck) || boxes.has(bk)) return false;
      rows.add(rk); cols.add(ck); boxes.add(bk);
    }
  }
  return true;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-coin-change": `function fn(args) {
  const [coins, amount] = args;
  const dp = new Array(amount + 1).fill(Infinity);
  dp[0] = 0;
  for (let i = 1; i <= amount; i++) {
    for (const c of coins) {
      if (c <= i) dp[i] = Math.min(dp[i], dp[i - c] + 1);
    }
  }
  return dp[amount] === Infinity ? -1 : dp[amount];
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-longest-valid-parentheses": `function fn(s) {
  let best = 0;
  const stack = [-1];
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "(") stack.push(i);
    else {
      stack.pop();
      if (stack.length === 0) stack.push(i);
      else best = Math.max(best, i - stack[stack.length - 1]);
    }
  }
  return best;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-median-two-sorted": `function fn(args) {
  const [a, b] = args;
  const merged = [...a, ...b].sort((x, y) => x - y);
  const n = merged.length;
  if (n % 2 === 1) return merged[Math.floor(n / 2)];
  return (merged[n / 2 - 1] + merged[n / 2]) / 2;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-sliding-window-max": `function fn(args) {
  const [nums, k] = args;
  const out = [];
  const dq = [];
  for (let i = 0; i < nums.length; i++) {
    while (dq.length && nums[dq[dq.length - 1]] <= nums[i]) dq.pop();
    dq.push(i);
    if (dq[0] <= i - k) dq.shift();
    if (i >= k - 1) out.push(nums[dq[0]]);
  }
  return out;
}

console.log(JSON.stringify(fn(${JSON.stringify(problem.testInput)})));`,
    "tmb-n-queens": `function fn(n) {
  let count = 0;
  const cols = new Set(), diag = new Set(), anti = new Set();
  function backtrack(row) {
    if (row === n) { count++; return; }
    for (let c = 0; c < n; c++) {
      if (cols.has(c) || diag.has(row - c) || anti.has(row + c)) continue;
      cols.add(c); diag.add(row - c); anti.add(row + c);
      backtrack(row + 1);
      cols.delete(c); diag.delete(row - c); anti.delete(row + c);
    }
  }
  backtrack(0);
  return count;
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-atoi": `function fn(s) {
  const str = String(s);
  let i = 0;
  while (i < str.length && str[i] === " ") i++;
  let sign = 1;
  if (i < str.length && (str[i] === "+" || str[i] === "-")) {
    if (str[i] === "-") sign = -1;
    i++;
  }
  let num = 0;
  while (i < str.length && str[i] >= "0" && str[i] <= "9") {
    num = num * 10 + (str.charCodeAt(i) - 48);
    if (num > 2147483648) { num = 2147483648; break; }
    i++;
  }
  num *= sign;
  if (num > 2147483647) return 2147483647;
  if (num < -2147483648) return -2147483648;
  return num;
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-csv-parse": `function fn(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  let fieldQuoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += ch;
    } else if (ch === '"') {
      inQuotes = true;
      fieldQuoted = true;
    } else if (ch === ",") {
      row.push(field); field = ""; fieldQuoted = false;
    } else if (ch === "\\n") {
      row.push(field); field = "";
      rows.push(row); row = []; fieldQuoted = false;
    } else field += ch;
  }
  row.push(field);
  if (row.length > 1 || row[0] !== "" || fieldQuoted) rows.push(row);
  return rows;
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-basic-calc": `function fn(expr) {
  let s = expr.replace(/\\s+/g, "");
  let norm = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if ((ch === "+" || ch === "-") && (i === 0 || "(-+*/".includes(s[i - 1]))) norm += "0";
    norm += ch;
  }
  s = norm;
  const tokens = [];
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if ("+-*/()".includes(ch)) { tokens.push(ch); continue; }
    let num = "";
    while (i < s.length && /\\d/.test(s[i])) { num += s[i]; i++; }
    tokens.push(num);
    i--;
  }
  const prec = { "+": 1, "-": 1, "*": 2, "/": 2 };
  const out = [];
  const ops = [];
  for (const t of tokens) {
    if (/\\d/.test(t)) out.push(t);
    else if (t === "(") ops.push(t);
    else if (t === ")") {
      while (ops.length && ops[ops.length - 1] !== "(") out.push(ops.pop());
      ops.pop();
    } else {
      while (ops.length && ops[ops.length - 1] !== "(" && prec[ops[ops.length - 1]] >= prec[t]) out.push(ops.pop());
      ops.push(t);
    }
  }
  while (ops.length) out.push(ops.pop());
  const stack = [];
  for (const t of out) {
    if (/\\d/.test(t)) stack.push(Number(t));
    else {
      const b = stack.pop(), a = stack.pop();
      if (t === "+") stack.push(a + b);
      else if (t === "-") stack.push(a - b);
      else if (t === "*") stack.push(a * b);
      else stack.push(Math.trunc(a / b));
    }
  }
  return stack[stack.length - 1];
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-text-justify": `function fn(args) {
  const [words, maxWidth] = args;
  const lines = [];
  let cur = [], curLen = 0;
  for (const w of words) {
    if (cur.length && curLen + cur.length + w.length > maxWidth) {
      lines.push([cur, curLen]);
      cur = []; curLen = 0;
    }
    cur.push(w);
    curLen += w.length;
  }
  if (cur.length) lines.push([cur, curLen]);
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const [ws, len] = lines[i];
    const isLast = i === lines.length - 1;
    const gapCount = ws.length - 1;
    const fill = maxWidth - len;
    if (isLast || gapCount === 0) {
      out.push(ws.join(" ") + " ".repeat(maxWidth - len - gapCount));
    } else {
      const base = Math.floor(fill / gapCount);
      let extra = fill % gapCount;
      let str = "";
      for (let j = 0; j < ws.length; j++) {
        if (j > 0) {
          str += " ".repeat(base + (extra > 0 ? 1 : 0));
          if (extra > 0) extra--;
        }
        str += ws[j];
      }
      out.push(str);
    }
  }
  return out;
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-wildcard-match": `function fn(args) {
  const [s, p] = args;
  const m = s.length, n = p.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(false));
  dp[0][0] = true;
  for (let j = 1; j <= n; j++) if (p[j - 1] === "*") dp[0][j] = dp[0][j - 1];
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (p[j - 1] === "*") dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
      else dp[i][j] = (p[j - 1] === "?" || p[j - 1] === s[i - 1]) && dp[i - 1][j - 1];
    }
  }
  return dp[m][n];
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-valid-number": `function fn(s) {
  const str = String(s);
  if (!str) return false;
  return /^[+-]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][+-]?\\d+)?$/.test(str);
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-course-schedule": `function fn(args) {
  const [numCourses, prerequisites] = args;
  const adj = Array.from({ length: numCourses }, () => []);
  const indeg = new Array(numCourses).fill(0);
  for (const [a, b] of prerequisites) { adj[b].push(a); indeg[a]++; }
  const q = [];
  for (let i = 0; i < numCourses; i++) if (indeg[i] === 0) q.push(i);
  let done = 0;
  while (q.length) {
    const c = q.shift();
    done++;
    for (const nx of adj[c]) { if (--indeg[nx] === 0) q.push(nx); }
  }
  return done === numCourses;
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-min-window": `function fn(args) {
  const [s, t] = args;
  if (!t) return "";
  const need = {};
  for (const ch of t) need[ch] = (need[ch] || 0) + 1;
  let have = 0;
  const want = Object.keys(need).length;
  let left = 0, best = "", bestLen = Infinity;
  const win = {};
  for (let right = 0; right < s.length; right++) {
    const ch = s[right];
    win[ch] = (win[ch] || 0) + 1;
    if (need[ch] && win[ch] === need[ch]) have++;
    while (have === want) {
      const len = right - left + 1;
      if (len < bestLen) { bestLen = len; best = s.slice(left, right + 1); }
      const lc = s[left];
      win[lc]--;
      if (need[lc] && win[lc] < need[lc]) have--;
      left++;
    }
  }
  return best;
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-regex-match": `function fn(args) {
  const [s, p] = args;
  const m = s.length, n = p.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(false));
  dp[0][0] = true;
  for (let j = 2; j <= n; j++) if (p[j - 1] === "*") dp[0][j] = dp[0][j - 2];
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (p[j - 1] === "*") {
        dp[i][j] = dp[i][j - 2] || ((p[j - 2] === "." || p[j - 2] === s[i - 1]) && dp[i - 1][j]);
      } else {
        dp[i][j] = (p[j - 1] === "." || p[j - 1] === s[i - 1]) && dp[i - 1][j - 1];
      }
    }
  }
  return dp[m][n];
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-max-points-line": `function fn(points) {
  if (points.length < 2) return points.length;
  let best = 1;
  for (let i = 0; i < points.length; i++) {
    const slopes = new Map();
    let same = 0;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const [x1, y1] = points[i];
      const [x2, y2] = points[j];
      if (x1 === x2 && y1 === y2) { same++; continue; }
      let dx = x2 - x1, dy = y2 - y1;
      const g = gcd(Math.abs(dx), Math.abs(dy));
      dx /= g; dy /= g;
      if (dx < 0) { dx = -dx; dy = -dy; }
      if (dx === 0) dy = Math.abs(dy);
      slopes.set(dx + "," + dy, (slopes.get(dx + "," + dy) || 0) + 1);
    }
    let max = 0;
    for (const v of slopes.values()) max = Math.max(max, v);
    best = Math.max(best, max + same + 1);
  }
  return best;
  function gcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,

    // --- trial candidates ---------------------------------------------------
    "tmb-cents-split": `function fn(args) {
  const [total, n] = args;
  const base = Math.floor(total / n);
  const extra = total - base * n;
  return Array.from({ length: n }, (_, i) => base + (i < extra ? 1 : 0));
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-truncate-utf8": `function fn(args) {
  const [s, maxBytes] = args;
  let out = "";
  for (const ch of s) {            // iterate by code point, not code unit
    if (Buffer.byteLength(out + ch, "utf8") > maxBytes) break;
    out += ch;
  }
  return out;
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-number-to-words": `const ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"];
const TENS = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"];
function under1000(n) {
  if (n < 20) return ONES[n];
  if (n < 100) {
    const r = n % 10;
    return TENS[Math.floor(n / 10)] + (r ? "-" + ONES[r] : "");
  }
  const r = n % 100;
  return ONES[Math.floor(n / 100)] + " hundred" + (r ? " " + under1000(r) : "");
}
function fn(n) {
  if (n === 0) return "zero";
  const parts = [];
  let rest = n;
  for (const [value, name] of [[1000000, "million"], [1000, "thousand"]]) {
    if (rest >= value) {
      parts.push(under1000(Math.floor(rest / value)) + " " + name);
      rest %= value;
    }
  }
  if (rest) parts.push(under1000(rest));
  return parts.join(" ");
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-path-normalize": `function fn(path) {
  const absolute = path.startsWith("/");
  const out = [];
  for (const part of path.split("/")) {
    if (part === "" || part === ".") continue;
    if (part === "..") {
      if (out.length && out[out.length - 1] !== "..") out.pop();
      else if (!absolute) out.push("..");
      continue;
    }
    out.push(part);
  }
  const joined = out.join("/");
  if (absolute) return "/" + joined;
  return joined || ".";
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
    "tmb-kv-serialize": `function fn(obj) {
  const esc = (s) => String(s).replace(/%/g, "%25").replace(/,/g, "%2C").replace(/=/g, "%3D");
  return Object.keys(obj).map((k) => esc(k) + "=" + esc(obj[k])).join(",");
}

console.log(JSON.stringify(fn(${JSON.stringify(problemTests(problem)[0].input)})));`,
  };
  // Strict: a missing fixture used to silently fall back to the JWT fixture,
  // which grades the wrong reference implementation against these tests.
  let code = FIXTURES[problem.id];
  if (!code) throw new Error(`No fixture for ${problem.id} — direct mode cannot validate it.`);
  // level=large validates EVERY expected value, the same way a model run is
  // graded. The fixture prints the first test itself, so only the remaining
  // calls get appended; grading then reads the last N stdout lines, as usual.
  const allTests = problemTests(problem);
  const tests = useAllTests ? allTests : [allTests[0]];
  const extra = tests.slice(1)
    .map((t) => `console.log(JSON.stringify(fn(${JSON.stringify(t.input)})));`).join("\n");
  if (extra) code += "\n" + extra;
  const grade = await runNodeScript(code, modelConfig.TMB_NODE, Number(problem.timeout_s) * 1000);
  grade.pass = gradeTests(tests.map((t) => t.expected), grade.stdout);
  grade.generated_code = code;
  grade.backend = "direct-fixture";
  grade.model = null;
  grade.problem = problem.id;
  grade.response = "(deterministic fixture; validates extraction/grading path)";
  return grade;
}

async function ollama(problem, modelConfig, useAllTests) {
  const started = performance.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), Number(modelConfig.TMB_TIMEOUT_S) * 1000);
  let responseText = "";
  try {
    if (bool(config.TMB_WARMUP)) {
      console.log("warmup: greeting the model...");
      const warmup = await fetch(`${normalizeUrl(modelConfig.TMB_OLLAMA_URL || modelConfig.OLLAMA_URL)}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: modelConfig.TMB_MODEL,
          messages: [{ role: "user", content: "Hello, let's begin." }],
          stream: false,
          think: bool(config.TMB_THINK),
          options: { temperature: 0, num_predict: 32 },
        }),
      });
      if (!warmup.ok) {
        const body = await warmup.json().catch(() => ({}));
        throw new Error(`Ollama warmup HTTP ${warmup.status}: ${JSON.stringify(body).slice(0, 300)}`);
      }
      console.log("warmup: done.");
    }

    PrintSpinner.start(`ollama · ${modelConfig.TMB_MODEL}`);

    const allTests = problemTests(problem);
    const tests = useAllTests ? allTests : [allTests[0]];
    const maxAttempts = int(config.TMB_ATTEMPTS, 3);
    const messages = [{ role: "user", content: modelPrompt(problem) }];
    let code = "";
    let grade = null;
    let attemptsUsed = 0;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      attemptsUsed = attempt;
      const response = await fetch(`${normalizeUrl(modelConfig.TMB_OLLAMA_URL || modelConfig.OLLAMA_URL)}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          model: modelConfig.TMB_MODEL,
          messages,
          stream: false,
          think: bool(config.TMB_THINK),
          options: { temperature: 0, num_predict: Number(modelConfig.TMB_NUM_PREDICT) },
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(`Ollama HTTP ${response.status}: ${JSON.stringify(data).slice(0, 1000)}`);
      responseText = data.message?.content || data.response || "";
      code = extractCode(responseText);

      if (!code) {
        if (attempt < maxAttempts) {
          messages.push({ role: "assistant", content: responseText });
          messages.push({ role: "user", content: "Your response did not contain a JavaScript function named fn. Return ONLY a code block defining function fn." });
          continue;
        }
        PrintSpinner.stop();
        return {
          backend: "ollama",
          model: config.TMB_MODEL,
          pass: false,
          elapsed_ms: Math.round(performance.now() - started),
          response: responseText.slice(-4000),
          error: "no JavaScript function named fn extracted",
          stdout: "",
          stderr: "",
          exit_code: null,
          problem: problem.id,
        };
      }

      const source = buildSource(code, tests);
      grade = await runNodeScript(source, config.TMB_NODE, Number(problem.timeout_s) * 1000);
      grade.pass = gradeTests(tests.map(t => t.expected), grade.stdout);

      if (grade.pass || attempt >= maxAttempts) break;

      // Self-repair: feed the failure back and ask for a fix.
      const feedback = `Your code did not pass all tests.\nstdout: ${fmtStdout(grade.stdout) || "(empty)"}${grade.error ? `\nerror: ${grade.error}` : ""}${grade.stderr ? `\nstderr: ${grade.stderr.slice(0, 300)}` : ""}\nFix the fn function and return ONLY the corrected code block.`;
      messages.push({ role: "assistant", content: responseText });
      messages.push({ role: "user", content: feedback });
    }

    PrintSpinner.stop();
    grade.generated_code = code;
    grade.response = responseText.slice(-4000);
    grade.backend = "ollama";
    grade.model = modelConfig.TMB_MODEL;
    grade.problem = problem.id;
    grade.attempts = attemptsUsed;
    // runNodeScript's elapsed_ms only measures the grading step. Report the
    // whole check instead (generation + any self-repair attempts + grading) so
    // the js and py harnesses' ms columns mean the same thing — tiny_benchmark.py
    // makes the identical override in its ollama().
    grade.elapsed_ms = Math.round(performance.now() - started);
    return grade;
  } catch (error) {
    PrintSpinner.stop();
    return {
      backend: "ollama",
      model: config.TMB_MODEL,
      pass: false,
      elapsed_ms: Math.round(performance.now() - started),
      response: responseText.slice(-4000),
      stdout: "",
      stderr: "",
      exit_code: null,
      error: error.name === "AbortError" ? `Ollama timeout after ${modelConfig.TMB_TIMEOUT_S}s` : String(error.message || error).slice(-1000),
      problem: problem.id,
    };
  } finally {
    clearTimeout(timer);
  }
}

/*
 * Overwriting single-line spinner. Use it only around the slow Ollama path.
 * printSpinner.start(label) begins overwriting one line with the label + frames.
 * printSpinner.stop() restores a clean line and returns the last frame text.
 */
const TTY = process.stdout.isTTY;

const PrintSpinner = (() => {
  const FRAMES = TTY ? ["\u280B", "\u2819", "\u280A", "\u2814"] : ["."];
  let timer = null;
  let frameIndex = 0;
  let line = "";
  let lastWritten = "";

  function write(text) {
    if (!TTY) return;
    process.stdout.write(`\x1b[2K\r${text}`);
    lastWritten = text;
  }

  function frame() {
    const label = line;
    const f = FRAMES[frameIndex % FRAMES.length];
    if (TTY) {
      write(`${f} ${label}`);
    } else {
      // Non-TTY: pulse a single short marker instead of spamming frames.
      write(`${label} ${".".repeat((frameIndex % 3) + 1)}`);
    }
    frameIndex++;
  }

  return {
    start(label) {
      if (timer) this.stop();
      line = label || "working";
      frameIndex = 0;
      frame();
      timer = setInterval(frame, TTY ? 120 : 700);
    },
    stop() {
      if (!timer) return "";
      clearInterval(timer);
      timer = null;
      if (TTY) {
        // Fully clear the progress line and return the cursor to column 0.
        process.stdout.write("\x1b[2K\r");
      }
      return line;
    },
  };
})();

function fmtStdout(text) {
  if (!text) return "";
  // Collapse the authoritative harness line to one line, keep a sensible cap.
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const tail = lines.slice(-1).join(" ").slice(0, 120);
  return tail || text.slice(0, 120);
}

function classifyNodeFailure(result) {
  if (!result || result.pass) return "";
  if (result.error && /timeout/i.test(result.error)) return "timeout";

  const haystack = `${result.stderr || ""}\n${result.error || ""}`;
  // Backend/transport failures are not the model's fault: an unloadable model
  // (e.g. an architecture llama.cpp does not know) or a dead server must never
  // be scored as a wrong answer, or a run that never generated reads as 0%.
  if (/Ollama HTTP \d|error loading model|unknown model architecture|ECONNREFUSED|fetch failed|socket hang up/i.test(haystack)) {
    return "infra";
  }
  if (/SyntaxError/i.test(haystack)) return "syntax";
  if (/ReferenceError|TypeError|RangeError|Error:/i.test(haystack)) return "runtime";
  if (result.error && /no JavaScript function named fn extracted/i.test(result.error)) return "extract";
  if (result.exit_code != null && result.exit_code !== 0) return "node-exit";
  return "wrong-answer";
}

function compactError(result) {
  if (!result || result.pass) return "";
  const preferred = result.error || "";
  const stderrLine = pickErrorHeadline(result.stderr || "");
  const message = preferred || stderrLine || "failed";
  return message.slice(0, 80);
}

function statusLine(result) {
  const icon = result.pass ? "✅" : "❌";
  const problem = result.problem ? ` ${result.problem}` : "";
  const model = result.model ? `  ${result.model}` : "";
  const time = result.elapsed_ms != null ? `  ${result.elapsed_ms}ms` : "";
  const exit = result.exit_code != null && result.exit_code !== 0 ? `  exit ${result.exit_code}` : "";
  console.log(`${icon}${problem} ${result.backend}${model}${time}${exit}`);
  if (result.error_compact || result.error) console.log(`   ⚠ ${result.error_compact || result.error}`);
  if (bool(config.TMB_DEBUG) && result.stderr && !result.pass) {
    const snippet = debugErrorSnippet(result.stderr);
    if (snippet) console.log(`   debug stderr: ${snippet}`);
  }
}

function printResultTable(results) {
  const headers = ["Problem", "Backend", "Run", "Model", "Pass", "Time", "Exit", "Type", "Err"];
  const rows = results.map((r) => [
    r.problem || "—",
    r.backend,
    r.run != null ? String(r.run) : "—",
    r.model || "—",
    r.pass ? "✅" : "❌",
    `${r.elapsed_ms}ms`,
    r.exit_code != null ? String(r.exit_code) : "—",
    r.failure_type || "—",
    r.error_compact || "—",
  ]);

  const colW = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => String(r[i]).length)));

  console.log("");
  console.log("┌" + colW.map((w) => "─".repeat(w + 2)).join("┬") + "┐");
  console.log("│ " + headers.map((h, i) => h.padEnd(colW[i])).join(" │ ") + " │");
  console.log("├" + colW.map((w) => "─".repeat(w + 2)).join("┼") + "┤");
  for (const row of rows) {
    console.log("│ " + row.map((c, i) => String(c).padEnd(colW[i])).join(" │ ") + " │");
  } console.log("└" + colW.map((w) => "─".repeat(w + 2)).join("┴") + "┘");
  console.log("");

  const failures = results.filter((r) => !r.pass);
  if (failures.length) {
    const keyFor = (r) => `${r.failure_type || "unknown"}||${r.error_compact || "failed"}`;
    const grouped = new Map();
    for (const r of failures) {
      const key = keyFor(r);
      if (!grouped.has(key)) grouped.set(key, { type: r.failure_type || "unknown", err: r.error_compact || "failed", count: 0 });
      grouped.get(key).count++;
    }

    const groupedRows = [...grouped.values()].sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
    const gh = ["Count", "Type", "Error"];
    const gr = groupedRows.map((g) => [String(g.count), g.type, g.err]);
    const gw = gh.map((h, i) => Math.max(h.length, ...gr.map((r) => String(r[i]).length)));

    console.log("Failure summary:");
    console.log("┌" + gw.map((w) => "─".repeat(w + 2)).join("┬") + "┐");
    console.log("│ " + gh.map((h, i) => h.padEnd(gw[i])).join(" │ ") + " │");
    console.log("├" + gw.map((w) => "─".repeat(w + 2)).join("┼") + "┤");
    for (const row of gr) {
      console.log("│ " + row.map((c, i) => String(c).padEnd(gw[i])).join(" │ ") + " │");
    }
    console.log("└" + gw.map((w) => "─".repeat(w + 2)).join("┴") + "┘");
    console.log("");
  }
}

/*
 * Short end-of-run report: one row per harness+model, so a js run and a py run
 * can be read side by side without scrolling the per-check table. Mirrors
 * print_summary_table() in tiny_benchmark.py — keep the two in step.
 */
function printSummaryTable(results, level) {
  const order = [];
  const groups = new Map();
  for (const r of results) {
    const key = `${r.backend}||${r.model || "—"}`;
    if (!groups.has(key)) {
      groups.set(key, { backend: r.backend, model: r.model || "—", rows: [] });
      order.push(key);
    }
    groups.get(key).rows.push(r);
  }
  if (!order.length) return;

  const headers = ["Harness", "Model", "Checks", "Pass", "Rate", "Timeouts", "Top failures"];
  const rows = order.map((key) => {
    const g = groups.get(key);
    const total = g.rows.length;
    const passed = g.rows.filter((r) => r.pass).length;
    const timeouts = g.rows.filter((r) => r.failure_type === "timeout").length;
    const counts = new Map();
    for (const r of g.rows) {
      if (r.pass || r.failure_type === "timeout") continue;
      const type = r.failure_type || "unknown";
      counts.set(type, (counts.get(type) || 0) + 1);
    }
    const top = [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([type, n]) => `${type} ${n}`)
      .join(", ");
    return [g.backend, g.model, String(total), String(passed),
            `${((100 * passed) / (total || 1)).toFixed(1)}%`, String(timeouts), top || "—"];
  });

  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => String(r[i]).length)));
  const line = (cells) => "│ " + cells.map((c, i) => String(c).padEnd(widths[i])).join(" │ ") + " │";
  console.log(`TinyMark report — js harness · level ${level} · ${results.length} checks`);
  console.log("┌" + widths.map((w) => "─".repeat(w + 2)).join("┬") + "┐");
  console.log(line(headers));
  console.log("├" + widths.map((w) => "─".repeat(w + 2)).join("┼") + "┤");
  for (const row of rows) console.log(line(row));
  console.log("└" + widths.map((w) => "─".repeat(w + 2)).join("┴") + "┘");
  console.log("");
}

const config = await loadLocalConfig();

const level = String(config.TMB_LEVEL || "med").toLowerCase();
const useAllTests = level === "large";
// --tests is a friendly alias for --problems: run one (or a few) tests individually.
const testSel = config.TMB_TESTS || config.TMB_PROBLEMS;
const problemIds = testSel
  ? String(testSel).split(",").map((s) => s.trim()).filter(Boolean)
  : (LEVELS[level] || LEVELS.med);

if (problemIds.length === 0) throw new Error("No problems to run.");
for (const id of problemIds) {
  if (!ALL_PROBLEMS[id]) throw new Error(`Unknown problem ${id}. Known: ${Object.keys(ALL_PROBLEMS).join(", ")}`);
}

const modelIds = config.TMB_MODELS
  ? String(config.TMB_MODELS).split(",").map((s) => s.trim()).filter(Boolean)
  : [config.TMB_MODEL];

if (modelIds.length === 0) throw new Error("No models to run.");

const modes = String(config.TMB_MODE).toLowerCase() === "both"
  ? ["direct", "ollama"]
  : [String(config.TMB_MODE).trim()].filter(Boolean).map((mode) => mode.toLowerCase());
const startedAt = new Date().toISOString();
const results = [];

console.table(
  problemIds.map((id) => ({
    problem: id,
    title: ALL_PROBLEMS[id].title,
    summary: ALL_PROBLEMS[id].summary,
    difficulty: ALL_PROBLEMS[id].difficulty,
  })),
);
console.table(
  modelIds.map((m) => ({
    model: m,
  })),
);
console.log("");
console.log("");

const runs = int(config.TMB_RUNS, 3);

for (const id of problemIds) {
  const problem = ALL_PROBLEMS[id];
  for (const model of modelIds) {
    const modelConfig = { ...config, TMB_MODEL: model };
    for (const mode of modes) {
      for (let runIndex = 1; runIndex <= runs; runIndex++) {
        let result;
        if (mode === "direct") {
          result = await directFixture(problem, modelConfig, useAllTests);
        } else if (mode === "ollama") {
          result = await ollama(problem, modelConfig, useAllTests);
        } else {
          result = { backend: mode, pass: false, elapsed_ms: 0, error: `unknown mode ${mode}`, problem: id, model };
        }
        result.run = runIndex;
        result.problem = id;
        result.model = model;
        result.failure_type = classifyNodeFailure(result);
        result.error_compact = compactError(result);
        results.push(result);
        statusLine(result);
      }
    }
  }
}

printResultTable(results);
printSummaryTable(results, level);
console.log("");

const output = {
  schema: "tiny-js-benchmark/v0",
  started_at: startedAt,
  finished_at: new Date().toISOString(),
  problems: problemIds.map((id) => ({
    id,
    title: ALL_PROBLEMS[id].title,
    summary: ALL_PROBLEMS[id].summary,
    difficulty: ALL_PROBLEMS[id].difficulty,
  })),
  models: modelIds,
  config: {
    mode: config.TMB_MODE,
    models: modelIds,
    ollama_url: normalizeUrl(config.TMB_OLLAMA_URL || config.OLLAMA_URL),
    warmup: bool(config.TMB_WARMUP),
    timeout_s: Math.max(...problemIds.map((id) => ALL_PROBLEMS[id].timeout_s)),
    num_predict: Number(config.TMB_NUM_PREDICT),
    think: bool(config.TMB_THINK),
    level,
    attempts: int(config.TMB_ATTEMPTS, 3),
  },
  results,
};
const outputPath = join(ROOT, config.TMB_RESULTS);
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, JSON.stringify(output, null, 2) + "\n", "utf8");
console.log(`results: ${outputPath}`);
console.log(`passed: ${results.filter((r) => r.pass).length}/${results.length}`);
