#!/usr/bin/env node
/**
 * Local harness self-test for tiny_benchmark.js.
 *
 * This never calls actual models. It forces --mode=direct so only deterministic
 * fixture code paths run, then validates the result JSON structure and pass counts.
 *
 * Usage:
 *   node scripts/tiny_benchmark_selftest.js
 */

import { mkdir, readFile, rm } from "node:fs/promises";

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const ROOT = join(import.meta.dirname, "..");
const OUT_DIR = join(ROOT, ".eval", "selftest");

const PROFILES = [
  {
    name: "small",
    problems: ["tmb-slug", "tmb-kv-parse"],
    runs: 1,
    maxMs: 60_000,
  },
  {
    name: "medium",
    problems: ["tmb-slug", "tmb-kv-parse", "tmb-fix-index-delim", "tmb-merge-ranges"],
    runs: 2,
    maxMs: 90_000,
  },
  {
    name: "large",
    problems: ["tmb-jwt", "tmb-slug", "tmb-kv-parse", "tmb-fix-index-delim", "tmb-merge-ranges", "tmb-rpn-eval", "tmb-lru-ops"],
    runs: 3,
    maxMs: 180_000,
  },
];

async function runProfile(profile) {
  const outFile = join(OUT_DIR, `result-${profile.name}.json`);
  const started = Date.now();

  const args = [
    join("scripts", "tiny_benchmark.js"),
    "--mode=direct",
    "--models=fixture-local",
    `--runs=${profile.runs}`,
    `--problems=${profile.problems.join(",")}`,
    `--results=${join(".eval", "selftest", `result-${profile.name}.json`)}`,
    "--debug=false",
  ];

  const child = await execFileAsync("node", args, {
    cwd: ROOT,
    timeout: profile.maxMs,
    maxBuffer: 1024 * 1024,
  });

  const elapsed = Date.now() - started;
  const payload = JSON.parse(await readFile(outFile, "utf8"));

  assert.equal(payload.schema, "tiny-js-benchmark/v0", `${profile.name}: unexpected schema`);
  assert.ok(Array.isArray(payload.results), `${profile.name}: results must be an array`);

  const expectedCount = profile.problems.length * profile.runs; // 1 direct backend x 1 model
  assert.equal(payload.results.length, expectedCount, `${profile.name}: unexpected result count`);

  for (const row of payload.results) {
    assert.equal(row.backend, "direct-fixture", `${profile.name}: backend must be direct-fixture`);
    assert.equal(row.pass, true, `${profile.name}: expected pass=true for ${row.problem}`);
  }

  return {
    name: profile.name,
    elapsedMs: elapsed,
    expectedCount,
    stdoutTail: String(child.stdout || "").split(/\r?\n/).slice(-3).join("\n"),
  };
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const summaries = [];

  for (const profile of PROFILES) {
    const summary = await runProfile(profile);
    summaries.push(summary);
    console.log(`${profile.name}: ok (${summary.expectedCount} cases, ${summary.elapsedMs}ms)`);
  }

  console.log("all self-tests passed");

  // Keep artifacts in .eval/selftest for inspection. Remove this block if you
  // prefer to retain all JSON outputs by default.
  if (process.env.TMB_SELFTEST_CLEANUP === "1") {
    await rm(OUT_DIR, { recursive: true, force: true });
    console.log("cleaned .eval/selftest");
  }
}

main().catch((error) => {
  console.error("self-test failed:", error?.message || error);
  process.exit(1);
});
