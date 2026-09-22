#!/usr/bin/env bash
set -euo pipefail

while true; do
  out="$(python3 scripts/local_eval.py status)"
  printf "%s\n" "$out"
  printf "-----\n"

  if ! printf "%s" "$out" | grep -q "RUNNING"; then
    printf "Done (not running).\n"
    python3 scripts/local_eval.py result
    break
  fi

  sleep 10
done
