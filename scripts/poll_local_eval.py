#!/usr/bin/env python3
import subprocess
import time

while True:
    out = subprocess.check_output(
        ["python3", "scripts/local_eval.py", "status"],
        text=True,
    )
    print(out, end="\n-----\n")

    if "RUNNING" not in out:
        print("Done (not running).")
        subprocess.run(["python3", "scripts/local_eval.py", "result"], check=False)
        break

    time.sleep(10)
