"""Summarize a completed OpenCL profile, optionally compare raw F32 encoder outputs.

Usage: python3 tools/unoq-profile-summary.py RUN_DIR [CPU_RUN_DIR]
Host finish time INCLUDES device execution; do not add those columns together.
"""
import array
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from collections import defaultdict


def floats(path):
    result = array.array("f")
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"partial float in {path}")
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path, nargs="?")
    parser.add_argument("--tensor", default="t5g_hidden.f32", help="raw F32 filename within each run")
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    root = args.run
    profile = root / "cl_profiling.csv"
    if profile.exists() and not args.no_profile:
        grouped = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        with profile.open(newline="") as stream:
            for row in csv.DictReader(stream):
                stats = grouped[row["kernel name"]]
                stats[0] += 1
                for i, column in enumerate(("exec duration (ms)", "host enqueue (ms)", "host finish (ms)"), 1):
                    stats[i] += float(row[column])
        print("count  device_ms  enqueue_ms  finish_ms(inclusive)  kernel")
        for kernel, stats in sorted(grouped.items(), key=lambda item: item[1][1], reverse=True):
            print(f"{stats[0]:5d} {stats[1]:10.3f} {stats[2]:11.3f} {stats[3]:12.3f}  {kernel}")
        print("totals:", [sum(row[i] for row in grouped.values()) for i in range(4)])
        events = json.loads((root / "cl_trace.json").read_text())
        print("valid trace JSON; events:", len(events))
    output = floats(root / args.tensor)
    bad = sum(not math.isfinite(x) for x in output)
    print("elements:", len(output), "nonfinite:", bad)
    if bad or not output:
        raise ValueError("missing/invalid output")
    if args.reference:
        reference = floats(args.reference / args.tensor)
        if len(reference) != len(output) or not all(math.isfinite(x) for x in reference):
            raise ValueError("incompatible reference")
        dot = sum(a*b for a, b in zip(output, reference))
        aa = sum(a*a for a in output)
        bb = sum(b*b for b in reference)
        rmse = math.sqrt(sum((a-b)**2 for a, b in zip(output, reference))/len(output))
        print("RMSE:", rmse, "cosine:", dot/math.sqrt(aa*bb) if aa and bb else None,
              "max_abs:", max(abs(a-b) for a, b in zip(output, reference)))


if __name__ == "__main__":
    main()
