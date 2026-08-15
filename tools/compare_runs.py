#!/usr/bin/env python3
"""Compare two sa3-train metrics.jsonl runs step by step.

Written for the quantized-base validation: a Q4_K_M run against a matched-seed
F16 control. Since both runs draw the same dataset order, timesteps and
inpainting masks from the seed, step N of one is directly comparable to step N of
the other, and a divergence in the *trajectory* -- not just the mean -- is what
would indicate the quantized backward is wrong rather than merely less precise.

Usage: compare_runs.py <run-dir-or-jsonl> <control-dir-or-jsonl> [--window N]
"""

import json
import os
import statistics
import sys


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(a, b):
    return float("nan") if b == 0 else (a - b) / b * 100.0


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 2
    window = 100
    for a in argv[1:]:
        if a.startswith("--window"):
            window = int(a.split("=", 1)[1]) if "=" in a else 100

    run, ctl = load(args[0]), load(args[1])
    n = min(len(run), len(ctl))
    if n == 0:
        print("no overlapping steps yet")
        return 1
    run, ctl = run[:n], ctl[:n]

    # The seed fixes the sample order, so a mismatch here means the two runs are
    # not actually comparable and every number below would be meaningless.
    drift = [i + 1 for i, (a, b) in enumerate(zip(run, ctl)) if a["id"] != b["id"]]
    print(f"steps compared: {n}  (run has {len(load(args[0]))}, control has {len(load(args[1]))})")
    if drift:
        print(f"!! sample order diverges at step(s) {drift[:5]}{'...' if len(drift) > 5 else ''}"
              f" -- these runs are NOT matched, treat the rest as noise")
    else:
        print("sample order: identical throughout (runs are matched)")

    print(f"\n{'window':>14}  {'loss run':>9} {'loss ctl':>9} {'delta':>8}   "
          f"{'gnorm run':>9} {'gnorm ctl':>9}   {'s/step run':>10} {'s/step ctl':>10}")
    for lo in range(0, n, window):
        hi = min(lo + window, n)
        r, c = run[lo:hi], ctl[lo:hi]
        rl = statistics.mean(x["loss"] for x in r)
        cl = statistics.mean(x["loss"] for x in c)
        print(f"{lo + 1:>6}-{hi:<7}  {rl:>9.4f} {cl:>9.4f} {pct(rl, cl):>+7.1f}%   "
              f"{statistics.mean(x['grad_norm'] for x in r):>9.4f} "
              f"{statistics.mean(x['grad_norm'] for x in c):>9.4f}   "
              f"{statistics.mean(x['step_s'] for x in r):>10.3f} "
              f"{statistics.mean(x['step_s'] for x in c):>10.3f}")

    rl = [x["loss"] for x in run]
    cl = [x["loss"] for x in ctl]
    print(f"\noverall loss   run {statistics.mean(rl):.4f}   control {statistics.mean(cl):.4f}"
          f"   {pct(statistics.mean(rl), statistics.mean(cl)):+.1f}%")

    # Per-step correlation separates "same trajectory, slightly offset" from
    # "different trajectory that happens to average out".
    if n > 2 and statistics.pstdev(rl) > 0 and statistics.pstdev(cl) > 0:
        mr, mc = statistics.mean(rl), statistics.mean(cl)
        cov = sum((a - mr) * (b - mc) for a, b in zip(rl, cl)) / n
        print(f"per-step loss correlation: {cov / (statistics.pstdev(rl) * statistics.pstdev(cl)):.5f}"
              f"   (near 1.0 = same trajectory, only precision differs)")

    steady = slice(min(100, n // 2), n)
    sr = statistics.mean(x["step_s"] for x in run[steady])
    sc = statistics.mean(x["step_s"] for x in ctl[steady])
    faster = pct(sc, sr)
    verdict = f"run {faster:+.1f}% faster" if faster >= 0 else f"run {-faster:.1f}% SLOWER"
    print(f"steady s/step  run {sr:.3f}   control {sc:.3f}   {verdict}")

    # Cross-run step times on a thermally-limited laptop are only meaningful if
    # each run's own step time is stable. Report the spread so a timing claim
    # cannot be read off two runs recorded hours apart under different load.
    def spread(rows):
        w = [statistics.mean(x["step_s"] for x in rows[lo:lo + window])
             for lo in range(0, len(rows), window)][1:]
        return (min(w), max(w)) if w else (float("nan"),) * 2

    rlo, rhi = spread(run)
    clo, chi = spread(ctl)
    print(f"  per-window range   run {rlo:.1f}-{rhi:.1f} s   control {clo:.1f}-{chi:.1f} s")
    if max(rhi - rlo, chi - clo) > 0.15 * min(sr, sc):
        print("  !! one or both runs swing more than the difference between them --")
        print("     treat this as inconclusive and re-measure back to back if it matters")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
