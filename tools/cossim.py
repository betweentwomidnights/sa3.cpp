#!/usr/bin/env python3
"""Compare C++ raw-f32 decoder checkpoints against the PyTorch .npy references.

The C++ tool writes each checkpoint in GGML memory order (ne0 fastest). For each
checkpoint we transpose the PyTorch reference into that same order, then report
cosine similarity + max abs error.

Usage:
  python tools/cossim.py --ref refdata --cpp <dir-with-*.f32>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

# checkpoint -> how to turn the pytorch npy into the C++ ggml ravel order.
# value is the np.transpose axes to apply to the squeezed (batch-dropped) array.
# ggml ne=[a,b,..] memory == numpy shape [..,b,a] C-order.
LAYOUT = {
    "after_in_proj":   (0, 1),   # pt [T,1536] ; ggml [1536,T] -> ravel (t,d) == pt C-order. no transpose
    "after_resampling":(1, 0),   # pt [512,128]; ggml [512,128](ne0=ch) ravel (t,ch) -> transpose pt to [128,512]
    "audio":           (0, 1),   # pt [2,32768] ; ggml [32768,2] ravel (ch,t) == pt C-order. no transpose
}


def load_ref(p):
    a = np.load(p).astype(np.float64)
    if a.shape[0] == 1:        # drop batch
        a = a[0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="refdata")
    ap.add_argument("--cpp", required=True)
    args = ap.parse_args()
    ref_dir, cpp_dir = Path(args.ref), Path(args.cpp)

    worst = 1.0
    for name, axes in LAYOUT.items():
        rp, cp = ref_dir / f"{name}.npy", cpp_dir / f"{name}.f32"
        if not cp.exists():
            print(f"  {name:18s}  (no C++ dump, skipped)")
            continue
        ref = load_ref(rp)
        ref = np.transpose(ref, axes).ravel()
        cpp = np.fromfile(cp, dtype=np.float32).astype(np.float64)
        if ref.size != cpp.size:
            print(f"  {name:18s}  SIZE MISMATCH ref={ref.size} cpp={cpp.size}")
            worst = 0.0
            continue
        cos = float(np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-20))
        maxerr = float(np.max(np.abs(ref - cpp)))
        flag = "OK " if cos > 0.999 else "!! "
        print(f"  {flag}{name:18s} cossim={cos:.6f}  max_abs_err={maxerr:.3e}  n={ref.size}")
        worst = min(worst, cos)

    print(f"\nworst cossim: {worst:.6f}  ->  {'PASS' if worst > 0.999 else 'FAIL'}")
    return 0 if worst > 0.999 else 1


if __name__ == "__main__":
    sys.exit(main())
