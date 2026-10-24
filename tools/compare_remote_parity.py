#!/usr/bin/env python3
"""Compare an sa3.cpp generation against a stable_audio_3 recording, stage by stage.

  --ref  directory written by tools/dump_remote_parity.py (torch layout .npy)
  --cpp  directory written by sa3-generate with SA3_PARITY_DUMP=<dir> (raw f32, time-major)
  --feed DIR   instead of comparing, write the reference's conditioning, per-step DiT inputs and
               final latent as sa3.cpp raw f32 into DIR, for SA3_PARITY_FEED=DIR runs
  --feed-only  NAMES  comma list of what to write with --feed (cross,global,steps,latent; default all)

Each stage prints the cosine similarity, the relative L2 error ||cpp-ref||/||ref|| and the max
absolute difference. With --wav-ref/--wav-cpp it also compares the decoded audio per frequency band.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
    return f"cos={cos:.6f}  rel={rel:.3e}  max|d|={np.abs(a - b).max():.3e}"


def load_cpp(d, name, shape_tm):
    p = Path(d) / name
    return np.fromfile(p, dtype=np.float32).reshape(shape_tm) if p.exists() else None


def bands(x, sr):
    """Energy (dB) per band of a [ch, n] signal."""
    edges = [0, 250, 1000, 4000, 8000, 12000, 16000, sr / 2]
    spec = np.abs(np.fft.rfft(x.astype(np.float64), axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / sr)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        out.append((lo, hi, 10 * np.log10(spec[..., m].sum() + 1e-30)))
    return out


def read_wav(p):
    import soundfile as sf
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    return x.T, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cpp")
    ap.add_argument("--feed")
    ap.add_argument("--feed-only", default="cross,global,steps,latent")
    ap.add_argument("--wav-ref")
    ap.add_argument("--wav-cpp")
    args = ap.parse_args()

    ref = Path(args.ref)
    R = lambda n: np.load(ref / n) if (ref / n).exists() else None
    noise0 = R("noise0.npy")
    io, T = noise0.shape
    steps = len([p for p in os.listdir(ref) if p.startswith("step") and p.endswith("_v.npy")])

    if args.feed:
        out = Path(args.feed)
        out.mkdir(parents=True, exist_ok=True)
        want = set(args.feed_only.split(","))
        w = lambda name, a: np.ascontiguousarray(a, dtype=np.float32).tofile(out / name)
        if "cross" in want: w("cross.f32", R("cross_attn_cond.npy"))
        if "global" in want: w("global.f32", R("global_cond.npy"))
        if "steps" in want:
            for i in range(steps):
                w(f"step{i}_x.f32", R(f"step{i}_x.npy").T)
        if "latent" in want: w("latent.f32", R("latent.npy").T)
        print(f"wrote {sorted(want)} for io={io} T={T} steps={steps} -> {out}")
        return 0

    if args.cpp:
        cpp = args.cpp
        cn = load_cpp(cpp, "noise0.f32", (-1, io))
        if cn is not None and cn.shape[0] != T:
            print(f"!! latent length differs: cpp T={cn.shape[0]} ref T={T}; per-element stages skipped")
            cs = load_cpp(cpp, "sigmas.f32", (-1,))
            print("sigmas ref", np.round(R("sigmas.npy").ravel(), 5), "\nsigmas cpp", np.round(cs, 5))
        else:
            cross = load_cpp(cpp, "cross.f32", (-1, R("cross_attn_cond.npy").shape[-1]))
            print(f"cross        {stats(cross, R('cross_attn_cond.npy'))}")
            ct = load_cpp(cpp, "cross.f32", (-1, 768))
            rt = R("cross_attn_cond.npy")
            print(f"  secs token {stats(ct[-1], rt[-1])}")
            print(f"global       {stats(load_cpp(cpp, 'global.f32', (-1,)), R('global_cond.npy'))}")
            print(f"noise0       {stats(cn.T, noise0)}")
            cs = load_cpp(cpp, "sigmas.f32", (-1,))
            rs = R("sigmas.npy").ravel()
            print(f"sigmas       max|d|={np.abs(cs - rs).max():.3e}  ref={np.round(rs, 5).tolist()}")
            for i in range(steps):
                cx, cv = load_cpp(cpp, f"step{i}_x.f32", (T, io)), load_cpp(cpp, f"step{i}_v.f32", (T, io))
                if cx is None:
                    break
                print(f"step {i} t={rs[i]:.4f}  x {stats(cx.T, R(f'step{i}_x.npy'))}   v {stats(cv.T, R(f'step{i}_v.npy'))}")
            cl = load_cpp(cpp, "latent.f32", (T, io))
            if cl is not None:
                print(f"latent       {stats(cl.T, R('latent.npy'))}")

    if args.wav_ref and args.wav_cpp:
        a, sr = read_wav(args.wav_cpp)
        b, _ = read_wav(args.wav_ref)
        n = min(a.shape[-1], b.shape[-1])
        a, b = a[:, :n], b[:, :n]
        print(f"audio        {stats(a, b)}  (n={n})")
        print("band energy dB   cpp      ref      cpp-ref   |diff|/ref")
        for (lo, hi, ea), (_, _, eb), (_, _, ed) in zip(bands(a, sr), bands(b, sr), bands(a - b, sr)):
            print(f"  {lo:>5.0f}-{hi:<5.0f}  {ea:8.2f} {eb:8.2f} {ea - eb:+8.2f}   {ed - eb:+8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
