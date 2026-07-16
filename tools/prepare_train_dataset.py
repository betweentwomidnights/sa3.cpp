#!/usr/bin/env python3
"""Arrange a flat folder of audio+caption pairs into sa3-train's manifest layout.

sa3-train expects `<out>/<split>/{filelist.txt, metadata.jsonl}` plus the audio (and a
same-stem .txt caption) reachable under each split. It requires distinct, non-contaminating
train/test/evaluation splits, so held-out splits are created (empty by default).

    python tools/prepare_train_dataset.py --input-dir "C:\\datasets\\my-audio" \
        --out "C:\\datasets\\my-training-set" --holdout 0

Requires only the standard library. Audio is copied (or symlinked with --symlink); ffmpeg in
sa3-train decodes whatever container you point it at (wav/mp3/...).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def find_pairs(input_dir: Path, caption_ext: str):
    pairs = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in AUDIO_EXTS:
            cap = p.with_suffix(caption_ext)
            pairs.append((p, cap if cap.exists() else None))
    return pairs


def write_split(out: Path, split: str, pairs, symlink: bool):
    split_dir = out / split
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    filelist, records = [], []
    for audio, cap in pairs:
        stem = audio.stem
        rel = f"audio/{audio.name}"
        dst = audio_dir / audio.name
        if not dst.exists():
            if symlink:
                os.symlink(audio, dst)
            else:
                shutil.copy2(audio, dst)
        # caption: same-stem .txt next to the audio (sa3-train derives it when metadata omits it)
        cap_dst = audio_dir / f"{stem}.txt"
        if cap is not None and not cap_dst.exists():
            shutil.copy2(cap, cap_dst)
        elif cap is None and not cap_dst.exists():
            cap_dst.write_text("", encoding="utf-8")  # empty caption placeholder
        filelist.append(rel)
        records.append({"id": stem, "split": split, "audio_path": rel})
    (split_dir / "filelist.txt").write_text("\n".join(filelist) + ("\n" if filelist else ""), encoding="utf-8")
    with open(split_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return len(filelist)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption-ext", default=".txt")
    ap.add_argument("--holdout", type=int, default=0, help="tracks moved to test/evaluation (split evenly)")
    ap.add_argument("--symlink", action="store_true", help="symlink audio instead of copying")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out = Path(args.out)
    pairs = find_pairs(input_dir, args.caption_ext)
    if not pairs:
        raise SystemExit(f"no audio files found in {input_dir}")

    h = max(0, args.holdout)
    test_pairs = pairs[:h // 2] if h else []
    eval_pairs = pairs[h // 2:h] if h else []
    train_pairs = pairs[h:] if h else pairs

    n_tr = write_split(out, "train", train_pairs, args.symlink)
    n_te = write_split(out, "test", test_pairs, args.symlink)
    n_ev = write_split(out, "evaluation", eval_pairs, args.symlink)
    print(f"wrote {n_tr} train / {n_te} test / {n_ev} evaluation tracks -> {out}")


if __name__ == "__main__":
    main()
