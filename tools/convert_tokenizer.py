#!/usr/bin/env python3
"""Bake the Gemma byte-fallback BPE tokenizer (tokenizer.json) into a GGUF.

Extracts the id-ordered vocab and the ranked merge list so the C++ tokenizer
needs no protobuf / sentencepiece at runtime. Matches the HF fast tokenizer that
produced our reference token ids.

Usage:
  python tools/convert_tokenizer.py --src <t5gemma/tokenizer.json> --out models/t5gemma-tok.gguf
"""
import argparse, json, sys
from pathlib import Path
from gguf import GGUFWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tj = json.loads(Path(args.src).read_text(encoding="utf-8"))
    model = tj["model"]
    assert model["type"] == "BPE", model["type"]
    vocab = model["vocab"]                 # piece -> id
    n = max(vocab.values()) + 1
    tokens = [""] * n
    for piece, idx in vocab.items():
        tokens[idx] = piece
    # added_tokens (specials) may not all be in vocab
    for a in tj.get("added_tokens", []):
        if a["id"] < n:
            tokens[a["id"]] = a["content"]

    merges = model["merges"]
    merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in merges]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="sa3-tokenizer")
    w.add_name("t5gemma-b-b-ul2 tokenizer (Gemma BPE)")
    w.add_array("tok.tokens", tokens)
    w.add_array("tok.merges", merges)
    w.add_uint32("tok.bos_id", vocab.get("<bos>", 2))
    w.add_uint32("tok.eos_id", vocab.get("<eos>", 1))
    w.add_uint32("tok.pad_id", vocab.get("<pad>", 0))
    w.add_uint32("tok.unk_id", vocab.get("<unk>", 3))
    w.add_bool("tok.byte_fallback", bool(model.get("byte_fallback", True)))
    w.add_bool("tok.add_bos", False)       # Gemma t5gemma tokenizer_config: add_bos_token=False
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}  ({n} tokens, {len(merges)} merges)")


if __name__ == "__main__":
    sys.exit(main())
