#!/usr/bin/env python3
"""Convert the Stable Audio 3 per-variant conditioner to a small sidecar GGUF for sa3.cpp.

The conditioner = the learned prompt padding embedding + the seconds_total NumberConditioner
(embedder Linear). These are trained per model variant, so — unlike the frozen/shared t5gemma
encoder — they ship as a tiny sidecar gguf alongside it. Loaded by sa3-generate via --cond.
See docs/DISTRIBUTION.md.

Usage:
  python tools/convert_conditioner.py --src <model.safetensors> --config <model_config.json> \
                                      --variant medium --out models/stable-audio-3-medium-conditioner-v1.0-F32.gguf
"""
import argparse, json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter
import gguf_meta

CMAP = {
    "conditioner.conditioners.prompt.padding_embedding": "te.padding_embedding",
    "conditioner.conditioners.seconds_total.embedder.embedding.1.weight": "te.secs.weight",
    "conditioner.conditioners.seconds_total.embedder.embedding.1.bias":   "te.secs.bias",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="model.safetensors (for conditioner.* tensors)")
    ap.add_argument("--config", required=True, help="model_config.json (seconds_total min/max)")
    ap.add_argument("--variant", default="medium", choices=list(gguf_meta.VARIANTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sa3cfg = json.loads(Path(args.config).read_text())
    secs = {}
    for c in sa3cfg.get("model", {}).get("conditioning", {}).get("configs", []):
        if c["id"] == "seconds_total":
            secs = c["config"]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="sa3-conditioner")
    w.add_float32("t5g.secs_min", float(secs.get("min_val", 0)))
    w.add_float32("t5g.secs_max", float(secs.get("max_val", 384)))
    w.add_uint32("t5g.secs_dim", 256)        # NumberEmbedder default dim

    n = 0
    with safe_open(args.src, framework="numpy") as f:
        for src_name, dst in CMAP.items():
            w.add_tensor(dst, np.ascontiguousarray(f.get_tensor(src_name).astype(np.float32)))
            n += 1

    gguf_meta.add_general(w, basename=f"stable-audio-3-{args.variant}-conditioner",
                          name=f"stable-audio-3-{args.variant} conditioner",
                          finetune=args.variant)   # tiny sidecar: size_label-exempt
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}  ({n} tensors)  secs=[{secs.get('min_val', 0)},{secs.get('max_val', 384)}]")


if __name__ == "__main__":
    main()
