#!/usr/bin/env python3
"""Convert the T5Gemma ENCODER (google/t5gemma-b-b-ul2) to GGUF for sa3.cpp.

Reads the BF16 safetensors directly (no torch needed) and upcasts to F32.
Bakes Gemma's RMSNorm (1 + weight) into the norm tensors so the C++ graph can
use a plain rms_norm. Encoder-only; the decoder half is skipped.

Usage:
  python tools/convert_t5gemma.py --src <t5gemma/model.safetensors> \
                                  --config <t5gemma/config.json> \
                                  --out models/t5gemma-enc-f32.gguf
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np
from gguf import GGUFWriter
import gguf_meta

PREF = "model.encoder."


def read_safetensors_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n  # data starts after the header


def load_f32(f, data_start, meta):
    off0, off1 = meta["data_offsets"]
    f.seek(data_start + off0)
    raw = f.read(off1 - off0)
    dt = meta["dtype"]
    if dt == "F32":
        a = np.frombuffer(raw, dtype=np.float32)
    elif dt == "F16":
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        a = (u16 << 16).view(np.float32)
    else:
        raise ValueError(f"unhandled dtype {dt}")
    return np.ascontiguousarray(a.reshape(meta["shape"]))


# Gemma RMSNorm weights: stored as (weight); applied as (1 + weight). Bake the +1.
NORM_SUFFIXES = ("pre_self_attn_layernorm", "post_self_attn_layernorm",
                 "pre_feedforward_layernorm", "post_feedforward_layernorm")


def rename(k):
    r = k[len(PREF):]
    if r == "embed_tokens.weight": return "te.embed.weight", False
    if r == "norm.weight":         return "te.norm.weight", "norm"
    if r.startswith("layers."):
        i, rest = r[len("layers."):].split(".", 1)
        m = {
            "pre_self_attn_layernorm.weight":   ("attn_norm.weight",  "norm"),
            "post_self_attn_layernorm.weight":  ("attn_post_norm.weight", "norm"),
            "pre_feedforward_layernorm.weight": ("ffn_norm.weight",   "norm"),
            "post_feedforward_layernorm.weight":("ffn_post_norm.weight", "norm"),
            "self_attn.q_proj.weight": ("q.weight", False),
            "self_attn.k_proj.weight": ("k.weight", False),
            "self_attn.v_proj.weight": ("v.weight", False),
            "self_attn.o_proj.weight": ("o.weight", False),
            "mlp.gate_proj.weight": ("gate.weight", False),
            "mlp.up_proj.weight":   ("up.weight", False),
            "mlp.down_proj.weight": ("down.weight", False),
        }
        if rest in m:
            sub, kind = m[rest]
            return f"te.{i}.{sub}", kind
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())["encoder"]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="sa3-t5gemma")

    dim = cfg["hidden_size"]
    w.add_uint32("t5g.dim",          dim)
    w.add_uint32("t5g.layers",       cfg["num_hidden_layers"])
    w.add_uint32("t5g.heads",        cfg["num_attention_heads"])
    w.add_uint32("t5g.head_dim",     cfg["head_dim"])
    w.add_uint32("t5g.intermediate", cfg["intermediate_size"])
    w.add_uint32("t5g.vocab",        cfg["vocab_size"])
    w.add_float32("t5g.eps",         cfg["rms_norm_eps"])
    w.add_float32("t5g.rope_theta",  cfg["rope_theta"])
    w.add_float32("t5g.attn_softcap",cfg["attn_logit_softcapping"])
    w.add_float32("t5g.query_scalar",cfg["query_pre_attn_scalar"])
    w.add_float32("t5g.normalizer",  float(dim) ** 0.5)
    w.add_uint32("t5g.max_length",   256)      # prompt tokenization length (shared, encoder-side)

    # Pure encoder weights only. The per-variant conditioner (learned prompt padding +
    # seconds_total NumberConditioner) is a separate sidecar gguf — see convert_conditioner.py.
    n_written, nparams = 0, 0
    with open(args.src, "rb") as f:
        header, data_start = read_safetensors_header(f)
        for k in sorted(header):
            if not k.startswith(PREF):
                continue
            name, kind = rename(k)
            if name is None:
                continue
            t = load_f32(f, data_start, header[k]).astype(np.float32)
            if kind == "norm":
                t = t + 1.0    # bake Gemma's (1 + weight)
            w.add_tensor(name, np.ascontiguousarray(t))
            n_written += 1; nparams += t.size

    gguf_meta.add_general(w, basename="t5gemma-b-b-ul2-encoder",
                          name="t5gemma-b-b-ul2 encoder", n_params=nparams)   # shared: no finetune
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}  ({n_written} tensors)  dim={dim} layers={cfg['num_hidden_layers']} "
          f"heads={cfg['num_attention_heads']}x{cfg['head_dim']} softcap={cfg['attn_logit_softcapping']}")


if __name__ == "__main__":
    sys.exit(main())
