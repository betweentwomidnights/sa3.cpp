#!/usr/bin/env python3
"""Convert the Stable Audio 3 medium DiT to GGUF for sa3.cpp.

Reads the `model.model.*` tensors (the DiffusionTransformer) from the medium
checkpoint, squeezes the k=1 pre/post convs to matrices, and renames to compact
GGML names. Skips the inpaint `to_local_embed` (Phase 5) for now. All F32.

Usage:
  python tools/convert_dit.py --src <model.safetensors> --config <model_config.json> \
                              --out models/sa3-dit-f32.gguf
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter
import gguf_meta

PREF = "model.model."
TR   = "transformer."


def rename(k):
    r = k[len(PREF):]
    direct = {
        "preprocess_conv.weight":  "dit.pre_conv.weight",
        "postprocess_conv.weight": "dit.post_conv.weight",
        "to_cond_embed.0.weight":   "dit.cond_embed.0.weight",
        "to_cond_embed.2.weight":   "dit.cond_embed.2.weight",
        "to_global_embed.0.weight": "dit.global_embed.0.weight",
        "to_global_embed.2.weight": "dit.global_embed.2.weight",
        "to_timestep_embed.0.weight":"dit.time_embed.0.weight",
        "to_timestep_embed.0.bias":  "dit.time_embed.0.bias",
        "to_timestep_embed.2.weight":"dit.time_embed.2.weight",
        "to_timestep_embed.2.bias":  "dit.time_embed.2.bias",
        TR+"global_cond_embedder.0.weight": "dit.gce.0.weight",
        TR+"global_cond_embedder.0.bias":   "dit.gce.0.bias",
        TR+"global_cond_embedder.2.weight": "dit.gce.2.weight",
        TR+"global_cond_embedder.2.bias":   "dit.gce.2.bias",
        TR+"memory_tokens":   "dit.memory_tokens",
        TR+"project_in.weight":  "dit.proj_in.weight",
        TR+"project_out.weight": "dit.proj_out.weight",
    }
    if r in direct:
        return direct[r]
    if r == TR+"rotary_pos_emb.inv_freq":
        return None   # recomputed in C++
    if r.startswith(TR+"layers."):
        i, rest = r[len(TR+"layers."):].split(".", 1)
        m = {
            "pre_norm.gamma":          "pre_norm.gamma",
            "ff_norm.gamma":           "ff_norm.gamma",
            "cross_attend_norm.gamma": "cross_norm.gamma",
            "self_attn.to_qkv.weight": "self.qkv.weight",
            "self_attn.to_out.weight": "self.out.weight",
            "self_attn.q_norm.gamma":  "self.q_norm.gamma",
            "self_attn.k_norm.gamma":  "self.k_norm.gamma",
            "cross_attn.to_q.weight":  "cross.q.weight",
            "cross_attn.to_kv.weight": "cross.kv.weight",
            "cross_attn.to_out.weight":"cross.out.weight",
            "cross_attn.q_norm.gamma": "cross.q_norm.gamma",
            "cross_attn.k_norm.gamma": "cross.k_norm.gamma",
            "ff.ff.0.proj.weight":     "ff.proj.weight",
            "ff.ff.0.proj.bias":       "ff.proj.bias",
            "ff.ff.2.weight":          "ff.out.weight",
            "ff.ff.2.bias":            "ff.out.bias",
            "to_scale_shift_gate":     "ssg",
            # inpaint per-block local conditioning: Linear(257->dim) -> SiLU -> Linear(dim->dim)
            "to_local_embed.0.weight": "local.0.weight",
            "to_local_embed.0.bias":   "local.0.bias",
            "to_local_embed.2.weight": "local.2.weight",
            "to_local_embed.2.bias":   "local.2.bias",
        }
        if rest in m:
            return f"dit.{i}.{m[rest]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="medium", choices=list(gguf_meta.VARIANTS),
                    help="model family this DiT belongs to (sets general.* metadata)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())["model"]["diffusion"]["config"]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out), arch="sa3-dit")

    dim = cfg["embed_dim"]
    w.add_uint32("dit.io",        cfg["io_channels"])      # 256
    w.add_uint32("dit.dim",       dim)                     # 1536
    w.add_uint32("dit.depth",     cfg["depth"])            # 24
    w.add_uint32("dit.heads",     cfg["num_heads"])        # 24
    w.add_uint32("dit.head_dim",  dim // cfg["num_heads"]) # 64
    w.add_uint32("dit.cond_dim",  cfg["cond_token_dim"])   # 768
    w.add_uint32("dit.mem_tokens",cfg["num_memory_tokens"])# 64
    w.add_uint32("dit.rot",       (dim // cfg["num_heads"]) // 2)  # 32 (partial rope)
    w.add_uint32("dit.time_dim",  cfg.get("timestep_features_dim", 256))  # 256
    w.add_uint32("dit.differential", 1 if cfg.get("attn_kwargs", {}).get("differential", False) else 0)
    w.add_uint32("dit.local_dim", cfg.get("local_add_cond_dim", 0) or 0)   # 257 for inpaint, 0 if absent
    w.add_float32("dit.rope_base", 10000.0)
    w.add_float32("dit.norm_eps",  1e-5)   # block RMSNorm
    w.add_float32("dit.qk_eps",    1e-6)   # qk RMSNorm
    w.add_float32("dit.time_min_freq", 0.5)
    w.add_float32("dit.time_max_freq", 10000.0)

    n, skip, nparams = 0, 0, 0
    with safe_open(args.src, framework="numpy") as f:
        for k in sorted(f.keys()):
            if not k.startswith(PREF):
                continue
            name = rename(k)
            if name is None:
                skip += 1; continue
            t = np.ascontiguousarray(f.get_tensor(k).astype(np.float32))
            if name in ("dit.pre_conv.weight", "dit.post_conv.weight"):
                t = t[:, :, 0]   # [256,256,1] -> [256,256]
            if name == "dit.memory_tokens":
                t = np.ascontiguousarray(t)   # [64,1536]
            w.add_tensor(name, t)
            n += 1; nparams += t.size

    gguf_meta.add_general(w, basename=f"stable-audio-3-{args.variant}-dit",
                          name=f"stable-audio-3-{args.variant} DiT",
                          finetune=args.variant, n_params=nparams)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {out}  ({n} tensors, skipped {skip})  dim={dim} depth={cfg['depth']} "
          f"heads={cfg['num_heads']}x{dim//cfg['num_heads']} mem={cfg['num_memory_tokens']}")


if __name__ == "__main__":
    sys.exit(main())
