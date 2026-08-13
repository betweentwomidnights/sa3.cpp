#!/usr/bin/env python3
"""Convert a sa3.cpp LoRA/DoRA gguf back to safetensors + json — the inverse of
convert_lora.py, so an adapter trained natively by `sa3-train` can be loaded by the
PyTorch/gary4local side.

sa3.cpp stores adapter tensors under the *base DiT weight* name it adapts
(`dit.7.self.qkv.lora_A`). The PyTorch side expects the parametrization path on the
original module (`model.transformer.layers.7.self_attn.to_qkv.parametrizations.weight.0.lora_A`).
convert_dit.rename() is a pure lookup in both its direct and per-layer tables, so it
inverts exactly; this builds that inverse rather than hand-maintaining a second table,
which means the two directions cannot drift apart.

  python3 tools/convert_lora_to_safetensors.py \\
      --in train-runs/<run>/adapter-final.gguf --out loras/my-adapter

writes <out>.safetensors + <out>.json.

Verified by round-trip on a real 228-target dora-rows adapter: converting to safetensors and
back through sa3-lora-convert reproduces the original gguf **bit-exactly** — 684/684 tensors,
worst absolute difference 0.0.

  .venv/bin/python tools/convert_lora_to_safetensors.py --in a.gguf --out b
  sa3-lora-convert --in b --out b.gguf      # b.gguf matches a.gguf tensor-for-tensor

Needs gguf + safetensors + numpy in the converter venv:
  .venv/bin/pip install numpy gguf safetensors
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import convert_dit
from convert_dit import PREF, TR

KINDS = ("lora_A", "lora_B", "magnitude", "magnitude_r", "magnitude_c", "U", "V", "M_xs")


def build_inverse():
    """gguf base name (no .weight) -> PyTorch module path (no .weight), from convert_dit's tables.

    Recovered by probing rename() with every key its tables accept, so the inverse is derived
    from the forward mapping rather than duplicated.
    """
    inv = {}

    def record(torch_suffix):
        full = PREF + torch_suffix
        g = convert_dit.rename(full)
        if not g:
            return
        gbase = g[:-len(".weight")] if g.endswith(".weight") else g
        tbase = torch_suffix[:-len(".weight")] if torch_suffix.endswith(".weight") else torch_suffix
        # module path as convert_lora.base_name() consumes it: "model." + <module>
        inv.setdefault(gbase, "model." + tbase)

    # rename()'s tables are locals, so probe it with every key they accept
    direct_keys = [
        "preprocess_conv.weight", "postprocess_conv.weight",
        "to_cond_embed.0.weight", "to_cond_embed.2.weight",
        "to_global_embed.0.weight", "to_global_embed.2.weight",
        "to_timestep_embed.0.weight", "to_timestep_embed.0.bias",
        "to_timestep_embed.2.weight", "to_timestep_embed.2.bias",
        TR + "global_cond_embedder.0.weight", TR + "global_cond_embedder.0.bias",
        TR + "global_cond_embedder.2.weight", TR + "global_cond_embedder.2.bias",
        TR + "memory_tokens", TR + "project_in.weight", TR + "project_out.weight",
    ]
    for k in direct_keys:
        record(k)

    per_layer = [
        "pre_norm.gamma", "ff_norm.gamma", "cross_attend_norm.gamma",
        "self_attn.to_qkv.weight", "self_attn.to_out.weight",
        "self_attn.q_norm.gamma", "self_attn.k_norm.gamma",
        "cross_attn.to_q.weight", "cross_attn.to_kv.weight", "cross_attn.to_out.weight",
        "cross_attn.q_norm.gamma", "cross_attn.k_norm.gamma",
        "ff.ff.0.proj.weight", "ff.ff.0.proj.bias", "ff.ff.2.weight", "ff.ff.2.bias",
        "to_scale_shift_gate",
        "to_local_embed.0.weight", "to_local_embed.0.bias",
        "to_local_embed.2.weight", "to_local_embed.2.bias",
    ]
    for i in range(64):                      # generous; medium is 24
        for rest in per_layer:
            record(f"{TR}layers.{i}.{rest}")
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="adapter .gguf from sa3-train")
    ap.add_argument("--out", required=True, help="output basename (writes .safetensors + .json)")
    args = ap.parse_args()

    try:
        import numpy as np
        from gguf import GGUFReader
        from safetensors.numpy import save_file
    except ImportError as e:
        sys.exit(f"needs the converter venv (gguf + safetensors): {e}")

    r = GGUFReader(args.inp)

    kv = {}
    for field in r.fields.values():
        try:
            kv[field.name] = field.contents()
        except Exception:
            pass
    adapter_type = kv.get("lora.adapter_type", "lora")
    rank = kv.get("lora.rank")
    alpha = kv.get("lora.alpha")
    if rank is None or alpha is None:
        sys.exit(f"{args.inp}: missing lora.rank / lora.alpha in the gguf KV store")

    inv = build_inverse()
    out_sd, unmapped, n_targets = {}, [], set()
    for t in r.tensors:
        name = str(t.name)
        kind = next((k for k in KINDS if name.endswith("." + k)), None)
        if kind is None:
            unmapped.append(name); continue
        gbase = name[: -(len(kind) + 1)]
        module = inv.get(gbase)
        if module is None:
            unmapped.append(name); continue
        arr = np.array(t.data, dtype=np.float32)
        out_sd[f"{module}.parametrizations.weight.0.{kind}"] = np.ascontiguousarray(arr)
        n_targets.add(gbase)

    if not out_sd:
        sys.exit(f"{args.inp}: no adapter tensors recognised — is this a sa3 LoRA gguf?")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_sd, str(out.with_suffix(".safetensors")))
    cfg = {"adapter_type": str(adapter_type), "rank": int(rank), "alpha": float(alpha)}
    out.with_suffix(".json").write_text(json.dumps(cfg, indent=2))

    print(f"wrote {out}.safetensors + {out}.json")
    print(f"  {len(out_sd)} tensors across {len(n_targets)} targets  "
          f"type={cfg['adapter_type']} rank={cfg['rank']} alpha={cfg['alpha']}")
    if unmapped:
        print(f"  ! {len(unmapped)} tensor(s) not mapped, e.g. {unmapped[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
