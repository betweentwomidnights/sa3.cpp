#!/usr/bin/env python3
"""Dump a DiT forward reference WITH a LoRA/DoRA applied, for validating the C++ apply pass.

Builds the full wrapper (so the official loader's name resolution works), applies the
adapter via the real `load_and_apply_loras` (default strength 1.0), then runs one DiT
forward on random inputs (same layout as dump_dit.py). Conditioner LoRA is inert here
because cross/global are fed directly as random tensors.

Run with the SA3 venv:
  .../env/Scripts/python.exe tools/dump_lora_dit.py --config <cfg> --src <model.safetensors> \
      --hf_home <...> --lora loras/kev.ckpt --out refdata_lora --frames 32 --t 0.5 --seed 0
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--hf_home", required=True)
    ap.add_argument("--lora", required=True, help=".ckpt adapter to apply")
    ap.add_argument("--out", default="refdata_lora")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    import torch
    torch.set_grad_enabled(False)
    from safetensors import safe_open
    import stable_audio_3.models.transformer as _T
    for s in ("flash_attn_func","flash_attn_kvpacked_func","flash_attn_varlen_func",
              "index_first_axis","pad_input","unpad_input"):
        setattr(_T, s, None)
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.models.lora.loader import load_and_apply_loras
    from stable_audio_3.models.lora.model import set_lora_strength

    config = json.loads(Path(args.config).read_text())
    wrapper = create_diffusion_cond_from_config(config).eval()
    sd = {}
    with safe_open(args.src, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).float()
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    wrapper.float()
    print(f"loaded wrapper | missing={len(missing)} unexpected={len(unexpected)}")

    load_and_apply_loras(wrapper, [args.lora], config["model_type"])
    set_lora_strength(wrapper.model.model, args.strength)
    print(f"applied lora {args.lora} at strength {args.strength}")

    dit = wrapper.model.model
    dmc = config["model"]["diffusion"]["config"]
    T, cond_dim, io = args.frames, dmc["cond_token_dim"], dmc["io_channels"]
    torch.manual_seed(args.seed)
    x = torch.randn(1, io, T); t = torch.tensor([args.t])
    cross = torch.randn(1, 257, cond_dim); glob = torch.randn(1, cond_dim)
    tfeat = dit.timestep_features(t[:, None])
    vel = dit(x, t, cross_attn_cond=cross, global_embed=glob, cfg_scale=1.0)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(x[0].numpy().T).tofile(out / "dit_x.f32")
    np.ascontiguousarray(tfeat[0].numpy()).tofile(out / "dit_tfeat.f32")
    np.ascontiguousarray(cross[0].numpy()).tofile(out / "dit_cross.f32")
    np.ascontiguousarray(glob[0].numpy()).tofile(out / "dit_global.f32")
    np.save(out / "dit_vel.npy", vel[0].numpy())
    print(f"velocity {tuple(vel.shape)} range [{vel.min():.3f},{vel.max():.3f}] -> {out}")


if __name__ == "__main__":
    sys.exit(main())
