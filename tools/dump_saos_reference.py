#!/usr/bin/env python3
"""Create a deterministic classic SAOS DiT + Oobleck parity fixture."""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import packaging
import torch
from safetensors import safe_open

# Older stable-audio-tools combinations expect this removed compatibility shim.
if "pkg_resources" not in sys.modules:
    stub = types.ModuleType("pkg_resources")
    stub.packaging = packaging
    sys.modules["pkg_resources"] = stub

from stable_audio_tools.models.autoencoders import OobleckDecoder
from stable_audio_tools.models.dit import DiffusionTransformer


def load_prefix(module, checkpoint, prefix, dtype):
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith(prefix)]
        state = {k[len(prefix):]: f.get_tensor(k).to(dtype=dtype) for k in keys}
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch for {prefix}: missing={missing}, unexpected={unexpected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditioning", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    diff = cfg["model"]["diffusion"]
    dtype = torch.float32 if a.dtype == "float32" else torch.float16
    model = DiffusionTransformer(**diff["config"],
                                 diffusion_objective=diff["diffusion_objective"])
    load_prefix(model, a.checkpoint, "model.model.", dtype)
    model = model.to(device=a.device, dtype=dtype).eval()

    cond_dim = int(diff["config"]["cond_token_dim"])
    cross_np = np.fromfile(a.conditioning + ".cross.f32", dtype=np.float32).reshape(-1, cond_dim)
    glob_np = np.fromfile(a.conditioning + ".global.f32", dtype=np.float32)
    cross = torch.from_numpy(cross_np).unsqueeze(0).to(a.device, dtype)
    glob = torch.from_numpy(glob_np).unsqueeze(0).to(a.device, dtype)
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    x0 = torch.randn(1, int(diff["config"]["io_channels"]), a.frames,
                     generator=gen, device=a.device, dtype=dtype)
    noises = [torch.randn(x0.shape, generator=gen, device=a.device, dtype=dtype)
              for _ in range(a.steps)]
    logsnr = torch.linspace(-6, 2, a.steps + 1, device=a.device, dtype=torch.float32)
    schedule = torch.sigmoid(-logsnr)
    schedule[0] = 1
    schedule[-1] = 0
    x = x0.clone()
    with torch.inference_mode():
        for i in range(a.steps):
            velocity = model(x, schedule[i:i+1], cross_attn_cond=cross,
                             global_embed=glob, cfg_scale=1.0)
            denoised = x - schedule[i] * velocity
            x = (1 - schedule[i + 1]) * denoised + schedule[i + 1] * noises[i]
    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(x0[0].float().cpu().numpy().T).tofile(str(prefix) + ".initial.f32")
    np.ascontiguousarray(np.stack([n[0].float().cpu().numpy().T for n in noises])).tofile(
        str(prefix) + ".step-noise.f32")
    np.ascontiguousarray(x[0].float().cpu().numpy().T).tofile(str(prefix) + ".latent.f32")
    del model, velocity, denoised, x0, noises
    torch.cuda.empty_cache()

    dec_cfg = cfg["model"]["pretransform"]["config"]["decoder"]["config"]
    decoder = OobleckDecoder(**dec_cfg)
    load_prefix(decoder, a.checkpoint, "pretransform.model.decoder.", dtype)
    decoder = decoder.to(device=a.device, dtype=dtype).eval()
    with torch.inference_mode():
        audio = decoder(x.to(a.device, dtype))
    np.ascontiguousarray(audio[0].float().cpu().numpy()).tofile(str(prefix) + ".audio.f32")
    Path(str(prefix) + ".json").write_text(json.dumps({
        "frames": a.frames, "steps": a.steps, "seed": a.seed, "dtype": a.dtype,
        "latent_shape": list(x.shape), "audio_shape": list(audio.shape)
    }, indent=2), encoding="utf-8")
    print(f"wrote {prefix}.* latent={tuple(x.shape)} audio={tuple(audio.shape)}")


if __name__ == "__main__":
    main()
