#!/usr/bin/env python3
"""Create a deterministic V-prediction SAT DiT/Oobleck parity fixture."""

import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import packaging
import torch
from safetensors import safe_open

if "pkg_resources" not in sys.modules:
    stub = types.ModuleType("pkg_resources")
    stub.packaging = packaging
    sys.modules["pkg_resources"] = stub

from stable_audio_tools.models.autoencoders import OobleckDecoder
from stable_audio_tools.models.dit import DiffusionTransformer


def load_prefix(module, checkpoint, prefix, dtype):
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        keys = [key for key in source.keys() if key.startswith(prefix)]
        state = {key[len(prefix):]: source.get_tensor(key).to(dtype=dtype) for key in keys}
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state mismatch for {prefix}: missing={missing}, unexpected={unexpected}")


def dpmpp_3m_sde_step(x, denoised, noise, sigma, sigma_next, history, eta=1.0):
    denoised_1, denoised_2, h_1, h_2 = history
    h = h_1
    if sigma_next == 0:
        x = denoised
    else:
        t, s = -sigma.log(), -sigma_next.log()
        h = s - t
        h_eta = h * (eta + 1)
        x = (-h_eta).exp() * x + (-h_eta).expm1().neg() * denoised
        if h_2 is not None:
            r0, r1 = h_1 / h, h_2 / h
            d1_0 = (denoised - denoised_1) / r0
            d1_1 = (denoised_1 - denoised_2) / r1
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            phi_2 = (-h_eta).expm1() / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            x = x + phi_2 * d1 - phi_3 * d2
        elif h_1 is not None:
            r = h_1 / h
            d = (denoised - denoised_1) / r
            phi_2 = (-h_eta).expm1() / h_eta + 1
            x = x + phi_2 * d
        if eta:
            scale = sigma_next * (-2 * h * eta).expm1().neg().sqrt()
            x = x + noise * scale
    return x, (denoised, denoised_1, h, h_1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--conditioning", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--sigma-min", type=float, default=0.5)
    parser.add_argument("--sigma-max", type=float, default=50.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    diffusion = config["model"]["diffusion"]
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model = DiffusionTransformer(**diffusion["config"])
    load_prefix(model, args.checkpoint, "model.model.", dtype)
    model = model.to(device=args.device, dtype=dtype).eval()

    cond_dim = int(diffusion["config"]["cond_token_dim"])
    cross_np = np.fromfile(args.conditioning + ".cross.f32", dtype=np.float32)
    cross = torch.from_numpy(cross_np.reshape(-1, cond_dim)).unsqueeze(0).to(args.device, dtype)
    global_np = np.fromfile(args.conditioning + ".global.f32", dtype=np.float32)
    global_cond = torch.from_numpy(global_np).unsqueeze(0).to(args.device, dtype)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    shape = (1, int(diffusion["config"]["io_channels"]), args.frames)
    initial = torch.randn(shape, generator=generator, device=args.device, dtype=dtype)
    noises = [torch.randn(shape, generator=generator, device=args.device, dtype=dtype)
              for _ in range(args.steps)]
    ramp = torch.linspace(1, 0, args.steps, device=args.device, dtype=torch.float32) ** args.rho
    sigmas = torch.exp(ramp * (math.log(args.sigma_max) - math.log(args.sigma_min))
                       + math.log(args.sigma_min))
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

    x = initial * sigmas[0]
    history = (None, None, None, None)
    with torch.inference_mode():
        for index in range(args.steps):
            sigma, sigma_next = sigmas[index], sigmas[index + 1]
            c_skip = 1 / (sigma ** 2 + 1)
            c_out = -sigma / (sigma ** 2 + 1).sqrt()
            c_in = 1 / (sigma ** 2 + 1).sqrt()
            timestep = (sigma.atan() / math.pi * 2).to(dtype)
            velocity = model(x * c_in, timestep[None], cross_attn_cond=cross,
                             global_embed=global_cond, cfg_scale=1.0)
            denoised = velocity * c_out + x * c_skip
            x, history = dpmpp_3m_sde_step(
                x, denoised, noises[index], sigma, sigma_next, history, args.eta)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(initial[0].float().cpu().numpy().T).tofile(
        str(prefix) + ".initial.f32")
    np.ascontiguousarray(np.stack([
        noise[0].float().cpu().numpy().T for noise in noises])).tofile(
            str(prefix) + ".step-noise.f32")
    np.ascontiguousarray(x[0].float().cpu().numpy().T).tofile(
        str(prefix) + ".latent.f32")

    decoder_config = config["model"]["pretransform"]["config"]["decoder"]["config"]
    del model, velocity, denoised
    torch.cuda.empty_cache()
    decoder = OobleckDecoder(**decoder_config)
    load_prefix(decoder, args.checkpoint, "pretransform.model.decoder.", dtype)
    decoder = decoder.to(device=args.device, dtype=dtype).eval()
    with torch.inference_mode():
        audio = decoder(x.to(args.device, dtype))
    np.ascontiguousarray(audio[0].float().cpu().numpy()).tofile(str(prefix) + ".audio.f32")
    Path(str(prefix) + ".json").write_text(json.dumps({
        "frames": args.frames, "steps": args.steps, "seed": args.seed,
        "sigma_min": args.sigma_min, "sigma_max": args.sigma_max,
        "rho": args.rho, "eta": args.eta, "dtype": args.dtype,
        "latent_shape": list(x.shape), "audio_shape": list(audio.shape),
    }, indent=2), encoding="utf-8")
    print(f"wrote {prefix}.* latent={tuple(x.shape)} audio={tuple(audio.shape)}")


if __name__ == "__main__":
    main()
