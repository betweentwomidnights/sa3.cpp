#!/usr/bin/env python3
"""Dump deterministic PyTorch Oobleck decoder fixtures from a Stable Audio checkpoint.

Run this with an environment containing torch, stable-audio-tools and safetensors (the
gary-localhost stable-audio environment is suitable). It instantiates only the Oobleck
decoder; T5 and the diffusion model are not loaded.

The NPZ arrays use the layouts consumed/returned by src/sat/oobleck.h:
  latent: [frames, latent_channels]
  audio:  [samples, audio_channels]

Usage:
  services/stable-audio/env/Scripts/python.exe tools/dump_sat_oobleck_refs.py \
      --src model.safetensors --config model_config.json \
      --out refdata/saos_oobleck_f32.npz --frames 4 --seed 1234
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


PREFIX_CANDIDATES = ("pretransform.model.", "model.pretransform.model.", "")


def find_prefix(keys):
    marker = "decoder.layers.0.weight_v"
    matches = [p for p in PREFIX_CANDIDATES if p + marker in keys]
    if len(matches) != 1:
        raise ValueError(f"could not uniquely locate decoder; matching prefixes: {matches}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-dtype", choices=("float32", "float16"), default="float32")
    ap.add_argument("--taps", action="store_true",
                    help="also save each top-level decoder-layer output")
    args = ap.parse_args()
    if args.frames < 1:
        ap.error("--frames must be positive")

    try:
        import packaging
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        sys.exit(f"needs torch, safetensors, and packaging: {exc}")

    # Some stable-audio-tools dependency combinations import `packaging` through the
    # removed pkg_resources compatibility module. Supplying exactly that attribute keeps
    # fixture generation independent of the environment's setuptools version.
    if "pkg_resources" not in sys.modules:
        stub = types.ModuleType("pkg_resources")
        stub.packaging = packaging
        sys.modules["pkg_resources"] = stub
    try:
        from stable_audio_tools.models.autoencoders import OobleckDecoder
    except ImportError as exc:
        sys.exit(f"cannot import stable_audio_tools OobleckDecoder: {exc}")

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        dec = cfg["model"]["pretransform"]["config"]["decoder"]
    except (KeyError, TypeError) as exc:
        sys.exit(f"not a Stable Audio model config: {exc}")
    if dec.get("type") != "oobleck":
        sys.exit("configured decoder is not Oobleck")
    dec_cfg = dict(dec["config"])

    decoder = OobleckDecoder(**dec_cfg).to(args.device)
    dtype = torch.float32 if args.model_dtype == "float32" else torch.float16
    decoder = decoder.to(dtype=dtype).eval()

    with safe_open(args.src, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        prefix = find_prefix(keys)
        source_prefix = prefix + "decoder."
        state = {
            key[len(source_prefix):]: f.get_tensor(key).to(dtype=dtype)
            for key in sorted(keys)
            if key.startswith(source_prefix)
        }
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"decoder state mismatch: missing={missing}, unexpected={unexpected}")

    taps = {}
    hooks = []
    if args.taps:
        for index, layer in enumerate(decoder.layers):
            def capture(_module, _inputs, output, i=index):
                taps[f"layer_{i:02d}"] = output.detach().float().cpu().numpy()
            hooks.append(layer.register_forward_hook(capture))

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    latent = torch.randn(
        1, int(dec_cfg["latent_dim"]), args.frames,
        device=args.device, dtype=dtype,
    )
    with torch.no_grad():
        audio = decoder(latent)
    for hook in hooks:
        hook.remove()

    arrays = {
        "latent": latent[0].float().cpu().numpy().T,
        "audio": audio[0].float().cpu().numpy().T,
    }
    for name, value in taps.items():
        arrays[name] = value[0].transpose(1, 0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    manifest = {
        "source": str(Path(args.src).resolve()),
        "config": str(Path(args.config).resolve()),
        "source_prefix": prefix,
        "seed": args.seed,
        "frames": args.frames,
        "model_dtype": args.model_dtype,
        "device": args.device,
        "latent_shape": list(arrays["latent"].shape),
        "audio_shape": list(arrays["audio"].shape),
        "tap_names": sorted(taps),
    }
    out.with_suffix(out.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out} latent={arrays['latent'].shape} audio={arrays['audio'].shape}")
    print(f"wrote {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
