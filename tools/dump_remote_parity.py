#!/usr/bin/env python3
"""Record a stable_audio_3 generation exactly as the gary4local SA3 service runs it.

The service (gary-localhost-installer .../services/sa3/api.py) loads the medium model with
model_half=True, attaches user adapters through load_lora + set_lora_strength (functional torch
parametrizations, not merged), and calls StableAudioModel.generate with the pingpong sampler.
This script makes the same calls and records every stage, so sa3.cpp can be compared against it
stage by stage (tools/compare_remote_parity.py):

  noise0.npy        [io, T]   the initial torch.randn draw (CUDA philox)
  sigmas.npy        [steps+1] the schedule the sampler actually used
  step{i}_x.npy     [io, T]   sampler state fed to the DiT at step i
  step{i}_v.npy     [io, T]   DiT output at step i (CFG-combined)
  step{i}_noise.npy [io, T]   the randn_like draw of step i
  <kwarg>.npy               every tensor the DiT received at step 0 (cross_attn_cond,
                            cross_attn_mask, global_cond, local_add_cond, padding_mask)
  latent.npy        [io, T]   sampler output
  audio.npy         [2, S]    decoded audio (clamped, truncated to the duration), when decoding

Run with a CUDA PyTorch env, pointing --sa3-pkg at the directory that holds the service's
stable_audio_3 package:
  python tools/dump_remote_parity.py --hf-home <hub root> --sa3-pkg <...>/services/sa3 \
      --lora loras/koan/koan.safetensors --prompt "..." --duration 10 --seed 42 --out parity/koan
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", required=True, help="HF_HOME holding stabilityai/stable-audio-3-medium")
    ap.add_argument("--sa3-pkg", required=True, help="directory containing the stable_audio_3 package to use")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--fp32", action="store_true", help="model_half=False (the service default is half)")
    ap.add_argument("--lora", action="append", default=[], help="user adapter (.safetensors/.ckpt); repeatable")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--decoder-lora", default=None, help="merged into the decoder like SA3_USE_DECODER_LORA")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--no-padding-mask", action="store_true",
                    help="diagnostic: withhold the attention padding mask from the DiT (sa3.cpp has none)")
    ap.add_argument("--no-clamp", action="store_true",
                    help="keep the decode unclamped, as the service's loudness path measures its peak")
    ap.add_argument("--decode-latent", default=None,
                    help="skip sampling: decode this latent.npy [io, T] and write audio.npy / torch.wav")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, args.sa3_pkg)
    import torch
    torch.set_grad_enabled(False)
    import stable_audio_3
    import stable_audio_3.inference.sampling as sampling
    from stable_audio_3.model import StableAudioModel
    print(f"stable_audio_3 from {Path(stable_audio_3.__file__).parent}")

    # decode-only keeps the DiT on the CPU so an fp32 autoencoder fits beside nothing else
    pipe = StableAudioModel.from_pretrained(args.model, model_half=not args.fp32,
                                            device="cpu" if args.decode_latent else None)
    if args.decode_latent and torch.cuda.is_available():
        pipe.model.pretransform.to("cuda")
    if args.decoder_lora:
        from stable_audio_3.models.lora import merge_lora
        pipe.load_lora([args.decoder_lora])
        merge_lora(pipe.model.pretransform.model.decoder)
        pipe.model.lora_names = []
        pipe.model.use_lora = False
        print(f"merged decoder adapter {args.decoder_lora}")
    lora_configs = None
    if args.lora:
        pipe.load_lora(args.lora)
        for i in range(len(args.lora)):
            pipe.set_lora_strength(args.lora_strength, lora_index=i)
        lora_configs = [{"lora_index": i, "interval": (0.0, 1.0), "layer_filter": ""}
                        for i in range(len(args.lora))]
        print(f"adapters {args.lora} at strength {args.lora_strength}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, t):
        np.save(out / name, t.detach().float().cpu().numpy())

    def decode(latents):
        pt = pipe.model.pretransform
        z = latents.to(device=next(pt.parameters()).device, dtype=next(pt.parameters()).dtype)
        audio = pt.decode(z).float()
        if not args.no_clamp:
            audio = audio.clamp(-1, 1)
        audio = audio[:, :, : int(args.duration * pipe.model.sample_rate)]
        save("audio.npy", audio[0])
        import soundfile as sf
        sf.write(out / "torch.wav", audio[0].cpu().numpy().T, pipe.model.sample_rate, subtype="FLOAT")

    if args.decode_latent:
        decode(torch.from_numpy(np.load(args.decode_latent))[None])
        print(f"decoded {args.decode_latent} -> {out / 'torch.wav'}")
        return 0

    rec = {"steps": 0}
    real_pingpong = sampling.sample_flow_pingpong
    real_randn_like = torch.randn_like

    def pingpong(model, x, sigmas, **kw):
        save("noise0.npy", x[0])
        save("sigmas.npy", sigmas)

        def wrapped(xi, t, **mkw):
            i = rec["steps"]
            if args.no_padding_mask:
                mkw = {k: v for k, v in mkw.items() if k != "padding_mask"}
            v = model(xi, t, **mkw)
            save(f"step{i}_x.npy", xi[0])
            save(f"step{i}_v.npy", v[0])
            if i == 0:
                for key, val in mkw.items():
                    if isinstance(val, torch.Tensor):
                        save(f"{key}.npy", val[0])
                rec["model_kwargs"] = sorted(k for k, val in mkw.items() if val is not None)
            rec["steps"] = i + 1
            return v

        def randn_like(t, *a, **k):
            r = real_randn_like(t, *a, **k)
            save(f"step{rec['steps'] - 1}_noise.npy", r[0])
            return r

        torch.randn_like = randn_like
        try:
            return real_pingpong(wrapped, x, sigmas, **kw)
        finally:
            torch.randn_like = real_randn_like

    sampling.sample_flow_pingpong = pingpong
    gen_kwargs = dict(prompt=args.prompt, negative_prompt=args.negative_prompt, duration=args.duration,
                      steps=args.steps, cfg_scale=args.cfg, seed=args.seed, sampler_type="pingpong",
                      return_latents=True)
    if lora_configs is not None:
        gen_kwargs["lora_configs"] = lora_configs
    latents = pipe.generate(**gen_kwargs)
    sampling.sample_flow_pingpong = real_pingpong
    save("latent.npy", latents[0])

    if not args.no_decode:
        decode(latents)

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    (out / "manifest.json").write_text(json.dumps({
        "prompt": args.prompt, "negative_prompt": args.negative_prompt, "duration": args.duration,
        "steps": args.steps, "cfg": args.cfg, "seed": args.seed, "half": not args.fp32,
        "loras": args.lora, "lora_strength": args.lora_strength, "decoder_lora": args.decoder_lora,
        "latent_shape": list(latents.shape[1:]), "model_kwargs": rec.get("model_kwargs"),
        "cuda": props.name if props else None,
        # what sa3-generate --rng-cuda-blocks needs to reproduce this GPU's noise
        "rng_cuda_blocks": props.multi_processor_count * (props.max_threads_per_multi_processor // 256)
                           if props else None,
    }, indent=2))
    print(f"recorded {rec['steps']} steps, latent {tuple(latents.shape)} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
