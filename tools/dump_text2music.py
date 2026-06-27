#!/usr/bin/env python3
"""End-to-end text2music reference for the A/B test.

Builds the full SA3 medium wrapper (DiT + SAME + T5Gemma conditioner), computes
real conditioning for a prompt+duration, runs a controlled manual ping-pong loop
(no padding mask, to match the GGML DiT) with pre-generated noise + schedule, and
decodes to a WAV. Dumps everything GGML needs to replay the exact same generation.

Run with the SA3 venv:
  .../services/sa3/env/Scripts/python.exe tools/dump_text2music.py \
      --config <model_config.json> --src <model.safetensors> \
      --hf_home <...Gary4LocalTest/hf-download-hotfix> --out refdata \
      --prompt "..." --frames 64 --steps 8 --seed 0
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--hf_home", required=True, help="HF cache root holding the medium repo (for t5gemma)")
    ap.add_argument("--out", default="refdata")
    ap.add_argument("--prompt", default="Upbeat funk groove with slap bass, bright horns, tight drums")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-wav", default=None, help="audio2audio: init WAV to encode + vary from")
    ap.add_argument("--init-noise-level", type=float, default=1.0, help="sigma_max for audio2audio")
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
    from stable_audio_3.inference.sampling import build_schedule

    dev = torch.device("cpu")
    config = json.loads(Path(args.config).read_text())
    wrapper = create_diffusion_cond_from_config(config).to(dev).eval()
    sd = {}
    with safe_open(args.src, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).float()
    missing, unexpected = wrapper.load_state_dict(sd, strict=False)
    print(f"loaded wrapper | missing={len(missing)} unexpected={len(unexpected)}")
    wrapper.float()

    sr = config["sample_rate"]; ds = config["model"]["pretransform"]["config"]["downsampling_ratio"]
    T = args.frames
    secs = T * ds / sr
    print(f"prompt={args.prompt!r}  frames={T}  ~{secs:.2f}s  steps={args.steps}")

    # --- real conditioning (assembled directly; skip the inpaint local-cond path) ---
    cond_t = wrapper.conditioner([{"prompt": args.prompt, "seconds_total": secs}], dev)
    cross = torch.cat([cond_t["prompt"][0], cond_t["seconds_total"][0]], dim=1)  # [1, 257, 768]
    glob  = cond_t["seconds_total"][0].squeeze(1)                                # [1, 768]
    print("cross_attn_cond", tuple(cross.shape), "global_embed", tuple(glob.shape))

    # --- audio2audio: encode the init WAV to z_init (= pretransform.encode, same as inference) ---
    sigma_max = args.init_noise_level if args.init_wav else 1.0
    z_init = None
    if args.init_wav:
        import torchaudio
        wav, in_sr = torchaudio.load(args.init_wav)            # [ch, samples]
        assert in_sr == sr, f"init WAV is {in_sr} Hz, expected {sr}"
        if wav.shape[0] == 1: wav = wav.repeat(2, 1)
        Lreq = T * ds
        wav = wav[:, :Lreq] if wav.shape[1] >= Lreq else torch.nn.functional.pad(wav, (0, Lreq - wav.shape[1]))
        z_init = wrapper.pretransform.encode(wav.unsqueeze(0).float())   # [1, io, T]
        print(f"init z_init {tuple(z_init.shape)} range [{z_init.min():.3f},{z_init.max():.3f}]  sigma_max={sigma_max}")

    # --- schedule (post dist-shift); sigma_max < 1 for audio2audio ---
    sigmas = build_schedule(steps=args.steps, sigma_max=sigma_max, dist_shift=wrapper.sampling_dist_shift,
                            fallback_seq_len=T, include_endpoint=True, device=dev).float()  # [steps+1]
    print("sigmas:", [round(float(s), 4) for s in sigmas])

    # --- controlled ping-pong (pre-generated noise, no padding mask) ---
    g = torch.Generator(device=dev).manual_seed(args.seed)
    io = config["model"]["io_channels"]
    noise0 = torch.randn(1, io, T, generator=g)
    if z_init is not None:                                      # mix: init*(1-sigma_max) + noise*sigma_max
        noise0 = z_init * (1 - sigma_max) + noise0 * sigma_max
    step_noise = [torch.randn(1, io, T, generator=g) for _ in range(args.steps)]
    dit = wrapper.model.model
    x = noise0.clone()
    with torch.no_grad():
        for i in range(args.steps):
            v = dit(x, sigmas[i].expand(1), cross_attn_cond=cross, global_embed=glob, cfg_scale=1.0)
            denoised = x - sigmas[i] * v
            x = (1 - sigmas[i+1]) * denoised + sigmas[i+1] * step_noise[i]
        latent = x
        audio = wrapper.pretransform.decode(latent).clamp(-1, 1)   # [1, 2, T*ds]

    # --- dump everything for GGML replay ---
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(cross[0].numpy()).tofile(out / "tm_cross.f32")       # ggml [768,257]
    np.ascontiguousarray(glob[0].numpy()).tofile(out / "tm_global.f32")       # [768]
    np.ascontiguousarray(sigmas.numpy()).tofile(out / "tm_sigmas.f32")        # [steps+1]
    np.ascontiguousarray(noise0[0].numpy().T).tofile(out / "tm_noise0.f32")   # ggml [io,T]
    np.ascontiguousarray(np.stack([n[0].numpy().T for n in step_noise])).tofile(out / "tm_stepnoise.f32")  # [steps, io, T] each ggml [io,T]
    np.save(out / "tm_latent.npy", latent[0].numpy())                          # [io, T]

    # write the PyTorch WAV
    import torchaudio
    torchaudio.save(str(out / "tm_pytorch.wav"), audio[0], sr)
    print(f"latent {tuple(latent.shape)} range [{latent.min():.3f},{latent.max():.3f}]")
    print(f"audio  {tuple(audio.shape)} -> {out}/tm_pytorch.wav")
    (out / "tm_manifest.json").write_text(json.dumps(
        {"prompt": args.prompt, "frames": T, "secs": secs, "steps": args.steps,
         "seed": args.seed, "sample_rate": sr}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
