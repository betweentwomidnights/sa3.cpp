# Parity with the PyTorch service

How to check sa3.cpp against stable_audio_3 as the gary4local SA3 service runs it (medium,
`model_half=True`, adapters attached as torch parametrizations, pingpong sampler), one stage at
a time with identical inputs.

## Same seed, same noise

`--rng torch-cuda` (CLI) or `"rng": "torch-cuda"` (server) draws what `torch.manual_seed(seed)`
gives stable_audio_3 on a CUDA GPU, in its order: `torch.randn([1, 256, T])` for the initial
latent, then one `randn_like` per ping-pong step. PyTorch's normal kernel is a grid-stride
loop, so once a canvas holds more than `256 * cap` values (about 20 s on an RTX 5070 Laptop)
the layout depends on the GPU: pass `--rng-cuda-blocks` with that GPU's
`multi_processor_count * (max_threads_per_multi_processor // 256)`. The text2music canvas is
sized as `_adapt_sample_size` sizes it (`int((seconds + 6) * 44100)` samples, rounded up to
whole frames and then to 2), so the noise tensor has the reference's shape.

## Recording and comparing

```bash
# the reference: run with a CUDA torch env; --sa3-pkg is the directory holding the service's stable_audio_3
python tools/dump_remote_parity.py --hf-home <hub root> --sa3-pkg <services/sa3> \
    --lora loras/koan/koan.ckpt --prompt "..." --duration 10 --seed 42 --out parity/ref
# sa3.cpp on the same seed, every stage dumped
SA3_PARITY_DUMP=parity/cpp sa3-generate --model medium --lora koan --prompt "..." --duration 10 \
    --seed 42 --rng torch-cuda --rng-cuda-blocks 216 --no-peak-normalize --no-limiter --out parity/cpp/cpp.wav
python tools/compare_remote_parity.py --ref parity/ref --cpp parity/cpp \
    --wav-ref parity/ref/torch.wav --wav-cpp parity/cpp/cpp.wav
```

To isolate a stage, write the reference's own inputs with `--feed DIR` and run sa3-generate
with `SA3_PARITY_FEED=DIR`: `cross`/`global` replace the conditioning, `steps` replaces the
state the DiT sees at each step (teacher forcing, so every step's velocity compares on equal
input), `latent` replaces what the decoder gets. `dump_remote_parity.py --decode-latent` decodes
any latent with the PyTorch decoder (fp16 or `--fp32`, optionally `--decoder-lora`).

Pass the adapter's `.ckpt` (or a `.safetensors` that carries `lora_config` metadata): given a
bare `.safetensors`, stable_audio_3's loader defaults to `adapter_type: "lora"` and silently
drops a DoRA's magnitudes.

## What it found (koan, seed 42, 8 steps, cfg 1)

| stage | before | after |
| --- | --- | --- |
| initial noise | (mt19937) | rel 2e-4 (the reference's fp16 rounding) |
| schedule | exact | exact |
| DiT velocity per step, teacher-forced, base model | 7.8-12% | 1.5-2.9% |
| DiT velocity per step, teacher-forced, koan | 9.5-12% | 2.3-5.0% |
| same, reference without its padding mask | | 0.6-1.3% |
| seconds token vs fp32 torch, koan | 2.1% | 0.04% |
| SAME-L decode of one latent vs fp32 torch | 0.42% | 0.42% |
| final latent, 10 s, vs fp16 torch | 35% | 9.7% |
| final latent, 30 s, vs fp16 torch | 50% (audio cos 0.73) | 16% (audio cos 0.97) |

PyTorch's own fp16 and fp32 runs differ by 0.9-1.3% per step and 5.3% in the final latent.

- **The DiT's local-cond block was skipped.** For plain generation stable_audio_3 still feeds
  every block's `to_local_embed` MLP an all-zero `[mask | masked_input]`, which adds
  `W2·silu(b1) + b2` (norm 0.5-1.9) to every frame in every block, plus whatever an adapter
  trained into those layers. sa3.cpp now feeds the zeros (`SA3_LOCAL_ZEROS=0` restores the
  skip, for A/B); it costs about 4% of DiT time.
- **The seconds_total conditioner's adapter was dropped.** stable_audio_3 attaches DiT adapters
  to the conditioner too, and koan carries a DoRA on the seconds projection. The converters now
  keep it as `te.secs`, and the pipeline applies it where it computes that token. Adapters
  converted before this carry no `te.secs`; convert them again to pick it up.
- The decoder, the squeakfix decoder adapter, F16 vs F32 weights, merged vs functional adapters,
  the schedule and CFG all match.
- Known remaining differences: the reference masks the 2 headroom frames past
  `seconds + 6 s` out of self-attention (`padding_mask`), which moves the kept frames by 1-2%;
  how much depends on its attention backend (flash-attn varlen excludes them, the SDPA fallback
  only zeroes their values). The fp16 service also quantizes the normalized seconds to fp16
  before 10 kHz Fourier features, which moves its seconds token by 7.5%; sa3.cpp's f32 value
  matches the fp32 reference.
