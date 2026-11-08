# Parity with the PyTorch service

How to check sa3.cpp against stable_audio_3 as the gary4local SA3 service runs it (medium,
`model_half=True`, adapters attached as torch parametrizations, pingpong sampler), one stage at
a time with identical inputs.

## Same seed, same noise

The default noise, `--rng torch-cuda` (CLI) / `"rng": "torch-cuda"` (server), draws what
`torch.manual_seed(seed)` gives stable_audio_3 on a CUDA GPU, in its order:
`torch.randn([1, 256, T])` for the initial latent, then one `randn_like` per ping-pong step.
It runs on the host, so every sa3.cpp backend (CUDA, Vulkan, Metal, CPU) gets the same noise.
PyTorch's normal kernel is a grid-stride loop, so once a canvas holds more than `256 * cap`
values the layout depends on the PyTorch side's GPU: `--rng-cuda-blocks` is that GPU's
`multi_processor_count * (max_threads_per_multi_processor // 256)`, by default 288 for the DGX
Spark that serves gary4juce's remote SA3 (216 for the RTX 5070 Laptop used below). The text2music canvas is
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
- **A negative prompt's padding reached the unconditioned pass.** stable_audio_3's CFG branch
  replaces the negative prompt's masked tokens with the null embedding
  (`torch.where(negative_cross_attn_mask, negative, 0)`); the conditioned prompt keeps its
  padding embedding, because the DiT disables that mask. sa3.cpp padded both. With "low quality"
  (3 valid tokens of 257) at cfg 2, that put the teacher-forced velocity 9-45% off per step and
  the final latent 3.4% off, against 0.6-4.5% and 0.24% at cfg 2 without a negative prompt.
  sa3.cpp now zeroes those rows of the unconditioned conditioning, and the run lands at
  0.7-4.1% / 0.24% (all with the fp32, no-padding-mask reference described below).
- The decoder, the squeakfix decoder adapter, F16 vs F32 weights, merged vs functional adapters,
  the schedule and CFG all match.
- Known remaining differences: the reference masks the 2 headroom frames past
  `seconds + 6 s` out of self-attention (`padding_mask`), which moves the kept frames by 1-2%;
  how much depends on its attention backend (flash-attn varlen excludes them, the SDPA fallback
  only zeroes their values). The fp16 service also quantizes the normalized seconds to fp16
  before 10 kHz Fourier features, which moves its seconds token by 7.5%; sa3.cpp's f32 value
  matches the fp32 reference.

## Lowering the floor

Against the fp16 service with its padding mask, even the base model differs 6-16% per step on
the DGX Spark, enough to hide a mistake of a few percent in whatever you are checking. Record
the reference in fp32 and without the padding mask (`--fp32 --no-padding-mask`) and compare
teacher-forced: the base model then sits at 0.3-2.9% per step and 0.14% in the final latent.

## Automation lanes (DGX Spark, medium, 10 s, 8 steps)

gary4juce's remote SA3 runs a PyTorch port of the lanes (`lora_lanes.py` in the service), so
the same request gives the same clip on either side. Compared as above (fp32 reference, no
padding mask, teacher-forced), with the registry's DoRA adapters converted from their `.ckpt`:

| case | DiT velocity per step | final latent |
| --- | --- | --- |
| base model, no adapter (the floor) | 0.34-2.9% | cos 0.999999, rel 0.14% |
| two lanes, kev 1 → 0 and keygen 0 → 1, cfg 1 | 0.29-1.7% | cos 0.999999, rel 0.12% |
| shared kev 0.5, a lane with its own prompt, the Automation tab's blend (`lane_average_until` 0.9, `lane_isolation` 1, `lane_isolation_until` 0), cfg 1 | 0.37-4.2% | cos 0.999999, rel 0.15% |
| the same at cfg 2 with a negative prompt | 0.9-5.3% | cos 0.999994, rel 0.35% |

The lanes move the latent 71% against the same seed without them, so the match isn't the
curves doing nothing. Every pass in these cases holds one adapter, or the shared adapter followed
by one that also sorts after it. In a pass with several DoRAs the PyTorch side applies them in
its load order rather than shared-then-lane, which can differ slightly because DoRA
renormalizes.
