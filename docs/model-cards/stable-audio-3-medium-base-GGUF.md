---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-medium-base
base_model_relation: quantized
tags:
- audio-generation
- diffusion
- gguf
- lora
- sa3.cpp
---

# Stable Audio 3 Medium Base DiT — GGUF for sa3.cpp

F32, F16 and Q4_K_M GGUF conversions of the training DiT from
[stabilityai/stable-audio-3-medium-base](https://huggingface.co/stabilityai/stable-audio-3-medium-base),
for native LoRA/DoRA training with [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp).

This is a training-only base DiT. Train the adapter on this model, then apply that adapter to
[Stable Audio 3 Medium](https://huggingface.co/thepatch/stable-audio-3-medium-GGUF) for inference.
Do not use the post-trained/ARC medium DiT as the training base.

## Files

| file | size | purpose |
|---|---:|---|
| `stable-audio-3-medium-base-dit-1.5B-v1.0-Q4_K_M.gguf` | 962 MB | smallest, and the fastest to train on |
| `stable-audio-3-medium-base-dit-1.5B-v1.0-F16.gguf` | 2773 MB | the reference training base |
| `stable-audio-3-medium-base-dit-1.5B-v1.0-F32.gguf` | 5545 MB | CPU/reference validation |
| `SHA256SUMS` | | release checksums |

The tokenizer, T5Gemma encoder, conditioner, SAME autoencoder, and inference DiT are fetched from
the normal medium model set; they are intentionally not duplicated here.

```bash
python tools/download_models.py --variant medium --encoding f16 --training-base
build-cuda/bin/sa3-train --model medium --models-dir models --dataset /path/to/dataset --out train-runs/example
```

## Training on the Q4_K_M base

Training against a quantized base is supported on **every** backend — CPU, CUDA, Vulkan and Metal.
The frozen base only ever enters the adapter path as a `mul_mat` argument, and the one backward that
needed it, `out_prod(W, transpose(grad))`, now accepts a quantized `src0` on all three GPU backends.

It is *faster* than F16 rather than a tradeoff, on a 2.9x smaller file — steady state, medium,
128 frames, `dora-rows`:

| backend | Q4_K_M | F16 |
|---|---:|---:|
| CUDA | 0.93 s/step | 1.08 s/step |
| Vulkan | 1.72 s/step | 1.90 s/step |
| Metal (M4) | 2.95 s/step | 3.39 s/step |
| CPU | 29.2 s/step | 46.9 s/step |

A 2000-step DoRA trained on the Q4_K_M base is **audibly indistinguishable** from the same run on
the F16 base; loss tracks 2.5-4% higher throughout, the offset a slightly degraded forward should
produce. Peak VRAM drops from 5243 to 3414 MiB, which is the part that decides whether a small
machine can train at all.

```bash
python tools/download_models.py --variant medium --encoding q4_k_m --training-base
build-cuda/bin/sa3-train --model medium --models-dir models \
  --dit models/stable-audio-3-medium-base-dit-1.5B-v1.0-Q4_K_M.gguf \
  --dataset /path/to/dataset --adapter-type dora-rows --rank 16 --out train-runs/example
```

The resulting adapter is an ordinary GGUF LoRA — nothing about it is quantization-specific, so it
applies to an F16 or a quantized inference DiT either way.

## Provenance

- Source revision: `b32993f73c3bdc3864043a72d8032606bba737c8`
- Conversion: `tools/convert_dit.py --variant medium --training-base`, then `tools/quantize_gguf.py`
  for F16 and `sa3-quantize --mix q4_k_m` for the quant
- Relationship: tensor rename/serialization and precision conversion only; no retraining
- The Q4_K_M base passes `sa3-quant-check` against F16 with `below-threshold=0` at cosine `0.990`

## License and attribution

Powered by Stability AI.

These converted weights remain under the Stability AI Community License. This repository includes
the upstream `LICENSE.md` and a `NOTICE` describing the conversion and retaining the required
Stability AI attribution. Organizations above the license's revenue threshold must obtain the
appropriate license from Stability AI.
