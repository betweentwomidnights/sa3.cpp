---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-small-music-base
base_model_relation: quantized
tags:
- audio-generation
- music
- diffusion
- gguf
- lora
- sa3.cpp
---

# Stable Audio 3 Small Music Base DiT — GGUF for sa3.cpp

F32, F16 and Q4_K_M GGUF conversions of the training DiT from
[stabilityai/stable-audio-3-small-music-base](https://huggingface.co/stabilityai/stable-audio-3-small-music-base),
for native LoRA/DoRA training with [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp).

This is a training-only base DiT. Train the adapter here, then apply it to
[Stable Audio 3 Small Music](https://huggingface.co/thepatch/stable-audio-3-small-music-GGUF)
for inference.

## Files

| file | size | purpose |
|---|---:|---|
| `stable-audio-3-small-music-base-dit-0.5B-v1.0-Q4_K_M.gguf` | 302 MB | smallest, and the fastest to train on |
| `stable-audio-3-small-music-base-dit-0.5B-v1.0-F16.gguf` | 877 MB | the reference training base |
| `stable-audio-3-small-music-base-dit-0.5B-v1.0-F32.gguf` | 1751 MB | CPU/reference validation |
| `SHA256SUMS` | | release checksums |

The normal small-music model set supplies all other components.

```bash
python tools/download_models.py --variant small-music --encoding f16 --training-base
build-cuda/bin/sa3-train --model small-music --models-dir models --dataset /path/to/dataset --out train-runs/example
```

## Training on the Q4_K_M base

Training against a quantized base is supported on **every** backend — CPU, CUDA, Vulkan and Metal.
The frozen base only ever enters the adapter path as a `mul_mat` argument, and the one backward that
needed it, `out_prod(W, transpose(grad))`, now accepts a quantized `src0` on all three GPU backends.
It is *faster* than F16 rather than a tradeoff, on a 2.9x smaller file, and the adapter it produces
is an ordinary GGUF LoRA that applies to an F16 or a quantized inference DiT either way.

At 302 MB this is the smallest trainable SA3 base published, which is the one that decides whether
a phone or a low-VRAM laptop can train at all.

```bash
python tools/download_models.py --variant small-music --encoding q4_k_m --training-base
build-cuda/bin/sa3-train --model small-music --models-dir models \
  --dit models/stable-audio-3-small-music-base-dit-0.5B-v1.0-Q4_K_M.gguf \
  --dataset /path/to/dataset --adapter-type dora-rows --rank 16 --out train-runs/example
```

## Provenance

- Source revision: `eab5ceee5ad9c1ed38800aff30a8e49d1161c539`
- Conversion: `tools/convert_dit.py --variant small-music --training-base`, then
  `tools/quantize_gguf.py` for F16 and `sa3-quantize --mix q4_k_m` for the quant
- Relationship: tensor rename/serialization and precision conversion only; no retraining
- The Q4_K_M base passes `sa3-quant-check` against F16 with `below-threshold=0` at cosine `0.990`

## License and attribution

Powered by Stability AI.

These converted weights remain under the Stability AI Community License. This repository includes
the upstream `LICENSE.md` and a `NOTICE` describing the conversion and retaining the required
Stability AI attribution.
