---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-medium-base
tags:
- audio-generation
- diffusion
- gguf
- lora
- sa3.cpp
---

# Stable Audio 3 Medium Base DiT — GGUF for sa3.cpp

F16 and F32 GGUF conversions of the training DiT from
[stabilityai/stable-audio-3-medium-base](https://huggingface.co/stabilityai/stable-audio-3-medium-base),
for native LoRA/DoRA training with [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp).

This is a training-only base DiT. Train the adapter on this model, then apply that adapter to
[Stable Audio 3 Medium](https://huggingface.co/thepatch/stable-audio-3-medium-GGUF) for inference.
Do not use the post-trained/ARC medium DiT as the training base.

## Files

| file | purpose |
|---|---|
| `stable-audio-3-medium-base-dit-1.5B-v1.0-F16.gguf` | recommended frozen training base |
| `stable-audio-3-medium-base-dit-1.5B-v1.0-F32.gguf` | CPU/reference validation |
| `SHA256SUMS` | release checksums |

The tokenizer, T5Gemma encoder, conditioner, SAME autoencoder, and inference DiT are fetched from
the normal medium model set; they are intentionally not duplicated here.

```bash
python tools/download_models.py --variant medium --encoding f16 --training-base
build-cuda/bin/sa3-train --model medium --models-dir models --dataset /path/to/dataset --out train-runs/example
```

## Provenance

- Source revision: `b32993f73c3bdc3864043a72d8032606bba737c8`
- Conversion: `tools/convert_dit.py --variant medium --training-base`, then `tools/quantize_gguf.py` for F16
- Relationship: tensor rename/serialization and precision conversion only; no retraining

## License and attribution

Powered by Stability AI.

These converted weights remain under the Stability AI Community License. This repository includes
the upstream `LICENSE.md` and a `NOTICE` describing the conversion and retaining the required
Stability AI attribution. Organizations above the license's revenue threshold must obtain the
appropriate license from Stability AI.
