---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-small-sfx-base
tags:
- audio-generation
- sound-effects
- diffusion
- gguf
- lora
- sa3.cpp
---

# Stable Audio 3 Small SFX Base DiT — GGUF for sa3.cpp

F16 and F32 GGUF conversions of the training DiT from
[stabilityai/stable-audio-3-small-sfx-base](https://huggingface.co/stabilityai/stable-audio-3-small-sfx-base),
for native LoRA/DoRA training with [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp).

This is a training-only base DiT. Train the adapter here, then apply it to
[Stable Audio 3 Small SFX](https://huggingface.co/thepatch/stable-audio-3-small-sfx-GGUF)
for inference.

## Files

| file | purpose |
|---|---|
| `stable-audio-3-small-sfx-base-dit-0.5B-v1.0-F16.gguf` | recommended frozen training base |
| `stable-audio-3-small-sfx-base-dit-0.5B-v1.0-F32.gguf` | CPU/reference validation |
| `SHA256SUMS` | release checksums |

The normal small-SFX model set supplies all other components.

```bash
python tools/download_models.py --variant small-sfx --encoding f16 --training-base
build-cuda/bin/sa3-train --model small-sfx --models-dir models --dataset /path/to/dataset --out train-runs/example
```

## Provenance

- Source revision: `cc5ddb990e30daa68336ac61c140c37c7033ab7c`
- Conversion: `tools/convert_dit.py --variant small-sfx --training-base`, then `tools/quantize_gguf.py` for F16
- Relationship: tensor rename/serialization and precision conversion only; no retraining

## License and attribution

Powered by Stability AI.

These converted weights remain under the Stability AI Community License. This repository includes
the upstream `LICENSE.md` and a `NOTICE` describing the conversion and retaining the required
Stability AI attribution.
