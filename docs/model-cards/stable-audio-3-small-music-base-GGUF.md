---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-small-music-base
tags:
- audio-generation
- music
- diffusion
- gguf
- lora
- sa3.cpp
---

# Stable Audio 3 Small Music Base DiT — GGUF for sa3.cpp

F16 and F32 GGUF conversions of the training DiT from
[stabilityai/stable-audio-3-small-music-base](https://huggingface.co/stabilityai/stable-audio-3-small-music-base),
for native LoRA/DoRA training with [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp).

This is a training-only base DiT. Train the adapter here, then apply it to
[Stable Audio 3 Small Music](https://huggingface.co/thepatch/stable-audio-3-small-music-GGUF)
for inference.

## Files

| file | purpose |
|---|---|
| `stable-audio-3-small-music-base-dit-0.5B-v1.0-F16.gguf` | recommended frozen training base |
| `stable-audio-3-small-music-base-dit-0.5B-v1.0-F32.gguf` | CPU/reference validation |
| `SHA256SUMS` | release checksums |

The normal small-music model set supplies all other components.

```bash
python tools/download_models.py --variant small-music --encoding f16 --training-base
build-cuda/bin/sa3-train --model small-music --models-dir models --dataset /path/to/dataset --out train-runs/example
```

## Provenance

- Source revision: `eab5ceee5ad9c1ed38800aff30a8e49d1161c539`
- Conversion: `tools/convert_dit.py --variant small-music --training-base`, then `tools/quantize_gguf.py` for F16
- Relationship: tensor rename/serialization and precision conversion only; no retraining

## License and attribution

Powered by Stability AI.

These converted weights remain under the Stability AI Community License. This repository includes
the upstream `LICENSE.md` and a `NOTICE` describing the conversion and retaining the required
Stability AI attribution.
