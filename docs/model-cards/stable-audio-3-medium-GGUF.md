<!-- v1 draft model card — may revise after Stability review. -->
---
language:
- en
license: other
license_name: stable-audio-community
license_link: https://stability.ai/license
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-medium
tags:
- audio-generation
- music
- sound-effects
- diffusion
- gguf
- sa3.cpp
---

# Stable Audio 3 Medium — GGUF (for sa3.cpp)

GGUF conversions of [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
for [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp) — a portable C++/GGML port of
Stable Audio 3, no PyTorch in the loop. Runs on CPU, CUDA, or Vulkan. Every component is validated
against the PyTorch reference at cosine similarity ~1.0.

## Files

This is a multi-file model. Grab the **DiT** + **SAME** at your chosen precision and the
**conditioner**, plus the shared **encoder + tokenizer** from the
[t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF) repo.

| component | file | notes |
|---|---|---|
| DiT (diffusion transformer) | `stable-audio-3-medium-dit-1.5B-v1.0-{F16,F32}.gguf` | pick one precision |
| autoencoder (SAME-L) | `stable-audio-3-medium-same-l-v1.0-{F16,F32}.gguf` | pick one precision |
| conditioner | `stable-audio-3-medium-conditioner-v1.0-F32.gguf` | tiny sidecar (prompt padding + seconds_total) |
| encoder + tokenizer | → [t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF) | **shared** across all SA3 variants |

**F16** is the production path (~3.5s for 12s of audio on an 8GB laptop GPU); **F32** is for CPU
validation. The conditioner + encoder + tokenizer stay F32 (small / quality-critical).

## Usage

```bash
# pip install huggingface_hub
python tools/download_models.py --variant medium --encoding f16   # fetches this set + the shared encoder

sa3-generate --tok models/t5gemma-b-b-ul2-v1.0-vocab.gguf \
    --t5 models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
    --cond models/stable-audio-3-medium-conditioner-v1.0-F32.gguf \
    --dit models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
    --same models/stable-audio-3-medium-same-l-v1.0-F16.gguf \
    --prompt "upbeat funk groove with slap bass" --frames 128 --steps 8 --out song.wav
```

## Performance

Roughly **3s for a 12s clip** at f16 on an 8GB laptop GPU (RTX 5070), and ~6s on an Apple M4 — end to end,
including model load. The sliding-window decoder keeps long generations linear (a 2-minute clip is ~9s on
the 5070). CPU works but is ~10× slower. Full numbers + the f16 / flash-attention levers:
[docs/BENCHMARKS.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/BENCHMARKS.md).

## License

These are format conversions of [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium),
whose weights Stability AI releases under the [Stability AI Community License](https://stability.ai/license):
free for organizations under $1M annual revenue, with commercial use, fine-tuning, and derivative works
permitted within that threshold (above it, contact Stability AI for an Enterprise License). Outputs are yours.
That license carries over to these converted weights.

The upstream [stable-audio-3 source code](https://github.com/Stability-AI/stable-audio-3) is released
separately under MIT. Pair these with the shared T5Gemma text encoder, which is Google's under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Relationship to the original

**Format conversions** (weights → GGUF) for inference in sa3.cpp — no retraining, no architectural
changes. See [sa3.cpp/docs/DISTRIBUTION.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/DISTRIBUTION.md)
for the naming convention and how the pieces fit together.
