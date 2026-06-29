---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-small-music
tags:
- audio-generation
- music
- diffusion
- gguf
- sa3.cpp
---

# Stable Audio 3 Small-Music — GGUF (for sa3.cpp)

GGUF conversions of [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
for [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp) — a portable C++/GGML port of
Stable Audio 3, no PyTorch in the loop. Runs on CPU, CUDA, Vulkan, or Metal (Apple Silicon). The small-music model is a
lighter, faster music generator (SAME-S autoencoder, 0.5B DiT). Validated against the PyTorch
reference at cosine similarity ~1.0.

## Files

Multi-file model. Grab the **DiT** + **SAME** at your chosen precision and the **conditioner**, plus
the shared **encoder + tokenizer** from
[t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF).

| component | file | notes |
|---|---|---|
| DiT (diffusion transformer) | `stable-audio-3-small-music-dit-0.5B-v1.0-{F16,F32}.gguf` | pick one precision |
| autoencoder (SAME-S) | `stable-audio-3-small-music-same-s-v1.0-{F16,F32}.gguf` | pick one precision |
| conditioner | `stable-audio-3-small-music-conditioner-v1.0-F32.gguf` | tiny sidecar (prompt padding + seconds_total) |
| encoder + tokenizer | → [t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF) | **shared** across all SA3 variants |

> **note:** SAME-S needs an **even** `--frames` count (the packed sequence must divide the chunk size).

## Usage

For use with [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp):

```bash
python tools/download_models.py --variant small-music --encoding f16

sa3-generate --tok models/t5gemma-b-b-ul2-v1.0-vocab.gguf \
    --t5 models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
    --cond models/stable-audio-3-small-music-conditioner-v1.0-F32.gguf \
    --dit models/stable-audio-3-small-music-dit-0.5B-v1.0-F16.gguf \
    --same models/stable-audio-3-small-music-same-s-v1.0-F16.gguf \
    --prompt "lo-fi hip hop beat, warm vinyl, mellow keys" --frames 128 --steps 8 --out song.wav
```

## Performance

Roughly **1.7s for a 12s clip** at f16 on an 8GB laptop GPU (RTX 5070) — about 2× faster than the medium
model. The sliding-window decoder keeps long generations linear. Full numbers + levers:
[docs/BENCHMARKS.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/BENCHMARKS.md).

## License

These are format conversions of [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music),
whose weights Stability AI releases under the [Stability AI Community License](https://stability.ai/license):
free for organizations under $1M annual revenue, with commercial use, fine-tuning, and derivative works
permitted within that threshold (above it, contact Stability AI for an Enterprise License). Outputs are yours.
That license carries over to these converted weights.

The upstream [stable-audio-3 source code](https://github.com/Stability-AI/stable-audio-3) is released
separately under MIT. Pair these with the shared T5Gemma text encoder, which is Google's under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Relationship to the original

**Format conversions** (weights → GGUF) for inference in sa3.cpp — no retraining. See
[sa3.cpp/docs/DISTRIBUTION.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/DISTRIBUTION.md).
