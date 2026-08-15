---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-small-sfx
base_model_relation: quantized
tags:
- audio-generation
- sound-effects
- diffusion
- gguf
- sa3.cpp
---

# Stable Audio 3 Small-SFX — GGUF (for sa3.cpp)

GGUF conversions of [stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
for [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp) — a portable C++/GGML port of
Stable Audio 3, no PyTorch in the loop. Runs on CPU, CUDA, Vulkan, or Metal (Apple Silicon). The small-sfx model targets
**sound effects / foley** (SAME-S autoencoder, 0.5B DiT). Validated against the PyTorch reference at
cosine similarity ~1.0.

## Files

Multi-file model. Grab the **DiT** + **SAME** at your chosen precision and the **conditioner**, plus
the shared **encoder + tokenizer** from
[t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF).

| component | file | notes |
|---|---|---|
| DiT (diffusion transformer) | `stable-audio-3-small-sfx-dit-0.5B-v1.0-{F32,F16,Q8_0,Q5_K_M,Q4_K_M}.gguf` | pick one encoding |
| autoencoder (SAME-S) | `stable-audio-3-small-sfx-same-s-v1.0-{F32,F16,Q8_0,Q5_K_M,Q4_K_M}.gguf` | **must match the DiT** |
| conditioner | `stable-audio-3-small-sfx-conditioner-v1.0-F32.gguf` | tiny sidecar (prompt padding + seconds_total) |
| encoder + tokenizer | → [t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF) | **shared** across all SA3 variants |

> **note:** SAME-S needs an **even** `--frames` count (the packed sequence must divide the chunk size).

## Encodings

`sa3-generate --encoding` resolves the DiT and the SAME with the same suffix, so download the pair.

| encoding | DiT | SAME-S | total |
|---|---:|---:|---:|
| F32 | 1751 MB | 413 MB | 2164 MB |
| F16 | 877 MB | 207 MB | 1084 MB |
| Q8_0 | 471 MB | 110 MB | 581 MB |
| Q5_K_M | 333 MB | 79 MB | 412 MB |
| Q4_K_M | 302 MB | 72 MB | 374 MB |

`q4_k_m` and `q5_k_m` promote the attention V, feed-forward down and embedding tensors to Q6_K;
`q8_0` is uniform. Every tier passes `sa3-quant-check` with `below-threshold=0` at cosine `0.990`
against the F16 reference, for the DiT and the SAME alike.

**Quantization buys footprint everywhere and speed only on some backends.** CUDA and Vulkan gain
roughly 33% end to end. **Metal is flat** — Q8_0 is 1.7% faster and Q4_K_M 2.1% *slower* than F16,
because the load-time saving and the added per-step dequant cancel out. On a Mac, pick a quant for
the memory, not for the speed.

## Usage

For use with [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp):

```bash
python tools/download_models.py --variant small-sfx --encoding f16

# --model resolves the gguf set in ./models by name
sa3-generate --model small-sfx --prompt "a dog barking in a large empty hall" --out sfx.wav
```

For a quantized set, pass the encoding to both — the downloader and the generator use the same names:

```bash
python tools/download_models.py --variant small-sfx --encoding q4_k_m
sa3-generate --model small-sfx --encoding q4_k_m --prompt "a dog barking in a large hall" --out sfx.wav
```

## Performance

Roughly **1.7s for a 12s clip** at f16 on an 8GB laptop GPU (RTX 5070) — about 2× faster than the medium
model. The sliding-window decoder keeps long generations linear. Full numbers + levers:
[docs/BENCHMARKS.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/BENCHMARKS.md).

## License

These are format conversions of [stabilityai/stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx),
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
