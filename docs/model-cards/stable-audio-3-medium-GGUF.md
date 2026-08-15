---
language:
- en
license: other
license_name: stable-audio-community
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: stabilityai/stable-audio-3-medium
base_model_relation: quantized
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
Stable Audio 3, no PyTorch in the loop. Runs on CPU, CUDA, Vulkan, or Metal (Apple Silicon). Every component is validated
against the PyTorch reference at cosine similarity ~1.0.

## Files

This is a multi-file model. Grab the **DiT** + **SAME** at your chosen precision and the
**conditioner**, plus the shared **encoder + tokenizer** from the
[t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF) repo.

| component | file | notes |
|---|---|---|
| DiT (diffusion transformer) | `stable-audio-3-medium-dit-1.5B-v1.0-{F32,F16,Q8_0,Q5_K_M,Q4_K_M}.gguf` | pick one encoding |
| autoencoder (SAME-L) | `stable-audio-3-medium-same-l-v1.0-{F32,F16,Q8_0,Q5_K_M,Q4_K_M}.gguf` | **must match the DiT** |
| conditioner | `stable-audio-3-medium-conditioner-v1.0-F32.gguf` | tiny sidecar (prompt padding + seconds_total) |
| encoder + tokenizer | → [t5gemma-b-b-ul2-GGUF](https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF) | **shared** across all SA3 variants |

**F16** is the production path (~3.5s for 12s of audio on an 8GB laptop GPU); **F32** is for CPU
validation. The conditioner + encoder + tokenizer stay F32 (small / quality-critical).

## Encodings

`sa3-generate --encoding` resolves the DiT and the SAME with the same suffix, so download the pair.

| encoding | DiT | SAME-L | total |
|---|---:|---:|---:|
| F32 | 5545 MB | 3251 MB | 8796 MB |
| F16 | 2773 MB | 1626 MB | 4399 MB |
| Q8_0 | 1483 MB | 865 MB | 2348 MB |
| Q5_K_M | 1053 MB | 617 MB | 1670 MB |
| Q4_K_M | 962 MB | 570 MB | 1532 MB |

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
# pip install huggingface_hub
python tools/download_models.py --variant medium --encoding f16   # fetches this set + the shared encoder

# --model resolves the 5 gguf files in ./models by name
sa3-generate --model medium --prompt "upbeat funk groove with slap bass" --out song.wav
```

For a quantized set, pass the encoding to both — the downloader and the generator use the same names:

```bash
python tools/download_models.py --variant medium --encoding q4_k_m
sa3-generate --model medium --encoding q4_k_m --prompt "upbeat funk groove with slap bass" --out song.wav
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
