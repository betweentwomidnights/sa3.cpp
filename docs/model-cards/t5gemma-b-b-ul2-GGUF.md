---
language:
- en
license: gemma
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model: google/t5gemma-b-b-ul2
tags:
- text-encoder
- gguf
- sa3.cpp
---

# T5Gemma-b-b-ul2 Encoder + Tokenizer — GGUF (for sa3.cpp)

The **shared** text encoder + tokenizer for [**sa3.cpp**](https://github.com/betweentwomidnights/sa3.cpp).
GGUF conversion of the frozen [google/t5gemma-b-b-ul2](https://huggingface.co/google/t5gemma-b-b-ul2)
encoder (encoder-only at inference) that Stable Audio 3 uses to embed text prompts.

This component is **identical across all three SA3 variants** (medium, small-music, small-sfx), so it
lives in its own repo and is fetched once — the per-variant conditioner ships separately in each model
repo. Validated against the PyTorch reference at cosine similarity ~1.0.

## Files

| component | file | size | notes |
|---|---|---:|---|
| text encoder | `t5gemma-b-b-ul2-encoder-0.3B-v1.0-F16.gguf` | 537 MiB | **default** — equivalent to F32 |
| text encoder | `t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf` | 1074 MiB | reference precision |
| text encoder | `t5gemma-b-b-ul2-encoder-0.3B-v1.0-Q8_0.gguf` | 285 MiB | smallest; a small, real tradeoff |
| tokenizer | `t5gemma-b-b-ul2-v1.0-vocab.gguf` | 14 MiB | Gemma byte-fallback BPE |

The encoder carries **no conditioner** — that ships per-variant in each model repo.

### picking an encoder precision

Measured against an F32 control, swapping *only* the encoder:

| encoder | conditioning cosine | generated-audio cosine |
|---|---:|---:|
| **F16** | 0.999995 | 0.999848 |
| Q8_0 | 0.999547 | 0.999108 |

`F16` is what `sa3.cpp` downloads and resolves by default, and it is a size win rather than a
fidelity substitution. `Q8_0` is offered for tight-memory installs; it is a small but real
tradeoff, so it is never auto-selected — ask for it by name.

`Q4_K_M` is **not published** for this encoder, on evidence rather than caution: it is *slower*
than F32 here (the T5 forward is too small to amortize dequant, unlike the DiT) while dropping
generated-audio cosine to ~0.80, which is a different piece of music rather than a degraded one.
`Q8_0` dominates it on every axis.

Select a precision with `--t5-encoding f16|f32|q8_0`, independently of the `--encoding` that picks
the DiT/SAME tier — the combination worth having on a small device is a quantized DiT with an F16
encoder. This works the same way at inference (`sa3-generate`) and at training (`sa3-train`).

Pair these with any SA3 variant repo's DiT + SAME + conditioner:
[medium](https://huggingface.co/thepatch/stable-audio-3-medium-GGUF) ·
[small-music](https://huggingface.co/thepatch/stable-audio-3-small-music-GGUF) ·
[small-sfx](https://huggingface.co/thepatch/stable-audio-3-small-sfx-GGUF).
`tools/download_models.py` fetches this repo automatically alongside whichever variant you pick.

## License

This is a format conversion of [google/t5gemma-b-b-ul2](https://huggingface.co/google/t5gemma-b-b-ul2),
released under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (including the use restrictions
in Section 3.2). Those terms carry over to this converted encoder + tokenizer.

## Relationship to the original

A **format conversion** (weights → GGUF) for inference in sa3.cpp — no retraining. See
[sa3.cpp/docs/DISTRIBUTION.md](https://github.com/betweentwomidnights/sa3.cpp/blob/main/docs/DISTRIBUTION.md).
