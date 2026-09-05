---
license: other
license_name: stability-ai-community
license_link: https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md
tags:
- audio-generation
- foundation-1
- stable-audio-open
- gguf
- sa3.cpp
---

# Foundation-1 GGUF

GGUF conversions of [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1)
for [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp)'s optional native
stable-audio-tools runtime. Foundation-1 is a full finetune of Stable Audio Open 1.0 and
uses the same T5-base and Oobleck architecture.

Each encoding is a complete three-file bundle:

| Tier | Complete bundle | Guidance |
| --- | ---: | --- |
| F16 | 2,377 MiB | reference |
| Q8_0 | 1,296 MiB | conservative high fidelity |
| Q5_K_M | 922 MiB | recommended default |
| Q4_K_M | 832 MiB | smallest footprint |

All quantized tensors passed the per-tensor 0.990 cosine gate. Matched CUDA renders
measured envelope/log-magnitude cosine of 0.9967/0.9901 at Q8, 0.9904/0.9550 at Q5,
and 0.9643/0.9284 at Q4. All four tiers passed listening tests.

## Musical timing

Foundation-1 was trained for 4- or 8-bar clips at 100, 110, 120, 128, 130, 140, or
150 BPM. The prompt should include the selected bar count and BPM. sa3.cpp's Foundation
profile calculates the exact sample crop, the whole-second `seconds_total` conditioning
value, and the padded Oobleck latent canvas independently to match RoyalCities' inference.

Unsupported BPMs are intentionally rejected by this model profile. We are considering an
optional pitch-preserving time-stretch layer for arbitrary host tempos; it is not part of
the current GGML inference primitives. Applications such as gary4local may map to a trained
BPM and stretch the result downstream.

RoyalCities' UI profile uses DPM++ 3M SDE, sigma 0.01–100, 100 steps, and CFG 7. The
gary4local fallback profile uses DPM++ 2M SDE, sigma 0.5–50, 100 steps, and CFG 7.

The SAT CLI is being generalized around these model profiles. A planned `--randomize` mode
will report the selected prompt and seed together with the output audio instead of exposing
a separate randomization API.

## Sources and licensing

- Foundation-1 source revision: `d3160956fa13a8f861d3f608ed24075d44e98554`.
- Stable Audio Open 1.0 base revision: `f21265c1e2710b3bd2386596943f0007f55f802e`.
- T5-base revision: `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`.

The Stability AI Community License is included as `LICENSE.md`; RoyalCities' source
license notice is retained as `LICENSE_FOUNDATION.md`, and T5-base's Apache 2.0 license is
included as `LICENSE_T5.md`. See `NOTICE` for conversion details.

Powered by Stability AI.
