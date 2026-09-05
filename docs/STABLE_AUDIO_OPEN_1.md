# Stable Audio Open 1.0 and Foundation-1

Stable Audio Open 1.0 and Foundation-1 use the optional classic
stable-audio-tools (`SA3_BUILD_SAT`) pipeline introduced for Stable Audio Open
Small. They do not require a second inference implementation.

## Component reuse

The two checkpoints have the same loadable topology: a 24-layer, width-1536
classic continuous DiT with 24 query heads, grouped 12-head cross-attention,
64 latent channels, V-prediction, and both `seconds_start` and `seconds_total`
conditioning. Foundation-1 is a full finetune of that topology.

Both models use T5-base with a 128-token context. The encoder weights and
tokenizer are the same as SAOS, but the GGUF is kept distinct because its
`sat.t5.max_length` metadata is 128 instead of 64.

The 365 Oobleck tensors are byte-equivalent after normalizing source precision
across SAOS, SAO 1.0, and Foundation-1. The published SAOS Oobleck GGUF can
therefore be reused directly at every encoding tier.

Only these settings vary per model:

| Model | DiT weights | Published sample window | Suggested inference profile |
| --- | --- | ---: | --- |
| Stable Audio Open 1.0 | SAO 1.0 | 2,097,152 samples | DPM++ 3M SDE, sigma 0.3–500, 100 steps, CFG 7 |
| Foundation-1 | Foundation-1 | 882,000 samples | application-selected; see below |

## Shared runtime API

`sa3::sat::Pipeline` accepts the same three component paths used by SAOS:

- DiT GGUF
- T5-base encoder GGUF
- Oobleck GGUF

SAO 1.0 adds no new model execution entry point. Applications use the existing
`GenerateParams` fields, with `seconds_start`, `sigma_min`, `sigma_max`,
`sigma_rho`, and `sde_eta` populated for V-prediction. The runtime supports
both DPM++ 2M SDE and DPM++ 3M SDE.

A service that already fronts SAOS can reuse its prompt, seed, duration,
negative-prompt, progress, WAV, and response-metadata plumbing. The family
descriptor only needs to select the DiT, the 128-token T5, the maximum sample
window, `seconds_start`, and a sampler profile.

Foundation currently has two useful named profiles:

| Profile | Sampler | Sigma range | Steps | CFG |
| --- | --- | ---: | ---: | ---: |
| Gary fallback | DPM++ 2M SDE | 0.5–50 | 100 | 7 |
| RoyalCities UI | DPM++ 3M SDE | 0.01–100 | 100 | 7 |

These remain application policy rather than hard-coded graph behavior.

## Reference validation

The canonical SAO 1.0 and Foundation checkpoints each converted to 379 GGUF
tensors containing 1,057,335,680 DiT and numeric-conditioner parameters. A
three-step real-checkpoint Foundation comparison against stable-audio-tools
measured:

- native conditioning cosine: 0.9999973
- final latent cosine: 0.9999964
- decoded stereo audio cosine: 0.9999923

The native CUDA pipeline also completed matched 11-second, 100-step renders for
both Foundation profiles and the official SAO 1.0 profile.

## Preliminary quantization matrix

The generic quantizer required no SAT-specific tensor rules. Per-tensor checks
at a 0.990 cosine threshold passed all tested artifacts:

| Component | Q8 compared / failed | Q5_K_M compared / failed | Q4_K_M compared / failed |
| --- | ---: | ---: | ---: |
| Foundation DiT | 178 / 0 | 175 / 0 | 175 / 0 |
| SAO 1.0 DiT | 178 / 0 | 175 / 0 | 175 / 0 |
| T5-base 128 | 73 / 0 | 73 / 0 | 73 / 0 |

The complete bundles below use the matching tier for DiT, T5, and the shared
Oobleck. Metrics compare matched prompts, initial-noise seeds, and SDE noise
streams against F16. Envelope and log-magnitude cosine are more useful than raw
waveform cosine after a long stochastic trajectory.

| Tier | Bundle size | Foundation envelope / log-mag | SAO 1.0 envelope / log-mag |
| --- | ---: | ---: | ---: |
| F16 | 2,377 MiB | reference | reference |
| Q8_0 | 1,296 MiB | 0.9967 / 0.9901 | 0.9959 / 0.9784 |
| Q5_K_M | 922 MiB | 0.9904 / 0.9550 | 0.9021 / 0.8742 |
| Q4_K_M | 832 MiB | 0.9643 / 0.9284 | 0.9071 / 0.8789 |

CUDA end-to-end real-time factors for the SAO 1.0 run were 0.879 at F16,
0.754 at Q8, 0.785 at Q5, and 0.778 at Q4 on an RTX 5070 Laptop GPU. Foundation
showed the same roughly 10–15 percent quantized speed improvement.

These paired metrics measure divergence from one F16 trajectory, not absolute
audio quality. Q8 is the conservative high-fidelity tier. Q5 and Q4 need the
same ear-test gate used for SAOS before assigning publication recommendations;
all tiers are technically healthy and remain candidates for publication.

## Remaining gates

1. Complete ear tests across more than one Foundation and SAO 1.0 prompt.
2. Finalize family-scoped publication filenames and model repository layout.
3. Add catalog/downloader aliases without changing default SA3 builds or downloads.
4. Validate the stacked SAOS and SAO 1.0/Foundation changes on Metal.
