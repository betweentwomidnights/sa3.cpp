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

`seconds` and `seconds_total` are deliberately separate. `seconds` describes the
requested output crop; `seconds_total` is the value passed to the learned numeric
conditioner and defaults to `seconds` for existing callers. Foundation's musical
profile needs these values to differ.

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

## Foundation musical timing

Foundation-1's trained grid is 4 or 8 bars at 100, 110, 120, 128, 130, 140,
or 150 BPM. `sat/foundation_timing.h` implements the complete grid, and
`sat/profiles.h` applies it to `GenerateParams` while appending the required
`N Bars, N BPM` prompt conditioning.

The profile matches RoyalCities' three separate lengths:

1. exact output samples: `round((60 / BPM) * 4 * bars * 44100)`;
2. learned `seconds_total`: the exact duration rounded up to a whole second;
3. latent canvas: that conditioned duration rounded up to the 2,048-sample
   Oobleck stride.

The decoded result is then cropped to the exact musical sample count. Unsupported
BPMs are rejected rather than silently mapped to a different tempo.

Pitch-preserving time stretching for arbitrary host tempos remains an open application
feature. It works well in gary4local, but bringing it into this repository would require
a deliberately selected DSP implementation and dependency/licensing policy; it does not
belong in the DiT/T5/Oobleck primitives.

The optional `sat-generate` frontend exposes Foundation-specific prompt randomization
without putting application policy into the pipeline. `--randomize` uses a deterministic
C++ port of RoyalCities' weighted vocabulary and M1/T1 organization. `--randomize-mode
mix` selects T1 and `--family` locks the anchor family. The audio seed drives the prompt
selection too, and the command reports the exact prompt, seed, variant, musical geometry,
and output path. This is deliberately one generate operation rather than a separate
randomization endpoint.

```powershell
sat-generate --model foundation-1 --randomize --randomize-mode mix `
  --family Synth --bars 4 --bpm 128 --key-root F# --key-mode minor --seed 42
```

## Publication layout

SAO 1.0 and Foundation-1 are packaged as separate, self-contained repositories:

- `thepatch/stable-audio-open-1.0-GGUF`
- `thepatch/foundation-1-GGUF`

Both use the same canonical 128-token T5 and Oobleck filenames, and those shared
artifacts are byte-identical between repositories. `tools/stage_sat_large_repos.py`
constructs both release directories, verifies source licenses, and writes SHA-256
manifests. `tools/download_models.py --sat --sat-model ...` resolves either family
without changing the default SA3 download.

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

The musical-timing integration was subsequently exercised through the packaged
Q5 Foundation bundle at both 4 bars / 128 BPM and the longest 8 bars / 100 BPM
boundary. The latter used 431 latent frames conditioned with `seconds_total=20`
and cropped to exactly 846,720 stereo samples (19.200 s), including the canvas
extension 688 samples beyond the nominal 882,000-sample checkpoint window.
The packaged SAO 1.0 Q8 bundle also completed its full 1,024-frame context and
cropped it to 2,072,700 samples (47.000 s). Both ran on CUDA against the pinned
GGML submodule.

## Quantization matrix

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
audio quality. All tiers passed listening tests and remain publication candidates.
Q8 is the conservative high-fidelity tier; Foundation Q5 is the recommended balance,
while SAO 1.0's larger measured trajectory drift makes Q8 its conservative default.

## Remaining gates

1. Validate the stacked SAOS and SAO 1.0/Foundation changes on Metal.
