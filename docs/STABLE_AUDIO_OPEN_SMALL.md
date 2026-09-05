# Stable Audio Open Small implementation notes

This branch adds the classic `stable-audio-tools` model family beside, rather than
inside, the Stable Audio 3 pipeline. The family namespace is `sa3::sat`; existing SA3
model configuration and public APIs remain unchanged.

## Compatibility boundary

Checkpoint compatibility is determined by tensor topology, not by the words “Stable
Audio” or “finetune.” SAOS uses a 16-layer, 1024-wide DiT with eight heads and T5
conditioning from `prompt` and `seconds_total`. SAO 1.0 and Foundation-1 use the larger
24-layer, 1536-wide, 24-head topology and add `seconds_start`. They can share T5-base and
the Oobleck architecture, but they cannot share a converted DiT or a single instantiated
DiT graph.

`sat::weight_topology_compatible` intentionally ignores inference-only settings such as
sample length, objective, sampler, step count, and CFG. A genuine SAOS finetune can
change those settings while retaining loadable tensor shapes.

## Landed primitives

- `src/sampling.h`: model-independent RF logSNR schedule and ping-pong update. The
  existing SA3 pipeline calls the same update primitive without changing its schedule.
- `src/sat/t5.h`: classic T5 encoder, including bidirectional learned relative-position
  buckets. This stays separate from the architecturally different T5Gemma encoder.
- `src/sat/tokenizer.h`: SentencePiece-Unigram/Viterbi tokenization using vocabulary and
  scores embedded in the T5 GGUF; no protobuf or SentencePiece runtime dependency.
- `src/sat/conditioning.h`: the learned `seconds_total` Fourier/linear conditioner.
- `src/sat/oobleck.h`: config-driven Oobleck decoder. The ConvTranspose1d implementation
  uses GEMM plus `ggml_col2im_1d`, generalized from the tested acestep.cpp implementation.
- `src/sat/dit.h`: the classic ContinuousTransformer DiT, including learned Fourier
  timestep features, prepended global token, partial NeoX RoPE, affine Q/K layer norms,
  cross attention, and SwiGLU feed-forward blocks.
- `src/sat/model_spec.h`: SAOS model geometry, inference defaults, validation, and
  checkpoint topology checks.
- `src/sat/pipeline.h`: reusable application component for staged T5, classic-DiT,
  and Oobleck execution, with objective-aware sampler/step/CFG defaults.
- `tools/saos-generate.cpp`: isolated experimental command-line and timing driver.
  It is a thin frontend over `sa3::sat::Pipeline` and does not add SAOS branches to
  `sa3_pipeline.h` or the existing public API.

## Conversion and validation

`tools/convert_sat_oobleck.py` fuses weight normalization before writing
regular convolution weights. ConvTranspose1d weights must additionally be transposed to
`[in_channels, kernel * out_channels]` for the GEMM/col2im graph. Snake parameters are
stored as `exp(alpha)` and `1 / exp(beta)` so the inference graph does not repeat those
constant transforms.

Canonical names are documented at the top of `src/sat/oobleck.h`. Keeping the conversion
transform explicit lets the runtime load tensors with the existing model-agnostic
`GgufModel` instead of introducing an Oobleck-specific loader.

`tools/convert_sat_dit.py` converts the classic DiT plus the checkpoint-specific learned
duration conditioner, and rejects missing or unknown classic-DiT tensors. Large matrix
weights default to F16 while normalization, bias, and learned Fourier state remain F32.
`tools/convert_sat_t5.py` packages the shared T5-base encoder and Unigram tokenizer into
a single GGUF. All converters store topology metadata in their GGUF.

`tools/dump_saos_reference.py` produces deterministic PyTorch trajectory and decoder
fixtures. On the production checkpoint, the F16 GGUF CUDA implementation measured
0.999997 latent cosine similarity and 0.999991 decoded-audio cosine similarity against a
float32 PyTorch reference for the one-step parity case.

## Current end-to-end workflow

The normal generation path is now entirely native. `tools/export_saos_conditioning.py`
remains only as an independent Transformers parity oracle. On the production artifacts,
native prompt conditioning measured 0.999998 cosine similarity and the learned duration
vector measured 1.000000 against that oracle.

## Optional build target

SAOS is deliberately excluded from the default build. The core `sa3` library and
existing SA3 command-line tools therefore do not acquire a Stable Audio Open Small
executable or model-specific runtime dependency.

Enable the isolated runner and its focused tests explicitly:

```powershell
cmake -S . -B build-saos -DSA3_BUILD_SAOS=ON
cmake --build build-saos --config Release --target saos-generate
```

This also creates the `sa3_saos` static-library target. An embedding application can
link that component directly and consume tightly packed planar float audio from
`sa3::sat::Pipeline::generate`; it does not need to invoke or link the CLI.

For CUDA, add `-DSA3_CUDA=ON` to the configure command. `SA3_BUILD_SAOS` creates the
component even when `SA3_BUILD_TOOLS=OFF`; the latter only controls whether the
`saos-generate` frontend and focused test executables are also created.

```powershell
python tools/convert_sat_dit.py --src model.safetensors --config model_config.json --out saos-dit-f16.gguf
python tools/convert_sat_oobleck.py --src model.safetensors --config model_config.json --out saos-oobleck-f16.gguf
python tools/convert_sat_t5.py --src t5-base/model.safetensors --config t5-base/config.json `
  --tokenizer t5-base/tokenizer.json --out t5-base-encoder-f16.gguf
$env:SA3_DEVICE="cuda"; $env:SA3_FLASH_ATTN="1"
saos-generate --t5 t5-base-encoder-f16.gguf `
  --prompt "A short, beautiful piano riff in C minor" --seconds 11 `
  --dit saos-dit-f16.gguf --ae saos-oobleck-f16.gguf `
  --frames 256 --steps 8 --samples 485100 --wav saos.wav
```

The model is sampled at its trained 256 latent frames (524,288 decoded samples), then
the WAV is cropped to exactly 485,100 samples/11 seconds. The timing line separates
weight loading, graph build/allocation, eight-step denoising, and Oobleck decoding.

For a topology-compatible pre-ARC/full-checkpoint finetune, convert its `.ckpt`
directly (Lightning `state_dict` and EMA wrappers are detected safely), then use Euler
or RF DPM++ with guidance. When omitted, a `rectified_flow` GGUF resolves to Euler,
50 steps, and CFG 4; `rf_denoiser` continues to resolve to ping-pong, 8 steps, and CFG 1.

```powershell
python tools/convert_sat_dit.py --src finetune.ckpt --config finetune_config.json `
  --out finetune-dit-f16.gguf --model-id my-saos-finetune
saos-generate --t5 t5-base-encoder-f16.gguf --dit finetune-dit-f16.gguf `
  --ae saos-oobleck-f16.gguf --prompt "..." --sampler dpmpp --steps 40 `
  --cfg-scale 4 --seconds 11 --peak-normalize --wav finetune.wav
```

Peak normalization is a CLI presentation option matching Gary's current service;
the reusable component returns unclipped planar float samples so embedding applications
retain control over loudness and limiting.

## Full-checkpoint validation

Two independent `rectified_flow` finetunes have been converted and run through the
native CUDA component with the shared SAOS T5 and Oobleck artifacts:

- `S3Sound/kickbass`, checkpoint `kickbass_v1_saos_e257_s18870.ckpt`: an unwrapped
  750-tensor full checkpoint. RF DPM++ at 40 steps/CFG 4 generated 11 seconds in
  about 2.5 seconds total on an RTX 5070 Laptop GPU.
- `thepatch/jerry_grunge`, checkpoint
  `jerry_encoded_bs64_epoch=374-step=3000.ckpt`: a 5.77 GB preencoded Lightning
  training snapshot. Conversion selected its 382-tensor EMA DiT, paired it with the
  frozen duration conditioner, and omitted optimizer, loss, and autoencoder state.
  RF DPM++ at 40 steps/CFG 4 likewise generated 11 seconds in about 2.5 seconds. The
  validation prompt was `chillhop synth warm bass dusty drums sidechain 90 bpm`.

The objective-driven no-override path was also exercised on KickBass and resolved to
Euler, 50 steps, and CFG 4 as intended.

## Quantization validation

The existing model-agnostic `sa3-quantize` path works without SAT-specific tensor
branches. ARC, KickBass, and Jerry Grunge were each converted from F16 to Q8_0,
Q5_K_M, and Q4_K_M. `sa3-quant-check` compared 122 quantized tensors at Q8_0 and
119 at each K-quant tier for every DiT; all nine artifacts had zero tensors below
0.990 cosine similarity.

Each complete bundle below uses the same tier for its DiT, T5-base, and Oobleck.
Audio comparisons use matched prompts/seeds against the all-F16 bundle. RMS-envelope
and log-magnitude-spectrum cosine are reported because raw waveform cosine overstates
small diffusion-trajectory and phase changes.

| Tier | Complete bundle | ARC envelope / log-mag | Jerry envelope / log-mag |
| --- | ---: | ---: | ---: |
| F16 | 1,010 MiB | reference | reference |
| Q8_0 | 569 MiB | 0.9995 / 0.9917 | 0.9999 / 0.9986 |
| Q5_K_M | 416 MiB | 0.9937 / 0.9895 | 0.9992 / 0.9977 |
| Q4_K_M | 379 MiB | 0.9383 / 0.9832 | 0.9973 / 0.9964 |

The all-Q4 KickBass bundle measured 0.9952 envelope and 0.9855 log-magnitude
cosine. DiT-only Q4 tests were stronger than the cumulative ARC result (0.9666
envelope and 0.9870 log-magnitude), confirming that ARC is more sensitive to the
combined T5 and DiT perturbation rather than identifying a corrupt quantized tensor.

The shared artifacts were also isolated:

- T5-base conditioning cosine was 0.999895 at Q8_0, 0.998273 at Q5_K_M, and
  0.994069 at Q4_K_M. A full Jerry generation with only T5 quantized retained
  0.9990, 0.9987, and 0.9979 log-magnitude cosine respectively.
- Oobleck Q8/Q5/Q4 retained 0.9990, 0.9987, and 0.9983 log-magnitude cosine.
  Most Oobleck convolution tensors remain F16 because the generic quantizer only
  quantizes compatible two-dimensional matrices, so its size reduction is modest.

Publish F16, Q8_0, Q5_K_M, and Q4_K_M components. Label Q5_K_M as the recommended
download: it is only 37 MiB larger than an all-Q4 bundle, while staying robust across
both objectives. Q8_0 is the near-transparent tier, F16 is the reference tier, and
Q4_K_M is the space-first tier. CUDA load time falls substantially with quantization;
end-to-end time improved by roughly 10–15% in back-to-back 11-second runs, while
denoising speed on this relatively small DiT was close enough to treat as backend- and
thermal-state-dependent rather than promise a fixed speedup.

## Remaining integration gates

1. Add the T5 precompiled Unicode normalization table; the current dependency-free
   tokenizer exactly covers ordinary prompt text but does not yet reproduce every NFKC-like
   normalization performed by SentencePiece for unusual Unicode compatibility characters.
2. Convert a Foundation-1 checkpoint and let metadata/topology validation determine
   which Oobleck/T5 artifacts it can share. Its larger DiT requires a separately sized
   graph even when the implementation is identical.
3. Publish the validated SAOS and finetune GGUFs. Prefer one SAOS GGUF model repository
   with shared T5/Oobleck artifacts at the top level and a directory per DiT variant
   (`arc`, `base`, `kickbass`, and `jerry-grunge`). Include source-checkpoint revision,
   conversion command, tensor/weight type, recommended sampler defaults, license and
   attribution, checksums, and a short reproducible generation example for each variant.

## Deferred LoRA question

The official SAOS repository includes the pre-ARC `base_model.ckpt` and matching
`base_model_config.json` alongside the released ARC checkpoint. This gives a future
base-to-ARC adapter experiment a controlled source pair with identical DiT topology.
It does not make SAO 1.0 or Foundation-1 adapters compatible with SAOS; those models
use the separate larger DiT topology described above. Native SAT LoRA inference and
training remain deliberately outside this initial inference milestone.

The Oobleck converter includes complete decoder
tensor coverage checks, weight-norm fusion, col2im transformation, and F16/F32 output.
`tools/dump_sat_oobleck_refs.py` loads only the upstream decoder and writes deterministic
time-major latent/audio NPZ fixtures plus a JSON provenance sidecar. Passing `--taps`
also records every top-level decoder layer for locating the first numerical divergence.

The encoder half of Oobleck is deferred until audio-to-audio is in scope; text-to-audio
generation only requires the decoder.
