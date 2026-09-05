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
- `tools/saos-generate.cpp`: isolated experimental ping-pong pipeline and timing driver.
  It does not add SAOS branches to `sa3_pipeline.h` or the existing public API.

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

For CUDA, add `-DSA3_CUDA=ON` to the configure command. `SA3_BUILD_SAOS` is only
effective when the general `SA3_BUILD_TOOLS` option is enabled; both default to the
appropriate non-application behavior (`SA3_BUILD_SAOS` is always off).

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

## Remaining integration gates

1. Add the T5 precompiled Unicode normalization table; the current dependency-free
   tokenizer exactly covers ordinary prompt text but does not yet reproduce every NFKC-like
   normalization performed by SentencePiece for unusual Unicode compatibility characters.
2. Decide whether to expose the family through a reusable SAT pipeline class or keep it
   as application composition; do not put model-family branching in the SA3 pipeline.
3. Convert a Foundation-1 checkpoint and let metadata/topology validation determine
   which Oobleck/T5 artifacts it can share. Its larger DiT requires a separately sized
   graph even when the implementation is identical.

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
