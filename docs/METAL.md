# metal backend — validation checklist (for the m4)

status: **GENERATION PASS on Apple M4.** medium f16 text-to-music generates successfully on the
ggml Metal backend, matches CPU closely, is deterministic run-to-run, and now exits cleanly with
ggml's default Metal residency sets enabled.

## 0. prerequisites

- Xcode command-line tools (`xcode-select --install`) — provides clang + the Metal toolchain.
- cmake on PATH (`brew install cmake`).
- Python for the converters/checks (`python3 -m pip install huggingface_hub numpy`).

## 1. build

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp
./build.sh metal        # -> build-metal/   (SA3_METAL=ON; ggml also auto-picks Accelerate BLAS)
```

ggml embeds the Metal shaders at build time, so there's no separate SDK to install (unlike Vulkan's
glslc). if the build can't find the Metal framework, confirm the Xcode CLT are installed.

## 2. get a model set

the GGUF repos are live under the `thepatch` org. **they're private for now** (until the Stability
review), so authenticate with a token that can read them, then download:

```bash
python3 -m pip install huggingface_hub
hf auth login                                              # paste a token with read access to thepatch
python3 tools/download_models.py --variant medium --encoding f16
```

this pulls the medium f16 set (~5.7 GB) + the shared encoder/tokenizer into `models/`. once the repos
are public the `hf auth login` step is unnecessary. while the repos are private, a fine-grained token
must grant `repo.content.read` on the `thepatch` org; a token scoped only to the user account can log in
successfully but still get a 404 for private org repos. (the binaries land in `build-metal/bin/`.)

## 3. device + memory notes

- one Apple GPU, so `make_backend` (src/gguf_model.h) picks it with no fuss; `SA3_DEVICE=cpu` forces
  the CPU backend for the A/B below. `SA3_GPU` is there if you ever need to pin a device.
- **unified memory is a big deal here:** M-series shares RAM with the GPU, so the 8GB-discrete VRAM
  thrash we fought on the 5070 (and the f32-medium OOM under Vulkan) should NOT happen — f32 medium
  likely fits. try f16 first (production path) but f32 is worth a shot for an exact-parity check.

## 4. validate (the right way)

**the key lesson from Vulkan:** raw waveform cosine is the WRONG end-to-end metric for a backend whose
matmuls aren't bit-exact. Vulkan's tensor-core path made full-gen raw cosine 0.72 vs CPU while the audio
was perceptually identical (log-mag spectrogram cosine 0.95, rms envelope ~perfect). Metal's matmul may
likewise differ slightly from CPU. so:

1. **determinism** — run the same gen twice on Metal, expect byte-identical output (cosine 1.0). if it
   drifts run-to-run, Metal is using nondeterministic reductions; flag it.
2. **GPU-vs-CPU, same binary** — generate once on Metal, once with `SA3_DEVICE=cpu`, same seed/prompt:
   ```bash
   B=./build-metal/bin/sa3-generate
   ARGS="--tok models/t5gemma-b-b-ul2-v1.0-vocab.gguf --t5 models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
         --cond models/stable-audio-3-medium-conditioner-v1.0-F32.gguf \
         --dit models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
         --same models/stable-audio-3-medium-same-l-v1.0-F16.gguf \
         --prompt 'upbeat funk groove with slap bass' --frames 128 --steps 8 --seed 0"
   $B $ARGS --out metal.wav
   SA3_DEVICE=cpu $B $ARGS --out cpu.wav
   ```
   then compare with **envelope + log-mag-spectrogram cosine**, not raw waveform cosine:
   ```python
   import wave, numpy as np
   def rd(p):
       w=wave.open(p,'rb'); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float64); w.close(); return a
   a,b=rd('metal.wav'),rd('cpu.wav'); n=min(len(a),len(b)); a,b=a[:n],b[:n]
   def logmag(x,win=2048,hop=512):
       f=np.array([np.abs(np.fft.rfft(x[i:i+win]*np.hanning(win))) for i in range(0,len(x)-win,hop)])
       return np.log1p(f).ravel()
   def env(x,w=2205): return np.sqrt(np.array([np.mean(x[i:i+w]**2) for i in range(0,len(x)-w,w)]))
   cos=lambda u,v:float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-12))
   print('raw waveform cosine :', cos(a,b), '  <- expected to be low if matmuls differ; not a failure')
   print('rms envelope cosine :', cos(env(a),env(b)))
   print('log-mag spec cosine :', cos(logmag(a),logmag(b)))
   ```
   **pass:** envelope + log-mag cosine ~0.95+ and the two clips sound the same. (if f32 medium fits in
   unified memory, an f32 Metal-vs-CPU run may even land near raw cosine 1.0 — worth recording.)
3. **isolated-forward parity (optional, stronger)** — if you bring over `refdata/` inputs from the
   Windows box, `sa3-dit` / `sa3-codec` give Metal-vs-CPU cosine on a single forward (these read ~1.0
   on CPU/CUDA; Vulkan's coopmat path read 0.9999). not required if the end-to-end perceptual check passes.
4. **speed** — time a 12s medium f16 gen; for reference CUDA ~3.5s, Vulkan ~4.1s on a 5070 laptop.

## 5. risks to watch

same op set as Vulkan, which all worked: cast, rms_norm, soft_max_ext with a 3D mask, 4D batched
mul_mat, the f16 `ggml_cpy`, and the SAME-L local-attn concat/views. ggml-metal is mature, so coverage
should be fine — but if any op is unsupported the run errors (not silently wrong); note which and we
either tweak the graph or fall back. when it's green, write up results here like VULKAN.md and update
the roadmap (Metal = the last backend).

## 6. m4 results (2026-06-29)

Test machine: Apple M4, 32 GB unified memory, macOS 15, ggml commit `eced84c8`.
Command shape: stable-audio-3-medium, DiT/SAME f16 GGUF, conditioner/T5 f32 GGUF, 8 steps,
prompt `upbeat funk groove with slap bass`, seed `0`.

Early bring-up used one Metal backend per GGUF and needed `GGML_METAL_NO_RESIDENCY=1` as a teardown
workaround. `sa3-generate` now shares one backend across T5, conditioner, DiT, and SAME, so the same
run exits cleanly with default residency sets enabled. The default-residency and no-residency outputs
were byte-identical for a 128-frame / 8-step check.

### device smoke

`sa3-smoke` sees:

- `GPU MTL0 Apple M4` with 21.3/21.3 GiB free
- `ACCEL BLAS Accelerate`
- `CPU Apple M4` with 32.0/32.0 GiB free

### parity

| comparison | byte-identical | raw waveform cosine | rms envelope cosine | log-mag spec cosine |
|---|---:|---:|---:|---:|
| Metal run 1 vs Metal run 2 | yes | 1.000000 | 1.000000 | 1.000000 |
| Metal vs CPU, same binary | no | 0.996556 | 0.999541 | 0.998519 |

### speed

End-to-end wall clock includes model load, Metal pipeline compilation, generation, decode, and WAV
write.

| backend | output | frames | steps | wall time |
|---|---:|---:|---:|---:|
| Metal | 12s | 128 | 8 | 6.21 s |
| CPU | 12s | 128 | 8 | 29.74 s |
| Metal | 30s | 323 | 8 | 12.14 s |
| Metal | 60s | 646 | 8 | 22.05 s |
| Metal | 120s | 1292 | 8 | 44.51 s |

One profiled 12s Metal run (`SA3_PROFILE=1`) reported:

| stage | time |
|---|---:|
| load tokenizer | 0.13 s |
| load T5 | 0.17 s |
| load DiT | 0.38 s |
| load SAME | 0.23 s |
| T5 compute | 0.06 s |
| DiT total | 3.03 s |
| SAME decode total | 1.92 s |

MLX comparison, using `sa3-mlx/scripts/benchmark_mlx_text_to_audio.py` with one warmup and one
measured iteration. Setup is torch checkpoint load + torch-to-MLX conversion in that script, so the
generation columns are the better steady-state UX proxy.

| backend | output | duration padding | latent length | measured generation | sampling | decode | setup |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLX generic SAME | 12s | 6s | 194 | 5.81 s | 3.21 s | 2.56 s | 30.20 s |
| MLX generic SAME | 12s | 0s | 130 | 4.16 s | 2.62 s | 1.50 s | 18.45 s |
| MLX generic SAME | 120s | 0s | 1292 | 92.68 s | 16.12 s | 76.47 s | 18.74 s |
| MLX generic SAME, chunked decode | 120s | 0s | 1292 | 45.00 s | 18.23 s | 26.72 s | 20.21 s |
| MLX official SAME-L decoder | 120s | 0s | 1292 | 31.69 s | 17.64 s | 13.95 s | 22.19 s |

Takeaway: ggml Metal is not the fastest short-generation path versus MLX, but it scales much better
than the generic MLX SAME decode. For long-form UX, it roughly ties generic MLX with chunked decode
and trails the official optimized MLX SAME-L decoder.

## 7. metal optimization notes

The practical Metal-specific fix was backend lifetime: keep one `ggml_backend_t` alive for every model
buffer used by a generation, then free it after all GGUF buffers are released. That avoids repeated
Metal initialization and fixes the residency-set teardown assert without disabling residency.

Runtime knobs tested on the 12s / 128-frame / 8-step medium run:

| setting | result |
|---|---|
| default Metal settings | best overall; fusion, concurrency, graph optimization, shared buffers on |
| `GGML_METAL_NO_RESIDENCY=1` | same output and same speed class; no longer required after shared backend |
| `GGML_METAL_TENSOR_ENABLE=1` | no effect on this M4; ggml still reports `has tensor = false` |
| `GGML_METAL_SHARED_BUFFERS_DISABLE=1` | no speedup |
| `GGML_METAL_FUSION_DISABLE=1` | slight regression |
| `GGML_METAL_CONCURRENCY_DISABLE=1` | slight regression |
| `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1` | slight regression |

Outer chunked SAME-L decode was added to `sa3-generate` as an opt-in compatibility/memory lever:
`--chunked-decode` (default `--decode-chunk-size 128 --decode-overlap 32`). It mirrors the
`decode_audio` overlap tiling used by the PyTorch/MLX path, but it is not a speedup on this GGML Metal
graph. The reason is structural: SAME-L attention is already computed with `nn::attn_sliding`, an
overlapping-block local-attention graph that scales linearly with length; outer tiling mostly repeats
decoder work in the overlap.

Chunked-decode checks on Apple M4, same prompt/seed/model/8-step setup:

| mode | output | chunks | SAME decode compute | wall time | audio delta vs monolithic |
|---|---:|---:|---:|---:|---:|
| monolithic | 12s | 1 | 1.92 s | 13.74 s* | reference |
| chunked 128/32 | 12s | 1 | 1.91 s | 7.34 s | byte-identical |
| monolithic | 23.8s | 1 | 3.92 s | 9.79 s | reference |
| chunked 128/32 | 23.8s | 3 | 6.55 s | 12.47 s | cosine 0.9999996 / -61.1 dB RMS |
| monolithic | 120s | 1 | 23.42 s | 50.65 s | reference |
| chunked 128/32 | 120s | 14 | 31.78 s | 59.17 s | cosine 0.9999986 / -55.5 dB RMS |

`*` the 12s monolithic run included a cold post-build Metal pipeline/library load; the single-chunk
comparison is useful for correctness, not wall-clock ranking.

`GGML_OP_FLASH_ATTN_EXT` is a real SAME-L decoder win on Metal when enabled for the decoder only:

```bash
SA3_FLASH_ATTN=0 SA3_SAME_FLASH_ATTN=1 ./build-metal/bin/sa3-generate ...
```

This path builds a full F16 band mask `[N, N]` for SAME-L and feeds the natural full K/V sequence to
`ggml_flash_attn_ext`. That is a quadratic mask/attention shape, unlike the default compact
`nn::attn_sliding` graph, so it is currently opt-in rather than the cross-backend default. On this M4,
the flash kernel is fast enough to win through the 120s UX target despite the larger mask.

Decoder-only flash-attention A/B:

| mode | output | frames | mask size | SAME decode compute | wall time | audio delta vs default |
|---|---:|---:|---:|---:|---:|---:|
| default `nn::attn_sliding` | 12s | 128 | compact | 1.92 s | 13.74 s* | reference |
| `SA3_SAME_FLASH_ATTN=1` | 12s | 128 | 9 MiB | 1.27 s | 5.75 s | cosine 0.9999969 / -51.9 dB RMS |
| default `nn::attn_sliding` | 23.8s | 256 | compact | 3.92 s | 9.79 s | reference |
| `SA3_SAME_FLASH_ATTN=1` | 23.8s | 256 | 36 MiB | 2.60 s | 8.52 s | cosine 0.9999954 / -50.4 dB RMS |
| default `nn::attn_sliding` | 60s | 646 | compact | 9.99 s | 21.99 s | reference |
| `SA3_SAME_FLASH_ATTN=1` | 60s | 646 | 230 MiB | 7.25 s | 19.17 s | cosine 0.9999931 / -48.6 dB RMS |
| default `nn::attn_sliding` | 120s | 1292 | compact | 23.42 s | 50.65 s | reference |
| `SA3_SAME_FLASH_ATTN=1` | 120s | 1292 | 920 MiB | 16.38 s | 40.44 s | cosine 0.9999917 / -47.8 dB RMS |

Use `SA3_SAME_FLASH_ATTN=1` for the decoder-only experiment. The broader `SA3_FLASH_ATTN=1` flag also
switches DiT attention to flash attention; that may be useful later, but it changes the sampled latents
more noticeably and is not the clean SAME-L A/B.

The local/windowed flash variant was also tested:

```bash
SA3_FLASH_ATTN=0 SA3_SAME_FLASH_ATTN=local ./build-metal/bin/sa3-generate ...
```

This keeps the compact `[3*sub_chunk, sub_chunk, nb]` SAME-L neighborhood shape and uses
`ggml_flash_attn_ext` inside each local block. It is correct and preserves linear mask memory, but the
current Metal flash-attention vector kernel is slower for these small windows than the default compact
matmul/softmax graph.

Local/windowed flash A/B:

| mode | output | frames | mask size | SAME decode compute | wall time | audio delta vs default |
|---|---:|---:|---:|---:|---:|---:|
| default `nn::attn_sliding` | 12s | 128 | compact F32 | 1.92 s | 13.74 s* | reference |
| `SA3_SAME_FLASH_ATTN=local` | 12s | 128 | 0.2 MiB | 2.53 s | 7.20 s | cosine 0.9999969 / -51.9 dB RMS |
| default `nn::attn_sliding` | 23.8s | 256 | compact F32 | 3.92 s | 9.79 s | reference |
| `SA3_SAME_FLASH_ATTN=local` | 23.8s | 256 | 0.4 MiB | 5.18 s | 11.68 s | cosine 0.9999954 / -50.4 dB RMS |
| default `nn::attn_sliding` | 60s | 646 | compact F32 | 9.99 s | 21.99 s | reference |
| `SA3_SAME_FLASH_ATTN=local` | 60s | 646 | 1.1 MiB | 14.38 s | 27.54 s | cosine 0.9999931 / -48.6 dB RMS |
| default `nn::attn_sliding` | 120s | 1292 | compact F32 | 23.42 s | 50.65 s | reference |
| `SA3_SAME_FLASH_ATTN=local` | 120s | 1292 | 2.1 MiB | 30.65 s | 57.29 s | cosine 0.9999917 / -47.8 dB RMS |

So the current recommendation is:

- use default `nn::attn_sliding` for the portable linear-memory path;
- use `SA3_SAME_FLASH_ATTN=1` on M4 when the full F16 mask memory is acceptable;
- keep `SA3_SAME_FLASH_ATTN=local` as an experimental reference unless ggml gains a faster small-window
  flash/local-attention Metal kernel.

Upstream check: standalone ggml is already at `eced84c8`, and current `llama.cpp`'s
`ggml/src/ggml-metal` matched this submodule during the test. There is no obvious "pull newer Metal"
win as of this run. Source references: ggml's
[`ggml-metal-device.m`](https://github.com/ggml-org/ggml/blob/master/src/ggml-metal/ggml-metal-device.m),
[`ggml-metal-context.m`](https://github.com/ggml-org/ggml/blob/master/src/ggml-metal/ggml-metal-context.m),
and llama.cpp's vendored
[`ggml/src/ggml-metal`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal).

Next high-leverage work is graph/model-shape optimization rather than toggling Metal flags:

- Port or reproduce any non-tiling pieces of the optimized SAME-L decoder path in ggml; outer chunked
  decode by itself is correct, but slower on Metal.
- Revisit local/windowed flash only if ggml's Metal backend gets a faster small-window/local flash
  kernel. The compact graph is implemented and correct, but is slower than both default sliding
  attention and full-mask flash on this M4.
- For an app/server backend, keep models resident and cache reusable graphs/allocators per duration
  bucket. The CLI numbers include load and pipeline setup; Gary's UX path should not pay that every
  request.
