# metal backend — validation checklist (for the m4)

status: **GENERATION + TRAINING PASS on Apple M4.** medium f16 text-to-music is deterministic and
native small/medium LoRA/DoRA training completes on ggml Metal. The training path is correct and
memory-efficient; the optimized `OUT_PROD` path brings matched 512-frame training to within 19% of
MLX while retaining the checkpointed graph's much lower memory use.

> **Every Metal number in this document comes from one machine: a MacBook Air (`Mac16,13`), base
> M4, 10-core GPU, 32 GB unified memory, fanless.** That is the *weakest* M4 Apple ships, and it is
> the only Apple hardware available for testing here. Treat these timings as the floor of the M4
> family rather than a representative Apple figure — an M4 Pro/Max has 2-4x the GPU cores, far more
> memory bandwidth, and active cooling.
>
> Cooling matters as much as core count for training. Sustained runs on this Air throttle roughly
> 50% from cold, plateauing around step 5-7 (`bora` at 512 frames: 26.97 s cold → 40.3 s
> sustained). Long training runs on a cooled Mac should hold closer to their cold-start rate, so
> the gap on real workloads is likely wider than the per-step numbers here suggest. Memory
> footprints, correctness results, and cross-family *ratios* carry across the family; absolute
> seconds do not.

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

Test machine: MacBook Air (`Mac16,13`), base M4, 10-core GPU, 32 GB unified memory, fanless,
macOS 15, ggml commit `eced84c8`. See the note at the top — this is the slowest M4 there is.
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

## 7. native training results (2026-07-16)

Test machine: MacBook Air (`Mac16,13`), base M4, 10-core GPU, 32 GB unified memory, fanless,
macOS 15.7.3 — the slowest M4, and it throttles on sustained runs. Candidate base: ggml v0.16.0 plus the
downstream Metal training branch. The implementation adds native Metal `REPEAT_BACK`, `OUT_PROD`,
`SILU_BACK`, `RMS_NORM_BACK`, and `SOFT_MAX_BACK`, supports the strided F32 binary views produced by
autodiff, and fixes the non-inplace `ACC` copy dispatch for rows wider than one Metal threadgroup.
Unsupported shapes remain explicit; training operations are not silently routed to CPU.

Correctness gates passed:

- all 37 registered CTest tests, including ggml's full backend-op suite;
- Metal backend-op coverage: `OUT_PROD` 92/92, `REPEAT_BACK` 10/10, `ACC` 7/7,
  `SILU_BACK` 1/1, `RMS_NORM_BACK` 10/10, and `SOFT_MAX_BACK` 24/24;
- a two-step small-music run with finite losses/gradient norms and valid adapter/trainer-state files;
- CPU trainer-state resume on Metal, with a finite resumed update and valid output checkpoints;
- matched one-step CPU/Metal parity using the exact same latent, noise, noised input, target, and
  conditioning tensors: aggregate adapter-gradient cosine `0.9999853`, relative L2 `0.005433`;
- all 576 first-step adapter gradients were compared, and every `lora_A` gradient was zero on both
  backends as required by zero-initialized `lora_B`;
- the frozen medium inference WAV remained byte-identical before and after the training patch
  (`SHA-256 32276296285ec02487d1882688a8baaecc58d07343028ec3a5fa40da1576a0e6`).

The real 512-frame gate used the ten-track Ratatat dataset and the exact medium latents produced by
the earlier gary4local/PyTorch job. Default medium DoRA-rows rank/alpha 16 adapts 228 DiT targets.
Steps 2-5 exclude graph setup and pipeline compilation:

| kernel | frames | sampled steps | T5 | prep | DiT fwd/bwd | AdamW | total / step | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar correctness baseline | 512 | 2-5 | 35.3 ms | 1.0 ms | 22,273.8 ms | 53.8 ms | **22,364.3 ms** | **5.76 GiB** |
| 32x16 SIMD-group tile | 512 | 2-5 | 29.8 ms | 1.0 ms | 8,574.5 ms | 43.0 ms | **8,648.8 ms** | **5.75 GiB** |

A matched local gary4local MLX run used the same medium RF architecture, 512-frame crop,
DoRA-rows rank/alpha 16, and all 228 targets. MLX averaged **7.29 s/step** over steps 2-5, so the
optimized C++ Metal path is **1.19x slower**, down from 3.07x for the scalar kernel. MLX used
**16.27 GiB maximum RSS** and a 23.79 GiB peak memory footprint; C++ used 5.75 GiB maximum RSS and
a 5.52 GiB peak footprint. This is a throughput comparison, not exact loss parity: the trainers
use different latent encodes, random streams, and slightly different optimizer details.

The optimized Metal `OUT_PROD` kernel stages 32x16 and 16x16 operand tiles in threadgroup memory
and uses eight SIMD-group matrix accumulators to produce a 32x16 output tile. It retains arbitrary
byte strides, batch broadcasting, partial tiles, F32/F32, and frozen-F16/F32 support. Relative to
the scalar correctness kernel it improves matched medium training by **2.59x**. The two-step small
adapter and five-step medium loss trajectory remained byte-for-byte/print-identical, the frozen
inference WAV remained byte-identical, all 92 focused `OUT_PROD` cases passed, and the complete
37-test suite passed. A 2,500-update run now projects to about **6 h 0 min**, versus 15 h 32 min for
the scalar path and 5 h 4 min for MLX.

The full 2,000-update Ratatat run then completed in **5 h 29 min 55 s** including startup and four
checkpoint writes. Steps 2-2,000 averaged 9.895 s under sustained load (10.301 s over the final 500
steps), with 5.76 GiB maximum RSS and a 5.52 GiB peak footprint. All metric records were finite,
the final adapter matched the step-2,000 checkpoint byte-for-byte, and an 8-step/CFG-1 base render
remained byte-identical to the retained healthy inference reference.

## 8. metal optimization notes

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

## 9. quantization results (2026-07-25)

Test machine: MacBook Air (`Mac16,13`), base M4, 10-core GPU, 32 GB unified memory, fanless,
macOS 15 (Darwin 24.6.0). Branch
`feature/q4km-quantization`, ggml submodule `e9c70fd6`. Model: stable-audio-3-medium, 30 s output
(324 frames), 8 steps, prompt `cinematic orchestral build with strings and timpani`, seed `99`.

Metal was the one backend the Q4_K_M / Q8_0 tooling had never run on. It is correct here, and the
size wins carry over intact — but **the speed win does not**. On Metal, quantization is a footprint
optimization, not a latency optimization.

### the ggml pin is inert on Metal, as predicted

The submodule bump adds CUDA k-quant `get_rows`; the diff against `main`'s pin touches only
`src/ggml-cuda/*` plus two additive `gguf_get_tensor_*` accessors, and no Metal source at all.
Confirmed empirically: an F16 run on this branch is **byte-identical** to the same run built from
`main` (`md5 cfd0e775d12a6115661768e717db00a0`), at the same warm speed (14.51 s vs 14.53 s).

Note this 14.5 s does *not* match the 12.14 s recorded for the same shape in section 6
(2026-06-29). That gap is present on `main` too, so it predates this work — it arrived with the
ggml base move to v0.16.0 plus the Metal training merge. Worth a separate look; not attributable
to quantization.

### size

| artifact | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| medium DiT | 2773.4 MiB | 1483.4 | 962.4 |
| medium SAME-L | 1626.1 MiB | 864.7 | 569.9 |
| **set total** | **4399.5 MiB** | **2348.1** | **1532.3** |

Byte-for-byte identical to the Windows outputs, as expected — quantization is host-side and
backend-independent, so this also confirms the CI story is reproducible across hosts.

`sa3-quant-check` against the F16 reference: DiT `compared=204 below-threshold=0`, SAME-L
`compared=100 below-threshold=0`, at `threshold=0.990`, for both Q8_0 and Q4_K_M.

### speed — no gain on Metal

Best of three warm runs, first run per encoding discarded:

| backend | F16 | Q8_0 | Q5_K_M | Q4_K_M |
|---|---:|---:|---:|---:|
| CUDA (RTX 5070 laptop) | 5.54 | 4.15 | 3.71 | 3.65 |
| Vulkan (same GPU) | 6.61 | 5.23 | 4.62 | 4.50 |
| CPU (default threads) | 122.70 | 79.68 | 84.32 | 72.48 |
| **Metal (M4)** | **14.51** | **14.27** | — | **14.81** |

Where the other two GPU backends gain ~33%, Metal is flat: Q8_0 is 1.7% faster, Q4_K_M is 2.1%
*slower*. Run-to-run spread was under 0.2 s, so this is signal, not noise.

`SA3_PROFILE=1` explains why. "load/init" is total minus the profiled stages:

| stage | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| load / init | 1986.7 ms | 1359.5 | 1079.8 |
| t5 compute | 49.6 ms | 41.6 | 49.0 |
| dit_compute | 6598.4 ms | 6851.0 | 7282.7 |
| dec_compute | 5688.6 ms | 5834.9 | 6089.8 |
| **total** | **14349.8 ms** | **14113.5** | **14530.8** |

Quantization does save load time — 0.63 s at Q8_0, 0.91 s at Q4_K_M — but it *costs* compute:
`dit_compute` +3.8% / +10.4% and `dec_compute` +2.6% / +7.0%. The two effects cancel.

This is the unified-memory result the handoff anticipated. On a discrete GPU the win comes from
moving fewer bytes across PCIe at load and from lower VRAM bandwidth pressure during the matmuls.
On M4 the first term barely exists — `dit_upload` is 0.5 ms because there is no copy — and at
324 frames the DiT matmuls run the `mul_mm` path, which is ALU-bound rather than bandwidth-bound.
There, dequantizing k-quant blocks into registers is pure added work against Metal's F16
simdgroup-matrix path. Quantization only pays on Metal when memory is the binding constraint.

### fidelity

Waveform comparison of the three 30 s renders, identical prompt and seed:

| comparison | raw waveform cosine | rms envelope cosine |
|---|---:|---:|
| F16 vs Q8_0 | 0.989968 | 0.999567 |
| F16 vs Q4_K_M | 0.529443 | 0.956489 |
| Q8_0 vs Q4_K_M | 0.524552 | 0.955519 |

Q4_K_M's 0.53 raw cosine is the expected number, matching the ~0.55 seen on the other backends:
quantization perturbs the diffusion trajectory, so the sample drifts while the musical envelope
holds at 0.96. All three renders are music of the same character. The three outputs also have three
distinct hashes, which is a direct check on the `--encoding` fix — before it, `q4_k_m` silently
generated from the F16 weights and produced a bit-identical file.

### quantized T5 encoder — the k-quant `get_rows` path

`te.embed.weight` is the pipeline's only `ggml_get_rows` input, so this is the one place k-quants
reach that kernel. Encoder is 1074 MiB at F32. Cosine of encoder hidden states vs F32, `--seed 7`:

| encoder mix | embed type | size | CUDA | CPU | Vulkan | **Metal** |
|---|---|---:|---:|---:|---:|---:|
| q8_0 | Q8_0 | 285 MiB | 0.999301 | 0.999242 | 0.999641 | **0.999686** |
| q4_k_m | Q6_K | 205 MiB | 0.979378 | 0.978046 | 0.980380 | **0.978471** |
| q5_k | Q5_K | 184 MiB | 0.987014 | 0.986894 | 0.988893 | **0.988871** |

Metal lands inside the ordinary backend-divergence band on all three. The Q5_K row is the
meaningful one: it exercises the index mapping corrected in ggml `e9c70fd6`, and Metal's
independent `kernel_get_rows_q5_K` agrees with the fixed CUDA kernel to 0.002 — effectively on top
of Vulkan. A fourth implementation concurring is good evidence the fix is right; had it been wrong,
CUDA would read as the outlier. An end-to-end `sa3-generate` with a Q5_K-embed encoder ran clean.

Reaching Q5_K at all requires the `q5_k` coverage mix, since `q4_k_m` / `q5_k_m` promote
`embed.weight` to Q6_K. `q5_k` remains a test-only mix, not for publication.

Pass `--t5 <path>` explicitly: the encoder is not selectable through `--encoding`, and two
`t5gemma-b-b-ul2-encoder-*.gguf` files in `models/` would make the prefix glob ambiguous.

### recommendation for Metal

Ship Q4_K_M and Q8_0 for the disk and memory savings — a 4399 → 1532 MiB set is the difference
between fitting and not fitting on smaller Apple silicon, and that is the real win here. Do not
advertise a speedup on Apple silicon; on M4 it is a wash. Q8_0 is the better default where space
allows, being visually indistinguishable in fidelity terms (0.99 waveform cosine) at no speed cost.
