# benchmarks

### testing ground: 5070 laptop gpu, 8gb vram

created some different optimization levers to pull here. all numbers below are
stable-audio-3-medium, 8 steps, cuda backend. `total` is end-to-end wall time
*including* loading the three nets from disk (~2-3s of it); `decode` is just the
SAME-L decoder compute, `dit` is the 8-step sampler.

## ggml v0.16.0: Intel iGPU inference validation

The v0.16.0 candidate was compared directly with the prior v0.15.3 pin on the Intel Graphics
iGPU (device `0x7d67`, driver `32.0.101.6629`). Both runs used medium F16, the prompt `upbeat funk
groove with slap bass`, seed 42, 12 seconds, eight steps, no duration padding, and
`SA3_GPU=intel`. One warm-up run was excluded and the table reports the median of three complete
processes, including model loading.

| ggml base | matrix-core path | median end-to-end | result |
|---|---|---:|---|
| v0.15.3 (`5a87d69c` SA3 pin) | none | **23.47 s** | baseline |
| v0.16.0 (`9915b8f1` SA3 candidate) | none | **23.94 s** | +2.0%, within run/power variance |

All six measured WAV files were byte-identical. The v0.16.0 Intel Xe1 detection recognizes the
device's subgroup and integer-dot capabilities, but this driver does not expose
`VK_KHR_cooperative_matrix`; ggml therefore correctly stays on the scalar matrix path. The new
version is a correctness-preserving upgrade on this machine, but it does not unlock an iGPU
inference speedup without cooperative-matrix support from the driver.

## the levers

- **fp16 quantization** (`tools/quantize_gguf.py`) — halves the weight matrices so all
  three nets (dit + same-l + t5) fit in 8gb at once (~5.7gb vs ~10.3gb at f32), and
  lights up the tensor cores. weights f16 x activations f32, the graph handles it natively.
  decode parity cossim 1.0, full gen 0.998 vs f32.
- **early-free offloading** (on by default; `--keep-models` to disable) — frees t5 before
  sampling and the dit before the decode, so whichever net is working gets the vram to
  itself. cheap insurance; turns out to matter a lot at the extremes.
- **sliding-window attention** in the same-l decoder — the decoder attends over
  `frames * 17` tokens, but (matching the pytorch impl) each token only sees a ~35-wide
  window. computed as overlapping blocks, linear in length, instead of dense n²
  (quadratic). this is what makes long generations possible at all.
- **lora/dora apply on the gpu** — adapters are dora (weight-decomposed), so each is a
  per-weight recompute `W_eff = magnitude * (W + B@A) / ||·||`, not a static merge (keeps
  runtime strength + multi-adapter blending). that recompute used to run in host loops
  (~14s for the medium's 228 weights); now it's a ggml graph done in-place on the gpu.

## optimization matrix (~12s output, 128 frames)

| precision | early-free | total |
|---|---|---|
| f32 | off (`--keep-models`) | 41.4 s  ← vram thrash (10.3gb > 8gb) |
| f32 | on | 7.4 s |
| f16 | off (`--keep-models`) | 3.5 s |
| f16 | on | 3.7 s |

at f16 everything fits, so early-free barely matters at short lengths — keep-models is
even a hair faster (no free overhead). at f32 it's the difference between usable and
thrashing the pcie bus.

## generation length scaling (f16, early-free on)

| output | frames | decoder tokens | dit | decode | total |
|---|---|---|---|---|---|
| 12s  | 128  | 2,176  | 0.45s | 0.21s | 3.6s |
| 30s  | 323  | 5,491  | 0.87s | 0.65s | 4.4s |
| 60s  | 646  | 10,982 | 1.87s | 1.22s | 6.0s |
| 120s | 1292 | 21,964 | 3.96s | 2.24s | 9.1s |

the decoder stays linear in length — that's the sliding window doing its job. (the model
goes up to ~6 min; the dit's full attention is over `64 + frames` tokens so it's the part
that eventually grows fastest, but it's still fine into the multi-minute range.)

## where early-free re-earns its keep (120s output)

| | decode | total |
|---|---|---|
| f16, `--keep-models` | 41.5 s | 48.2 s |
| f16, early-free (default) | 2.24 s | 9.1 s |

at long lengths the decoder's feed-forward activations get big (the swiglu inner is
`[12288, 21964]` ≈ 1gb), and with all three models resident there isn't room for it — so
it thrashes again. freeing the dit first gives the decode buffer space and it drops 18x.

and the dense decoder isn't even on the table here: at 120s its attention score tensor
would be ~46 gb *per layer* (`n² × 24 heads × 4b`). the sliding window keeps it bounded —
vram held at ~6gb through the entire 120s decode.

## lora / dora apply (~12s output, f16)

| run | total | apply overhead |
|---|---|---|
| no lora | 3.50 s | — |
| + kev | 3.63 s | +0.13 s |
| + kev + keygen | 3.91 s | +0.41 s |

with the apply on the gpu and written in-place (no second copy of the weights), a lora
generation is basically a base generation plus ~0.1s per adapter. the in-place part matters:
a separate copy of the dora'd dit (~2.9gb) would push vram back over 8gb and thrash. on the
host loops this same apply was ~14s.

## flash attention (opt-in)

`ggml_flash_attn_ext` is a fused attention op every gpu backend implements, so the same flag
works across cuda/vulkan/metal (skip it on cpu). it's **off by default** — it accumulates in
f16, so the output drifts slightly from the bit-exact reference path (raw waveform cosine ~0.95
vs no-flash, but log-mag spectrogram ~0.996 — perceptually identical). turn it on for speed:

- `SA3_FLASH_ATTN=1` — flashes the DiT (and, unless overridden, the SAME-L decoder too).
- `SA3_SAME_FLASH_ATTN=full` (default when flash is on) / `=local` / `=0` — controls just the
  SAME-L decoder. **use `full`.** `local` (compact 3-block neighborhoods) is an Apple-Metal
  experiment that's *slower* even there, gives no win on Vulkan, and isn't supported by ggml's
  CUDA flash kernel (sa3-generate falls back to `full` with a warning on CUDA).

decode is where it pays off (medium f16, 12s, this 5070 laptop):

| backend | dit_compute | dec_compute |
|---|---|---|
| CUDA, no flash | ~486 ms | ~221 ms |
| CUDA, flash (full) | ~442 ms | **~166 ms (−25%)** |
| Vulkan, no flash | ~435 ms | ~245 ms |
| Vulkan, flash (full) | ~454 ms | **~166 ms (−32%)** |

the DiT gain is modest/within-noise; the decoder is the real win. on metal it helped through
120s (see docs/METAL.md). vulkan pays a one-time shader compile on the first flash run.

## CPU thread count (small-music)

testing ground: Intel Core Ultra 9 275HX, CPU backend, `small-music` f16,
`--duration 5 --steps 4 --duration-padding 0`. `--threads N` sets ggml's CPU
backend thread count; `SA3_THREADS=N` does the same for any surface.

| threads | total | t5_compute | dit_compute | dec_compute |
|---|---:|---:|---:|---:|
| 1  | 20.90 s | 0.87 s | 15.07 s | 3.54 s |
| 2  | 8.80 s  | 0.46 s | 5.64 s  | 1.26 s |
| 4  | 5.46 s  | 0.27 s | 3.02 s  | 0.74 s |
| 8  | 3.90 s  | 0.17 s | 1.85 s  | 0.43 s |
| 12 | 3.33 s  | 0.14 s | 1.41 s  | 0.33 s |
| 16 | 3.09 s  | 0.12 s | 1.20 s  | 0.30 s |
| 24 | 2.86 s  | 0.11 s | 1.07 s  | 0.25 s |

for a longer radio-style chunk on the same CPU, `--duration 60 --steps 4
--duration-padding 0` took 25.6s at the build default thread count and 9.4s
with `--threads 24`.

## CPU backend variants (small-music)

On Windows, `build.cmd cpu-variants` builds a CPU-only `GGML_BACKEND_DL=ON`
tree with `GGML_CPU_ALL_VARIANTS=ON`. GGML then loads the highest-scoring
supported `ggml-cpu-*.dll` at startup without requiring CUDA or Vulkan SDKs.

testing ground: Intel Core Ultra 9 275HX, `small-music` f16,
`--duration 30 --steps 1 --duration-padding 0 --threads 8`. Each variant was
forced by running from a temp directory containing only that CPU backend DLL.

| backend | loaded DLL | total | t5_compute | dit_compute | dec_compute |
|---|---|---:|---:|---:|---:|
| stock native CPU | linked/static | 6.79 s | 0.17 s | 1.64 s | 3.27 s |
| x64 | `ggml-cpu-x64.dll` | 43.43 s | 1.65 s | 13.99 s | 26.02 s |
| sse42 | `ggml-cpu-sse42.dll` | 43.21 s | 1.69 s | 13.67 s | 26.16 s |
| sandybridge | `ggml-cpu-sandybridge.dll` | 80.60 s | 0.19 s | 27.31 s | 51.27 s |
| haswell | `ggml-cpu-haswell.dll` | 6.57 s | 0.18 s | 1.61 s | 3.08 s |
| alderlake | `ggml-cpu-alderlake.dll` | 6.45 s | 0.15 s | 1.52 s | 3.11 s |

Takeaway: SAME-S CPU performance really wants an AVX2/FMA/F16C-class backend.
Generic/SSE fallback is too slow for realtime goals; the runtime variant bundle
is mainly useful for shipping one CPU package that can still select the best
backend available on the user's machine.
