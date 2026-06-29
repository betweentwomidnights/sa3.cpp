# benchmarks

### testing ground: 5070 laptop gpu, 8gb vram

created some different optimization levers to pull here. all numbers below are
stable-audio-3-medium, 8 steps, cuda backend. `total` is end-to-end wall time
*including* loading the three nets from disk (~2-3s of it); `decode` is just the
SAME-L decoder compute, `dit` is the 8-step sampler.

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

