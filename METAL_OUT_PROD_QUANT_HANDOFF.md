# Handoff: Metal OUT_PROD with a quantized src0

Working doc for the M4. Delete once Metal lands and the durable parts move into `docs/GGML_FORK.md`.

ggml branch `feature/vulkan-out-prod-quant` (`c162c6ce`), off the merged v0.16.0 line.
sa3.cpp branch `feature/quant-distribution`.

## Why

The backward of `mul_mat` is `out_prod(W, transpose(grad))`. With `W` quantized that is the **only**
op between a quantized base model and LoRA training on it — the forward already works, because the
functional adapter path never does anything with `W` except pass it to `mul_mat`. That is the same
property that made quantized *inference* work in #24.

Measured payoff (CPU, medium, 64 frames, plain lora, matched seed):

| base | loss s1 / s2 | gnorm | s/step |
|---|---|---|---|
| Q4_K_M | 0.737046 / 1.116067 | 0.2057 / 0.2915 | **29.2** |
| F16 | 0.705220 / 1.093278 | 0.1952 / 0.3137 | **46.9** |

1.6x faster on a 2.9x smaller base. On top of that sits the QLoRA argument — training the adapter
against the weights it will actually deploy on — which is untested and is the reason kev wants this.

## Done

- **CUDA** (`c162c6ce`) — nearly free. `out-prod.cu` already dequantized a non-f32 `src0` into a
  transient pool buffer for frozen f16 weights, and `ggml_get_to_fp32_cuda` covers every k-quant, so
  the change was widening `is f16` to `is not f32`.
- **Vulkan** (`18db5476`) — new `out_prod_quant.comp`, inline dequant, generated for q4_K / q5_K /
  q6_K / q8_0. `test-backend-ops -o OUT_PROD`: **124/124 on both Vulkan devices, 3/3 backends**.
- **Host gap** (sa3.cpp, `train_lora.h`) — `train_tensor_to_f32` asked
  `ggml_backend_tensor_get` for `n` floats from a quantized tensor and tripped its bounds assert
  before any graph ran, so dora/bora aborted regardless of backend. Now delegates to `lora.h`'s
  `read_to_f32`.

Cross-check worth trusting: CUDA and Vulkan agree to the fifth decimal on a real training step
(`0.727405/0.894911` vs `0.727389/0.894905`).

## The trap that cost me a cycle — read before writing the kernel

`dequantize()` returns a **pair**, and which two elements depends on `QUANT_R`. Vulkan's
`get_rows_quant.comp` encodes the convention as `y_offset = QUANT_R == 1 ? 1 : QUANT_K/2`:

- `QUANT_R == 1` — pair is `(iqs, iqs+1)`, and **`iqs` must be EVEN**. The k-quants live here and
  halve `iqs` internally, so an odd `iqs` silently reads a different pair.
- `QUANT_R == 2` — pair is `(iqs, iqs + QUANT_K/2)`, two interleaved halves.

**Q8_0 hides this completely.** Its `.x` is element `iqs` for any `iqs`, so my first cut passed Q8_0
and returned garbage for Q4_K at `ERR ~1.0`. If Metal's equivalent passes Q8_0, that proves nothing
— check a k-quant explicitly.

Metal's helpers are shaped differently (block-based, 16 values at a time, `dequantize_q4_K(device
const block_q4_K *xb, short il, thread type4x4 & reg)`), so the mapping will not port literally —
but the same class of off-by-a-pair is the thing to watch.

## Two routes, and why the easy one does not work

1. **Inline dequant in the kernel** (what Vulkan does). `kernel_out_prod` is our SIMD-group tiled
   version templated on `T0`; the tile_a load at `ggml-metal.metal:1506` is where the dequant goes.
2. **Dequantize to scratch, reuse the f32 kernel** (what CUDA does). **This does not cover the case
   we care about**: Metal's `CPY` supports a quantized *source* only for the legacy quants — Q1_0,
   Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (`ggml-metal-device.m`, the `GGML_OP_CPY` switch). No Q4_K/Q5_K/Q6_K,
   so `q4_k_m` would be left out, which is the whole point.

Possible third route worth a look before committing to (1): Metal's `GET_ROWS` claims support for
everything except NVFP4 (`ggml-metal-device.m:1388`). A `get_rows` with an identity index vector is
a dequantize-to-f32 in disguise, and it reuses a tested kernel. Ugly, but it may be the fastest
correct path and a good way to prove the training result before writing a tiled quant kernel.

## Validation

`test-backend-ops -o OUT_PROD` is the gate and already generates quantized cases — `base_types`
includes Q4_K and Q8_0, so they were being reported "not supported" on Vulkan before this branch.
Note `supports_op` is not the only gate: Vulkan also has a blanket assert in `ggml_vk_op_f32`
allowing quantized `src0` only for `GET_ROWS` and `CPY`, which OUT_PROD had to be added to. Check
whether Metal has an equivalent.

Then end to end, which is what actually matters:

```
sa3-quantize --in models/stable-audio-3-medium-base-dit-1.5B-v1.0-F16.gguf \
             --out medium-base-q4.gguf --mix q4_k_m
sa3-train --model medium --dit medium-base-q4.gguf --dataset <ds> --latents-dir <lat> \
          --adapter-type dora-rows --frames 128 --steps 3 --seed 42 --out probe-q4
```

Expect finite losses in the same range as an F16 base, roughly 5% higher. A quantized forward should
cost a little accuracy; it should not change the shape of the curve.

## Speed: quantized is FASTER than f16 on all three measured backends

Steady state, mean of steps 6-15 (never trust a 2-step mean — it is mostly graph build, and it
misled me twice on this exact question):

| backend | q4_K_M | f16 | |
|---|---:|---:|---|
| CUDA | **0.93 s** | 1.08 s | medium, 128 frames, dora-rows |
| Vulkan | **1.72 s** | 1.90 s | same |
| CPU | **29.2 s** | 46.9 s | 64 frames, plain lora |

So there is no speed penalty to design around — the target is "at least match f16", and inline
dequant reaches it.

The Vulkan shader needed one optimization to get there (`4d20817d`): every type here is
`QUANT_R == 1`, so `dequantize()` returns two **adjacent** elements, which are two adjacent
`tile_a` slots. Loading a pair per call rather than fetching a pair per element and discarding half
halves the work — and for a k-quant that work is dominated by unpacking the block's scale bytes,
not by the element. **Metal's helpers return 16 values at a time (`thread type4x4 &`), so the same
idea should go further there**, though its tile is indexed `[ti][tk]` and its load loop varies `tk`
fastest, so an invocation's consecutive loads walk rows rather than elements — that loop needs
restructuring before a run of elements can share a call.

Also worth knowing: the bulk-dequant route (what CUDA does) is **not** a small change on Vulkan and
probably is not on Metal either. `mul_mat` has a bespoke path precisely because dequant-to-scratch
needs two-phase `prealloc_size_x` sizing plus descriptor-set requests; `out_prod` goes through the
generic `ggml_vk_op_f32` helper and would need the same plumbing. Inline dequant turned out to be
both smaller and fast enough, so prefer it.

## Known-not-done

Nothing blocking. The obvious further optimization is widening the pair load to a longer run so the
block scale unpack is amortised over more elements — worth more on Metal (16-wide helpers) than on
Vulkan (2-wide).
