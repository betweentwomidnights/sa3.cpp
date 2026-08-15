# Maintaining the SA3 ggml fork

`sa3.cpp` pins an exact commit from
[`betweentwomidnights/ggml`](https://github.com/betweentwomidnights/ggml). The fork carries a small
training-oriented patch stack that is not yet available in upstream ggml.

The initial patch branch is `feature/sa3-training-v0.15.3`, based on upstream ggml `v0.15.3`
(`eced84c`). It contains six focused commits:

1. CPU strided-source binary operations.
2. Autodiff backward support for `GGML_OP_CONCAT`.
3. CUDA strided-source/destination unary operations.
4. Additional allocator free-block capacity for large functional LoRA graphs.
5. CPU and CUDA F16-weight support in `OUT_PROD` backward.
6. Contiguous materialization for strided `GGML_OP_CONT` gradients.

Vulkan training is layered on that reviewed pin in
`feature/sa3-training-vulkan-v0.15.3`. Its four additional commits add Vulkan `OUT_PROD`
backward support, tile and thread-tile the shader, cover the new F32/F16 and partial-tile cases in
ggml's backend-op tests, and prevent a stale Windows `MATH_LIBRARY-NOTFOUND` cache entry from
breaking reconfiguration.

The v0.16.0 validation line is `feature/sa3-training-vulkan-v0.16.0`, based on upstream tag
`v0.16.0` (`524f974b`). The complete CPU/CUDA/Vulkan patch stack cherry-picks without conflicts.
At the pre-Metal candidate `9915b8f1`, all three sa3.cpp builds and their 16 registered tests pass,
Vulkan `OUT_PROD` passes 91/91 backend-op cases on both Intel and NVIDIA, and matched Intel inference
and training outputs are byte-identical to the v0.15.3 pin.

Metal training was developed on `feature/sa3-training-metal-v0.16.0` and merged into the v0.16.0
patch line through ggml PR #1. It adds native Metal `REPEAT_BACK`, F32/F16-weight `OUT_PROD`,
`SILU_BACK`, `RMS_NORM_BACK`, and `SOFT_MAX_BACK`; generalizes F32 binary operations for autodiff's
strided views; and fixes the wide-row non-inplace `ACC` copy dispatch. On Apple M4, all 37 registered
tests pass, Metal `OUT_PROD` passes 92/92 cases, small CPU/Metal aggregate gradient cosine is
0.9999853, CPU trainer state resumes on Metal, medium-base trains at 512 frames, and the frozen
inference WAV is byte-identical before and after the patch. A 32x16 threadgroup/SIMD-group
`OUT_PROD` tile reduces the matched medium steady step from 22.364 s to 8.649 s while keeping the
adapter byte-identical and peak RSS at 5.75 GiB. The immutable tag `sa3-training-v1-metal` points to
the exact audited commit `922875a6`; PR #1's merge commit `f75b63f6` has the same source tree.

Three pins have followed on the same v0.16.0 line. None is a new backend milestone, so none carries
a new tag. The first two are merged and reachable from `feature/sa3-training-vulkan-v0.16.0`, which
is what the pin policy below requires; the third is still on its own branch and has to merge there
before it can be pinned on `main`.

**Q4_K_M / q5_K `get_rows`** (ggml PR #2, merge `f561ab0d`, pinned at `e9c70fd6`). Fixes k-quant
element dequantization in `get_rows` and adds gguf tensor ndims accessors. Landed with the
quantization work.

**Wide-row non-inplace `SET`** (ggml PR #3, merge `5e4d3a8c`, contains `74d2abbd`).
`ggml_metal_op_set` dispatched only `ne01` threadgroups for its `src0 -> dst` copy while
`kernel_cpy_t_t` derives its row-chunk index as `tgpig[0]/ne01`, pinning that index at 0 and
truncating every row at `nth = min(1024, ne00)` elements. The untouched tail kept whatever the
destination buffer held. sa3's DiT prepends 64 memory tokens to a `[1536, T]` activation with two
non-inplace `ggml_set` calls, so 512 of every row's 1536 channels were stale — zeros on step 1,
since fresh buffers are zeroed, and previous graph data afterwards. Scoped training on Apple was
silently corrupted: `--lora-scope core` completed normally with finite losses and a broken adapter.

Note this is the **same defect the Metal milestone above already fixed in `ACC`** — the two
functions are near-identical copies, `ggml_metal_op_acc` carries the multiplier as `ne01*nwg`, and
`ggml_metal_op_set` was missed. Treat them as a pair when touching either. `test-backend-ops`
passed 12/12 throughout because it exercises `SET` only at `ne00 = 6`, well under the 1024
threshold; `tests/set_wide_row_test.cpp` in this repo covers the real shape on whichever GPU
backend is present.

**Quantized `src0` for `OUT_PROD`** (branch `feature/vulkan-out-prod-quant`, pinned at `2f6e2a7c`;
the branch name predates the CUDA and Metal commits on it). This is a new capability, not a fix: it
is what lets a LoRA train on a quantized base. The backward of `mul_mat` is
`out_prod(W, transpose(grad))`, and with `W` quantized that is the **only** op in the way — the
forward already worked, because a functional adapter never does anything with `W` except pass it to
`mul_mat`, the same property that made quantized inference work.

- **CUDA** (`c162c6ce`) — nearly free. `out-prod.cu` already dequantized a non-f32 `src0` into a
  transient pool buffer for frozen f16 weights and `ggml_get_to_fp32_cuda` covers every k-quant, so
  widening `is f16` to `is not f32` was the change.
- **Vulkan** (`18db5476`, `4d20817d`) — new `out_prod_quant.comp` with inline dequant.
- **Metal** (`2f6e2a7c`) — new `kernel_out_prod_q`, inline dequant, same SIMD-group tiling.

All three are generated for q4_K, q5_K, q6_K and q8_0 only, the types a `q4_k_m` / `q5_k_m` /
`q8_0` mix actually produces. The rest of the type table is deliberately absent until someone
validates it.

**Quantized is faster than F16 on every backend measured**, so there is no accuracy/speed tradeoff
to reason about — steady state, mean of steps 6-15, medium with 128 frames and `dora-rows` except
where noted:

| backend | q4_K_M | F16 |
|---|---:|---:|
| CUDA | 0.93 s | 1.08 s |
| Metal (M4) | 2.77 s | 3.28 s |
| Vulkan | 1.72 s | 1.90 s |
| CPU | 29.2 s | 46.9 s (64 frames, plain lora) |

Never trust a two-step mean here: it is mostly graph build and it inverted this exact comparison
twice during development.

Two traps are worth carrying forward, because they are the same class of bug wearing different
clothes. On Vulkan, `dequantize()` returns a **pair** whose members depend on `QUANT_R`, and for
`QUANT_R == 1` — every k-quant — `iqs` must be **even**, since the k-quants halve it internally. On
Metal the helpers instead take a *group* index and emit 16 values, so a work item must stage a whole
16-wide run of a row rather than one element; doing it per element repeats the block's scale unpack
sixteen times, and on a k-quant that unpack is the cost, not the element arithmetic. In both cases
**Q8_0 masks the mistake completely** — its layout is insensitive to the index convention — so a
Q8_0 pass is not evidence. Check a k-quant explicitly.

The gate is `test-backend-ops -o OUT_PROD`: 128/128 on Metal and 124/124 on both Vulkan devices,
220/220 on CUDA. `base_types` reaches Q4_K and Q8_0 but not Q5_K or Q6_K, and a `q4_k_m` mix emits
Q6_K, so explicit cases for those two were added alongside the Metal kernel. `supports_op` is not
always the only gate: Vulkan has a blanket assert in `ggml_vk_op_f32` permitting a quantized `src0`
only for `GET_ROWS` and `CPY`, which `OUT_PROD` had to join. Metal has no equivalent, but its
`supports_op` and the op's own assert now share `ggml_metal_op_out_prod_supports_src0` so the
supported list cannot drift from the kernel instantiations.

Two routes were rejected on Metal and the reasoning generalizes. Dequantizing to scratch and reusing
the f32 kernel, which is what CUDA does, does not cover the case that matters: Metal's `CPY` accepts
a quantized *source* only for the legacy quants (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0), so `q4_k_m`
would be left out. And the bulk-dequant plumbing is not a small change on either Vulkan or Metal —
`mul_mat` has a bespoke path precisely because dequant-to-scratch needs two-phase `prealloc` sizing,
which `out_prod` does not have. Inline dequant is smaller and already faster than F16.

## Updating the fork

Keep the official repository as `upstream` and the SA3 fork as `origin` inside the submodule:

```sh
git -C ggml remote add upstream https://github.com/ggml-org/ggml.git
git -C ggml fetch upstream --tags
```

Do not force-push a patch branch after an `sa3.cpp` commit references it. For a new upstream ggml
version, create a new branch such as `feature/sa3-training-v0.16.0`, rebase or cherry-pick the six
patches onto the reviewed upstream commit, and resolve each patch independently. Drop a patch only
after confirming that upstream contains an equivalent implementation.

Before updating the parent repository's gitlink:

1. Build every affected CPU, CUDA, Vulkan, or Metal configuration.
2. Run the registered CTest suite on every affected configuration.
3. Run ggml backend-op coverage for any newly supported operation and device class.
4. Run one medium-base and one small-base training step.
5. Apply each resulting adapter through `sa3-generate`.
6. Push the ggml branch and verify its exact commit is visible on the fork.

Then update the pinned commit in `sa3.cpp`. A fresh-clone check is required before merging:

```sh
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git sa3-clean
git -C sa3-clean/ggml rev-parse HEAD
```

The parent repository intentionally pins a commit instead of following a moving branch. This keeps
builds reproducible while still making the downstream patch lineage explicit.

## Pin and release policy

The gitlink in each `sa3.cpp` commit is the source of truth. Fork branches are development lines,
not dependency selectors: do not add `branch = ...` to `.gitmodules`, and do not ask users to run
`git submodule update --remote`.

Create a new immutable annotated tag for each reviewed backend milestone. Never move an existing
tag when Vulkan, Metal, or another backend adds patches; push a new tag and update the `sa3.cpp`
gitlink to its exact commit. Keep every published pin reachable from the public fork so old
`sa3.cpp` revisions remain buildable.

| sa3.cpp milestone | immutable tag | upstream base | fork branch | pinned commit | trained backends |
| --- | --- | --- | --- | --- | --- |
| trainer v1 | `sa3-training-v1-cpu-cuda` | ggml `v0.15.3` (`eced84c`) | `feature/sa3-training-v0.15.3` | `cfec69c` | CPU, CUDA |
| Vulkan v1 | `sa3-training-v1-vulkan` | ggml `v0.15.3` (`eced84c`) | `feature/sa3-training-vulkan-v0.15.3` | `5a87d69c` | CPU, CUDA, Vulkan |
| Metal v1 | `sa3-training-v1-metal` | ggml `v0.16.0` (`524f974b`) | `feature/sa3-training-vulkan-v0.16.0` (PR #1) | `922875a6` | CPU, CUDA, Vulkan, Metal |
| Q4_K_M `get_rows` fix | — (bug fix) | ggml `v0.16.0` (`524f974b`) | `feature/sa3-training-vulkan-v0.16.0` (PR #2) | `e9c70fd6` | CPU, CUDA, Vulkan, Metal |
| Wide-row `SET` fix | — (bug fix) | ggml `v0.16.0` (`524f974b`) | `feature/sa3-training-vulkan-v0.16.0` (PR #3) | `5e4d3a8c` | CPU, CUDA, Vulkan, Metal |

Add a row when the parent pin changes. Later backend milestones receive new immutable tags and rows
rather than changing any existing trainer-v1 tag. A pin that only fixes a bug on an existing line
gets a row but no tag: the tags mark audited backend milestones, while the requirement that every
published pin stay reachable is met by the fork branch itself.

To read the pin from a checkout rather than trusting this table:

```sh
git -C ggml rev-parse HEAD            # the exact pinned commit
git -C ggml describe --tags           # e.g. sa3-training-v1-metal-6-g5e4d3a8c
```

`describe` is the quick sanity check — the suffix counts commits past the last audited milestone,
so `-6-` means six commits of fixes have landed since `sa3-training-v1-metal`.

## Updating existing clones and downstream forks

For a direct clone following `main`, disable recursive submodule fetching during the one-time
superproject update, then synchronize the cached URL and check out the exact tested pin:

```sh
git -c fetch.recurseSubmodules=false pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

The fetch override matters during the URL migration: recursive fetching otherwise consults the
old cached `ggml-org/ggml` URL before the new `.gitmodules` file is present and can fail with
`not our ref`.

For a downstream fork, use the same ordering around its normal merge or rebase workflow:

```sh
git -c fetch.recurseSubmodules=false fetch upstream
git merge upstream/main                 # or: git rebase upstream/main
git submodule sync --recursive
git submodule update --init --recursive
```

Fresh `--recurse-submodules` clones already use the correct URL. Later pin-only updates within this
same fork normally require only `submodule update`, but the two-command form is deliberately safe
to repeat and also handles a future URL change.

Normal downstream changes to the server, CLI, or web UI do not conflict with the ggml gitlink. A
submodule conflict occurs only when both sides intentionally change the ggml pin; resolve that by
choosing or integrating the desired ggml commit, validating it, and recording the resulting exact
gitlink. Never use `--force` in general update instructions because downstream contributors may
have local ggml work.
