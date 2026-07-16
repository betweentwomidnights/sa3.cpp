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
At candidate `9915b8f1`, all three sa3.cpp builds and their 16 registered tests pass, Vulkan
`OUT_PROD` passes 91/91 backend-op cases on both Intel and NVIDIA, and matched Intel inference and
training outputs are byte-identical to the v0.15.3 pin. Do not publish a release tag or update the
table below until final downstream terminal validation is complete.

Metal training is layered on that candidate in `feature/sa3-training-metal-v0.16.0`. The untagged
development line adds native Metal `REPEAT_BACK`, F32/F16-weight `OUT_PROD`, `SILU_BACK`,
`RMS_NORM_BACK`, and `SOFT_MAX_BACK`; generalizes F32 binary operations for autodiff's strided
views; and fixes the wide-row non-inplace `ACC` copy dispatch. On Apple M4, all 37 registered tests
pass, Metal `OUT_PROD` passes 92/92 cases, small CPU/Metal aggregate gradient cosine is 0.9999853,
CPU trainer state resumes on Metal, medium-base trains at 512 frames, and the frozen inference WAV
is byte-identical before and after the patch. The branch remains untagged until the user's final
terminal and ear checks pass.

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

Add a row when the parent pin changes. Metal and later backend milestones receive new immutable
tags and rows rather than changing either existing trainer-v1 tag.

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
