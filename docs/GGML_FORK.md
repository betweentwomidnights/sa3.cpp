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

1. Build both CPU and CUDA configurations.
2. Run the registered CTest suite on both configurations.
3. Run one medium-base and one small-base training step.
4. Apply each resulting adapter through `sa3-generate`.
5. Push the ggml branch and verify its exact commit is visible on the fork.

Then update the pinned commit in `sa3.cpp`. A fresh-clone check is required before merging:

```sh
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git sa3-clean
git -C sa3-clean/ggml rev-parse HEAD
```

The parent repository intentionally pins a commit instead of following a moving branch. This keeps
builds reproducible while still making the downstream patch lineage explicit.
