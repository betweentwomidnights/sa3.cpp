# Codex handoff: ggml v0.16.0 and Metal LoRA training

> Temporary internal handoff for continuing this work on the Apple M4 MacBook. Keep it on the
> development branch while Metal is being implemented; remove it or convert the durable parts into
> `docs/METAL.md` / `docs/GGML_FORK.md` before the final merge.

## Objective and decision boundary

Bring native SA3 LoRA/DoRA training up on ggml Metal, using the already validated ggml v0.16.0
patch stack as the candidate base.

Do **not** merge v0.16.0 merely because it builds. Windows A/B testing found essentially no Intel
or CUDA runtime change. Merge the dependency update only if it is a sound base for Metal training
or produces a demonstrated Metal benefit. Prefer one eventual parent-repo gitlink update containing
v0.16.0 plus Metal training over two consecutive submodule-only updates.

Do not tag either candidate branch until Metal validation and the user's terminal/ear checks pass.

## Exact starting state

Parent repository:

- repository: `https://github.com/betweentwomidnights/sa3.cpp.git`
- base main: `a5e17a0de6d7cb6b7581e29e7697131f53920fa2`
- handoff branch: `feature/ggml-v0.16.0-validation`
- handoff commit before this file: `7619f0bd9ad75030d05235036e280aead3049c19`
- the branch pins the ggml commit below

ggml submodule:

- repository: `https://github.com/betweentwomidnights/ggml.git`
- upstream base: tag `v0.16.0`, commit `524f974b`
- patch branch: `feature/sa3-training-vulkan-v0.16.0`
- validated candidate: `9915b8f1fcd24b066ec938c359be7297d88bbe55`
- no immutable v0.16.0 release tag exists yet, intentionally

The v0.15.3 immutable pins remain untouched:

- `sa3-training-v1-cpu-cuda` -> `cfec69c1`
- `sa3-training-v1-vulkan` -> `5a87d69c`

## Clone or update on the MacBook

Fresh clone:

```bash
git clone --branch feature/ggml-v0.16.0-validation --recurse-submodules \
  https://github.com/betweentwomidnights/sa3.cpp.git sa3.cpp
cd sa3.cpp
git submodule sync --recursive
git submodule update --init --recursive
```

Existing clone:

```bash
git status --short --branch
git -c fetch.recurseSubmodules=false fetch origin
git switch --track origin/feature/ggml-v0.16.0-validation
git submodule sync --recursive
git submodule update --init --recursive
```

Verify before editing:

```bash
git rev-parse HEAD
git -C ggml rev-parse HEAD
git -C ggml describe --tags --always
```

The parent hash will include this handoff commit. The required ggml hash is exactly
`9915b8f1fcd24b066ec938c359be7297d88bbe55`.

Create new Metal work branches rather than committing directly to the validation lines:

```bash
git switch -c feature/metal-training
git -C ggml switch -c feature/sa3-training-metal-v0.16.0 9915b8f1
```

The parent worktree will then correctly show `ggml` as modified when the submodule advances.

## What v0.16.0 changes for Metal

Between ggml v0.15.3 and v0.16.0, upstream Metal gained:

- `COL2IM_1D` for F32/F16/BF16 (`fda9d536`)
- F16-source `SET_ROWS` (`20f8dce7`)
- depthwise `CONV_2D_DW` (`a1329a7a`)

These are useful upstream coverage improvements, but none has been established as an SA3 hot-path
speedup. Metal AdamW already existed in v0.15.3. Most importantly, v0.16.0 still has **no Metal
`GGML_OP_OUT_PROD` implementation**. That is the expected hard blocker for LoRA backward.

The current downstream patch stack, rebased cleanly onto v0.16.0, contains:

1. CPU strided binary source support (`408b2172`)
2. `CONCAT` autodiff backward (`0d0170d9`)
3. CUDA strided unary source/destination support (`b6ff1a35`)
4. larger allocator free-block capacity (`31111d50`)
5. CPU/CUDA F16-weight `OUT_PROD` support (`fcd20c16`)
6. contiguous materialization for strided `CONT` gradients (`ae8ff532`)
7. patch-stack documentation (`c69e6187`)
8. Vulkan `OUT_PROD` (`f9eed3fb`)
9. Windows stale `MATH_LIBRARY` cache fix (`96694e19`)
10. tiled Vulkan `OUT_PROD` (`1a22b0b4`)
11. four-output Vulkan `OUT_PROD` (`cd7f29a5`)
12. v0.16.0 patch-base documentation (`9915b8f1`)

Do not squash these while developing. The functional separation is intentional.

## Known-good validation already completed on Windows

At parent `7619f0b` / ggml `9915b8f1`:

- CPU, CUDA, and Vulkan complete builds passed.
- all 16 registered sa3.cpp tests passed on each backend.
- Vulkan `OUT_PROD` passed 91/91 backend-op cases on Intel and NVIDIA.
- real medium CUDA and small-music Vulkan training steps passed.
- Intel v0.15.3 and v0.16.0 inference WAVs were byte-identical.
- matched eight-step Intel training produced byte-identical final adapters and trainer states.
- v0.16.0 was about 0.9% faster in the matched Intel training sample and about 2% slower in the
  matched inference median; both are within power/thermal variance.
- Intel driver `32.0.101.6629` does not expose `VK_KHR_cooperative_matrix`, so the new Xe1 path
  remained inactive.
- CUDA runtime remained in the existing performance class. The approximately 25-minute first
  Blackwell build was a cold rebuild, not a demonstrated v0.16.0 compile regression.

See `docs/BENCHMARKS.md`, `docs/TRAINING_BENCHMARKS.md`, and `docs/GGML_FORK.md`.

## Existing M4 Metal inference baseline

`docs/METAL.md` records a successful Apple M4, 32 GB unified-memory inference baseline on the old
ggml v0.15.3 base (`eced84c8`):

- medium F16, 128 frames, 8 steps: 6.21 seconds end to end
- Metal run-to-run output: byte-identical
- Metal versus CPU: raw cosine 0.996556, RMS-envelope cosine 0.999541, log-mag cosine 0.998519
- shared Metal backend lifetime fixed the old residency-set teardown failure

Before changing ggml Metal code, reproduce one matched v0.16.0 inference run. This distinguishes an
upstream Metal regression from the later training patch.

## Initial Mac build and inference gate

Prerequisites:

```bash
xcode-select --install   # only if the command-line tools are missing
brew install cmake       # only if cmake is missing
```

Download medium inference plus training-base models if the Mac does not already have them:

```bash
./models.sh --variant medium --encoding f16 --training-base
```

Configure ggml backend tests as part of the Metal tree, then build:

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DSA3_METAL=ON -DGGML_BUILD_TESTS=ON
cmake --build build-metal --config Release -j "$(sysctl -n hw.ncpu)"
ctest --test-dir build-metal --output-on-failure
```

Matched v0.16.0 inference smoke:

```bash
SA3_PROFILE=1 ./build-metal/bin/sa3-generate \
  --model medium \
  --prompt "upbeat funk groove with slap bass" \
  --duration 12 --steps 8 --duration-padding 0 --seed 42 \
  --out train-runs/v0160-metal-inference-baseline.wav
```

Record total, T5, DiT, and decoder timings; confirm the file sounds healthy and a second identical
run is byte-identical. Do not optimize inference during initial training bring-up.

## Expected first training result

Use a small-music smoke first if that model set and a small dataset are already present; otherwise a
short medium run is acceptable. Use pre-encoded latents when available so audio encoding does not
obscure the DiT failure.

```bash
SA3_TRAIN_PROFILE=1 ./build-metal/bin/sa3-train \
  --model small-music \
  --dataset /path/to/dataset \
  --latents-dir /path/to/preencoded/latents/sa3-medium \
  --frames 256 --steps 2 --checkpoint-every 0 --seed 42 \
  --out train-runs/metal-pre-outprod-smoke
```

The expected pre-patch failure is unsupported `GGML_OP_OUT_PROD` during LoRA backward. Capture the
exact assertion/error before editing. If a different operation fails first, investigate it rather
than assuming `OUT_PROD` is the only blocker.

## Metal `OUT_PROD` implementation target

Use CPU as the semantic reference and the downstream Vulkan implementation/tests as the closest
training-oriented precedent. The minimum SA3 contract is:

- F32 left operand + F32 right operand -> F32 output
- frozen F16 left operand + F32 right operand -> F32 output
- transposed-right/backward shapes generated by `MUL_MAT` autodiff
- partial/non-multiple tile shapes
- deterministic accumulation on repeated runs

Likely ggml files to touch include:

- `src/ggml-metal/ggml-metal-device.m` for support detection/pipeline creation
- `src/ggml-metal/ggml-metal-ops.cpp` and headers for dispatch/encoding
- `src/ggml-metal/ggml-metal.metal` for the kernel
- `tests/test-backend-ops.cpp` only when additional Metal-relevant cases are genuinely missing

Do not route the op back to CPU silently. A working Metal trainer must execute the backward op on
Metal and report unsupported shapes explicitly.

Start correct and simple. Benchmark before introducing threadgroup tiling or SIMD-group matrix
optimizations. The Vulkan path showed that a scalar implementation can establish correctness while
the tiled version is developed independently.

## Required post-implementation gates

1. Rebuild and run the 16 registered tests.
2. Run ggml backend-op coverage:

   ```bash
   ./build-metal/bin/test-backend-ops test -o OUT_PROD -b Metal
   ```

   The downstream Vulkan suite reports 91/91 supported cases; Metal should pass every case it
   claims to support, including the F16/F32 transposed LoRA case and `LORA_ZERO_B` invariant.
3. Run a two-step small-music training smoke and confirm finite loss/gradient norm.
4. Run the same seed/data/frames step on CPU and Metal. Compare printed loss and gradient norm.
5. For a stronger check, set `SA3_DUMP_STEP` and `SA3_DUMP_GRADS` on matched one-step CPU and Metal
   runs, then compare the forward velocity and every adapter gradient. Expected first-step
   `lora_A` gradients are zero because `lora_B` initializes to zero.
6. Resume a tiny checkpoint once to confirm the trainer state remains backend-portable.
7. Run medium-base at 512 frames after small-music passes; record peak memory and steady step time.
8. Re-run the inference baseline to prove the Metal training patch did not regress generation.
9. Give the user copy-pastable terminal commands for the final training and inference ear checks.

Do not call Metal training complete based only on backend-op tests.

## Branch, commit, and publication policy

- Keep parent work on `feature/metal-training`.
- Keep ggml work on `feature/sa3-training-metal-v0.16.0`.
- Commit the Metal support as focused ggml commits; keep documentation separate.
- Push the ggml commit before pushing a parent commit that pins it.
- Do not force-push either published patch branch.
- Do not move the existing immutable tags.
- After final validation, create a new immutable tag such as `sa3-training-v1-metal` and record it
  in `docs/GGML_FORK.md`.
- Update `docs/METAL.md` and the training benchmark document with measured M4 results.
- Remove this temporary handoff before the final merge, after its durable conclusions are in docs.

## User intent and collaboration context

The user prioritizes backend-complete native training and especially practical performance on
consumer hardware. Correctness comes first, but a merely functional Metal path should be profiled
and optimized before calling it v1.

Avoid a standalone ggml v0.16.0 merge if Metal does not justify it. Minimizing unnecessary gitlink
churn matters because downstream forks, including Pillo Paus's web-interface work, build on main.
The submodule remains exact-pinned for reproducibility; never change `.gitmodules` to follow a moving
branch and never instruct users to run `git submodule update --remote`.
