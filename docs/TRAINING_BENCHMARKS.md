# Training Benchmarks

These measurements cover native `sa3-train` update speed. They are separate from the generation
benchmarks in [BENCHMARKS.md](BENCHMARKS.md). Unless noted otherwise, projected run times multiply
the mean steady-step time by the requested step count and exclude model loading, pre-encoding, and
checkpoint writes.

## Test system

- Windows, Intel Core Ultra 9 275HX (24 cores / 24 logical processors)
- Intel Graphics, driver `32.0.101.6629`, Vulkan UMA backend
- NVIDIA GeForce RTX 5070 Laptop GPU, Vulkan discrete-GPU backend with NV cooperative matrices
- 64 GB system memory
- F16 training-base DiTs, default DoRA-rows rank/alpha 16 recipe
- Batch size 1, pre-encoded ratatat latents, seed 42
- sa3.cpp `d7939a2`, ggml `5a87d69c`

## ggml v0.16.0 matched Intel validation

Before changing the dependency pin, the v0.15.3 and v0.16.0 builds were run back-to-back with the
same small-music F16 base, 256 frames, pre-encoded latents, dataset order, seed, timesteps, and
inpainting masks. Step 1 was excluded for graph setup; the table averages steps 2-8 from each
eight-step run.

| ggml base | T5 | prep | DiT | AdamW | total / step | projected 2,500 steps |
|---|---:|---:|---:|---:|---:|---:|
| v0.15.3 (`5a87d69c` SA3 pin) | 111.3 ms | 1.0 ms | 6,054.3 ms | 34.1 ms | **6,200.4 ms** | **4 h 18 min** |
| v0.16.0 (`9915b8f1` SA3 candidate) | 114.0 ms | 1.0 ms | 5,995.3 ms | 33.7 ms | **6,143.7 ms** | **4 h 16 min** |

The candidate is 0.9% faster in this matched pass, which is too small to claim beyond normal
iGPU power and thermal variance. More importantly, every printed loss and gradient norm matched,
and the final adapters and trainer-state files were byte-identical across versions. This run was
slower than the earlier 5,518.6 ms sample below on both builds, illustrating why the back-to-back
comparison is more useful for judging the ggml update than either absolute result.

The Intel device still reports `matrix cores: none`: driver `32.0.101.6629` does not expose
`VK_KHR_cooperative_matrix`, so ggml v0.16.0 cannot activate its new Xe1 cooperative-matrix path.
The dependency update is therefore correctness-neutral here, not the hoped-for iGPU training
breakthrough.

## Headline: medium-base CUDA versus PyTorch

This is the default production recipe: medium-base F16, 512 latent frames (~23.8 seconds), batch
size 1, DoRA-rows rank/alpha 16, checkpointed backward graphs, and the same pre-encoded ratatat
latents. Both implementations ran on the RTX 5070 Laptop GPU.

| implementation | timing source | steady step | throughput | 2,500 updates |
|---|---|---:|---:|---:|
| **sa3.cpp CUDA** | steps 2-50, `SA3_TRAIN_PROFILE=1` | **1,065.4 ms** | **0.939 steps/s** | **44 min 24 s projected** |
| PyTorch 2.7.1 + CUDA 12.8 | checkpoint timestamps, steps 500-2,500 | 1,775.0 ms | 0.563 steps/s | 73 min 58 s steady / **74 min 2 s recorded process** |

The native C++ trainer is **1.67x faster** on this workload, reducing projected training time by
40.0%. The completed PyTorch process time includes model startup and five step checkpoints; the C++
figure is a steady-step projection and excludes startup and checkpoint writes. A full native
2,500-step run should be recorded before replacing the projection with an observed wall time.

The C++ steady-step breakdown was:

| sampled steps | T5 | prep | DiT forward/backward | AdamW | total / step |
|---:|---:|---:|---:|---:|---:|
| 2-50 | 8.2 ms | 1.8 ms | 986.2 ms | 69.2 ms | **1,065.4 ms** |

The C++ measurement used:

```bat
set "SA3_GPU=" && set "SA3_TRAIN_PROFILE=1" && build-cuda\bin\Release\sa3-train.exe --dataset C:\dev\datasets\ratatat-train --latents-dir "C:\path\to\encoded\latents\sa3-medium" --steps 50 --out train-runs\cuda-medium-defaults-benchmark-50
```

The successful PyTorch reference job used mixed-F16 base weights, variable-length Flash Attention,
and 229 adapter targets. The sa3.cpp default intentionally excludes the tiny `seconds_total`
conditioner target and trains the 228 DiT Linear/Conv targets recommended for small datasets. This
one-target scope difference is disclosed for precision; it is not large enough to explain the
measured speedup by itself.

## Apple M4: native Metal training and matched MLX A/B

The ggml v0.16.0 Metal candidate completed the full native training path on an Apple M4 with 32 GB
unified memory. The reference gate used medium-base F16, 512 latent frames, batch size 1,
DoRA-rows rank/alpha 16, all 228 DiT targets, and the exact pre-encoded Ratatat latents from the
completed gary4local/PyTorch job. Step 1 was excluded for graph setup and Metal pipeline compilation.

| implementation | sampled steps | target count | steady step | throughput | projected 2,500 updates | maximum RSS |
|---|---:|---:|---:|---:|---:|---:|
| sa3.cpp Metal, scalar `OUT_PROD` | 2-5 | 228 | **22.364 s** | 0.0447 steps/s | **15 h 32 min** | **5.76 GiB** |
| **sa3.cpp Metal, 32x16 SIMD-group tile** | 2-5 | 228 | **8.649 s** | 0.1156 steps/s | **6 h 0 min** | **5.75 GiB** |
| gary4local MLX | 2-5 | 228 | **7.285 s** | 0.1373 steps/s | **5 h 4 min** | **16.27 GiB** |

The matched target inventory matters. Gary's existing production-style MLX run adapts only the 36
Linear/Conv layers in transformer blocks 20-23 and reached roughly 2.2-2.3 s/step at 512 frames;
that is not a fair comparison with the native trainer's full 228-target default. With MLX explicitly
set to all 228 targets, optimized C++ Metal is **1.19x slower**. The original scalar kernel was
3.07x slower; tiling and SIMD-group matrix accumulation improved the native path by **2.59x**.

Memory moves in the opposite direction. `/usr/bin/time -l` reported a 5.52 GiB peak memory
footprint for C++ and 23.79 GiB for MLX, in addition to the maximum-RSS values above. The native
checkpointed graph therefore uses about 35% of MLX's maximum RSS and 23% of its peak footprint.

The trainers are matched for model architecture, crop, adapter family/rank, learning rate, and
target count, but this is a throughput A/B rather than an exact loss comparison. The C++ run reads
the PyTorch job's F32 pre-encoded latents and uses the native reference optimizer/scheduler recipe;
the MLX trainer performs its own F16 latent encode and has different random-stream and AdamW details.
The native small-model parity gate separately compared identical inputs and all adapter gradients.

The C++ steady-step breakdown was:

| kernel | sampled steps | T5 | prep | DiT forward/backward | AdamW | total / step |
|---|---:|---:|---:|---:|---:|---:|
| scalar | 2-5 | 35.3 ms | 1.0 ms | 22,273.8 ms | 53.8 ms | **22,364.3 ms** |
| 32x16 SIMD-group tile | 2-5 | 29.8 ms | 1.0 ms | 8,574.5 ms | 43.0 ms | **8,648.8 ms** |

The optimized kernel cooperatively stages a 32x16 A tile and 16x16 B tile, then uses eight Metal
SIMD groups to compute the 32x16 result. It preserves the scalar path's arbitrary-stride, broadcast,
partial-tile, F32/F32, and F16/F32 contract. The final adapter is byte-identical to the scalar/tiled
reference for matched two-step runs; all 92 focused operation cases and all 37 project tests pass.

The native measurement used:

```sh
SA3_TRAIN_PROFILE=1 ./build-metal/bin/sa3-train \
  --model medium \
  --dataset train-runs/ratatat-macos-dataset \
  --latents-dir "$HOME/Downloads/ratatat-3-1784005242/encoded/latents/sa3-medium" \
  --prompt-config "$HOME/Downloads/ratatat-3-1784005242/ratatat-3-1784005242_dataset.json" \
  --frames 512 --steps 5 --checkpoint-every 0 --seed 42 \
  --out train-runs/metal-medium-ratatat-512-simdgroup-final-5
```

## Validated full run: small-music CUDA

The `train-runs/ratatat-train` run provides a complete native end-to-end measurement rather than a
short-run projection:

| model | frames | updates | latent source | checkpoints | wall time | effective time / update |
|---|---:|---:|---|---|---:|---:|
| `small-music` F16 base | 512 | 2,000 | native pre-encode | 500, 1,000, 1,500, 2,000 | **16 min 10 s observed** | **485 ms** |

The output directory and configuration snapshot were created at 2:04:39 PM; the step-2,000 and
final adapters were written at 2:20:49 PM. This wall time therefore includes model startup, native
audio pre-encoding, all training updates, four checkpoints, and the final adapter write.

The checkpoint intervals were:

| interval | elapsed | average step |
|---|---:|---:|
| start to 500 | 4 min 8 s | 496 ms |
| 500 to 1,000 | 3 min 54 s | 468 ms |
| 1,000 to 1,500 | 4 min 2 s | 484 ms |
| 1,500 to 2,000 | 4 min 6 s | 492 ms |

The command recorded by the trainer was:

```bat
sa3-train --dataset C:\dev\datasets\ratatat-train --steps 2000 --model small-music
```

## Intel iGPU: small-music Vulkan training

The backend was pinned with `SA3_GPU=intel`. Step 1 was excluded as graph setup/warm-up; the table
uses the subsequent stable steps printed by `SA3_TRAIN_PROFILE=1`.

| latent frames | audio context | sampled steps | T5 | prep | DiT | AdamW | total / step | projected 2,500 steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | ~23.8 s | 2-5 | 104.8 ms | 2.0 ms | 9,034.3 ms | 32.5 ms | **9,173.5 ms** | **6 h 22 min** |
| 256 | ~11.9 s | 2-6 | 100.2 ms | 1.0 ms | 5,383.8 ms | 33.2 ms | **5,518.6 ms** | **3 h 50 min** |

Reducing the crop from 512 to 256 frames cut steady total step time by 39.8% (1.66x throughput).
This is not strict reference-recipe parity: the shorter crop reduces the musical context seen by
each update. Stable Audio 3 is a variable-length model family, however, and maintainer guidance
indicates that adapters trained on roughly 12-second clips can still generate well at longer
durations during inference. A 256-frame crop is therefore a practical iGPU training option rather
than merely a smoke-test setting, although 512 frames remains the reference-style default.

The 256-frame run was launched from Command Prompt with:

```bat
set "SA3_GPU=intel" && set "SA3_TRAIN_PROFILE=1" && build-vulkan\bin\Release\sa3-train.exe --model small-music --dataset C:\dev\datasets\ratatat-train --latents-dir "C:\path\to\encoded\latents\sa3-medium" --frames 256 --steps 50 --out train-runs\vulkan-intel-small-256
```

The existing latent cache deliberately removes native audio pre-encoding from the comparison. The
profiled `total` is the training-step path; initial model loading and graph setup are also excluded.

## 256-frame backend comparison

The discrete-GPU and CPU comparisons used the same 256-frame model, latents, seed, dataset order,
timesteps, and inpainting masks. The matching stochastic trajectory and closely aligned
losses/gradient norms are also a useful cross-backend correctness check.

| backend | sampled steps | T5 | prep | DiT | AdamW | total / step | projected 2,500 steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 5070 Laptop GPU, Vulkan | 2-6 | 12.6 ms | 1.6 ms | 666.0 ms | 34.2 ms | **714.2 ms** | **29 min 46 s** |
| Intel Graphics, Vulkan | 2-6 | 100.2 ms | 1.0 ms | 5,383.8 ms | 33.2 ms | **5,518.6 ms** | **3 h 50 min** |
| Core Ultra 9 275HX, CPU, 24 threads | 2-6 | 115.4 ms | 1.0 ms | 13,990.8 ms | 59.2 ms | **14,166.8 ms** | **9 h 50 min** |

On this machine the discrete Vulkan GPU provides 7.73x the Intel iGPU throughput and 19.84x the CPU
throughput. The Intel iGPU still provides 2.57x the CPU throughput, making it the more practical
fallback for long runs when a discrete GPU is unavailable.

The NVIDIA Vulkan run was pinned independently of CUDA with:

```bat
set "SA3_GPU=nvidia" && set "SA3_TRAIN_PROFILE=1" && build-vulkan\bin\Release\sa3-train.exe --model small-music --dataset C:\dev\datasets\ratatat-train --latents-dir "C:\path\to\encoded\latents\sa3-medium" --frames 256 --steps 50 --out train-runs\vulkan-nvidia-small-256
```

The CPU run explicitly cleared `SA3_GPU` to avoid carrying device selection over from the Vulkan
terminal session:

```bat
set "SA3_GPU=" && set "SA3_TRAIN_PROFILE=1" && build\bin\Release\sa3-train.exe --model small-music --dataset C:\dev\datasets\ratatat-train --latents-dir "C:\path\to\encoded\latents\sa3-medium" --frames 256 --threads 24 --steps 50 --out train-runs\cpu-small-256-t24-50
```

Record steps 2 onward so graph setup does not skew the steady-state comparison.
