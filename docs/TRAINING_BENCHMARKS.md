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
