# OpenCL backend

`./build.sh opencl` builds sa3.cpp and the pinned ggml OpenCL backend in `build-opencl/`.
The build uses embedded OpenCL C kernels and enables ggml's Adreno kernels. The Arduino UNO Q
currently needs the experimental Rusticl compatibility mode described below.

## Arduino UNO Q status

An end-to-end `small-music` Q4 generation has completed on an UNO Q's FD702 GPU and produced a
finite, non-silent 44.1 kHz stereo WAV. This is a correctness milestone, not yet a fast inference
path.

The tested board is Debian 13 on AArch64 with 3.6 GiB system RAM. Its physical Adreno 702 is
available at `/dev/dri/renderD128`; the login user must be in the `render` group. Both the system
Mesa 25.2.6 Turnip driver and a user-local Mesa 26.2.1 Turnip driver expose only Vulkan 1.0 for the
physical device. ggml requires Vulkan 1.2, so OpenCL/Rusticl is the usable GPU route. The Vulkan
1.4 `llvmpipe` device is CPU software rendering, not an alternative GPU path.

Two driver details matter:

- The image's Mesa/Rusticl 25.2.6 enumerates FD702, but does not expose the subgroup support ggml
  needs. The successful run used Mesa/Rusticl 26.2.1 from a user-local prefix.
- Mesa 26.2.1 exposes the Intel-compatible subgroup interface when `RUSTICL_FEATURES=intel` is
  set. `GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT=1` opts ggml into the corresponding at-most-32-thread
  generic dispatch path, uses whole-work-group local-memory reduction where the physical subgroup
  width cannot be trusted, disables unstable fusions, and serializes kernel dispatches to avoid
  delayed FD702 device faults. That serialization is intentionally conservative and is a major
  performance suspect; device execution and host wait costs still need separate profiling.

Do not replace the board's system Mesa packages just to reproduce this experiment. The tested
setup extracts the newer Debian packages under `/home/arduino/mesa26/root` and selects them only
for the sa3 process. The compatibility layer is experimental and may change as Mesa evolves.

## Build

Install the native build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build build-essential git \
    ocl-icd-opencl-dev opencl-headers clinfo
```

Clone and build:

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp
./build.sh opencl
```

For the tested user-local Mesa 26.2.1 prefix, use this environment when inspecting or running the
GPU:

```bash
export LD_LIBRARY_PATH=/home/arduino/mesa26/root/usr/lib/aarch64-linux-gnu
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export LIBCLC_PATH=/home/arduino/mesa26/root/usr/lib/clc
export RUSTICL_FEATURES=intel
export GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT=1

clinfo -l
```

The local Mesa package expects Clang resources below `/usr/lib/llvm-21`. On the test image that
path is a symlink to `/home/arduino/mesa26/root/usr/lib/llvm-21`. Treat this as a board-image
integration detail, not a portable installation recipe.

At startup, confirm sa3 selects `GPUOpenCL` / `FD702`. `SA3_DEVICE=cpu` forces the CPU backend for
matched A/B runs.

## OpenCL-compatible T5 encoding

The published F16 and all-Q8 T5 encoders reach tiled GEMM kernels that request 128 work-items;
Rusticl reports a smaller resource-limited maximum for those kernels on FD702. Use the custom
`q4_k_m_opencl` mix instead:

- Q4_K for the main weights;
- Q6_K for accuracy-sensitive projections;
- Q8_0 for the token-embedding table, served by the OpenCL SOA `GET_ROWS` kernel.

Create it from the F32 encoder already downloaded into the models directory:

```bash
./build-opencl/bin/sa3-quantize \
  --in  /home/arduino/sa3-models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
  --out /home/arduino/sa3-models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-Q4_K_M_OPENCL.gguf \
  --mix q4_k_m_opencl
```

On the test board conversion took 52.63 seconds, peaked at about 1.51 GiB RSS, and produced a
251 MiB file.

## Reproduce the smoke test

With the environment above and the model files in `/home/arduino/sa3-models`:

```bash
/usr/bin/time -v ./build-opencl/bin/sa3-generate \
  --model small-music \
  --models-dir /home/arduino/sa3-models \
  --encoding q4_k_m \
  --ae-encoding q4_k_m \
  --t5-encoding q4_k_m_opencl \
  --prompt "short electronic tone" \
  --duration 0.1 \
  --duration-padding 0 \
  --steps 1 \
  --threads 4 \
  --seed 1 \
  --out /home/arduino/sa3-opencl-smoke.wav
```

The corrected warm-cache run took 206.51 seconds wall-clock, peaked at 516,020 KiB RSS, used no
swap, and wrote a 17,684-byte PCM WAV with every sample nonzero. The identical CPU run took 14.20
seconds and peaked at 647,920 KiB RSS. `clFinish` after every GPU dispatch makes this a safe
compatibility baseline rather than a useful estimate of the FD702's potential throughput.

Phase timings for this deliberately tiny, one-step workload were:

| Phase | CPU | OpenCL compatibility | Ratio |
|---|---:|---:|---:|
| T5 compute | 3.91 s | 60.70 s | 15.5x slower |
| DiT compute | 6.42 s | 79.70 s | 12.4x slower |
| decode compute | 0.49 s | 6.89 s | 14.1x slower |

For context, removing per-dispatch synchronization reduced the reported T5 compute time to 4.65
seconds, but the driver then failed at readback with `CL_OUT_OF_RESOURCES`. This is not a valid
completed T5 benchmark. Attempts to finish every second or eighth kernel also failed. These
results motivate investigating the workaround first, but do not prove that synchronization alone
explains the slowdown or establish the GPU's raw arithmetic performance. The fault's root cause
(kernel correctness, resource pressure, buffer lifetime, or driver behavior) remains unresolved.

CPU/OpenCL diagnostic tensors remained finite after the work-group softmax fix. Cosine similarity
was 0.99859 for T5 cross-attention, 0.99802 for the post-DiT latent, and 0.84044 for decoded float
audio. Exact equality is not expected from a quantized GPU path.

## What fits

Correction (2026-09-02): medium Q4 is plausible in frugal mode, but has not been tested on this
board. The previous conclusion that it required new streaming/offload work incorrectly added
models that the existing pipeline frees between phases.

The driver reported about 1833 MiB (1.79 GiB) of global memory. ggml prints this as "MB", but divides
by 1024 squared. Its reported 809 MiB "free" is not a live measurement: the OpenCL backend
subtracts a fixed 1 GiB safety margin from the total.

The CLI and server default to `keep_models=false`. For plain text-to-audio, the pipeline frees T5
before loading DiT, then frees DiT and its graph allocator before loading SAME for decode. Startup
also loads and frees each model separately. Approximate medium weight sizes are:

| Phase | Weight-file size |
|---|---:|
| OpenCL-compatible T5 | 251 MiB |
| Medium Q4 DiT | 963 MiB |
| SAME-L Q4 | 570 MiB |

The relevant budget is the maximum of each phase's weights plus its working memory, not the sum
of all three weight files. File sizes are only a starting estimate: tensor alignment, upload
temporaries, activations, host memory, and driver allocations must be measured. Audio-to-audio,
adapters, and switching from a previously resident request need separate peak-memory checks.

The [iOS harness](https://github.com/betweentwomidnights/sa3.cpp-ios/blob/main/SA3OnDevice/SA3Engine.swift)
uses the same frugal setting and requests outer SAME-L decode chunks of 128 frames with 32-frame
overlap. The user's successful medium Q4 inference on iPhone 13 is a useful precedent, not proof
of identical memory behavior under Rusticl. Small-music remains the primary kernel-development
target; medium feasibility and GPU speed are separate questions.

## Validation and next work

The validated OpenCL cases include F32 elementwise operations, whole-work-group masked softmax,
Q4_K and Q6_K matrix multiplication at T5-sized shapes, and Q8_0 embedding `GET_ROWS`, all checked
against the CPU backend. Before trusting longer output:

1. Preserve a reproducible correctness baseline. Repeat operator tests and the deterministic
   0.1-second smoke test after every driver or kernel change. Non-silence alone is insufficient:
   track finite values and CPU-reference errors at T5, DiT, and decode. The observed decoded-audio
   cosine of 0.84044 deserves isolated decoder tests with identical input latents; it is not an
   established quality tolerance. Add a longer, multi-step listening test once stable.
2. Make profiling preserve execution semantics. Currently `GGML_OPENCL_PROFILING` takes a different
   enqueue branch that bypasses the Rusticl per-dispatch `clFinish`. Unify the safety behavior
   before comparing profiling builds. Record kernel names, tensor shapes, work sizes, event
   execution times, host enqueue/wait times, and phase-boundary live/peak buffer allocations.
   Bound and release profiling events so instrumentation does not accumulate resource pressure.
3. Reduce the no-serialization failure to a short repeatable kernel sequence, starting with T5
   because it fails before sampling. Capture the first failing boundary and device diagnostics;
   audit tensor bounds, subbuffer alignment/lifetimes, scratch reuse, and kernel resource limits.
   The backend already uses an in-order queue, so an ordering bug is a hypothesis, not a diagnosis.
   Keep the safe default while testing targeted synchronization or bounded in-flight work. Require
   repeated successful readback and CPU-reference agreement before calling an asynchronous path
   faster. If a driver issue is isolated, retain a minimal reproducer for upstream reporting.
4. Tune the measured hotspots. Start with Q4_K/Q6_K matrix kernels if profiling confirms their
   importance, then reductions, layout conversions, and safe fusions. Query per-kernel work-group
   limits and local/private memory; tune tile sizes to actual FD702 capabilities instead of
   assuming 128 work-items are valid or that fewer threads always means less memory/faster work.
   Revisit F16/Q8 GEMMs after the current Q4 path is reliable. Treat work-group geometry separately
   from model residency and full-graph activation memory.
5. Benchmark progressively: retain the 0.1-second test for regression, then use longer clips and
   multiple denoising steps, repeated warm runs, and a separately reported cold start. Match CPU
   and GPU models, duration/padding, steps, seed, and frugal settings; report per-phase and
   end-to-end time, peak memory, swap, temperature, and throttling. GPU wins are a measured goal,
   not guaranteed for every workload. If only some phases win, evaluate explicit hybrid placement
   including transfer costs rather than forcing all phases onto the GPU.
6. Validate medium separately after shared kernels stabilize. SAME-S currently runs as one outer
   decode graph but uses internal block-diagonal/shifted attention; SAME-L uses sliding-window
   attention and supports outer overlap tiling. Medium also changes matrix shapes and resource
   pressure, so small-model tuning does not establish medium coverage. Start with a short frugal
   medium run, then measure SAME-L chunk sizes/overlap and phase peaks as duration increases.

Set `SA3_PROFILE=1` for phase timings. For cross-backend numerical diagnosis, create an empty
directory and set `SA3_DUMP_GEN` to it; sa3 writes raw F32 conditioning, latent, and decoded-audio
tensors there.

Useful operator-test examples:

```bash
./build-opencl/bin/test-backend-ops test -b GPUOpenCL -o GET_ROWS -p q8_0
./build-opencl/bin/test-backend-ops test -b GPUOpenCL -o MUL_MAT \
  -p 'm=768,n=256,k=768'
./build-opencl/bin/test-backend-ops test -b GPUOpenCL -o SOFT_MAX \
  -p 'ne=\[1024,16,1,1\].*sinks=0,m_prec=f32'
```

Mesa 26.2.1 currently prints a warning about not detecting a patched `libclc` when using the
extracted prefix. The tested kernels still compile and pass the CPU-reference cases, but the
warning should remain visible until the board image has a fully integrated Mesa/libclc stack.
