# OpenCL backend

## Local experimental project

Development is isolated on `feature/unoq-opencl-experimental` in both sa3.cpp and its ggml
submodule. This is a local-only experiment: no push, published fork, or upstream integration is
implied. The known-working serialized baseline is checkpointed at sa3.cpp `0b355e3` and ggml
`e7ee7182`. Commit ggml changes first, then update the parent gitlink so each checkpoint identifies
both source trees. These unpublished pins are not fetchable from GitHub; preserve both repositories
(or local Git bundles), not just the parent repository. Existing remotes are unchanged.

The UNO Q checkout is the deployment/build copy. Keep the original `build-opencl` as the baseline
and use `build-opencl-profile` for instrumented work. Match its existing build settings, including
`GGML_OPENCL_USE_ADRENO_KERNELS=OFF` for the tested generic Rusticl dispatch path; the build helper's
Adreno-enabled default is not the exact configuration used by this board's baseline.

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
  set. `GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT=1` opts ggml into a generic dispatch path with small
  K-quant work-groups, uses whole-work-group local-memory reduction where the physical subgroup
  width cannot be trusted, disables unstable fusions, and serializes kernel dispatches to avoid
  delayed FD702 device faults. That serialization is intentionally conservative. The first
  event-timed T5 profile below finds slow kernel execution, not synchronization overhead, to be
  the dominant cost in that workload.

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
2. Make profiling preserve execution semantics. The initial `GGML_OPENCL_PROFILING` branch bypassed
   the Rusticl per-dispatch `clFinish`; the experimental instrumentation now shares the enqueue
   and synchronization path between both build modes. Record kernel names, tensor shapes, work sizes, event
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

### Standalone T5 fault reduction

Initial profiling result (2026-09-02): a one-layer, 256-token T5 graph passed three CPU and three
serialized GPU repetitions. CPU compute was 0.327-0.333 s; GPU compute was 5.166-5.168 s.
Across all three GPU repetitions, event execution totaled 15.351 s versus 15.460 s spent inside
`clFinish` (inclusive of execution). Q4_K matrix-vector kernels accounted for 10.465 s, Q6_K for
2.821 s, and F32 attention matrix multiplication for 1.853 s. Thus Q4/Q6 kernels account for about
87% of device time; removing host waits alone cannot explain or fix this slowdown. CPU/GPU
hidden-state cosine was 0.99998697, RMSE 0.0301843, with all 196,608 values finite.

With per-dispatch finishes disabled, the profiled one-layer graph also passed three repetitions
at 5.150-5.166 s and produced identical output to the serialized GPU run. A full 12-layer graph
then completed at 60.627 s with finite output. These runs did not reproduce the original fault;
event profiling and flushed dispatch tracing can affect driver submission behavior. The original
unprofiled full-generation fault remains unresolved. Existing kernel logs show hang recovery and,
on one earlier attempt, GPU read-translation faults; neither establishes the root cause.

This evidence changes the priority: retain safe dispatch while testing an opt-in 32x32 tiled
Q4_K/Q6_K GEMM (`GGML_OPENCL_RUSTICL_TILED_K=1`). Its 32-work-item geometry replaces the unavailable
128-work-item tile only in Rusticl compatibility mode, with a per-kernel limit check and fallback.
Partial tiles and broadcast batches need CPU-reference tests before relying on model output.

The first tiled candidate passed all six targeted Q4_K/Q6_K operator cases: T5-sized
`m=768,n=256,k=768`, partial `m=35,n=33,k=256`, and the latter with broadcast batches.
The one-layer GPU graph ran in 1.688-1.702 s during the test-binary build, then 1.747-1.750 s in a
separate three-repeat run after compilation finished. The matched CPU repeat was 0.327-0.336 s.
This is roughly a 3x improvement over the safe matrix-vector path, not a GPU win over CPU.

The tiled, serialized, profiled full small-music smoke test completed with these phase timings:

| Measurement | Previous safe fallback | Tiled candidate |
|---|---:|---:|
| T5 compute | 60.70 s | 20.70 s |
| DiT compute | 79.70 s | 36.78 s |
| SAME-S decode compute | 6.89 s | 3.38 s |
| Process wall time | 206.51 s | 120.57 s |

Peak RSS was 516,068 KiB with no swap. The candidate wrote a valid stereo 44.1 kHz WAV with all
8,820 PCM samples nonzero. Against the previous safe GPU float dumps, cosine was 0.999999999998
for conditioning, 0.999999999997 for latent, and 0.99999619 for decoded audio (decode RMSE 0.01035).
This checks that tiling preserved the prior result; it does not resolve that baseline's larger
CPU/GPU audio discrepancy or establish useful musical quality from a one-step, 0.1-second test.

Full-run device execution totaled 58.46 s: Q6_K GEMM 22.95 s, Q4_K GEMM 19.37 s, and F32 matrix
multiplication 14.00 s (about 96% combined). Those are the next compute targets. Roughly 60 s
remains outside the measured compute phases; separately time initialization and weight loading
before attributing that cost. These are diagnostic comparisons, with tracing/profiling enabled
for the candidate and some compilation overlapping its startup, not controlled serving benchmarks.
The original CPU full smoke result remains 14.20 s.

Artifacts are retained on the board under `/home/arduino/sa3-experiments/` and copied locally to
the ignored `bench/unoq/` directory. Important runs: `textenc-safe-L1-S256-bxUTsX` (fallback),
`textenc-safe-L1-S256-Nudpel` (final tiled repeat), `textenc-cpu-L1-S256-chDHcr` (matched CPU),
`generate-tiled-FAvhUt` (full generation), and `ops-tiled-VHIC5o` (6/6 operator checks).
Both profiling and non-profiling C++ branches compile (the latter checked with `-fsyntax-only`);
runtime validation in this iteration used the profiling build. Tiling remains opt-in and the
original `build-opencl` binary is preserved.

Build the instrumented binary separately from the preserved baseline:

```bash
cmake -S . -B build-opencl-profile -DCMAKE_BUILD_TYPE=Release \
  -DSA3_OPENCL=ON -DGGML_OPENCL_TARGET_VERSION=300 \
  -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
  -DGGML_OPENCL_PROFILING=ON -DGGML_BUILD_TESTS=ON
cmake --build build-opencl-profile --target sa3-textenc sa3-generate test-backend-ops -j2
bash tools/unoq-textenc.sh cpu 1 3 256
bash tools/unoq-textenc.sh safe 1 3 256
```

Runner arguments are mode, encoder layers (`-1` means all), repetitions, and sequence length.
Each run creates a directory under `/home/arduino/sa3-experiments`, with source/build identity,
process resource statistics, and `t5g_hidden.f32`. A truncated encoder still uses the final norm
and loads the whole weight file; it is a diagnostic graph, not valid full-model conditioning or a
test of reduced weight residency. Both CPU and GPU must use the same truncation for comparisons.

`GGML_OPENCL_DISPATCH_TRACE=/absolute/path.tsv` flushes each enqueue/finish boundary to disk to
survive a process abort. Profiling writes `cl_profiling.csv` and valid `cl_trace.json` on clean
backend teardown. Host finish time **includes** device execution; do not add them together.
`python3 tools/unoq-profile-summary.py GPU_RUN_DIR CPU_RUN_DIR` summarizes kernels and compares
the float outputs. Trace I/O and profiling overhead mean these are diagnostic timings.

Only after validating the safe case, `bash tools/unoq-textenc.sh async 1 1 256` opts into a fault
experiment. Internally `GGML_OPENCL_RUSTICL_FINISH_INTERVAL=0` disables per-dispatch finishing;
positive N finishes every Nth wrapped dispatch. Unset means the safe default of 1. These controls
only affect Rusticl compatibility mode. Other existing barriers remain, and asynchronous profiling
waits to collect/release a batch at 2048 events, so it is not a barrier-free execution test.
An enqueue or graph submission time followed by failed readback is never a successful benchmark.

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
