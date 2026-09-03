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

End-to-end `small-music` Q4 and F16 smoke generations have completed on an UNO Q's FD702 GPU and produced a
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

On the preserved baseline, the published F16 and all-Q8 T5 encoders reach tiled GEMM kernels that request 128 work-items;
Rusticl reports a smaller resource-limited maximum for those kernels on FD702. Use the custom
`q4_k_m_opencl` mix for that baseline. The opt-in float-tile candidate below now also runs the
published F16 encoder. The custom mix is:

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
6. Validate medium separately after shared kernels stabilize. SAME-S defaults to one outer
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

## Float-tile precision experiment

`GGML_OPENCL_RUSTICL_TILED_FLOAT=1`, together with Rusticl compatibility mode, opts into a
32-work-item, 32x32 output / K=32 tile for F16 or F32 weights and F32 activations/accumulation.
Shared rows have a 33-float pitch (8.25 KiB total), an experiment in reducing bank conflicts.
It supports byte-strided inputs and host-dispatched broadcast batches; it does not imply native
half arithmetic. The candidate is selected only when its reported per-kernel work-group limit
allows 32 work-items. The older F16 tile now also checks its 128-work-item requirement before
selection. Other devices retain their existing float dispatch paths.

Run correctness before performance, without overlapping builds or other benchmark processes:

```bash
bash tools/unoq-precision.sh test GPUOpenCL
bash tools/unoq-precision.sh perf CPU
bash tools/unoq-precision.sh perf GPUOpenCL
GGML_OPENCL_RUSTICL_TILED_K=1 GGML_OPENCL_RUSTICL_TILED_FLOAT=1 \
  bash tools/unoq-textenc.sh safe 1 3 256
SA3_UNOQ_T5_WEIGHTS=/home/arduino/sa3-models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F16.gguf \
  GGML_OPENCL_RUSTICL_TILED_FLOAT=1 bash tools/unoq-textenc.sh safe 1 3 256
```

The operator sweep uses identical M/N/K shapes, not identical quantized values across precisions.
Each correctness case compares against CPU at the same weight precision. The performance suite
uses its own warmup/repetition policy and the board's four CPU threads. It remains an instrumented
microbenchmark, not end-to-end latency. Existing K-quant kernels fuse tile dequantization into
floating-point multiplication; measured kernel time does not isolate dequantization cost.
Use `tools/unoq-profile-summary.py RUN --profile-only` to inspect operator-only profiles.

`tools/unoq-generate.sh DEVICE DIT_PRECISION AE_PRECISION T5_PRECISION` runs the established
one-step, 0.1-second smoke with stage dumps and resource statistics, using both candidate tile
switches. Defaults are GPU / Q4_K_M / Q4_K_M / Q4_K_M_OPENCL. It intentionally does not keep all
models resident. Compare CPU/GPU with matching encodings; compare fixed-latent codec output
separately before attributing generated-waveform differences to codec precision.

Reference lessons from Stability AI's `optimized/tflite` implementation: precision is a per-stage
decision, weight-only quantization is not integer arithmetic, and fixed-shape shared-weight decoder
plans apply to SAME-S as well as SAME-L. SAME-S outer tiling is gated by the separate experimental
switch documented below; it is not enabled by the kernel switches. The UNO Q CPU does
not advertise `asimddp` or `i8mm`, so modern-CPU INT8 performance claims do not transfer directly.
Medium Q4 remains a separate memory-constrained target; small-model float results do not settle it.

Hexagon remains a capability investigation, not an enabled backend: the board exposes root-only
`/dev/fastrpc-adsp` with a running audio DSP, but no CDSP device was found. The pinned ggml backend
expects CDSP and targets v73/v75/v79/v81. Audio DSP presence alone does not establish compatibility.

Published F16 small-music DiT and SAME-S weights were fetched from `thepatch/stable-audio-3-small-music-GGUF`
at revision `e06bd8e9a23de11207334cb57c0599c9c3e5bc42`, without changing the existing Q4 files.
Their SHA-256 values matched the published LFS hashes before the temporary downloads were renamed:

- DiT, 919,265,984 bytes: `d00baebc5ed850701ff19904d750dd98c03fdadc7c24a8ab0abcad8efba36cb4`.
- SAME-S, 216,721,728 bytes: `ae34819e33cb188e54c4102125e0926d6739978772923b05e00ce9e74684e736`.

### Operator results (2026-09-02 local / 2026-09-03 UTC)

The final padded candidate passed **22/22** CPU-reference cases (`precision-test-GPUOpenCL-KLfg9Y`).
These include K=37 tails, M=35/N=33 edges, nonzero view offsets, padded rows, permuted batch/row
strides and broadcasting in both batch dimensions. Two earlier fixture attempts aborted before
completing: transposed A is rejected by ggml's graph API, and transposed B is unsupported by this
CPU reference implementation. Neither abort was a GPU fault; the final fixtures keep K contiguous.

Matched M=768, N=256, K=768, milliseconds per matrix multiplication (lower is better):

| Weight type | CPU, four threads | GPU, initial 32-float pitch | GPU, final 33-float pitch |
|---|---:|---:|---:|
| Q4_K | 20.52 | 76.45 | 79.27 |
| Q6_K | 27.02 | 100.82 | 101.79 |
| F16 | 33.11 | 88.65 | 89.74 |
| F32 | 42.46 | 86.60 | 88.23 |

The unchanged quantized paths provide context for run-to-run drift; the small float differences
do not establish a padding benefit. Float storage is not a speed win on this shape. The initial
GPU profile recorded 115.95 s of device execution and 117.07 s of inclusive host finishes across
warmup plus 332 timed iterations per type, confirming compute rather than host synchronization
dominates. Profiles explicitly contain `mul_mat_f16_f32_small` and `mul_mat_f32_f32_small`.
Artifacts: `precision-perf-CPU-E5zppQ`, `precision-perf-GPUOpenCL-IZahB8` (initial) and
`precision-perf-GPUOpenCL-QTlyos` (final). No builds/downloads overlapped these timing sweeps.

One-layer T5, sequence 256, three repetitions:

| Encoding/backend | Compute time | CPU-reference output cosine |
|---|---:|---:|
| F16 CPU | 0.454–0.477 s | reference |
| F16 GPU, float tile | 1.268–1.278 s | 0.9999999983 |
| Q4 custom GPU, both tiles | 1.206–1.209 s | 0.9999869695 |
| Q4 custom GPU, previous K tile only | 1.747–1.750 s | 0.99998697 |

All outputs were finite. The FP16 encoder is now runnable, with RMSE 0.000347 against the matching
CPU result; it uses a 537.14 MiB weight buffer and 7 MiB graph buffer. CPU and GPU peak RSS were
1,034,344 and 619,344 KiB respectively, with zero swap. The Q4 run uses a 251.35 MiB weight buffer.
The Q4 graph improved about 31% in latency, mostly from F32 attention: its three-repeat device time
fell from about 1.84 s to 0.173 s, while CPU-reference agreement remained unchanged. This does not
make the overall GPU faster than CPU, nor establish that Q4 T5 has equal quality to F16 T5.
Artifacts: `textenc-cpu-L1-S256-6jqR2H`, `textenc-safe-L1-S256-fn7RR4` (F16), and
`textenc-safe-L1-S256-cOQXrE` (Q4 with both tiles).

For fixed-latent SAME-S comparisons, build `sa3-codec` and run
`bash tools/unoq-codec.sh DEVICE ENCODING /absolute/latent.f32 FRAMES` with the same latent file
for every case. This runs three synchronized repetitions, checks finite output, writes the
intermediate taps and raw `audio.f32`, and records model/input hashes and process memory statistics.
Compare with `tools/unoq-profile-summary.py RUN REFERENCE --tensor audio.f32`; this bypasses sampler
divergence and does not apply the pipeline's loudness processing. It is not a musical quality test.

Cold-start follow-up: `Pipeline::load` currently loads/uploads each complete model sequentially
to extract configuration, frees it, and then `generate` loads it again at the relevant phase.
This preserves low simultaneous weight residency but duplicates I/O/upload work. A metadata-only
configuration reader (plus the small `ae.running_std` tensor) is a concrete next optimization.
Its contribution to wall time has not yet been isolated; do not attribute all unprofiled time to it.

### Complete small-music smoke and fixed-latent results

One step, 0.1 seconds requested, seed 1, four CPU threads, no simultaneous model residency:

| Configuration | T5 compute | DiT compute | Decode compute | Process wall | Peak RSS (KiB) |
|---|---:|---:|---:|---:|---:|
| Q4 CPU | 3.908 s | 6.411 s | 0.487 s | 14.81 s | 644756 |
| Q4 GPU, both tiles | 14.387 s | 32.014 s | 3.429 s | 110.38 s | 514960 |
| F16 CPU | 5.446 s | 8.504 s | 0.645 s | 29.87 s | 1140084 |
| F16 GPU, float tile | 14.928 s | 31.971 s | 3.480 s | 99.91 s | 714500 |

All four wrote valid 44.1 kHz stereo, 4410-frame WAVs; all float stage dumps were finite, and
no process swapped. RSS is not total GPU/system memory usage. This establishes F16 feasibility
for the short smoke, not arbitrary-duration memory fit. The apparent F16 process-time advantage
on GPU is outside measured compute (50.38 s F16 versus 49.83 s Q4); loading and cache state need
separate control before attributing that advantage to precision. CPU remains faster.

CPU/GPU cosine by stage: Q4 conditioning 0.998590, latent 0.998025, decoded audio 0.840407;
F16 conditioning 0.999999719, latent 0.999999390, decoded audio 0.997569. Neither is an exact
waveform match or a musical-quality evaluation. Q4's prior larger discrepancy remains present.

Fixed input `generate-tiled-FAvhUt/latent.f32` (512 values, RMS 104.86, an extreme one-step latent),
SAME-S T=2, three repetitions with the original input restored before each repeat:

| SAME-S precision | CPU compute | GPU compute | CPU/GPU audio cosine |
|---|---:|---:|---:|
| Q4_K_M | 0.491–0.530 s | 3.432–3.481 s | 0.928944 |
| F16 | 0.629–0.664 s | 3.436–3.520 s | 0.998243 |

Q4/F16 CPU audio cosine on that same input was 0.835315. These are numerical diagnostics on an
unrepresentative short sampler output, not evidence about perceptual quality on real music.
The decoder itself contributes to the divergence, so do not attribute all waveform differences
to the sampler. Use F16 as the clearer correctness reference while retaining Q4 as the compact
baseline, and validate realistic latents/listening before selecting production precision.

An initial repetition-harness bug failed to restore the latent after graph execution reused its
storage. Those earlier codec runs (`aSCHL2`, `Tat1vY`, `hrws7V`, `RqzANK`) are **invalid fixed-input
comparisons**. The corrected Q4 CPU three-repeat output is bit-identical to an independent one-pass
run (`codec-repeat-check`). Valid codec artifacts: `codec-cpu-Q4_K_M-2GGAl5`,
`codec-gpu-Q4_K_M-N1eB9U`, `codec-cpu-F16-d3Pwrf`, `codec-gpu-F16-KLIYPy`.
Full generation artifacts: `generate-cpu-q4_k_m-q4_k_m-q4_k_m_opencl-Nymcaz`,
`generate-gpu-q4_k_m-q4_k_m-q4_k_m_opencl-IoDqTY`, `generate-cpu-f16-f16-f16-ZsNWh5`,
and `generate-gpu-f16-f16-f16-rvdFSK`.

### Experimental fixed-shape SAME-S decoding

`SA3_EXPERIMENTAL_SAME_S_CHUNKED=1` allows the existing outer decode planner to use SAME-S.
Default behavior remains monolithic. This is **decode only**; SAME-S outer encoding is unchanged.
The candidate requires the known two-latent attention-block geometry, an even chunk size, and
overlap at least 16 and divisible by 4 (also strictly smaller than the chunk). It reuses one
allocated graph and the already loaded AE weights across chunks, restoring each input before
execution. Retained output regions are stitched without cross-fading.

Absolute RoPE positions must be preserved for both attention halves. Restarting each chunk at
zero produced cosine 0.73193 against monolithic CPU output, with errors beginning exactly at the
second chunk's retained region. With global position offsets, the 4-second CPU comparison is
**bit-identical** for conditioning, latent, all 360,448 decoded float samples, and therefore has
zero boundary error on this test. This is why relative-position equivalence alone was insufficient
for enabling this path. SAME-L and encoder position behavior is unchanged.

```bash
SA3_UNOQ_DURATION=4 bash tools/unoq-generate.sh cpu
SA3_UNOQ_DURATION=4 SA3_UNOQ_DECODE_CHUNK=32 \
  SA3_EXPERIMENTAL_SAME_S_CHUNKED=1 bash tools/unoq-generate.sh cpu
```

The 44-frame reference used a 36,116,480-byte decoder graph buffer; the reused 32-frame graph uses
26,665,376 bytes (about 26% less). Decode compute increased from 7.39 to 10.92 seconds because two
overlapping chunks process 64 frames in total. Whole-process peak RSS remained about 645,000 KiB,
dominated by other phases: do not describe the graph-buffer reduction as an overall RSS reduction.
CPU artifacts: `generate-cpu-q4_k_m-q4_k_m-q4_k_m_opencl-COJMWA` (monolithic),
`generate-cpu-q4_k_m-q4_k_m-q4_k_m_opencl-dX82w9` (absolute-position chunks).
`...-J22GwG` is the failed zero-origin candidate and must not be used as a valid tiling result.
Medium/SAME-L remains a separate validation target; these SAME-S results do not certify medium.

### Large-dispatch watchdog investigation

The first 4-second GPU monolithic reference (`...-YW8oO5`) failed during SAME-S with OpenCL event
status -14 and a kernel-driver GPU hang/recovery report. A decoder-only traced replay
(`codec-gpu-Q4_K_M-kxUKkO`) also failed. Its last submission was Q4_K GEMM with global geometry
4608x25x1, local 32x1x1; the inclusive finish wait was 1.484 seconds. This suggests a long-submission
watchdog problem, but a finish wait that includes failure/recovery is not a valid kernel benchmark.
Neither failed run produced valid decoded output. No reboot, watchdog disabling, or system-driver
change was used.

`GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS=256` opts the candidate F16/F32 and Q4_K/Q6_K tiles into column
striping. It additionally limits estimated work to approximately 500 MFLOP per submission, rounding
to 32-column boundaries and accounting for batches in the K-quant kernels. This is a heuristic,
not a guaranteed execution-time ceiling; a minimum tile can still be too expensive on another
shape/device. Full batch strides and offsets are retained, including the final partial stripe.
The value must be a positive multiple of 32; unset disables splitting. The precision, generation,
and codec experiment runners now enable 256 by default. Earlier timing tables predate splitting.

With the limit forced to **32**, all 22 operator cases passed again
(`precision-test-GPUOpenCL-eu6NLV`), exercising split output columns, tails, offset views and batches.
Unchanged work-group geometry and K accumulation order make this a submission-size change rather
than another arithmetic precision change. Keep serialized dispatch enabled for these tests.

The identical decoder-only 44-frame input then completed with splitting
(`codec-gpu-Q4_K_M-CSCR1Y`): 41.00 s compute, all output taps finite, a valid 13,514-event profile,
and a maximum recorded device execution of 152.58 ms per dispatch. No new driver hang/recovery
was logged. This supports the long-dispatch explanation and provides a mitigation for this
workload; it does not resolve every previous asynchronous driver fault or certify all shapes.

With splitting enabled, the one-layer Q4 T5 reference (`textenc-safe-L1-S256-2znhUH`) took
1.198 s per repeat, retaining the earlier float-tile speedup. Both full 4-second, one-step GPU
runs then completed without a new driver recovery:

| SAME-S decode | Graph buffer | Decode compute | Process wall | Peak RSS |
|---|---:|---:|---:|---:|
| Monolithic, 44 frames | 36,116,608 bytes | 37.60 s | 154.17 s | 516708 KiB |
| Chunk size 32, overlap 16 | 26,665,472 bytes | 56.03 s | 172.56 s | 516764 KiB |

Artifacts: `generate-gpu-q4_k_m-q4_k_m-q4_k_m_opencl-4UBcH6` and
`generate-gpu-q4_k_m-q4_k_m-q4_k_m_opencl-SOHItp`. Conditioning, latent, and all 360,448 decoded
floats were **bit-identical between these GPU runs**, matching the CPU chunk-equivalence result.
Both WAVs contain 176,400 stereo frames at 44.1 kHz. All compared float values were finite.
The profiles contained 18,345 and 24,841 kernel events, respectively, with maximum device durations
151.65 and 152.59 ms. This validates the tested four-second shape, not all durations or music quality.
Chunking saves graph memory but costs overlap compute; whole-process RSS did not decrease.

Invalid chunk size 31 and overlap 12 were also rejected with the expected diagnostic and exit 1
before generation (`...-vp22ZZ`, `...-IIsnQF`). The latest backend passes a non-profiling C++ syntax
check; experiment scripts pass `bash -n` and the profile helper passes Python compilation.

Next work should focus on the remaining GEMM device time and avoiding startup's redundant model
loads. Neither a metadata-only loader nor a Hexagon/Vulkan implementation is included here.
Keep F16 as an accuracy reference, Q4 as the compact baseline, and medium as a separate validation
target. The experimental GPU path still does not beat the CPU.
