# Linux runtime packaging

`ci/package-linux.py` builds native x64 or ARM64 Linux packages for gary4local.
This is an experimental packaging path. Windows/macOS release publishing is
unchanged; Linux CI artifacts are dry runs and are not attached to releases.

| Archive | Contents |
| --- | --- |
| `sa3-vX.Y.Z-linux-{x64,arm64}-core.zip` | SA3/SAT servers, generation/training/conversion/analyzer tools, libsa3, ggml core and runtime-selected CPU variants, licenses, prompt examples and build metadata |
| `sa3-vX.Y.Z-linux-{x64,arm64}-cuda.zip` | `libggml-cuda.so` only |
| `sa3-vX.Y.Z-linux-{x64,arm64}-vulkan.zip` | `libggml-vulkan.so` only |
| `SHA256SUMS-linux-{x64,arm64}` | Checksums for this invocation's verified archives |

Extract core and **one** GPU backend into the same directory. The archive root
is flat except for prompt examples. Executables keep their Unix permissions;
shared library aliases are regular copies rather than ZIP symlinks, matching
gary4local's extraction behavior. Model weights are never included.

The core can be checked on its own. GPU driver/loader libraries and CUDA user
libraries remain **system dependencies**, with their exact external SONAMEs and
glibc/libstdc++ symbol requirements recorded in `BUILD-INFO.json`. These packages
do not include or claim compatibility with the Windows CUDA redistributable.
Defining and pinning shared Linux CUDA redistributables in gary4local is separate
work. A successful Spark check does not establish compatibility with older Linux
distributions or Linux x64 machines.

## Build dependencies

- Native Linux x64 or ARM64; C/C++ compiler, CMake 3.20+, CTest, binutils (`readelf`),
  `ldd`, Git with the pinned ggml submodule, and Python 3.10+ for packaging/tests.
  Installed runtimes do not require Python.
- The pinned ggml's full ARM CPU variant set needs SME compiler support (tested
  with GCC 14). The Spark's GCC 13 rejects those flags. Set `CC`/`CXX` to a
  capable local toolchain; the packager checks this before compiling GPU code.
  This is a build requirement for runtime support, not a CPU-generation target.
- CUDA: a compatible CUDA toolkit and driver. The script does not download one.
- Vulkan: Vulkan development headers/loader, `glslc`, and SPIRV-Headers CMake
  configuration. `VULKAN_SDK`/`CMAKE_PREFIX_PATH` can select a local toolchain.

The default builds CUDA and Vulkan. Select only the dependencies you have:

```sh
# Spark smoke build; targets the attached GPU, not a portable CUDA release.
python3 ci/package-linux.py --version v0.2.3 --backends cuda \
  --cuda-arch native --require-gpu --jobs 4

# Vulkan with the same portable CPU variants, in separate owned build/output dirs.
python3 ci/package-linux.py --version v0.2.3 --backends vulkan --require-gpu \
  --build-dir build-dist-linux-vulkan --out-dir dist-linux-vulkan --jobs 2
```

Use the version from `project(VERSION)`. `--cuda-arch` marks metadata as
`gpu-smoke`; omit it to use ggml's portable architecture list. `--require-clean`
rejects local source changes. The script checks that ggml is clean at the pinned
commit. `--skip-tests` skips CTest, but never skips extracted package checks.
No packages, releases, models, or dependencies are downloaded by the script.

CUDA and Vulkan are the supported generation paths for gary4local. CPU libraries
remain necessary for runtime support and model-free tests; CPU generation and
CPU-only releases are not a validation target. A core-only invocation is useful
for package diagnostics, not a supported gary4local deployment.

Build/output directories must be separate, dedicated directories inside the
checkout. A marker identifies directories the script owns; a nonempty directory
without that marker is refused. Repeated runs reuse the compiler output and
replace only the packager's staging folder. Archives reach `dist-linux*` after
verification and the checksum file is written last. Use separate output directories
for different backend sets; the checksum file covers the current invocation,
not historical ZIPs left beside it.

## What is checked

1. CMake builds shared libraries with `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`,
   `GGML_CPU_ALL_VARIANTS=ON`, `SA3_BUILD_SAT=ON`, and `$ORIGIN` runtime paths.
2. CTest runs with a bounded timeout per test.
3. ELF metadata records actual dependencies and rejects runtime paths into the
   checkout/toolchain. The CPU core contains no GPU backend.
4. Core and each core-plus-backend archive combination are extracted elsewhere
   and checked with `LD_LIBRARY_PATH` removed. Internal ggml/libsa3 dependencies
   must resolve to the extracted package, not an installed/build-tree copy.
5. Both servers answer `--version` and `--props` from an empty working directory;
   SA3 server/trainer/analyzer controls are verified by the shared package checker.
   `--require-gpu` additionally requires a GPU device in the chosen backend in
   **both** servers' properties. Software Vulkan alone is insufficient.

These are packaging/control checks, not audio-quality, generation, training,
or API parity checks. Run short generation and server tests with existing
weights separately before proposing a release or consumer manifest change.

The **Linux packaging dry run** workflow exercises native x64 and ARM64 Vulkan
packages on packaging pull requests or manual dispatch on Ubuntu 24.04, with
7-day artifact retention. The ARM64 runner installs GCC 14 for SME support.
Hosted CI does not validate GPU execution; CUDA packaging and real CUDA/Vulkan
generation checks currently happen on Spark. There is no self-hosted runner
registration or automatic Spark job.

The existing `release.yml` also calls this workflow for a **no-tag** dispatch
with `platforms=all` or `platforms=linux`. An empty tag retains artifacts without
publishing. `platforms=linux` skips Windows/macOS. Release events and tagged
rebuilds continue to publish only Windows/macOS packages.

```sh
gh workflow run release.yml -R betweentwomidnights/sa3.cpp \
  --ref feature/linux-runtime-packaging -f platforms=all
```

## Spark workspace

Use `/home/kev/gary4local-packaging/sa3.cpp`, an independent checkout, with
`WORKSPACE.md` and timestamped `storage.tsv` in its parent. All build artifacts,
logs, temporary files, downloaded build-tool packages, and caches belong inside
`/home/kev/gary4local-packaging`. Export `TMPDIR`, `XDG_CACHE_HOME`, and
`CUDA_CACHE_PATH` to its `tmp`/`cache` folders for each run.

No production containers or service ports are changed. Reuse existing models
through explicit paths instead of downloading or copying them. Keep generation
outputs here too; do not write adapters or derived caches into an existing
model store. Before cleanup, measure `du -h -d 2`, preserve useful artifacts/logs,
and stop packaging-owned processes. This checkout is not a worktree of the
existing production/development checkout, so deleting the dedicated workspace
does not leave worktree registrations behind.

## Spark validation, November 12, 2026

Built from sa3.cpp v0.2.3 (`3f300d8` plus the packaging changes), with ggml
`e3db5210`, on Ubuntu 24.04.5 ARM64 / NVIDIA GB10 / driver 580.178.04.
GCC 14.2 and Vulkan build tools were extracted under `toolchains/`, without
installing system packages. CUDA uses the existing toolkit 13.0.88. The CUDA
smoke flags resolve to `compute_121a` / `sm_121a`; this is not a general portable
CUDA release. The Vulkan build used loader/headers 1.4.328.1, SPIRV-Headers
1.4.328.1 and shaderc 2023.8 (KHR cooperative matrices available; optional
NVIDIA cooperative-matrix2 features unavailable).

| Check | CUDA | Vulkan |
| --- | --- | --- |
| Native build and CTest | 52/52 pass | 52/52 pass |
| Extracted core and core-plus-backend controls | Pass | Pass |
| Both servers' required GPU `--props` | Pass | Pass |
| SHA-256 archive checks | Pass | Pass |
| Short generation from extracted ZIPs | CUDA0 / NVIDIA GB10 | Vulkan0 / NVIDIA GB10 |
| Output | 88,200 frames, stereo PCM16, 44.1 kHz, nonzero | Same format/length, nonzero |
| Total CLI wall time, including model load | 9.5 s | 7.8 s |

The generation used existing medium F16 DiT, F16 T5 and F32 SAME weights from
`/home/kev/sa3.cpp/models`, without adapters: 2 seconds, 2 steps, seed 42, CFG 2,
negative prompt `low quality`, zero duration padding, 4 CPU support threads.
The selected device was explicitly restricted to NVIDIA. The existing GGUF
files' sizes and modification times were unchanged. This checks execution,
including negative-prompt CFG, not perceptual quality, backend parity or speed.
No CPU generation, training, LoRA/continuation generation, SAT-family generation,
or gary4local adapter integration was validated in this run.

The actual ELF requirements are glibc **2.38** and GLIBCXX **3.4.32** for these
builds. CUDA additionally needs the driver and system `libcudart.so.13` /
`libcublas.so.13` (including their dependencies). Vulkan needs `libvulkan.so.1`
and a functioning driver. These results do not cover Ubuntu 22.04 or a machine
with only CUDA 12 user libraries. The package checker removes `LD_LIBRARY_PATH`
and verifies that internal libraries come from the extracted archive.

At the end of the initial GPU checks, the workspace used about **1.3 GiB**:
roughly 754 MiB source/builds/packages, 264 MiB toolchains, 105 MiB local CUDA
cache, and 116 MiB extracted validation runtimes/audio/logs. The CUDA ZIP set was
31 MiB and Vulkan 18 MiB. See the parent `storage.tsv` for subsequent measurements.
The older 6.2 GiB control-center exploration folder is separate. The workspace's
runner and smoke scripts record the exact commands and keep temporary files here.

## Versioned Linux previews

The release workflow can promote an existing successful Linux CI run without
rebuilding its ZIPs. Set `platforms=linux`, `linux_preview_run_id`, the exact
`linux_preview_source_sha`, and `linux_preview_number`. With publication false,
the workflow retains a concrete release plan and checksum-verified artifacts.
Set `publish_linux_preview=true` only for the approved plan. Tags use
`linux-vX.Y.Z-preview.N`; engine versions and asset basenames stay `vX.Y.Z`.
Previews are never marked latest and do not replace Windows/macOS assets.

The script verifies repository/run identity, both native Linux jobs, the outer
artifact digests, inner SHA256SUMS, source/ggml commits, portable Vulkan flags
and test metadata. Uploads are checked while draft before publication; existing
tags are never moved and conflicting assets are refused. A matching interrupted
draft can resume. Publishing is restricted to the repository workflow token,
which avoids recursively starting unrelated release builds. The immutable
LINUX-PREVIEW.json index preserves provenance after CI artifacts expire.
