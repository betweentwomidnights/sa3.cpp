# stable-audio-3 in c++

## built with sa3.cpp

These applications run sa3.cpp outside this repository:

| Application | What it uses | Download |
|---|---|---|
| [Foundation Keys](https://github.com/betweentwomidnights/foundation-1.2-iplug2) | Foundation-1.2 Keybeds text-to-synth through the Stable Audio Tools backend | [v0.1.2 for Windows and macOS](https://github.com/betweentwomidnights/foundation-1.2-iplug2/releases/tag/v0.1.2) |
| [SA3 iPlug2 demo](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo) | Stable Audio 3 VST3 and REAPER extension using the embedded C ABI | [v0.4.0 for Windows](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo/releases/tag/v0.4.0) |
| [SA3 Ableton extension](https://github.com/betweentwomidnights/sa3-ableton-extension) | Stable Audio 3 inside Ableton Live using the embedded C ABI | [v0.2.0 for Windows](https://github.com/betweentwomidnights/sa3-ableton-extension/releases/tag/v0.2.0) |
| [sa3.cpp iOS](https://github.com/betweentwomidnights/sa3.cpp-ios) | Experimental on-device LoRA training and inference on an iPhone 13 | [Source and build instructions](https://github.com/betweentwomidnights/sa3.cpp-ios) (no release yet) |

sa3.cpp runs Stable Audio 3 and the Stable Audio Tools family locally with ggml.
It provides generation and LoRA training CLIs, an embeddable C ABI, and a local
browser studio for Stable Audio 3 inference and training. The released applications above show
these runtimes in DAWs and on device. See [embedding](docs/EMBEDDING.md),
[training](docs/TRAINING.md), and [runtime packaging](docs/RUNTIME_RELEASE.md).

The browser inference and training interfaces originated with
[pillopaus-project](https://github.com/pillopaus-project/sa3.cpp). Both original
commit authorship and visible UI credit are preserved. Run `sa3-server` on port
8006 and `sa3-train-web` on port 8016 to use the two linked views; see
[server](docs/SERVER.md) and [training web UI](docs/TRAINING_WEB.md).
The planned sample pad and take workflow is described in the
[studio roadmap](docs/STUDIO_ROADMAP.md).

## quickstart

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp

# 1. build a backend (own dir each, so they coexist)
./build.sh cuda        # or: cpu | vulkan | hip | metal | all     (windows: build.cmd cuda)

# 2. download a model set into ./models  (no python — curl from HuggingFace)
./models.sh            # windows: models.cmd    (add --training-base for native LoRA training)
./models.sh --encoding q4_k_m --training-base   # q4 dit (+ f32 autoencoder) and a q4 base to train on
./models.sh --encoding q4_k_m --ae-encoding q4_k_m --training-base   # ...quantize the autoencoder too

# 3. put the tools on PATH for this shell (points SA3_MODELS_DIR at ./models too)
source ./env.sh        # windows:  env.cmd  (cmd)   or   . .\env.ps1  (powershell)

# 4. generate — --model resolves the gguf set in ./models by name
sa3-generate --model medium --prompt "upbeat funk groove with slap bass" --out song.wav
sa3-generate --model small-music --duration 12 --prompt "upbeat funk groove with slap bass" --out song.wav

# adapters resolve the same way: --lora <name> finds models/lora-<name>-*.gguf
sa3-generate --model medium --lora kev --lora keygen --prompt "neo-classical lofi hiphop 90bpm C# minor" --out song.wav
```

The classic Stable Audio family is an optional component, excluded from ordinary builds. Enable
`SA3_BUILD_SAT` and build its `sat-generate` CLI explicitly. The default SAOS download is the
all-F16 reference bundle; pass `--encoding q5_k_m` for the recommended compact tier:

```bash
cmake -S . -B build-sat -DSA3_BUILD_SAT=ON -DSA3_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-sat --target sat-generate
./models.sh --sat
SA3_DEVICE=metal build-sat/bin/sat-generate --model arc \
  --prompt "A short, beautiful piano riff in C minor" --seconds 11 --out saos.wav
```

See [docs/STABLE_AUDIO_OPEN_SMALL.md](docs/STABLE_AUDIO_OPEN_SMALL.md) for finetunes,
samplers, explicit component paths, conversion, and quantization results.

Stable Audio Open 1.0 and Foundation-1 use the same optional build. Their
self-contained bundles are downloaded with the Python helper (F16 is the default):

```bash
python3 -m pip install -U "huggingface_hub"
python3 tools/download_models.py --sat --sat-model foundation-1
SA3_DEVICE=metal build-sat/bin/sat-generate --model foundation-1 \
  --randomize --bars 4 --bpm 128 --seed 42 --out foundation.wav
```

Pass `--sat-model sao1` to the downloader and `--model sao1` to `sat-generate`
for Stable Audio Open 1.0. See
[docs/STABLE_AUDIO_OPEN_1.md](docs/STABLE_AUDIO_OPEN_1.md) for timing controls,
Foundation prompt randomization, named profiles, and quantization results.

updating an existing checkout across the one-time ggml URL migration:

```bash
git -c fetch.recurseSubmodules=false pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

sa3.cpp pins an exact revision of its public
[`betweentwomidnights/ggml`](https://github.com/betweentwomidnights/ggml) fork. existing checkouts
cache the old submodule URL. disabling recursive submodule fetching for that first pull lets Git
receive the new parent revision before `sync` replaces the cached URL; otherwise some Git setups
try to fetch the new pin from the old `ggml-org/ggml` remote and report `not our ref`. `update`
then moves ggml to the backend patch revision tested by that sa3.cpp commit. none of these commands
uses a moving ggml branch. if you have local changes inside `ggml/`, commit or stash them first.

(`--model` is a convenience over the explicit `--tok/--t5/--cond/--dit/--same` flags, which still
work and override it per-slot. `--encoding f32` and `--models-dir DIR` adjust what it resolves.
Use `--duration SEC` for an exact output length, or `--frames N` for the lower-level latent length.)

**configuration.** the model/adapter dirs (and the backend knobs) read from env vars, so a downstream
app sets them in the process it spawns and never touches the CLI. drop a `.env` in the working dir to
set them locally — see [`.env.example`](.env.example). precedence is **flag > env var > `.env` > default**:

| | env var | flag | default |
|---|---|---|---|
| base ggufs | `SA3_MODELS_DIR` | `--models-dir` | `models/` |
| adapters (`--lora <name>`) | `SA3_ADAPTERS_DIR` | `--adapters-dir` | = models dir |
| source adapter exports | `SA3_SOURCE_LORAS_DIR` | `--source-loras-dir` | `loras/` |
| prompt dice pools | `SA3_PROMPTS_DIR` | `--prompts-dir` | `prompts/` |
| device / cpu threads / flash | `SA3_DEVICE` `SA3_GPU` `SA3_THREADS` `SA3_FLASH_ATTN` | `--threads` | auto |

build needs cmake + a c++17 compiler (Visual Studio 2022 on windows). cuda needs the CUDA
Toolkit; vulkan needs the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home); metal is macOS-only.
backend + packaging details: [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) ·
[docs/VULKAN.md](docs/VULKAN.md) · [docs/METAL.md](docs/METAL.md) · [docs/HIP.md](docs/HIP.md).
The opt-in classic stable-audio-tools family is documented in
[docs/STABLE_AUDIO_OPEN_SMALL.md](docs/STABLE_AUDIO_OPEN_SMALL.md) and
[docs/STABLE_AUDIO_OPEN_1.md](docs/STABLE_AUDIO_OPEN_1.md); it is excluded from default builds.
there's also a small HTTP server (`./server.sh` / `server.cmd`) — see [docs/SERVER.md](docs/SERVER.md).
native adapter training is documented in [docs/TRAINING.md](docs/TRAINING.md), with measured backend
and PyTorch comparisons in [docs/TRAINING_BENCHMARKS.md](docs/TRAINING_BENCHMARKS.md).
With the matching base DiT downloaded (`models.cmd --training-base` on Windows), the common training
path is deliberately just one command after `env.cmd`:

```powershell
sa3-train --dataset C:\dev\datasets\my-training-set --steps 1500
```

On the 8 GB laptop 5070 used for development, the default medium-base CUDA recipe currently averages
1.065 seconds per update: a projected 44 minutes for 2,500 steps, versus 74 minutes for the completed
PyTorch reference job on the same machine.

The validated medium-base DoRA recipe is the default; model, optimizer, crop, conditioning, and
output settings remain available as overrides for advanced runs. Periodic checkpoints are
restart-safe: `--resume trainer-state-step-N.gguf --steps TOTAL` restores the adapter, AdamW,
dataset cursor, and stochastic streams exactly.

what works:

- text2music, audio2audio, inpainting / continuation
- both sizes: medium (same-l) + small-music (same-s)
- lora / dora / bora adapters (+ xs variants) — runtime strength + multi-adapter blending,
  applied in weight space (not a static merge)
- cuda backend + fp16 — medium generation ~3.5s end-to-end on an 8gb laptop 5070, and long-form
  (sliding-window decoder) scales linearly. see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- vulkan backend + fp16 — validated on NVIDIA dGPU, Intel iGPU, and AMD Radeon 780M/RADV,
  including long-form via `--chunked-decode`. this is the working path on AMD (not HIP).
  see [docs/VULKAN.md](docs/VULKAN.md).

what's next:

- [x] same-s / stable-audio-3-small-music and sfx
- [x] audio2audio
- [x] inpainting
- [x] loras (lora/dora/bora + xs variants, runtime strength + multi-adapter blending)
- [x] cuda backend + fp16
- [x] benchmark generation times and stuff ([docs/BENCHMARKS.md](docs/BENCHMARKS.md))
- [x] vulkan backend, including iGPU/APU selection ([docs/VULKAN.md](docs/VULKAN.md))
- [x] vulkan/radv long-form stability — 780M/RADV lost the device around 200s on a monolithic
  decode; `--chunked-decode` fixes it (thanks @bakamomi for confirming on 780M/RADV)
- [x] metal backend builds + smoke-tests on Apple M4
- [ ] hip/rocm for amd — not working: selects the device but gfx1103/780M aborts in HIP
  code-object loading (#6). on that hardware use the vulkan backend + `--chunked-decode` instead.
  **[ROCm tester wanted](docs/HIP.md)** 🙏
- [ ] cross-backend seed reproducibility — the same seed gives a *different-but-valid* result on cuda vs vulkan
  (tensor-core matmul accumulation, not the RNG — the noise is already deterministic). worth a "precise mode"
  toggle / philox-style approach (cf. [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)) so a seed
  carries across backends for A/B testing

> note: **generation** handles all eight families. dora-rows and bora are validated against trained
> checkpoints at cossim 1.0 (kev/keygen for dora-rows; a trained koan bora adapter for bora);
> dora-cols and the -xs variants are formula-validated only (no trained checkpoint to a/b yet) but
> share the same apply path.
>
> **training an adapter here** covers all eight, at 0.8–2.5 s/step on a laptop 5070 (medium, rank
> 16, the reference 512 frames) — a 2000-step run measured 34 min for dora-rows and 64 min for
> dora-cols. see [docs/TRAINING.md](docs/TRAINING.md).

credits:

[dada-bots/underfit](https://github.com/dada-bots/underfit) is our source of truth for all things
LoRA/DoRA training with Stable Audio 3.

[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) was used as a bit of a guide here.

classic stable-audio-tools references:

[Stability AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) is the original
PyTorch implementation. [RoyalCities/RC-stable-audio-tools](https://github.com/RoyalCities/RC-stable-audio-tools)
is the Foundation inference fork, and the original
[Foundation-1 model card](https://huggingface.co/RoyalCities/Foundation-1) documents the checkpoint.

official upstream repo:

https://github.com/Stability-AI/stable-audio-3

License: MIT
