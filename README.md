# stable-audio-3 in c++

> **update 7/16/2026 — Vulkan backend LoRA training is now implemented via CLI. Metal is up next.**
>
> **i'm still not super happy with training speed using iGPUs. hoping to improve that over the next
> few days once we get the Metal backend working and i can circle back.**
>
> **nothing should be different for you at inference time, but plz let me know if any issues surface.**

trying to make this as composable and extensible as i can without over-engineering it too much. my hope is that this might eventually replace the sa3 backend i already use in [gary4local](https://github.com/betweentwomidnights/gary-localhost-installer), and start unifying that application for mac/pc. 

it might also just allow us to embed sa3 directly inside a JUCE/iPlug2 project. see [docs/EMBEDDING.md](docs/EMBEDDING.md).

because this is my first ggml project, i wanted to be the first to actually run it in downstream apps instead of just benchmarking it. so both surfaces are already tested end-to-end:

- the **http server** (`sa3-server`) as an optional backend in [sa3-ableton-extension](https://github.com/betweentwomidnights/sa3-ableton-extension/tree/backend/sa3.cpp) (branch `backend/sa3.cpp`) — there's an [embedded version of the extension](https://github.com/betweentwomidnights/sa3-ableton-extension/tree/backends/embedded-sa3) now too, running libsa3 in-process
- **libsa3** (the embedded c abi) runs the model in-process inside [sa3.cpp-iplug2-demo](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo)

there's also a browser web UI for the `sa3-server` http backend being built and validated on [pillopaus-project's fork](https://github.com/pillopaus-project/sa3.cpp).

## quickstart

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp

# 1. build a backend (own dir each, so they coexist)
./build.sh cuda        # or: cpu | vulkan | hip | metal | all     (windows: build.cmd cuda)

# 2. download a model set into ./models  (no python — curl from HuggingFace)
./models.sh            # windows: models.cmd    (add --training-base for native LoRA training)

# 3. put the tools on PATH for this shell (points SA3_MODELS_DIR at ./models too)
source ./env.sh        # windows:  env.cmd  (cmd)   or   . .\env.ps1  (powershell)

# 4. generate — --model resolves the gguf set in ./models by name
sa3-generate --model medium --prompt "upbeat funk groove with slap bass" --out song.wav
sa3-generate --model small-music --duration 12 --prompt "upbeat funk groove with slap bass" --out song.wav

# adapters resolve the same way: --lora <name> finds models/lora-<name>-*.gguf
sa3-generate --model medium --lora kev --lora keygen --prompt "neo-classical lofi hiphop 90bpm C# minor" --out song.wav
```

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

> note: **dora-rows and bora are both validated end-to-end against trained checkpoints at cossim 1.0**
> (kev/keygen for dora-rows; a trained koan bora adapter for bora). dora-cols and the -xs variants are
> formula-validated only (no trained checkpoint to a/b yet), but share the same apply path.

credits:

[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) was used as a bit of a guide here.

official upstream repo:

https://github.com/Stability-AI/stable-audio-3

License: MIT
