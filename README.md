# stable-audio-3 in c++

a portable c++/ggml port of stable audio 3 — prompt string → music, no pytorch in the loop.
runs on cpu, cuda, vulkan, or metal; weights in f32 or f16. every component is validated against the
pytorch reference at cosine similarity ~1.0.

## quickstart

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp

# 1. build a backend (own dir each, so they coexist)
./build.sh cuda        # or: cpu | vulkan | hip | metal | all     (windows: build.cmd cuda)

# 2. download a model set into ./models  (needs: python3 -m pip install huggingface_hub)
python3 tools/download_models.py --variant medium --encoding f16

# 3. generate — --model resolves the gguf set in ./models by name
./build-cuda/bin/sa3-generate --model medium --prompt "upbeat funk groove with slap bass" --out song.wav

# adapters resolve the same way: --lora <name> finds models/lora-<name>-*.gguf
./build-cuda/bin/sa3-generate --model medium --lora kev --lora keygen --prompt "breakcore 140bpm" --out song.wav
```

(`--model` is a convenience over the explicit `--tok/--t5/--cond/--dit/--same` flags, which still
work and override it per-slot. `--encoding f32` and `--models-dir DIR` adjust what it resolves.)

**configuration.** the model/adapter dirs (and the backend knobs) read from env vars, so a downstream
app sets them in the process it spawns and never touches the CLI. drop a `.env` in the working dir to
set them locally — see [`.env.example`](.env.example). precedence is **flag > env var > `.env` > default**:

| | env var | flag | default |
|---|---|---|---|
| base ggufs | `SA3_MODELS_DIR` | `--models-dir` | `models/` |
| adapters (`--lora <name>`) | `SA3_ADAPTERS_DIR` | `--adapters-dir` | = models dir |
| device / flash | `SA3_DEVICE` `SA3_GPU` `SA3_FLASH_ATTN` | — | auto |

build needs cmake + a c++17 compiler (Visual Studio 2022 on windows). cuda needs the CUDA
Toolkit; vulkan needs the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home); metal is macOS-only.
backend + packaging details: [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) ·
[docs/VULKAN.md](docs/VULKAN.md) · [docs/METAL.md](docs/METAL.md) · [docs/HIP.md](docs/HIP.md).
there's also a small HTTP server (`./server.sh` / `server.cmd`) — see [docs/SERVER.md](docs/SERVER.md).

what works:

- text2music, audio2audio, inpainting / continuation
- both sizes: medium (same-l) + small-music (same-s)
- lora / dora / bora adapters (+ xs variants) — runtime strength + multi-adapter blending,
  applied in weight space (not a static merge)
- cuda backend + fp16 — medium generation ~3.5s end-to-end on an 8gb laptop 5070, and long-form
  (sliding-window decoder) scales linearly. see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

what's next:

- [x] same-s / stable-audio-3-small-music and sfx
- [x] audio2audio
- [x] inpainting
- [x] loras (lora/dora/bora + xs variants, runtime strength + multi-adapter blending)
- [x] cuda backend + fp16
- [x] benchmark generation times and stuff ([docs/BENCHMARKS.md](docs/BENCHMARKS.md))
- [x] vulkan backend ([docs/VULKAN.md](docs/VULKAN.md))
- [x] metal backend builds + smoke-tests on Apple M4
- [ ] hip/rocm for amd — scaffolded (`./build.sh hip`) but **untested, [ROCm tester wanted](docs/HIP.md)** 🙏

> note: **dora-rows and bora are both validated end-to-end against trained checkpoints at cossim 1.0**
> (kev/keygen for dora-rows; a trained koan bora adapter for bora). dora-cols and the -xs variants are
> formula-validated only (no trained checkpoint to a/b yet), but share the same apply path.

credits:

[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) was used as a bit of a guide here.

official upstream repo:

https://github.com/Stability-AI/stable-audio-3

License: MIT
