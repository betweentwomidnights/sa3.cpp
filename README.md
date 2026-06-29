# stable-audio-3 in c++

a portable c++/ggml port of stable audio 3 — prompt string → music, no pytorch in the loop.
runs on cpu, cuda, or vulkan; weights in f32 or f16. every component is validated against the
pytorch reference at cosine similarity ~1.0.

## quickstart

```bash
git clone --recurse-submodules https://github.com/betweentwomidnights/sa3.cpp.git
cd sa3.cpp

# 1. build a backend (own dir each, so they coexist)
./build.sh cuda        # or: cpu | vulkan | hip | metal | all     (windows: build.cmd cuda)

# 2. download a model set into ./models  (needs: python3 -m pip install huggingface_hub)
python3 tools/download_models.py --variant medium --encoding f16

# 3. generate
./build-cuda/bin/sa3-generate --tok models/t5gemma-b-b-ul2-v1.0-vocab.gguf \
    --t5 models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf \
    --cond models/stable-audio-3-medium-conditioner-v1.0-F32.gguf \
    --dit models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
    --same models/stable-audio-3-medium-same-l-v1.0-F16.gguf \
    --prompt "upbeat funk groove with slap bass" --frames 128 --steps 8 --out song.wav
```

build needs cmake + a c++17 compiler (Visual Studio 2022 on windows). cuda needs the CUDA
Toolkit; vulkan needs the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home); metal is macOS-only.
backend + packaging details: [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) ·
[docs/VULKAN.md](docs/VULKAN.md) · [docs/METAL.md](docs/METAL.md).

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
- [ ] hip/rocm for amd

> note: still want to do more testing to confirm the adapters are working just like the pytorch version
> (dora-rows is validated end-to-end at cossim 1.0; the other adapter types are formula-validated but
> not yet a/b'd against a trained checkpoint — a bora training run is in progress for that)

credits:

[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) was used as a bit of a guide here.

official upstream repo:

https://github.com/Stability-AI/stable-audio-3

License: MIT
