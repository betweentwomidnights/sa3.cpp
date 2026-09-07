---
license: other
license_name: stability-ai-community
license_link: https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md
tags:
- audio-generation
- stable-audio-open
- gguf
- sa3.cpp
---

# Stable Audio Open 1.0 GGUF

GGUF conversions of [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
for [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp)'s optional native
stable-audio-tools runtime.

Each encoding is a self-contained DiT, 128-token T5-base encoder, and Oobleck decoder
bundle. The T5 and Oobleck artifacts are byte-identical to the files in the Foundation-1
GGUF repository.

| Tier | Complete bundle | Guidance |
| --- | ---: | --- |
| F16 | 2,377 MiB | reference and runtime default |
| Q8_0 | 1,296 MiB | conservative high fidelity |
| Q5_K_M | 922 MiB | compact |
| Q4_K_M | 832 MiB | smallest footprint |

All quantized tensors passed the per-tensor 0.990 cosine gate. Matched CUDA renders
measured envelope/log-magnitude cosine of 0.9959/0.9784 at Q8, 0.9021/0.8742 at Q5,
and 0.9071/0.8789 at Q4. All four tiers passed listening tests; Q8 remains the
conservative recommendation for this model.

The official inference profile is DPM++ 3M SDE, sigma 0.3–500, 100 steps, and CFG 7.
Stable Audio Open 1.0 supports up to approximately 47 seconds at 44.1 kHz stereo.

```powershell
cmake -S . -B build-sat -DSA3_BUILD_SAT=ON -DSA3_CUDA=ON
cmake --build build-sat --config Release --target sat-generate
python tools/download_models.py --sat --sat-model stable-audio-open-1.0
$env:SA3_DEVICE="cuda"
sat-generate --model stable-audio-open-1.0 --prompt "cinematic ambient soundscape" `
  --seconds 47 --seed 42 --out sao1.wav
```

## Sources and licensing

- Stable Audio Open 1.0 source revision: `f21265c1e2710b3bd2386596943f0007f55f802e`.
- T5-base revision: `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`.

The Stable Audio weights remain under the Stability AI Community License included as
`LICENSE.md`. T5-base is covered by `LICENSE_T5.md`. See `NOTICE` for conversion details.
These artifacts rename tensors, serialize them as GGUF, and quantize selected tensors;
they do not retrain the model.

Powered by Stability AI.
