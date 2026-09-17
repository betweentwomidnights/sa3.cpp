---
license: other
license_name: stability-ai-community
license_link: https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md
base_model: RoyalCities/Foundation-1
tags:
- audio-generation
- foundation-1
- keybed
- sampler
- stable-audio-open
- gguf
- sa3.cpp
---

# Foundation-1.2 Keybeds GGUF

GGUF conversions of RoyalCities' [Foundation-1.2 Keybeds](https://huggingface.co/RoyalCities/Foundation-1)
checkpoint for [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp)'s native
stable-audio-tools runtime. Keybeds is a further finetune of Foundation-1 trained to keep one
sound's identity consistent across pitch, so a text prompt can become a playable instrument.
It shares Foundation-1's architecture, T5-base encoder, and Stable Audio Open Oobleck autoencoder.

Each encoding is a complete three-file bundle:

| Tier | Complete bundle | Guidance |
| --- | ---: | --- |
| F16 | 2,377 MiB | reference and runtime default |
| Q8_0 | 1,296 MiB | conservative high fidelity |
| Q5_K_M | 922 MiB | compact tier |
| Q4_K_M | 832 MiB | smallest footprint |

All quantized DiT tensors passed the per-tensor 0.990 cosine gate (Q8 178/0, Q5 175/0, Q4
175/0). Matched CUDA renders of two 12-note keybed previews (Rhodes and sine prompts, seed 42,
same SDE noise streams) were sliced into 24 notes per tier and compared against F16:

| Tier | Max pitch error | Octave errors | Envelope / log-mag cosine vs F16 |
| --- | ---: | ---: | ---: |
| F16 | 2.5 cents | 0 | reference |
| Q8_0 | 2.5 cents | 0 | 0.9998 / 0.9985 |
| Q5_K_M | 2.5 cents | 0 | 0.9992 / 0.9951 |
| Q4_K_M | 3.0 cents | 0 | 0.9980 / 0.9894 |

These paired metrics measure divergence from one F16 trajectory, not absolute audio quality.

## Keybeds

The model renders chromatic runs: a chunk of up to six notes, 3.0 s each with 0.25 s gaps, in one
conditioned window (`seconds_total` 20 for six notes). sa3.cpp's `sat/keybed.h` follows
RoyalCities' keybed tab and exporter (RC-stable-audio-tools `43dcbb4b`):

```text
Keybed, Sequence, Timbre Profile, Grand Piano, Warm, Dry, Chromatic Chunk, Note Sequence, C4, C#4, D4, D#4, E4, F4
```

- one seed shared by every chunk; RC's keyboards are prompt labels C2-B5, C2-F6, or C2-B6;
- DPM++ 3M SDE, 80 steps, CFG 6, sigma 0.03-500, no loudness normalization;
- slices on the 3.25 s grid, conservative -60 dB tail trim, 120 ms terminal fade.

**Pitch.** In our measurements every note renders exactly one octave below its prompt label
(piano, sine, and bass prompts; "A4" in a sine prompt is a 220 Hz tone). sa3.cpp maps samples
by sounding pitch, so a C2-B5 prompt range becomes MIDI keys C1-B4.

```powershell
cmake -S . -B build -DSA3_BUILD_SAT=ON -DSA3_CUDA=ON
cmake --build build --config Release --target sat-generate
python tools/download_models.py --sat --sat-model foundation-1.2-keybeds
sat-generate --model keybeds --prompt "Rhodes Piano, Warm, Soft" --seed 1234 --out-dir kit
```

This writes one WAV per sounding note plus `kit.sfz`. A full 48-note kit took 79.5 s at F16 on
an RTX 5070 Laptop GPU. `--keybed-preview C4:6` renders a single chunk.

Through libsa3's C ABI, create a context with `variant = "foundation-1.2-keybeds"` and send one
Generate request per chunk (`duration_seconds = 19.25`, `conditioning_seconds_total = 20`,
`sampler = SA3_SAMPLER_DPMPP_3M_SDE_V1`). The
[SA3Keybed](https://github.com/betweentwomidnights/foundation-1.2-iplug2) VST3/CLAP instrument
does this and plays the result from MIDI.

## Sources and licensing

- Foundation-1.2 Keybeds source: `RoyalCities/Foundation-1` revision
  `7b10fbbbc1be2f54cbc5540aab89ee383bc94e4a`, file `Foundation-1.2-Keybeds.safetensors`.
- Stable Audio Open 1.0 base revision: `f21265c1e2710b3bd2386596943f0007f55f802e`.
- T5-base revision: `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`.

The Stability AI Community License is included as `LICENSE.md`; RoyalCities' source
license notice is retained as `LICENSE_FOUNDATION.md`, and T5-base's Apache 2.0 license is
included as `LICENSE_T5.md`. See `NOTICE` for conversion details.

Powered by Stability AI.
