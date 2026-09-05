---
license: other
license_name: stability-ai-community
license_link: https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE
tags:
- audio-generation
- stable-audio-open-small
- gguf
- sa3.cpp
---

# Stable Audio Open Small GGUF

GGUF conversions for [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp)'s optional native
stable-audio-tools runtime. This repository is deliberately scoped to Stable Audio Open
Small (SAOS). Stable Audio Open 1.0 and Foundation-1 belong in separate model repositories.

The root contains the released ARC-trained SAOS DiT and the shared T5-base encoder and
Oobleck decoder. Validated full-checkpoint finetunes are nested under `finetunes/`:

- `finetunes/kickbass/`: `S3Sound/kickbass`, checkpoint
  `kickbass_v1_saos_e257_s18870.ckpt`.
- `finetunes/jerry-grunge/`: `thepatch/jerry_grunge`, preencoded checkpoint
  `jerry_encoded_bs64_epoch=374-step=3000.ckpt`.

## Quantization choices

Every component is published at F16, Q8_0, Q5_K_M, and Q4_K_M. Use the same tier for
all three components unless you have a reason to mix them.

| Tier | Complete bundle | Intended use |
| --- | ---: | --- |
| F16 | 1,010 MiB | reference |
| Q8_0 | 569 MiB | near-transparent |
| Q5_K_M | 416 MiB | recommended default |
| Q4_K_M | 379 MiB | smallest footprint |

Matched 11-second renders validated all four tiers on ARC, KickBass, and Jerry Grunge.
Q5_K_M was consistently robust and is only 37 MiB larger than the all-Q4 bundle.

## Download and run

SAOS is not part of the default sa3.cpp build. Enable the optional SAT component:

```powershell
cmake -S . -B build-saos -DSA3_BUILD_SAT=ON -DSA3_CUDA=ON
cmake --build build-saos --config Release --target saos-generate
python tools/download_models.py --sat --sat-model saos --saos-variant arc
```

Then generate directly from the canonical filenames:

```powershell
$env:SA3_DEVICE="cuda"
$env:SA3_FLASH_ATTN="1"
saos-generate --model arc --prompt "A short, beautiful piano riff in C minor" `
  --seconds 11 --seed 1234 --out saos.wav
```

For a finetune, download and select the same variant name:

```powershell
python tools/download_models.py --sat --saos-variant jerry-grunge
saos-generate --model jerry-grunge `
  --prompt "chillhop synth warm bass dusty drums sidechain 90 bpm" `
  --seconds 11 --seed 4242 --out jerry-grunge.wav
```

ARC metadata selects ping-pong sampling with 8 steps and CFG 1. The two ordinary
rectified-flow finetunes select Euler with 50 steps and CFG 4. `--sampler dpmpp --steps 40
--cfg-scale 4` is another validated finetune configuration.

## Sources and licensing

- SAOS source: `stabilityai/stable-audio-open-small`, revision
  `dc620d91535857b72ebb59b4ca45978db6d417f5`.
- KickBass source: `S3Sound/kickbass`, revision
  `44e51ff50d14f34654a9305580687e9fe875223f`. Redistribution here is with direct
  permission from S3Sound.
- Jerry Grunge source: `thepatch/jerry_grunge`, revision
  `d23abf859f51af6587da6f1397306d18c8c42edd`.
- T5-base source: `google-t5/t5-base`, revision
  `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`, under Apache License 2.0.

The Stable Audio weights and derived finetunes remain subject to the Stability AI
Community License included as `LICENSE.md`. T5-base is covered by `LICENSE_T5.md`.
See `NOTICE` for attribution and modification details. These artifacts rename tensors,
serialize them as GGUF, and quantize selected tensors; they do not merge the finetunes
or change their model architecture.

Powered by Stability AI.
