# Native LoRA Training

`sa3-train` trains a DiT LoRA/DoRA adapter directly in C++/ggml from MP3/caption pairs and writes an adapter GGUF that loads through the existing `sa3-generate --lora` path.

Training is currently validated on CUDA, Vulkan, and CPU. Vulkan v1 supports both discrete GPUs and
integrated GPUs, although iGPU throughput still has substantial room for improvement. Metal is
planned next. These training additions do not alter the existing inference path.

Measured training throughput and reproducible backend comparisons are collected in
[TRAINING_BENCHMARKS.md](TRAINING_BENCHMARKS.md).

## Build

```sh
./build.sh cpu
./build.sh cuda
./build.sh vulkan
```

CUDA is the fastest validated backend for real training runs. Vulkan training is supported on both
integrated and discrete GPUs, and CPU training honors the same thread controls as inference. See
[TRAINING_BENCHMARKS.md](TRAINING_BENCHMARKS.md) for measured throughput and
[NON_GPU_TESTS.md](NON_GPU_TESTS.md) for the registered CTest suite.

## Models

Download the normal inference GGUF set and its matching training base:

```sh
python tools/download_models.py --variant medium --encoding f16 --training-base
```

Training additionally needs the matching base DiT. Adapters are trained on `medium-base`,
`small-music-base`, or `small-sfx-base`, then applied to `medium`, `small-music`, or `small-sfx`
respectively at inference. `sa3-train --model <variant>` resolves the normal tokenizer, T5/Gemma
encoder, conditioner, and SAME files, but requires a separate
`stable-audio-3-<variant>-base-dit-*-F16.gguf`; model-name resolution does not fall back to the
ARC/post-trained inference DiT. See [the models README](../models/README.md#lora-training-bases) for
conversion details. The trainer checks `dit.training_base=true` and the matching model-family
metadata before allocating the backend, so merely renaming an inference GGUF is rejected.
Adapter checkpoint metadata and tensor shapes are documented in
[lora_checkpoint_contract.md](lora_checkpoint_contract.md).

F16 is the recommended training encoding. The frozen base DiT weights remain F16 while adapter
parameters and optimizer state remain F32, matching the reference trainer's memory-saving
`--base_precision fp16` configuration.

## Dataset

The expected dataset layout is:

```text
datasets/my-training-set/
  train/filelist.txt
  train/metadata.jsonl
  train/audio/*.mp3
  train/audio/*.txt
  test/...
  evaluation/...
```

Training honors `train/filelist.txt`. Test and evaluation splits are loaded only for validation/evaluation and are rejected if any train item overlaps by basename, canonical path, or `audio_sha256`.

## Train

After `env.cmd` (Command Prompt) or `. .\env.ps1` (PowerShell), a normal Windows run is one line:

```powershell
sa3-train --dataset C:\datasets\my-training-set --steps 1500
```

This uses the validated underfit-style recipe by default: medium-base F16, DoRA-rows rank/alpha 16,
512-frame random latent crops, seed 42, AdamW (`0.9/0.95` betas, `0.01` weight decay), gradient
clipping at 1.0, truncated-logit-normal timesteps with the model's distribution shift, CFG dropout,
inpainting, and the inverse-LR schedule. It also loads `prompt_config.json` from the dataset root when
present and creates a non-colliding `train-runs/<dataset-name>` output directory. Use flags or a JSON
config to override any of these defaults.

CPU builds use ggml's automatic thread count by default. As with inference, `SA3_THREADS` or
`--threads` selects it explicitly; for example:

```powershell
sa3-train --dataset C:\datasets\my-training-set --steps 1500 --threads 24
```

At successful completion the trainer prints a copy-pastable `sa3-generate` command containing the
run's inference model variant, final adapter checkpoint, first held-out evaluation caption, model
directory, and a `preview.wav` path inside the run directory.

The expanded equivalent is useful when documenting or changing individual settings:

```sh
build-cuda/bin/sa3-train \
  --model medium \
  --models-dir models \
  --dataset ../datasets/my-training-set \
  --adapter-type dora-rows \
  --rank 16 \
  --alpha 16 \
  --learning-rate 0.0001 \
  --frames 512 \
  --batch-size 1 \
  --steps 10000 \
  --checkpoint-every 500 \
  --out train-runs/my-training-run
```

The current graph executes one physical sample at a time. Use `--batch-size 1`.

The native target inventory contains the DiT's Linear/Conv weights: 228 targets for medium and 192
for small with the default inpainting path enabled. Disabling inpainting removes the 40 small-model
local-conditioning targets, leaving 152. The trainer intentionally does not adapt the separate
`seconds_total` conditioner Linear (the optional 229th medium or 193rd small reference target).
Stable Audio 3's training guide recommends `--exclude seconds_total` for small datasets to avoid
conditioner hijacking; use that exclusion for direct cross-framework comparisons.

Small-model training uses the same command with `--model small-music` or `--model small-sfx` and
requires its matching `*-base` DiT. Small-music and small-sfx share the same standard-attention
architecture and local/inpainting conditioning shape; a small-music smoke therefore exercises the
same native graph structure used by small-sfx.

`--adapter-type` accepts `lora`, `dora-rows`, `dora-cols`, `bora`, and their `-xs`
variants (`lora-xs`, `dora-rows-xs`, `dora-cols-xs`, `bora-xs`). The `-xs` families
freeze `U`/`V` as the top-`rank` SVD bases of each base weight and train only the small
`M_xs` core (plus DoRA/BoRA magnitudes). Those bases are computed natively by default;
pass `--svd-bases bases.gguf` to load precomputed bases instead — generate them with
`python tools/compute_svd_bases.py --dit models/... --rank 8 --out bases.gguf` for exact
`torch.linalg.svd` parity with the reference implementation.

Each sample is conditioned by its caption text. A dataset-level `prompt_config.json` can optionally
compose that caption with general metadata tags or path-derived text.

## Outputs

The output directory contains:

- `adapter-step-*.gguf`
- `trainer-state-step-*.gguf`
- `adapter-final.gguf`
- `metrics.jsonl`
- `config.snapshot.txt`
- `config.resume.snapshot.txt` (latest resumed invocation, when applicable)
- `command.txt`

## Resume Training

Every step checkpoint is an immutable pair. `adapter-step-N.gguf` is the normal, lean adapter and
can be passed directly to `sa3-generate`; `trainer-state-step-N.gguf` is its training-only sidecar.
The sidecar contains the AdamW moments, optimizer/LR-scheduler step, dataset cursor and shuffled
order, plus the stochastic states used for crops, prompts, CFG dropout, inpainting, timesteps, and
diffusion noise. Keeping those tensors separate prevents inference from loading optimizer state.

Pass either member of the pair to `--resume`. `--steps` is the total target step, not the number of
additional updates. For example, this continues a step-500 CPU or CUDA run through step 1500:

```powershell
sa3-train --dataset C:\datasets\my-training-set --resume C:\dev\sa3.cpp\train-runs\my-training-set\trainer-state-step-500.gguf --steps 1500
```

When `--out` is omitted, a resumed run continues in the checkpoint's directory and appends to
`metrics.jsonl` and `command.txt`. Use `--out` to branch the continuation into a new directory.
The trainer rejects changes to trajectory-defining model, dataset, adapter, optimizer, conditioning,
or sampling settings. `--steps`, checkpoint cadence, output location, and evaluation-only settings
may change. Resume the latest checkpoint when continuing in place because step pairs are immutable;
this prevents an older continuation from silently overwriting newer state.

The final update is always saved as a resumable numbered pair even when it does not fall on
`--checkpoint-every`; `adapter-final.gguf` remains the convenient inference copy. Weight-only
warm-starting is intentionally not called resume and may later be exposed separately as
`--init-adapter`.

## Generate With The Adapter

```sh
build-cuda/bin/sa3-generate \
  --model medium \
  --models-dir models \
  --lora train-runs/my-training-run/adapter-final.gguf \
  --prompt "the held-out caption to evaluate" \
  --frames 128 \
  --steps 8 \
  --seed 42 \
  --out train-runs/my-training-run/evaluation.wav
```

## Troubleshooting

- Missing model files: run `python tools/download_models.py --variant medium --training-base`.
- Unmarked training DiT: delete any pre-release base-named GGUF and redownload it; the downloader
  resumes existing files and cannot replace an older complete file automatically.
- Missing captions or audio: inspect the split `filelist.txt` and `metadata.jsonl`; training fails before model loading.
- Split contamination: remove the duplicate item from train or held-out splits.
- FFmpeg decode failure: confirm `ffmpeg` is installed and can decode the MP3.
- CUDA build failure: verify the CUDA Toolkit is installed and rerun `./build.sh cuda`.
- Vulkan build failure: verify the Vulkan SDK is installed and rerun `./build.sh vulkan`.
