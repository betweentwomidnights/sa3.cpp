# Native LoRA Training

`sa3-train` trains a DiT LoRA/DoRA adapter directly in C++/ggml from MP3/caption pairs and writes an adapter GGUF that loads through the existing `sa3-generate --lora` path.

Training is currently validated on CUDA, Vulkan, Metal, and CPU. Vulkan v1 supports both discrete
and integrated GPUs. Metal training is validated on Apple M4 with both small and medium models;
its correctness and memory use are strong, and the tiled Metal `OUT_PROD` kernel brings the
full-target medium trainer close to matched MLX throughput. Further graph and model-shape
optimization remains possible. These training additions do not alter the existing inference path.

Measured training throughput and reproducible backend comparisons are collected in
[TRAINING_BENCHMARKS.md](TRAINING_BENCHMARKS.md).

## Quickstart

Everything below has a working default. In practice these are the four knobs worth touching:

```sh
sa3-train --dataset /path/to/dataset --steps 2000 \
          --rank 16 --duration 23 --lora-scope core
```

| flag | default | what it does |
|---|---|---|
| `--steps N` | `10000` | optimizer updates. 2000 is a reasonable first run on a small dataset. |
| `--rank N` | `16` | adapter capacity. `--alpha` follows it automatically, so raising rank does **not** silently weaken the adapter. |
| `--duration SEC` | — | seconds of audio per training crop. Without it you get `--frames 512`, which is **47.6 s**. |
| `--lora-scope full\|core` | `full` | `core` adapts only the 24 blocks' own projections (168 weights instead of 228): ~10% faster steps and ~11% smaller adapters, and in a matched 2000-step run the loss curve was within 0.0013 of `full` throughout. |

Two things that are easy to miss:

- **`--duration` is in seconds; `--frames` is in latent frames.** One latent frame is 4096 samples at
  44.1 kHz, so 512 frames = 47.6 s, 256 = 23.8 s. `--duration` is the friendlier knob and wins when
  both are given. Shorter crops train faster and use less memory; the reference regime is ~47 s.
- **MP3 datasets need `ffmpeg` on `PATH`.** Decoding shells out to it (see [Dataset](#dataset)).
  WAV inputs do not.

Training prints what it is actually doing at startup, which is worth a glance before walking away:

```
[train] lora scope: core -> 168 targets (168 per-block, 0 elsewhere)  rank 16 alpha 16 scale 1
```

`scale` is `alpha/rank` and should normally be `1`. Adapter strength is multiplied by it in both
training and inference, so a `scale` well below 1 means the adapter is being attenuated.

## Build

```sh
./build.sh cpu
./build.sh cuda
./build.sh vulkan
./build.sh metal
```

CUDA is the fastest validated backend for real training runs. Vulkan training is supported on both
integrated and discrete GPUs, Metal training is supported on Apple Silicon, and CPU training honors
the same thread controls as inference. See [TRAINING_BENCHMARKS.md](TRAINING_BENCHMARKS.md) for measured throughput and
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

**MP3 decoding requires `ffmpeg` on `PATH`.** `sa3-train` shells out to it (`ffmpeg -f f32le …`) to
read compressed audio; there is no built-in MP3 decoder. Inference has no such dependency — it reads
WAV directly — so a machine that generates fine can still fail to train. Check with `ffmpeg -version`
before a long run; a missing binary surfaces as a decode failure per file rather than a clear
"ffmpeg not found".

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

All eight train, and all eight are within about 3x of each other per step:

| adapter type | s/step | first step |
|---|---|---|
| `lora` | 0.80 | 1.02 |
| `lora-xs` | 1.05 | 1.33 |
| `dora-rows` (default) | 1.25 | 6.17 |
| `dora-rows-xs` | 1.57 | 2.00 |
| `dora-cols-xs` | 1.84 | 2.31 |
| `dora-cols` | 1.95 | 2.35 |
| `bora-xs` | 2.33 | 2.67 |
| `bora` | 2.47 | 2.88 |

CUDA / RTX 5070 Laptop, medium, rank 16, the reference 512 frames, averaged over steps 10-40. A
2000-step run is 30-90 minutes depending on family; the two measured end to end came in at 34 min
(`dora-rows`) and 64 min (`dora-cols`).

Measure over enough steps to reach steady state. The first step carries graph build and, for
`dora-rows`, the one-off `base_norm_sq` reduction over every targeted weight — 6.17 s against a
1.25 s steady step, so a 2- or 3-step average is mostly setup and ranks the families wrongly.

Two things used to make this table much worse, both fixed:

- `dora-cols` and `dora-cols-xs` aborted during the backward pass on
  `GGML_ASSERT(ggml_is_padded_1d(a))`. ggml differentiates transpose into a non-contiguous view
  and `ggml_scale` rejects one, so the column-norm helper's transposed gradient reached the
  scale on the low-rank delta. It now interposes a `cont`, whose backward re-materializes the
  gradient contiguously.
- Six of the eight ran a monolithic backward asking for 24.6 GiB of allocator on an 8 GiB card,
  so the driver paged it to system RAM: `dora-cols` measured 92.76 s/step at 512 frames against
  1.95 now, with the allocator down to 1126 MiB. Gradient checkpointing had been gated to the two
  families `dit_lin` can apply functionally; the rest now build their effective weight inside
  whichever segment graph consumes it, so they are checkpointed too. See `--checkpoint-backward`.

`dora-rows` remains the default and the best-trodden path.

Each sample is conditioned by its caption text. A dataset-level `prompt_config.json` can optionally
compose that caption with general metadata tags or path-derived text.

### Path-derived captions are off by default

`use_paths` defaults to **false**, which is a deliberate departure from underfit, where filenames
are part of the dice pool. Turn it on only if your filenames actually read as descriptions of the
music:

```json
{ "prompt_config": { "use_tags": true, "use_paths": true, "balance": { "tags": 40, "paths": 30 } } }
```

With that on, roughly a third of steps train on the file path verbatim — and `path_hide_ext`,
`path_hide_dirs` and `path_hide_topmost` all default to false, so a track lands in the model as

```
03 - Gettysburg [Q-gtqz_il-w].wav
```

A track number, a YouTube ID and a file extension are not a description of a piece of music, and
training on them teaches the adapter to emit your material in response to noise. That dilutes the
conditioning the rest of the run is trying to build, and it shows up at inference as an adapter
that responds best to no prompt at all rather than to a prompt from its training distribution.

If your filenames are genuinely descriptive, `use_paths` is worth having, and `hide_ext` /
`hide_dirs` / `hide_topmost_dir` trim the parts that are not. Otherwise leave it off and let the
tags plus the CFG dropout (`--cfg-dropout-prob`, default 0.1) do the generalizing.

Check what a run is actually training on before committing hours to it — the per-step log prints
the composed caption, and a quick tally of `prompt="..."` across the log will show the mix.

## Target Scopes

Which DiT weights get an adapter. Every 2D `dit.*.weight` is eligible; a scope narrows that.

| scope | targets | what it covers |
|---|---|---|
| `full` (default) | 228 | the 9 per-block projections plus the 12 outside the blocks (`proj_in`/`proj_out`, `pre_conv`/`post_conv`, `time_embed`, `cond_embed`, `global_embed`, `gce`) |
| `core` | 168 | the 7 per-block projections only — `self.qkv`, `self.out`, `cross.q`, `cross.kv`, `cross.out`, `ff.proj`, `ff.out` — across 24 blocks |

`core` is the scope the MLX trainer uses, where it is spelled `include transformer.layers` /
`exclude to_local_embed`. Those are upstream names; ours are `dit.<N>.…` and `local.{0,2}`, so the
filters do not transfer literally — see [TENSOR-MAP.md](TENSOR-MAP.md) for the full mapping.

Measured on medium, 23 s crops, rank 16, 2000 steps, CUDA:

| | full (228) | core (168) |
|---|---|---|
| final loss | 0.8830 | 0.8843 |
| s/step | 0.718 | 0.642 |
| adapter | 82.3 MB | 72.7 MB |

Same quality — the loss curves stayed within 0.0014 of each other in every 250-step window — for
~10% faster steps and ~11% smaller adapters.

Worth knowing why those savings are ~11% and not the 26% the target count suggests: **the dropped
weights are the small ones.** `core` drops 26% of the targets but only 11.6% of the adapter
parameters (19.06 M kept vs 2.51 M dropped), because the large projections are all kept and
`local.{0,2}` plus the head/tail weights are comparatively tiny. The step-time saving tracks the
parameter ratio rather than the target ratio, since the frozen base still runs in full either way.
`core` does skip one graph entirely — the head backward pass has nothing to differentiate — but that
turns out to be nearly free.

For anything the named scopes do not carve out, `--lora-include` and `--lora-exclude` take
comma-separated substrings matched against tensor names, applied after the scope:

```sh
--lora-scope core --lora-exclude ff.        # 120 targets: core minus both feed-forward projections
--lora-scope full --lora-include cross.     #  72 targets: cross-attention only
```

A scope change invalidates a checkpoint — the resume fingerprint covers every target name — so
`--resume` refuses rather than continuing into a different parametrization. Pick a scope before a
long run, not partway through.

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
additional updates. For example, this continues a step-500 CPU, CUDA, or Vulkan run through step
1500:

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
