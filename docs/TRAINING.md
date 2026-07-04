# Native LoRA Training

`sa3-train` trains a DiT LoRA/DoRA adapter directly in C++/ggml from MP3/caption pairs and writes an adapter GGUF that loads through the existing `sa3-generate --lora` path.

## Build

```sh
./build.sh cpu
./build.sh cuda
```

CUDA is the target backend for real training runs. CPU builds are useful for parser, dataset, and checkpoint tests.

## Models

Download the base GGUF models first:

```sh
python tools/download_models.py --variant medium
```

`sa3-train --model medium --models-dir models` resolves the tokenizer, T5/Gemma encoder, conditioner, DiT, and SAME GGUFs with the same naming convention as `sa3-generate`.

## Dataset

The expected dataset layout is:

```text
datasets/mnesia-audio-v1/
  train/filelist.txt
  train/metadata.jsonl
  train/audio/*.mp3
  train/audio/*.txt
  test/...
  evaluation/...
```

Training honors `train/filelist.txt`. Test and evaluation splits are loaded only for validation/evaluation and are rejected if any train item overlaps by basename, canonical path, or `audio_sha256`.

## Train

```sh
build-cuda/bin/sa3-train \
  --model medium \
  --models-dir models \
  --dataset ../datasets/mnesia-audio-v1 \
  --adapter-type lora \
  --rank 8 \
  --alpha 8 \
  --learning-rate 0.0001 \
  --frames 128 \
  --batch-size 1 \
  --checkpoint-every 1 \
  --out train-runs/mnesia-lora
```

The current graph executes one physical sample at a time. Use `--batch-size 1`.

## Outputs

The output directory contains:

- `adapter-step-*.gguf`
- `adapter-final.gguf`
- `metrics.jsonl`
- `config.snapshot.txt`
- `command.txt`

## Generate With The Adapter

```sh
build-cuda/bin/sa3-generate \
  --model medium \
  --models-dir models \
  --lora train-runs/mnesia-lora/adapter-final.gguf \
  --prompt "$(cat ../datasets/mnesia-audio-v1/evaluation/audio/single-moment.txt)" \
  --frames 128 \
  --steps 8 \
  --seed 42 \
  --out train-runs/mnesia-lora/eval-single-moment.wav
```

## Troubleshooting

- Missing model files: run `python tools/download_models.py --variant medium`.
- Missing captions or audio: inspect the split `filelist.txt` and `metadata.jsonl`; training fails before model loading.
- Split contamination: remove the duplicate item from train or held-out splits.
- FFmpeg decode failure: confirm `ffmpeg` is installed and can decode the MP3.
- CUDA build failure: verify the CUDA Toolkit is installed and rerun `./build.sh cuda`.
