# models

the GGUF models live here (the default `SA3_MODELS_DIR`). one variant needs five files — its DiT, SAME
autoencoder, and conditioner, plus the shared T5Gemma encoder + tokenizer:

| File | Role |
|------|------|
| `stable-audio-3-<variant>-dit-<size>-v1.0-<F16\|F32>.gguf` | DiT — the diffusion transformer |
| `stable-audio-3-<variant>-<same-l\|same-s>-v1.0-<F16\|F32>.gguf` | SAME autoencoder — audio ↔ latent |
| `stable-audio-3-<variant>-conditioner-v1.0-F32.gguf` | per-variant conditioner (prompt + seconds embeddings) |
| `t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf` | shared T5Gemma text encoder |
| `t5gemma-b-b-ul2-v1.0-vocab.gguf` | shared tokenizer |

`<variant>` is `medium` (DiT 1.5B, SAME-L) or `small-music` / `small-sfx` (DiT 0.5B, SAME-S). `F16` is the
production path; `F32` is for CPU / bit-exact validation. The conditioner, encoder, and tokenizer are small and
quality-critical, so they're always F32.

## get them (no python)

```bash
./models.sh                          # medium f16   (Windows: models.cmd)
./models.sh --variant small-music    # or small-sfx; --encoding f32; --out DIR
./models.sh --variant medium --training-base  # inference set + LoRA training base
```

## faster official downloader

If plain `curl` is slow on your connection, use the Hugging Face SDK downloader. It keeps the
same filenames in `models/`, but uses Hugging Face's cache/local-dir metadata, concurrent file
fetching, and recent `hf_xet` support:

```bash
python -m pip install -U "huggingface_hub"
python tools/download_models.py --variant small-music --encoding f16
```

Set `HF_XET_HIGH_PERFORMANCE=1` before running it if you want Hugging Face's high-throughput Xet mode:

```powershell
$env:HF_XET_HIGH_PERFORMANCE = "1"
python tools\download_models.py --variant small-music --encoding f16
```

```cmd
set HF_XET_HIGH_PERFORMANCE=1
python tools\download_models.py --variant small-music --encoding f16
```

```bash
HF_XET_HIGH_PERFORMANCE=1 python tools/download_models.py --variant small-music --encoding f16
```

or download by hand from huggingface:

- https://huggingface.co/thepatch/stable-audio-3-medium-GGUF
- https://huggingface.co/thepatch/stable-audio-3-small-music-GGUF
- https://huggingface.co/thepatch/stable-audio-3-small-sfx-GGUF
- https://huggingface.co/thepatch/t5gemma-b-b-ul2-GGUF  (shared encoder + tokenizer)

## troubleshooting

If generation fails with `[gguf] short read ...`, the named GGUF is incomplete, usually from an interrupted
download. Rerun `models.sh` / `models.cmd`; the curl downloader resumes existing partial files. If it still
fails, delete the named `.gguf` and run the downloader again.

## LoRA training bases

Adapters are trained on the matching base checkpoint, then applied to the ARC/post-trained model for
inference:

| `sa3-train --model` | Training checkpoint | Inference checkpoint |
|---|---|---|
| `medium` | `stabilityai/stable-audio-3-medium-base` | `stabilityai/stable-audio-3-medium` |
| `small-music` | `stabilityai/stable-audio-3-small-music-base` | `stabilityai/stable-audio-3-small-music` |
| `small-sfx` | `stabilityai/stable-audio-3-small-sfx-base` | `stabilityai/stable-audio-3-small-sfx` |

`sa3-train` therefore requires `stable-audio-3-<variant>-base-dit-*-F16.gguf` (or F32 according to
`--encoding`) and deliberately refuses to fall back to the inference DiT. Before loading weights it
also requires `dit.training_base=true` and a matching `general.finetune` value in the GGUF metadata;
renaming an inference model cannot bypass the check. Convert the base checkpoint with
`tools/convert_dit.py --training-base`, quantize it with `tools/quantize_gguf.py`, or pass its marked
GGUF explicitly with `--dit`. The tokenizer, T5 encoder, conditioner, and SAME autoencoder remain the
normal inference files.

Download the complete inference + training set with:

```sh
python tools/download_models.py --variant medium --encoding f16 --training-base
```

`sa3-generate --model <variant>` resolves this set from the naming convention above. lora adapters resolve the
same way (`lora-<name>-*.gguf`) — see [`../loras`](../loras).
