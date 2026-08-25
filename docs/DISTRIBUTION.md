# distribution — gguf naming, hf repos, model cards

how the sa3.cpp model weights are packaged and published so anyone (and us, on a fresh
machine) can `clone → download → build → run`. this is the canonical reference; the download
script and the model cards follow it exactly.

## the models are multi-file families

one sa3 "model" is assembled from five ggufs: three **per-variant** (DiT, SAME autoencoder,
and a tiny conditioner sidecar) and two **shared** (the frozen T5Gemma text encoder + tokenizer,
identical across all three variants, fetched once).

| variant | DiT | autoencoder | conditioner |
|---|---|---|---|
| medium      | `sa3-dit` (1.45B) | SAME-L | `sa3-conditioner` (~0.8MB) |
| small-music | `sa3-dit` (0.46B) | SAME-S | `sa3-conditioner` |
| small-sfx   | `sa3-dit` (0.46B) | SAME-S | `sa3-conditioner` |
| *(shared)*  | — | T5Gemma encoder (0.28B) + tokenizer | — |

**why the conditioner is split out (sidecar):** the T5Gemma encoder is frozen/shared, but the
learned prompt padding embedding + the `seconds_total` NumberConditioner are **trained per
variant**. bundling them into the encoder gguf would silently make the 1.1GB "shared" encoder
per-variant. so they ship as a few-KB sidecar (`general.architecture = sa3-conditioner`) loaded
alongside the encoder — the gguf convention's `Sidecar` pattern (cf. `mmproj`). `sa3-generate`
takes `--cond <gguf>`; if omitted it falls back to reading the conditioner from the `--t5` model
(so legacy bundled encoder ggufs still work). validated: pure-encoder + sidecar reproduces the
old bundled path byte-for-byte.

## naming convention

follows the gguf spec (`ggml/docs/gguf.md`):
`<BaseName>-<SizeLabel>-<Version>-<Encoding>[-<Type>].gguf`, `-`-delimited, minimum
BaseName + SizeLabel + Version. our application:

- **BaseName** — descriptive `model-component`, e.g. `stable-audio-3-medium-dit`.
- **SizeLabel** — param-count class, **on model-like components only** (DiT, text encoder).
  the SAME autoencoder and the tokenizer are convention-exempt (no natural param class — same
  call acestep.cpp makes for its VAE).
- **Version** — `v1.0` (bump on any weight/conversion change).
- **Encoding** — `F32`, `F16` now; `Q8_0` / `Q6_K` / `Q5_K_M` / `Q4_K_M` later.
- **Type** — `vocab` for the tokenizer, `LoRA` for adapters; omitted for normal tensor models.

### filenames

```
# medium  (repo: stable-audio-3-medium-GGUF)
stable-audio-3-medium-dit-1.5B-v1.0-F32.gguf
stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf
stable-audio-3-medium-same-l-v1.0-F32.gguf
stable-audio-3-medium-same-l-v1.0-F16.gguf
stable-audio-3-medium-conditioner-v1.0-F32.gguf       # tiny per-variant sidecar

# small-music  (repo: stable-audio-3-small-music-GGUF)
stable-audio-3-small-music-dit-0.5B-v1.0-{F32,F16}.gguf
stable-audio-3-small-music-same-s-v1.0-{F32,F16}.gguf
stable-audio-3-small-music-conditioner-v1.0-F32.gguf

# small-sfx  (repo: stable-audio-3-small-sfx-GGUF)
stable-audio-3-small-sfx-dit-0.5B-v1.0-{F32,F16}.gguf
stable-audio-3-small-sfx-same-s-v1.0-{F32,F16}.gguf
stable-audio-3-small-sfx-conditioner-v1.0-F32.gguf

# shared text encoder + tokenizer  (repo: t5gemma-b-b-ul2-GGUF)
t5gemma-b-b-ul2-encoder-0.3B-v1.0-{F16,F32,Q8_0}.gguf
t5gemma-b-b-ul2-v1.0-vocab.gguf

# adapters (live with the repo they target, or a loras repo)
<name>-v1.0-F32-LoRA.gguf        # e.g. kev-v1.0-F32-LoRA.gguf

# training-only base DiTs (one dedicated repo per variant)
stable-audio-3-medium-base-dit-1.5B-v1.0-{F32,F16}.gguf
stable-audio-3-small-music-base-dit-0.5B-v1.0-{F32,F16}.gguf
stable-audio-3-small-sfx-base-dit-0.5B-v1.0-{F32,F16}.gguf
```

## metadata (`general.*`) the converters must stamp

the converters set `general.architecture` (`sa3-dit` / `sa3-ae` / `sa3-t5gemma` /
`sa3-conditioner` / `sa3-tokenizer` / `sa3-lora` — our loaders key off these, keep them) and,
via the shared `tools/gguf_meta.py` helper, the convention/catalog fields below (DONE for
dit/ae/t5gemma/conditioner converters):

| key | value |
|---|---|
| `general.basename`   | e.g. `stable-audio-3-medium-dit` |
| `general.size_label` | `1.5B` / `0.5B` / `0.3B` (DiT + encoder only; omit for SAME / conditioner / tokenizer) |
| `general.finetune`   | `medium` / `small-music` / `small-sfx` (omitted on the shared encoder) |
| `general.version`    | `v1.0` |
| `general.license`    | `stabilityai-community` |

Training-base DiTs additionally set `dit.training_base = true` and the standard
`general.base_model.0.{name,organization,version,repo_url}` fields. `version` is the exact pinned
upstream revision, so the source of a standalone GGUF remains recoverable without its model card.

## hf repo layout

de-facto gguf norm (TheBloke / acestep.cpp): **one repo per model, all components + all
quants as files, one card with a "grab one of each" table** — not a repo per quant.

| repo | holds |
|---|---|
| `stable-audio-3-medium-GGUF`       | medium DiT + SAME-L + conditioner, all encodings |
| `stable-audio-3-small-music-GGUF`  | small-music DiT + SAME-S + conditioner |
| `stable-audio-3-small-sfx-GGUF`    | small-sfx DiT + SAME-S + conditioner |
| `t5gemma-b-b-ul2-GGUF` *(shared)*  | encoder + tokenizer |
| `stable-audio-3-<variant>-base-GGUF` | training-only base DiT, F16 + F32 |

grouped under an hf **collection** "Stable Audio 3 (GGUF)". the `models` downloader fetches one
variant repo (DiT + SAME + conditioner) + the shared encoder repo. Passing `--training-base`
also fetches the matching dedicated base-DiT repo used by `sa3-train`.

## quant matrix

published per variant, for **both** the DiT and the SAME: `F32`, `F16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`.
`--encoding` selects the pair (`sa3-generate` resolves DiT and SAME with the same suffix). the
conditioner and tokenizer stay `F32` — those two really are small (793 KiB and 14 MiB).

**the text encoder is published `F16` (default), `F32` and `Q8_0`**, selected by `--t5-encoding`
independently of `--encoding` — the combination worth having on a small device is a quantized DiT
with an F16 encoder. an earlier revision of this section lumped the encoder in with the conditioner
and tokenizer as "small and quality-critical, nothing to gain". it is not small: at F32 it is
**1074 MiB**, the largest single file in the set and bigger than a small-music DiT.

measured on small-music/Metal against an F32 control, swapping *only* the encoder:

| encoder | size | conditioning cosine | generated-audio cosine | small-music train peak |
|---|---:|---:|---:|---:|
| F32 | 1074 MiB | — | — | 2.28 GiB |
| **F16** | **537 MiB** | 1.000000000 | 0.999971 | **1.78 GiB** |
| Q8_0 | 285 MiB | 0.999789 | 0.991496 | 1.52 GiB |
| ~~Q4_K_M~~ | 206 MiB | 0.977908 | **0.797692** | 1.46 GiB |

`Q4_K_M` is **not published** for the encoder, on evidence rather than caution: it is **11.8% slower
than F32** here — the T5 forward is too small to amortize dequant, the opposite of the DiT result
above — while dropping generated-audio cosine to 0.798, which is a different piece of music rather
than a degraded one. `Q8_0` dominates it on every axis.

> ⚠️ **do not validate an encoder with a loss curve.** `Q4_K_M` moved mean training loss by **0.02%**
> over 100 matched steps while producing that 0.798. diffusion loss averages velocity error over
> every latent position, so a shifted prompt embedding still predicts velocity well — just for a
> subtly different prompt. use conditioning cosine (`SA3_DUMP_COND`) and generated audio.

an earlier revision of this section said to keep the SAME at F16/F32 rather than quantize it. that
was a precaution written before the tooling existed, and measurement superseded it: `sa3-quant-check`
reports `below-threshold=0` at `threshold=0.990` for every DiT and SAME at all three quant tiers,
across all three variants.

| | Q8_0 | Q5_K_M | Q4_K_M |
|---|---:|---:|---:|
| medium DiT (F16 2773 MB) | 1483 | 1053 | 962 |
| medium SAME-L (F16 1626 MB) | 865 | 617 | 570 |
| small-music / small-sfx DiT (F16 877 MB) | ~471-481 | ~333-343 | ~302-312 |
| small-* SAME-S (F16 207 MB) | 110 | 79 | 72 |
| medium base DiT (F16 2773 MB) | — | — | 962 |
| small-* base DiT (F16 877 MB) | — | — | 302 |

**quantization buys footprint everywhere and speed only on some backends.** CUDA and Vulkan gain
~33% end to end; Metal is flat (Q8_0 1.7% faster, Q4_K_M 2.1% *slower*) because the load-time saving
and the added per-step compute cancel. say so on the cards — a Mac user picking Q4_K_M for speed
would be picking it for the wrong reason. see `docs/METAL.md` §9.

**training bases are published `F32`, `F16` and `Q4_K_M`.** training on a quantized base works on
every backend as of ggml PR #4, which taught `out_prod` to take a quantized `src0` — that backward,
`out_prod(W, transpose(grad))`, was the only op in the way, since the functional adapter path
otherwise keeps the frozen base as a plain `mul_mat` argument. it is *faster* than F16 on every
backend measured, not a tradeoff, and a 2000-step adapter trained on a Q4_K_M base is audibly
indistinguishable from an F16-trained control. see `docs/TRAINING.md`.

only Q4_K_M is published as a base: it is the tier where the 2.9x footprint win matters, and Q5_K_M
/ Q8_0 bases would be three more large files for a saving nobody asked for. requesting one of those
tiers with `--training-base` resolves the base DiT — and only the base DiT — to F16, which
`--dry-run` prints.

## license

stable-audio-3 is under the **Stability AI Community License**. user is in the Stability org and
the published GGUF repositories mirror the upstream license/gating pattern. Each training-base
repository includes the pinned upstream `LICENSE.md`, the required Stability attribution in
`NOTICE`, its model card, and `SHA256SUMS`.

## training-base release staging

`tools/stage_training_base_repos.py` is optional maintainer tooling, not a runtime downloader or an
automatic uploader. Given the converted F16/F32 GGUF directory, it prepares all three repository
trees, fetches and checksum-verifies the pinned upstream license, refuses to replace mismatched
files, hard-links large GGUFs when possible, and writes release checksums. Review the staged trees
before uploading them with the standard Hugging Face tooling.

```sh
python tools/stage_training_base_repos.py --gguf-dir /path/to/converted --out /path/to/staging
```

## download contract

`models.{sh,cmd}` (or a python `hf download` wrapper — better on windows) pulls a default set
into `models/`: one DiT + one SAME (chosen encoding) + the conditioner for the requested variant,
plus the shared encoder + tokenizer. flags for variant (`--medium`/`--small-music`/`--small-sfx`)
and encoding (`--f16`/`--f32`). mirrors acestep's `models.sh`. `sa3-generate` is then run with
`--t5 <encoder> --cond <conditioner> --dit <dit> --same <same> --tok <tokenizer>`.
