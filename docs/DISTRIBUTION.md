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
t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf
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

## quant matrix (initial)

ship **F32 + F16** first (f16 = `tools/quantize_gguf.py`, the production path; f32 for cpu
validation). integer quants (Q8_0/Q6_K/Q5_K_M/Q4_K_M) are a later pass — note in the card which
exist. the SAME autoencoder is quality-critical and small, so keep it F16/F32 (don't aggressively
quantize), mirroring acestep keeping its VAE at BF16.

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
