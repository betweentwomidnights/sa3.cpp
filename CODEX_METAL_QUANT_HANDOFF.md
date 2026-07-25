# Codex handoff: validate Q4_K_M / Q8_0 quantization on Metal

Gate for merging `feature/q4km-quantization`. Windows validation (CUDA, Vulkan, CPU) is
done; Metal is the one backend nobody has run this on. Do not merge the PR until the
gates below pass or a failure is understood.

## Objective and decision boundary

Confirm that quantized DiT/SAME weights load and generate correctly on Metal, and record
M4 numbers alongside the existing CUDA/Vulkan/CPU table.

In scope: building the branch, quantizing locally, the gates below, appending results to
`docs/METAL.md`.

Out of scope, do not do these without asking:
- publishing anything to Hugging Face
- merging either PR
- changing the ggml submodule pin
- touching pillo's web app / server changes (a later, separate stage)

## Exact starting state

- sa3.cpp branch `feature/q4km-quantization`, 4 commits on top of `main` (`c81a201`):
  - `0bbcd0b` quantization tooling + four bug fixes
  - `2c6f92c` `--mix` table (`q4_k_m` / `q5_k_m` / `q5_k` / `q8_0`)
  - `f11820c` ggml submodule pin bump
  - `ae2e556` canonical `Q4_K_M` / `Q5_K_M` naming
- ggml submodule pinned at `e9c70fd6` on branch `feature/sa3-q4km-getrows`
  (betweentwomidnights/ggml PR #2, base `feature/sa3-training-vulkan-v0.16.0`).
- `.gitmodules` still points at `betweentwomidnights/ggml`. If you ever see
  `pillopaus-project/ggml` in there, something pulled in the wrong branch — revert it.

## Clone or update on the MacBook

```sh
cd ~/dev/sa3.cpp
git fetch origin
git checkout feature/q4km-quantization
git submodule update --init --recursive
git -C ggml log --oneline -2      # expect e9c70fd6 then 74e73f67
./build.sh metal
source env.sh
```

## What the ggml pin does and does not affect

The submodule bump adds **CUDA** k-quant `get_rows`. Metal already has
`kernel_get_rows_q2_K` through `q6_K`, so the pin changes nothing for Metal —
it should be inert here. That is itself worth confirming: an unchanged Metal
inference baseline is a pass, not a null result.

`te.embed.weight` (`src/t5gemma.h:47`) is the only `ggml_get_rows` input in the whole
pipeline, and it lives in the T5 encoder gguf. Quantized DiT/SAME never reach `get_rows`.

## Gate 1 — inference is unregressed at F16

Before touching quantization, confirm the branch did not disturb the existing path.
`--encoding f16` must still resolve and generate, and should match the M4 baseline in
`docs/METAL.md` section 6.

```sh
sa3-generate --model medium --encoding f16 --prompt "cinematic orchestral build with strings and timpani" \
  --duration 30 --seed 99 --out metal_f16.wav
```

## Gate 2 — quantize locally

The medium DiT is 2.7 GiB at F16. `sa3-quantize` reads the whole input into RAM, so
expect a ~4 GiB peak; fine on the M4 but do not run it alongside a training job.

```sh
sa3-quantize --in models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
             --out models/stable-audio-3-medium-dit-1.5B-v1.0-Q4_K_M.gguf --mix q4_k_m
sa3-quantize --in models/stable-audio-3-medium-same-l-v1.0-F16.gguf \
             --out models/stable-audio-3-medium-same-l-v1.0-Q4_K_M.gguf --mix q4_k_m
sa3-quantize --in models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
             --out models/stable-audio-3-medium-dit-1.5B-v1.0-Q8_0.gguf --mix q8_0
sa3-quantize --in models/stable-audio-3-medium-same-l-v1.0-F16.gguf \
             --out models/stable-audio-3-medium-same-l-v1.0-Q8_0.gguf --mix q8_0
```

Expected sizes (should match Windows byte-for-byte — quantization is host-side and
backend-independent, so a mismatch means something is wrong):

| file | MiB |
| --- | --- |
| medium DiT Q4_K_M | 962 |
| medium SAME-L Q4_K_M | 570 |
| medium DiT Q8_0 | 1483 |
| medium SAME-L Q8_0 | 864 |

## Gate 3 — per-tensor validation

```sh
sa3-quant-check --ref models/stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf \
                --quant models/stable-audio-3-medium-dit-1.5B-v1.0-Q4_K_M.gguf
```

Expect `compared=204  below-threshold=0  threshold=0.990`. Same for SAME-L with
`compared=100`.

## Gate 4 — generate and listen

```sh
for enc in f16 q8_0 q4_k_m; do
  sa3-generate --model medium --encoding $enc \
    --prompt "cinematic orchestral build with strings and timpani" \
    --duration 30 --seed 99 --out metal_medium_$enc.wav
done
```

Must be music, not noise, and recognisably the same *kind* of piece across the three.
Do not expect waveform equality: quantization perturbs the diffusion trajectory, so
Q4_K_M diverges from F16 at roughly 0.55 waveform cosine while sounding fine. Judge by
ear, not by cosine. Noise, silence, or obvious glitching is a real failure.

## Gate 5 — speed, warm

Discard the first run of each encoding (page cache + shader warm-up; on Vulkan the first
Q4_K_M run came in at 11.2 s against a 4.5 s steady state). Take best of three.

Reference numbers to compare against, medium / 30 s / 8 steps, best of 3 warm:

| backend | F16 | Q8_0 | Q5_K_M | Q4_K_M |
| --- | --- | --- | --- | --- |
| CUDA (RTX 5070 laptop) | 5.54 | 4.15 | 3.71 | 3.65 |
| Vulkan (same GPU) | 6.61 | 5.23 | 4.62 | 4.50 |
| CPU (default threads) | 122.70 | 79.68 | 84.32 | 72.48 |
| Metal (M4) | ? | ? | — | ? |

On both GPU backends Q4_K_M ran ~33% faster than F16, and per-stage profiling
(`SA3_PROFILE=1`) showed the gain split between real compute and reduced load time.
The M4 is the interesting case: unified memory means the load-time component may behave
unlike either a discrete GPU or CPU, so capture `SA3_PROFILE=1` output for F16 and
Q4_K_M and report `dit_compute` / `dec_compute` separately from wall time.

`q5_k_m` is optional on Metal. It is speed-neutral against `q4_k_m` on GPU, 9% larger,
and loses to `q8_0` on CPU in both speed and fidelity, so it is unlikely to ship.

## Optional — the quantized encoder

Only if the above passes and there is appetite. This is the one place the k-quant
`get_rows` path is reachable, and the biggest single size win left (encoder is 1074 MiB
at F32).

```sh
sa3-quantize --in models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf --out t5_enc_Q8_0.gguf   --mix q8_0
sa3-quantize --in models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf --out t5_enc_Q4_K_M.gguf --mix q4_k_m
sa3-quant-eval --model models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf t5_enc_Q8_0.gguf   --seed 7
sa3-quant-eval --model models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf t5_enc_Q4_K_M.gguf --seed 7
```

Cosine vs F32 measured on the other three backends:

| encoder | CUDA | CPU | Vulkan | Metal |
| --- | --- | --- | --- | --- |
| Q8_0 (embed Q8_0) | 0.999301 | 0.999242 | 0.999641 | ? |
| Q4_K_M (embed Q6_K) | 0.979378 | 0.978046 | 0.980380 | ? |

Metal should land in the same band. A number far below these means Metal's
`kernel_get_rows_q6_K` disagrees with the CPU reference, which would be a real find.

Note the encoder is **not** selectable via `--encoding` — `ModelPaths::resolve` gives the
`t5` slot a fixed `.gguf` suffix. Pass it explicitly with `--t5 <path>`. Do not leave two
`t5gemma-b-b-ul2-encoder-*.gguf` files in `models/`; the resolver globs on prefix and would
pick one arbitrarily (`resolve_one` flags the ambiguity but callers ignore the flag).

## After the gates

Append a dated section to `docs/METAL.md` in the style of sections 6 and 7 with the
numbers above filled in, and report back before merging. If Q4_K_M sounds good on Metal
too, the release shortlist is Q4_K_M plus Q8_0; Q5_K_M only earns a slot if it clearly
beats Q4_K_M by ear.

## Collaboration context

The quantization work is ported from @pillopaus-project's `q4km-webapp-fixes` branch,
stripped of the web app so `main` stays primitives-only. His authorship is preserved on
the ggml commit. Both PRs are deliberately scoped to just the Q4_K_M work so he can point
his `.gitmodules` back at our fork; his other changes (web app, config/env rework, trainer
UI) are later separate stages. Four bugs were fixed on top of his port — a 32-bit `ftell`
that broke any gguf ≥ 2 GiB, a `% 32` block-size guard that should have been `% 256`, a
C++20 designated initializer, and a `q5_K` `get_rows` index mapping that was wrong for
189 of every 256 elements. Worth telling him about that last one regardless of whether we
ever ship Q5.
