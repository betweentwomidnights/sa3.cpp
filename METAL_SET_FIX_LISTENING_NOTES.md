# Listening notes: what the Metal `SET` fix changed about trained adapters

Companion to `METAL_SCOPE_GNORM_HANDOFF.md`, which covers the bug itself. This is about its
*effect on adapters*, which turned out to be more interesting than the bug.

Temporary; fold the durable parts into `docs/TRAINING.md` once we know what to do about it.

## The pair

Two 2000-step DoRA-rows r16 runs on the same 10-track Ratatat set, **identical commands**, only the
ggml `SET` fix differing:

| | run | note |
|---|---|---|
| pre-fix | `metal-medium-ratatat-512-simdgroup-2000` | trained through the corrupted forward |
| post-fix | `metal-full-FIXED-512-2000` | correct memory tokens |

Both exported to `loras/ratatat-{prefix-simdgroup,postfix-setfix}-2000.safetensors` for analysis
elsewhere. Same seed, same latents, same `--prompt-config` (so the same ~60/40 real/filename caption
mix), same 512-frame crops, full 228-target scope.

**Inference never touched the bug** — `dit_head` uses `ggml_concat` when `!autodiff_safe`
(`dit.h:301`) and only the training path uses the two `ggml_set` calls. So this was a
train/inference mismatch, not corrupted generation.

## Training curves: the difference is entirely in the first 500 steps

| | pre-fix | post-fix |
|---|---|---|
| steps 1-500 | loss 0.7137, gnorm **0.4340** | loss 0.6505, gnorm **0.1429** |
| steps 501-1000 | 0.6268, 0.1613 | 0.6251, 0.1628 |
| steps 1001-1500 | 0.6319, 0.1651 | 0.6306, 0.1650 |
| steps 1501-2000 | 0.6099, 0.1658 | 0.6085, 0.1730 |
| first-10 -> last-10 | 0.9673 -> **0.5892** | 0.6730 -> **0.5881** |
| max loss | 2.2723 | 1.1536 |
| clipped (`grad_clip` 1.0) | **61/2000** | **8/2000** |

They converge to the same loss (0.5892 vs 0.5881) and behave identically from step 500 on. The
adapter was not "compensating for a wrong model" for 2000 steps — it was pushed into a different
basin during the first quarter, then trained normally inside it.

## What it sounds like

kev's read, which is more precise than any metric here: the pre-fix adapter **leaves out Ratatat's
guitar work** and produces something like a Ratatat *backing track* — lots of space, no transient
squeaks in the percussion fills ("the banshee"). The post-fix adapter is **fuller and much more
recognisably Ratatat**, but brings the banshee with it.

So the bug was not adding character so much as **suppressing a class of content**, and what it
suppressed included both the artifacts and some of the musical complexity.

## High-frequency measurements, and where they mislead

Crude proxy: first-difference energy / total energy, left channel. Sensitive to transient density
and percussive attack as much as to hiss, so treat as directional only.

25s renders, post-fix is higher in **every** pair:

| prompt | pre-fix | post-fix | ratio |
|---|---:|---:|---:|
| Chiptune 111 B minor | 0.0153 | 0.0248 | 1.62x |
| Chiptune 92 D minor | 0.0150 | 0.0210 | 1.40x |
| Indie rock 91 A major | 0.0651 | 0.1748 | 2.69x |
| folk funk 103 D minor | 0.1660 | 0.2034 | 1.23x |
| Synthwave s99 | 0.0434 | 0.0711 | 1.64x |
| Synthwave s7 | 0.0282 | 0.0580 | 2.05x |

**But it is not monotonic in adapter strength**, which kills the obvious "post-fix just pushes
harder" reading:

```
pre-fix   strength 1.0 : 0.0153   <- target
post-fix  strength 1.0 : 0.0248
post-fix  strength 0.8 : 0.0323
post-fix  strength 0.6 : 0.0566   <- weaker adapter, MORE high end
```

Weakening the corrected adapter makes the top end worse, which suggests the squeak is substantially
the **base model's own** behaviour and the pre-fix adapter was damping it. You cannot dial the old
sound back with `--lora-strength`.

Blending overshoots both parents at 25s (`pre-fix 1.0 + post-fix 0.5` = 0.0320 against 0.0153 and
0.0248), and order barely matters despite `dora-rows` being non-commutative (0.0320 vs 0.0318
reversed).

**Over 2 minutes the relationship inverts partway through** — per 30s quarter:

| | 0-30s | 30-60s | 60-90s | 90-120s |
|---|---:|---:|---:|---:|
| pre-fix | 0.3352 | 0.0562 | 0.0564 | 0.0772 |
| post-fix | 0.1033 | 0.1103 | 0.0898 | 0.1107 |
| blend | 0.1666 | 0.1333 | 0.1201 | 0.1126 |

Pre-fix front-loads then settles far below post-fix for the remaining 90 seconds; post-fix is flat
throughout. The 25s clips only ever sampled the opening region, where the metric and the ear
disagree — kev hears pre-fix as cleaner across the whole piece, and the settled quarters agree with
him even though the first one does not. **Do not draw conclusions about this from short renders.**

Long generation is fine regardless of crop length: a 23.8s-crop adapter generates coherent
multi-minute output. Crop length is not a constraint on render length here.

## Where this might go

The interesting question is whether the suppression can be reproduced deliberately, since
corrupting memory tokens is not a knob anyone should ship. Two things point at how:

- the divergence is confined to the **first ~500 steps**, so a controlled version is plausibly a
  *schedule* (perturb early, anneal off) rather than a constant;
- the perturbation was specific — 512 of 1536 channels on 64 memory-token rows, training-only —
  which is a small, well-defined intervention rather than generic noise.

kev's note is worth keeping attached: "more chiptune, less guitar" is probably only a *gift* on a
dataset like Ratatat, and a liability elsewhere. Any deliberate version should be an off-by-default
flag, not a change to the recipe.

Both adapters are exported to safetensors for weight-space comparison against the decoder-LoRA work,
which targets the same banshee artifacts from the other end.
