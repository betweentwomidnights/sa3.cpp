# Handoff: Metal scoped-training gradient blow-up

Temporary working doc for the M4 <-> PC investigation. Delete once the cause is understood and the
durable parts land in `docs/METAL.md` / `docs/TRAINING.md`.

Branch `experiment/metal-lora-scope-gnorm`, off `79f57b8` (post-#28 main). Nothing here blocks
anything shipped — #28 merged clean and this reproduces on `main`.

## The problem in one table

300 steps, medium, 512 frames, seed 42, dora-rows r16, same `ratatat-3-1784005242` latents.
Each machine compared against itself, so dataset/caption/file-order/RNG differences cancel.

| scope | M4 Metal gnorm | PC CUDA gnorm | ratio |
|---|---:|---:|---:|
| `full` (228) | 0.6192 | 0.1529 | 4.0x |
| `core` (168) | 2.6253 | 0.0548 | **47.9x** |

Two effects stacked: a ~4x baseline elevation on Apple even at full scope, plus something far larger
that appears only when the scope is narrowed.

**The ordering inverts between machines**, which is the sharper fact:

- PC: `core` 0.0548 -> `full` 0.1529 (widening *raises* the norm, as an L2 over more terms should)
- M4: `core` 2.6253 -> `full` 0.6192 (narrowing *quadruples* it)

Loss agrees: on the PC `core` and `full` land together (0.6611 / 0.6599); on Metal they are 1.2312
vs 0.7629. Neither PC run clips at all (0/300 both scopes); Metal clips ~9% at `core` with
`grad_clip` 1.0.

## What the instrumentation shows

`SA3_GRAD_DEBUG=1` (landed on this branch) stages the accumulation. Metal, 512 frames, step 2:

| stage | `core` | `full` |
|---|---:|---:|
| after `T` | 0.000000 (0 tail params) | 0.826411 (2 tail params) |
| after `B_l` | **5.636396** | 0.941640 |
| `\|gctx_sum\|` | **0.404941** | 0.006271 |
| `\|ggcond_sum\|` | **9.243105** | 1.873588 |
| after `H` | 5.636396 (skipped) | 2.489338 (ran) |

Inflation enters through the **per-block graphs** (~12x), and the host-side activation-gradient
accumulators are inflated with it (`gctx_sum` 64x, `ggcond_sum` 5x).

**Step 1 is identical between scopes** — `gctx_sum` 0.000457 in both, `ggcond_sum` 0.321407 vs
0.321394. Nothing is wrong at init; step 1's *update* creates the divergence.

## It is specifically `H`, not "a segment with no trainable params"

`core` has zero tail params as well as zero head params, so `T` also runs with nothing trainable —
the same hazard the `H` skip-guard exists for. Split by adding exactly one target:

| run | targets | `T` contributes | after `B_l` | `\|gctx_sum\|` | `\|ggcond_sum\|` | gnorm |
|---|---|---:|---:|---:|---:|---:|
| blocks only | 72 (0 elsewhere) | 0.000000 | 4.982893 | 0.383294 | 8.885983 | **4.98** |
| + `proj_out` (tail) | 73 (1 elsewhere) | 0.615436 | 5.007677 | 0.386406 | 8.841840 | **5.01** |
| + `proj_in` (head) | 73 (1 elsewhere) | 0.000000 | 0.437671 | 0.006269 | 1.872781 | **0.59** |

Giving `T` a real parameter changes nothing. Only `H` executing collapses it. Note `H` runs *after*
the block loop, so it cannot affect the same step's block gradients — the chain must be
`H` skipped -> step 1's update differs -> step 2's activation gradients explode.

## Ruled out, each with a measurement

| candidate | how it died |
|---|---|
| Accelerate / BLAS | `build-metal` (links `libggml-blas.0.dylib`) vs `build-metal-noblas` (does not): **bit-identical to 4 decimals**, both scopes, all mask types. Linkage verified with `otool -L`, not assumed. |
| Stale ggml libs | both builds load `0.16.0`; the `0.15.3` dylibs in the build dir are not linked |
| Nondeterminism | identical command twice -> identical output |
| Adapter magnitude | lr swept 1e-15 -> 1e-35, loss perfectly flat, scope-dependence unchanged |
| Data pipeline | `id` / `t` / `mask` / `n_gen` / `n_ctx` / `lr` identical across scopes, all steps |
| `T` with zero trainable params | table above |
| My own instrumentation | stashed, both builds rebuilt clean -> identical results |

## Claims of mine that are dead — do not build on them

- **"Metal-specific"** — wrong as originally framed; the CPU backend on this Mac reproduced it too,
  which is what pointed at BLAS. BLAS is now also out.
- **"`core` is inherently weaker at style transfer"** — speculation, contradicted by kev's PC runs
  where 168 vs 228 costs almost nothing.
- **"`T` with zero params"** — tested and dead (above).
- **Per-node checksum dumps** — methodologically invalid. gallocr reuses scratch buffers, so reading
  node outputs after the graph completes returns whatever last occupied that memory. Any conclusion
  from that pass, including a dramatic "node 0 diverges", is an artefact.
- **Cross-machine step-by-step loss comparison** — meaningless. Dataset directories differ, so file
  enumeration and the RNG stream diverge; seed 42 does not give the same samples.

## The measurement that would split the remaining hypotheses

On CUDA, same probe, `--lora-include self.qkv,self.out,ff.proj` (72 targets, H skipped), 512 frames,
seed 42:

```
SA3_GRAD_DEBUG=1 sa3-train --model medium --dataset <ds> --latents-dir <lat> \
  --frames 512 --steps 3 --checkpoint-every 0 --seed 42 \
  --lora-include self.qkv,self.out,ff.proj --out probe-blocksonly
```

and the same with `,proj_in` appended. The numbers that matter are **`|gctx_sum|` and
`|ggcond_sum|` at step 2**.

| CUDA result | reading |
|---|---|
| same ~60x gap in `gctx_sum` as Metal, but block gradients stay small | the accumulation is a red herring; the fault is in how step 1's update is applied |
| no gap at all | the activation-gradient accumulation itself diverges on Metal — dig there |

## Open question nobody has answered

The **direction**. Widening the scope raises the norm on CUDA and lowers it on Metal, with identical
host-side optimizer code. Whatever the mechanism is, it has to explain a sign flip, not just a
magnitude.

## Practical impact while this is open

Narrow. Full scope (the default) is unaffected on Metal and produces good adapters. Only scoped
training on Apple is affected, and the PC — where production training happens — is clean at both
scopes. Workaround if anyone needs scoped training on a Mac: include any head target (`proj_in`
alone is enough) so `H` is built.
