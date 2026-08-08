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

### CUDA answer: no gap at all

Run on the PC (RTX 5070 Laptop, CUDA), `SA3_GRAD_DEBUG=1`, 512 frames, seed 42, 3 steps, same
latents. Step 2:

| run | H | after `T` | after `B_l` | `\|gctx_sum\|` | `\|ggcond_sum\|` | gnorm |
|---|---|---:|---:|---:|---:|---:|
| blocks only (72) | skipped | 0.000000 | 0.047340 | **0.008253** | **0.139518** | 0.0473 |
| + `proj_in` (73) | ran | 0.000000 | 0.047408 | **0.008254** | **0.139784** | 0.0741 |
| + `proj_out` (73) | skipped | 0.080837 | 0.093617 | 0.008219 | 0.139446 | 0.0936 |

`|gctx_sum|` agrees to four significant figures across scopes and `|ggcond_sum|` to 0.2%, against
Metal's 64x and 5x. **So the activation-gradient accumulation is where Metal diverges.** Block
gradients also stay put on CUDA (0.047340 vs 0.047408), so nothing is inflating there either.

Step 1 on CUDA is likewise scope-independent (`|gctx_sum|` 0.000163 / 0.000162 / 0.000163), which
matches your Metal step 1. The difference is entirely in what step 2 does with it.

**Why that is hard to explain as a legitimate parameter effect.** Step 1 runs at
`lr = 5.000e-07`, so whatever the optimizer writes moves the adapter by ~1e-7 relative. The only
parameter that differs between blocks-only and +`proj_in` after step 1 is `proj_in`'s own adapter
— the 72 block adapters get identical gradients either way, since `H` contributes only to head
params. A ~1e-7 change in one adapter cannot move step 2's activation gradients 64x. On CUDA it
moves them by 1 part in 8000, which is the scale you would expect.

**Two more things CUDA confirms, both matching your Metal findings:**

- Giving `T` a real parameter changes nothing about the accumulators here either
  (+`proj_out`: `|gctx_sum|` 0.008219 vs blocks-only 0.008253). Your "`T` with zero params is not
  the hazard" conclusion holds on both machines.
- The head/tail targets really are gradient-heavy. `proj_out` **alone** contributes 0.080837
  through `T`, against 0.047340 for all 72 block targets combined; adding `proj_in` roughly
  doubles the total. That is the same effect behind CUDA's full-vs-core ordering (0.1529 vs
  0.0548), and it is what makes the Metal direction so strange: excluding the gradient-heaviest
  targets should *lower* the norm, as it does here, not quadruple it.

Losses are scope-independent on CUDA at every step (0.6534 / 0.8558 / 0.3111 across all three
runs, varying only in the 4th decimal), so none of this perturbs the trajectory here.

## Open question nobody has answered

The **direction**. Widening the scope raises the norm on CUDA and lowers it on Metal, with identical
host-side optimizer code. Whatever the mechanism is, it has to explain a sign flip, not just a
magnitude.

## Suggested next probes on Metal, given the CUDA result

The accumulation diverging while step 1 is clean narrows it to state carried across the step
boundary. Cheapest discriminators, roughly in order:

1. **The frozen-adapter invariant — run this first.** Use **`--adapter-type lora`** (not
   `dora-rows`) with `--learning-rate 1e-35`. Plain lora has `B = 0` at init, so an adapted target
   adds exactly `+0` to the forward, and a frozen adapter keeps it that way. Adding `proj_in` to
   the scope therefore *cannot* change the forward at all, and `H` contributes only head
   gradients, which are never applied. So:

   > blocks-only and +`proj_in` must produce **bit-identical** `loss`, `|gctx_sum|` and
   > `|ggcond_sum|` at every step. Only `gnorm` may differ — the +`proj_in` run has one extra
   > adapted target contributing a term to the global L2.

   Verified on CUDA, 512 frames, seed 42, 3 steps:

   | | loss (steps 1-3) | `\|gctx_sum\|` | `\|ggcond_sum\|` | gnorm |
   |---|---|---|---|---|
   | blocks only | 0.653466 / 0.855654 / 0.311041 | 0.000163 / 0.008253 / 0.000129 | 0.045958 / 0.139590 / 0.046064 | 0.0161 |
   | + `proj_in` | **identical** | **identical** | **identical** | 0.0321 |

   If Metal breaks that equality, the fault is upstream of the optimizer entirely and `H`'s mere
   presence is changing the forward or the accumulation. If Metal holds it, the divergence needs a
   real parameter update to appear, which points back at what gets written between steps.

   Note this is exact only for plain `lora`. With `dora-rows` the same probe shows ~5e-5 loss
   differences even frozen, because the row scale is `magnitude/||W + s·A@B||` — numerically ~1
   but not exactly 1 — so adapting a target perturbs the forward at init. That is expected and not
   a bug; it just makes `dora-rows` useless as an exact invariant.
2. **Read back the adapter after step 1.** `ggml_backend_tensor_get` one block adapter's `lora_B`
   at the top of step 2 and compare against the host `TrainLoraState`. The 72 block adapters must
   be identical between the two scopes at that point, since `H` contributes only head gradients.
   If they differ on Metal, the divergence is in what got written, not what got computed.
3. **`Gctx_in` / `Ggcond_in` when `H` is skipped.** They are uploaded every step and then consumed
   by nobody. Worth confirming that an unconsumed persistent-buffer write is not interacting with
   the next step's uploads — e.g. by skipping the upload entirely when `hgraph` is null and seeing
   whether the blow-up survives.

(1) is the one I would run first: it separates "forward differs" from "gradients differ" in a
single pair of runs, and on CUDA it is guaranteed to come back identical, so a Metal difference is
immediately meaningful.

## Practical impact while this is open

Narrow. Full scope (the default) is unaffected on Metal and produces good adapters. Only scoped
training on Apple is affected, and the PC — where production training happens — is clean at both
scopes. Workaround if anyone needs scoped training on a Mac: include any head target (`proj_in`
alone is enough) so `H` is built.
