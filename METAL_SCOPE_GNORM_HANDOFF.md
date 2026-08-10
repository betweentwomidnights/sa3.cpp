# Handoff: Metal scoped-training gradient blow-up

Temporary working doc for the M4 <-> PC investigation. Delete once the cause is understood and the
durable parts land in `docs/METAL.md` / `docs/TRAINING.md`.

Branch `experiment/metal-lora-scope-gnorm`, off `79f57b8` (post-#28 main). Nothing here blocks
anything shipped — #28 merged clean and this reproduces on `main`.

> ## STATUS: ROOT-CAUSED. One question left, and it is a code read.
>
> **Cause.** `fgraph` node 25, `GGML_OP_SET` (`ne=[1536,576]`, `dst != src0`), does not write its
> full destination. It is the memory-token concat — 576 = 64 memory tokens + 512 frames — and the
> 64 rows it does not write are supposed to come from `src0`. `dst` is a recycled buffer, so those
> rows inherit whatever was there. Proven: poisoning F's scratch with `0xDEADBEEF` drives `xb[0]`
> from 1.21e6 to 2.05e23, a 13-probe bisect isolates node 25, and zeroing F's scratch
> (`SA3_ZERO_F_SCRATCH=1`) makes the broken configuration reproduce the correct trajectory exactly.
>
> **`H`, `--lora-scope core` and Metal were all correlates, never causes.** They change gallocr's
> plan, which changes which recycled buffer node 25 gets, which changes whether the stale rows
> matter. Sections below written before this was known — notably "It is specifically `H`" — are
> **superseded**; they are kept for the evidence, not the conclusions.
>
> **UPDATE:** the kernel read came back — Metal's `SET` implements the copy and passes 12/12 unit
> tests, so the general "Metal omits the copy" hypothesis is **dead**. But poisoning node 25 alone
> still leaves `0xDEADBEEF` in `xb[0]` rows 0..63. Next step is now a **minimal standalone `SET`
> reproducer at `ne=[1536,576]`** — see the M4 answer section. Superseded instruction follows:
>
> **~~NEXT STEP (PC): read ggml's Metal `SET` kernel.~~** CPU's `ggml_compute_forward_set` copies
> `src0` into `dst` when not inplace. Does `ggml-metal`'s? And is that copy contractually required
> of `ggml_set()` non-inplace? If Metal omits it, this is an upstream ggml bug affecting any
> non-inplace `SET` onto a recycled buffer, well beyond this graph.
>
> This is a **read, not a run** — it does not reproduce on the PC, so there is nothing to bisect
> there. Non-reproduction is itself consistent with CUDA's `SET` doing the copy correctly, which
> would confirm the diagnosis from the other side.
>
> **Do not gate `--lora-scope core` as a fix.** The trigger is an allocation plan, not the scope.
> Any scope or graph shape can land on a plan that hands node 25 a dirty buffer. If a stopgap is
> needed before the kernel is fixed, `SA3_ZERO_F_SCRATCH` is the sound one.


## PC answer: confirmed non-reproducing on CUDA — but the SET read does not support the diagnosis

### 1. CUDA is immune, tested directly rather than by absence

Ran your own probes on CUDA (plain `lora`, `lr=1e-35`, `--random-crop false`, 512 frames, seed 42,
blocks-only so `H` is skipped — the configuration that breaks on Metal):

| run | loss s1 / s2 / s3 | gnorm |
|---|---|---|
| base | 0.434523 / 0.602845 / 0.404239 | 0.0137 / 0.0251 / 0.0594 |
| **`SA3_POISON_F` (4438 of 4637 nodes)** | **identical** | **identical** |
| **`SA3_ZERO_F_SCRATCH`** | **identical** | **identical** |

`SA3_FWD_DEBUG` agrees at the hash level, not just the magnitude — `xb[0]`, `xb[depth]` and
`gcond_p` have identical FNV values across all three runs at every step
(step 1 `xb[0]` `abssum=1.180930e+06 fnv=0264391687cb8cb7` in all three; your Metal poison moved
`xb[0]` 1.21e6 -> 2.05e23).

So on CUDA `F` fully writes every scratch buffer it reads. This is the strong form of
non-reproduction: not "we did not see the symptom" but "we deliberately filled 96% of F's scratch
with `0xDEADBEEF` and the output did not move a bit."

**Your probes needed a portability fix to run here** (committed): `SA3_ZERO_F_SCRATCH` and
`SA3_POISON_F` wrote to `t->data` with `memset`/`memcpy`, which is a host write to a *device*
pointer on CUDA/Vulkan and faults with an access violation. They now go through
`ggml_backend_tensor_set`, which is a plain memcpy on unified-memory Metal, so your results are
unaffected. Worth knowing before you conclude anything from these knobs on a non-Apple backend.

### 2. Metal's `SET` does NOT omit the `src0 -> dst` copy

This was the read you asked for, and the answer is the opposite of the hypothesis. All three
backends implement the same contract:

| backend | non-inplace behaviour |
|---|---|
| CPU | `ops.cpp:4595-4605` — `memcpy(dst, src0, ggml_nbytes(dst))` under `if (!inplace)`, then the `src1` region |
| CUDA | `ggml-cuda/set.cu:22-24` — `if (!inplace) ggml_cuda_cpy(ctx, src0, dst);` then `src1` into a dst view |
| **Metal** | `ggml-metal/ggml-metal-ops.cpp:2042-2080` — `if (!inplace)` dispatches a full `cpy` pipeline, `ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, ...)` over **src0's whole row count**, then `ggml_metal_op_concurrency_reset` (a real `ggml_metal_encoder_memory_barrier`), then the `src1` copy |

For `ne=[1536,576]` that first dispatch is 576 threadgroups — every row. And the generic op loop
already inserts a barrier when a node reads a range a previous node wrote
(`ggml_metal_op_concurrency_check` at `ggml-metal-ops.cpp:221`), so the ordering is covered too.

I could not find the missing write in the source. That does not invalidate your measurements —
poisoning changing the result and zeroing fixing it are real — but "Metal's `SET` skips the copy"
is not what the code says, so the upstream-bug framing should not go out on it yet.

### 3. Why the bisect may not name the op it looks like it names

gallocr **recycles**: one buffer region backs many tensors across a graph. Poisoning "node 25's
data" poisons a *region*, and that region is also the storage for other tensors at other points in
the schedule. So the bisect isolates a **memory range**, not necessarily the op that owns it at
index 25. This is the same aliasing that invalidated the earlier per-node checksum pass, in a
different disguise.

### 4. Two tests that would settle it, in order

1. **`test-backend-ops -o SET` on Metal.** ggml ships a `test_set` case for `GGML_OP_SET`
   (`ggml/tests/test-backend-ops.cpp:2872`) that runs the op on the backend and against the CPU
   reference. If it passes, Metal's `SET` is correct for those shapes and the hypothesis is dead
   without any diffusion model involved. If it fails, you have a minimal, self-contained upstream
   reproducer — far stronger than anything derived from this graph.
2. **Read node 25's destination directly after `F`.** Poison, run `F`, then dump the 64
   memory-token rows of that tensor. If they hold `0xDEADBEEF`, `SET` genuinely did not write them
   and (1) should have failed. If they hold `src0`'s values, `SET` did its job and the misread is
   somewhere downstream — which is where the region-vs-op distinction above starts to matter.

Until one of those lands, `SA3_ZERO_F_SCRATCH` remains the sound stopgap, and your point that
`--lora-scope core` must not be gated as the fix stands — the trigger is an allocation plan.

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

## ~~It is specifically `H`, not "a segment with no trainable params"~~ (SUPERSEDED)

> **Superseded.** `H` is a correlate, not a cause — see the status block at the top. The measurements
> below are sound; the conclusion drawn from them is not.

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
- **"It is specifically `H`"** — the biggest one. `H` correlates perfectly (48->49 targets flips it)
  but causes nothing; it changes gallocr's plan, which changes which recycled buffer node 25 gets.
  I held this through several rounds and PC Claude reached it independently, so it is worth naming
  explicitly rather than quietly dropping.
- **"the ordering inverts between machines" as a mystery** — real observation, but it needed no
  special explanation once the cause was allocation layout.
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

### Metal answer to probe 1: the invariant breaks, and it isolates the decomposition

Ran it on Metal exactly as specified (`--adapter-type lora`, `--learning-rate 1e-35`, 512 frames,
seed 42, 3 steps), and added a monolithic pair so the decomposition itself is under test.

| run | H | step 1 | step 2 | step 3 | gnorm (steps 1-3) |
|---|---|---|---|---|---|
| `ckpt` blocks only | skipped | 0.876117 | **1.04967** | 1.81555 | 0.0491 / 4.1408 / 1.7276 |
| `ckpt` + `proj_in` | ran | 0.876117 | **1.42897** | 1.27942 | 0.0801 / 0.3483 / 0.2761 |
| `mono` blocks only | n/a | 0.876117 | **1.41798** | 1.79573 | 0.0491 / 0.4473 / 1.1438 |
| `mono` + `proj_in` | n/a | 0.876117 | **1.42897** | 1.27942 | 0.0801 / 0.3483 / 0.2761 |

`|gctx_sum|` / `|ggcond_sum|` at step 2, checkpointed: blocks-only `0.388184 / 9.194522` against
+`proj_in` `0.006269 / 1.872330`. Step 1 is identical in both (`0.000457 / 0.321336`), matching CUDA.

**Three things fall out.**

1. **`ckpt` + `proj_in` is bit-identical to `mono` + `proj_in`** — loss *and* gnorm, all three steps.
   When `H` runs, the checkpointed decomposition reproduces the monolithic graph exactly.
2. **`ckpt` blocks-only is the lone outlier**, ~26% off the other three at step 2. At `lr=1e-35`
   nothing moves, so all four step-2 forwards should be the same base model on the same sample.
   Only checkpointed-with-`H`-skipped is not.
3. **`mono` breaks the invariant too, but by 0.8%** (1.41798 vs 1.42897) — small, scope-dependent,
   and present with no decomposition involved at all. Plausibly the same thing as the ~4x baseline
   elevation at full scope, and separate from the `H` effect.

So the large effect is **not** training dynamics: with `H` skipped the checkpointed path computes a
different forward, and step 1 being clean means it is state carried across the step boundary rather
than anything wrong at build time. Your reading of the CUDA result — that the accumulation is where
Metal diverges — is where this points, with the added constraint that it only diverges when `H` is
absent, since `H` present reproduces `mono` bit-for-bit.

Worth noting probes 2 and 3 are now the obvious follow-ups, and (3) is especially interesting given
the above: `Gctx_in`/`Ggcond_in` are uploaded and consumed by nobody when `H` is skipped, which is
precisely the configuration that breaks. Skipping that upload when `hgraph` is null is a two-line
change and would test it directly.

### Probes 3 and 1 (fixed-crop): dead writes exonerated, and the bug is pinned to the decomposition

**Probe 3 is dead.** Skipping the `Gctx_in`/`Ggcond_in` upload entirely when `hgraph` is null
(`SA3_SKIP_DEAD_GCTX=1`, on this branch) changes **nothing** — bit-identical to baseline, loss and
both accumulator norms, all three steps. Unconsumed persistent-buffer writes are not the mechanism.

**Probe 1 re-run with `--random-crop false`**, removing the last unlogged variable. Plain `lora`,
`lr=1e-35`, 512 frames, seed 42:

| run | H | step 1 | step 2 | step 3 |
|---|---|---:|---:|---:|
| `ckpt` blocks only | skipped | 0.704558 | **3.04466** | 2.60996 |
| `mono` blocks only | n/a | 0.704558 | **1.31317** | 1.27079 |
| `ckpt` + `proj_in` | ran | 0.704558 | **1.35079** | 0.836514 |
| `mono` + `proj_in` | n/a | 0.704558 | **1.35079** | 0.836514 |

Sampling is confirmed identical across scopes — same `id`, `t`, `mask`, `n_gen`/`n_ctx` at every
step — so this is not an RNG/sampling artefact, and with the crop fixed there is no unlogged input
left.

- **`ckpt` + `proj_in` == `mono` + `proj_in`, exactly.** With `H` built, the decomposition
  reproduces the monolithic graph bit-for-bit.
- **`ckpt` blocks-only is 2.3x off its own monolithic reference** (3.04466 vs 1.31317), same scope,
  same everything. Adapters are provably contributing `+0`. So the checkpointed path computes a
  **wrong forward** when `H` is absent. This is a computational bug, not training dynamics.
- **`mono` blocks-only vs `mono` + `proj_in` still differ by 2.8%** (1.31317 vs 1.35079) with no
  decomposition involved at all. Second, smaller, independent effect — likely the same thing as the
  ~4x baseline elevation at full scope.

### CUDA control for the fixed-crop matrix: all four cells are bit-identical

Same four runs on CUDA (plain `lora`, `lr=1e-35`, `--random-crop false`, 512 frames, seed 42):

| run | H | loss s1 / s2 / s3 | gnorm s1 / s2 / s3 |
|---|---|---|---|
| `ckpt` blocks only | skipped | 0.434523 / 0.602845 / 0.404239 | 0.0137 / 0.0251 / 0.0594 |
| `mono` blocks only | n/a | **identical** | **identical** |
| `ckpt` + `proj_in` | ran | **identical** | 0.0233 / 0.0465 / 0.0684 |
| `mono` + `proj_in` | n/a | **identical** | identical to `ckpt` |

`|gctx_sum|` / `|ggcond_sum|` are identical across scopes at every step too
(0.000175/0.038836, 0.007143/0.110514, 0.000551/0.173036). Only `gnorm` moves, and only with
scope — the extra L2 term from one more adapted target.

So the matrix is **fully degenerate on CUDA**, which makes both of your deviations unambiguous:

- `ckpt` blocks-only 2.3x off its own `mono` — the `H`-skip bug.
- **`mono` blocks-only vs `mono` +`proj_in` differing 2.8%** — on CUDA these are *exactly* equal,
  so your second effect is real and genuinely independent of the decomposition. Two separate Metal
  bugs, confirmed from this side.

### Is `mono` even a valid reference on the M4? At this scope, yes — but check it

Worth stating, since monolithic OOMs on that machine in other configurations. The two regimes are
far apart:

| config | monolithic allocator |
|---|---|
| plain `lora`, 72-73 targets, 512 frames (these probes) | **6,412 MiB** (CUDA measurement) |
| materialized families (`dora-cols`/`bora`), 228 targets | 19,000-30,400 MiB -> hard OOM on 22.9 GB |

The probes use the functional path, which never materializes a `W_eff`, so ~6.4 GB against a
22.9 GB working set should fit with room. The earlier OOMs were a different regime entirely.

**But confirm it rather than assume**, because the failure is silent in exactly the way that would
poison this: an OOM leaves the Metal backend in an error state while `sa3-train` still exits 0 and
prints finite losses — your own finding. Please report the `[train] DiT fwd+bwd graph: N nodes,
gallocr buffer X MiB` line for the `mono` runs and confirm no
`kIOGPUCommandBufferCallbackErrorOutOfMemory` in their output. If either `mono` cell was quietly
OOMing, the 2.8% second effect evaporates and only the `H` bug is real.

### What that leaves

When `hgraph` is null the differences are: H's graph is never built (so `halloc` is never
allocated), H never computes, H's gradients are never read, and — now excluded — its input uploads
are skipped. Since step 1 is clean and step 2 is wrong, whatever it is has to be state that
survives the step boundary.

Two candidates worth trying next, in order:

1. **Allocation layout.** `halloc` is a real backend buffer that exists in one case and not the
   other, which shifts every later allocation. Allocating a same-sized dummy buffer when `H` is
   skipped is a cheap discriminator: if the blow-up disappears, this is a layout/aliasing effect
   rather than anything about `H` itself. (Earlier range dumps showed no overlap between F/T/H/block
   scratch and the persistent buffer, but those were min/max spans and would miss a gap.)
2. **H's compute as an implicit barrier.** H is the last GPU work in the step when present. An
   end-of-step `ggml_backend_synchronize` was already tested and did *not* help, which weakens this,
   but H does more than synchronise — it reads `x`/`tfeat`/`cross`/`global` and runs a full head
   forward, so it touches persistent inputs the next step's F also reads.

3. **Read the persistent boundary tensors after F — this one is methodologically safe.** Your
   scratch-checksum pass was invalid because gallocr reuses node buffers, but `xb[]`,
   `context_p` and `gcond_p` live in the **persistent** buffer (`pbuf`), which no gallocr ever
   touches. Reading them back with `ggml_backend_tensor_get` immediately after `F` computes is
   therefore meaningful, and it splits the remaining space in one run:

   > With the adapter frozen and inert, `ckpt` blocks-only and `ckpt` +`proj_in` see the same base
   > model on the same sample, so `xb[depth]`, `context_p` and `gcond_p` after `F` must be
   > identical at **every** step, including step 2.

   If they already differ at step 2, `F` is computing a wrong forward and everything downstream —
   block gradients, `gctx_sum`, `gnorm` — is just the consequence. If they match, `F` is fine and
   the fault is in the backward decomposition (`T`/`B_l`), which would be surprising given the
   adapter contributes nothing.

   That is the cheapest way to find out whether you are chasing a forward bug or a backward one,
   and it is the question everything else currently hinges on. `loss` alone cannot distinguish
   them because it is computed in `T`, downstream of both.

### Probe 3 (safe forward read): F computes a wrong `x0`, with everything upstream bit-identical

Ran it. `xb[]`, `context_p`, `gcond_p` read from the persistent buffer immediately after F, plus
F's inputs and the shared adapter tensors. Plain `lora`, `lr=1e-35`, `--random-crop false`,
512 frames, seed 42, checkpointed, blocks-only vs +`proj_in`. FNV over raw bytes, so any bit
difference shows.

**Step 1: every tensor bit-identical.** **Step 2:**

| tensor | blocks-only | +`proj_in` | |
|---|---|---|---|
| `IN:x` | `1b1de7f35f1a3484` | `1b1de7f35f1a3484` | identical |
| `IN:target` | `46c2f6e91696cdb3` | `46c2f6e91696cdb3` | identical |
| `IN:local` | `be05445d6c098a83` | `be05445d6c098a83` | identical |
| `IN:cross` | `885c52e59a4a3c36` | `885c52e59a4a3c36` | identical |
| `ADPT:A[0]` | `6754814d3f32d4a3` | `6754814d3f32d4a3` | identical |
| `ADPT:B[0]` | `4a5ba5a04ab0c598` | `4a5ba5a04ab0c598` | identical (abssum 9.56e-33) |
| `context_p` | `6bdaa5c2c8f53962` | `6bdaa5c2c8f53962` | identical |
| `gcond_p` | `6a1c57b46977eca7` | `6a1c57b46977eca7` | identical |
| **`xb[0]`** | `27d945097a8d4c2f` | `0386318a8691ba7c` | **differs, abssum 1.474e6 vs 1.377e6 (7%)** |
| **`xb[depth]`** | `fc1b9484783dc1e5` | `d3aa93d289504f44` | **differs** |

So at step 2: identical inputs, identical shared adapters, frozen base weights, `B ~ 9.6e-33`.
The sole difference between the runs is that one extra target carries four graph nodes
(`mul_mat`, `mul_mat`, `scale`, `add`) contributing ~1e-33. **`x0` moves 7%.**

**This answers your fork: it is a forward bug, not a backward one.** Everything downstream —
block gradients, `gctx_sum`, `gnorm`, loss — is consequence.

**And it localises inside `dit_head` itself.** `x0`, `context` and `gcond` all come out of the same
`dit_head()` call. Two are bit-identical; only `x0` diverges — and `x0` is the branch containing
`proj_in`, the target whose adapter is being added.

RNG draw-order is excluded: `lora_A` is bit-identical across the two runs despite the differing
target count, so the shared 72 adapters are the same objects. Sampling is excluded (all inputs
identical, crop fixed). The optimizer is excluded (`B` identical at step 2; it only diverges at
step 3, as a consequence).

Note the trigger: at step 1 `B` is **exactly** 0 and `x0` is bit-identical; at step 2 `B` is
~1e-33 and `x0` is 7% off. So it is not the adapter's magnitude but the transition from
"contributes exactly zero" to "contributes a denormal-scale value" — or, equivalently, whatever
changes in F's execution once those four nodes carry nonzero data.

Given no overlap was found between F/T/H/block scratch and the persistent buffer (min/max spans
only, so a gap could hide), your allocation-layout candidate is now the leading one: the four extra
nodes change F's gallocr plan, and F is the graph whose output is wrong.

Mono OOM check you asked for, on the fixed-crop matrix: `mono` blocks-only
`DiT fwd+bwd graph: 7884 nodes, gallocr buffer 6412.1 MiB`, +`proj_in` 7907 nodes / 6420.9 MiB —
matching your 6,412 MiB CUDA figure. Zero `Insufficient Memory`, zero command-buffer failures in
all four cells. **The 2.8% second effect is not an OOM artefact.**

## ROOT CAUSE: F reads scratch it does not fully write

`SA3_ZERO_F_SCRATCH=1` (on this branch) memsets every non-persistent tensor in `fgraph` before each
`ggml_backend_graph_compute`. That alone fixes the broken configuration:

| run | step 1 | step 2 | step 3 | `xb[0]` abssum |
|---|---:|---:|---:|---:|
| blocks-only, baseline | 0.704558 | **3.04466** | 2.60996 | 1.474471e+06 |
| blocks-only, **F scratch zeroed** | 0.704558 | **1.35079** | 0.836514 | **1.376813e+06** |
| +`proj_in` (known-correct reference) | 0.704558 | **1.35079** | 0.836514 | 1.376813e+06 |

The zeroed blocks-only run reproduces the reference trajectory exactly and its `xb[0]` matches to
seven digits. (The FNV differs in the last bits because the two runs carry different adapter sets,
so the ~1e-33 residual is not identical — irrelevant at this scale.)

**Mechanism.** F's allocation is planned once at build and never changes, so "step 1 right, step 2
wrong" was never explicable by a bad plan. What it is: F reads a scratch buffer it does not fully
write. Step 1 is correct only because a freshly allocated backend buffer happens to be zeroed; from
step 2 that memory holds F's own step-1 leftovers, and `x0` comes out wrong. Whether the
uninitialised read lands on live data depends on the gallocr plan, which is why adding `proj_in`'s
four nodes flipped it — and why `H` was only ever a correlate. `H` never caused anything.

**This also retires the earlier confusion.** `context_p`/`gcond_p` were always bit-identical because
their producing nodes do not touch the affected buffer; only the `x0` branch does. And the CPU
backend reproducing it is consistent — gallocr is backend-agnostic, so the same plan and the same
under-written destination occur there too.

### What is NOT yet known

Which op under-writes its destination. Candidates worth checking in `dit_head`'s `x0` path: anything
producing a tensor larger than the region its kernel writes — a padded/strided destination, a
`ggml_cpy`/`ggml_cont` over a view that does not cover the full destination, or the memory-token
concat. Identifying it is the difference between a real fix and the blunt workaround.

### Ruled out along the way (all with measurements)

Aliasing between graph buffers — dumped scratch extents for F/T/B/H plus persistent in both configs:
**no overlaps** except `B[0]`/`B[1]`, which share one allocator by design (#28's shape grouping).
Also excluded earlier: BLAS/Accelerate, RNG draw order (`lora_A` bit-identical across scopes),
sampling (all F inputs bit-identical, crop fixed), the optimizer (`B` bit-identical at step 2),
adapter magnitude (lr swept 20 orders), dead `Gctx_in`/`Ggcond_in` writes, and my own instrumentation.

### Fix options

1. **Find and fix the under-writing op.** Correct, and fixes it for every backend and every graph
   that happens to hit the same plan — this is not necessarily specific to `--lora-scope core`, it
   is specific to an allocation layout, so other scopes could hit it too.
2. **Zero F's scratch each step.** Blunt but cheap (67 MiB memset, ~ms against a 12 s step) and
   provably correct here. Reasonable as a stopgap.
3. **Gate `--lora-scope core` on Metal.** Narrowest, but note the finding above: if the trigger is
   an allocation layout rather than the scope itself, gating `core` does not guarantee other scopes
   are safe.

## THE OP: node 25, `GGML_OP_SET`, does not write its full destination

Poison probe (`SA3_POISON_F="lo:hi"`, on this branch) fills fgraph nodes in a range with
`0xDEADBEEF` before F runs, then reads `xb[0]`. A node whose kernel fully writes its destination is
immune; one that is read before being written propagates the pattern. **Step 1 only**, so the clean
run is known-correct and any change is caused by the fill.

```
clean         xb[0] abssum=+1.214878e+06
poison all    xb[0] abssum=+2.051229e+23     <- F reads uninitialised scratch
```

Binary search over the node range, 13 probes, converged on **node 25**:

```
[node] 0021 SET  ne=[1536,576] dst=0x160d9b000 src0=0x160738000 separate  src: NONE[1536,576] NONE[1536,64]
[node] 0025 SET  ne=[1536,576] dst=0x1610fb000 src0=0x160d9b000 separate  src: SET[1536,576] MUL_MAT[1536,512]
```

**This is the memory-token concat.** `576 = 64 memory tokens + 512 frames`. Node 21 writes the 64
memory-token rows; node 25 writes the 512 projected-latent rows into the same `[1536,576]` tensor.
`ggml_set` writes only its target region — the other 64 rows are supposed to come from `src0`.

`dst` is a **different buffer from `src0`** (`0x1610fb000` vs `0x160d9b000`), so for the result to be
correct the kernel must copy `src0` into `dst` before writing its region. **It does not** — which is
exactly what the poison probe proves: if `dst` were fully written, pre-existing content could not
affect the output.

So rows 0..64 of node 25's output are whatever happened to be in that buffer. On step 1 the backend
buffer is freshly allocated; from step 2 it holds F's own leftovers.

**Why the scope changes the outcome.** Both configurations have `dst != src0` — aliasing is *not*
the discriminator. What differs is which recycled buffer gallocr hands node 25, and therefore what
stale content those 64 rows inherit. Adding `proj_in`'s six nodes shifts the plan (the same `SET`
sits at index 31 there, `dst=0x162ee3000`), and that buffer happens to carry benign content. Nothing
about `H`, `--lora-scope core`, or Metal is causal — they are all downstream of which buffer got
reused.

**Confirming fix, already measured:** zeroing F's scratch every step
(`SA3_ZERO_F_SCRATCH=1`) makes the broken configuration reproduce the correct trajectory exactly.

### What to check next, and it is a read not a run

`ggml_compute_forward_set` on CPU copies `src0` into `dst` when not inplace (guarded by an
`inplace` flag). The question is whether ggml's **Metal** `SET` kernel does the same, and whether
`ggml_set()` non-inplace is contractually required to. If the Metal kernel omits that copy, this is
a ggml bug affecting any non-inplace `SET` whose destination is a recycled buffer — not specific to
this graph, and worth an upstream report.

Two reasons this is worth reading rather than bisecting further: it does not reproduce on the PC, so
there is nothing to bisect there; and the answer is a few lines of kernel source either way.

### Impact reassessment

Wider than "scoped training on Apple". The trigger is a gallocr plan that hands node 25 a dirty
buffer. Any scope, any backend and any graph shape could land on such a plan — `--lora-scope core`
on Metal is simply one instance we found. Gating `core` would not make the rest safe.

## ~~Open question nobody has answered~~ — ANSWERED

> The sign flip is explained: narrowing the scope changes gallocr's plan, so node 25 gets a
> different recycled buffer and inherits different stale rows. The direction was never meaningful.

<details><summary>original wording</summary>

The **direction**. Widening the scope raises the norm on CUDA and lowers it on Metal, with identical
host-side optimizer code. Whatever the mechanism is, it has to explain a sign flip, not just a
magnitude.

</details>

## ~~Suggested next probes on Metal~~ — ALL THREE RUN, see results above

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
