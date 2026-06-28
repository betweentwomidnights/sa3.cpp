# sa3.cpp — Roadmap

**Goal:** a portable C++/GGML implementation of Stable Audio 3 (`medium` DiT + `SAME-L` AE, the ARC
ping-pong variant), supporting text2music, audio2audio, and inpainting/continuation, with the ability
to load PyTorch-compatible LoRA adapters. Model: acestep.cpp.

> **STATUS (2026-06-27): Phases 0–6 complete + CUDA/f16 (Phase 7a).** text2music, audio2audio,
> inpaint/continuation, BOTH sizes (medium SAME-L + small-music SAME-S), and lora/dora/bora(+xs)
> adapters are all validated vs PyTorch at cossim ~1.0. Runs on CPU and CUDA; f16 quantization done;
> medium generation ~3.5s on an 8GB 5070 with the sliding-window decoder scaling linearly to multi-minute
> output. Benchmarks: BENCHMARKS.md. **Vulkan backend DONE (Phase 7b): VULKAN.md** — whole stack runs with
> zero custom ops, ~4.1s on the 5070, op-level parity ~1.0 (end-to-end differs as a different-but-valid
> sample due to tensor-core accumulation; perceptually identical). **NEXT: Metal.** (The backend is selected
> via ggml's device registry, so the Vulkan build did "just work" — no graph changes.) NOTE: Phase 6 below
> ("static merge") is **superseded** — the real adapters are **DoRA**
> (non-linear in strength, non-additive across adapters), applied at runtime in weight space via an
> in-place ggml graph (src/lora.h), NOT statically merged. The richer state lives in the goal memory.

Guiding principle: **validate every component against the PyTorch reference (cosine similarity per
tensor) before moving on.** acestep does exactly this in its `tests/`; we copy that discipline. Build
inward-out along the *output* path so each phase produces something audible.

---

## Phase 0 — Architecture spike & validation harness
- Pull `model_config.json` + weights for `stable-audio-3-medium` and `SAME-L` from HF; record exact
  dims/depths/heads/latent_dim and resolve the Phase-0 open questions in ARCHITECTURE-NOTES.md.
- Stand up the repo skeleton (CMake, GGML submodule — start **vanilla**, add patches only if forced).
- Write `convert.py` (safetensors → GGUF) for SAME-L first; bundle config + tokenizer.
- Build a reference-dump harness: run the PyTorch model, save intermediate activations to compare
  against (`tools/dump_refs.py` + a cossim checker). **Exit criteria:** can dump and diff tensors.

## Phase 1 — SAME-L decoder (latent → audio)  ⭐ first audible milestone
- Port `TransformerResamplingBlock` (upsampling path), differential attention, WNConv1d folding.
- Implement tiled decode (mirror `decode_audio` overlap/chunk).
- Tool: `sa3-codec --decode` (feed a PyTorch-dumped latent → WAV).
- **Exit:** decoded WAV matches PyTorch decode, cossim > 0.999 on the waveform/latent path.

## Phase 2 — SAME-L encoder (audio → latent)
- Port the downsampling path; reuse the block from Phase 1.
- Tool: `sa3-codec --encode`; validate encode→decode roundtrip and encoder output vs PyTorch.
- **Exit:** encoder latents match PyTorch; roundtrip is clean. Unblocks audio2audio + inpainting.

## Phase 3 — T5Gemma text encoder
- Port (or adapt an existing GGML gemma graph for) the t5gemma encoder; produce `cross_attn_cond`.
- Tool: `sa3-textenc "prompt"` dumping the conditioning tensor.
- **Exit:** conditioning tensor matches PyTorch for several prompts.

## Phase 4 — DiT + ping-pong sampler (text2music)  ⭐ first full song
- Port the DiT forward graph (timestep Fourier embed in fp32, cross-attn, global/prepend cond,
  pre/post residual convs). Wire `pingpong` into the solver registry; implement `build_schedule` +
  `dist_shift`; seeded noise via Philox.
- Start with CFG off (ARC default). Add batched CFG + APG afterward if needed.
- Tool: `sa3-synth` (prompt → latent → SAME-L decode → WAV).
- **Exit:** per-layer DiT cossim vs PyTorch passes; an end-to-end text2music render sounds right.

## Phase 5 — audio2audio + inpainting/continuation
- `init_data` mixing (`noise = init*(1-σmax) + noise*σmax`); inpaint-mask construction from
  start/end seconds (multi-region); `input_concat_cond` packing (masked latents + mask).
- **Exit:** audio2audio and inpaint renders match PyTorch behavior on reference clips.

## Phase 6 — LoRA adapter loading  ✅ (DoRA, runtime weight-space — NOT static merge)
- Adapters are **DoRA** (`dora-rows`), not plain LoRA. `W_eff = magnitude·(W + (alpha/rank)·strength·B@A)/‖·‖_row`
  is non-linear in strength and non-additive across adapters, so static merge is out. Implemented as a
  per-weight recompute (runtime strength + multi-adapter chaining) via an in-place ggml graph that runs on
  the GPU (~0.13s/adapter). Converters: `.ckpt → safetensors → gguf`. All 8 families (lora/dora/bora + -xs)
  supported; dora-rows A/B'd against trained kev/keygen at cossim 1.0, the rest formula-validated.
- **Exit:** ✅ a known DoRA changes output the same way it does in PyTorch (incl. adjustable strength).

## Phase 7 — Productionize
- ✅ f16 quantization (`tools/quantize_gguf.py`); ✅ CUDA backend; ✅ benchmarks (BENCHMARKS.md);
  ✅ Vulkan backend (VULKAN.md).
- **NEXT:** Metal backend. (Optional later: integer quant Q8/Q6/Q4 + quality table;
  CLI/HTTP server; broader examples.)

---

## Sequencing rationale
Output path first (decoder → encoder → text enc → DiT) means every phase yields something you can
*hear* and *diff*, and the hardest-to-validate piece (the DiT denoiser) lands only after its inputs
(conditioning) and outputs (decode) are already trusted. text2music is the first full-pipeline
milestone (Phase 4); audio2audio/inpainting (Phase 5) reuse the encoder + a mask path; LoRA (Phase 6)
is an isolated load-time transform.

## Top risks
1. **t5gemma encoder** has no acestep analogue — may need a hand-written GGML graph. (Phase 3)
2. **Differential attention** in SAME must match exactly. (Phase 1)
3. **Numerical fidelity** of the rf_denoiser/logsnr path (fp32 timestep discipline). (Phase 4)
4. **Gated HF repos / licensing** for weights. (Phase 0)
5. Whether **vanilla GGML** truly suffices (expected yes; confirm in Phase 1).
