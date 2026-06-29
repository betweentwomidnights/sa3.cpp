# SAME-L decoder — implementation spec (Phase 1)

Full forward trace of `pretransform.decode(z)` for `stable-audio-3-medium`, read from the PyTorch
source (`refs/stable-audio-3/stable_audio_3/models/{autoencoders,transformer,bottleneck,pretransforms}.py`).
This is what the GGML decoder must reproduce. Latent `z`: `(B, 256, T)` at ≈10.77 Hz → audio `(B, 2, T·4096)` @ 44.1 kHz.

## Top-level pipeline (`AudioAutoencoder.decode`)
1. **SoftNorm decode** (`bottleneck.py`): `z = z * running_std` (a single scalar). `scaling_factor`/`bias`
   are **encode-only**. There's an inference noise term (`+ randn·running_std·1e-3`) — **disable for
   validation** (and omit in C++; it's a regularizer).
2. **SAMEDecoder** (`z: B,256,T → B,512,T·16`):
   - `Transpose → Linear(256→1536) → Transpose` (this is `decoder.layers.1.{weight,bias}`).
   - `TransformerResamplingBlock` (decoder, in=1536 out=512, stride=16) — see below.
3. **Unpatchify** (`PatchedPretransform.decode`): `rearrange("b (c h) l -> b c (l h)", h=256)`:
   `(B,512,L) → (B,2,L·256)`. i.e. `out[b, c, l*256+h] = x[b, c*256+h, l]`. No postfilter, no oversampling.
4. `soft_clip = False` → no final tanh.
(`decode_audio` adds chunked overlap tiling for long audio. `sa3-generate` mirrors this as optional
outer tiling with `--chunked-decode`, but the GGML default remains monolithic because SAME-L attention
is already computed as linear sliding-window blocks internally.)
For Metal, `SA3_SAME_FLASH_ATTN=1` can replace the compact sliding-window attention subgraph with
`GGML_OP_FLASH_ATTN_EXT` plus a full F16 band mask; this is faster on Apple M4 through 120s, but uses
quadratic mask memory, so it remains opt-in. `SA3_SAME_FLASH_ATTN=local` keeps the compact local
neighborhood shape and is numerically correct, but is slower on the current ggml Metal kernels.

## TransformerResamplingBlock — DECODER, variable-stride, sliding-window
Config: `stride=16`, `variable_stride=True`, `sliding_window=[1,1]`, `transformer_depth=12`,
`sinusoidal_blocks=8`, `dim_heads=64`, `differential=True`, `dyt=True`, `ff_mult=3`, `mask_noise=0.1`.
Derived: `sub_chunk_size = stride+1 = 17`, `output_seg_size = stride = 16`, `input_seg_size = 1`,
`sliding_window_seq = [17,17]` (= `1*(stride+1)`).

Forward (input `x: B,1536,T`):
1. `x = rearrange("b c t -> b t c")` → `(B,T,1536)`.
2. Pad seq to multiple of `input_seg_size=1` (no-op); `rearrange("b (n c) d -> (b n) c d", c=1)` → `(B·T,1,1536)`.
3. **Insert learned upsampling tokens**: `new_tokens` param shape `(1,1,1536)` → expand to `(B·T,16,1536)`.
   `mask_noise>0` adds `randn·0.1` → **disable for validation**. `x = cat([x, new_tokens], dim=-2)` →
   `(B·T,17,1536)` (1 real frame + 16 query slots).
4. `rearrange("(b n) c d -> b (n c) d")` → `(B, T·17, 1536)`.
5. Since `sliding_window` is set, **no chunk folding** — run all 12 layers over the full `(B, T·17, 1536)`
   sequence with **sliding-window self-attention, band ±17** (`flash_attn_sliding_window=[17,17]`).
6. `rearrange("b (n c) d -> (b n) c d", c=17)`; `x = x[:, -16:, :]` (drop the 1 real, keep 16 filled);
   `rearrange("(b n) c d -> b d (n c)")` → `(B, 1536, T·16)`.
7. **Mapping** (decoder applies it last): `WNConv1d(1536→512, k=1)` → `(B, 512, T·16)`. Fuse weight-norm
   (`mapping.weight_g`,`mapping.weight_v`) → plain conv weight at convert time.

## TransformerBlock (AE variant — no adaLN, no cross-attn)
`x = x + self_attn(pre_norm(x), rope)` then `x = x + ff(ff_norm(x))`. `layer_scale=False` → scales are
identity. `zero_init_branch_outputs` only affects init, not inference.

- **Norm = DynamicTanh (DyT)**, NOT RMSNorm: `y = tanh(alpha · x) · gamma + beta`
  (`alpha` scalar, `gamma`/`beta` per-dim). Applies to `pre_norm`, `ff_norm` (dim 1536) and to qk-norm
  (dim 64). No eps. (`norm_type='dyt'`, `qk_norm='dyt'`.)
- **Differential attention** (`dim_heads=64`, fused `to_qkv: 1536→7680 = 5×1536`):
  - split into `q,k,v,q_diff,k_diff`, each `→ (B,h,N,64)`, `h=1536/64=24`.
  - qk-norm: DyT over the 64-dim head, applied to `[q;q_diff]` and `[k;k_diff]` (eps 1e-3, but DyT ignores eps).
  - **partial RoPE** on first 32 of 64 head dims (`RotaryEmbedding(32)`, `inv_freq[16]`, base 10000),
    computed over the full packed length (T·17), in fp32; applied to q and k (both main and diff).
  - `out = softmax(q·kᵀ/√64)·v − softmax(q_diff·k_diffᵀ/√64)·v` (two SDPAs, **subtracted**; no learnable λ).
  - merge heads → `to_out` (1536→1536, no bias).
  - **sliding-window band ±17** on the attention scores (mask scores where `|i−j|>17`).
- **FF = SwiGLU**: `proj: 1536 → 2·(1536·3)=9216`; `chunk → (x, gate)`; `x · act(gate)`;
  `Linear(4608→1536)`. Activation `act` is `SiLU`, except the **last 7 blocks** (`i≥5`, since
  `(12−i)<8`) use `Sin(x)=sin(π·x)`.

## Weights (names from `docs/TENSOR-MAP.md` / `tensors-medium.txt`)
`pretransform.model.bottleneck.running_std` (scalar); `pretransform.model.decoder.layers.1.{weight,bias}`
(Linear 256→1536); `...decoder.layers.3.mapping.{weight_g,weight_v}` (WNConv1d→fuse);
`...decoder.layers.3.new_tokens [1,1,1536]`; `...decoder.layers.3.transformers.{0..11}.` with
`pre_norm/ff_norm.{alpha,beta,gamma}`, `self_attn.to_qkv [7680,1536]`, `self_attn.{q,k}_norm.{alpha,beta,gamma}[64]`,
`self_attn.to_out [1536,1536]`, `self_attn.rope.inv_freq [16]`, `ff.ff.0.proj [9216,1536]`, `ff.ff.2 [1536,4608]`.

## Determinism for validation
Disable both inference noises (`mask_noise` on new_tokens; SoftNorm decode noise) in the reference dump
**and** omit them in C++, so cossim is meaningful. Otherwise expect tiny mismatch.

## GGML op coverage check (all stock)
DyT = tanh/mul/add; differential attn = 2× `ggml_soft_max`/`mul_mat` (or flash-attn) then sub;
partial RoPE = `ggml_rope` on a view; SwiGLU = `ggml_silu`/elementwise (+ `ggml_sin` for Sin blocks);
WNConv1d k=1 = `ggml_mul_mat`; new-token insert/reshape = `ggml_concat`/views; sliding band = additive
mask into soft_max. **No custom ops needed** — confirms the vanilla-GGML thesis on the hardest component.
