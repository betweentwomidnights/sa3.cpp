// same_ae.h — the SAME-L ("taae_v2") autoencoder for Stable Audio 3 medium.
// Currently the decode path; the encoder (Phase 2) reuses same_block() directly.
// See docs/DECODER-DESIGN.md for the full forward trace this mirrors.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "nn.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace sa3 {

struct SameConfig {
    int dim, latent, dim_heads, n_heads, depth;
    int sub_chunk, output_seg, sliding_window, sinusoidal_blocks;
    int out_channels, patch_size, rot;
    float rope_base, running_std;
    bool chunk, dec_conv_mapping;   // SAME-S: chunk+midpoint-shift attn / k=3 decoder mapping
    int  chunk_size, eff_chunk, split, shift;

    static SameConfig from(const GgufModel& m) {
        SameConfig c;
        auto u32opt = [&](const char* k, int def){ int i = gguf_find_key(m.gguf, k); return i < 0 ? def : (int)gguf_get_val_u32(m.gguf, i); };
        c.chunk            = u32opt("sa3.ae.chunk", 0) != 0;
        c.chunk_size       = u32opt("sa3.ae.chunk_size", 0);
        c.dec_conv_mapping = u32opt("sa3.ae.dec_conv_mapping", 0) != 0;
        c.dim              = m.u32("sa3.ae.dim");
        c.latent           = m.u32("sa3.ae.latent_dim");
        c.dim_heads        = m.u32("sa3.ae.dim_heads");
        c.n_heads          = m.u32("sa3.ae.n_heads");
        c.depth            = m.u32("sa3.ae.depth");
        c.sub_chunk        = m.u32("sa3.ae.sub_chunk");
        c.output_seg       = m.u32("sa3.ae.output_seg");
        c.sliding_window   = m.u32("sa3.ae.sliding_window");
        c.sinusoidal_blocks= m.u32("sa3.ae.sinusoidal_blocks");
        c.out_channels     = m.u32("sa3.ae.out_channels");
        c.patch_size       = m.u32("sa3.ae.patch_size");
        c.rope_base        = m.f32("sa3.ae.rope_base");
        c.rot              = c.dim_heads / 2;
        c.running_std      = m.scalar("ae.running_std");
        // chunk geometry (decoder stride == output_seg): eff = chunk + chunk//stride
        c.eff_chunk = c.chunk ? c.chunk_size + c.chunk_size / c.output_seg : 0;
        c.split     = c.depth / 2;
        c.shift     = c.eff_chunk / 2;
        return c;
    }
};

// Host-side [M,M] additive attention mask (row-major (q,k)) for a SAME AE:
// block-diagonal over eff_chunk (chunked, SAME-S) or sliding-window band (SAME-L).
// 0 where attention is allowed, -inf where masked. Shared by every driver so the
// SAME-L band and SAME-S block-diag rule live in exactly one place.
inline std::vector<float> build_attn_mask(const SameConfig& c, int M) {
    std::vector<float> b((size_t)M * M);
    for (int q = 0; q < M; q++) for (int k = 0; k < M; k++)
        b[(size_t)q*M + k] = c.chunk ? ((q/c.eff_chunk == k/c.eff_chunk) ? 0.0f : -INFINITY)
                                     : ((std::abs(q-k) <= c.sliding_window) ? 0.0f : -INFINITY);
    return b;
}

// One SAME TransformerBlock over a packed sequence x:[dim, N]:
//   x = x + diff_attn(dyt(x))   then   x = x + swiglu_ff(dyt(x))
// Differential attention = sdpa(q,k,v) - sdpa(q_diff,k_diff,v) with a band mask;
// per-head DyT qk-norm; partial NeoX RoPE. `sinusoidal` swaps SiLU for sin(pi*x).
inline ggml_tensor* same_block(ggml_context* ctx, const GgufModel& W, const std::string& p,
                               ggml_tensor* x, ggml_tensor* pos, ggml_tensor* mask,
                               const SameConfig& c, bool sinusoidal) {
    const int   dim = c.dim, dh = c.dim_heads, nh = c.n_heads;
    const int64_t N = x->ne[1];
    const float scale = 1.0f / sqrtf((float)dh);

    // ---- differential self-attention ----
    ggml_tensor* h = nn::dyt(ctx, x, W.get(p+"pre_norm.alpha"), W.get(p+"pre_norm.gamma"), W.get(p+"pre_norm.beta"));
    ggml_tensor* qkv = ggml_mul_mat(ctx, W.get(p+"self_attn.to_qkv.weight"), h); // [5*dim, N]

    auto slice = [&](int i) {            // chunk i of `dim` along ne0
        return ggml_view_2d(ctx, qkv, dim, N, qkv->nb[1], (size_t)i*dim*sizeof(float));
    };
    auto heads = [&](ggml_tensor* a) {   // [dim,N] -> [dh,nh,N]
        return ggml_reshape_3d(ctx, ggml_cont(ctx, a), dh, nh, N);
    };
    ggml_tensor* q = heads(slice(0)), *k = heads(slice(1)), *v = heads(slice(2)),
                *qd = heads(slice(3)), *kd = heads(slice(4));

    ggml_tensor *qa=W.get(p+"self_attn.q_norm.alpha"), *qg=W.get(p+"self_attn.q_norm.gamma"), *qb=W.get(p+"self_attn.q_norm.beta");
    ggml_tensor *ka=W.get(p+"self_attn.k_norm.alpha"), *kg=W.get(p+"self_attn.k_norm.gamma"), *kb=W.get(p+"self_attn.k_norm.beta");
    q = nn::dyt(ctx,q,qa,qg,qb); qd = nn::dyt(ctx,qd,qa,qg,qb);
    k = nn::dyt(ctx,k,ka,kg,kb); kd = nn::dyt(ctx,kd,ka,kg,kb);

    q = nn::rope_neox(ctx,q,pos,c.rot,c.rope_base); qd = nn::rope_neox(ctx,qd,pos,c.rot,c.rope_base);
    k = nn::rope_neox(ctx,k,pos,c.rot,c.rope_base); kd = nn::rope_neox(ctx,kd,pos,c.rot,c.rope_base);

    auto toAttn = [&](ggml_tensor* a){ return ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3)); }; // [dh,nh,N]->[dh,N,nh]
    q=toAttn(q); k=toAttn(k); v=toAttn(v); qd=toAttn(qd); kd=toAttn(kd);

    ggml_tensor* o = ggml_sub(ctx, nn::sdpa(ctx,q,k,v,mask,scale), nn::sdpa(ctx,qd,kd,v,mask,scale)); // [dh,N,nh]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));            // [dh,nh,N]
    o = ggml_reshape_2d(ctx, o, dim, N);
    o = ggml_mul_mat(ctx, W.get(p+"self_attn.to_out.weight"), o);
    x = ggml_add(ctx, x, o);

    // ---- SwiGLU feed-forward (sin activation on the late blocks) ----
    ggml_tensor* f = nn::dyt(ctx, x, W.get(p+"ff_norm.alpha"), W.get(p+"ff_norm.gamma"), W.get(p+"ff_norm.beta"));
    f = nn::linear(ctx, W.get(p+"ff.proj.weight"), f, W.get(p+"ff.proj.bias")); // [2*inner, N]
    const int inner = f->ne[0] / 2;
    ggml_tensor* val  = ggml_view_2d(ctx, f, inner, N, f->nb[1], 0);
    ggml_tensor* gate = ggml_view_2d(ctx, f, inner, N, f->nb[1], (size_t)inner*sizeof(float));
    ggml_tensor* act  = sinusoidal
        ? ggml_sin(ctx, ggml_scale(ctx, ggml_cont(ctx, gate), 3.14159265358979324f))
        : ggml_silu(ctx, ggml_cont(ctx, gate));
    f = ggml_mul(ctx, ggml_cont(ctx, val), act);
    f = nn::linear(ctx, W.get(p+"ff.out.weight"), f, W.get(p+"ff.out.bias"));
    return ggml_add(ctx, x, f);
}

// Encoder checkpoints (validation taps).
struct EncodeOut { ggml_tensor* after_resampling; ggml_tensor* latent; ggml_tensor* z; };

// Full encode: audio [L, audio_channels] -> latent z [latent, T] (T = L / 4096).
// The mirror of same_decode: mapping conv runs first, each chunk packs `stride`
// real frames + 1 new token, we keep the last (the new token's output).
// pos: I32 [N=T*sub_chunk]; mask: [N, N] sliding-window band.
inline EncodeOut same_encode(ggml_context* ctx, const GgufModel& W, ggml_tensor* audio,
                             const SameConfig& c, int T, ggml_tensor* pos, ggml_tensor* mask) {
    const int dim = c.dim, stride = c.output_seg;   // output_seg == stride == 16
    const int ch = c.out_channels / c.patch_size;   // 2
    const int Tp = stride * T;                       // patch-frames (= L / patch_size)
    const int64_t N = (int64_t)T * c.sub_chunk;
    EncodeOut out{};
    // SAME-L (sliding-window) encode only; the SAME-S chunked encoder (Phase 5) will
    // mirror same_decode's midpoint-shift split. Fail loudly until it lands.
    GGML_ASSERT(!c.chunk && "SAME-S chunked encoder not implemented yet");

    // patchify: audio [L, ch] -> [patch, ch, Tp] -> [patch, Tp, ch] -> [patch*ch=512, Tp]
    ggml_tensor* pa = ggml_reshape_3d(ctx, audio, c.patch_size, Tp, ch);
    pa = ggml_cont(ctx, ggml_permute(ctx, pa, 0, 2, 1, 3));          // [patch, ch, Tp]
    ggml_tensor* x = ggml_reshape_2d(ctx, pa, c.out_channels, Tp);   // [512, Tp]

    // mapping conv 512 -> dim (encoder applies it first; WNConv1d has a bias)
    x = nn::linear(ctx, W.get("ae.enc.mapping.weight"), x, W.get("ae.enc.mapping.bias")); // [dim, Tp]

    // pack each chunk of `stride` real frames + 1 new token -> [dim, sub_chunk, T] -> [dim, N]
    ggml_tensor* xg = ggml_reshape_3d(ctx, x, dim, stride, T);       // [dim, 16, T]
    ggml_tensor* nt = ggml_reshape_3d(ctx, W.get("ae.enc.new_tokens"), dim, 1, 1);
    ggml_tensor* nt_rep = ggml_repeat(ctx, nt, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 1, T));
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_concat(ctx, xg, nt_rep, 1)), dim, N); // [dim, N]

    for (int l = 0; l < c.depth; l++)   // encoder has no sinusoidal blocks
        x = same_block(ctx, W, "ae.enc." + std::to_string(l) + ".", x, pos, mask, c, /*sinusoidal=*/false);

    // keep the last 1 of each sub_chunk (the new token's output) -> [dim, T]
    ggml_tensor* x17 = ggml_reshape_3d(ctx, x, dim, c.sub_chunk, T);
    ggml_tensor* kept = ggml_cont(ctx, ggml_view_3d(ctx, x17, dim, 1, T, x17->nb[1], x17->nb[2],
                                                    (size_t)(c.sub_chunk - 1) * x17->nb[1]));
    ggml_tensor* res = ggml_reshape_2d(ctx, kept, dim, T);          // [dim, T]
    out.after_resampling = res; ggml_set_output(res);

    // out_proj dim -> latent, then SoftNorm.encode: (lat * scaling_factor + bias) / running_std
    ggml_tensor* lat = nn::linear(ctx, W.get("ae.out_proj.weight"), res, W.get("ae.out_proj.bias")); // [latent,T]
    out.latent = lat; ggml_set_output(lat);
    ggml_tensor* z = ggml_add(ctx, ggml_mul(ctx, lat, W.get("ae.scaling_factor")), W.get("ae.bias"));
    z = ggml_scale(ctx, z, 1.0f / c.running_std);
    out.z = z; ggml_set_output(z);
    return out;
}

// Decoder checkpoints (also the validation taps).
struct DecodeOut { ggml_tensor* after_in_proj; ggml_tensor* after_resampling; ggml_tensor* audio; };

// Full decode: z [latent, T] -> audio.
// Sliding-window AE (SAME-L): pos [N=T*sub_chunk], mask [N,N] band; pos2/mask2 unused.
// Chunked AE (SAME-S): pos/mask = block-diag over N (first split layers);
//   pos2 [N+2*shift], mask2 [N+2*shift,N+2*shift] = block-diag (shifted half, after edge-pad).
inline DecodeOut same_decode(ggml_context* ctx, const GgufModel& W, ggml_tensor* z,
                             const SameConfig& c, int T, ggml_tensor* pos, ggml_tensor* mask,
                             ggml_tensor* pos2 = nullptr, ggml_tensor* mask2 = nullptr) {
    const int dim = c.dim;
    const int64_t N = (int64_t)T * c.sub_chunk;
    DecodeOut out{};
    // chunked (SAME-S) decode needs the shifted-half graph inputs; make the contract
    // explicit so a 6-arg caller on a SAME-S model fails loudly, not via a null deref.
    GGML_ASSERT(!c.chunk || (pos2 && mask2));

    // bottleneck.decode (z * running_std) + in_proj (latent -> dim)
    ggml_tensor* x = ggml_scale(ctx, z, c.running_std);
    x = nn::linear(ctx, W.get("ae.in_proj.weight"), x, W.get("ae.in_proj.bias"));   // [dim, T]
    out.after_in_proj = x; ggml_set_output(x);

    // pack each frame with `output_seg` learned new_tokens -> [dim, sub_chunk, T] -> [dim, N]
    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, dim, 1, T);
    ggml_tensor* nt = ggml_reshape_3d(ctx, W.get("ae.dec.new_tokens"), dim, 1, 1);
    ggml_tensor* nt_rep = ggml_repeat(ctx, nt, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, c.output_seg, T));
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_concat(ctx, x3, nt_rep, 1)), dim, N);

    auto blk = [&](ggml_tensor* xx, int l, ggml_tensor* p, ggml_tensor* m){
        return same_block(ctx, W, "ae.dec." + std::to_string(l) + ".", xx, p, m, c, (c.depth - l) < c.sinusoidal_blocks);
    };
    if (!c.chunk) {
        for (int l = 0; l < c.depth; l++) x = blk(x, l, pos, mask);
    } else {
        // first `split` layers: block-diag attention over N
        for (int l = 0; l < c.split; l++) x = blk(x, l, pos, mask);
        // shifted half: edge-pad concat([x[:,:shift], x, x[:,-shift:]]) -> run -> slice [shift:shift+N]
        ggml_tensor* lft = ggml_cont(ctx, ggml_view_2d(ctx, x, dim, c.shift, x->nb[1], 0));
        ggml_tensor* rgt = ggml_cont(ctx, ggml_view_2d(ctx, x, dim, c.shift, x->nb[1], (size_t)(N - c.shift)*x->nb[1]));
        x = ggml_concat(ctx, ggml_concat(ctx, lft, x, 1), rgt, 1);          // [dim, N+2*shift]
        for (int l = c.split; l < c.depth; l++) x = blk(x, l, pos2, mask2);
        x = ggml_cont(ctx, ggml_view_2d(ctx, x, dim, N, x->nb[1], (size_t)c.shift * x->nb[1]));  // [dim, N]
    }

    // keep the last `output_seg` of each `sub_chunk` -> [dim, T*output_seg], then mapping dim->out_channels
    ggml_tensor* x17 = ggml_reshape_3d(ctx, x, dim, c.sub_chunk, T);
    ggml_tensor* kept = ggml_cont(ctx, ggml_view_3d(ctx, x17, dim, c.output_seg, T, x17->nb[1], x17->nb[2], x17->nb[1]));
    ggml_tensor* up = ggml_reshape_2d(ctx, kept, dim, (int64_t)c.output_seg * T);   // [dim, L]
    ggml_tensor* mapped;
    if (!c.dec_conv_mapping) {
        mapped = nn::linear(ctx, W.get("ae.dec.mapping.weight"), up, W.get("ae.dec.mapping.bias")); // k=1 = matmul
    } else {
        // k=3 WNConv1d, padding 'same' = sum of 3 tap-matmuls over zero-padded shifts:
        //   out[:,t] = w0@x[:,t-1] + w1@x[:,t] + w2@x[:,t+1]   (F32-exact, no im2col).
        const int64_t L = up->ne[1];
        ggml_tensor* zc = ggml_scale(ctx, ggml_view_2d(ctx, up, dim, 1, up->nb[1], 0), 0.0f); // [dim,1] zeros
        ggml_tensor* xp = ggml_concat(ctx, ggml_concat(ctx, zc, up, 1), zc, 1);               // [dim, L+2]
        auto shifted = [&](int off){ return ggml_view_2d(ctx, xp, dim, L, xp->nb[1], (size_t)off*xp->nb[1]); };
        ggml_tensor* m0 = ggml_mul_mat(ctx, W.get("ae.dec.mapping.w0"), shifted(0));  // x[:,t-1]
        ggml_tensor* m1 = ggml_mul_mat(ctx, W.get("ae.dec.mapping.w1"), shifted(1));  // x[:,t]
        ggml_tensor* m2 = ggml_mul_mat(ctx, W.get("ae.dec.mapping.w2"), shifted(2));  // x[:,t+1]
        mapped = ggml_add(ctx, ggml_add(ctx, ggml_add(ctx, m0, m1), m2), W.get("ae.dec.mapping.bias"));
    }
    out.after_resampling = mapped; ggml_set_output(mapped);

    // unpatchify: [out_channels, L] -> [patch, ch, L] -> [patch, L, ch] -> [L*patch, ch]
    const int L = c.output_seg * T;
    const int ch = c.out_channels / c.patch_size;
    ggml_tensor* pm = ggml_reshape_3d(ctx, mapped, c.patch_size, ch, L);
    pm = ggml_cont(ctx, ggml_permute(ctx, pm, 0, 2, 1, 3));
    out.audio = ggml_reshape_2d(ctx, pm, (int64_t)c.patch_size * L, ch); ggml_set_output(out.audio);
    return out;
}

} // namespace sa3
