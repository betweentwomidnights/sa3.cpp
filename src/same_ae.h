// same_ae.h — the SAME-L ("taae_v2") autoencoder for Stable Audio 3 medium.
// Currently the decode path; the encoder (Phase 2) reuses same_block() directly.
// See docs/DECODER-DESIGN.md for the full forward trace this mirrors.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "nn.h"

#include <cmath>
#include <string>

namespace sa3 {

struct SameConfig {
    int dim, latent, dim_heads, n_heads, depth;
    int sub_chunk, output_seg, sliding_window, sinusoidal_blocks;
    int out_channels, patch_size, rot;
    float rope_base, running_std;

    static SameConfig from(const GgufModel& m) {
        SameConfig c;
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
        return c;
    }
};

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

// Decoder checkpoints (also the validation taps).
struct DecodeOut { ggml_tensor* after_in_proj; ggml_tensor* after_resampling; ggml_tensor* audio; };

// Full decode: z [latent, T] -> audio [T*patch*output_seg-as-time, audio_channels].
// pos: I32 [N=T*sub_chunk]; mask: [N, N] sliding-window band.
inline DecodeOut same_decode(ggml_context* ctx, const GgufModel& W, ggml_tensor* z,
                             const SameConfig& c, int T, ggml_tensor* pos, ggml_tensor* mask) {
    const int dim = c.dim;
    const int64_t N = (int64_t)T * c.sub_chunk;
    DecodeOut out{};

    // bottleneck.decode (z * running_std) + in_proj (latent -> dim)
    ggml_tensor* x = ggml_scale(ctx, z, c.running_std);
    x = nn::linear(ctx, W.get("ae.in_proj.weight"), x, W.get("ae.in_proj.bias"));   // [dim, T]
    out.after_in_proj = x; ggml_set_output(x);

    // pack each frame with `output_seg` learned new_tokens -> [dim, sub_chunk, T] -> [dim, N]
    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, dim, 1, T);
    ggml_tensor* nt = ggml_reshape_3d(ctx, W.get("ae.dec.new_tokens"), dim, 1, 1);
    ggml_tensor* nt_rep = ggml_repeat(ctx, nt, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, c.output_seg, T));
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_concat(ctx, x3, nt_rep, 1)), dim, N);

    for (int l = 0; l < c.depth; l++) {
        bool sinusoidal = (c.depth - l) < c.sinusoidal_blocks;
        x = same_block(ctx, W, "ae.dec." + std::to_string(l) + ".", x, pos, mask, c, sinusoidal);
    }

    // keep the last `output_seg` of each `sub_chunk` -> [dim, T*output_seg], then mapping dim->out_channels
    ggml_tensor* x17 = ggml_reshape_3d(ctx, x, dim, c.sub_chunk, T);
    ggml_tensor* kept = ggml_cont(ctx, ggml_view_3d(ctx, x17, dim, c.output_seg, T, x17->nb[1], x17->nb[2], x17->nb[1]));
    ggml_tensor* up = ggml_reshape_2d(ctx, kept, dim, (int64_t)c.output_seg * T);
    ggml_tensor* mapped = ggml_mul_mat(ctx, W.get("ae.dec.mapping.weight"), up);    // [out_channels, L]
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
