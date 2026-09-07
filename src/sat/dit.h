// sat/dit.h -- classic stable-audio-tools ContinuousTransformer DiT.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "nn.h"
#include "sat/model_spec.h"

#include <cmath>
#include <string>

namespace sa3::sat {

inline DitSpec dit_spec_from(const GgufModel& m) {
    if (m.string("sat.architecture") != "stable-audio-tools" ||
        m.u32("sat.format_version") != 1)
        throw gguf_error("unsupported stable-audio-tools DiT GGUF");
    DitSpec c;
    c.io_channels = (int)m.u32("sat.dit.io_channels");
    c.embed_dim = (int)m.u32("sat.dit.width");
    c.depth = (int)m.u32("sat.dit.depth");
    c.num_heads = (int)m.u32("sat.dit.heads");
    c.cond_token_dim = (int)m.u32("sat.dit.cond_dim");
    c.global_cond_dim = (int)m.u32("sat.dit.global_dim");
    c.qk_layer_norm = m.string("sat.dit.qk_norm") == "ln";
    if (!c.qk_layer_norm || c.embed_dim % c.num_heads)
        throw gguf_error("unsupported classic DiT attention geometry");
    return c;
}

inline ggml_tensor* sat_layer_norm(ggml_context* ctx, ggml_tensor* x,
                                    ggml_tensor* gamma, ggml_tensor* beta,
                                    float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), gamma), beta);
}

inline ggml_tensor* sat_mlp(ggml_context* ctx, const GgufModel& W,
                            const std::string& p, ggml_tensor* x, bool bias) {
    x = nn::linear(ctx, W.get(p + "0.weight"), x,
                   bias ? W.get(p + "0.bias") : nullptr);
    x = ggml_silu(ctx, x);
    return nn::linear(ctx, W.get(p + "2.weight"), x,
                      bias ? W.get(p + "2.bias") : nullptr);
}

inline ggml_tensor* sat_heads(ggml_context* ctx, ggml_tensor* x,
                              int head_dim, int heads, int64_t tokens) {
    return ggml_reshape_3d(ctx, ggml_cont(ctx, x), head_dim, heads, tokens);
}

inline ggml_tensor* sat_to_attn(ggml_context* ctx, ggml_tensor* x) {
    return ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)); // [hd,tokens,heads]
}

inline ggml_tensor* sat_merge_heads(ggml_context* ctx, ggml_tensor* x,
                                     int width, int64_t tokens) {
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)); // [hd,heads,tokens]
    return ggml_reshape_2d(ctx, x, width, tokens);
}

inline ggml_tensor* sat_qk_norm(ggml_context* ctx, const GgufModel& W,
                                const std::string& p, ggml_tensor* x) {
    // torch.nn.LayerNorm(head_dim, eps=1e-6), independently per head/token.
    return sat_layer_norm(ctx, x, W.get(p + ".weight"), W.get(p + ".bias"), 1.0e-6f);
}

inline ggml_tensor* classic_dit_block(ggml_context* ctx, const GgufModel& W,
                                       const std::string& p, ggml_tensor* x,
                                       ggml_tensor* context, ggml_tensor* pos,
                                       const DitSpec& c, int rotary_dims) {
    const int width = c.embed_dim;
    const int heads = c.num_heads;
    const int hd = width / heads;
    const int64_t tokens = x->ne[1];
    const int64_t cond_tokens = context->ne[1];
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    auto slice = [&](ggml_tensor* a, int part, int64_t n) {
        return ggml_view_2d(ctx, a, width, n, a->nb[1],
                            (size_t)part * width * sizeof(float));
    };
    auto attend = [&](ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, int64_t nq) {
        ggml_tensor* o = nn::sdpa(ctx, sat_to_attn(ctx, q), sat_to_attn(ctx, k),
                                  sat_to_attn(ctx, v), nullptr, attn_scale);
        return sat_merge_heads(ctx, o, width, nq);
    };

    // Pre-norm self attention.  Rotary positions include the prepended global token.
    ggml_tensor* h = sat_layer_norm(ctx, x, W.get(p + "pre_norm.gamma"),
                                    W.get(p + "pre_norm.beta"), 1.0e-5f);
    ggml_tensor* qkv = nn::linear(ctx, W.get(p + "self.qkv.weight"), h);
    ggml_tensor* q = sat_heads(ctx, slice(qkv, 0, tokens), hd, heads, tokens);
    ggml_tensor* k = sat_heads(ctx, slice(qkv, 1, tokens), hd, heads, tokens);
    ggml_tensor* v = sat_heads(ctx, slice(qkv, 2, tokens), hd, heads, tokens);
    q = sat_qk_norm(ctx, W, p + "self.q_norm", q);
    k = sat_qk_norm(ctx, W, p + "self.k_norm", k);
    q = nn::rope_neox(ctx, q, pos, rotary_dims, 10000.0f);
    k = nn::rope_neox(ctx, k, pos, rotary_dims, 10000.0f);
    h = nn::linear(ctx, W.get(p + "self.out.weight"), attend(q, k, v, tokens));
    x = ggml_add(ctx, x, h);

    // Cross attention. Context is already projected to model width and has no RoPE.
    h = sat_layer_norm(ctx, x, W.get(p + "cross_norm.gamma"),
                       W.get(p + "cross_norm.beta"), 1.0e-5f);
    ggml_tensor* qf = nn::linear(ctx, W.get(p + "cross.q.weight"), h);
    ggml_tensor* kv = nn::linear(ctx, W.get(p + "cross.kv.weight"), context);
    q = sat_heads(ctx, qf, hd, heads, tokens);
    k = sat_heads(ctx, slice(kv, 0, cond_tokens), hd, heads, cond_tokens);
    v = sat_heads(ctx, slice(kv, 1, cond_tokens), hd, heads, cond_tokens);
    q = sat_qk_norm(ctx, W, p + "cross.q_norm", q);
    k = sat_qk_norm(ctx, W, p + "cross.k_norm", k);
    h = nn::linear(ctx, W.get(p + "cross.out.weight"), attend(q, k, v, tokens));
    x = ggml_add(ctx, x, h);

    // Pre-norm SwiGLU feed-forward: first half values, second half gates.
    h = sat_layer_norm(ctx, x, W.get(p + "ff_norm.gamma"),
                       W.get(p + "ff_norm.beta"), 1.0e-5f);
    h = nn::linear(ctx, W.get(p + "ff.in.weight"), h, W.get(p + "ff.in.bias"));
    const int inner = (int)h->ne[0] / 2;
    ggml_tensor* values = ggml_cont(ctx, ggml_view_2d(ctx, h, inner, tokens, h->nb[1], 0));
    ggml_tensor* gates = ggml_cont(ctx, ggml_view_2d(ctx, h, inner, tokens, h->nb[1],
                                                     (size_t)inner * sizeof(float)));
    h = ggml_mul(ctx, values, ggml_silu(ctx, gates));
    h = nn::linear(ctx, W.get(p + "ff.out.weight"), h, W.get(p + "ff.out.bias"));
    return ggml_add(ctx, x, h);
}

// x:[io,T], time:[1], cross:[cond_dim,C], global:[global_dim], pos:[T+1].
// Returns RF velocity [io,T].
inline ggml_tensor* classic_dit_forward(ggml_context* ctx, const GgufModel& W,
                                        ggml_tensor* x_in, ggml_tensor* time,
                                        ggml_tensor* cross, ggml_tensor* global,
                                        ggml_tensor* pos, const DitSpec& c) {
    const int64_t frames = x_in->ne[1];
    const int64_t tokens = frames + 1;
    const int rotary_dims = (int)W.u32("sat.dit.rotary_dims");

    ggml_tensor* freq = nn::linear(ctx, W.get("dit.time_fourier.weight"), time);
    freq = ggml_scale(ctx, freq, 6.2831853071795864769f);
    ggml_tensor* time_feat = ggml_concat(ctx, ggml_cos(ctx, freq), ggml_sin(ctx, freq), 0);
    ggml_tensor* time_embed = sat_mlp(ctx, W, "dit.time.", time_feat, true);
    ggml_tensor* global_embed = sat_mlp(ctx, W, "dit.global.", global, false);
    ggml_tensor* global_token = ggml_reshape_2d(ctx, ggml_add(ctx, global_embed, time_embed),
                                                c.embed_dim, 1);
    ggml_tensor* context = sat_mlp(ctx, W, "dit.cond.", cross, false);

    ggml_tensor* x = ggml_add(ctx, nn::linear(ctx, W.get("dit.pre.weight"), x_in), x_in);
    x = nn::linear(ctx, W.get("dit.in.weight"), x);
    x = ggml_concat(ctx, global_token, x, 1);
    for (int i = 0; i < c.depth; ++i)
        x = classic_dit_block(ctx, W, "dit.blocks." + std::to_string(i) + ".",
                              x, context, pos, c, rotary_dims);

    x = ggml_cont(ctx, ggml_view_2d(ctx, x, c.embed_dim, frames, x->nb[1], x->nb[1]));
    x = nn::linear(ctx, W.get("dit.out.weight"), x);
    return ggml_add(ctx, nn::linear(ctx, W.get("dit.post.weight"), x), x);
}

} // namespace sa3::sat
