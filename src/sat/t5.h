// sat/t5.h -- classic T5 encoder used by stable-audio-tools conditioners.
//
// This is deliberately not folded into t5gemma.h: T5-base uses learned relative
// position bias, bias-free RMS norms and a ReLU feed-forward block, whereas T5Gemma
// uses RoPE, soft-capping, sandwich norms and GeGLU.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "nn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace sa3::sat {

struct T5EncoderConfig {
    int dim = 768;
    int layers = 12;
    int heads = 12;
    int head_dim = 64;
    int intermediate = 3072;
    int vocab = 32128;
    int relative_buckets = 32;
    int relative_max_distance = 128;
    float eps = 1.0e-6f;

    static T5EncoderConfig from(const GgufModel& m) {
        T5EncoderConfig c;
        c.dim                   = m.u32("sat.t5.dim");
        c.layers                = m.u32("sat.t5.layers");
        c.heads                 = m.u32("sat.t5.heads");
        c.head_dim              = m.u32("sat.t5.head_dim");
        c.intermediate          = m.u32("sat.t5.intermediate");
        c.vocab                 = m.u32("sat.t5.vocab");
        c.relative_buckets      = m.u32("sat.t5.relative_buckets");
        c.relative_max_distance = m.u32("sat.t5.relative_max_distance");
        c.eps                   = m.f32("sat.t5.eps");
        return c;
    }
};

// Port of transformers.T5Attention._relative_position_bucket for bidirectional
// encoder attention.  relative_position = key_position - query_position.
inline int t5_relative_position_bucket(int relative_position,
                                       int num_buckets = 32,
                                       int max_distance = 128) {
    const int half = num_buckets / 2;
    int n = -relative_position;
    int bucket = n < 0 ? half : 0;
    n = std::abs(n);
    const int max_exact = half / 2;
    if (n < max_exact) return bucket + n;
    const float denom = std::log((float)max_distance / (float)max_exact);
    int large = max_exact;
    if (denom > 0.0f) {
        large += (int)(std::log((float)n / (float)max_exact) / denom *
                       (float)(half - max_exact));
    }
    large = std::min(large, half - 1);
    return bucket + large;
}

// GGML I32 [key, query] contents for ggml_get_rows(relative_attention_bias, ids).
inline std::vector<int32_t> t5_relative_position_buckets(int seq,
                                                         int num_buckets = 32,
                                                         int max_distance = 128) {
    std::vector<int32_t> out((size_t)seq * seq);
    for (int q = 0; q < seq; ++q) {
        for (int k = 0; k < seq; ++k) {
            out[(size_t)q * seq + k] =
                t5_relative_position_bucket(k - q, num_buckets, max_distance);
        }
    }
    return out;
}

// Encode ids to last_hidden_state [dim, seq].
// mask is additive F32 [key, query] (zero for valid keys, -inf for padding).
// relative_buckets is flattened I32 [key * query], normally populated by the helper
// above; ggml_get_rows uses a 1-D index tensor here, then we restore the matrix shape.
inline ggml_tensor* t5_encode(ggml_context* ctx,
                              const GgufModel& W,
                              ggml_tensor* ids,
                              ggml_tensor* mask,
                              ggml_tensor* relative_buckets,
                              const T5EncoderConfig& c) {
    const int64_t seq = ids->ne[0];
    const int inner = c.heads * c.head_dim;

    ggml_tensor* x = ggml_get_rows(ctx, W.get("te.embed.weight"), ids); // [dim, seq]

    // Only block 0 owns the learned relative-attention table. Transformers computes
    // its position_bias once and passes the same tensor through all encoder blocks.
    ggml_tensor* position_bias =
        ggml_get_rows(ctx, W.get("te.relative_attention_bias.weight"), relative_buckets);
    position_bias = ggml_reshape_3d(ctx, position_bias, c.heads, seq, seq);
    // get_rows: [heads, key, query] -> attention layout [key, query, heads].
    position_bias = ggml_cont(ctx, ggml_permute(ctx, position_bias, 2, 0, 1, 3));
    position_bias = ggml_add(ctx, position_bias, mask);

    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "te." + std::to_string(l) + ".";

        ggml_tensor* residual = x;
        ggml_tensor* h = nn::rms_norm(ctx, x, W.get(p + "attn_norm.weight"), c.eps);
        ggml_tensor* q = nn::linear(ctx, W.get(p + "q.weight"), h);
        ggml_tensor* k = nn::linear(ctx, W.get(p + "k.weight"), h);
        ggml_tensor* v = nn::linear(ctx, W.get(p + "v.weight"), h);
        q = ggml_reshape_3d(ctx, q, c.head_dim, c.heads, seq);
        k = ggml_reshape_3d(ctx, k, c.head_dim, c.heads, seq);
        v = ggml_reshape_3d(ctx, v, c.head_dim, c.heads, seq);
        auto attention_layout = [&](ggml_tensor* a) {
            return ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3)); // [head_dim, seq, heads]
        };
        q = attention_layout(q);
        k = attention_layout(k);
        v = attention_layout(v);

        // T5 does not apply the usual 1/sqrt(head_dim) factor; its initialization
        // accounts for the scale.
        ggml_tensor* scores = ggml_mul_mat(ctx, k, q); // [key, query, heads]
        scores = ggml_soft_max(ctx, ggml_add(ctx, scores, position_bias));
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
        ggml_tensor* o = ggml_mul_mat(ctx, vt, scores); // [head_dim, query, heads]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
        o = ggml_reshape_2d(ctx, o, inner, seq);
        x = ggml_add(ctx, residual, nn::linear(ctx, W.get(p + "o.weight"), o));

        residual = x;
        h = nn::rms_norm(ctx, x, W.get(p + "ffn_norm.weight"), c.eps);
        h = ggml_relu(ctx, nn::linear(ctx, W.get(p + "wi.weight"), h));
        h = nn::linear(ctx, W.get(p + "wo.weight"), h);
        x = ggml_add(ctx, residual, h);
    }
    return nn::rms_norm(ctx, x, W.get("te.norm.weight"), c.eps);
}

} // namespace sa3::sat
