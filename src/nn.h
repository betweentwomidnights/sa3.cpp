// nn.h — stateless GGML neural-net building blocks shared across the SA3 stack.
// Free functions over a ggml_context; no weights or state of their own.
#pragma once

#include "ggml.h"

namespace sa3::nn {

// DynamicTanh: y = tanh(alpha * x) * gamma + beta.
// alpha is [1] (broadcast); gamma/beta are [ne0] (broadcast over the rest).
inline ggml_tensor* dyt(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* alpha, ggml_tensor* gamma, ggml_tensor* beta) {
    ggml_tensor* y = ggml_tanh(ctx, ggml_mul(ctx, x, alpha));
    y = ggml_mul(ctx, y, gamma);
    y = ggml_add(ctx, y, beta);
    return y;
}

// RMSNorm with a learned gamma (no bias). Used by the DiT (Phase 4).
inline ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), gamma);
}

// Linear: w is [in,out] (ggml ne), x is [in,N] -> [out,N]; optional bias [out].
inline ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b = nullptr) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// Partial NeoX rotary embedding over the first n_dims of each head.
// a: [head_dim, n_head, seq]; pos: I32 [seq].
inline ggml_tensor* rope_neox(ggml_context* ctx, ggml_tensor* a, ggml_tensor* pos,
                              int n_dims, float base) {
    return ggml_rope_ext(ctx, a, pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX,
                         /*n_ctx_orig=*/0, base, /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
}

// Scaled-dot-product attention with an additive mask.
// q,k,v: [d, N, H]; mask: [Nk, Nq] (0 / -inf). Returns [d, Nq, H].
inline ggml_tensor* sdpa(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                         ggml_tensor* mask, float scale) {
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                 // [Nk, Nq, H]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);        // softmax over Nk
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [Nk, d, H]
    return ggml_mul_mat(ctx, vt, kq);                          // [d, Nq, H]
}

} // namespace sa3::nn
