// nn.h — stateless GGML neural-net building blocks shared across the SA3 stack.
// Free functions over a ggml_context; no weights or state of their own.
#pragma once

#include "ggml.h"

#include <cstdlib>
#include <cstring>

namespace sa3::nn {

inline bool flash_attn_enabled() {
    const char* p = getenv("SA3_FLASH_ATTN");
    return p && strcmp(p, "0") != 0;
}

inline int same_flash_attn_mode() {
    const char* p = getenv("SA3_SAME_FLASH_ATTN");
    if (!p) return flash_attn_enabled() ? 1 : 0; // legacy global flag: full-mask flash
    if (!strcmp(p, "0")) return 0;
    if (!strcmp(p, "local") || !strcmp(p, "window") || !strcmp(p, "compact")) return 2;
    return 1; // non-zero / "full"
}

inline bool same_flash_attn_enabled() {
    return same_flash_attn_mode() != 0;
}

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

// Block-diagonal self-attention: q,k,v [d, N, H] with N = cc*nb; each block of cc tokens
// attends only within itself. Mathematically identical to dense sdpa with a block-diagonal
// mask, but O(N*cc) instead of O(N^2) — the SAME-S (chunked) decoder/encoder path.
inline ggml_tensor* attn_blockdiag(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k,
                                   ggml_tensor* v, int cc, float scale) {
    const int64_t d = q->ne[0], N = q->ne[1], H = q->ne[2], nb = N / cc;
    auto blk = [&](ggml_tensor* t){ return ggml_reshape_4d(ctx, t, d, cc, nb, H); };   // [d, cc, nb, H]
    ggml_tensor* kq = ggml_mul_mat(ctx, blk(k), blk(q));               // [cc(k), cc(q), nb, H]
    kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);            // softmax over keys, no mask
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, blk(v), 1, 0, 2, 3)); // [cc, d, nb, H]
    ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                       // [d, cc, nb, H]
    return ggml_reshape_3d(ctx, o, d, N, H);
}

// Sliding-window (banded) attention via overlapping blocks: q,k,v [d, N, H], N = b*nb.
// Each block of b tokens attends to its 3-block neighborhood [i-1, i, i+1] (3b keys, edge-padded);
// `bias` [3b, b, nb] additively encodes the exact band + edge validity (built host-side). With
// b >= window this is identical to dense band sdpa, but O(N*3b) not O(N^2) — the SAME-L decoder.
inline ggml_tensor* attn_sliding(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                 int b, ggml_tensor* bias, float scale) {
    const int64_t d = q->ne[0], N = q->ne[1], H = q->ne[2], nb = N / b;
    auto blk = [&](ggml_tensor* t){ return ggml_reshape_4d(ctx, t, d, b, nb, H); };   // [d, b, nb, H]
    auto pad = [&](ggml_tensor* xb){                                                   // zero block each side -> [d,b,nb+2,H]
        ggml_tensor* z = ggml_scale(ctx, ggml_cont(ctx,
            ggml_view_4d(ctx, xb, d, b, 1, H, xb->nb[1], xb->nb[2], xb->nb[3], 0)), 0.0f);
        return ggml_concat(ctx, ggml_concat(ctx, z, xb, 2), z, 2);
    };
    auto neigh = [&](ggml_tensor* xp){                                                 // 3 block-slices -> [d, 3b, nb, H]
        auto sl = [&](int off){ return ggml_cont(ctx, ggml_view_4d(ctx, xp, d, b, nb, H,
            xp->nb[1], xp->nb[2], xp->nb[3], (size_t)off*xp->nb[2])); };
        return ggml_concat(ctx, ggml_concat(ctx, sl(0), sl(1), 1), sl(2), 1);
    };
    ggml_tensor* Kn = neigh(pad(blk(k)));                              // [d, 3b, nb, H]
    ggml_tensor* Vn = neigh(pad(blk(v)));
    ggml_tensor* kq = ggml_mul_mat(ctx, Kn, blk(q));                  // [3b, b, nb, H]
    kq = ggml_soft_max_ext(ctx, kq, bias, scale, 0.0f);              // fused scale+bias+softmax (bias [3b,b,nb])
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, Vn, 1, 0, 2, 3)); // [3b, d, nb, H]
    ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                      // [d, b, nb, H]
    return ggml_reshape_3d(ctx, o, d, N, H);
}

// Same compact sliding-window attention as attn_sliding(), but the per-block attention
// itself is computed by ggml_flash_attn_ext. q,k,v [d,N,H], bias [3b,b,nb] F16.
inline ggml_tensor* attn_sliding_flash_ext(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                           int b, ggml_tensor* bias, float scale) {
    const int64_t d = q->ne[0], N = q->ne[1], H = q->ne[2], nb = N / b;
    auto blk = [&](ggml_tensor* t){ return ggml_reshape_4d(ctx, t, d, b, nb, H); };   // [d, b, nb, H]
    auto pad = [&](ggml_tensor* xb){
        ggml_tensor* z = ggml_scale(ctx, ggml_cont(ctx,
            ggml_view_4d(ctx, xb, d, b, 1, H, xb->nb[1], xb->nb[2], xb->nb[3], 0)), 0.0f);
        return ggml_concat(ctx, ggml_concat(ctx, z, xb, 2), z, 2);
    };
    auto neigh = [&](ggml_tensor* xp){
        auto sl = [&](int off){ return ggml_cont(ctx, ggml_view_4d(ctx, xp, d, b, nb, H,
            xp->nb[1], xp->nb[2], xp->nb[3], (size_t)off*xp->nb[2])); };
        return ggml_concat(ctx, ggml_concat(ctx, sl(0), sl(1), 1), sl(2), 1);          // [d, 3b, nb, H]
    };

    ggml_tensor* out = ggml_flash_attn_ext(ctx, blk(q), neigh(pad(blk(k))), neigh(pad(blk(v))), bias, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);                                  // [d, nb, b, H]
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));                          // [d, b, nb, H]
    return ggml_reshape_3d(ctx, out, d, N, H);
}

// Scaled-dot-product attention with an additive mask.
// q,k,v: [d, N, H]; mask: [Nk, Nq] (0 / -inf). Returns [d, Nq, H].
inline ggml_tensor* sdpa_flash_ext(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                   ggml_tensor* mask, float scale) {
    ggml_tensor* out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)); // [d, Nq, H, 1]
    return ggml_reshape_3d(ctx, out, q->ne[0], q->ne[1], q->ne[2]);
}

inline ggml_tensor* sdpa(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                         ggml_tensor* mask, float scale) {
    if (flash_attn_enabled() && (!mask || mask->type == GGML_TYPE_F16)) {
        return sdpa_flash_ext(ctx, q, k, v, mask, scale);
    }
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                 // [Nk, Nq, H]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);        // softmax over Nk
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [Nk, d, H]
    return ggml_mul_mat(ctx, vt, kq);                          // [d, Nq, H]
}

} // namespace sa3::nn
