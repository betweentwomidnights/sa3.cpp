// dit.h — the Stable Audio 3 medium DiffusionTransformer (rf_denoiser / ARC) in GGML.
// Continuous transformer: project_in + 64 memory tokens, 24 blocks of
// [adaLN-zero -> differential self-attn (partial RoPE, RMS qk-norm)
//  -> differential cross-attn (to T5Gemma+seconds cond) -> adaLN-zero -> SwiGLU FF],
// then project_out, with zero-init residual 1x1 pre/post convs. Predicts velocity.
// Timestep ExpoFourierFeatures are computed host-side and passed as `t_feat`.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "nn.h"

#include <cmath>
#include <map>
#include <string>

namespace sa3 {

struct DitConfig {
    int io, dim, depth, heads, head_dim, cond_dim, mem_tokens, rot, time_dim, local_dim;
    float rope_base, norm_eps, qk_eps, time_min_freq, time_max_freq;
    bool differential;   // medium=true (5x qkv), small=false (3x qkv)

    static DitConfig from(const GgufModel& m) {
        DitConfig c;
        int dk = gguf_find_key(m.gguf, "dit.differential");
        c.differential = (dk < 0) ? true : (gguf_get_val_u32(m.gguf, dk) != 0);
        int lk = gguf_find_key(m.gguf, "dit.local_dim");
        c.local_dim = (lk < 0) ? 0 : (int)gguf_get_val_u32(m.gguf, lk);   // 257 inpaint, 0 if absent
        c.io         = m.u32("dit.io");
        c.dim        = m.u32("dit.dim");
        c.depth      = m.u32("dit.depth");
        c.heads      = m.u32("dit.heads");
        c.head_dim   = m.u32("dit.head_dim");
        c.cond_dim   = m.u32("dit.cond_dim");
        c.mem_tokens = m.u32("dit.mem_tokens");
        c.rot        = m.u32("dit.rot");
        c.time_dim   = m.u32("dit.time_dim");
        c.rope_base  = m.f32("dit.rope_base");
        c.norm_eps   = m.f32("dit.norm_eps");
        c.qk_eps     = m.f32("dit.qk_eps");
        c.time_min_freq = m.f32("dit.time_min_freq");
        c.time_max_freq = m.f32("dit.time_max_freq");
        return c;
    }
};

// Functional LoRA/DoRA application, used only by the training graph (see functional-lora-speed-plan).
// A DitLora maps a weight-tensor name -> its trainable adapter tensors. When `dl` is null (all
// inference callers) or has no entry for `name`, dit_lin is byte-identical to nn::linear:
// mul_mat(W,x) then optional bias. When an entry exists, the adapter is applied FUNCTIONALLY as
// small rank-sized matmuls, so autodiff never forms a full [in,out] weight gradient (the out_prod
// that made the materialized-effective-weight path ~27x slower than PyTorch).
struct DitLoraParam {
    ggml_tensor* A = nullptr;            // [in, rank]   trainable
    ggml_tensor* B = nullptr;            // [rank, out]  trainable
    ggml_tensor* magnitude = nullptr;    // dora-rows [out] trainable, else null
    ggml_tensor* base_norm_sq = nullptr; // dora-rows [out] constant (Σ_in W² + in·eps), else null
    float scale = 1.0f;                  // alpha / rank
    bool  dora = false;                  // apply dora-rows column normalization
    int64_t in = 0;                      // input dim (reference / eps bookkeeping)
};
using DitLora = std::map<std::string, DitLoraParam>;

// y = mul_mat(W,x) [+ functional adapter] [+ bias]. See DitLora above. `x` is [in, seq].
inline ggml_tensor* dit_lin(ggml_context* ctx, const GgufModel& W, const std::string& name,
                            ggml_tensor* x, ggml_tensor* bias, const DitLora* dl) {
    ggml_tensor* w = W.get(name);
    const DitLoraParam* p = nullptr;
    if (dl) {
        auto it = dl->find(name);
        if (it != dl->end()) p = &it->second;
    }
    // The frozen base stays in its stored precision even when adapted (training): the mul_mat
    // backward computes grad w.r.t. x via out_prod(src0=W, …), and out_prod handles f16 src0 by
    // converting per-op into a transient buffer. An explicit in-graph f32 cast here would instead
    // stay live from its forward use until its backward use — for ~228 adapted targets that held a
    // full f32 copy of the DiT (~5.8 GB) resident mid-graph and pushed the training working set
    // past VRAM into driver sysmem paging (the 25x slowdown).
    ggml_tensor* wl = w;
    ggml_tensor* y = ggml_mul_mat(ctx, wl, x);                 // [out, seq]
    if (p) {
        // LoRA applied functionally: y += scale · B @ (A @ x)  (== mul_mat(W + s·A@B, x)).
        ggml_tensor* ax  = ggml_mul_mat(ctx, p->A, x);          // [rank, seq]
        ggml_tensor* bax = ggml_mul_mat(ctx, p->B, ax);         // [out, seq]
        y = ggml_add(ctx, y, ggml_scale(ctx, bax, p->scale));
        if (p->dora) {
            // dora-rows: multiply each output row by magnitude[out] / ||(W+s·A@B)_col[out]||.
            // The per-out scalar factors out of the matmul, so it scales the LoRA output above.
            // norm_sq[out] = base_norm_sq + 2s·term2 + s²·term3, with (over rank r):
            //   term2[o] = Σ_r (Wᵀ A)[o,r] · B[r,o]
            //   term3[o] = Σ_r B[r,o] · (AᵀA · B)[r,o]
            // computed without ever materializing W + s·A@B.
            const float s = p->scale;
            const int64_t out = y->ne[0];
            ggml_tensor* WtA = ggml_mul_mat(ctx, wl, p->A);                          // [out, rank]
            ggml_tensor* Bt  = ggml_cont(ctx, ggml_transpose(ctx, p->B));            // [out, rank]
            ggml_tensor* t2  = ggml_sum_rows(ctx,
                ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, WtA, Bt))));        // [1, out]
            ggml_tensor* AtA  = ggml_mul_mat(ctx, p->A, p->A);                        // [rank, rank]
            ggml_tensor* AtAB = ggml_mul_mat(ctx, AtA, p->B);                         // [rank, out]
            ggml_tensor* t3   = ggml_sum_rows(ctx, ggml_mul(ctx, p->B, AtAB));        // [1, out]
            ggml_tensor* nsq = ggml_add(ctx,
                ggml_add(ctx, p->base_norm_sq,
                              ggml_scale(ctx, ggml_reshape_1d(ctx, t2, out), 2.0f * s)),
                ggml_scale(ctx, ggml_reshape_1d(ctx, t3, out), s * s));              // [out]
            ggml_tensor* scale_vec = ggml_div(ctx, p->magnitude, ggml_sqrt(ctx, nsq)); // [out]
            y = ggml_mul(ctx, y, scale_vec);                                          // broadcast over seq
        }
    }
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

// dual-softmax differential attention; q,k,v,qd,kd already in [head_dim, S, n_head].
inline ggml_tensor* dit_diff_attn(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                  ggml_tensor* qd, ggml_tensor* kd, int dim, float scale) {
    ggml_tensor* o = ggml_sub(ctx, nn::sdpa(ctx, q, k, v, nullptr, scale),
                                   nn::sdpa(ctx, qd, kd, v, nullptr, scale)); // [hd, Sq, nh]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                     // [hd, nh, Sq]
    return ggml_reshape_2d(ctx, o, dim, o->ne[2]);
}

// One DiT block. x:[dim,S]; context:[dim,Ctx]; gcond:[6*dim] adaLN signal; ones:[1]=1.0.
// local_cond:[local_dim,T] (inpaint) or nullptr — projected per-block and added to the T real tokens.
// autodiff_safe expands broadcasts and view operands for ggml backward; keep it false for inference
// so generation retains the original, validated graph and its backend fusion/numerical behavior.
inline ggml_tensor* dit_block(ggml_context* ctx, const GgufModel& W, const std::string& p,
                              ggml_tensor* x, ggml_tensor* context, ggml_tensor* gcond,
                              ggml_tensor* pos, ggml_tensor* ones, const DitConfig& c,
                              ggml_tensor* local_cond = nullptr, const DitLora* dl = nullptr,
                              bool autodiff_safe = false) {
    const int dim = c.dim, hd = c.head_dim, nh = c.heads;
    const int64_t S = x->ne[1], Ctx = context->ne[1];
    const float scale = 1.0f / sqrtf((float)hd);

    // adaLN-zero: (per-layer ssg) + gcond, split into 6 modulation vectors [dim]
    ggml_tensor* ssg = ggml_add(ctx, W.get(p+"ssg"), gcond);     // [6*dim]
    auto mod = [&](int i){ return ggml_view_1d(ctx, ssg, dim, (size_t)i*dim*sizeof(float)); };
    ggml_tensor *sc_a=mod(0),*sh_a=mod(1),*g_a=mod(2),*sc_f=mod(3),*sh_f=mod(4),*g_f=mod(5);
    auto modulate = [&](ggml_tensor* h, ggml_tensor* sc, ggml_tensor* sh){
        if (!autodiff_safe)
            return ggml_add(ctx, ggml_mul(ctx, h, ggml_add(ctx, sc, ones)), sh);
        ggml_tensor* onev = ggml_repeat(ctx, ones, sc);
        ggml_tensor* scale = ggml_repeat(ctx, ggml_add(ctx, sc, onev), h);
        ggml_tensor* shift = ggml_repeat(ctx, sh, h);
        ggml_tensor* hc = ggml_cont(ctx, h);
        return ggml_add(ctx, ggml_mul(ctx, hc, scale), shift);  // h*(1+sc)+sh
    };
    auto gate = [&](ggml_tensor* h, ggml_tensor* g){
        if (!autodiff_safe)
            return ggml_mul(ctx, h, ggml_sigmoid(ctx, ggml_add(ctx, ggml_neg(ctx, g), ones)));
        ggml_tensor* onev = ggml_repeat(ctx, ones, g);
        ggml_tensor* denom = ggml_add(ctx, ggml_exp(ctx, ggml_sub(ctx, g, onev)), onev);
        ggml_tensor* hc = ggml_cont(ctx, h);
        return ggml_div(ctx, hc, ggml_repeat(ctx, denom, hc)); // *sigmoid(1-g)
    };
    auto heads = [&](ggml_tensor* a, int64_t n){ return ggml_reshape_3d(ctx, ggml_cont(ctx, a), hd, nh, n); };
    auto qknorm = [&](ggml_tensor* a, const std::string& g){ return nn::rms_norm(ctx, a, W.get(g), c.qk_eps); };
    auto toAttn = [&](ggml_tensor* a){ return ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3)); };

    auto sl = [&](ggml_tensor* t, int i){ return ggml_view_2d(ctx, t, dim, t->ne[1], t->nb[1], (size_t)i*dim*sizeof(float)); };
    auto single_attn = [&](ggml_tensor* qa, ggml_tensor* ka, ggml_tensor* va){  // q,k,v [hd,seq,nh]
        ggml_tensor* o = nn::sdpa(ctx, qa, ka, va, nullptr, scale);
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
        return ggml_reshape_2d(ctx, o, dim, o->ne[2]);
    };

    // --- self-attention with adaLN + partial RoPE (differential or standard) ---
    ggml_tensor* res = x;
    ggml_tensor* h = modulate(nn::rms_norm(ctx, x, W.get(p+"pre_norm.gamma"), c.norm_eps), sc_a, sh_a);
    ggml_tensor* qkv = dit_lin(ctx, W, p+"self.qkv.weight", h, nullptr, dl); // [3 or 5 *dim, S]
    ggml_tensor* q = qknorm(heads(sl(qkv,0),S), p+"self.q_norm.gamma");
    ggml_tensor* k = qknorm(heads(sl(qkv,1),S), p+"self.k_norm.gamma");
    ggml_tensor* v = heads(sl(qkv,2),S);
    q = nn::rope_neox(ctx,q,pos,c.rot,c.rope_base); k = nn::rope_neox(ctx,k,pos,c.rot,c.rope_base);
    ggml_tensor* o;
    if (c.differential) {
        ggml_tensor* qd = qknorm(heads(sl(qkv,3),S), p+"self.q_norm.gamma");
        ggml_tensor* kd = qknorm(heads(sl(qkv,4),S), p+"self.k_norm.gamma");
        qd = nn::rope_neox(ctx,qd,pos,c.rot,c.rope_base); kd = nn::rope_neox(ctx,kd,pos,c.rot,c.rope_base);
        o = dit_diff_attn(ctx, toAttn(q),toAttn(k),toAttn(v),toAttn(qd),toAttn(kd), dim, scale);
    } else {
        o = single_attn(toAttn(q), toAttn(k), toAttn(v));
    }
    o = dit_lin(ctx, W, p+"self.out.weight", o, nullptr, dl);
    x = ggml_add(ctx, autodiff_safe ? ggml_cont(ctx, res) : res, gate(o, g_a));

    // --- cross-attention (no adaLN, no RoPE), kv from context ---
    ggml_tensor* hc = nn::rms_norm(ctx, x, W.get(p+"cross_norm.gamma"), c.norm_eps);
    ggml_tensor* qf = dit_lin(ctx, W, p+"cross.q.weight", hc, nullptr, dl);   // [1 or 2 *dim, S]
    ggml_tensor* kv = dit_lin(ctx, W, p+"cross.kv.weight", context, nullptr, dl);// [2 or 3 *dim, Ctx]
    ggml_tensor* cq = qknorm(heads(sl(qf,0),S), p+"cross.q_norm.gamma");
    ggml_tensor* ck = qknorm(heads(sl(kv,0),Ctx), p+"cross.k_norm.gamma");
    ggml_tensor* oc;
    if (c.differential) {
        ggml_tensor* cqd = qknorm(heads(sl(qf,1),S),   p+"cross.q_norm.gamma");
        ggml_tensor* ckd = qknorm(heads(sl(kv,1),Ctx), p+"cross.k_norm.gamma");
        ggml_tensor* cv  = heads(sl(kv,2),Ctx);
        oc = dit_diff_attn(ctx, toAttn(cq),toAttn(ck),toAttn(cv),toAttn(cqd),toAttn(ckd), dim, scale);
    } else {
        ggml_tensor* cv = heads(sl(kv,1),Ctx);
        oc = single_attn(toAttn(cq), toAttn(ck), toAttn(cv));
    }
    oc = dit_lin(ctx, W, p+"cross.out.weight", oc, nullptr, dl);
    x = ggml_add(ctx, autodiff_safe ? ggml_cont(ctx, x) : x, oc);

    // --- inpaint local conditioning (per block, after cross-attn): add to the T real tokens only ---
    // le = Linear(local_dim->dim) -> SiLU -> Linear(dim->dim); memory tokens (first mem_tokens) get +0.
    if (local_cond) {
        ggml_tensor* le = dit_lin(ctx, W, p+"local.0.weight", local_cond, W.get(p+"local.0.bias"), dl); // [dim,T]
        le = ggml_silu(ctx, le);
        le = dit_lin(ctx, W, p+"local.2.weight", le, W.get(p+"local.2.bias"), dl);                      // [dim,T]
        const int64_t Tt = local_cond->ne[1];
        ggml_tensor* xm = ggml_cont(ctx, ggml_view_2d(ctx, x, dim, c.mem_tokens, x->nb[1], 0));                    // [dim,mem]
        ggml_tensor* xt = ggml_cont(ctx, ggml_view_2d(ctx, x, dim, Tt, x->nb[1], (size_t)c.mem_tokens*x->nb[1]));  // [dim,T]
        x = ggml_concat(ctx, xm, ggml_add(ctx, autodiff_safe ? ggml_cont(ctx, xt) : xt, le), 1);                 // [dim,mem+T]
    }

    // --- SwiGLU feed-forward with adaLN ---
    res = x;
    ggml_tensor* f = modulate(nn::rms_norm(ctx, x, W.get(p+"ff_norm.gamma"), c.norm_eps), sc_f, sh_f);
    f = dit_lin(ctx, W, p+"ff.proj.weight", f, W.get(p+"ff.proj.bias"), dl); // [2*inner, S]
    const int inner = f->ne[0] / 2;
    ggml_tensor* val  = ggml_view_2d(ctx, f, inner, S, f->nb[1], 0);
    ggml_tensor* glu  = ggml_view_2d(ctx, f, inner, S, f->nb[1], (size_t)inner*sizeof(float));
    f = ggml_mul(ctx, ggml_cont(ctx, val), ggml_silu(ctx, ggml_cont(ctx, glu)));
    f = dit_lin(ctx, W, p+"ff.out.weight", f, W.get(p+"ff.out.bias"), dl);
    return ggml_add(ctx, autodiff_safe ? ggml_cont(ctx, res) : res, gate(f, g_f));
}

// MLP: linear(0) -> SiLU -> linear(2), optional biases. prefix like "dit.time_embed."
inline ggml_tensor* mlp_silu(ggml_context* ctx, const GgufModel& W, const std::string& p,
                             ggml_tensor* x, bool bias, const DitLora* dl = nullptr) {
    ggml_tensor* h = dit_lin(ctx, W, p+"0.weight", x, bias ? W.get(p+"0.bias") : nullptr, dl);
    h = ggml_silu(ctx, h);
    return dit_lin(ctx, W, p+"2.weight", h, bias ? W.get(p+"2.bias") : nullptr, dl);
}

// Head of the DiT (everything before the block stack): conditioning signals + x projection +
// memory-token concat. x0 is the block-0 input [dim, mem+T]; context [dim,Ctx] and gcond [6*dim]
// are the shared tensors every block consumes. Split out of dit_forward so the checkpointed
// training backward can rebuild the head as its own graph; inference behavior is unchanged.
struct DitHeadOut {
    ggml_tensor* x0 = nullptr;
    ggml_tensor* context = nullptr;
    ggml_tensor* gcond = nullptr;
};

inline DitHeadOut dit_head(ggml_context* ctx, const GgufModel& W, ggml_tensor* x_in,
                           ggml_tensor* t_feat, ggml_tensor* cross, ggml_tensor* glob,
                           const DitConfig& c, const DitLora* dl = nullptr,
                           bool autodiff_safe = false) {
    DitHeadOut h;
    // conditioning signals
    ggml_tensor* te = mlp_silu(ctx, W, "dit.time_embed.", t_feat, true, dl);      // [dim]
    ggml_tensor* g  = mlp_silu(ctx, W, "dit.global_embed.", glob, false, dl);     // [dim]
    g = ggml_add(ctx, g, te);
    h.gcond = mlp_silu(ctx, W, "dit.gce.", g, true, dl);                          // [6*dim]
    h.context = mlp_silu(ctx, W, "dit.cond_embed.", cross, false, dl);            // [dim, Ctx]

    // x path: residual pre-conv, project in, prepend memory tokens
    ggml_tensor* x = ggml_add(ctx, dit_lin(ctx, W, "dit.pre_conv.weight", x_in, nullptr, dl), x_in); // [io,T]
    x = dit_lin(ctx, W, "dit.proj_in.weight", x, nullptr, dl);                    // [dim, T]
    if (!autodiff_safe) {
        h.x0 = ggml_concat(ctx, W.get("dit.memory_tokens"), x, 1);
        return h;
    }
    ggml_tensor* full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, c.mem_tokens + x->ne[1]);
    full = ggml_set(ctx, full, W.get("dit.memory_tokens"), full->nb[1], full->nb[2], full->nb[3], 0);
    x = ggml_set(ctx, full, x, full->nb[1], full->nb[2], full->nb[3], (size_t)c.mem_tokens * full->nb[1]); // [dim, mem+T]
    h.x0 = ggml_cont(ctx, x);
    return h;
}

// Tail of the DiT (everything after the block stack): drop memory tokens, project out,
// residual post-conv. x is the final block output [dim, mem+T]; returns velocity [io, T].
inline ggml_tensor* dit_tail(ggml_context* ctx, const GgufModel& W, ggml_tensor* x,
                             const DitConfig& c, const DitLora* dl = nullptr,
                             bool autodiff_safe = false) {
    const int64_t T = x->ne[1] - c.mem_tokens;
    x = ggml_cont(ctx, ggml_view_2d(ctx, x, c.dim, T, x->nb[1], (size_t)c.mem_tokens * x->nb[1])); // [dim,T]
    x = dit_lin(ctx, W, "dit.proj_out.weight", x, nullptr, dl);                   // [io, T]
    x = ggml_add(ctx, dit_lin(ctx, W, "dit.post_conv.weight", x, nullptr, dl),
                 autodiff_safe ? ggml_cont(ctx, x) : x);
    return x;
}

// Full DiT forward. x:[io,T]; t_feat:[time_dim]; cross:[cond_dim,Ctx]; glob:[cond_dim];
// pos:[mem+T]; ones:[1]=1.0. local_cond:[local_dim,T] for inpaint, or nullptr. Returns velocity [io, T].
inline ggml_tensor* dit_forward(ggml_context* ctx, const GgufModel& W, ggml_tensor* x_in,
                                ggml_tensor* t_feat, ggml_tensor* cross, ggml_tensor* glob,
                                ggml_tensor* pos, ggml_tensor* ones, const DitConfig& c,
                                ggml_tensor* local_cond = nullptr, const DitLora* dl = nullptr,
                                bool autodiff_safe = false) {
    DitHeadOut h = dit_head(ctx, W, x_in, t_feat, cross, glob, c, dl, autodiff_safe);
    ggml_tensor* x = h.x0;
    for (int l = 0; l < c.depth; l++)
        x = dit_block(ctx, W, "dit." + std::to_string(l) + ".", x, h.context, h.gcond, pos, ones, c,
                      local_cond, dl, autodiff_safe);
    return dit_tail(ctx, W, x, c, dl, autodiff_safe);
}

} // namespace sa3
