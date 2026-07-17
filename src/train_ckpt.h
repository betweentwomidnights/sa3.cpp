// train_ckpt.h - gradient-checkpointed DiT training graphs (per-block backward).
//
// The monolithic fwd+bwd graph keeps every forward activation alive until its backward consumer
// runs: ~9.6 GB of gallocr buffer at 512 frames, far past the 8 GB card this trains on, so the
// driver pages the buffer to system RAM and every kernel crawls (the diagnosed ~25x slowdown —
// see functional-lora-speed-plan). Checkpointing decomposes the step:
//
//   F    forward-only graph. Runs head + all blocks once, copying each block input x_l
//        (depth+1 tensors of [dim, mem+frames], ~90 MB total) plus the shared context/gcond
//        into persistent tensors.
//   T    tail graph (proj_out/post_conv + the real loss), fwd+bwd. Emits dL/dx_depth.
//   B_l  one graph per block, l = depth-1 .. 0, all sharing ONE gallocr. Recomputes block l
//        forward from the stored x_l and backprops the incoming gradient G with the VJP trick:
//        loss_l = sum(block_out * G) is a scalar whose gradient w.r.t. any upstream tensor is
//        exactly G^T * Jacobian (the sum/mul backward reproduces G bit-for-bit as the seed).
//        Emits the block's adapter grads, dL/dx_l (ping-ponged into the next graph's G input),
//        and the block's contribution to dL/dcontext and dL/dgcond (accumulated host-side —
//        those two are consumed by every block).
//   H    head graph, fwd+bwd, closing the chain with a 3-term VJP against dL/dx_0, dL/dcontext
//        and dL/dgcond. Emits the head adapter grads.
//
// Peak activation memory = one block's working set instead of 24, so the whole step stays
// VRAM-resident; the price is one extra forward pass. All step inputs, adapter tensors and
// boundary/grad carriers live in ONE persistent backend buffer that no gallocr ever touches
// (this also retires the "re-upload base_norm_sq every step" workaround — constants uploaded at
// build survive). Functional adapter families only (lora / dora-rows); others use the
// monolithic path in train_dit.h.
#pragma once

#include "dit.h"
#include "gguf_model.h"
#include "train_dit.h"
#include "train_lora.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <string>
#include <vector>

namespace sa3 {

// Segment owning a LoRA target stem: 0..depth-1 = that block, -1 = head, -2 = tail.
inline int train_ckpt_segment(const std::string& stem, int depth) {
    if (stem == "dit.proj_out" || stem == "dit.post_conv") return -2;
    if (stem.rfind("dit.", 0) == 0) {
        const size_t p = 4;
        const size_t q = stem.find('.', p);
        if (q != std::string::npos && q > p) {
            bool digits = true;
            for (size_t i = p; i < q; ++i) digits = digits && stem[i] >= '0' && stem[i] <= '9';
            if (digits) {
                const int l = std::atoi(stem.substr(p, q - p).c_str());
                if (l >= 0 && l < depth) return l;
            }
        }
    }
    return -1;  // time_embed / global_embed / gce / cond_embed / pre_conv / proj_in
}

struct TrainCkptBlock {
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* grad_ctx = nullptr;    // this block's dL/dcontext contribution [dim, Ctx]
    ggml_tensor* grad_gcond = nullptr;  // this block's dL/dgcond contribution [6*dim]
    std::vector<size_t> param_idx;      // indices into lora.params / TrainDitCkpt::params
};

struct TrainDitCkpt {
    // --- persistent tensors (one backend buffer; uploads survive all graph allocations) ---
    ggml_context* pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;
    ggml_tensor* x = nullptr;           // [io, frames] noisy latent input
    ggml_tensor* tfeat = nullptr;       // [time_dim]
    ggml_tensor* cross = nullptr;       // [cond_dim, ctx_len]
    ggml_tensor* global = nullptr;      // [cond_dim]
    ggml_tensor* pos = nullptr;         // [mem+frames] i32 iota (constant, uploaded at build)
    ggml_tensor* ones = nullptr;        // [1] = 1.0 (constant)
    ggml_tensor* target = nullptr;      // [io, frames]
    ggml_tensor* local = nullptr;       // [local_dim, frames] inpaint local-add cond, or null
    ggml_tensor* loss_weight = nullptr; // [io, frames] inpaint loss weight, or null
    std::vector<ggml_tensor*> xb;       // depth+1 block boundaries [dim, mem+frames]
    ggml_tensor* context_p = nullptr;   // [dim, ctx_len] shared cross-attn context (copied by F)
    ggml_tensor* gcond_p = nullptr;     // [6*dim] shared adaLN signal (copied by F)
    ggml_tensor* G_a = nullptr;         // ping-pong dL/dx carriers [dim, mem+frames]
    ggml_tensor* G_b = nullptr;
    ggml_tensor* Gctx_in = nullptr;     // [dim, ctx_len] summed dL/dcontext (uploaded before H)
    ggml_tensor* Ggcond_in = nullptr;   // [6*dim] summed dL/dgcond
    std::vector<TrainDitParamTensors> params;  // adapter tensors, indexed like lora.params
    DitLora dl;

    // --- graphs ---
    ggml_context* fctx = nullptr;       // F: forward + boundary copies
    ggml_gallocr_t falloc = nullptr;
    ggml_cgraph* fgraph = nullptr;
    ggml_context* tctx = nullptr;       // T: tail + real loss, fwd+bwd
    ggml_gallocr_t talloc = nullptr;
    ggml_cgraph* tgraph = nullptr;
    ggml_tensor* velocity = nullptr;
    ggml_tensor* loss = nullptr;
    std::vector<size_t> tail_param_idx;
    ggml_gallocr_t balloc = nullptr;    // shared by all block graphs (re-alloc'd per use)
    std::vector<TrainCkptBlock> blocks;
    ggml_context* hctx = nullptr;       // H: head VJP, fwd+bwd
    ggml_gallocr_t halloc = nullptr;
    ggml_cgraph* hgraph = nullptr;
    std::vector<size_t> head_param_idx;

    int depth = 0, frames = 0, cond_dim = 0, ctx_len = 0;

    // dL/dx carrier feeding block l's graph (T writes g_in(depth-1); B_l writes g_in(l-1) == g_out(l)).
    ggml_tensor* g_in(int l) const { return ((depth - 1 - l) % 2 == 0) ? G_a : G_b; }
    ggml_tensor* g_out(int l) const { return ((depth - 1 - l) % 2 == 0) ? G_b : G_a; }
};

inline void free_train_dit_ckpt(TrainDitCkpt& ck) {
    if (ck.falloc) ggml_gallocr_free(ck.falloc);
    if (ck.talloc) ggml_gallocr_free(ck.talloc);
    if (ck.balloc) ggml_gallocr_free(ck.balloc);
    if (ck.halloc) ggml_gallocr_free(ck.halloc);
    for (TrainCkptBlock& b : ck.blocks) if (b.ctx) ggml_free(b.ctx);
    if (ck.fctx) ggml_free(ck.fctx);
    if (ck.tctx) ggml_free(ck.tctx);
    if (ck.hctx) ggml_free(ck.hctx);
    if (ck.pbuf) ggml_backend_buffer_free(ck.pbuf);
    if (ck.pctx) ggml_free(ck.pctx);
    ck = TrainDitCkpt{};
}

inline bool build_train_dit_ckpt(GgufModel& dit, const DitConfig& dc, const TrainLoraState& lora,
                                 int frames, int cond_dim, int ctx_len,
                                 TrainDitCkpt& out, std::string& err, bool inpaint = false) {
    if (lora.adapter_type != "lora" && lora.adapter_type != "dora-rows") {
        err = "checkpointed backward supports functional families only (lora, dora-rows)";
        return false;
    }
    if (frames <= 0 || cond_dim <= 0 || ctx_len <= 0) {
        err = "invalid DiT training graph dimensions";
        return false;
    }
    if (inpaint && dc.local_dim <= 0) {
        err = "inpainting requested but DiT has no local-cond weights (dit.local_dim <= 0)";
        return false;
    }
    out = TrainDitCkpt{};
    out.depth = dc.depth;
    out.frames = frames;
    out.cond_dim = cond_dim;
    out.ctx_len = ctx_len;
    const int64_t S = (int64_t)dc.mem_tokens + frames;
    const bool dora = lora.adapter_type == "dora-rows";
    auto fail = [&](const std::string& msg) { err = msg; free_train_dit_ckpt(out); return false; };

    // --- persistent tensors ---
    {
        const size_t n_tensors = 24 + (size_t)(dc.depth + 1) + lora.params.size() * 4;
        ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 64), nullptr, true };
        out.pctx = ggml_init(ip);
        if (!out.pctx) return fail("ggml_init failed for persistent training tensors");
        out.x = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.io, frames);
        out.tfeat = ggml_new_tensor_1d(out.pctx, GGML_TYPE_F32, dc.time_dim);
        out.cross = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, cond_dim, ctx_len);
        out.global = ggml_new_tensor_1d(out.pctx, GGML_TYPE_F32, cond_dim);
        out.pos = ggml_new_tensor_1d(out.pctx, GGML_TYPE_I32, S);
        out.ones = ggml_new_tensor_1d(out.pctx, GGML_TYPE_F32, 1);
        out.target = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.io, frames);
        if (inpaint) {
            out.local = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.local_dim, frames);
            out.loss_weight = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.io, frames);
        }
        out.xb.resize((size_t)dc.depth + 1);
        for (ggml_tensor*& t : out.xb) {
            t = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.dim, S);
            ggml_set_param(t);   // every boundary's grad is needed (dL/dx_l feeds the next graph)
        }
        out.context_p = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.dim, ctx_len);
        out.gcond_p = ggml_new_tensor_1d(out.pctx, GGML_TYPE_F32, 6 * (int64_t)dc.dim);
        ggml_set_param(out.context_p);
        ggml_set_param(out.gcond_p);
        out.G_a = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.dim, S);
        out.G_b = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.dim, S);
        out.Gctx_in = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, dc.dim, ctx_len);
        out.Ggcond_in = ggml_new_tensor_1d(out.pctx, GGML_TYPE_F32, 6 * (int64_t)dc.dim);

        for (const TrainLoraParam& hp : lora.params) {
            TrainDitParamTensors tp;
            tp.stem = hp.target.stem;
            tp.lora_A = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, hp.target.in, lora.rank);
            tp.lora_B = ggml_new_tensor_2d(out.pctx, GGML_TYPE_F32, lora.rank, hp.target.out);
            ggml_set_param(tp.lora_A);
            ggml_set_param(tp.lora_B);
            DitLoraParam dp;
            dp.A = tp.lora_A;
            dp.B = tp.lora_B;
            dp.scale = lora.alpha / (float)lora.rank;
            dp.in = hp.target.in;
            if (dora) {
                tp.magnitude = ggml_new_tensor_1d(out.pctx, GGML_TYPE_F32, (int64_t)hp.magnitude.size());
                ggml_set_param(tp.magnitude);
                tp.base_norm_sq = ggml_new_tensor_1d(out.pctx, GGML_TYPE_F32, hp.target.out);
                tp.base_norm_sq_host = train_row_norm_sq(dit.get(hp.target.weight_name), hp.target.in,
                                                         hp.target.out, 1e-12f);
                dp.dora = true;
                dp.magnitude = tp.magnitude;
                dp.base_norm_sq = tp.base_norm_sq;
            }
            out.dl[hp.target.weight_name] = dp;
            out.params.push_back(tp);
        }

        out.pbuf = ggml_backend_alloc_ctx_tensors(out.pctx, dit.backend);
        if (!out.pbuf) return fail("failed to allocate persistent training tensors");

        // constants: uploaded once — the persistent buffer is never re-planned by a gallocr
        std::vector<int32_t> pos_host((size_t)S);
        for (size_t i = 0; i < pos_host.size(); ++i) pos_host[i] = (int32_t)i;
        ggml_backend_tensor_set(out.pos, pos_host.data(), 0, pos_host.size() * sizeof(int32_t));
        const float one = 1.0f;
        ggml_backend_tensor_set(out.ones, &one, 0, sizeof(float));
        for (const TrainDitParamTensors& tp : out.params) {
            if (tp.base_norm_sq && !tp.base_norm_sq_host.empty())
                ggml_backend_tensor_set(tp.base_norm_sq, tp.base_norm_sq_host.data(), 0,
                                        tp.base_norm_sq_host.size() * sizeof(float));
        }
    }

    // segment ownership of every adapter target
    std::vector<std::vector<size_t>> block_param_idx((size_t)dc.depth);
    for (size_t i = 0; i < lora.params.size(); ++i) {
        const int seg = train_ckpt_segment(lora.params[i].target.stem, dc.depth);
        if (seg >= 0) block_param_idx[(size_t)seg].push_back(i);
        else if (seg == -2) out.tail_param_idx.push_back(i);
        else out.head_param_idx.push_back(i);
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(dit.backend);
    auto graph_ctx = [&](size_t graph_size, bool grads) -> ggml_context* {
        ggml_init_params ip = { ggml_tensor_overhead() * graph_size * 3 +
                                ggml_graph_overhead_custom(graph_size, grads), nullptr, true };
        return ggml_init(ip);
    };
    // Keep the grads we read after compute alive through the trailing cpy nodes: without the
    // output flag the graph allocator may hand a grad's buffer to a later node before we read it.
    auto keep_param_grads = [&](ggml_cgraph* g, const std::vector<size_t>& idx) {
        for (size_t i : idx) {
            const TrainDitParamTensors& tp = out.params[i];
            for (ggml_tensor* p : { tp.lora_A, tp.lora_B, tp.magnitude }) {
                if (!p) continue;
                ggml_tensor* gr = ggml_graph_get_grad(g, p);
                if (gr) ggml_set_output(gr);
            }
        }
    };

    // --- F: forward + boundary copies ---
    {
        out.fctx = graph_ctx(16384, false);
        if (!out.fctx) return fail("ggml_init failed for forward graph");
        out.fgraph = ggml_new_graph_custom(out.fctx, 16384, false);
        DitHeadOut h = dit_head(out.fctx, dit, out.x, out.tfeat, out.cross, out.global, dc, &out.dl, true);
        ggml_build_forward_expand(out.fgraph, ggml_cpy(out.fctx, h.context, out.context_p));
        ggml_build_forward_expand(out.fgraph, ggml_cpy(out.fctx, h.gcond, out.gcond_p));
        ggml_tensor* xc = h.x0;
        ggml_build_forward_expand(out.fgraph, ggml_cpy(out.fctx, xc, out.xb[0]));
        for (int l = 0; l < dc.depth; ++l) {
            xc = dit_block(out.fctx, dit, "dit." + std::to_string(l) + ".", xc, h.context, h.gcond,
                           out.pos, out.ones, dc, out.local, &out.dl, true);
            ggml_build_forward_expand(out.fgraph, ggml_cpy(out.fctx, xc, out.xb[(size_t)l + 1]));
        }
        out.falloc = ggml_gallocr_new(buft);
        if (!out.falloc || !ggml_gallocr_alloc_graph(out.falloc, out.fgraph))
            return fail("failed to allocate checkpointed forward graph");
    }

    // --- T: tail + real loss, fwd+bwd; emits dL/dx_depth ---
    {
        out.tctx = graph_ctx(2048, true);
        if (!out.tctx) return fail("ggml_init failed for tail graph");
        out.tgraph = ggml_new_graph_custom(out.tctx, 2048, true);
        ggml_tensor* vel = ggml_cont(out.tctx, dit_tail(out.tctx, dit, out.xb[(size_t)dc.depth], dc, &out.dl, true));
        out.velocity = vel;
        ggml_set_output(vel);
        ggml_tensor* sq = ggml_sqr(out.tctx, ggml_sub(out.tctx, vel, out.target));
        out.loss = inpaint ? ggml_sum(out.tctx, ggml_mul(out.tctx, sq, out.loss_weight))
                           : ggml_scale(out.tctx, ggml_sum(out.tctx, sq), 1.0f / (float)(dc.io * frames));
        ggml_set_loss(out.loss);
        ggml_set_output(out.loss);
        ggml_build_forward_expand(out.tgraph, out.loss);
        ggml_build_backward_expand(out.tctx, out.tgraph, nullptr);
        ggml_tensor* gx = ggml_graph_get_grad(out.tgraph, out.xb[(size_t)dc.depth]);
        if (!gx) return fail("tail graph produced no dL/dx gradient");
        ggml_build_forward_expand(out.tgraph, ggml_cpy(out.tctx, gx, out.g_in(dc.depth - 1)));
        keep_param_grads(out.tgraph, out.tail_param_idx);
        out.talloc = ggml_gallocr_new(buft);
        if (!out.talloc || !ggml_gallocr_alloc_graph(out.talloc, out.tgraph))
            return fail("failed to allocate tail training graph");
    }

    // --- B_l: per-block VJP fwd+bwd, shared allocator (re-planned per use in the runner) ---
    {
        out.balloc = ggml_gallocr_new(buft);
        if (!out.balloc) return fail("failed to create block graph allocator");
        out.blocks.resize((size_t)dc.depth);
        for (int l = 0; l < dc.depth; ++l) {
            TrainCkptBlock& B = out.blocks[(size_t)l];
            B.param_idx = block_param_idx[(size_t)l];
            B.ctx = graph_ctx(6144, true);
            if (!B.ctx) return fail("ggml_init failed for block graph");
            B.graph = ggml_new_graph_custom(B.ctx, 6144, true);
            ggml_tensor* xo = dit_block(B.ctx, dit, "dit." + std::to_string(l) + ".", out.xb[(size_t)l],
                                        out.context_p, out.gcond_p, out.pos, out.ones, dc, out.local, &out.dl, true);
            ggml_tensor* vjp = ggml_sum(B.ctx, ggml_mul(B.ctx, xo, out.g_in(l)));
            ggml_set_loss(vjp);
            ggml_build_forward_expand(B.graph, vjp);
            ggml_build_backward_expand(B.ctx, B.graph, nullptr);
            ggml_tensor* gx = ggml_graph_get_grad(B.graph, out.xb[(size_t)l]);
            if (!gx) return fail("block graph produced no dL/dx gradient");
            ggml_build_forward_expand(B.graph, ggml_cpy(B.ctx, gx, out.g_out(l)));
            B.grad_ctx = ggml_graph_get_grad(B.graph, out.context_p);
            B.grad_gcond = ggml_graph_get_grad(B.graph, out.gcond_p);
            if (!B.grad_ctx || !B.grad_gcond) return fail("block graph missing context/gcond gradient");
            ggml_set_output(B.grad_ctx);
            ggml_set_output(B.grad_gcond);
            keep_param_grads(B.graph, B.param_idx);
        }
    }

    // --- H: head VJP fwd+bwd ---
    {
        out.hctx = graph_ctx(4096, true);
        if (!out.hctx) return fail("ggml_init failed for head graph");
        out.hgraph = ggml_new_graph_custom(out.hctx, 4096, true);
        DitHeadOut h = dit_head(out.hctx, dit, out.x, out.tfeat, out.cross, out.global, dc, &out.dl, true);
        ggml_tensor* vjp = ggml_add(out.hctx,
            ggml_add(out.hctx,
                ggml_sum(out.hctx, ggml_mul(out.hctx, h.x0, out.g_out(0))),
                ggml_sum(out.hctx, ggml_mul(out.hctx, h.context, out.Gctx_in))),
            ggml_sum(out.hctx, ggml_mul(out.hctx, h.gcond, out.Ggcond_in)));
        ggml_set_loss(vjp);
        ggml_build_forward_expand(out.hgraph, vjp);
        ggml_build_backward_expand(out.hctx, out.hgraph, nullptr);
        keep_param_grads(out.hgraph, out.head_param_idx);
        out.halloc = ggml_gallocr_new(buft);
        if (!out.halloc || !ggml_gallocr_alloc_graph(out.halloc, out.hgraph))
            return fail("failed to allocate head training graph");
    }

    std::fprintf(stderr, "[train] checkpointed graphs: F %d nodes %.1f MiB, T %d nodes %.1f MiB, "
                 "%d block graphs %d nodes each, H %d nodes %.1f MiB, persistent %.1f MiB\n",
                 ggml_graph_n_nodes(out.fgraph), ggml_gallocr_get_buffer_size(out.falloc, 0) / (1024.0 * 1024.0),
                 ggml_graph_n_nodes(out.tgraph), ggml_gallocr_get_buffer_size(out.talloc, 0) / (1024.0 * 1024.0),
                 dc.depth, ggml_graph_n_nodes(out.blocks[0].graph),
                 ggml_graph_n_nodes(out.hgraph), ggml_gallocr_get_buffer_size(out.halloc, 0) / (1024.0 * 1024.0),
                 ggml_backend_buffer_get_size(out.pbuf) / (1024.0 * 1024.0));
    return true;
}

} // namespace sa3
