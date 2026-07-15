// train_dit.h - DiT training graph assembly for native SA3 LoRA training.
#pragma once

#include "dit.h"
#include "gguf_model.h"
#include "train_lora.h"

#include "ggml-alloc.h"

#include <cstdio>
#include <string>
#include <vector>

namespace sa3 {

struct TrainDitParamTensors {
    std::string stem;
    ggml_tensor* lora_A = nullptr;      // standard families (trainable)
    ggml_tensor* lora_B = nullptr;      // standard families (trainable)
    ggml_tensor* U = nullptr;           // -xs frozen basis (input, not trained)
    ggml_tensor* V = nullptr;           // -xs frozen basis (input, not trained)
    ggml_tensor* M_xs = nullptr;        // -xs trainable core
    ggml_tensor* magnitude = nullptr;
    ggml_tensor* magnitude_r = nullptr;
    ggml_tensor* magnitude_c = nullptr;
    ggml_tensor* base_norm_sq = nullptr;    // functional dora-rows constant input [out]
    std::vector<float> base_norm_sq_host;   // its (constant) host values, re-uploaded every step
};

struct TrainDitGraph {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t ctx_buf = nullptr;   // legacy whole-context buffer (unused with gallocr)
    ggml_gallocr_t alloc = nullptr;            // reusing graph allocator (peak = working set)
    ggml_cgraph* graph = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* tfeat = nullptr;
    ggml_tensor* cross = nullptr;
    ggml_tensor* global = nullptr;
    ggml_tensor* pos = nullptr;
    ggml_tensor* ones = nullptr;
    ggml_tensor* target = nullptr;
    ggml_tensor* local = nullptr;        // [local_dim, frames] inpaint local-add cond (Stage 12), or null
    ggml_tensor* loss_weight = nullptr;  // [io, frames] per-position loss weight (Stage 12), or null
    ggml_tensor* velocity = nullptr;
    ggml_tensor* loss = nullptr;
    std::vector<TrainDitParamTensors> params;
    DitLora dl;                          // functional adapter map (lora / dora-rows); empty otherwise
};

inline void free_train_dit_graph(TrainDitGraph& g) {
    if (g.alloc) ggml_gallocr_free(g.alloc);
    if (g.ctx_buf) ggml_backend_buffer_free(g.ctx_buf);
    if (g.ctx) ggml_free(g.ctx);
    g = TrainDitGraph{};
}

inline bool build_train_dit_forward_graph(GgufModel& dit, const DitConfig& dc, const TrainLoraState& lora,
                                          int frames, int cond_dim, int ctx_len,
                                          TrainDitGraph& out, std::string& err, bool inpaint = false) {
    if (frames <= 0 || cond_dim <= 0 || ctx_len <= 0) {
        err = "invalid DiT training graph dimensions";
        return false;
    }
    if (inpaint && dc.local_dim <= 0) {
        err = "inpainting requested but DiT has no local-cond weights (dit.local_dim <= 0)";
        return false;
    }
    out = TrainDitGraph{};
    ggml_init_params ip = { (size_t)768 * 1024 * 1024, nullptr, true };
    out.ctx = ggml_init(ip);
    if (!out.ctx) {
        err = "ggml_init failed for DiT training graph";
        return false;
    }
    out.x = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, dc.io, frames);
    out.tfeat = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, dc.time_dim);
    out.cross = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, cond_dim, ctx_len);
    out.global = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, cond_dim);
    out.pos = ggml_new_tensor_1d(out.ctx, GGML_TYPE_I32, dc.mem_tokens + frames);
    out.ones = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, 1);
    out.target = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, dc.io, frames);
    for (ggml_tensor* t : {out.x, out.tfeat, out.cross, out.global, out.pos, out.ones, out.target}) ggml_set_input(t);
    if (inpaint) {
        out.local = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, dc.local_dim, frames);
        out.loss_weight = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, dc.io, frames);
        ggml_set_input(out.local);
        ggml_set_input(out.loss_weight);
    }

    const bool xs = lora.adapter_type.size() >= 3 &&
                    lora.adapter_type.compare(lora.adapter_type.size() - 3, 3, "-xs") == 0;
    // Functional adapter application (see functional-lora-speed-plan): keeps the backward off the
    // full-weight out_prod that made the materialized-effective-weight path ~27x slower. Supported
    // for plain lora and dora-rows (the ratatat-2 reference config); other families fall back to the
    // materialized dit.overrides path below.
    const bool functional = (lora.adapter_type == "lora" || lora.adapter_type == "dora-rows");
    const bool dora_rows = (lora.adapter_type == "dora-rows");
    std::vector<std::string> overridden;
    for (const TrainLoraParam& hp : lora.params) {
        TrainDitParamTensors tp;
        tp.stem = hp.target.stem;
        TrainLoraGraphParam gp;
        if (xs) {
            // Frozen SVD bases are graph inputs (uploaded, never trained); only M_xs is a parameter.
            tp.U = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, lora.rank, hp.target.out);
            tp.V = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, lora.rank, hp.target.in);
            tp.M_xs = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, lora.rank, lora.rank);
            ggml_set_input(tp.U);
            ggml_set_input(tp.V);
            ggml_set_param(tp.M_xs);
            ggml_set_input(tp.M_xs);
            gp.U = tp.U;
            gp.V = tp.V;
            gp.M_xs = tp.M_xs;
        } else {
            tp.lora_A = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hp.target.in, lora.rank);
            tp.lora_B = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, lora.rank, hp.target.out);
            ggml_set_param(tp.lora_A);
            ggml_set_param(tp.lora_B);
            ggml_set_input(tp.lora_A);
            ggml_set_input(tp.lora_B);
            gp.lora_A = tp.lora_A;
            gp.lora_B = tp.lora_B;
        }
        if (!hp.magnitude.empty()) {
            tp.magnitude = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, (int64_t)hp.magnitude.size());
            ggml_set_param(tp.magnitude);
            ggml_set_input(tp.magnitude);
            gp.magnitude = tp.magnitude;
        }
        if (!hp.magnitude_r.empty()) {
            tp.magnitude_r = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, (int64_t)hp.magnitude_r.size());
            ggml_set_param(tp.magnitude_r);
            ggml_set_input(tp.magnitude_r);
            gp.magnitude_r = tp.magnitude_r;
        }
        if (!hp.magnitude_c.empty()) {
            tp.magnitude_c = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, (int64_t)hp.magnitude_c.size());
            ggml_set_param(tp.magnitude_c);
            ggml_set_input(tp.magnitude_c);
            gp.magnitude_c = tp.magnitude_c;
        }
        ggml_tensor* base = dit.get(hp.target.weight_name);
        if (functional) {
            // Register the adapter tensors for functional application in dit_lin. The tiny A,B
            // (+ dora magnitude) stay the only trained params; the frozen base is read only for the
            // dora column-norm (cheaply, via mul_mat(W,A)), never materialized into a full W_eff.
            DitLoraParam dp;
            dp.A = tp.lora_A;
            dp.B = tp.lora_B;
            dp.scale = lora.alpha / (float)lora.rank;
            dp.in = hp.target.in;
            if (dora_rows) {
                dp.dora = true;
                dp.magnitude = tp.magnitude;
                tp.base_norm_sq = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, hp.target.out);
                ggml_set_input(tp.base_norm_sq);
                dp.base_norm_sq = tp.base_norm_sq;
                tp.base_norm_sq_host = train_row_norm_sq(base, hp.target.in, hp.target.out, 1e-12f);
            }
            out.dl[hp.target.weight_name] = dp;
        } else {
            dit.overrides[hp.target.weight_name] =
                train_lora_effective_weight(out.ctx, base, gp, lora.adapter_type, lora.rank, lora.alpha);
            overridden.push_back(hp.target.weight_name);
        }
        out.params.push_back(tp);
    }

    const DitLora* dl = functional ? &out.dl : nullptr;
    out.velocity = ggml_cont(out.ctx, dit_forward(out.ctx, dit, out.x, out.tfeat, out.cross,
                                                  out.global, out.pos, out.ones, dc, out.local, dl));
    ggml_set_output(out.velocity);
    ggml_tensor* sq = ggml_sqr(out.ctx, ggml_sub(out.ctx, out.velocity, out.target));
    if (inpaint) {
        // Stage 12: sum(mse * loss_weight). loss_weight folds in the reference's
        // mean_gen(mse) + mask_loss_weight*mean_ctx(mse) (per-frame 1/(io*N_gen) or w/(io*N_ctx)),
        // so this equals the uniform mean when no frame is masked (FULL keep can't happen).
        out.loss = ggml_sum(out.ctx, ggml_mul(out.ctx, sq, out.loss_weight));
    } else {
        out.loss = ggml_scale(out.ctx, ggml_sum(out.ctx, sq), 1.0f / (float)(dc.io * frames));
    }
    ggml_set_loss(out.loss);
    ggml_set_output(out.loss);
    out.graph = ggml_new_graph_custom(out.ctx, 65536, true);
    ggml_build_forward_expand(out.graph, out.loss);
    ggml_build_backward_expand(out.ctx, out.graph, nullptr);
    for (const std::string& name : overridden) dit.overrides.erase(name);

    // Mark the trainable-parameter gradients as graph outputs so the reusing allocator keeps them
    // resident through compute — we read them back on the host each step for gradient accumulation.
    // (Without this, gallocr could reuse a grad buffer for a later intermediate.)
    for (const TrainDitParamTensors& tp : out.params) {
        for (ggml_tensor* p : {tp.lora_A, tp.lora_B, tp.M_xs, tp.magnitude, tp.magnitude_r, tp.magnitude_c}) {
            if (!p) continue;
            ggml_tensor* g = ggml_graph_get_grad(out.graph, p);
            if (!g) g = ggml_graph_get_grad_acc(out.graph, p);
            if (g) ggml_set_output(g);
        }
    }

    // Allocate the forward+backward graph with a reusing allocator (ggml_gallocr): peak memory is
    // the max concurrent working set, not the sum of every intermediate + gradient. The prior
    // ggml_backend_alloc_ctx_tensors path made every tensor persistent (~100 GB for medium DiT),
    // which only "worked" on CPU via pagefile over-commit and hard-failed on GPU VRAM.
    out.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dit.backend));
    if (!out.alloc || !ggml_gallocr_alloc_graph(out.alloc, out.graph)) {
        err = "failed to allocate DiT training graph (gallocr)";
        return false;
    }
    std::fprintf(stderr, "[train] DiT fwd+bwd graph: %d nodes, gallocr buffer %.1f MiB\n",
                 ggml_graph_n_nodes(out.graph),
                 ggml_gallocr_get_buffer_size(out.alloc, 0) / (1024.0 * 1024.0));
    // NB: the functional dora-rows base_norm_sq constants are uploaded per step in
    // upload_train_lora_state, not here: gallocr reuses an input tensor's buffer once its last
    // consumer has run, so a single build-time upload would be clobbered after the first step.
    return true;
}

} // namespace sa3
