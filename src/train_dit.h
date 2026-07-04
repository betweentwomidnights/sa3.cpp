// train_dit.h - DiT training graph assembly for native SA3 LoRA training.
#pragma once

#include "dit.h"
#include "gguf_model.h"
#include "train_lora.h"

#include <string>
#include <vector>

namespace sa3 {

struct TrainDitParamTensors {
    std::string stem;
    ggml_tensor* lora_A = nullptr;
    ggml_tensor* lora_B = nullptr;
    ggml_tensor* magnitude = nullptr;
    ggml_tensor* magnitude_r = nullptr;
    ggml_tensor* magnitude_c = nullptr;
};

struct TrainDitGraph {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t ctx_buf = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* tfeat = nullptr;
    ggml_tensor* cross = nullptr;
    ggml_tensor* global = nullptr;
    ggml_tensor* pos = nullptr;
    ggml_tensor* ones = nullptr;
    ggml_tensor* target = nullptr;
    ggml_tensor* velocity = nullptr;
    ggml_tensor* loss = nullptr;
    std::vector<TrainDitParamTensors> params;
};

inline void free_train_dit_graph(TrainDitGraph& g) {
    if (g.ctx_buf) ggml_backend_buffer_free(g.ctx_buf);
    if (g.ctx) ggml_free(g.ctx);
    g = TrainDitGraph{};
}

inline bool build_train_dit_forward_graph(GgufModel& dit, const DitConfig& dc, const TrainLoraState& lora,
                                          int frames, int cond_dim, int ctx_len,
                                          TrainDitGraph& out, std::string& err) {
    if (frames <= 0 || cond_dim <= 0 || ctx_len <= 0) {
        err = "invalid DiT training graph dimensions";
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

    std::vector<std::string> overridden;
    for (const TrainLoraParam& hp : lora.params) {
        TrainDitParamTensors tp;
        tp.stem = hp.target.stem;
        tp.lora_A = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, hp.target.in, lora.rank);
        tp.lora_B = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, lora.rank, hp.target.out);
        ggml_set_param(tp.lora_A);
        ggml_set_param(tp.lora_B);
        ggml_set_input(tp.lora_A);
        ggml_set_input(tp.lora_B);
        TrainLoraGraphParam gp;
        gp.lora_A = tp.lora_A;
        gp.lora_B = tp.lora_B;
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
        dit.overrides[hp.target.weight_name] =
            train_lora_effective_weight(out.ctx, base, gp, lora.adapter_type, lora.rank, lora.alpha);
        overridden.push_back(hp.target.weight_name);
        out.params.push_back(tp);
    }

    out.velocity = ggml_cont(out.ctx, dit_forward(out.ctx, dit, out.x, out.tfeat, out.cross,
                                                  out.global, out.pos, out.ones, dc, nullptr));
    ggml_set_output(out.velocity);
    out.loss = ggml_scale(out.ctx,
                          ggml_sum(out.ctx, ggml_sqr(out.ctx, ggml_sub(out.ctx, out.velocity, out.target))),
                          1.0f / (float)(dc.io * frames));
    ggml_set_loss(out.loss);
    ggml_set_output(out.loss);
    out.graph = ggml_new_graph_custom(out.ctx, 65536, true);
    ggml_build_forward_expand(out.graph, out.loss);
    ggml_build_backward_expand(out.ctx, out.graph, nullptr);
    for (const std::string& name : overridden) dit.overrides.erase(name);
    return true;
}

} // namespace sa3
