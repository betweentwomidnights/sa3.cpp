// train_conditioning.h - caption conditioning for native SA3 LoRA training.
#pragma once

#include "gguf_model.h"
#include "sa3_pipeline.h"
#include "t5gemma.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace sa3 {

struct TrainConditioning {
    std::vector<float> cross;  // [cond_dim, max_len + 1]
    std::vector<float> global; // [cond_dim]
    int cond_dim = 0;
    int ctx_len = 0;
    int token_count = 0;
};

inline bool encode_train_caption_conditioning(Tokenizer& tok, GgufModel& te, const GgufModel& cond,
                                              const T5GemmaConfig& tc, const std::string& caption,
                                              float seconds, TrainConditioning& out, std::string& err) {
    const int max_len = (int)te.u32("t5g.max_length");
    const int cond_dim = tc.dim;
    std::vector<int32_t> encoded = tok.encode(caption);
    const int used = std::min((int)encoded.size(), max_len);
    std::vector<int32_t> ids((size_t)max_len, tok.pad_id), attn((size_t)max_len, 0), pos((size_t)max_len);
    for (int i = 0; i < used; ++i) {
        ids[(size_t)i] = encoded[(size_t)i];
        attn[(size_t)i] = 1;
    }
    for (int i = 0; i < max_len; ++i) pos[(size_t)i] = i;

    std::vector<float> hidden((size_t)cond_dim * max_len);
    ggml_init_params ip = { (size_t)256 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        err = "ggml_init failed for T5 conditioning graph";
        return false;
    }
    ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len);
    ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len);
    ggml_tensor* mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_len, max_len);
    ggml_set_input(ids_t);
    ggml_set_input(pos_t);
    ggml_set_input(mask_t);
    ggml_tensor* h = ggml_cont(ctx, t5gemma_encode(ctx, te, ids_t, pos_t, mask_t, tc));
    ggml_set_output(h);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, h);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(te.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    std::vector<float> mb((size_t)max_len * max_len);
    for (int q = 0; q < max_len; ++q) {
        for (int k = 0; k < max_len; ++k) mb[(size_t)q * max_len + k] = attn[(size_t)k] ? 0.0f : -INFINITY;
    }
    ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask_t, mb.data(), 0, mb.size() * sizeof(float));
    ggml_backend_graph_compute(te.backend, gf);
    ggml_backend_tensor_get(h, hidden.data(), 0, hidden.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    std::vector<float> pad_emb = tensor_to_host(cond, "te.padding_embedding");
    for (int p = 0; p < max_len; ++p) {
        if (!attn[(size_t)p]) std::memcpy(&hidden[(size_t)p * cond_dim], pad_emb.data(), cond_dim * sizeof(float));
    }

    const float smin = cond.f32("t5g.secs_min");
    const float smax = cond.f32("t5g.secs_max");
    const int sdim = (int)cond.u32("t5g.secs_dim");
    const float sclamp = seconds < smin ? smin : (seconds > smax ? smax : seconds);
    const float snorm = (sclamp - smin) / (smax - smin);
    std::vector<float> ef;
    expo_features(snorm, ef, sdim, 0.5f, 10000.0f);
    std::vector<float> sw = tensor_to_host(cond, "te.secs.weight");
    std::vector<float> sb = tensor_to_host(cond, "te.secs.bias");
    out.global.assign((size_t)cond_dim, 0.0f);
    for (int d = 0; d < cond_dim; ++d) {
        float acc = sb[(size_t)d];
        for (int i = 0; i < sdim; ++i) acc += ef[(size_t)i] * sw[(size_t)d * sdim + i];
        out.global[(size_t)d] = acc;
    }

    out.cond_dim = cond_dim;
    out.ctx_len = max_len + 1;
    out.token_count = used;
    out.cross.assign((size_t)cond_dim * out.ctx_len, 0.0f);
    std::memcpy(out.cross.data(), hidden.data(), hidden.size() * sizeof(float));
    std::memcpy(&out.cross[(size_t)cond_dim * max_len], out.global.data(), out.global.size() * sizeof(float));
    return true;
}

} // namespace sa3
