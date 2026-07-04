// train_lora.h - trainable LoRA/DoRA adapter state helpers.
#pragma once

#include "gguf_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace sa3 {

struct TrainLoraTarget {
    std::string weight_name;
    std::string stem;
    int64_t in = 0;
    int64_t out = 0;
};

struct TrainLoraParam {
    TrainLoraTarget target;
    std::vector<float> lora_A;      // [rank, in] host row-major by rank
    std::vector<float> lora_B;      // [out, rank]
    std::vector<float> U;           // [out, rank] for -xs
    std::vector<float> V;           // [in, rank] for -xs
    std::vector<float> M_xs;        // [rank, rank] for -xs
    std::vector<float> magnitude;   // dora-rows [out] or dora-cols [in]
    std::vector<float> magnitude_r; // bora [out]
    std::vector<float> magnitude_c; // bora [in]
};

struct TrainLoraState {
    std::string adapter_type = "lora";
    int rank = 0;
    float alpha = 1.0f;
    std::vector<TrainLoraParam> params;
};

struct TrainLoraGraphParam {
    ggml_tensor* lora_A = nullptr;      // [in, rank]
    ggml_tensor* lora_B = nullptr;      // [rank, out]
    ggml_tensor* magnitude = nullptr;   // dora rows [out] or cols [in]
    ggml_tensor* magnitude_r = nullptr; // bora rows [out]
    ggml_tensor* magnitude_c = nullptr; // bora cols [in]
};

inline std::vector<TrainLoraTarget> enumerate_train_lora_targets(const GgufModel& dit) {
    std::vector<TrainLoraTarget> out;
    for (const auto& kv : dit.tensors) {
        const std::string& name = kv.first;
        ggml_tensor* t = kv.second;
        if (name.rfind("dit.", 0) != 0) continue;
        if (name.size() < 7 || name.compare(name.size() - 7, 7, ".weight") != 0) continue;
        if (ggml_n_dims(t) != 2) continue;
        TrainLoraTarget target;
        target.weight_name = name;
        target.stem = name.substr(0, name.size() - 7);
        target.in = t->ne[0];
        target.out = t->ne[1];
        if (target.in > 0 && target.out > 0) out.push_back(std::move(target));
    }
    std::sort(out.begin(), out.end(), [](const TrainLoraTarget& a, const TrainLoraTarget& b) {
        return a.weight_name < b.weight_name;
    });
    return out;
}

inline void train_tensor_to_f32(ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize((size_t)n);
    if (t->buffer) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    } else if (t->data) {
        if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), t->data, out.size() * sizeof(float));
        else std::fill(out.begin(), out.end(), 0.0f);
    } else {
        std::fill(out.begin(), out.end(), 0.0f);
    }
}

inline std::vector<float> train_row_norms(ggml_tensor* w, int64_t in, int64_t out) {
    std::vector<float> data;
    train_tensor_to_f32(w, data);
    std::vector<float> norms((size_t)out, 1.0f);
    if (data.size() < (size_t)(in * out)) return norms;
    for (int64_t o = 0; o < out; ++o) {
        double s = 0.0;
        for (int64_t i = 0; i < in; ++i) {
            const float v = data[(size_t)o * in + i];
            s += (double)v * v;
        }
        norms[(size_t)o] = (float)std::sqrt(s);
    }
    return norms;
}

inline std::vector<float> train_col_norms(ggml_tensor* w, int64_t in, int64_t out) {
    std::vector<float> data;
    train_tensor_to_f32(w, data);
    std::vector<float> norms((size_t)in, 1.0f);
    if (data.size() < (size_t)(in * out)) return norms;
    for (int64_t i = 0; i < in; ++i) {
        double s = 0.0;
        for (int64_t o = 0; o < out; ++o) {
            const float v = data[(size_t)o * in + i];
            s += (double)v * v;
        }
        norms[(size_t)i] = (float)std::sqrt(s);
    }
    return norms;
}

inline bool init_train_lora_state(const GgufModel& dit, const std::vector<TrainLoraTarget>& targets,
                                  const std::string& adapter_type, int rank, float alpha,
                                  unsigned long long seed, TrainLoraState& out, std::string& err) {
    if (rank <= 0) {
        err = "rank must be positive";
        return false;
    }
    out = TrainLoraState{};
    out.adapter_type = adapter_type;
    out.rank = rank;
    out.alpha = alpha;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.01f);
    const bool xs = adapter_type.size() >= 3 && adapter_type.compare(adapter_type.size() - 3, 3, "-xs") == 0;
    for (const TrainLoraTarget& t : targets) {
        TrainLoraParam p;
        p.target = t;
        ggml_tensor* w = dit.get(t.weight_name);
        if (xs) {
            p.U.resize((size_t)t.out * rank);
            p.V.resize((size_t)t.in * rank);
            p.M_xs.assign((size_t)rank * rank, 0.0f);
            for (float& v : p.U) v = normal(rng);
            for (float& v : p.V) v = normal(rng);
        } else {
            p.lora_A.resize((size_t)rank * t.in);
            p.lora_B.assign((size_t)t.out * rank, 0.0f);
            for (float& v : p.lora_A) v = normal(rng);
        }
        if (adapter_type == "dora-rows" || adapter_type == "dora-rows-xs") {
            p.magnitude = train_row_norms(w, t.in, t.out);
        } else if (adapter_type == "dora-cols" || adapter_type == "dora-cols-xs") {
            p.magnitude = train_col_norms(w, t.in, t.out);
        } else if (adapter_type == "bora" || adapter_type == "bora-xs") {
            p.magnitude_r = train_row_norms(w, t.in, t.out);
            p.magnitude_c = train_col_norms(w, t.in, t.out);
        }
        out.params.push_back(std::move(p));
    }
    return true;
}

inline ggml_tensor* train_lora_apply_col_norm(ggml_context* ctx, ggml_tensor* v,
                                              ggml_tensor* magnitude, int64_t in, int64_t out) {
    ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, v)); // [out, in]
    ggml_tensor* mag = ggml_reshape_2d(ctx, magnitude, 1, in); // [1, in]
    ggml_tensor* vn = ggml_scale(ctx, ggml_rms_norm(ctx, vt, 1e-12f), 1.0f / sqrtf((float)out));
    return ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, vn, mag)));
}

inline ggml_tensor* train_lora_effective_weight(ggml_context* ctx, ggml_tensor* base,
                                                const TrainLoraGraphParam& p,
                                                const std::string& adapter_type,
                                                int rank, float alpha) {
    const int64_t in = base->ne[0], out = base->ne[1];
    ggml_tensor* delta = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, p.lora_A)), p.lora_B);
    ggml_tensor* v = ggml_add(ctx, base, ggml_scale(ctx, delta, alpha / (float)rank));
    if (adapter_type == "lora") return v;
    if (adapter_type == "dora-rows") {
        ggml_tensor* mag = ggml_reshape_2d(ctx, p.magnitude, 1, out);
        ggml_tensor* vn = ggml_scale(ctx, ggml_rms_norm(ctx, v, 1e-12f), 1.0f / sqrtf((float)in));
        return ggml_mul(ctx, vn, mag);
    }
    if (adapter_type == "dora-cols") {
        return train_lora_apply_col_norm(ctx, v, p.magnitude, in, out);
    }
    if (adapter_type == "bora") {
        ggml_tensor* mag_r = ggml_reshape_2d(ctx, p.magnitude_r, 1, out);
        ggml_tensor* rown = ggml_scale(ctx, ggml_rms_norm(ctx, v, 1e-12f), 1.0f / sqrtf((float)in));
        ggml_tensor* rowed = ggml_mul(ctx, rown, mag_r);
        return train_lora_apply_col_norm(ctx, rowed, p.magnitude_c, in, out);
    }
    return v;
}

} // namespace sa3
