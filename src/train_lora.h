// train_lora.h - trainable LoRA/DoRA adapter state helpers.
#pragma once

#include "gguf_model.h"
#include "train_svd.h"

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
    ggml_tensor* lora_A = nullptr;      // [in, rank]        (standard families)
    ggml_tensor* lora_B = nullptr;      // [rank, out]       (standard families)
    ggml_tensor* U = nullptr;           // [rank, out]       (-xs: frozen SVD basis)
    ggml_tensor* V = nullptr;           // [rank, in]        (-xs: frozen SVD basis)
    ggml_tensor* M_xs = nullptr;        // [rank, rank]      (-xs: trainable core)
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
    if (t->type == GGML_TYPE_F16) {                       // convert f16 base weights correctly
        std::vector<ggml_fp16_t> tmp((size_t)n);
        if (t->buffer) ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
        else if (t->data) std::memcpy(tmp.data(), t->data, tmp.size() * sizeof(ggml_fp16_t));
        else { std::fill(out.begin(), out.end(), 0.0f); return; }
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
    } else if (t->buffer) {
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

// Per-output squared column norm Σ_in W[in,out]² + in·eps, for the functional dora-rows path.
// This is the constant base term of ||W + s·A@B||_col² (the A,B contributions are added in-graph);
// folding in·eps here reproduces the materialized rms_norm denominator sqrt(Σv² + in·eps).
inline std::vector<float> train_row_norm_sq(ggml_tensor* w, int64_t in, int64_t out, float eps) {
    std::vector<float> data;
    train_tensor_to_f32(w, data);
    std::vector<float> nsq((size_t)out, (float)((double)in * eps));
    if (data.size() < (size_t)(in * out)) return nsq;
    for (int64_t o = 0; o < out; ++o) {
        double s = 0.0;
        for (int64_t i = 0; i < in; ++i) {
            const float v = data[(size_t)o * in + i];
            s += (double)v * v;
        }
        nsq[(size_t)o] = (float)(s + (double)in * eps);
    }
    return nsq;
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

// Copy the first `rank` basis columns of a bases tensor stored ggml [stored_rank, rows]
// (raw row-major [rows][stored_rank]) into dst as [rows][rank]. Returns false on shape mismatch.
inline bool train_copy_svd_basis(ggml_tensor* t, int64_t rows, int rank, std::vector<float>& dst) {
    if (!t || t->ne[1] != rows || t->ne[0] < rank) return false;
    const int64_t sr = t->ne[0];
    std::vector<float> raw;
    train_tensor_to_f32(t, raw);
    dst.resize((size_t)rows * rank);
    for (int64_t r = 0; r < rows; ++r)
        for (int a = 0; a < rank; ++a) dst[(size_t)r * rank + a] = raw[(size_t)r * sr + a];
    return true;
}

inline bool init_train_lora_state(const GgufModel& dit, const std::vector<TrainLoraTarget>& targets,
                                  const std::string& adapter_type, int rank, float alpha,
                                  unsigned long long seed, TrainLoraState& out, std::string& err,
                                  const GgufModel* svd_bases = nullptr) {
    if (rank <= 0) {
        err = "rank must be positive";
        return false;
    }
    out = TrainLoraState{};
    out.adapter_type = adapter_type;
    out.rank = rank;
    out.alpha = alpha;
    std::mt19937_64 rng(seed);
    const bool xs = adapter_type.size() >= 3 && adapter_type.compare(adapter_type.size() - 3, 3, "-xs") == 0;
    for (const TrainLoraTarget& t : targets) {
        if (rank > t.in || rank > t.out) {
            err = "rank is infeasible for target " + t.stem;
            return false;
        }
        TrainLoraParam p;
        p.target = t;
        ggml_tensor* w = dit.get(t.weight_name);
        if (xs) {
            // LoRA-XS: freeze U/V as the top-rank SVD bases of the base weight, train only M_xs.
            // W is read as PyTorch [out, in] (ggml [in, out] raw -> W[o*in + i]); M_xs starts at 0
            // so the adapter is a no-op at init, matching the reference implementation.
            bool loaded = false;
            if (svd_bases) {
                ggml_tensor* Ut = svd_bases->has(t.stem + ".U") ? svd_bases->get(t.stem + ".U") : nullptr;
                ggml_tensor* Vt = svd_bases->has(t.stem + ".V") ? svd_bases->get(t.stem + ".V") : nullptr;
                if (Ut || Vt) {
                    if (!train_copy_svd_basis(Ut, t.out, rank, p.U) ||
                        !train_copy_svd_basis(Vt, t.in, rank, p.V)) {
                        err = "svd bases shape mismatch for target " + t.stem;
                        return false;
                    }
                    loaded = true;
                }
            }
            if (!loaded) {
                std::vector<float> wf;
                train_tensor_to_f32(w, wf);
                randomized_svd_topr(wf, (int)t.out, (int)t.in, rank,
                                    seed + (unsigned long long)out.params.size() + 1, p.U, p.V);
            }
            p.M_xs.assign((size_t)rank * rank, 0.0f);
        } else {
            // Reference init (models/lora/model.py): A ~ kaiming_uniform_(a=sqrt(5)) = U(-1/sqrt(in),
            // +1/sqrt(in)) — per-layer scale, NOT a fixed std — and B = 0. The previous fixed
            // normal(0, 0.01) started A ~32% cooler on dim-1536 layers (and off for every other in),
            // skewing the A/B learning balance relative to the reference trainer.
            p.lora_A.resize((size_t)rank * t.in);
            p.lora_B.assign((size_t)t.out * rank, 0.0f);
            const float bound = 1.0f / std::sqrt((float)t.in);
            std::uniform_real_distribution<float> kaiming(-bound, bound);
            for (float& v : p.lora_A) v = kaiming(rng);
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

// Low-rank update delta in ggml [in, out] layout. Standard families use B@A; -xs families
// use the frozen SVD bases with the trainable core, delta = U @ M_xs @ V^T (contracting rank).
inline ggml_tensor* train_lora_delta(ggml_context* ctx, const TrainLoraGraphParam& p, bool xs) {
    if (xs) {
        ggml_tensor* pmv = ggml_mul_mat(ctx, p.M_xs, p.V); // [rank, in]
        return ggml_mul_mat(ctx, pmv, p.U);                // [in, out]
    }
    return ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, p.lora_A)), p.lora_B); // [in, out]
}

inline ggml_tensor* train_lora_effective_weight(ggml_context* ctx, ggml_tensor* base,
                                                const TrainLoraGraphParam& p,
                                                const std::string& adapter_type,
                                                int rank, float alpha) {
    const int64_t in = base->ne[0], out = base->ne[1];
    // Upcast the (possibly f16) frozen base to f32 so all adapter math runs in full precision,
    // matching the reference forward (W_2d = W.to(lora_A.dtype)). Adapter params are already f32.
    ggml_tensor* w = base->type == GGML_TYPE_F32 ? base : ggml_cast(ctx, base, GGML_TYPE_F32);
    const bool xs = adapter_type.size() >= 3 && adapter_type.compare(adapter_type.size() - 3, 3, "-xs") == 0;
    const std::string fam = xs ? adapter_type.substr(0, adapter_type.size() - 3) : adapter_type;
    ggml_tensor* delta = train_lora_delta(ctx, p, xs);
    ggml_tensor* v = ggml_add(ctx, w, ggml_scale(ctx, delta, alpha / (float)rank));
    if (fam == "lora") return v;
    if (fam == "dora-rows") {
        ggml_tensor* mag = ggml_reshape_2d(ctx, p.magnitude, 1, out);
        ggml_tensor* vn = ggml_scale(ctx, ggml_rms_norm(ctx, v, 1e-12f), 1.0f / sqrtf((float)in));
        return ggml_mul(ctx, vn, mag);
    }
    if (fam == "dora-cols") {
        return train_lora_apply_col_norm(ctx, v, p.magnitude, in, out);
    }
    if (fam == "bora") {
        ggml_tensor* mag_r = ggml_reshape_2d(ctx, p.magnitude_r, 1, out);
        ggml_tensor* rown = ggml_scale(ctx, ggml_rms_norm(ctx, v, 1e-12f), 1.0f / sqrtf((float)in));
        ggml_tensor* rowed = ggml_mul(ctx, rown, mag_r);
        return train_lora_apply_col_norm(ctx, rowed, p.magnitude_c, in, out);
    }
    return v;
}

} // namespace sa3
