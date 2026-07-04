// train_checkpoint.h - GGUF checkpoint writing for native SA3 LoRA training.
#pragma once

#include "gguf.h"
#include "train_lora.h"

#include <cstring>
#include <map>
#include <string>

namespace sa3 {

inline void train_checkpoint_tensor_to_f32(ggml_tensor* t, std::vector<float>& out) {
    out.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
}

inline void train_checkpoint_add_tensor(ggml_context* ctx, gguf_context* g, const std::string& name,
                                        const std::vector<float>& data, int n_dims,
                                        const int64_t* ne) {
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims, ne);
    ggml_set_name(t, name.c_str());
    std::memcpy(t->data, data.data(), data.size() * sizeof(float));
    gguf_add_tensor(g, t);
}

inline bool write_train_lora_gguf(const TrainLoraState& state, const std::string& out_path, std::string& err) {
    if (state.rank <= 0 || state.params.empty()) {
        err = "cannot write empty LoRA checkpoint";
        return false;
    }
    size_t data_bytes = 0;
    for (const TrainLoraParam& p : state.params) {
        data_bytes += (p.lora_A.size() + p.lora_B.size() + p.U.size() + p.V.size() + p.M_xs.size() +
                       p.magnitude.size() + p.magnitude_r.size() + p.magnitude_c.size()) * sizeof(float);
    }
    ggml_init_params ip = { data_bytes + state.params.size() * 12 * (ggml_tensor_overhead() + 64) + (1u << 20),
                            nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        err = "ggml_init failed while writing LoRA checkpoint";
        return false;
    }
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "sa3-lora");
    gguf_set_val_str(g, "general.name", "sa3 native trained adapter");
    gguf_set_val_str(g, "lora.adapter_type", state.adapter_type.c_str());
    gguf_set_val_u32(g, "lora.rank", (uint32_t)state.rank);
    gguf_set_val_f32(g, "lora.alpha", state.alpha);
    gguf_set_val_u32(g, "lora.n_targets", (uint32_t)state.params.size());

    for (const TrainLoraParam& p : state.params) {
        if (!p.lora_A.empty()) {
            int64_t ne[2] = { p.target.in, state.rank };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".lora_A", p.lora_A, 2, ne);
        }
        if (!p.lora_B.empty()) {
            int64_t ne[2] = { state.rank, p.target.out };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".lora_B", p.lora_B, 2, ne);
        }
        if (!p.U.empty()) {
            int64_t ne[2] = { state.rank, p.target.out };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".U", p.U, 2, ne);
        }
        if (!p.V.empty()) {
            int64_t ne[2] = { state.rank, p.target.in };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".V", p.V, 2, ne);
        }
        if (!p.M_xs.empty()) {
            int64_t ne[2] = { state.rank, state.rank };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".M_xs", p.M_xs, 2, ne);
        }
        if (!p.magnitude.empty()) {
            int64_t ne[1] = { (int64_t)p.magnitude.size() };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".magnitude", p.magnitude, 1, ne);
        }
        if (!p.magnitude_r.empty()) {
            int64_t ne[1] = { (int64_t)p.magnitude_r.size() };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".magnitude_r", p.magnitude_r, 1, ne);
        }
        if (!p.magnitude_c.empty()) {
            int64_t ne[1] = { (int64_t)p.magnitude_c.size() };
            train_checkpoint_add_tensor(ctx, g, p.target.stem + ".magnitude_c", p.magnitude_c, 1, ne);
        }
    }
    const bool ok = gguf_write_to_file(g, out_path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) {
        err = "failed to write " + out_path;
        return false;
    }
    return true;
}

inline bool load_train_lora_gguf(const std::string& path, TrainLoraState& state, std::string& err) {
    GgufModel g;
    try {
        g = load_gguf(path.c_str());
        int ti = gguf_find_key(g.gguf, "lora.adapter_type");
        state = TrainLoraState{};
        state.adapter_type = ti < 0 ? "lora" : gguf_get_val_str(g.gguf, ti);
        state.rank = (int)g.u32("lora.rank");
        state.alpha = g.f32("lora.alpha");
        std::map<std::string, TrainLoraParam> by_stem;
        auto stem_kind = [](const std::string& name, std::string& stem, std::string& kind) {
            static const char* kinds[] = {".lora_A", ".lora_B", ".U", ".V", ".M_xs",
                                          ".magnitude", ".magnitude_r", ".magnitude_c"};
            for (const char* k : kinds) {
                const std::string ks = k;
                if (name.size() > ks.size() && name.compare(name.size() - ks.size(), ks.size(), ks) == 0) {
                    stem = name.substr(0, name.size() - ks.size());
                    kind = ks.substr(1);
                    return true;
                }
            }
            return false;
        };
        for (const auto& kv : g.tensors) {
            std::string stem, kind;
            if (!stem_kind(kv.first, stem, kind)) continue;
            TrainLoraParam& p = by_stem[stem];
            p.target.stem = stem;
            p.target.weight_name = stem + ".weight";
            if (kind == "lora_A") {
                p.target.in = kv.second->ne[0];
                train_checkpoint_tensor_to_f32(kv.second, p.lora_A);
            } else if (kind == "lora_B") {
                p.target.out = kv.second->ne[1];
                train_checkpoint_tensor_to_f32(kv.second, p.lora_B);
            } else if (kind == "U") {
                p.target.out = kv.second->ne[1];
                train_checkpoint_tensor_to_f32(kv.second, p.U);
            } else if (kind == "V") {
                p.target.in = kv.second->ne[1];
                train_checkpoint_tensor_to_f32(kv.second, p.V);
            } else if (kind == "M_xs") {
                train_checkpoint_tensor_to_f32(kv.second, p.M_xs);
            } else if (kind == "magnitude") {
                train_checkpoint_tensor_to_f32(kv.second, p.magnitude);
            } else if (kind == "magnitude_r") {
                train_checkpoint_tensor_to_f32(kv.second, p.magnitude_r);
            } else if (kind == "magnitude_c") {
                train_checkpoint_tensor_to_f32(kv.second, p.magnitude_c);
            }
        }
        for (auto& kv : by_stem) state.params.push_back(std::move(kv.second));
        g.free();
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (state.params.empty()) {
        err = "no adapter tensors found in " + path;
        return false;
    }
    return true;
}

} // namespace sa3
