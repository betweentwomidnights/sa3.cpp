// lora.h — runtime LoRA/DoRA adapters for sa3.cpp.
//
// Adapters are applied in WEIGHT SPACE: for every targeted base weight W0 we recompute
// an effective weight W_eff = f(W0; A,B,magnitude,strength) and register it as an override
// on the base GgufModel (so the existing graph uses it via W.get()). This is done once per
// strength setting (the transform doesn't depend on activations), which both matches the
// PyTorch parametrization and is far cheaper than a per-step graph op. Multiple adapters
// COMPOSE IN ORDER (each reads the previous adapter's W_eff) — DoRA is non-commutative.
//
// dora-rows (our ckpts): V = W0 + (alpha/rank)*strength*(B@A);
//                        W_eff = magnitude[:,None] * V / (rownorm(V, over in) + 1e-12).
// plain lora:            W_eff = W0 + (alpha/rank)*strength*(B@A).  (additive, commutative)
#pragma once

#include "ggml.h"
#include "gguf_model.h"

#include <cmath>
#include <string>
#include <vector>

namespace sa3 {

struct LoraAdapter {
    GgufModel   gguf;
    std::string type;       // "lora" | "dora-rows" | ...
    int         rank = 0;
    float       alpha = 0.0f;
    float       strength = 1.0f;
};

inline LoraAdapter load_lora(const char* path, float strength = 1.0f) {
    LoraAdapter a;
    a.gguf = load_gguf(path);
    int ti = gguf_find_key(a.gguf.gguf, "lora.adapter_type");
    a.type     = ti < 0 ? "lora" : gguf_get_val_str(a.gguf.gguf, ti);
    a.rank     = (int)a.gguf.u32("lora.rank");
    a.alpha    = a.gguf.f32("lora.alpha");
    a.strength = strength;
    return a;
}

// W_eff storage: own a context + backend buffer holding the override tensors.
struct LoraStack {
    ggml_context*         ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    void free() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        ctx = nullptr; buf = nullptr;
    }
};

// Apply `adapters` (in order) to `base`, filling base.overrides with W_eff for every
// targeted weight. Returns a LoraStack owning the override tensors (free after the model).
inline LoraStack apply_loras(GgufModel& base, std::vector<LoraAdapter>& adapters) {
    LoraStack st;
    // 1. collect the set of base weights any adapter targets (base name = "<X>.weight",
    //    adapter tensors are "<X>.lora_A/.lora_B/.magnitude").
    std::vector<std::string> targets;
    for (auto& kv : base.tensors) {
        const std::string& wname = kv.first;
        if (wname.size() < 7 || wname.compare(wname.size()-7, 7, ".weight") != 0) continue;
        std::string stem = wname.substr(0, wname.size()-7);
        for (auto& a : adapters)
            if (a.gguf.has(stem + ".lora_A")) { targets.push_back(wname); break; }
    }

    // 2. allocate a context holding one override tensor per target
    ggml_init_params ip = { (size_t)targets.size()*ggml_tensor_overhead() + (1<<20), nullptr, /*no_alloc=*/true };
    st.ctx = ggml_init(ip);
    std::vector<ggml_tensor*> outs; outs.reserve(targets.size());
    for (auto& wname : targets) {
        ggml_tensor* W0 = base.tensors[wname];
        ggml_tensor* t = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, W0->ne[0], W0->ne[1]);
        ggml_set_name(t, wname.c_str());
        outs.push_back(t);
    }
    st.buf = ggml_backend_alloc_ctx_tensors(st.ctx, base.backend);

    // 3. compute W_eff per target (host), chaining adapters in order
    std::vector<float> w0, A, B, mag, V;
    for (size_t ti = 0; ti < targets.size(); ti++) {
        const std::string& wname = targets[ti];
        std::string stem = wname.substr(0, wname.size()-7);
        ggml_tensor* W0 = base.tensors[wname];
        const int in = (int)W0->ne[0], out = (int)W0->ne[1];
        w0.resize((size_t)in*out);
        ggml_backend_tensor_get(W0, w0.data(), 0, w0.size()*sizeof(float));   // w0[o*in + i]

        for (auto& a : adapters) {
            if (!a.gguf.has(stem + ".lora_A")) continue;
            if (a.strength == 0.0f) continue;                                 // identity
            ggml_tensor *At = a.gguf.get(stem+".lora_A"), *Bt = a.gguf.get(stem+".lora_B");
            const int rank = (int)At->ne[1];                                  // A ggml [in, rank]
            A.resize((size_t)rank*in); B.resize((size_t)out*rank); mag.resize(out);
            ggml_backend_tensor_get(At, A.data(), 0, A.size()*sizeof(float)); // A[r*in + i]
            ggml_backend_tensor_get(Bt, B.data(), 0, B.size()*sizeof(float)); // B[o*rank + r]
            const bool dora = a.type.rfind("dora", 0) == 0;
            if (dora) ggml_backend_tensor_get(a.gguf.get(stem+".magnitude"), mag.data(), 0, out*sizeof(float));
            const float sc = (a.alpha / rank) * a.strength;

            V.resize(in);
            for (int o = 0; o < out; o++) {
                float* w0o = &w0[(size_t)o*in];
                for (int i = 0; i < in; i++) V[i] = w0o[i];
                const float* Bo = &B[(size_t)o*rank];
                for (int r = 0; r < rank; r++) {                              // V += sc * B[o,r] * A[r,:]
                    const float b = sc * Bo[r];
                    const float* Ar = &A[(size_t)r*in];
                    for (int i = 0; i < in; i++) V[i] += b * Ar[i];
                }
                if (dora) {                                                   // renormalize per output row
                    double s = 0; for (int i = 0; i < in; i++) s += (double)V[i]*V[i];
                    const float inv = mag[o] / (float)(std::sqrt(s) + 1e-12);
                    for (int i = 0; i < in; i++) w0o[i] = V[i] * inv;
                } else {
                    for (int i = 0; i < in; i++) w0o[i] = V[i];               // plain LoRA: additive
                }
            }
        }
        ggml_backend_tensor_set(outs[ti], w0.data(), 0, w0.size()*sizeof(float));
        base.overrides[wname] = outs[ti];
    }
    return st;
}

} // namespace sa3
