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
            if (a.gguf.has(stem + ".lora_A") || a.gguf.has(stem + ".M_xs")) { targets.push_back(wname); break; }
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

    // 3. compute W_eff per target (host), chaining adapters in order. Buffers reused across targets.
    // W is the running effective weight, row-major w[o*in+i]. For each adapter:
    //   delta = B@A  (standard)  or  U@M_xs@V^T  (-xs);  V = W + (alpha/rank)*strength*delta;
    //   then a per-type post-transform (additive / dora-rows / dora-cols / bora). Spec: lora_spec_test.py.
    std::vector<float> w, A, B, U, Vm, M, mv, dl, Vb, mag, magc, inter, coln;
    auto getv = [&](GgufModel& g, const std::string& n, std::vector<float>& dst){
        ggml_tensor* t = g.get(n); dst.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size()*sizeof(float));
    };
    for (size_t ti = 0; ti < targets.size(); ti++) {
        const std::string& wname = targets[ti];
        std::string stem = wname.substr(0, wname.size()-7);
        ggml_tensor* W0 = base.tensors[wname];
        const int in = (int)W0->ne[0], out = (int)W0->ne[1];
        w.resize((size_t)in*out);
        ggml_backend_tensor_get(W0, w.data(), 0, w.size()*sizeof(float));     // w[o*in + i]

        for (auto& a : adapters) {
            const bool xs  = a.gguf.has(stem + ".M_xs");
            const bool std_= a.gguf.has(stem + ".lora_A");
            if ((!xs && !std_) || a.strength == 0.0f) continue;               // not targeted / identity
            const float sc = (a.alpha / a.rank) * a.strength;

            dl.assign((size_t)out*in, 0.0f);                                  // delta[o*in + i]
            if (xs) {                                                         // delta = U @ M_xs @ V^T
                getv(a.gguf, stem+".U", U); getv(a.gguf, stem+".V", Vm); getv(a.gguf, stem+".M_xs", M);
                const int rk = a.rank;
                mv.assign((size_t)rk*in, 0.0f);                               // mv[a,i] = sum_b M[a,b]*V[i,b]
                for (int aa = 0; aa < rk; aa++) for (int i = 0; i < in; i++) {
                    float s = 0; const float* Mr = &M[(size_t)aa*rk]; const float* Vi = &Vm[(size_t)i*rk];
                    for (int b = 0; b < rk; b++) s += Mr[b]*Vi[b]; mv[(size_t)aa*in+i] = s;
                }
                for (int o = 0; o < out; o++) { const float* Uo = &U[(size_t)o*rk]; float* dlo = &dl[(size_t)o*in];
                    for (int aa = 0; aa < rk; aa++) { const float u = Uo[aa]; const float* mva = &mv[(size_t)aa*in];
                        for (int i = 0; i < in; i++) dlo[i] += u*mva[i]; } }
            } else {                                                          // delta = B @ A
                getv(a.gguf, stem+".lora_A", A); getv(a.gguf, stem+".lora_B", B);
                const int rk = a.rank;
                for (int o = 0; o < out; o++) { const float* Bo = &B[(size_t)o*rk]; float* dlo = &dl[(size_t)o*in];
                    for (int r = 0; r < rk; r++) { const float b = Bo[r]; const float* Ar = &A[(size_t)r*in];
                        for (int i = 0; i < in; i++) dlo[i] += b*Ar[i]; } }
            }
            Vb.resize((size_t)out*in);
            for (size_t k = 0; k < Vb.size(); k++) Vb[k] = w[k] + sc*dl[k];   // V = W + sc*delta

            const std::string& ty = a.type;
            if (ty == "lora" || ty == "lora-xs") {                            // additive
                w.swap(Vb);
            } else if (ty == "dora-rows" || ty == "dora-rows-xs") {           // row norm + magnitude[out]
                getv(a.gguf, stem+".magnitude", mag);
                for (int o = 0; o < out; o++) { const float* Vo = &Vb[(size_t)o*in]; float* wo = &w[(size_t)o*in];
                    double s = 0; for (int i = 0; i < in; i++) s += (double)Vo[i]*Vo[i];
                    const float inv = mag[o]/(float)(std::sqrt(s)+1e-12);
                    for (int i = 0; i < in; i++) wo[i] = Vo[i]*inv; }
            } else if (ty == "dora-cols" || ty == "dora-cols-xs") {           // col norm + magnitude[in]
                getv(a.gguf, stem+".magnitude", mag);
                coln.assign(in, 0.0);
                for (int o = 0; o < out; o++) for (int i = 0; i < in; i++) coln[i] += Vb[(size_t)o*in+i]*Vb[(size_t)o*in+i];
                for (int i = 0; i < in; i++) coln[i] = std::sqrt(coln[i]) + 1e-12f;
                for (int o = 0; o < out; o++) for (int i = 0; i < in; i++) w[(size_t)o*in+i] = mag[i]*Vb[(size_t)o*in+i]/coln[i];
            } else if (ty == "bora" || ty == "bora-xs") {                     // row-norm*mag_r, then col-norm*mag_c
                getv(a.gguf, stem+".magnitude_r", mag); getv(a.gguf, stem+".magnitude_c", magc);
                inter.resize((size_t)out*in);
                for (int o = 0; o < out; o++) { const float* Vo = &Vb[(size_t)o*in];
                    double s = 0; for (int i = 0; i < in; i++) s += (double)Vo[i]*Vo[i];
                    const float invr = mag[o]/(float)(std::sqrt(s)+1e-12);
                    for (int i = 0; i < in; i++) inter[(size_t)o*in+i] = Vo[i]*invr; }
                coln.assign(in, 0.0);
                for (int o = 0; o < out; o++) for (int i = 0; i < in; i++) coln[i] += inter[(size_t)o*in+i]*inter[(size_t)o*in+i];
                for (int i = 0; i < in; i++) coln[i] = std::sqrt(coln[i]) + 1e-12f;
                for (int o = 0; o < out; o++) for (int i = 0; i < in; i++) w[(size_t)o*in+i] = magc[i]*inter[(size_t)o*in+i]/coln[i];
            } else {
                fprintf(stderr, "[lora] unknown adapter_type '%s'\n", ty.c_str()); exit(1);
            }
        }
        ggml_backend_tensor_set(outs[ti], w.data(), 0, w.size()*sizeof(float));
        base.overrides[wname] = outs[ti];
    }
    return st;
}

} // namespace sa3
