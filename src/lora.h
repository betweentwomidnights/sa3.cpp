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

inline LoraAdapter load_lora(const char* path, float strength = 1.0f, ggml_backend_t backend = nullptr) {
    LoraAdapter a;
    a.gguf = load_gguf(path, backend);   // load onto the base's backend so the GPU apply graph can read it
    int ti = gguf_find_key(a.gguf.gguf, "lora.adapter_type");
    a.type     = ti < 0 ? "lora" : gguf_get_val_str(a.gguf.gguf, ti);
    a.rank     = (int)a.gguf.u32("lora.rank");
    a.alpha    = a.gguf.f32("lora.alpha");
    a.strength = strength;
    return a;
}

// Read any tensor (F32 or F16) into an F32 host buffer.
inline void read_to_f32(ggml_tensor* t, std::vector<float>& dst) {
    const int64_t n = ggml_nelements(t);
    dst.resize(n);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n*sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), dst.data(), n);
    } else {
        ggml_backend_tensor_get(t, dst.data(), 0, n*sizeof(float));
    }
}
// Write an F32 host buffer into a tensor of t->type (F32 or F16) — keeps W_eff at the base dtype.
inline void write_from_f32(ggml_tensor* t, const std::vector<float>& src) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_fp32_to_fp16_row(src.data(), tmp.data(), n);
        ggml_backend_tensor_set(t, tmp.data(), 0, n*sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(t, src.data(), 0, n*sizeof(float));
    }
}

// W_eff storage: own a context + persistent backend buffer holding the override tensors.
struct LoraStack {
    ggml_context*         ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    void free() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        ctx = nullptr; buf = nullptr;
    }
};

// GPU/graph path handles additive lora + dora-rows (covers our DoRA adapters). The
// column-norm / -xs families stay on the host fallback (rare; no trained test adapters).
inline bool lora_graph_ok(const std::vector<LoraAdapter>& adapters) {
    for (auto& a : adapters) if (a.type != "lora" && a.type != "dora-rows") return false;
    return true;
}

// Build one ggml graph that recomputes every W_eff and run it on base.backend (GPU when CUDA).
//   delta = B@A ; V = W + (alpha/rank)*strength*delta ;
//   dora-rows: W_eff = magnitude[:,None] * V / ||V||_row  (= magnitude * rms_norm(V)/sqrt(in)).
// W_eff is written back IN-PLACE over the base weight (cast(W0) reads it once up front, the final
// ggml_cpy writes it last — no aliasing), so there's no second copy of the model in VRAM. Adapters
// must already be loaded on base.backend (load_lora(..., base.backend)).
inline LoraStack apply_loras_graph(GgufModel& base, std::vector<LoraAdapter>& adapters,
                                   const std::vector<std::string>& targets) {
    const size_t nn = targets.size()*20 + 64;
    ggml_init_params ip = { nn*ggml_tensor_overhead() + ggml_graph_overhead_custom(nn, false) + (1<<20), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, nn, false);
    for (auto& wname : targets) {
        std::string stem = wname.substr(0, wname.size()-7);
        ggml_tensor* W0 = base.tensors[wname];
        const int64_t in = W0->ne[0], out = W0->ne[1];
        ggml_tensor* Wc = ggml_cast(ctx, W0, GGML_TYPE_F32);                          // running weight [in,out] (f32 copy)
        for (auto& a : adapters) {
            if (!a.gguf.has(stem + ".lora_A") || a.strength == 0.0f) continue;
            ggml_tensor* A = a.gguf.get(stem+".lora_A");                              // [in, rank]
            ggml_tensor* B = a.gguf.get(stem+".lora_B");                              // [rank, out]
            ggml_tensor* delta = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, A)), B); // [in,out]
            ggml_tensor* V = ggml_add(ctx, Wc, ggml_scale(ctx, delta, (a.alpha/a.rank)*a.strength));
            if (a.type == "dora-rows") {
                ggml_tensor* mag = ggml_reshape_2d(ctx, a.gguf.get(stem+".magnitude"), 1, out); // [1,out]
                ggml_tensor* Vn = ggml_scale(ctx, ggml_rms_norm(ctx, V, 1e-12f), 1.0f/sqrtf((float)in));
                Wc = ggml_mul(ctx, Vn, mag);
            } else {
                Wc = V;                                                               // additive lora
            }
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Wc, W0));                         // write W_eff back over W0
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(base.backend));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_graph_compute(base.backend, gf);
    ggml_gallocr_free(alloc); ggml_free(ctx);
    return LoraStack{};   // weights modified in place; nothing to keep
}

// Host fallback: compute W_eff per target with plain loops (all 8 adapter families), in place.
inline LoraStack apply_loras_host(GgufModel& base, std::vector<LoraAdapter>& adapters,
                                  const std::vector<std::string>& targets) {
    // compute W_eff per target (host), chaining adapters in order. Buffers reused across targets.
    // W is the running effective weight, row-major w[o*in+i]. For each adapter:
    //   delta = B@A  (standard)  or  U@M_xs@V^T  (-xs);  V = W + (alpha/rank)*strength*delta;
    //   then a per-type post-transform (additive / dora-rows / dora-cols / bora). Spec: lora_spec_test.py.
    std::vector<float> w, A, B, U, Vm, M, mv, dl, Vb, mag, magc, inter, coln;
    auto getv = [&](GgufModel& g, const std::string& n, std::vector<float>& dst){ read_to_f32(g.get(n), dst); };
    for (size_t ti = 0; ti < targets.size(); ti++) {
        const std::string& wname = targets[ti];
        std::string stem = wname.substr(0, wname.size()-7);
        ggml_tensor* W0 = base.tensors[wname];
        const int in = (int)W0->ne[0], out = (int)W0->ne[1];
        read_to_f32(W0, w);                                                  // w[o*in + i] (handles f16 base)

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
        write_from_f32(W0, w);                                               // write W_eff back over W0 (in place)
    }
    return LoraStack{};
}

// Apply `adapters` (in order) to `base`, filling base.overrides with W_eff for every targeted
// weight. Dispatches to the GPU/graph path for additive+dora-rows, else the host fallback.
inline LoraStack apply_loras(GgufModel& base, std::vector<LoraAdapter>& adapters) {
    std::vector<std::string> targets;
    for (auto& kv : base.tensors) {
        const std::string& wname = kv.first;
        if (wname.size() < 7 || wname.compare(wname.size()-7, 7, ".weight") != 0) continue;
        std::string stem = wname.substr(0, wname.size()-7);
        for (auto& a : adapters)
            if (a.gguf.has(stem + ".lora_A") || a.gguf.has(stem + ".M_xs")) { targets.push_back(wname); break; }
    }
    return lora_graph_ok(adapters) ? apply_loras_graph(base, adapters, targets)
                                   : apply_loras_host(base, adapters, targets);
}

} // namespace sa3
