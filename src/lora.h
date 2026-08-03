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

#include "dit.h"          // DitLora / DitLoraParam, for the functional (unmerged) path below
#include "ggml.h"
#include "gguf_model.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
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

// Read any tensor (F32, F16, or a quantized type) into an F32 host buffer.
inline void read_to_f32(ggml_tensor* t, std::vector<float>& dst) {
    const int64_t n = ggml_nelements(t);
    dst.resize(n);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n*sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), dst.data(), n);
    } else if (ggml_is_quantized(t->type)) {
        // Pull the packed bytes down and unpack with the type's own reference dequantizer.
        // Asking ggml_backend_tensor_get for n floats would over-read the tensor (a Q4_K row
        // is ~0.56 bytes/element) and trip its bounds assert.
        const size_t nbytes = ggml_nbytes(t);
        std::vector<char> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const ggml_type_traits* tr = ggml_get_type_traits(t->type);
        const int64_t ne0 = t->ne[0], rows = n / ne0;
        const size_t row_bytes = ggml_row_size(t->type, ne0);
        for (int64_t r = 0; r < rows; r++)
            tr->to_float(raw.data() + (size_t)r*row_bytes, dst.data() + r*ne0, ne0);
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
                throw std::runtime_error("[lora] unknown adapter_type '" + ty + "'");
            }
        }
        write_from_f32(W0, w);                                               // write W_eff back over W0 (in place)
    }
    return LoraStack{};
}

// Apply `adapters` (in order) to `base`, updating every targeted weight in place. Dispatches to
// the GPU/graph path for additive+dora-rows, else the host fallback.
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

// ---------------------------------------------------------------------------------------
// Functional (graph-time) adapters — the path that works over a QUANTIZED base.
//
// apply_loras above rewrites W in place, which needs to read W out and write W_eff back.
// Neither has a route for k-quant types: the graph path dies in ggml_cast (no q6_K->f32 cpy)
// and the host path cannot re-pack an f32 result into Q4_K storage. So a quantized base
// fundamentally cannot carry a merged adapter.
//
// Instead hand the adapter tensors to dit_lin (src/dit.h), which folds them into the graph:
//     y = W@x + scale·B@(A@x)                                    [+ dora row scale]
// W is only ever an argument to mul_mat, and quantized mul_mat is supported on every
// backend, so the base never has to be dequantized or rewritten. This is the same
// formulation the training graph already uses; inference simply passed dl = nullptr.
//
// Limits (both fall back to the merge path):
//   - one adapter at a time. DitLoraParam holds a single A/B per target, and DoRA does not
//     commute, so composing several functionally is not a matter of summing them.
//   - "lora" and "dora-rows" only, matching what dit_lin implements.
struct FunctionalLora {
    DitLora               map;
    ggml_context*         ctx = nullptr;   // owns the base_norm_sq constants
    ggml_backend_buffer_t buf = nullptr;
    bool active = false;
    void free() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        ctx = nullptr; buf = nullptr; map.clear(); active = false;
    }
};

// base_norm_sq depends only on the base weights, never on the adapter, so it is the same for
// every LoRA ever loaded against a given model. Computing it means pulling every targeted
// weight to the host and summing squares, which costs seconds; cache it in a sidecar next to
// the gguf so only the first run pays. Keyed on the model's byte size, which changes for any
// re-conversion or re-quantization.
namespace normcache {

inline std::string path_for(const std::string& model_path) { return model_path + ".norms"; }

inline bool load(const std::string& model_path, std::map<std::string, std::vector<float>>& out) {
    std::error_code ec;
    const uintmax_t model_sz = std::filesystem::file_size(model_path, ec);
    if (ec) return false;
    std::ifstream f(path_for(model_path), std::ios::binary);
    if (!f) return false;
    char magic[8] = {0};
    uint64_t src_sz = 0, n = 0;
    f.read(magic, 8);
    if (std::memcmp(magic, "SA3NORM1", 8) != 0) return false;
    f.read((char*)&src_sz, 8);
    if (!f || src_sz != (uint64_t)model_sz) return false;      // model changed -> recompute
    f.read((char*)&n, 8);
    if (!f || n > (1u << 20)) return false;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t nl = 0; uint64_t cnt = 0;
        f.read((char*)&nl, 4);
        if (!f || nl == 0 || nl > 512) return false;
        std::string name(nl, '\0');
        f.read(&name[0], nl);
        f.read((char*)&cnt, 8);
        if (!f || cnt == 0 || cnt > (1u << 24)) return false;
        std::vector<float> v(cnt);
        f.read((char*)v.data(), (std::streamsize)(cnt*sizeof(float)));
        if (!f) return false;
        out[name] = std::move(v);
    }
    return true;
}

inline void save(const std::string& model_path, const std::map<std::string, std::vector<float>>& in) {
    std::error_code ec;
    const uintmax_t model_sz = std::filesystem::file_size(model_path, ec);
    if (ec) return;
    // Write to a temp then rename, so a killed run cannot leave a half-written cache that
    // would later be read as valid.
    const std::string tmp = path_for(model_path) + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        const uint64_t src_sz = (uint64_t)model_sz, n = in.size();
        f.write("SA3NORM1", 8);
        f.write((const char*)&src_sz, 8);
        f.write((const char*)&n, 8);
        for (auto& kv : in) {
            const uint32_t nl = (uint32_t)kv.first.size();
            const uint64_t cnt = kv.second.size();
            f.write((const char*)&nl, 4);
            f.write(kv.first.data(), nl);
            f.write((const char*)&cnt, 8);
            f.write((const char*)kv.second.data(), (std::streamsize)(cnt*sizeof(float)));
        }
        if (!f) { f.close(); std::filesystem::remove(tmp, ec); return; }
    }
    std::filesystem::rename(tmp, path_for(model_path), ec);
    if (ec) std::filesystem::remove(tmp, ec);
}

} // namespace normcache

inline bool functional_lora_ok(const std::vector<LoraAdapter>& adapters) {
    if (adapters.size() != 1) return false;
    const std::string& t = adapters[0].type;
    return t == "lora" || t == "dora-rows";
}

// Build the dit_lin adapter map for `adapters[0]` over `base`. Adapter tensors must already
// live on base.backend (load_lora(..., base.backend)); only the dora-rows base_norm_sq
// constants are allocated here, computed on the host from W (quantized W included).
inline FunctionalLora build_functional_lora(GgufModel& base, std::vector<LoraAdapter>& adapters,
                                            const std::string& base_path = "") {
    FunctionalLora fl;
    if (!functional_lora_ok(adapters)) return fl;
    LoraAdapter& a = adapters[0];
    const bool dora = (a.type == "dora-rows");

    std::map<std::string, std::vector<float>> nsq_cache;
    bool cache_hit = false, cache_dirty = false;
    if (dora && !base_path.empty()) cache_hit = normcache::load(base_path, nsq_cache);

    std::vector<std::string> targets;
    for (auto& kv : base.tensors) {
        const std::string& wname = kv.first;
        if (wname.size() < 7 || wname.compare(wname.size()-7, 7, ".weight") != 0) continue;
        if (a.gguf.has(wname.substr(0, wname.size()-7) + ".lora_A")) targets.push_back(wname);
    }
    if (targets.empty()) return fl;

    // Two persistent f32 [out] constants per dora-rows target: base_norm_sq (host-computed)
    // and row_scale (filled by the one-shot graph below).
    ggml_init_params ip = { (2*targets.size()+4)*ggml_tensor_overhead(), nullptr, true };
    fl.ctx = ggml_init(ip);
    std::vector<ggml_tensor*> norms, scales;
    std::vector<std::vector<float>> norm_host;
    for (auto& wname : targets) {
        ggml_tensor* W0 = base.tensors[wname];
        const int64_t in = W0->ne[0], out = W0->ne[1];
        std::string stem = wname.substr(0, wname.size()-7);
        DitLoraParam p;
        p.A     = a.gguf.get(stem + ".lora_A");
        p.B     = a.gguf.get(stem + ".lora_B");
        p.scale = (a.alpha / a.rank) * a.strength;
        p.dora  = dora;
        p.in    = in;
        if (dora) {
            p.magnitude = a.gguf.get(stem + ".magnitude");
            // base_norm_sq[o] = Σ_in W[o,i]² + in·eps, matching train_row_norm_sq. Reading W
            // here is the one place the base is touched outside a matmul, and read_to_f32
            // dequantizes on the host, so it is fine for a quantized base. It is also the
            // expensive part, hence the sidecar cache.
            std::vector<float> nsq;
            auto it = nsq_cache.find(wname);
            if (it != nsq_cache.end() && it->second.size() == (size_t)out) {
                nsq = it->second;
            } else {
                std::vector<float> w;  read_to_f32(W0, w);
                nsq.assign((size_t)out, 0.0f);
                for (int64_t o = 0; o < out; o++) {
                    double s = 0.0;
                    const float* wo = &w[(size_t)o*in];
                    for (int64_t i = 0; i < in; i++) s += (double)wo[i]*wo[i];
                    nsq[(size_t)o] = (float)(s + (double)in * 1e-12);
                }
                nsq_cache[wname] = nsq;
                cache_dirty = true;
            }
            ggml_tensor* nt = ggml_new_tensor_1d(fl.ctx, GGML_TYPE_F32, out);
            ggml_tensor* st = ggml_new_tensor_1d(fl.ctx, GGML_TYPE_F32, out);
            norms.push_back(nt); scales.push_back(st); norm_host.push_back(std::move(nsq));
            p.base_norm_sq = nt;
            p.row_scale    = st;
        }
        fl.map[wname] = p;
    }
    fl.buf = ggml_backend_alloc_ctx_tensors(fl.ctx, base.backend);
    for (size_t i = 0; i < norms.size(); i++)
        ggml_backend_tensor_set(norms[i], norm_host[i].data(), 0, norm_host[i].size()*sizeof(float));
    if (cache_dirty && !base_path.empty()) {
        normcache::save(base_path, nsq_cache);
        printf("lora: wrote base-norm cache %s\n", normcache::path_for(base_path).c_str());
    } else if (cache_hit) {
        printf("lora: base-norm cache hit (%zu tensors)\n", nsq_cache.size());
    }

    // Fill row_scale once. Deliberately the same op sequence dit_lin uses for the per-step
    // path, so the shortcut cannot drift from the reference: nsq = base_norm_sq + 2s·t2 + s²·t3,
    // row_scale = magnitude / sqrt(nsq). Every input here is frozen at inference, which is why
    // this is a constant rather than per-step work.
    if (!scales.empty()) {
        const size_t nn = targets.size()*24 + 64;
        ggml_init_params tp = { nn*ggml_tensor_overhead() + ggml_graph_overhead_custom(nn, false) + (1<<20),
                                nullptr, true };
        ggml_context* tctx = ggml_init(tp);
        ggml_cgraph* gf = ggml_new_graph_custom(tctx, nn, false);
        size_t si = 0;
        for (auto& wname : targets) {
            DitLoraParam& p = fl.map[wname];
            ggml_tensor* wl  = base.tensors[wname];
            const int64_t out = wl->ne[1];
            const float s = p.scale;
            ggml_tensor* WtA  = ggml_mul_mat(tctx, wl, p.A);                            // [out,rank]
            ggml_tensor* Bt   = ggml_cont(tctx, ggml_transpose(tctx, p.B));             // [out,rank]
            ggml_tensor* t2   = ggml_sum_rows(tctx,
                ggml_cont(tctx, ggml_transpose(tctx, ggml_mul(tctx, WtA, Bt))));        // [1,out]
            ggml_tensor* AtA  = ggml_mul_mat(tctx, p.A, p.A);                           // [rank,rank]
            ggml_tensor* AtAB = ggml_mul_mat(tctx, AtA, p.B);                           // [rank,out]
            ggml_tensor* t3   = ggml_sum_rows(tctx, ggml_mul(tctx, p.B, AtAB));         // [1,out]
            ggml_tensor* nsq  = ggml_add(tctx,
                ggml_add(tctx, p.base_norm_sq,
                               ggml_scale(tctx, ggml_reshape_1d(tctx, t2, out), 2.0f*s)),
                ggml_scale(tctx, ggml_reshape_1d(tctx, t3, out), s*s));                 // [out]
            ggml_tensor* sv = ggml_div(tctx, p.magnitude, ggml_sqrt(tctx, nsq));        // [out]
            ggml_build_forward_expand(gf, ggml_cpy(tctx, sv, scales[si++]));
        }
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(base.backend));
        ggml_gallocr_alloc_graph(alloc, gf);
        ggml_backend_graph_compute(base.backend, gf);
        ggml_gallocr_free(alloc);
        ggml_free(tctx);
    }
    fl.active = true;
    return fl;
}

} // namespace sa3
