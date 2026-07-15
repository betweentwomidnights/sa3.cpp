// Native C++ port of tools/convert_lora.py: convert an exported LoRA/DoRA (safetensors + json from
// lora_ckpt_export.py, all tensors saved as float32) into a sa3.cpp LoRA gguf, with NO Python.
//
// The exported .safetensors keys look like "<module>.parametrizations.weight.0.<kind>" where kind is
// lora_A|lora_B|magnitude|magnitude_r|magnitude_c|U|V|M_xs. Each <module> is mapped to the base DiT weight
// it adapts (reusing convert_dit's rename) and written as "<base>.<kind>"; adapter metadata (type/rank/alpha)
// goes in the gguf KV store. The C++ apply pass (src/lora.h) recomputes W_eff, so the base gguf is untouched.
//
// This is the same output as tools/convert_lora.py, produced without numpy/gguf/safetensors Python deps —
// so an embedded host (libsa3) can import a .safetensors LoRA in-process.
#pragma once

#include "ggml.h"
#include "gguf.h"
#include "yyjson.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sa3 {

// Port of tools/convert_dit.py rename(): "model.model.<...>.weight" -> "dit....weight" | "" (unmapped).
inline std::string dit_rename(const std::string& k) {
    static const std::string PREF = "model.model.";
    static const std::string TR   = "transformer.";
    if (k.rfind(PREF, 0) != 0) return "";
    const std::string r = k.substr(PREF.size());
    static const std::unordered_map<std::string, std::string> direct = {
        {"preprocess_conv.weight",  "dit.pre_conv.weight"},
        {"postprocess_conv.weight", "dit.post_conv.weight"},
        {"to_cond_embed.0.weight",   "dit.cond_embed.0.weight"},
        {"to_cond_embed.2.weight",   "dit.cond_embed.2.weight"},
        {"to_global_embed.0.weight", "dit.global_embed.0.weight"},
        {"to_global_embed.2.weight", "dit.global_embed.2.weight"},
        {"to_timestep_embed.0.weight", "dit.time_embed.0.weight"},
        {"to_timestep_embed.0.bias",   "dit.time_embed.0.bias"},
        {"to_timestep_embed.2.weight", "dit.time_embed.2.weight"},
        {"to_timestep_embed.2.bias",   "dit.time_embed.2.bias"},
        {TR + "global_cond_embedder.0.weight", "dit.gce.0.weight"},
        {TR + "global_cond_embedder.0.bias",   "dit.gce.0.bias"},
        {TR + "global_cond_embedder.2.weight", "dit.gce.2.weight"},
        {TR + "global_cond_embedder.2.bias",   "dit.gce.2.bias"},
        {TR + "memory_tokens",   "dit.memory_tokens"},
        {TR + "project_in.weight",  "dit.proj_in.weight"},
        {TR + "project_out.weight", "dit.proj_out.weight"},
    };
    if (auto it = direct.find(r); it != direct.end()) return it->second;
    if (r == TR + "rotary_pos_emb.inv_freq") return "";   // recomputed in C++
    const std::string layers = TR + "layers.";
    if (r.rfind(layers, 0) == 0) {
        const std::string rest = r.substr(layers.size());   // "<i>.<sub>"
        const size_t dot = rest.find('.');
        if (dot == std::string::npos) return "";
        const std::string i   = rest.substr(0, dot);
        const std::string sub = rest.substr(dot + 1);
        static const std::unordered_map<std::string, std::string> m = {
            {"pre_norm.gamma",          "pre_norm.gamma"},
            {"ff_norm.gamma",           "ff_norm.gamma"},
            {"cross_attend_norm.gamma", "cross_norm.gamma"},
            {"self_attn.to_qkv.weight", "self.qkv.weight"},
            {"self_attn.to_out.weight", "self.out.weight"},
            {"self_attn.q_norm.gamma",  "self.q_norm.gamma"},
            {"self_attn.k_norm.gamma",  "self.k_norm.gamma"},
            {"cross_attn.to_q.weight",  "cross.q.weight"},
            {"cross_attn.to_kv.weight", "cross.kv.weight"},
            {"cross_attn.to_out.weight","cross.out.weight"},
            {"cross_attn.q_norm.gamma", "cross.q_norm.gamma"},
            {"cross_attn.k_norm.gamma", "cross.k_norm.gamma"},
            {"ff.ff.0.proj.weight",     "ff.proj.weight"},
            {"ff.ff.0.proj.bias",       "ff.proj.bias"},
            {"ff.ff.2.weight",          "ff.out.weight"},
            {"ff.ff.2.bias",            "ff.out.bias"},
            {"to_scale_shift_gate",     "ssg"},
            {"to_local_embed.0.weight", "local.0.weight"},
            {"to_local_embed.0.bias",   "local.0.bias"},
            {"to_local_embed.2.weight", "local.2.weight"},
            {"to_local_embed.2.bias",   "local.2.bias"},
        };
        if (auto it = m.find(sub); it != m.end()) return "dit." + i + "." + it->second;
    }
    return "";
}

// LoRA module name -> base DiT weight stem (without ".weight"), or "" if not a DiT weight
// (e.g. conditioners.* — not in the DiT gguf). Mirrors convert_lora.py base_name().
inline std::string lora_base_name(const std::string& module) {
    static const std::string M = "model.";
    if (module.rfind(M, 0) != 0) return "";
    const std::string full = "model.model." + module.substr(M.size()) + ".weight";
    const std::string n = dit_rename(full);
    static const std::string suf = ".weight";
    if (n.size() >= suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0)
        return n.substr(0, n.size() - suf.size());
    return "";
}

// Split "<module>.parametrizations.weight.0.<kind>" -> (module, kind); kind must be a known adapter part.
inline bool lora_parse_key(const std::string& key, std::string& module, std::string& kind) {
    static const std::string MID = ".parametrizations.weight.0.";
    const size_t p = key.find(MID);
    if (p == std::string::npos) return false;
    module = key.substr(0, p);
    kind   = key.substr(p + MID.size());
    static const std::unordered_map<std::string, int> kinds = {
        {"lora_A", 1}, {"lora_B", 1}, {"magnitude", 1}, {"magnitude_r", 1},
        {"magnitude_c", 1}, {"U", 1}, {"V", 1}, {"M_xs", 1},
    };
    return kinds.count(kind) != 0;
}

// Convert safetensors_path (+ json_path config) to a sa3-lora gguf at out_gguf_path. false + err on failure.
inline bool convert_lora_safetensors(const std::string& safetensors_path,
                                     const std::string& json_path,
                                     const std::string& out_gguf_path,
                                     std::string& err) {
    // --- config json: adapter_type / rank / alpha ---
    std::string adapter_type = "lora";
    uint32_t rank = 0;
    float alpha = 1.0f;
    {
        std::ifstream f(json_path, std::ios::binary);
        if (!f) { err = "cannot open " + json_path; return false; }
        std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        yyjson_doc* doc = yyjson_read(js.data(), js.size(), 0);
        if (!doc) { err = "invalid json: " + json_path; return false; }
        yyjson_val* root = yyjson_doc_get_root(doc);
        if (yyjson_val* v = yyjson_obj_get(root, "adapter_type"); v && yyjson_is_str(v)) adapter_type = yyjson_get_str(v);
        if (yyjson_val* v = yyjson_obj_get(root, "rank"); v && yyjson_is_num(v)) rank = (uint32_t)yyjson_get_num(v);
        if (yyjson_val* v = yyjson_obj_get(root, "alpha"); v && yyjson_is_num(v)) alpha = (float)yyjson_get_num(v);
        yyjson_doc_free(doc);
    }
    if (rank == 0) { err = "config missing/zero 'rank' in " + json_path; return false; }

    // --- safetensors: read whole file, parse the JSON header, then the F32 tensor blobs ---
    std::vector<uint8_t> buf;
    {
        std::ifstream f(safetensors_path, std::ios::binary | std::ios::ate);
        if (!f) { err = "cannot open " + safetensors_path; return false; }
        const std::streamsize sz = f.tellg();
        if (sz < 8) { err = "truncated safetensors: " + safetensors_path; return false; }
        buf.resize((size_t)sz);
        f.seekg(0);
        f.read((char*)buf.data(), sz);
    }
    uint64_t hlen = 0;
    std::memcpy(&hlen, buf.data(), 8);   // little-endian header length
    if (8 + hlen > buf.size()) { err = "bad safetensors header length"; return false; }
    const size_t data_base = 8 + (size_t)hlen;

    // one adapter tensor to emit: gguf name + shape (safetensors order) + pointer into buf
    struct Emit { std::string name; std::vector<int64_t> shape; const void* data; size_t n; bool f16; };
    std::vector<Emit> emits;
    int n_targets = 0;
    {
        yyjson_doc* doc = yyjson_read((char*)buf.data() + 8, (size_t)hlen, 0);
        if (!doc) { err = "invalid safetensors header json"; return false; }
        yyjson_val* root = yyjson_doc_get_root(doc);
        // count distinct mapped modules for lora.n_targets (parity with convert_lora.py)
        std::unordered_map<std::string, int> mapped_modules;
        yyjson_obj_iter it; yyjson_obj_iter_init(root, &it);
        yyjson_val* key;
        while ((key = yyjson_obj_iter_next(&it))) {
            const std::string kname = yyjson_get_str(key);
            if (kname == "__metadata__") continue;
            std::string module, kind;
            if (!lora_parse_key(kname, module, kind)) continue;   // unrecognized key -> skip (like the python)
            const std::string base = lora_base_name(module);
            if (base.empty()) continue;                            // non-DiT (conditioners.*) -> skip
            yyjson_val* t = yyjson_obj_iter_get_val(key);
            yyjson_val* vd = yyjson_obj_get(t, "dtype");
            const std::string dtype = vd ? yyjson_get_str(vd) : "?";
            if (dtype != "F32" && dtype != "F16") {   // f16 exports (gary4local fp16 training) are upcast below
                err = "expected F32/F16 tensor, got '" + dtype + "' for " + kname;
                yyjson_doc_free(doc); return false;
            }
            Emit e; e.name = base + "." + kind; e.f16 = dtype == "F16";
            yyjson_val* vs = yyjson_obj_get(t, "shape");
            size_t n = 1; yyjson_val* dv; yyjson_arr_iter ai; yyjson_arr_iter_init(vs, &ai);
            while ((dv = yyjson_arr_iter_next(&ai))) { const int64_t d = yyjson_get_sint(dv); e.shape.push_back(d); n *= (size_t)d; }
            yyjson_val* vo = yyjson_obj_get(t, "data_offsets");
            const uint64_t begin = yyjson_get_uint(yyjson_arr_get(vo, 0));
            e.data = buf.data() + data_base + begin;
            e.n = n;
            emits.push_back(std::move(e));
            mapped_modules[module] = 1;
        }
        n_targets = (int)mapped_modules.size();
        yyjson_doc_free(doc);
    }
    if (emits.empty()) { err = "no DiT LoRA tensors found in " + safetensors_path; return false; }

    // --- build the gguf: KV + tensors (ggml ne = reversed safetensors shape) ---
    size_t data_bytes = 0;
    for (const Emit& e : emits) data_bytes += e.n * sizeof(float);
    const size_t mem = data_bytes + emits.size() * (ggml_tensor_overhead() + 64) + (1u << 20);
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) { err = "ggml_init failed"; return false; }
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "sa3-lora");
    gguf_set_val_str(g, "general.name", ("sa3 " + adapter_type + " adapter").c_str());
    gguf_set_val_str(g, "lora.adapter_type", adapter_type.c_str());
    gguf_set_val_u32(g, "lora.rank", rank);
    gguf_set_val_f32(g, "lora.alpha", alpha);
    gguf_set_val_u32(g, "lora.n_targets", (uint32_t)n_targets);

    for (const Emit& e : emits) {
        const int nd = (int)e.shape.size();
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int d = 0; d < nd; ++d) ne[d] = e.shape[nd - 1 - d];   // reverse to ggml ne order
        ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        ggml_set_name(t, e.name.c_str());
        if (e.f16) ggml_fp16_to_fp32_row((const ggml_fp16_t*)e.data, (float*)t->data, (int64_t)e.n);
        else       std::memcpy(t->data, e.data, e.n * sizeof(float));
        gguf_add_tensor(g, t);
    }

    const bool ok = gguf_write_to_file(g, out_gguf_path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) { err = "failed to write " + out_gguf_path; return false; }
    return true;
}

} // namespace sa3
