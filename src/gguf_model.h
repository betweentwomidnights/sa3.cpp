// gguf_model.h — minimal, model-agnostic GGUF weight loader for GGML.
// Loads every tensor in a .gguf into a backend buffer and exposes lookups by
// name plus typed metadata accessors. Reusable by any GGML port.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace sa3 {

struct GgufModel {
    ggml_context*         ctx     = nullptr;
    gguf_context*         gguf    = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    bool has(const std::string& n) const { return tensors.count(n) != 0; }

    ggml_tensor* get(const std::string& n) const {
        auto it = tensors.find(n);
        if (it == tensors.end()) { fprintf(stderr, "[gguf] missing tensor: %s\n", n.c_str()); exit(1); }
        return it->second;
    }

    uint32_t u32(const char* k) const {
        int i = gguf_find_key(gguf, k);
        if (i < 0) { fprintf(stderr, "[gguf] missing key: %s\n", k); exit(1); }
        return gguf_get_val_u32(gguf, i);
    }
    float f32(const char* k) const {
        int i = gguf_find_key(gguf, k);
        if (i < 0) { fprintf(stderr, "[gguf] missing key: %s\n", k); exit(1); }
        return gguf_get_val_f32(gguf, i);
    }
    // read a single-element tensor's value to the host
    float scalar(const std::string& n) const {
        float v = 0.0f;
        ggml_backend_tensor_get(get(n), &v, 0, sizeof(float));
        return v;
    }

    void free() {
        if (buf)     ggml_backend_buffer_free(buf);
        if (gguf)    gguf_free(gguf);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

// Load a GGUF into the given backend (CPU if null). Reads tensor data straight
// from the file at the offsets gguf reports.
inline GgufModel load_gguf(const char* path, ggml_backend_t backend = nullptr) {
    GgufModel m;
    m.backend = backend ? backend : ggml_backend_cpu_init();

    gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/&m.ctx };
    m.gguf = gguf_init_from_file(path, gp);
    if (!m.gguf) { fprintf(stderr, "[gguf] failed to open %s\n", path); exit(1); }

    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.backend);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[gguf] cannot read %s\n", path); exit(1); }
    const size_t data_off = gguf_get_data_offset(m.gguf);
    std::vector<char> tmp;
    for (ggml_tensor* cur = ggml_get_first_tensor(m.ctx); cur; cur = ggml_get_next_tensor(m.ctx, cur)) {
        const char* name = ggml_get_name(cur);
        const int idx = gguf_find_tensor(m.gguf, name);
        const size_t off = data_off + gguf_get_tensor_offset(m.gguf, idx);
        const size_t nb  = ggml_nbytes(cur);
        tmp.resize(nb);
        fseek(f, (long)off, SEEK_SET);
        if (fread(tmp.data(), 1, nb, f) != nb) { fprintf(stderr, "[gguf] short read %s\n", name); exit(1); }
        ggml_backend_tensor_set(cur, tmp.data(), 0, nb);
        m.tensors[name] = cur;
    }
    fclose(f);
    return m;
}

} // namespace sa3
