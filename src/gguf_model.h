// gguf_model.h — minimal, model-agnostic GGUF weight loader for GGML.
// Loads every tensor in a .gguf into a backend buffer and exposes lookups by
// name plus typed metadata accessors. Reusable by any GGML port.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace sa3 {

// Pick the compute backend: a registered GPU device (CUDA when the CUDA backend is
// linked) unless SA3_DEVICE=cpu forces CPU. In a CPU-only build the registry has no
// GPU device, so this transparently returns the CPU backend — same code, both builds.
//
// Device choice among GPUs: the Vulkan backend registers EVERY Vulkan device, so on a
// laptop with an Intel iGPU + a discrete NVIDIA GPU the first-by-type device may be the
// iGPU (wrong: tiny VRAM, slow). We therefore enumerate all GPU devices and, by default,
// prefer the one with the most total memory (the discrete GPU). SA3_GPU overrides this:
// a 0-based index into the GPU list, or a case-insensitive substring of the device name
// (e.g. SA3_GPU=nvidia). CUDA builds expose a single GPU device, so this is a no-op there.
inline ggml_backend_t make_backend() {
    const char* dev = getenv("SA3_DEVICE");
    if (!(dev && strcmp(dev, "cpu") == 0)) {
        // Collect all GPU devices in registry order.
        std::vector<ggml_backend_dev_t> gpus;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) gpus.push_back(d);
        }
        if (!gpus.empty()) {
            ggml_backend_dev_t chosen = nullptr;
            const char* sel = getenv("SA3_GPU");
            if (sel && *sel) {
                // Try index first; a pure-integer string selects by position.
                char* end = nullptr;
                long idx = strtol(sel, &end, 10);
                if (end && *end == '\0' && idx >= 0 && (size_t)idx < gpus.size()) {
                    chosen = gpus[(size_t)idx];
                } else {
                    // Otherwise match a substring of the device name (case-insensitive).
                    for (ggml_backend_dev_t d : gpus) {
                        std::string name = ggml_backend_dev_name(d);
                        std::string desc = ggml_backend_dev_description(d);
                        std::string hay = name + " " + desc, needle = sel;
                        for (char& c : hay)    c = (char)tolower((unsigned char)c);
                        for (char& c : needle) c = (char)tolower((unsigned char)c);
                        if (hay.find(needle) != std::string::npos) { chosen = d; break; }
                    }
                    if (!chosen) fprintf(stderr, "[sa3] SA3_GPU='%s' matched no device; using default\n", sel);
                }
            }
            if (!chosen) {
                // Default: the GPU with the most total memory (the discrete one).
                size_t best = 0;
                for (ggml_backend_dev_t d : gpus) {
                    size_t free = 0, total = 0;
                    ggml_backend_dev_memory(d, &free, &total);
                    if (total > best) { best = total; chosen = d; }
                }
                if (!chosen) chosen = gpus[0];
            }
            ggml_backend_t b = ggml_backend_dev_init(chosen, nullptr);
            if (b) {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(chosen, &free, &total);
                fprintf(stderr, "[sa3] backend: %s (%s, %.1f GB)\n",
                        ggml_backend_name(b), ggml_backend_dev_description(chosen),
                        total / (1024.0 * 1024.0 * 1024.0));
                return b;
            }
        }
    }
    return ggml_backend_cpu_init();
}

struct GgufModel {
    ggml_context*         ctx     = nullptr;
    gguf_context*         gguf    = nullptr;
    ggml_backend_t        backend = nullptr;
    bool                  owns_backend = true;
    ggml_backend_buffer_t buf     = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::map<std::string, ggml_tensor*> overrides;   // LoRA-effective weights (see lora.h)

    GgufModel() = default;
    ~GgufModel() { free(); }
    GgufModel(const GgufModel&) = delete;
    GgufModel& operator=(const GgufModel&) = delete;
    GgufModel(GgufModel&& other) noexcept { move_from(other); }
    GgufModel& operator=(GgufModel&& other) noexcept {
        if (this != &other) {
            free();
            move_from(other);
        }
        return *this;
    }

    bool has(const std::string& n) const { return tensors.count(n) != 0; }

    ggml_tensor* get(const std::string& n) const {
        auto ov = overrides.find(n);                 // LoRA W_eff transparently shadows the base weight
        if (ov != overrides.end()) return ov->second;
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
        if (backend && owns_backend) ggml_backend_free(backend);
        buf = nullptr; gguf = nullptr; ctx = nullptr; backend = nullptr;
        tensors.clear(); overrides.clear(); owns_backend = true;
    }

private:
    void move_from(GgufModel& other) noexcept {
        ctx = other.ctx;
        gguf = other.gguf;
        backend = other.backend;
        owns_backend = other.owns_backend;
        buf = other.buf;
        tensors = std::move(other.tensors);
        overrides = std::move(other.overrides);

        other.ctx = nullptr;
        other.gguf = nullptr;
        other.backend = nullptr;
        other.owns_backend = true;
        other.buf = nullptr;
        other.tensors.clear();
        other.overrides.clear();
    }
};

// Load a GGUF into the given backend (CPU if null). Reads tensor data straight
// from the file at the offsets gguf reports.
inline GgufModel load_gguf(const char* path, ggml_backend_t backend = nullptr) {
    GgufModel m;
    m.backend = backend ? backend : make_backend();
    m.owns_backend = backend == nullptr;

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
#ifdef _WIN32
        _fseeki64(f, (long long)off, SEEK_SET);   // MSVC `long` is 32-bit; files exceed 2 GB
#else
        fseeko(f, (off_t)off, SEEK_SET);
#endif
        if (fread(tmp.data(), 1, nb, f) != nb) { fprintf(stderr, "[gguf] short read %s\n", name); exit(1); }
        ggml_backend_tensor_set(cur, tmp.data(), 0, nb);
        m.tensors[name] = cur;
    }
    fclose(f);
    return m;
}

} // namespace sa3
