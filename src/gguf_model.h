// gguf_model.h — minimal, model-agnostic GGUF weight loader for GGML.
// Loads every tensor in a .gguf into a backend buffer and exposes lookups by
// name plus typed metadata accessors. Reusable by any GGML port.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/types.h>
#endif

namespace sa3 {

inline std::runtime_error gguf_error(const std::string& message) {
    return std::runtime_error("[gguf] " + message);
}

inline bool gguf_file_seek(FILE* f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
}

inline bool gguf_file_size(FILE* f, uint64_t& out) {
#ifdef _WIN32
    const long long cur = _ftelli64(f);
    if (cur < 0) return false;
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
    const long long end = _ftelli64(f);
    if (_fseeki64(f, cur, SEEK_SET) != 0 || end < 0) return false;
    out = (uint64_t)end;
    return true;
#else
    const off_t cur = ftello(f);
    if (cur < 0) return false;
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    const off_t end = ftello(f);
    if (fseeko(f, cur, SEEK_SET) != 0 || end < 0) return false;
    out = (uint64_t)end;
    return true;
#endif
}

inline std::string gguf_short_read_message(const char* path, const char* name,
                                           uint64_t off, size_t expected,
                                           size_t got, uint64_t file_size,
                                           bool have_file_size) {
    std::string msg = "short read while loading tensor " + std::string(name) +
        " from " + std::string(path) + ": expected " + std::to_string(expected) +
        " bytes at offset " + std::to_string(off) + ", got " + std::to_string(got);
    if (have_file_size) msg += " (file size " + std::to_string(file_size) + " bytes)";
    msg += ". The GGUF is probably incomplete; rerun models.sh/models.cmd to resume the download, "
           "or delete this file and download it again.";
    return msg;
}

inline int cpu_threads_from_env() {
    const char* v = getenv("SA3_THREADS");
    if (!v || !*v) return 0;
    char* end = nullptr;
    long n = strtol(v, &end, 10);
    if (!end || *end != '\0' || n <= 0 || n > 1024) {
        fprintf(stderr, "[sa3] ignoring invalid SA3_THREADS='%s' (expected a positive integer)\n", v);
        return 0;
    }
    return (int)n;
}

inline void load_dynamic_backends_once() {
    static bool loaded = false;
    if (!loaded) {
        ggml_backend_load_all();
        loaded = true;
    }
}

inline void configure_cpu_threads(ggml_backend_t b, int n_threads) {
    if (!b || n_threads <= 0) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) return;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto set_threads = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) {
        set_threads(b, n_threads);
        fprintf(stderr, "[sa3] CPU threads: %d\n", n_threads);
    }
}

// Pick the compute backend: a registered GPU/iGPU device (CUDA when the CUDA backend is
// linked) unless SA3_DEVICE=cpu forces CPU. In a CPU-only build the registry has no
// GPU/iGPU device, so this transparently returns the CPU backend — same code, both builds.
//
// Device choice among GPUs: the Vulkan backend registers EVERY Vulkan device, so on a
// laptop with an Intel iGPU + a discrete NVIDIA GPU the first-by-type device may be the
// iGPU (wrong: tiny VRAM, slow). We therefore enumerate all GPU devices and, by default,
// prefer a discrete GPU, then the one with the most total memory. If only an integrated
// GPU/APU is present, use that instead of silently falling back to CPU. SA3_GPU overrides this:
// a 0-based index into the GPU/iGPU list, or a case-insensitive substring of the device name
// (e.g. SA3_GPU=nvidia). CUDA builds expose a single GPU device, so this is a no-op there.
// device: explicit device request from the API (nullptr/empty falls back to the
// SA3_DEVICE env var, so CLI usage is unchanged). "cpu" forces the CPU backend;
// anything else selects a GPU (optionally narrowed by SA3_GPU) with CPU fallback.
inline ggml_backend_t make_backend(int cpu_threads = 0, const char* device = nullptr) {
    load_dynamic_backends_once();
    std::string dev_str = (device && *device) ? device : "";
    if (dev_str.empty()) { const char* e = getenv("SA3_DEVICE"); if (e) dev_str = e; }
    const char* dev = dev_str.empty() ? nullptr : dev_str.c_str();
    if (!(dev && strcmp(dev, "cpu") == 0)) {
        // Collect all GPU/iGPU devices in registry order.
        std::vector<ggml_backend_dev_t> gpus;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(d);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)
                gpus.push_back(d);
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
                // Default: prefer discrete GPUs over integrated GPUs/APUs. Within that class,
                // pick the device with the most total memory.
                size_t best = 0;
                for (ggml_backend_dev_t d : gpus) {
                    if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
                    size_t free = 0, total = 0;
                    ggml_backend_dev_memory(d, &free, &total);
                    if (total > best) { best = total; chosen = d; }
                }
                if (!chosen) {
                    for (ggml_backend_dev_t d : gpus) {
                        size_t free = 0, total = 0;
                        ggml_backend_dev_memory(d, &free, &total);
                        if (total > best) { best = total; chosen = d; }
                    }
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
    ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (b) {
        configure_cpu_threads(b, cpu_threads > 0 ? cpu_threads : cpu_threads_from_env());
        ggml_backend_dev_t d = ggml_backend_get_device(b);
        if (d) {
            fprintf(stderr, "[sa3] backend: %s (%s)\n",
                    ggml_backend_name(b), ggml_backend_dev_description(d));
        } else {
            fprintf(stderr, "[sa3] backend: %s\n", ggml_backend_name(b));
        }
    }
    return b;
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
        if (it == tensors.end()) throw gguf_error("missing tensor: " + n);
        return it->second;
    }

    uint32_t u32(const char* k) const {
        int i = gguf_find_key(gguf, k);
        if (i < 0) throw gguf_error("missing key: " + std::string(k));
        return gguf_get_val_u32(gguf, i);
    }
    float f32(const char* k) const {
        int i = gguf_find_key(gguf, k);
        if (i < 0) throw gguf_error("missing key: " + std::string(k));
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
    if (!m.gguf) throw gguf_error("failed to open " + std::string(path));

    std::unique_ptr<FILE, int(*)(FILE*)> f(fopen(path, "rb"), fclose);
    if (!f) throw gguf_error("cannot read " + std::string(path));
    const uint64_t data_off = (uint64_t)gguf_get_data_offset(m.gguf);
    uint64_t file_size = 0;
    const bool have_file_size = gguf_file_size(f.get(), file_size);

    struct TensorRead {
        ggml_tensor* tensor;
        std::string name;
        uint64_t off;
        size_t nb;
    };
    std::vector<TensorRead> reads;
    for (ggml_tensor* cur = ggml_get_first_tensor(m.ctx); cur; cur = ggml_get_next_tensor(m.ctx, cur)) {
        const char* name = ggml_get_name(cur);
        const int idx = gguf_find_tensor(m.gguf, name);
        if (idx < 0) throw gguf_error("metadata missing tensor offset for " + std::string(name) + " in " + std::string(path));
        const uint64_t off = data_off + (uint64_t)gguf_get_tensor_offset(m.gguf, idx);
        const size_t nb  = ggml_nbytes(cur);
        if (have_file_size && (off > file_size || nb > file_size - off)) {
            const uint64_t avail64 = off < file_size ? file_size - off : 0;
            const size_t available = avail64 > (uint64_t)nb ? nb : (size_t)avail64;
            throw gguf_error(gguf_short_read_message(path, name, off, nb, available, file_size, true));
        }
        reads.push_back({cur, name, off, nb});
    }

    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.backend);

    std::vector<char> tmp;
    for (const TensorRead& r : reads) {
        tmp.resize(r.nb);
        if (!gguf_file_seek(f.get(), r.off)) {
            throw gguf_error("seek failed while loading tensor " + r.name + " from " +
                             std::string(path) + ": " + std::strerror(errno));
        }
        const size_t got = fread(tmp.data(), 1, r.nb, f.get());
        if (got != r.nb) throw gguf_error(gguf_short_read_message(path, r.name.c_str(), r.off, r.nb, got, file_size, have_file_size));
        ggml_backend_tensor_set(r.tensor, tmp.data(), 0, r.nb);
        m.tensors[r.name] = r.tensor;
    }
    return m;
}

// ggml_backend_graph_compute returns a status that we used to discard everywhere, which is fine
// right up until a backend starts failing. Metal in particular LATCHES the failure: once one
// command buffer comes back != Completed, ggml_metal_graph_compute sets has_error and every later
// call returns GGML_STATUS_FAILED immediately without touching a single output tensor
// (ggml-metal-context.m: "recreate the backend to recover"). Since one backend is shared by the
// autoencoder, T5 and the DiT, a single failed buffer turns the rest of a training run into
// instant no-ops reading back whatever was already in the buffers -- pre-encode that races through
// producing all-zero latents, then steps at a fraction of a second reporting loss 0 and gnorm 0.
// Nothing about that looks like an error, so check the status and say so.
inline bool graph_compute_checked(ggml_backend_t backend, ggml_cgraph* graph,
                                  const char* what, std::string& err) {
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st == GGML_STATUS_SUCCESS) return true;
    const char* name = ggml_backend_name(backend);
    err = std::string(what) + ": backend '" + (name ? name : "(unknown)") + "' returned " +
          ggml_status_to_string(st) +
          ". The backend is in an error state and cannot be recovered within this run; see the "
          "backend's own log for the failing command buffer. On Metal this is usually memory "
          "pressure -- retry with a smaller --frames, a more quantized --encoding, or --device cpu";
    return false;
}

} // namespace sa3
