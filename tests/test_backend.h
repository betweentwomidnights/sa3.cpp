#pragma once

#include "gguf_model.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

inline ggml_backend_t sa3_test_cpu_backend() {
    static ggml_backend_t backend = [] {
        sa3::load_dynamic_backends_once();
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }();
    if (!backend) {
        std::fprintf(stderr, "FAIL: no dynamically registered CPU backend\n");
        std::abort();
    }
    return backend;
}

inline void sa3_test_compute(ggml_cgraph* graph) {
    if (ggml_backend_graph_compute(sa3_test_cpu_backend(), graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: CPU backend graph compute failed\n");
        std::abort();
    }
}

struct Sa3TestBackend {
    const char* name;
    ggml_backend_t backend;
};

// BLAS registers an ACCEL device on macOS but implements little beyond matmul. make_backend()
// only ever takes a GPU/IGPU or falls back to CPU, so a model never lands on one.
inline bool sa3_test_is_model_hostable(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return true;
        default:                            return false;
    }
}

// Every backend this build can host a model on: build/ gets CPU, build-metal CPU + Metal.
inline const std::vector<Sa3TestBackend>& sa3_test_backends() {
    static std::vector<Sa3TestBackend> backends = [] {
        sa3::load_dynamic_backends_once();
        std::vector<Sa3TestBackend> out;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!sa3_test_is_model_hostable(dev)) continue;
            if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr))
                out.push_back({ ggml_backend_dev_name(dev), b });
        }
        return out;
    }();
    if (backends.empty()) {
        std::fprintf(stderr, "FAIL: no registered backends\n");
        std::abort();
    }
    return backends;
}
