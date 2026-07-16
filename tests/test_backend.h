#pragma once

#include "gguf_model.h"

#include <cstdio>
#include <cstdlib>

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
