// sa3-smoke: minimal sanity check that the GGML toolchain links and runs.
// Enumerates available backend devices (so we can see CPU/CUDA/Vulkan on this
// machine) and runs a trivial tensor add to confirm compute works.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <vector>

int main() {
    printf("sa3.cpp smoke test\n");

    const size_t n_dev = ggml_backend_dev_count();
    printf("ggml backend devices: %zu\n", n_dev);
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        const char* kind = "other";
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   kind = "CPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   kind = "GPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  kind = "IGPU";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: kind = "ACCEL"; break;
            case GGML_BACKEND_DEVICE_TYPE_META:  kind = "META";  break;
        }
        printf("  [%zu] %-5s %-20s %s (%.1f/%.1f GiB free)\n",
               i, kind, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               free_mem / 1073741824.0, total_mem / 1073741824.0);
    }

    // Trivial compute check on the CPU backend: c = a + b.
    ggml_backend_t backend = ggml_backend_cpu_init();
    struct ggml_init_params p = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx = ggml_init(p);
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor* c = ggml_add(ctx, a, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const float av[4] = {1, 2, 3, 4};
    const float bv[4] = {10, 20, 30, 40};
    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));

    ggml_backend_graph_compute(backend, gf);

    float cv[4] = {0};
    ggml_backend_tensor_get(c, cv, 0, sizeof(cv));
    printf("compute check  a+b = [%.0f %.0f %.0f %.0f] (expect 11 22 33 44)\n",
           cv[0], cv[1], cv[2], cv[3]);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
