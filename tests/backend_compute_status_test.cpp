#include "gguf_model.h"
#include "ggml-alloc.h"

#include <cstdio>
#include <string>

static bool abort_compute(void*) { return true; }

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!device) { std::fprintf(stderr, "CPU backend unavailable\n"); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) { std::fprintf(stderr, "CPU backend initialization failed\n"); return 1; }
    auto set_abort = reinterpret_cast<ggml_backend_set_abort_callback_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),
                                          "ggml_backend_set_abort_callback"));
    if (!set_abort) { std::fprintf(stderr, "CPU abort callback unavailable\n"); return 1; }

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(input);
    ggml_tensor* output = ggml_add(ctx, input, input);
    ggml_set_output(output);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, graph);
    const float values[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    ggml_backend_tensor_set(input, values, 0, sizeof(values));

    std::string error;
    const bool success = sa3::graph_compute_checked(backend, graph, "T5 encode", error);
    set_abort(backend, abort_compute, nullptr);
    const bool incorrectly_succeeded = sa3::graph_compute_checked(backend, graph,
                                                                    "SAME decode", error);
    set_abort(backend, nullptr, nullptr);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!success || incorrectly_succeeded ||
        error.find("SAME decode") == std::string::npos ||
        error.find("aborted") == std::string::npos ||
        error.find("CPU") == std::string::npos) {
        std::fprintf(stderr, "checked compute did not report an aborted CPU graph: %s\n",
                     error.c_str());
        return 1;
    }
    std::puts("backend compute status test passed");
    return 0;
}
