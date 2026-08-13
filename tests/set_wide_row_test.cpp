// Regression test for non-inplace ggml_set on wide rows, on whichever GPU backend is present.
//
// Metal's ggml_metal_op_set dispatched only ne01 threadgroups for its src0->dst copy, while
// kernel_cpy_t_t derives its row-chunk index as tgpig[0]/ne01 and copies one element per thread.
// That pinned the chunk index at 0 and truncated every row at nth = min(1024, ne00) elements, so
// the tail kept whatever the destination buffer already held. Invisible below 1024 wide, which is
// why test-backend-ops (SET only at ne00=6) passed 12/12 while corrupting a 1536-wide tensor.
// Fixed in the ggml fork; this guards the next bump, on any backend.
//
// Replicates dit.h:305-308 exactly and nothing else:
//
//   full = ggml_new_tensor_2d(F32, dim, mem + T)         // UNINITIALISED leaf
//   full = ggml_set(full, memory_tokens, .., offset 0)   // writes rows 0..mem
//   x    = ggml_set(full, x,            .., offset mem)  // writes rows mem..mem+T
//   out  = ggml_cont(x)
//
// Both SETs are non-inplace, so each must copy src0 into dst before writing its own region.
// The scratch buffer is poisoned with 0xDEADBEEF first, so any byte the op fails to write is
// visible in the output instead of being silently plausible.
//
// Expected: out rows [0,mem) == 1.0 (memory tokens), rows [mem, mem+T) == 2.0 (x).
// Poison surviving anywhere means a destination was not fully written.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static const uint32_t POISON = 0xDEADBEEFu;

static int run(ggml_backend_t backend, const char* label, int64_t dim, int64_t mem, int64_t T,
               bool poison) {
    const int64_t rows = mem + T;

    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, rows);   // uninitialised
    ggml_tensor* memt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, mem);
    ggml_tensor* x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, T);
    ggml_set_input(memt);
    ggml_set_input(x);

    ggml_tensor* a   = ggml_set(ctx, full, memt, full->nb[1], full->nb[2], full->nb[3], 0);
    ggml_tensor* b   = ggml_set(ctx, a,    x,    full->nb[1], full->nb[2], full->nb[3],
                                (size_t)mem * full->nb[1]);
    ggml_tensor* out = ggml_cont(ctx, b);
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { std::printf("%s: alloc failed\n", label); return 2; }

    // Poison every allocated tensor in the graph, then write the real inputs on top.
    if (poison) {
        std::vector<uint32_t> pat;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor* t = ggml_graph_node(gf, i);
            if (!t->data || !t->buffer) continue;
            pat.assign((size_t)ggml_nbytes(t) / 4, POISON);
            ggml_backend_tensor_set(t, pat.data(), 0, pat.size() * 4);
        }
        if (full->data && full->buffer) {
            pat.assign((size_t)ggml_nbytes(full) / 4, POISON);
            ggml_backend_tensor_set(full, pat.data(), 0, pat.size() * 4);
        }
    }

    std::vector<float> hm((size_t)dim * mem, 1.0f), hx((size_t)dim * T, 2.0f);
    ggml_backend_tensor_set(memt, hm.data(), 0, hm.size() * sizeof(float));
    ggml_backend_tensor_set(x,    hx.data(), 0, hx.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::printf("%s: compute failed\n", label); return 2;
    }

    std::vector<float> o((size_t)dim * rows);
    ggml_backend_tensor_get(out, o.data(), 0, o.size() * sizeof(float));

    long long bad_mem = 0, bad_x = 0, pois = 0; long long firstbad = -1;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < dim; ++c) {
            const float v = o[(size_t)r * dim + c];
            uint32_t bits; std::memcpy(&bits, &v, 4);
            if (bits == POISON) { ++pois; if (firstbad < 0) firstbad = r; continue; }
            const float want = (r < mem) ? 1.0f : 2.0f;
            if (v != want) { if (r < mem) ++bad_mem; else ++bad_x; if (firstbad < 0) firstbad = r; }
        }
    }
    const bool ok = (pois == 0 && bad_mem == 0 && bad_x == 0);
    std::printf("%-22s ne=[%lld,%lld] mem=%lld poison=%-3s -> %s  poison_words=%lld "
                "wrong_memrows=%lld wrong_xrows=%lld%s\n",
                label, (long long)dim, (long long)rows, (long long)mem, poison ? "yes" : "no",
                ok ? "PASS" : "FAIL", pois, bad_mem, bad_x,
                firstbad >= 0 ? (" first_bad_row=" + std::to_string(firstbad)).c_str() : "");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = nullptr;
    std::string gpu_label = "GPU";
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_init(d, nullptr);
            gpu_label = std::string("GPU(") + ggml_backend_name(gpu) + ")";
            std::printf("GPU backend: %s\n", ggml_backend_dev_description(d));
            break;
        }
    }
    // sa3 medium: dim 1536, 64 memory tokens, 512 frames. Plus the shape test_set uses.
    struct { int64_t dim, mem, T; } cases[] = {
        {   6, 2,   3}, {  32, 8,  16}, {  64, 8,  16}, { 128, 8,  16},
        { 256, 8,  16}, { 512, 8,  16}, {1024, 8,  16}, {1536, 8,  16}, {2048, 8, 16},
        {1536,64, 512},
    };
    int fails = 0;
    for (auto& c : cases) {
        fails += run(cpu, "CPU", c.dim, c.mem, c.T, true);
        if (gpu) fails += run(gpu, gpu_label.c_str(), c.dim, c.mem, c.T, true);
    }
    std::printf(fails ? "RESULT: %d FAILING CASES\n" : "RESULT: all pass\n", fails);
    return fails ? 1 : 0;
}
