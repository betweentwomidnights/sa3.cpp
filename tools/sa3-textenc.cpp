// sa3-textenc: run the T5Gemma encoder on token ids and dump last_hidden_state
// (raw f32, ggml [dim, seq]) for tools/cossim.py. Reads ids/attn produced by
// tools/dump_t5gemma.py.
#include "gguf_model.h"
#include "t5gemma.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static std::vector<int32_t> read_i32(const char* path, size_t n) {
    std::vector<int32_t> b(n);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    if (fread(b.data(), sizeof(int32_t), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return b;
}

static int run(int argc, char** argv) {
    const char* gguf_path = nullptr; const char* ids_path = nullptr;
    const char* attn_path = nullptr; const char* outdir = ".";
    int seq = 256;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf") && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--ids")  && i+1 < argc) ids_path = argv[++i];
        else if (!strcmp(argv[i], "--attn") && i+1 < argc) attn_path = argv[++i];
        else if (!strcmp(argv[i], "--seq")  && i+1 < argc) seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")  && i+1 < argc) outdir = argv[++i];
    }
    if (!gguf_path || !ids_path || !attn_path) {
        fprintf(stderr, "usage: sa3-textenc --gguf <f> --ids <ids.i32> --attn <attn.i32> --seq N --out <dir>\n");
        return 1;
    }

    sa3::GgufModel W = sa3::load_gguf(gguf_path);
    const sa3::T5GemmaConfig c = sa3::T5GemmaConfig::from(W);

    ggml_init_params ip = { (size_t)128*1024*1024, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_input(ids); ggml_set_input(pos); ggml_set_input(mask);

    ggml_tensor* hidden = ggml_cont(ctx, sa3::t5gemma_encode(ctx, W, ids, pos, mask, c));
    ggml_set_output(hidden);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, hidden);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    std::vector<int32_t> idbuf  = read_i32(ids_path, seq);
    std::vector<int32_t> attn   = read_i32(attn_path, seq);
    ggml_backend_tensor_set(ids, idbuf.data(), 0, seq*sizeof(int32_t));

    std::vector<int32_t> posbuf(seq); for (int i = 0; i < seq; i++) posbuf[i] = i;
    ggml_backend_tensor_set(pos, posbuf.data(), 0, seq*sizeof(int32_t));

    // bidirectional mask: key k is masked (-inf) for all queries if it is padding
    std::vector<float> mbuf((size_t)seq*seq);
    for (int q = 0; q < seq; q++) for (int k = 0; k < seq; k++)
        mbuf[(size_t)q*seq + k] = attn[k] ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, mbuf.data(), 0, mbuf.size()*sizeof(float));

    ggml_backend_graph_compute(W.backend, gf);

    std::vector<float> h(ggml_nelements(hidden));
    ggml_backend_tensor_get(hidden, h.data(), 0, h.size()*sizeof(float));
    std::string fn = std::string(outdir) + "/t5g_hidden.f32";
    FILE* f = fopen(fn.c_str(), "wb"); fwrite(h.data(), sizeof(float), h.size(), f); fclose(f);
    printf("done. hidden ne=[%lld,%lld] seq=%d\n", (long long)hidden->ne[0], (long long)hidden->ne[1], seq);

    ggml_gallocr_free(alloc); ggml_free(ctx); W.free();
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
