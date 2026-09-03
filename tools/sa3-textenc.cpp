// sa3-textenc: run the T5Gemma encoder on token ids and dump last_hidden_state
// (raw f32, ggml [dim, seq]) for tools/cossim.py. Reads ids/attn produced by
// tools/dump_t5gemma.py.
#include "gguf_model.h"
#include "t5gemma.h"
#include "tokenizer.h"

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
    const char* tok_path = nullptr; const char* prompt = "short electronic tone";
    int seq = 256, layers = -1, repeat = 1, threads = 4;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf") && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--ids")  && i+1 < argc) ids_path = argv[++i];
        else if (!strcmp(argv[i], "--attn") && i+1 < argc) attn_path = argv[++i];
        else if (!strcmp(argv[i], "--seq")  && i+1 < argc) seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")  && i+1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--tok") && i+1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i+1 < argc) layers = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i+1 < argc) repeat = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i+1 < argc) threads = std::stoi(argv[++i]);
        else throw std::runtime_error(std::string("unknown or incomplete argument: ") + argv[i]);
    }
    if (!gguf_path || (!tok_path && (!ids_path || !attn_path))) {
        fprintf(stderr, "usage: sa3-textenc --gguf <f> (--ids <ids.i32> --attn <attn.i32> | --tok <gguf> [--prompt TEXT])\n"
                        "       [--seq N] [--layers N] [--repeat N] [--threads N] [--out <existing-dir>]\n"
                        "--layers truncates the encoder for fault reduction; output is NOT full-model conditioning.\n");
        return 1;
    }
    if (seq <= 0 || repeat <= 0 || threads <= 0 || layers < -1 || (tok_path && (ids_path || attn_path)))
        throw std::runtime_error("invalid dimensions, repeat count, threads, or mixed token inputs");

    const int64_t load_start = ggml_time_us();
    ggml_backend_t backend = sa3::make_backend(threads);
    sa3::GgufModel W = sa3::load_gguf(gguf_path, backend);
    W.owns_backend = true;
    sa3::T5GemmaConfig c = sa3::T5GemmaConfig::from(W);
    if (seq > (int)W.u32("t5g.max_length") || layers > c.layers)
        throw std::runtime_error("sequence or layers exceed model configuration");
    if (layers >= 0) c.layers = layers;
    fprintf(stderr, "textenc: backend=%s layers=%d seq=%d weight_buffer=%.2f MiB load=%.3f s\n",
            ggml_backend_name(W.backend), c.layers, seq, ggml_backend_buffer_get_size(W.buf)/1048576.0,
            (ggml_time_us()-load_start)/1.e6);

    ggml_init_params ip = { (size_t)128*1024*1024, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_input(ids); ggml_set_input(pos); ggml_set_input(mask);

    ggml_tensor* hidden = ggml_cont(ctx, sa3::t5gemma_encode(ctx, W, ids, pos, mask, c));
    ggml_set_output(hidden);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, hidden);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) throw std::runtime_error("graph allocation failed");
    fprintf(stderr, "textenc: graph_buffer=%.2f MiB nodes=%d\n",
            ggml_gallocr_get_buffer_size(alloc, 0)/1048576.0, ggml_graph_n_nodes(gf));

    std::vector<int32_t> idbuf, attn;
    if (tok_path) {
        const sa3::Tokenizer tok = sa3::Tokenizer::load(tok_path);
        const std::vector<int32_t> encoded = tok.encode(prompt);
        idbuf.assign(seq, tok.pad_id); attn.assign(seq, 0);
        for (size_t i = 0; i < encoded.size() && i < (size_t)seq; ++i) {
            idbuf[i] = encoded[i]; attn[i] = 1;
        }
    } else {
        idbuf = read_i32(ids_path, seq); attn = read_i32(attn_path, seq);
    }
    ggml_backend_tensor_set(ids, idbuf.data(), 0, seq*sizeof(int32_t));

    std::vector<int32_t> posbuf(seq); for (int i = 0; i < seq; i++) posbuf[i] = i;
    ggml_backend_tensor_set(pos, posbuf.data(), 0, seq*sizeof(int32_t));

    // bidirectional mask: key k is masked (-inf) for all queries if it is padding
    std::vector<float> mbuf((size_t)seq*seq);
    for (int q = 0; q < seq; q++) for (int k = 0; k < seq; k++)
        mbuf[(size_t)q*seq + k] = attn[k] ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, mbuf.data(), 0, mbuf.size()*sizeof(float));

    std::vector<float> h(ggml_nelements(hidden));
    for (int iteration = 0; iteration < repeat; ++iteration) {
        const int64_t start = ggml_time_us();
        std::string err;
        if (!sa3::graph_compute_checked(W.backend, gf, "textenc", err)) throw std::runtime_error(err);
        const int64_t computed = ggml_time_us();
        ggml_backend_tensor_get(hidden, h.data(), 0, h.size()*sizeof(float));
        size_t nonfinite = 0;
        double sum2 = 0;
        for (float value : h) { nonfinite += !std::isfinite(value); sum2 += (double)value*value; }
        fprintf(stderr, "textenc: repeat=%d compute=%.6f s readback=%.6f s nonfinite=%zu rms=%.8f\n",
                iteration+1, (computed-start)/1.e6, (ggml_time_us()-computed)/1.e6, nonfinite, sqrt(sum2/h.size()));
        if (nonfinite) throw std::runtime_error("non-finite encoder output");
    }
    std::string fn = std::string(outdir) + "/t5g_hidden.f32";
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open output: " + fn);
    const size_t written = fwrite(h.data(), sizeof(float), h.size(), f);
    const int closed = fclose(f);
    if (written != h.size() || closed) throw std::runtime_error("cannot write output: " + fn);
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
