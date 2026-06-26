// sa3-codec: decode a SAME-L latent to audio (Phase 1, CPU/f32) and dump the
// validation checkpoints (after_in_proj, after_resampling, audio) as raw f32 in
// GGML memory order for tools/cossim.py. Thin driver over src/same_ae.h.
#include "gguf_model.h"
#include "same_ae.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* gguf_path = nullptr;
    const char* z_path = nullptr;
    int frames = 8;
    const char* outdir = ".";
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf")   && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--z")      && i+1 < argc) z_path = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) outdir = argv[++i];
    }
    if (!gguf_path || !z_path) {
        fprintf(stderr, "usage: sa3-codec --gguf <f> --z <z.f32> --frames N --out <dir>\n");
        return 1;
    }

    sa3::GgufModel W = sa3::load_gguf(gguf_path);
    const sa3::SameConfig c = sa3::SameConfig::from(W);
    const int T = frames;
    const int64_t N = (int64_t)T * c.sub_chunk;

    ggml_init_params ip = { (size_t)256*1024*1024, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* z    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.latent, T);
    ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    ggml_set_input(z); ggml_set_input(pos); ggml_set_input(mask);

    sa3::DecodeOut out = sa3::same_decode(ctx, W, z, c, T, pos, mask);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out.audio);
    ggml_build_forward_expand(gf, out.after_in_proj);
    ggml_build_forward_expand(gf, out.after_resampling);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    std::vector<float> zbuf((size_t)c.latent*T);
    { FILE* f = fopen(z_path, "rb"); fread(zbuf.data(), sizeof(float), zbuf.size(), f); fclose(f); }
    ggml_backend_tensor_set(z, zbuf.data(), 0, zbuf.size()*sizeof(float));

    std::vector<int32_t> posbuf(N); for (int i = 0; i < N; i++) posbuf[i] = i;
    ggml_backend_tensor_set(pos, posbuf.data(), 0, posbuf.size()*sizeof(int32_t));

    std::vector<float> mbuf((size_t)N*N);
    for (int q = 0; q < N; q++) for (int kk = 0; kk < N; kk++)
        mbuf[(size_t)q*N+kk] = (std::abs(q-kk) <= c.sliding_window) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, mbuf.data(), 0, mbuf.size()*sizeof(float));

    ggml_backend_graph_compute(W.backend, gf);

    auto dump = [&](ggml_tensor* t, const char* name) {
        std::vector<float> b(ggml_nelements(t));
        ggml_backend_tensor_get(t, b.data(), 0, b.size()*sizeof(float));
        std::string fn = std::string(outdir) + "/" + name + ".f32";
        FILE* f = fopen(fn.c_str(), "wb"); fwrite(b.data(), sizeof(float), b.size(), f); fclose(f);
        printf("  dumped %-18s ne=[%lld,%lld,%lld]\n", name,
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
    };
    dump(out.after_in_proj,    "after_in_proj");
    dump(out.after_resampling, "after_resampling");
    dump(out.audio,            "audio");
    printf("done. T=%d N=%lld running_std=%.5f\n", T, (long long)N, c.running_std);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    W.free();
    return 0;
}
