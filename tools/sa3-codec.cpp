// sa3-codec: decode a SAME-L latent to audio (Phase 1, CPU/f32) and dump the
// validation checkpoints (after_in_proj, after_resampling, audio) as raw f32 in
// GGML memory order for tools/cossim.py. Thin driver over src/same_ae.h.
#include "gguf_model.h"
#include "same_ae.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static std::vector<float> read_f32(const char* path, size_t n) {
    std::vector<float> b(n);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    if (fread(b.data(), sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return b;
}

static int run(int argc, char** argv) {
    const char* gguf_path = nullptr;
    const char* in_path = nullptr;
    const char* mode = "decode";
    int frames = 8;
    const char* outdir = ".";
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf")   && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--mode")   && i+1 < argc) mode = argv[++i];
        else if ((!strcmp(argv[i], "--z") || !strcmp(argv[i], "--in")) && i+1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) outdir = argv[++i];
    }
    if (!gguf_path || !in_path) {
        fprintf(stderr, "usage: sa3-codec --gguf <f> --mode decode|encode --in <raw.f32> --frames N --out <dir>\n");
        return 1;
    }
    const bool encode = !strcmp(mode, "encode");

    sa3::GgufModel W = sa3::load_gguf(gguf_path);
    const sa3::SameConfig c = sa3::SameConfig::from(W);
    const int T = frames;
    const int64_t N = (int64_t)T * c.sub_chunk;

    ggml_init_params ip = { (size_t)256*1024*1024, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);

    const bool chunked = c.chunk;                           // SAME-S enc+dec both need a shifted 2nd mask
    const int64_t N2 = chunked ? N + 2*c.shift : 0;

    ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    // SAME-L: mask = sliding-window bias [3*sub_chunk, sub_chunk, N/sub_chunk]; SAME-S: block-diagonal needs none.
    ggml_tensor* mask = chunked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N)
                                : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3*c.sub_chunk, c.sub_chunk, N/c.sub_chunk);
    ggml_set_input(pos); ggml_set_input(mask);
    ggml_tensor *pos2 = nullptr, *mask2 = nullptr;
    if (chunked) {                                            // SAME-S shifted half: pos2 used (RoPE), mask2 unused
        pos2  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N2);
        mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N2, N2);
        ggml_set_input(pos2); ggml_set_input(mask2);
    }

    // input tensor + checkpoints depend on mode
    ggml_tensor* in = nullptr;
    std::vector<std::pair<ggml_tensor*, const char*>> taps;
    if (encode) {
        const int ch = c.out_channels / c.patch_size;
        const int L = T * c.patch_size * c.output_seg;       // = T * 4096
        in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, ch);  // [L, ch]
        ggml_set_input(in);
        sa3::EncodeOut e = sa3::same_encode(ctx, W, in, c, T, pos, mask, pos2, mask2);
        taps = {{e.after_resampling, "enc_after_resampling"}, {e.latent, "enc_latent"}, {e.z, "z_enc"}};
    } else {
        in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.latent, T); // [latent, T]
        ggml_set_input(in);
        sa3::DecodeOut d = sa3::same_decode(ctx, W, in, c, T, pos, mask, pos2, mask2);
        taps = {{d.after_in_proj, "after_in_proj"}, {d.after_resampling, "after_resampling"}, {d.audio, "audio"}};
    }

    // Make every checkpoint a real contiguous tensor: dumping a view whose buffer
    // gallocr later recycles yields garbage. cont + set_output protects the readout.
    for (auto& t : taps) { t.first = ggml_cont(ctx, t.first); ggml_set_output(t.first); }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    for (auto& t : taps) ggml_build_forward_expand(gf, t.first);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    std::vector<float> inbuf = read_f32(in_path, ggml_nelements(in));
    ggml_backend_tensor_set(in, inbuf.data(), 0, inbuf.size()*sizeof(float));

    auto set_pos = [&](ggml_tensor* p, int64_t n){ std::vector<int32_t> b(n); for (int i=0;i<n;i++) b[i]=i; ggml_backend_tensor_set(p, b.data(), 0, n*sizeof(int32_t)); };
    auto set_swa = [&](ggml_tensor* mt){       // SAME-L sliding-window bias; no-op when unused (SAME-S)
        if (!mt->buffer) return;
        std::vector<float> mb = sa3::build_swa_bias(c, N);
        ggml_backend_tensor_set(mt, mb.data(), 0, mb.size()*sizeof(float));
    };
    set_pos(pos, N); set_swa(mask);
    if (chunked) set_pos(pos2, N2);            // mask2 unused (block-diagonal)

    ggml_backend_graph_compute(W.backend, gf);

    for (auto& t : taps) {
        std::vector<float> b(ggml_nelements(t.first));
        ggml_backend_tensor_get(t.first, b.data(), 0, b.size()*sizeof(float));
        std::string fn = std::string(outdir) + "/" + t.second + ".f32";
        FILE* f = fopen(fn.c_str(), "wb"); fwrite(b.data(), sizeof(float), b.size(), f); fclose(f);
        printf("  dumped %-20s ne=[%lld,%lld,%lld]\n", t.second,
               (long long)t.first->ne[0], (long long)t.first->ne[1], (long long)t.first->ne[2]);
    }
    printf("done. mode=%s T=%d N=%lld running_std=%.5f\n", mode, T, (long long)N, c.running_std);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    W.free();
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
