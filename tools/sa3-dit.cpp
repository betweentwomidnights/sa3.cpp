// sa3-dit: run one DiT forward (velocity prediction) and dump it as raw f32
// (ggml [io, T]) for tools/cossim.py. Inputs come from tools/dump_dit.py.
#include "gguf_model.h"
#include "dit.h"
#include "lora.h"

#include <cstdio>
#include <cstring>
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

int main(int argc, char** argv) {
    const char* gguf_path = nullptr; const char* dir = "refdata"; const char* outdir = "cppout";
    const char* lora_path = nullptr; float lora_strength = 1.0f;
    int frames = 32, ctx_len = 257;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf")   && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--in")     && i+1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx")    && i+1 < argc) ctx_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--lora")   && i+1 < argc) lora_path = argv[++i];
        else if (!strcmp(argv[i], "--lora-strength") && i+1 < argc) lora_strength = (float)atof(argv[++i]);
    }
    if (!gguf_path) { fprintf(stderr, "usage: sa3-dit --gguf <f> --in <dir> --frames N --ctx 257 --out <dir> [--lora <gguf> --lora-strength S]\n"); return 1; }

    sa3::GgufModel W = sa3::load_gguf(gguf_path);
    const sa3::DitConfig c = sa3::DitConfig::from(W);

    std::vector<sa3::LoraAdapter> adapters;
    sa3::LoraStack lstack;
    if (lora_path) {
        adapters.push_back(sa3::load_lora(lora_path, lora_strength));
        lstack = sa3::apply_loras(W, adapters);
        printf("applied lora %s (type=%s rank=%d alpha=%.1f strength=%.2f, %zu overrides)\n",
               lora_path, adapters[0].type.c_str(), adapters[0].rank, adapters[0].alpha, lora_strength, W.overrides.size());
    }
    const int T = frames, S = c.mem_tokens + T;

    ggml_init_params ip = { (size_t)512*1024*1024, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.io, T);
    ggml_tensor* tfeat = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.time_dim);
    ggml_tensor* cross = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.cond_dim, ctx_len);
    ggml_tensor* glob  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.cond_dim);
    ggml_tensor* pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_tensor* ones  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    for (ggml_tensor* t : {x, tfeat, cross, glob, pos, ones}) ggml_set_input(t);

    // inpaint: if the DiT carries local-cond weights and a dit_local.f32 ref exists, feed it
    std::string local_path = std::string(dir) + "/dit_local.f32";
    ggml_tensor* local = nullptr;
    if (c.local_dim > 0) { FILE* lf = fopen(local_path.c_str(), "rb"); if (lf) { fclose(lf);
        local = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.local_dim, T); ggml_set_input(local); } }

    ggml_tensor* vel = ggml_cont(ctx, sa3::dit_forward(ctx, W, x, tfeat, cross, glob, pos, ones, c, local));
    ggml_set_output(vel);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, /*grads=*/false); // DiT has many nodes
    ggml_build_forward_expand(gf, vel);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    auto setf = [&](ggml_tensor* t, const std::string& name){
        std::vector<float> b = read_f32((std::string(dir) + "/" + name).c_str(), ggml_nelements(t));
        ggml_backend_tensor_set(t, b.data(), 0, b.size()*sizeof(float));
    };
    setf(x, "dit_x.f32"); setf(tfeat, "dit_tfeat.f32"); setf(cross, "dit_cross.f32"); setf(glob, "dit_global.f32");
    if (local) setf(local, "dit_local.f32");
    std::vector<int32_t> posbuf(S); for (int i = 0; i < S; i++) posbuf[i] = i;
    ggml_backend_tensor_set(pos, posbuf.data(), 0, posbuf.size()*sizeof(int32_t));
    float one = 1.0f; ggml_backend_tensor_set(ones, &one, 0, sizeof(float));

    ggml_backend_graph_compute(W.backend, gf);

    std::vector<float> out(ggml_nelements(vel));
    ggml_backend_tensor_get(vel, out.data(), 0, out.size()*sizeof(float));
    std::string fn = std::string(outdir) + "/dit_vel.f32";
    FILE* f = fopen(fn.c_str(), "wb"); fwrite(out.data(), sizeof(float), out.size(), f); fclose(f);
    printf("done. velocity ne=[%lld,%lld] T=%d S=%d\n", (long long)vel->ne[0], (long long)vel->ne[1], T, S);

    ggml_gallocr_free(alloc); ggml_free(ctx); W.free();
    return 0;
}
