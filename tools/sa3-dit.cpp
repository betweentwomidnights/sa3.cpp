// sa3-dit: run one DiT forward (velocity prediction) and dump it as raw f32
// (ggml [io, T]) for tools/cossim.py. Inputs come from tools/dump_dit.py, or are
// generated in-process with --random SEED (for self-contained A/B comparisons).
//
// Adapters can be applied two ways for cross-validation of the two code paths:
//   --lora <gguf>            merged, via lora.h apply_loras (the inference path)
//   --lora-functional <gguf> functional, via dit_lin's DitLora hook (the training path)
// Running both on identical inputs and diffing dit_vel.f32 checks that what training
// optimizes is exactly what inference applies.
#include "gguf_model.h"
#include "dit.h"
#include "lora.h"
#include "train_checkpoint.h"
#include "train_lora.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
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
    const char* gguf_path = nullptr; const char* dir = "refdata"; const char* outdir = "cppout";
    std::vector<std::pair<std::string,float>> lora_specs;   // (gguf, strength) in flag order
    const char* functional_path = nullptr;
    long long random_seed = -1;
    bool local_zeros = false;
    int frames = 32, ctx_len = 257;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gguf")   && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--in")     && i+1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx")    && i+1 < argc) ctx_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--lora")   && i+1 < argc) lora_specs.push_back({argv[++i], 1.0f});
        else if (!strcmp(argv[i], "--lora-strength") && i+1 < argc) {
            if (lora_specs.empty()) { fprintf(stderr, "--lora-strength must follow a --lora\n"); return 1; }
            lora_specs.back().second = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--lora-functional") && i+1 < argc) functional_path = argv[++i];
        else if (!strcmp(argv[i], "--random") && i+1 < argc) random_seed = strtoll(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--local-zeros")) local_zeros = true;
    }
    if (!gguf_path) { fprintf(stderr, "usage: sa3-dit --gguf <f> (--in <dir> | --random SEED) --frames N --ctx 257 --out <dir> [--lora <gguf> --lora-strength S | --lora-functional <gguf>]\n"); return 1; }

    sa3::GgufModel W = sa3::load_gguf(gguf_path);
    const sa3::DitConfig c = sa3::DitConfig::from(W);

    std::vector<sa3::LoraAdapter> adapters;
    for (auto& ls : lora_specs) adapters.push_back(sa3::load_lora(ls.first.c_str(), ls.second, W.backend));
    if (!adapters.empty()) {
        sa3::apply_loras(W, adapters);
        printf("applied %zu lora(s):\n", adapters.size());
        for (size_t k = 0; k < adapters.size(); k++)
            printf("  [%zu] %s strength=%.2f\n", k, lora_specs[k].first.c_str(), adapters[k].strength);
    }

    // Functional application: the adapter tensors feed dit_lin's DitLora hook exactly like the
    // training graph builds it (train_dit.h/train_ckpt.h), instead of merging into the weights.
    sa3::DitLora dl;
    ggml_context* pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;
    sa3::TrainLoraState flora;
    std::vector<std::vector<float>> bnsq_host;
    if (functional_path) {
        std::string err;
        if (!sa3::load_train_lora_gguf(functional_path, flora, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        if (flora.adapter_type != "lora" && flora.adapter_type != "dora-rows") {
            fprintf(stderr, "--lora-functional supports lora/dora-rows only (got %s)\n", flora.adapter_type.c_str());
            return 1;
        }
        const bool dora = flora.adapter_type == "dora-rows";
        ggml_init_params pip = { ggml_tensor_overhead() * (flora.params.size() * 4 + 16), nullptr, true };
        pctx = ggml_init(pip);
        struct Up { ggml_tensor* t; const float* d; size_t n; };
        std::vector<Up> ups;
        bnsq_host.reserve(flora.params.size());
        for (sa3::TrainLoraParam& hp : flora.params) {
            sa3::DitLoraParam dp;
            ggml_tensor* A = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, hp.target.in, flora.rank);
            ggml_tensor* B = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, flora.rank, hp.target.out);
            ups.push_back({A, hp.lora_A.data(), hp.lora_A.size()});
            ups.push_back({B, hp.lora_B.data(), hp.lora_B.size()});
            dp.A = A;
            dp.B = B;
            dp.scale = flora.alpha / (float)flora.rank;
            dp.in = hp.target.in;
            if (dora) {
                ggml_tensor* M = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, (int64_t)hp.magnitude.size());
                ups.push_back({M, hp.magnitude.data(), hp.magnitude.size()});
                bnsq_host.push_back(sa3::train_row_norm_sq(W.get(hp.target.weight_name), hp.target.in,
                                                           hp.target.out, 1e-12f));
                ggml_tensor* N = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, hp.target.out);
                ups.push_back({N, bnsq_host.back().data(), bnsq_host.back().size()});
                dp.dora = true;
                dp.magnitude = M;
                dp.base_norm_sq = N;
            }
            dl[hp.target.weight_name] = dp;
        }
        pbuf = ggml_backend_alloc_ctx_tensors(pctx, W.backend);
        if (!pbuf) { fprintf(stderr, "failed to allocate functional adapter tensors\n"); return 1; }
        for (const Up& u : ups) ggml_backend_tensor_set(u.t, u.d, 0, u.n * sizeof(float));
        printf("functional lora: %s (%zu targets, %s)\n", functional_path, flora.params.size(), flora.adapter_type.c_str());
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

    // inpaint: if the DiT carries local-cond weights and a dit_local.f32 ref exists, feed it.
    // --local-zeros instead feeds an all-zero local cond (the reference plain-generation input).
    std::string local_path = std::string(dir) + "/dit_local.f32";
    ggml_tensor* local = nullptr;
    if (c.local_dim > 0 && local_zeros) {
        local = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.local_dim, T); ggml_set_input(local);
    } else if (c.local_dim > 0) { FILE* lf = fopen(local_path.c_str(), "rb"); if (lf) { fclose(lf);
        local = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.local_dim, T); ggml_set_input(local); } }

    ggml_tensor* vel = ggml_cont(ctx, sa3::dit_forward(ctx, W, x, tfeat, cross, glob, pos, ones, c, local,
                                                       functional_path ? &dl : nullptr));
    ggml_set_output(vel);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, /*grads=*/false); // DiT has many nodes
    ggml_build_forward_expand(gf, vel);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    if (random_seed >= 0) {
        // Self-contained inputs: seeded gaussians (identical across invocations for A/B diffing).
        std::mt19937_64 rng((unsigned long long)random_seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto setr = [&](ggml_tensor* t){
            std::vector<float> b((size_t)ggml_nelements(t));
            for (float& v : b) v = nd(rng);
            ggml_backend_tensor_set(t, b.data(), 0, b.size()*sizeof(float));
        };
        setr(x); setr(tfeat); setr(cross); setr(glob);
        if (local) {
            std::vector<float> lb((size_t)ggml_nelements(local), 0.0f);
            if (!local_zeros) for (float& v : lb) v = nd(rng);
            ggml_backend_tensor_set(local, lb.data(), 0, lb.size()*sizeof(float));
        }
    } else {
        auto setf = [&](ggml_tensor* t, const std::string& name){
            std::vector<float> b = read_f32((std::string(dir) + "/" + name).c_str(), ggml_nelements(t));
            ggml_backend_tensor_set(t, b.data(), 0, b.size()*sizeof(float));
        };
        setf(x, "dit_x.f32"); setf(tfeat, "dit_tfeat.f32"); setf(cross, "dit_cross.f32"); setf(glob, "dit_global.f32");
        if (local) setf(local, "dit_local.f32");
    }
    std::vector<int32_t> posbuf(S); for (int i = 0; i < S; i++) posbuf[i] = i;
    ggml_backend_tensor_set(pos, posbuf.data(), 0, posbuf.size()*sizeof(int32_t));
    float one = 1.0f; ggml_backend_tensor_set(ones, &one, 0, sizeof(float));

    ggml_backend_graph_compute(W.backend, gf);

    std::vector<float> out(ggml_nelements(vel));
    ggml_backend_tensor_get(vel, out.data(), 0, out.size()*sizeof(float));
    std::string fn = std::string(outdir) + "/dit_vel.f32";
    FILE* f = fopen(fn.c_str(), "wb"); fwrite(out.data(), sizeof(float), out.size(), f); fclose(f);
    printf("done. velocity ne=[%lld,%lld] T=%d S=%d\n", (long long)vel->ne[0], (long long)vel->ne[1], T, S);

    ggml_gallocr_free(alloc); ggml_free(ctx);
    if (pbuf) ggml_backend_buffer_free(pbuf);
    if (pctx) ggml_free(pctx);
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
