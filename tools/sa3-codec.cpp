// sa3-codec: GGML implementation of the SAME-L decoder (Phase 1, CPU/f32).
// Loads the decoder GGUF, decodes a latent (raw f32, ggml [latent,T]) to audio,
// and dumps the three validation checkpoints (after_in_proj, after_resampling,
// audio) as raw f32 in GGML memory order for tools/cossim.py.
//
// Single-file first attempt; refactor into src/ headers once it validates.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// tiny GGUF weight loader
// ----------------------------------------------------------------------------
struct Weights {
    ggml_context*   ctx   = nullptr;
    gguf_context*   gguf  = nullptr;
    ggml_backend_t  backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> t;

    ggml_tensor* get(const std::string& n) {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "missing tensor: %s\n", n.c_str()); exit(1); }
        return it->second;
    }
    uint32_t u32(const char* k) {
        int i = gguf_find_key(gguf, k);
        if (i < 0) { fprintf(stderr, "missing key: %s\n", k); exit(1); }
        return gguf_get_val_u32(gguf, i);
    }
    float f32(const char* k) {
        int i = gguf_find_key(gguf, k);
        if (i < 0) { fprintf(stderr, "missing key: %s\n", k); exit(1); }
        return gguf_get_val_f32(gguf, i);
    }
};

static Weights load_gguf(const char* path) {
    Weights w;
    w.backend = ggml_backend_cpu_init();
    gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/&w.ctx };
    w.gguf = gguf_init_from_file(path, gp);
    if (!w.gguf) { fprintf(stderr, "failed to open gguf %s\n", path); exit(1); }

    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, w.backend);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    const size_t data_off = gguf_get_data_offset(w.gguf);
    std::vector<char> tmp;
    for (ggml_tensor* cur = ggml_get_first_tensor(w.ctx); cur; cur = ggml_get_next_tensor(w.ctx, cur)) {
        const char* name = ggml_get_name(cur);
        int idx = gguf_find_tensor(w.gguf, name);
        size_t off = data_off + gguf_get_tensor_offset(w.gguf, idx);
        size_t nb  = ggml_nbytes(cur);
        tmp.resize(nb);
        fseek(f, (long)off, SEEK_SET);
        if (fread(tmp.data(), 1, nb, f) != nb) { fprintf(stderr, "short read %s\n", name); exit(1); }
        ggml_backend_tensor_set(cur, tmp.data(), 0, nb);
        w.t[name] = cur;
    }
    fclose(f);
    return w;
}

// ----------------------------------------------------------------------------
// graph helpers
// ----------------------------------------------------------------------------
struct G { ggml_context* ctx; };

// DynamicTanh: y = tanh(alpha*x)*gamma + beta   (alpha [1], gamma/beta [dim] over ne0)
static ggml_tensor* dyt(G g, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* gamma, ggml_tensor* beta) {
    ggml_tensor* y = ggml_tanh(g.ctx, ggml_mul(g.ctx, x, alpha));
    y = ggml_mul(g.ctx, y, gamma);
    y = ggml_add(g.ctx, y, beta);
    return y;
}

// Linear: w ne=[in,out], x ne=[in,N] -> [out,N]; optional bias [out]
static ggml_tensor* linear(G g, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b = nullptr) {
    ggml_tensor* y = ggml_mul_mat(g.ctx, w, x);
    if (b) y = ggml_add(g.ctx, y, b);
    return y;
}

// scaled-dot-product attention with an additive mask.
// q,k,v: [d, N, H]; mask: [Nk, Nq] (0 / -inf). returns [d, Nq, H]
static ggml_tensor* sdpa(G g, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                         ggml_tensor* mask, float scale) {
    ggml_tensor* kq = ggml_mul_mat(g.ctx, k, q);                 // [Nk, Nq, H]
    kq = ggml_soft_max_ext(g.ctx, kq, mask, scale, 0.0f);        // softmax over Nk
    // v: [d, Nk, H] -> [Nk, d, H] so mul_mat(v, kq) -> [d, Nq, H]
    ggml_tensor* vt = ggml_cont(g.ctx, ggml_permute(g.ctx, v, 1, 0, 2, 3));
    ggml_tensor* o = ggml_mul_mat(g.ctx, vt, kq);                // [d, Nq, H]
    return o;
}

int main(int argc, char** argv) {
    const char* gguf_path = nullptr;
    const char* z_path = nullptr;
    int frames = 8;
    const char* outdir = ".";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gguf")  && i+1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--z") && i+1 < argc) z_path = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) outdir = argv[++i];
    }
    if (!gguf_path || !z_path) { fprintf(stderr, "usage: sa3-codec --gguf <f> --z <z.f32> --frames N --out <dir>\n"); return 1; }

    Weights w = load_gguf(gguf_path);
    const int dim     = w.u32("sa3.ae.dim");          // 1536
    const int latent  = w.u32("sa3.ae.latent_dim");   // 256
    const int dh      = w.u32("sa3.ae.dim_heads");    // 64
    const int nh      = w.u32("sa3.ae.n_heads");      // 24
    const int depth   = w.u32("sa3.ae.depth");        // 12
    const int sub     = w.u32("sa3.ae.sub_chunk");    // 17
    const int oseg    = w.u32("sa3.ae.output_seg");   // 16
    const int win     = w.u32("sa3.ae.sliding_window"); // 17
    const int sinb    = w.u32("sa3.ae.sinusoidal_blocks"); // 8
    const int outch   = w.u32("sa3.ae.out_channels"); // 512
    const int patch   = w.u32("sa3.ae.patch_size");   // 256
    const int rot     = dh / 2;                        // 32 rotary dims
    const float rope_base = w.f32("sa3.ae.rope_base"); // 10000
    const int T = frames;
    const int N = T * sub;                             // packed sequence length
    float running_std;
    ggml_backend_tensor_get(w.get("ae.running_std"), &running_std, 0, sizeof(float));

    // --- build compute graph (no_alloc, gallocr) ---
    size_t mem = (size_t)256*1024*1024;
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context* cctx = ggml_init(ip);
    G g{cctx};

    // inputs
    ggml_tensor* z = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, latent, T);  // [256,T]
    ggml_set_name(z, "z"); ggml_set_input(z);
    ggml_tensor* pos = ggml_new_tensor_1d(cctx, GGML_TYPE_I32, N);
    ggml_set_name(pos, "pos"); ggml_set_input(pos);
    ggml_tensor* mask = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, N, N);    // [Nk,Nq]
    ggml_set_name(mask, "mask"); ggml_set_input(mask);

    // bottleneck.decode: z * running_std, then in_proj 256->1536
    ggml_tensor* x = ggml_scale(cctx, z, running_std);           // [256,T]
    x = linear(g, w.get("ae.in_proj.weight"), x, w.get("ae.in_proj.bias")); // [1536,T]
    ggml_tensor* after_in_proj = x;
    ggml_set_output(after_in_proj);

    // build packed sequence: per frame [x_t, nt x16] -> [1536,17,T] -> [1536,N]
    ggml_tensor* x3 = ggml_reshape_3d(cctx, x, dim, 1, T);       // [1536,1,T]
    ggml_tensor* nt = ggml_reshape_3d(cctx, w.get("ae.dec.new_tokens"), dim, 1, 1);
    ggml_tensor* nt_rep = ggml_repeat(cctx, nt, ggml_new_tensor_3d(cctx, GGML_TYPE_F32, dim, oseg, T)); // [1536,16,T]
    ggml_tensor* packed = ggml_concat(cctx, x3, nt_rep, 1);      // [1536,17,T]
    packed = ggml_reshape_2d(cctx, ggml_cont(cctx, packed), dim, N); // [1536,N]
    x = packed;

    const float scale = 1.0f / sqrtf((float)dh);
    for (int l = 0; l < depth; l++) {
        std::string p = "ae.dec." + std::to_string(l) + ".";
        // ---- self-attention (differential) ----
        ggml_tensor* h = dyt(g, x, w.get(p+"pre_norm.alpha"), w.get(p+"pre_norm.gamma"), w.get(p+"pre_norm.beta"));
        ggml_tensor* qkv = ggml_mul_mat(cctx, w.get(p+"self_attn.to_qkv.weight"), h); // [7680,N]
        auto slice = [&](int idx)->ggml_tensor* {                 // chunk of 1536 along ne0
            return ggml_view_2d(cctx, qkv, dim, N, qkv->nb[1], (size_t)idx*dim*sizeof(float));
        };
        ggml_tensor* q  = slice(0), *k = slice(1), *v = slice(2), *qd = slice(3), *kd = slice(4);
        auto heads = [&](ggml_tensor* a)->ggml_tensor* {          // [1536,N] -> [dh,nh,N]
            return ggml_reshape_3d(cctx, ggml_cont(cctx, a), dh, nh, N);
        };
        q = heads(q); k = heads(k); v = heads(v); qd = heads(qd); kd = heads(kd);
        // qk DyT norm over head dim
        ggml_tensor* qa=w.get(p+"self_attn.q_norm.alpha"), *qg=w.get(p+"self_attn.q_norm.gamma"), *qb=w.get(p+"self_attn.q_norm.beta");
        ggml_tensor* ka=w.get(p+"self_attn.k_norm.alpha"), *kg=w.get(p+"self_attn.k_norm.gamma"), *kb=w.get(p+"self_attn.k_norm.beta");
        q = dyt(g,q,qa,qg,qb); qd = dyt(g,qd,qa,qg,qb);
        k = dyt(g,k,ka,kg,kb); kd = dyt(g,kd,ka,kg,kb);
        // partial RoPE (NeoX, first 32 dims)
        auto rope = [&](ggml_tensor* a){ return ggml_rope_ext(cctx, a, pos, nullptr, rot, GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f); };
        q = rope(q); qd = rope(qd); k = rope(k); kd = rope(kd);
        // to attention layout [d, N, H]
        auto toAttn = [&](ggml_tensor* a){ return ggml_cont(cctx, ggml_permute(cctx, a, 0, 2, 1, 3)); };
        q=toAttn(q); k=toAttn(k); v=toAttn(v); qd=toAttn(qd); kd=toAttn(kd);
        ggml_tensor* o1 = sdpa(g, q, k, v, mask, scale);
        ggml_tensor* o2 = sdpa(g, qd, kd, v, mask, scale);
        ggml_tensor* o  = ggml_sub(cctx, o1, o2);                 // [d, N, H]
        // merge heads -> [1536, N]
        o = ggml_cont(cctx, ggml_permute(cctx, o, 0, 2, 1, 3));   // [d, H, N]
        o = ggml_reshape_2d(cctx, o, dim, N);
        o = ggml_mul_mat(cctx, w.get(p+"self_attn.to_out.weight"), o);
        x = ggml_add(cctx, x, o);

        // ---- feed-forward (SwiGLU; Sin activation on last `sinb` blocks) ----
        ggml_tensor* f = dyt(g, x, w.get(p+"ff_norm.alpha"), w.get(p+"ff_norm.gamma"), w.get(p+"ff_norm.beta"));
        f = linear(g, w.get(p+"ff.proj.weight"), f, w.get(p+"ff.proj.bias")); // [9216,N]
        int inner = f->ne[0] / 2;
        ggml_tensor* a_   = ggml_view_2d(cctx, f, inner, N, f->nb[1], 0);
        ggml_tensor* gate = ggml_view_2d(cctx, f, inner, N, f->nb[1], (size_t)inner*sizeof(float));
        bool sinusoidal = (depth - l) < sinb;
        ggml_tensor* act = sinusoidal
            ? ggml_sin(cctx, ggml_scale(cctx, ggml_cont(cctx, gate), 3.14159265358979324f))
            : ggml_silu(cctx, ggml_cont(cctx, gate));
        f = ggml_mul(cctx, ggml_cont(cctx, a_), act);            // [4608,N]
        f = linear(g, w.get(p+"ff.out.weight"), f, w.get(p+"ff.out.bias")); // [1536,N]
        x = ggml_add(cctx, x, f);
    }

    // keep last 16 of each 17 -> [1536,16,T] -> [1536,T*16]
    ggml_tensor* x17 = ggml_reshape_3d(cctx, x, dim, sub, T);
    ggml_tensor* kept = ggml_view_3d(cctx, x17, dim, oseg, T, x17->nb[1], x17->nb[2], x17->nb[1]); // skip first token
    kept = ggml_cont(cctx, kept);
    ggml_tensor* up = ggml_reshape_2d(cctx, kept, dim, oseg*T);  // [1536, T*16]
    // mapping conv 1536->512
    ggml_tensor* mapped = ggml_mul_mat(cctx, w.get("ae.dec.mapping.weight"), up); // [512, T*16]
    ggml_tensor* after_resampling = mapped;
    ggml_set_output(after_resampling);

    // unpatchify: [512, L] -> reshape [256,2,L] -> permute [256,L,2] -> [L*256, 2]
    const int L = oseg*T;
    ggml_tensor* pm = ggml_reshape_3d(cctx, mapped, patch, outch/patch, L);   // [256,2,L]
    pm = ggml_cont(cctx, ggml_permute(cctx, pm, 0, 2, 1, 3));                 // [256,L,2]
    ggml_tensor* audio = ggml_reshape_2d(cctx, pm, patch*L, outch/patch);     // [L*256, 2]
    ggml_set_output(audio);

    // --- allocate + set inputs + compute ---
    ggml_cgraph* gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, audio);
    ggml_build_forward_expand(gf, after_in_proj);
    ggml_build_forward_expand(gf, after_resampling);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    // z
    std::vector<float> zbuf((size_t)latent*T);
    { FILE* f = fopen(z_path, "rb"); fread(zbuf.data(), sizeof(float), zbuf.size(), f); fclose(f); }
    ggml_backend_tensor_set(z, zbuf.data(), 0, zbuf.size()*sizeof(float));
    // pos
    std::vector<int32_t> posbuf(N); for (int i=0;i<N;i++) posbuf[i]=i;
    ggml_backend_tensor_set(pos, posbuf.data(), 0, posbuf.size()*sizeof(int32_t));
    // sliding-window band mask [Nk,Nq]: 0 if |q-k|<=win else -inf
    std::vector<float> mbuf((size_t)N*N);
    for (int q=0;q<N;q++) for (int kk=0;kk<N;kk++)
        mbuf[(size_t)q*N+kk] = (std::abs(q-kk) <= win) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, mbuf.data(), 0, mbuf.size()*sizeof(float));

    ggml_backend_graph_compute(w.backend, gf);

    auto dump = [&](ggml_tensor* t, const char* name){
        std::vector<float> b(ggml_nelements(t));
        ggml_backend_tensor_get(t, b.data(), 0, b.size()*sizeof(float));
        std::string fn = std::string(outdir) + "/" + name + ".f32";
        FILE* f = fopen(fn.c_str(), "wb"); fwrite(b.data(), sizeof(float), b.size(), f); fclose(f);
        printf("  dumped %-18s ne=[%lld,%lld,%lld]\n", name,
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2]);
    };
    dump(after_in_proj, "after_in_proj");
    dump(after_resampling, "after_resampling");
    dump(audio, "audio");
    printf("done. T=%d N=%d running_std=%.5f\n", T, N, running_std);
    return 0;
}
