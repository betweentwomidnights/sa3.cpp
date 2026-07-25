// sa3-quant-eval: compare original vs quantized SA3 model outputs.
// Loads one model at a time (never both). Auto-detects architecture from
// general.architecture. Reports MSE, cosine similarity, max abs error, L1.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "gguf_model.h"
#include "t5gemma.h"
#include "dit.h"
#include "same_ae.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// ── Tiny deterministic RNG (xorshift64*) ────────────────────────────────
struct Rng {
    uint64_t s;
    Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return (s * 0x2545F4914F6CDD1DULL) >> 32; }
    float    f32()   { return (next() >> 8) * 0x1p-24f; }
    int32_t  i32(int lo, int hi) { return lo + (int32_t)(next() % (uint32_t)(hi - lo)); }
};

// ── Comparison metrics ─────────────────────────────────────────────────
struct CompareResult {
    const char* name;
    double mse, cosine, max_abs, l1;
    int nan_a, nan_b, inf_a, inf_b;
    int64_t n;
};

static CompareResult compare(const float* a, const float* b, int64_t n, const char* name) {
    CompareResult r{name, 0, 0, 0, 0, 0, 0, 0, 0, n};
    double sum_sq = 0, sum_l1 = 0, dot = 0, nrm_a = 0, nrm_b = 0;
    for (int64_t i = 0; i < n; i++) {
        float va = a[i], vb = b[i];
        if (std::isnan(va)) { r.nan_a++; va = 0; }
        if (std::isnan(vb)) { r.nan_b++; vb = 0; }
        if (std::isinf(va)) { r.inf_a++; va = 0; }
        if (std::isinf(vb)) { r.inf_b++; vb = 0; }
        double d = (double)va - (double)vb;
        sum_sq += d * d;
        sum_l1 += std::abs(d);
        double ad = d > 0 ? d : -d;
        if (ad > r.max_abs) r.max_abs = ad;
        dot   += (double)va * (double)vb;
        nrm_a += (double)va * (double)va;
        nrm_b += (double)vb * (double)vb;
    }
    r.mse     = n ? sum_sq / n : 0;
    r.l1      = n ? sum_l1 / n : 0;
    double dn = sqrt(nrm_a * nrm_b);
    r.cosine  = dn > 0 ? dot / dn : 0;
    return r;
}

static void print_result(const CompareResult& r) {
    fprintf(stdout, "  %-10s  MSE=%.3e  cos=%.6f  max|Δ|=%.3e  L1=%.3e  n=%lld",
            r.name, r.mse, r.cosine, r.max_abs, r.l1, (long long)r.n);
    if (r.nan_a || r.nan_b) fprintf(stdout, "  NaN(orig=%d quant=%d)", r.nan_a, r.nan_b);
    if (r.inf_a || r.inf_b) fprintf(stdout, "  Inf(orig=%d quant=%d)", r.inf_a, r.inf_b);
    fprintf(stdout, "\n");
}

// ── Architecture detection (reads GGUF header only) ────────────────────
static std::string detect_arch(const char* path) {
    gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context* ctx = gguf_init_from_file(path, p);
    if (!ctx) { fprintf(stderr, "error: cannot open %s\n", path); exit(1); }
    int k = gguf_find_key(ctx, "general.architecture");
    std::string arch = k >= 0 ? gguf_get_val_str(ctx, k) : "unknown";
    gguf_free(ctx);
    return arch;
}

// ── T5Gemma eval ───────────────────────────────────────────────────────
static void eval_t5gemma(const char* orig_path, const char* quant_path,
                         int seq, uint64_t seed, const char* outdir,
                         ggml_backend_t forced_backend) {
    fprintf(stdout, "\n--- sa3-t5gemma: %s vs %s ---\n", orig_path, quant_path);
    Rng rng(seed);
    std::vector<int32_t> idbuf(seq);
    std::vector<int32_t> posbuf(seq);
    std::vector<float> mbuf((size_t)seq * seq);
    for (int i = 0; i < seq; i++) idbuf[i] = rng.i32(0, 32000);
    for (int i = 0; i < seq; i++) posbuf[i] = i;
    for (size_t j = 0; j < mbuf.size(); j++) mbuf[j] = 0.0f;

    const ggml_init_params ip = { (size_t)256 * 1024 * 1024, nullptr, /*no_alloc=*/true };
    std::vector<float> orig_out, quant_out;
    int64_t n_elems = 0;

    for (int phase = 0; phase < 2; phase++) {
        const char* path = phase == 0 ? orig_path : quant_path;
        auto W = sa3::load_gguf(path, forced_backend);
        auto c = sa3::T5GemmaConfig::from(W);
        n_elems = (int64_t)c.dim * seq;

        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq); ggml_set_input(ids);
        ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq); ggml_set_input(pos);
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq); ggml_set_input(mask);
        ggml_tensor* hidden = ggml_cont(ctx, sa3::t5gemma_encode(ctx, W, ids, pos, mask, c));
        ggml_set_output(hidden);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, hidden);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(ids,  idbuf.data(), 0, seq * sizeof(int32_t));
        ggml_backend_tensor_set(pos,  posbuf.data(), 0, seq * sizeof(int32_t));
        ggml_backend_tensor_set(mask, mbuf.data(),  0, mbuf.size() * sizeof(float));
        ggml_backend_graph_compute(W.backend, gf);

        std::vector<float> out(n_elems);
        ggml_backend_tensor_get(hidden, out.data(), 0, out.size() * sizeof(float));
        (phase == 0 ? orig_out : quant_out) = std::move(out);

        if (outdir) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s/t5_hidden_%s.f32", outdir, phase == 0 ? "orig" : "quant");
            FILE* f = fopen(fn, "wb");
            fwrite((phase == 0 ? orig_out : quant_out).data(), sizeof(float), n_elems, f);
            fclose(f);
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        W.free();
    }

    auto r = compare(orig_out.data(), quant_out.data(), n_elems, "hidden");
    print_result(r);
}

// ── DiT eval ───────────────────────────────────────────────────────────
static void eval_dit(const char* orig_path, const char* quant_path,
                     int T, int ctx_len, uint64_t seed, const char* outdir,
                     ggml_backend_t forced_backend) {
    fprintf(stdout, "\n--- sa3-dit: %s vs %s ---\n", orig_path, quant_path);
    Rng rng(seed);

    int64_t n_elems = 0;
    std::vector<float> x_inp, tfeat, cross, glob;
    const ggml_init_params ip = { (size_t)512 * 1024 * 1024, nullptr, /*no_alloc=*/true };
    std::vector<float> orig_out, quant_out;

    for (int phase = 0; phase < 2; phase++) {
        const char* path = phase == 0 ? orig_path : quant_path;
        auto W = sa3::load_gguf(path, forced_backend);
        auto c = sa3::DitConfig::from(W);
        const int S = c.mem_tokens + T;
        n_elems = (int64_t)c.io * T;

        if (phase == 0) {
            x_inp.resize((size_t)c.io * T);
            tfeat.resize(c.time_dim);
            cross.resize((size_t)c.cond_dim * ctx_len);
            glob.resize(c.cond_dim);
            for (auto& v : x_inp) v = rng.f32() * 2 - 1;
            for (auto& v : tfeat) v = rng.f32() * 2 - 1;
            for (auto& v : cross) v = rng.f32() * 2 - 1;
            for (auto& v : glob)  v = rng.f32() * 2 - 1;
        }

        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* x_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.io, T); ggml_set_input(x_t);
        ggml_tensor* tf   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.time_dim); ggml_set_input(tf);
        ggml_tensor* cr   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.cond_dim, ctx_len); ggml_set_input(cr);
        ggml_tensor* gl   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.cond_dim); ggml_set_input(gl);
        ggml_tensor* pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S); ggml_set_input(pos);
        ggml_tensor* ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); ggml_set_input(ones);

        ggml_tensor* vel = ggml_cont(ctx, sa3::dit_forward(ctx, W, x_t, tf, cr, gl, pos, ones, c));
        ggml_set_output(vel);

        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, vel);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(x_t, x_inp.data(), 0, x_inp.size() * sizeof(float));
        ggml_backend_tensor_set(tf,  tfeat.data(), 0, tfeat.size() * sizeof(float));
        ggml_backend_tensor_set(cr,  cross.data(), 0, cross.size() * sizeof(float));
        ggml_backend_tensor_set(gl,  glob.data(),  0, glob.size() * sizeof(float));
        std::vector<int32_t> posbuf(S);
        for (int i = 0; i < S; i++) posbuf[i] = i;
        ggml_backend_tensor_set(pos, posbuf.data(), 0, S * sizeof(int32_t));
        float one = 1.0f;
        ggml_backend_tensor_set(ones, &one, 0, sizeof(float));
        ggml_backend_graph_compute(W.backend, gf);

        std::vector<float> out(n_elems);
        ggml_backend_tensor_get(vel, out.data(), 0, out.size() * sizeof(float));
        (phase == 0 ? orig_out : quant_out) = std::move(out);

        if (outdir) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s/dit_velocity_%s.f32", outdir, phase == 0 ? "orig" : "quant");
            FILE* f = fopen(fn, "wb");
            fwrite((phase == 0 ? orig_out : quant_out).data(), sizeof(float), n_elems, f);
            fclose(f);
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        W.free();
    }

    auto r = compare(orig_out.data(), quant_out.data(), n_elems, "velocity");
    print_result(r);
}

// ── SAME encode / decode eval ──────────────────────────────────────────
static void eval_same(const char* orig_path, const char* quant_path,
                      const char* wav_path, int T, const char* outdir,
                      ggml_backend_t forced_backend) {
    fprintf(stdout, "\n--- sa3-ae: %s vs %s ---\n", orig_path, quant_path);

    int n_ch = 0, n_samp = 0, sr = 0;
    std::vector<float> wav = sa3::read_wav_planar(wav_path, n_samp, n_ch, sr);
    fprintf(stdout, "  WAV: %d samples, %d ch, %d Hz\n", n_samp, n_ch, sr);

    const ggml_init_params ip = { (size_t)512 * 1024 * 1024, nullptr, /*no_alloc=*/true };

    std::vector<float> orig_latent, quant_latent;
    std::vector<float> orig_audio, quant_audio;
    int64_t latent_elems = 0, audio_elems = 0;

    // ── Phase 1: orig ──────────────────────────────────────────────
    {
        auto W = sa3::load_gguf(orig_path, forced_backend);
        auto c = sa3::SameConfig::from(W);
        const int64_t L = (int64_t)T * c.patch_size * c.output_seg;
        const int64_t N = (int64_t)T * c.sub_chunk;
        latent_elems = (int64_t)c.latent * T;

        if (n_samp < L) {
            fprintf(stderr, "error: WAV too short (%d < %lld)\n", n_samp, (long long)L);
            exit(1);
        }
        if (n_ch != (c.out_channels / c.patch_size)) {
            fprintf(stderr, "error: WAV ch=%d but model expects %d\n", n_ch, c.out_channels / c.patch_size);
            exit(1);
        }

        std::vector<float> audio_buf((size_t)L * n_ch);
        for (int64_t s = 0; s < L; s++)
            for (int c2 = 0; c2 < n_ch; c2++)
                audio_buf[(size_t)c2 * L + s] = wav[(size_t)c2 * n_samp + s];

        // ── Encode ──
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* audio = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, n_ch); ggml_set_input(audio);
        ggml_tensor* pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N); ggml_set_input(pos);
        ggml_tensor* mask  = c.chunk
            ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N)
            : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * c.sub_chunk, c.sub_chunk, N / c.sub_chunk);
        ggml_set_input(mask);
        ggml_tensor *pos2 = nullptr, *mask2 = nullptr;
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            pos2  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N2); ggml_set_input(pos2);
            mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N2, N2); ggml_set_input(mask2);
        }

        auto enc = sa3::same_encode(ctx, W, audio, c, T, pos, mask, pos2, mask2);
        ggml_tensor* enc_latent = ggml_cont(ctx, enc.latent);
        ggml_set_output(enc_latent);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, enc_latent);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(audio, audio_buf.data(), 0, audio_buf.size() * sizeof(float));
        std::vector<int32_t> posbuf((size_t)N);
        for (int i = 0; i < (int)N; i++) posbuf[i] = i;
        ggml_backend_tensor_set(pos, posbuf.data(), 0, N * sizeof(int32_t));
        if (c.chunk) {
            std::vector<float> mb = sa3::build_attn_mask(c, (int)N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        } else {
            std::vector<float> mb = sa3::build_swa_bias(c, N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        }
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            std::vector<int32_t> p2((size_t)N2);
            for (int i = 0; i < (int)N2; i++) p2[i] = i;
            ggml_backend_tensor_set(pos2, p2.data(), 0, N2 * sizeof(int32_t));
            std::vector<float> mb2 = sa3::build_attn_mask(c, (int)N2);
            ggml_backend_tensor_set(mask2, mb2.data(), 0, mb2.size() * sizeof(float));
        }
        ggml_backend_graph_compute(W.backend, gf);

        orig_latent.resize(latent_elems);
        ggml_backend_tensor_get(enc_latent, orig_latent.data(), 0, orig_latent.size() * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);

        // ── Decode (from orig_latent) ──
        ctx = ggml_init(ip);
        ggml_tensor* z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.latent, T); ggml_set_input(z);
        pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N); ggml_set_input(pos);
        mask = c.chunk
            ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N)
            : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * c.sub_chunk, c.sub_chunk, N / c.sub_chunk);
        ggml_set_input(mask);
        pos2 = mask2 = nullptr;
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            pos2  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N2); ggml_set_input(pos2);
            mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N2, N2); ggml_set_input(mask2);
        }

        auto dec = sa3::same_decode(ctx, W, z, c, T, pos, mask, pos2, mask2);
        ggml_tensor* dec_audio = ggml_cont(ctx, dec.audio);
        ggml_set_output(dec_audio);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dec_audio);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(z, orig_latent.data(), 0, orig_latent.size() * sizeof(float));
        ggml_backend_tensor_set(pos, posbuf.data(), 0, N * sizeof(int32_t));
        if (c.chunk) {
            std::vector<float> mb = sa3::build_attn_mask(c, (int)N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        } else {
            std::vector<float> mb = sa3::build_swa_bias(c, N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        }
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            std::vector<int32_t> p2((size_t)N2);
            for (int i = 0; i < (int)N2; i++) p2[i] = i;
            ggml_backend_tensor_set(pos2, p2.data(), 0, N2 * sizeof(int32_t));
        }
        ggml_backend_graph_compute(W.backend, gf);

        orig_audio.resize(ggml_nelements(dec_audio));
        ggml_backend_tensor_get(dec_audio, orig_audio.data(), 0, orig_audio.size() * sizeof(float));
        audio_elems = orig_audio.size();

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        W.free();
    }

    // ── Phase 2: quant ──────────────────────────────────────────────
    {
        auto W = sa3::load_gguf(quant_path, forced_backend);
        auto c = sa3::SameConfig::from(W);
        const int64_t L = (int64_t)T * c.patch_size * c.output_seg;
        const int64_t N = (int64_t)T * c.sub_chunk;

        std::vector<float> audio_buf((size_t)L * n_ch);
        for (int64_t s = 0; s < L; s++)
            for (int c2 = 0; c2 < n_ch; c2++)
                audio_buf[(size_t)c2 * L + s] = wav[(size_t)c2 * n_samp + s];

        // ── Encode ──
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* audio = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, n_ch); ggml_set_input(audio);
        ggml_tensor* pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N); ggml_set_input(pos);
        ggml_tensor* mask  = c.chunk
            ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N)
            : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * c.sub_chunk, c.sub_chunk, N / c.sub_chunk);
        ggml_set_input(mask);
        ggml_tensor *pos2 = nullptr, *mask2 = nullptr;
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            pos2  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N2); ggml_set_input(pos2);
            mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N2, N2); ggml_set_input(mask2);
        }

        auto enc = sa3::same_encode(ctx, W, audio, c, T, pos, mask, pos2, mask2);
        ggml_tensor* enc_latent = ggml_cont(ctx, enc.latent);
        ggml_set_output(enc_latent);

        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, enc_latent);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(audio, audio_buf.data(), 0, audio_buf.size() * sizeof(float));
        std::vector<int32_t> posbuf((size_t)N);
        for (int i = 0; i < (int)N; i++) posbuf[i] = i;
        ggml_backend_tensor_set(pos, posbuf.data(), 0, N * sizeof(int32_t));
        if (c.chunk) {
            std::vector<float> mb = sa3::build_attn_mask(c, (int)N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        } else {
            std::vector<float> mb = sa3::build_swa_bias(c, N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        }
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            std::vector<int32_t> p2((size_t)N2);
            for (int i = 0; i < (int)N2; i++) p2[i] = i;
            ggml_backend_tensor_set(pos2, p2.data(), 0, N2 * sizeof(int32_t));
            std::vector<float> mb2 = sa3::build_attn_mask(c, (int)N2);
            ggml_backend_tensor_set(mask2, mb2.data(), 0, mb2.size() * sizeof(float));
        }
        ggml_backend_graph_compute(W.backend, gf);

        quant_latent.resize(latent_elems);
        ggml_backend_tensor_get(enc_latent, quant_latent.data(), 0, quant_latent.size() * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx);

        // ── Decode (from orig_latent for quality isolation) ──
        ctx = ggml_init(ip);
        ggml_tensor* z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.latent, T); ggml_set_input(z);
        pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N); ggml_set_input(pos);
        mask = c.chunk
            ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N)
            : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * c.sub_chunk, c.sub_chunk, N / c.sub_chunk);
        ggml_set_input(mask);
        pos2 = mask2 = nullptr;
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            pos2  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N2); ggml_set_input(pos2);
            mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N2, N2); ggml_set_input(mask2);
        }

        auto dec = sa3::same_decode(ctx, W, z, c, T, pos, mask, pos2, mask2);
        ggml_tensor* dec_audio = ggml_cont(ctx, dec.audio);
        ggml_set_output(dec_audio);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dec_audio);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(z, orig_latent.data(), 0, orig_latent.size() * sizeof(float));
        ggml_backend_tensor_set(pos, posbuf.data(), 0, N * sizeof(int32_t));
        if (c.chunk) {
            std::vector<float> mb = sa3::build_attn_mask(c, (int)N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        } else {
            std::vector<float> mb = sa3::build_swa_bias(c, N);
            ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
        }
        if (c.chunk) {
            const int64_t N2 = N + 2 * c.shift;
            std::vector<int32_t> p2((size_t)N2);
            for (int i = 0; i < (int)N2; i++) p2[i] = i;
            ggml_backend_tensor_set(pos2, p2.data(), 0, N2 * sizeof(int32_t));
        }
        ggml_backend_graph_compute(W.backend, gf);

        quant_audio.resize(audio_elems);
        ggml_backend_tensor_get(dec_audio, quant_audio.data(), 0, quant_audio.size() * sizeof(float));

        if (outdir) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s/same_latent_orig.f32", outdir);
            FILE* f = fopen(fn, "wb"); fwrite(orig_latent.data(), sizeof(float), latent_elems, f); fclose(f);
            snprintf(fn, sizeof(fn), "%s/same_latent_quant.f32", outdir);
            f = fopen(fn, "wb"); fwrite(quant_latent.data(), sizeof(float), latent_elems, f); fclose(f);
            snprintf(fn, sizeof(fn), "%s/same_audio_orig.f32", outdir);
            f = fopen(fn, "wb"); fwrite(orig_audio.data(), sizeof(float), audio_elems, f); fclose(f);
            snprintf(fn, sizeof(fn), "%s/same_audio_quant.f32", outdir);
            f = fopen(fn, "wb"); fwrite(quant_audio.data(), sizeof(float), audio_elems, f); fclose(f);
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        W.free();
    }

    auto r_lat = compare(orig_latent.data(), quant_latent.data(), latent_elems, "latent");
    print_result(r_lat);
    auto r_aud = compare(orig_audio.data(), quant_audio.data(), audio_elems, "audio");
    print_result(r_aud);
}

// ── main ───────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    const char* wav_path = nullptr;
    const char* outdir = nullptr;
    uint64_t seed = 42;
    int t5_seq = 64;
    int dit_T = 4, dit_ctx = 16;
    int same_T = 4;
    bool use_cpu = false;

    struct ModelPair { const char* orig; const char* quant; };
    std::vector<ModelPair> models;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 2 < argc) {
            models.push_back({argv[++i], argv[++i]});
        } else if (!strcmp(argv[i], "--wav") && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            outdir = argv[++i];
        } else if (!strcmp(argv[i], "--cpu")) {
            use_cpu = true;
        } else if (!strcmp(argv[i], "--t5-seq") && i + 1 < argc) {
            t5_seq = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--dit-frames") && i + 1 < argc) {
            dit_T = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--dit-ctx") && i + 1 < argc) {
            dit_ctx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--same-frames") && i + 1 < argc) {
            same_T = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: sa3-quant-eval --model <orig.gguf> <quant.gguf> [--model ...]\n"
                            "       [--wav <in.wav>] [--seed N] [--out <dir>] [--cpu]\n"
                            "       [--t5-seq N] [--dit-frames N] [--dit-ctx N] [--same-frames N]\n");
            return 1;
        }
    }

    if (models.empty()) {
        fprintf(stderr, "error: at least one --model pair required\n");
        return 1;
    }

    bool need_wav = false;
    for (auto& m : models) {
        std::string arch = detect_arch(m.orig);
        if (arch == "sa3-ae") need_wav = true;
        if (arch != "sa3-t5gemma" && arch != "sa3-dit" && arch != "sa3-ae") {
            fprintf(stderr, "error: unsupported architecture '%s' in %s\n", arch.c_str(), m.orig);
            return 1;
        }
    }
    if (need_wav && !wav_path) {
        fprintf(stderr, "error: --wav <file.wav> required for sa3-ae model\n");
        return 1;
    }

    // If --cpu, create a CPU backend; else null = auto-select
    ggml_backend_t forced_backend = nullptr;
    if (use_cpu) {
        forced_backend = ggml_backend_cpu_init();
        if (!forced_backend) { fprintf(stderr, "error: failed to init CPU backend\n"); return 1; }
        fprintf(stdout, "using CPU backend\n");
    }

    for (auto& m : models) {
        std::string arch = detect_arch(m.orig);
        if (arch == "sa3-t5gemma") {
            eval_t5gemma(m.orig, m.quant, t5_seq, seed, outdir, forced_backend);
        } else if (arch == "sa3-dit") {
            eval_dit(m.orig, m.quant, dit_T, dit_ctx, seed, outdir, forced_backend);
        } else if (arch == "sa3-ae") {
            eval_same(m.orig, m.quant, wav_path, same_T, outdir, forced_backend);
        }
    }

    if (forced_backend) ggml_backend_free(forced_backend);
    return 0;
}
