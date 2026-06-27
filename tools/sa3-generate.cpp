// sa3-generate: standalone text2music. prompt string -> WAV, no PyTorch in the loop.
// tokenize -> T5Gemma encode -> conditioning assembly (learned padding + seconds)
// -> ping-pong sampler over the DiT -> SAME-L decode -> WAV.
#include "gguf_model.h"
#include "tokenizer.h"
#include "t5gemma.h"
#include "dit.h"
#include "same_ae.h"
#include "rng.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ExpoFourierFeatures(dim, min, max): [cos, sin](t * f * 2pi), f log-spaced.
static void expo_features(float t, std::vector<float>& out, int dim, float min_f, float max_f) {
    const int half = dim / 2;
    const float lmin = logf(min_f), lmax = logf(max_f), TWO_PI = 6.28318530717958648f;
    out.resize(dim);
    for (int i = 0; i < half; i++) {
        float ramp = half > 1 ? (float)i / (half - 1) : 0.0f;
        float freq = expf(ramp * (lmax - lmin) + lmin);
        float arg = t * freq * TWO_PI;
        out[i] = cosf(arg); out[half + i] = sinf(arg);
    }
}

static std::vector<float> tensor_to_host(const sa3::GgufModel& M, const std::string& name) {
    ggml_tensor* t = M.get(name);
    std::vector<float> b(ggml_nelements(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size() * sizeof(float));
    return b;
}

int main(int argc, char** argv) {
    const char* tok_p = nullptr; const char* t5_p = nullptr; const char* dit_p = nullptr; const char* same_p = nullptr;
    std::string prompt = "Upbeat funk groove with slap bass, bright horns, tight drums";
    const char* wav_p = "song.wav";
    int frames = 128, steps = 8; uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--tok")    && i+1 < argc) tok_p = argv[++i];
        else if (!strcmp(argv[i], "--t5")     && i+1 < argc) t5_p = argv[++i];
        else if (!strcmp(argv[i], "--dit")    && i+1 < argc) dit_p = argv[++i];
        else if (!strcmp(argv[i], "--same")   && i+1 < argc) same_p = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")  && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")   && i+1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) wav_p = argv[++i];
    }
    if (!tok_p || !t5_p || !dit_p || !same_p) {
        fprintf(stderr, "usage: sa3-generate --tok <f> --t5 <f> --dit <f> --same <f> --prompt \"...\" --frames N --steps N --seed S --out song.wav\n");
        return 1;
    }

    sa3::Tokenizer tok = sa3::Tokenizer::load(tok_p);
    sa3::GgufModel TE = sa3::load_gguf(t5_p), DIT = sa3::load_gguf(dit_p), AE = sa3::load_gguf(same_p);
    const sa3::T5GemmaConfig tc = sa3::T5GemmaConfig::from(TE);
    const sa3::DitConfig     dc = sa3::DitConfig::from(DIT);
    const sa3::SameConfig    sc = sa3::SameConfig::from(AE);
    const int T = frames, max_len = (int)TE.u32("t5g.max_length");
    const int cond_dim = tc.dim, ctx_len = max_len + 1;     // t5gemma tokens + 1 seconds token
    const int N = T * dc.io;

    // ---------- tokenize + pad ----------
    std::vector<int32_t> enc = tok.encode(prompt);
    const int L = std::min((int)enc.size(), max_len);
    std::vector<int32_t> ids(max_len, tok.pad_id), attn(max_len, 0);
    for (int i = 0; i < L; i++) { ids[i] = enc[i]; attn[i] = 1; }
    printf("prompt: \"%s\"  (%d tokens, ~%.2fs)\n", prompt.c_str(), L, (float)T * (sc.patch_size * sc.output_seg) / 44100.0f);

    // ---------- T5Gemma encode ----------
    std::vector<float> hidden;
    {
        ggml_init_params ip = { (size_t)256*1024*1024, nullptr, true };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* ids_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len);
        ggml_tensor* pos_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len);
        ggml_tensor* mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_len, max_len);
        for (ggml_tensor* t : {ids_t, pos_t, mask_t}) ggml_set_input(t);
        ggml_tensor* h = ggml_cont(ctx, sa3::t5gemma_encode(ctx, TE, ids_t, pos_t, mask_t, tc));
        ggml_set_output(h);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, h);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(TE.backend));
        ggml_gallocr_alloc_graph(alloc, gf);
        ggml_backend_tensor_set(ids_t, ids.data(), 0, max_len*sizeof(int32_t));
        std::vector<int32_t> pos(max_len); for (int i = 0; i < max_len; i++) pos[i] = i;
        ggml_backend_tensor_set(pos_t, pos.data(), 0, max_len*sizeof(int32_t));
        std::vector<float> mb((size_t)max_len*max_len);
        for (int q = 0; q < max_len; q++) for (int k = 0; k < max_len; k++)
            mb[(size_t)q*max_len+k] = attn[k] ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(mask_t, mb.data(), 0, mb.size()*sizeof(float));
        ggml_backend_graph_compute(TE.backend, gf);
        hidden.resize((size_t)cond_dim*max_len);
        ggml_backend_tensor_get(h, hidden.data(), 0, hidden.size()*sizeof(float));
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }

    // ---------- conditioning assembly (host) ----------
    // learned padding: replace padded token columns with the padding embedding
    std::vector<float> pad_emb = tensor_to_host(TE, "te.padding_embedding");   // [cond_dim]
    for (int p = 0; p < max_len; p++)
        if (!attn[p]) memcpy(&hidden[(size_t)p*cond_dim], pad_emb.data(), cond_dim*sizeof(float));
    // seconds_total NumberConditioner: expo(clamp(secs)/range) -> Linear(secs_dim->cond_dim)
    const float secs = (float)T * (sc.patch_size * sc.output_seg) / 44100.0f;
    const float smin = TE.f32("t5g.secs_min"), smax = TE.f32("t5g.secs_max");
    const int sdim = (int)TE.u32("t5g.secs_dim");
    float sclamp = secs < smin ? smin : (secs > smax ? smax : secs);
    float snorm = (sclamp - smin) / (smax - smin);
    std::vector<float> ef; expo_features(snorm, ef, sdim, 0.5f, 10000.0f);
    std::vector<float> sw = tensor_to_host(TE, "te.secs.weight");   // ggml [sdim, cond_dim]
    std::vector<float> sb = tensor_to_host(TE, "te.secs.bias");     // [cond_dim]
    std::vector<float> secs_embed(cond_dim);
    for (int d = 0; d < cond_dim; d++) {
        float acc = sb[d];
        for (int i = 0; i < sdim; i++) acc += ef[i] * sw[(size_t)d*sdim + i];
        secs_embed[d] = acc;
    }
    // cross [cond_dim, 257] = [hidden(256 cols) | seconds(1 col)] ; global = seconds
    std::vector<float> crossb((size_t)cond_dim*ctx_len);
    memcpy(crossb.data(), hidden.data(), hidden.size()*sizeof(float));
    memcpy(&crossb[(size_t)cond_dim*max_len], secs_embed.data(), cond_dim*sizeof(float));
    std::vector<float>& globb = secs_embed;
    if (const char* dc_dir = getenv("SA3_DUMP_COND")) {   // debug: validate conditioning vs PyTorch
        FILE* f1 = fopen((std::string(dc_dir)+"/gen_cross.f32").c_str(), "wb");
        fwrite(crossb.data(), sizeof(float), crossb.size(), f1); fclose(f1);
        FILE* f2 = fopen((std::string(dc_dir)+"/gen_global.f32").c_str(), "wb");
        fwrite(globb.data(), sizeof(float), globb.size(), f2); fclose(f2);
    }

    // ---------- schedule (LogSNRShift rate=0: sigmoid((end-start)*t - end), endpoints forced) ----------
    const float logsnr_start = -6.2f, logsnr_end = 2.0f, coef = logsnr_end - logsnr_start;
    std::vector<float> sigmas(steps+1);
    for (int i = 0; i <= steps; i++) {
        float t_lin = 1.0f - (float)i / steps;
        sigmas[i] = (i == 0) ? 1.0f : (i == steps) ? 0.0f : 1.0f / (1.0f + expf(-(coef*t_lin - logsnr_end)));
    }

    // ---------- noise ----------
    sa3::Rng rng(seed);
    std::vector<float> host_x(N); rng.fill_normal(host_x.data(), N);
    std::vector<float> stepnoise((size_t)steps*N); rng.fill_normal(stepnoise.data(), stepnoise.size());

    // ---------- DiT ping-pong ----------
    const int S = dc.mem_tokens + T;
    ggml_init_params dip = { (size_t)512*1024*1024, nullptr, true };
    ggml_context* dctx = ggml_init(dip);
    ggml_tensor* x_in  = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.io, T);
    ggml_tensor* tfeat = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, dc.time_dim);
    ggml_tensor* cross = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, cond_dim, ctx_len);
    ggml_tensor* glob  = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, cond_dim);
    ggml_tensor* pos_d = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, S);
    ggml_tensor* ones  = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, 1);
    for (ggml_tensor* t : {x_in, tfeat, cross, glob, pos_d, ones}) ggml_set_input(t);
    ggml_tensor* vel = ggml_cont(dctx, sa3::dit_forward(dctx, DIT, x_in, tfeat, cross, glob, pos_d, ones, dc));
    ggml_set_output(vel);
    ggml_cgraph* gf_dit = ggml_new_graph_custom(dctx, 32768, false);
    ggml_build_forward_expand(gf_dit, vel);
    ggml_gallocr_t alloc_dit = ggml_gallocr_new(ggml_backend_get_default_buffer_type(DIT.backend));
    ggml_gallocr_alloc_graph(alloc_dit, gf_dit);
    std::vector<int32_t> posb(S); for (int i = 0; i < S; i++) posb[i] = i;
    const float one = 1.0f;
    std::vector<float> tf, vbuf(N);
    for (int i = 0; i < steps; i++) {
        ggml_backend_tensor_set(cross, crossb.data(), 0, crossb.size()*sizeof(float));  // re-set each step (gallocr recycle)
        ggml_backend_tensor_set(glob,  globb.data(),  0, globb.size()*sizeof(float));
        ggml_backend_tensor_set(pos_d, posb.data(),   0, posb.size()*sizeof(int32_t));
        ggml_backend_tensor_set(ones,  &one, 0, sizeof(float));
        ggml_backend_tensor_set(x_in,  host_x.data(), 0, N*sizeof(float));
        expo_features(sigmas[i], tf, dc.time_dim, dc.time_min_freq, dc.time_max_freq);
        ggml_backend_tensor_set(tfeat, tf.data(), 0, tf.size()*sizeof(float));
        ggml_backend_graph_compute(DIT.backend, gf_dit);
        ggml_backend_tensor_get(vel, vbuf.data(), 0, N*sizeof(float));
        const float tcur = sigmas[i], tnext = sigmas[i+1];
        for (int j = 0; j < N; j++) {
            float denoised = host_x[j] - tcur * vbuf[j];
            host_x[j] = (1.0f - tnext) * denoised + tnext * stepnoise[(size_t)i*N + j];
        }
        printf("  step %d/%d  t=%.4f\n", i+1, steps, tcur);
    }
    ggml_gallocr_free(alloc_dit); ggml_free(dctx);

    // ---------- decode -> WAV ----------
    const int Ndec = T * sc.sub_chunk;
    ggml_init_params eip = { (size_t)512*1024*1024, nullptr, true };
    ggml_context* ectx = ggml_init(eip);
    ggml_tensor* z = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, sc.latent, T);
    const int N2 = sc.chunk ? Ndec + 2*sc.shift : 0;       // SAME-S needs a shifted 2nd mask
    ggml_tensor* pos_e = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, Ndec);
    ggml_tensor* mask_e = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, Ndec, Ndec);
    ggml_set_input(z); ggml_set_input(pos_e); ggml_set_input(mask_e);
    ggml_tensor *pos2_e = nullptr, *mask2_e = nullptr;
    if (sc.chunk) {
        pos2_e  = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, N2);
        mask2_e = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, N2, N2);
        ggml_set_input(pos2_e); ggml_set_input(mask2_e);
    }
    ggml_tensor* audio = ggml_cont(ectx, sa3::same_decode(ectx, AE, z, sc, T, pos_e, mask_e, pos2_e, mask2_e).audio);
    ggml_set_output(audio);
    ggml_cgraph* gf_dec = ggml_new_graph_custom(ectx, 32768, false);
    ggml_build_forward_expand(gf_dec, audio);
    ggml_gallocr_t alloc_dec = ggml_gallocr_new(ggml_backend_get_default_buffer_type(AE.backend));
    ggml_gallocr_alloc_graph(alloc_dec, gf_dec);
    ggml_backend_tensor_set(z, host_x.data(), 0, N*sizeof(float));
    auto set_pos = [&](ggml_tensor* p, int n){ std::vector<int32_t> b(n); for (int i=0;i<n;i++) b[i]=i; ggml_backend_tensor_set(p, b.data(), 0, n*sizeof(int32_t)); };
    auto set_mask = [&](ggml_tensor* mt, int M){
        std::vector<float> b = sa3::build_attn_mask(sc, M);
        ggml_backend_tensor_set(mt, b.data(), 0, b.size()*sizeof(float));
    };
    set_pos(pos_e, Ndec); set_mask(mask_e, Ndec);
    if (sc.chunk) { set_pos(pos2_e, N2); set_mask(mask2_e, N2); }
    ggml_backend_graph_compute(AE.backend, gf_dec);
    const int n_samp = (int)audio->ne[0], n_ch = (int)audio->ne[1];
    std::vector<float> ab((size_t)n_samp*n_ch);
    ggml_backend_tensor_get(audio, ab.data(), 0, ab.size()*sizeof(float));
    sa3::write_wav_planar(wav_p, ab.data(), n_samp, n_ch, 44100);
    printf("wrote %s  (%.2fs, seed %llu)\n", wav_p, (float)n_samp/44100.0f, (unsigned long long)seed);

    ggml_gallocr_free(alloc_dec); ggml_free(ectx);
    TE.free(); DIT.free(); AE.free();
    return 0;
}
