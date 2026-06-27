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
    const char* init_p = nullptr;            // audio2audio / inpaint: source WAV (encoded to z_init)
    float init_noise_level = 0.85f;          // sigma_max for audio2audio (1.0 == text2music)
    float inpaint_start = -1.0f, inpaint_end = -1.0f;   // inpaint: regenerate this [start,end] sec region
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
        else if (!strcmp(argv[i], "--init")   && i+1 < argc) init_p = argv[++i];
        else if (!strcmp(argv[i], "--init-noise-level") && i+1 < argc) init_noise_level = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--inpaint-start") && i+1 < argc) inpaint_start = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--inpaint-end")   && i+1 < argc) inpaint_end   = (float)atof(argv[++i]);
    }
    const bool inpaint = (inpaint_start >= 0.0f || inpaint_end >= 0.0f);   // inpaint mode (needs --init source)
    if (!tok_p || !t5_p || !dit_p || !same_p) {
        fprintf(stderr, "usage: sa3-generate --tok <f> --t5 <f> --dit <f> --same <f> --prompt \"...\" --frames N --steps N --seed S --out song.wav\n");
        return 1;
    }

    sa3::Tokenizer tok = sa3::Tokenizer::load(tok_p);
    sa3::GgufModel TE = sa3::load_gguf(t5_p), DIT = sa3::load_gguf(dit_p), AE = sa3::load_gguf(same_p);
    const sa3::T5GemmaConfig tc = sa3::T5GemmaConfig::from(TE);
    const sa3::DitConfig     dc = sa3::DitConfig::from(DIT);
    const sa3::SameConfig    sc = sa3::SameConfig::from(AE);
    const int ds = sc.patch_size * sc.output_seg;          // downsampling ratio (4096 samples/frame)

    // ---------- load source WAV; derive output T (overrides --frames) ----------
    // audio2audio: output length = the source. inpaint/continuation: output length =
    // max(source, --inpaint-end) so a short clip can be extended (mask_end = total duration).
    std::vector<float> init_audio; int init_L = 0;         // padded planar [init_L, 2]
    if (init_p) {
        int n_samp, n_ch, sr;
        std::vector<float> raw = sa3::read_wav_planar(init_p, n_samp, n_ch, sr);
        if (n_ch != sc.out_channels / sc.patch_size) { fprintf(stderr, "init WAV must be %d-channel\n", sc.out_channels / sc.patch_size); return 1; }
        if (sr != 44100) fprintf(stderr, "warning: init WAV is %d Hz, expected 44100\n", sr);
        int want = n_samp;
        if (inpaint && inpaint_end > 0.0f) want = std::max(want, (int)(inpaint_end * 44100.0f));
        // pad up so T is an integer (and EVEN for SAME-S, which needs T*17 divisible by eff_chunk)
        const int mult = sc.chunk ? 2 * ds : ds;
        init_L = ((want + mult - 1) / mult) * mult;
        init_audio.assign((size_t)init_L * n_ch, 0.0f);
        const int copy = std::min(n_samp, init_L);          // source at the start, zero-padded tail
        for (int c = 0; c < n_ch; c++)
            memcpy(&init_audio[(size_t)c*init_L], &raw[(size_t)c*n_samp], copy * sizeof(float));
        frames = init_L / ds;
        printf("%s: src \"%s\" %.2fs -> output T=%d (%.2fs)\n",
               inpaint ? "inpaint" : "audio2audio", init_p, (float)n_samp/44100.0f, frames, (float)init_L/44100.0f);
    }

    const int T = frames, max_len = (int)TE.u32("t5g.max_length");
    const int cond_dim = tc.dim, ctx_len = max_len + 1;     // t5gemma tokens + 1 seconds token
    const int N = T * dc.io;
    // inpaint runs from pure noise (sigma_max=1) + local_add_cond; audio2audio mixes init at sigma_max<1
    const float sigma_max = (init_p && !inpaint) ? init_noise_level : 1.0f;
    if (inpaint && (!init_p || dc.local_dim <= 0)) {
        fprintf(stderr, "inpaint needs --init <wav> (source audio) and a DiT with local-cond weights (dit.local_dim>0)\n");
        return 1;
    }

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
        float t_in = sigma_max * (1.0f - (float)i / steps);    // linspace(sigma_max,0) before the LogSNR shift
        sigmas[i] = (i == 0) ? sigma_max : (i == steps) ? 0.0f : 1.0f / (1.0f + expf(-(coef*t_in - logsnr_end)));
    }

    // ---------- audio2audio: encode init audio -> latent z_init [latent, T] ----------
    std::vector<float> z_init;
    if (init_p) {
        z_init.resize(N);
        const int Nenc  = T * sc.sub_chunk;
        const int Nenc2 = sc.chunk ? Nenc + 2*sc.shift : 0;
        ggml_init_params encp = { (size_t)512*1024*1024, nullptr, true };
        ggml_context* enctx = ggml_init(encp);
        ggml_tensor* a_in   = ggml_new_tensor_2d(enctx, GGML_TYPE_F32, init_L, sc.out_channels / sc.patch_size);
        ggml_tensor* pos_a  = ggml_new_tensor_1d(enctx, GGML_TYPE_I32, Nenc);
        ggml_tensor* mask_a = ggml_new_tensor_2d(enctx, GGML_TYPE_F32, Nenc, Nenc);
        ggml_set_input(a_in); ggml_set_input(pos_a); ggml_set_input(mask_a);
        ggml_tensor *pos_a2 = nullptr, *mask_a2 = nullptr;
        if (sc.chunk) {
            pos_a2  = ggml_new_tensor_1d(enctx, GGML_TYPE_I32, Nenc2);
            mask_a2 = ggml_new_tensor_2d(enctx, GGML_TYPE_F32, Nenc2, Nenc2);
            ggml_set_input(pos_a2); ggml_set_input(mask_a2);
        }
        ggml_tensor* zt = ggml_cont(enctx, sa3::same_encode(enctx, AE, a_in, sc, T, pos_a, mask_a, pos_a2, mask_a2).z);
        ggml_set_output(zt);
        ggml_cgraph* gf_enc = ggml_new_graph_custom(enctx, 32768, false);
        ggml_build_forward_expand(gf_enc, zt);
        ggml_gallocr_t alloc_enc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(AE.backend));
        ggml_gallocr_alloc_graph(alloc_enc, gf_enc);
        ggml_backend_tensor_set(a_in, init_audio.data(), 0, init_audio.size()*sizeof(float));
        auto set_pos  = [&](ggml_tensor* p, int n){ std::vector<int32_t> b(n); for (int i=0;i<n;i++) b[i]=i; ggml_backend_tensor_set(p, b.data(), 0, n*sizeof(int32_t)); };
        auto set_mask = [&](ggml_tensor* mt, int M){ std::vector<float> b = sa3::build_attn_mask(sc, M); ggml_backend_tensor_set(mt, b.data(), 0, b.size()*sizeof(float)); };
        set_pos(pos_a, Nenc); set_mask(mask_a, Nenc);
        if (sc.chunk) { set_pos(pos_a2, Nenc2); set_mask(mask_a2, Nenc2); }
        ggml_backend_graph_compute(AE.backend, gf_enc);
        ggml_backend_tensor_get(zt, z_init.data(), 0, N*sizeof(float));
        ggml_gallocr_free(alloc_enc); ggml_free(enctx);
    }

    // ---------- noise (audio2audio: noise = init*(1-sigma_max) + noise*sigma_max) ----------
    sa3::Rng rng(seed);
    std::vector<float> host_x(N); rng.fill_normal(host_x.data(), N);
    std::vector<float> stepnoise((size_t)steps*N); rng.fill_normal(stepnoise.data(), stepnoise.size());
    if (init_p && !inpaint)
        for (int j = 0; j < N; j++) host_x[j] = z_init[j]*(1.0f - sigma_max) + host_x[j]*sigma_max;

    // ---------- inpaint: build local_add_cond = [mask(1) | z_init*mask(256)] in ggml [local_dim, T] ----------
    // mask: 1 = keep (known context), 0 = inpaint (regenerate); the [start,end] sec window is masked out.
    std::vector<float> localb;
    if (inpaint) {
        // match the reference: mask at audio-sample res (int() trunc) then nearest-downsample to frames
        // == frame f masked iff f*ds in [int(start*sr), int(end*sr)) == f0=ceil(int(start*sr)/ds).
        auto ceil_div = [](int a, int b){ return (a + b - 1) / b; };
        const int sa = (int)(inpaint_start * 44100.0f);
        const int ea = inpaint_end < 0 ? T*ds : (int)(inpaint_end * 44100.0f);
        const int f0 = std::max(0, std::min(T, ceil_div(sa, ds)));
        const int f1 = std::max(f0, std::min(T, ceil_div(ea, ds)));
        localb.assign((size_t)dc.local_dim * T, 0.0f);
        for (int t = 0; t < T; t++) {
            float m = (t >= f0 && t < f1) ? 0.0f : 1.0f;          // 0 inside the inpaint window
            localb[(size_t)t*dc.local_dim + 0] = m;               // channel 0 = mask
            for (int c = 0; c < dc.io; c++)                       // channels 1..256 = z_init * mask
                localb[(size_t)t*dc.local_dim + 1 + c] = z_init[(size_t)t*dc.io + c] * m;
        }
        printf("inpaint: regenerating frames [%d,%d) of %d (%.2f-%.2fs), keeping the rest\n",
               f0, f1, T, f0 * (float)ds / 44100.0f, f1 * (float)ds / 44100.0f);
    }

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
    ggml_tensor* local = nullptr;
    if (inpaint) { local = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.local_dim, T); ggml_set_input(local); }
    ggml_tensor* vel = ggml_cont(dctx, sa3::dit_forward(dctx, DIT, x_in, tfeat, cross, glob, pos_d, ones, dc, local));
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
        if (local) ggml_backend_tensor_set(local, localb.data(), 0, localb.size()*sizeof(float));  // re-set (gallocr recycle)
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
