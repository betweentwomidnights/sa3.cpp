// sa3-generate: standalone text2music. prompt string -> WAV, no PyTorch in the loop.
// tokenize -> T5Gemma encode -> conditioning assembly (learned padding + seconds)
// -> ping-pong sampler over the DiT -> SAME-L decode -> WAV.
#include "gguf_model.h"
#include "tokenizer.h"
#include "t5gemma.h"
#include "dit.h"
#include "same_ae.h"
#include "lora.h"
#include "rng.h"
#include "wav.h"

#include <algorithm>
#include <chrono>
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

static double wall_time_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static bool profile_enabled() {
    const char* p = getenv("SA3_PROFILE");
    return p && strcmp(p, "0") != 0;
}

static void profile_log(bool enabled, const char* label, double seconds) {
    if (enabled) fprintf(stderr, "[sa3-profile] %-18s %8.3f ms\n", label, seconds * 1000.0);
}

static std::vector<float> tensor_to_host(const sa3::GgufModel& M, const std::string& name) {
    ggml_tensor* t = M.get(name);
    std::vector<float> b(ggml_nelements(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size() * sizeof(float));
    return b;
}

// SAME-L sliding-window bias upload; no-op when the mask tensor is unused (SAME-S block-diagonal).
static void set_swa_bias(ggml_tensor* mt, const sa3::SameConfig& sc, int64_t N) {
    if (!mt->buffer || sc.chunk) return;
    if (mt->type == GGML_TYPE_F16) {
        const bool compact = mt->ne[0] == 3*sc.sub_chunk && mt->ne[1] == sc.sub_chunk;
        std::vector<ggml_fp16_t> bias = compact ? sa3::build_swa_bias_f16(sc, N) : sa3::build_swa_full_bias_f16(sc, N);
        ggml_backend_tensor_set(mt, bias.data(), 0, bias.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<float> bias = sa3::build_swa_bias(sc, N);
        ggml_backend_tensor_set(mt, bias.data(), 0, bias.size() * sizeof(float));
    }
}

int main(int argc, char** argv) {
    const bool prof = profile_enabled();
    const double t_total0 = wall_time_s();
    const char* tok_p = nullptr; const char* t5_p = nullptr; const char* dit_p = nullptr; const char* same_p = nullptr;
    const char* cond_p = nullptr;            // per-variant conditioner sidecar gguf (optional; falls back to --t5 if bundled)
    std::string prompt = "Upbeat funk groove with slap bass, bright horns, tight drums";
    const char* wav_p = "song.wav";
    const char* init_p = nullptr;            // audio2audio / inpaint: source WAV (encoded to z_init)
    float init_noise_level = 0.85f;          // sigma_max for audio2audio (1.0 == text2music)
    float inpaint_start = -1.0f, inpaint_end = -1.0f;   // inpaint: regenerate this [start,end] sec region
    std::vector<std::pair<std::string,float>> lora_specs;   // (gguf, strength) applied in flag order
    bool keep_models = false;        // --keep-models: don't free TE/DIT early (keep all resident)
    int decode_chunk_size = 0, decode_overlap = 32;     // outer SAME-L decode tiling; 0 = monolithic
    int frames = 128, steps = 8; uint64_t seed = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--tok")    && i+1 < argc) tok_p = argv[++i];
        else if (!strcmp(argv[i], "--t5")     && i+1 < argc) t5_p = argv[++i];
        else if (!strcmp(argv[i], "--dit")    && i+1 < argc) dit_p = argv[++i];
        else if (!strcmp(argv[i], "--same")   && i+1 < argc) same_p = argv[++i];
        else if (!strcmp(argv[i], "--cond")   && i+1 < argc) cond_p = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")  && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")   && i+1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) wav_p = argv[++i];
        else if (!strcmp(argv[i], "--init")   && i+1 < argc) init_p = argv[++i];
        else if (!strcmp(argv[i], "--init-noise-level") && i+1 < argc) init_noise_level = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--inpaint-start") && i+1 < argc) inpaint_start = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--inpaint-end")   && i+1 < argc) inpaint_end   = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lora")   && i+1 < argc) lora_specs.push_back({argv[++i], 1.0f});
        else if (!strcmp(argv[i], "--lora-strength") && i+1 < argc) {   // sets the most recent --lora
            if (lora_specs.empty()) { fprintf(stderr, "--lora-strength must follow a --lora\n"); return 1; }
            lora_specs.back().second = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--chunked-decode")) decode_chunk_size = 128;
        else if (!strcmp(argv[i], "--decode-chunk-size") && i+1 < argc) decode_chunk_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-overlap") && i+1 < argc) decode_overlap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keep-models")) keep_models = true;   // disable progressive VRAM frees
    }
    const bool inpaint = (inpaint_start >= 0.0f || inpaint_end >= 0.0f);   // inpaint mode (needs --init source)
    if (!tok_p || !t5_p || !dit_p || !same_p) {
        fprintf(stderr, "usage: sa3-generate --tok <f> --t5 <f> --dit <f> --same <f> --prompt \"...\" --frames N --steps N --seed S --out song.wav\n");
        return 1;
    }

    ggml_backend_t shared_backend = sa3::make_backend();

    double t0 = wall_time_s();
    sa3::Tokenizer tok = sa3::Tokenizer::load(tok_p);
    profile_log(prof, "load_tokenizer", wall_time_s() - t0);
    t0 = wall_time_s();
    sa3::GgufModel TE = sa3::load_gguf(t5_p, shared_backend);
    profile_log(prof, "load_t5", wall_time_s() - t0);
    t0 = wall_time_s();
    sa3::GgufModel DIT = sa3::load_gguf(dit_p, shared_backend);
    profile_log(prof, "load_dit", wall_time_s() - t0);
    t0 = wall_time_s();
    sa3::GgufModel AE = sa3::load_gguf(same_p, shared_backend);
    profile_log(prof, "load_same", wall_time_s() - t0);
    // Per-variant conditioner: separate sidecar gguf when given, else read from the (bundled) encoder.
    sa3::GgufModel* cond_model = cond_p ? new sa3::GgufModel(sa3::load_gguf(cond_p, shared_backend)) : nullptr;
    const sa3::GgufModel& CD = cond_model ? *cond_model : TE;
    const sa3::T5GemmaConfig tc = sa3::T5GemmaConfig::from(TE);
    const sa3::DitConfig     dc = sa3::DitConfig::from(DIT);
    const sa3::SameConfig    sc = sa3::SameConfig::from(AE);
    const int same_l_flash_mode = sc.chunk ? 0 : sa3::nn::same_flash_attn_mode();
    const bool same_l_flash_attn = same_l_flash_mode != 0;
    if (decode_chunk_size < 0) { fprintf(stderr, "--decode-chunk-size must be >= 0\n"); return 1; }
    if (decode_overlap < 0) { fprintf(stderr, "--decode-overlap must be >= 0\n"); return 1; }
    if (decode_chunk_size > 0 && decode_overlap >= decode_chunk_size) {
        fprintf(stderr, "--decode-overlap must be smaller than --decode-chunk-size\n");
        return 1;
    }
    const int ds = sc.patch_size * sc.output_seg;          // downsampling ratio (4096 samples/frame)

    // ---------- LoRA/DoRA adapters: recompute W_eff in weight space (chained in flag order) ----------
    std::vector<sa3::LoraAdapter> adapters;
    sa3::LoraStack lstack;
    for (auto& ls : lora_specs) adapters.push_back(sa3::load_lora(ls.first.c_str(), ls.second, DIT.backend));
    if (!adapters.empty()) {
        lstack = sa3::apply_loras(DIT, adapters);
        printf("lora: applied %zu adapter(s) -> %zu overridden weights:\n", adapters.size(), DIT.overrides.size());
        for (size_t i = 0; i < adapters.size(); i++)
            printf("  [%zu] %s  type=%s strength=%.2f\n", i, lora_specs[i].first.c_str(),
                   adapters[i].type.c_str(), adapters[i].strength);
    }

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
    if (same_l_flash_attn) {
        const int64_t n_same = (int64_t)T * sc.sub_chunk;
        const int64_t mask_elems = same_l_flash_mode == 2 ? (int64_t)3 * sc.sub_chunk * n_same : n_same * n_same;
        const double mask_mib = (double)mask_elems * sizeof(ggml_fp16_t) / (1024.0 * 1024.0);
        fprintf(stderr, "[sa3] SAME-L %s flash attention enabled: F16 band mask %.1f MiB\n",
                same_l_flash_mode == 2 ? "local" : "full", mask_mib);
    }
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
        const double t_t5_total = wall_time_s();
        double tp = wall_time_s();
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
        profile_log(prof, "t5_build", wall_time_s() - tp);
        tp = wall_time_s();
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(TE.backend));
        ggml_gallocr_alloc_graph(alloc, gf);
        profile_log(prof, "t5_alloc", wall_time_s() - tp);
        tp = wall_time_s();
        ggml_backend_tensor_set(ids_t, ids.data(), 0, max_len*sizeof(int32_t));
        std::vector<int32_t> pos(max_len); for (int i = 0; i < max_len; i++) pos[i] = i;
        ggml_backend_tensor_set(pos_t, pos.data(), 0, max_len*sizeof(int32_t));
        std::vector<float> mb((size_t)max_len*max_len);
        for (int q = 0; q < max_len; q++) for (int k = 0; k < max_len; k++)
            mb[(size_t)q*max_len+k] = attn[k] ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(mask_t, mb.data(), 0, mb.size()*sizeof(float));
        profile_log(prof, "t5_upload", wall_time_s() - tp);
        tp = wall_time_s();
        ggml_backend_graph_compute(TE.backend, gf);
        profile_log(prof, "t5_compute", wall_time_s() - tp);
        tp = wall_time_s();
        hidden.resize((size_t)cond_dim*max_len);
        ggml_backend_tensor_get(h, hidden.data(), 0, hidden.size()*sizeof(float));
        profile_log(prof, "t5_download", wall_time_s() - tp);
        ggml_gallocr_free(alloc); ggml_free(ctx);
        profile_log(prof, "t5_total", wall_time_s() - t_t5_total);
    }

    // ---------- conditioning assembly (host) ----------
    t0 = wall_time_s();
    // learned padding: replace padded token columns with the padding embedding
    std::vector<float> pad_emb = tensor_to_host(CD, "te.padding_embedding");   // [cond_dim]
    for (int p = 0; p < max_len; p++)
        if (!attn[p]) memcpy(&hidden[(size_t)p*cond_dim], pad_emb.data(), cond_dim*sizeof(float));
    // seconds_total NumberConditioner: expo(clamp(secs)/range) -> Linear(secs_dim->cond_dim)
    const float secs = (float)T * (sc.patch_size * sc.output_seg) / 44100.0f;
    const float smin = CD.f32("t5g.secs_min"), smax = CD.f32("t5g.secs_max");
    const int sdim = (int)CD.u32("t5g.secs_dim");
    float sclamp = secs < smin ? smin : (secs > smax ? smax : secs);
    float snorm = (sclamp - smin) / (smax - smin);
    std::vector<float> ef; expo_features(snorm, ef, sdim, 0.5f, 10000.0f);
    std::vector<float> sw = tensor_to_host(CD, "te.secs.weight");   // ggml [sdim, cond_dim]
    std::vector<float> sb = tensor_to_host(CD, "te.secs.bias");     // [cond_dim]
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
    profile_log(prof, "conditioning", wall_time_s() - t0);

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
        ggml_tensor* mask_a = sc.chunk ? ggml_new_tensor_2d(enctx, GGML_TYPE_F32, Nenc, Nenc)
                            : same_l_flash_mode == 1 ? ggml_new_tensor_4d(enctx, GGML_TYPE_F16, Nenc, Nenc, 1, 1)
                            : same_l_flash_mode == 2 ? ggml_new_tensor_3d(enctx, GGML_TYPE_F16, 3*sc.sub_chunk, sc.sub_chunk, Nenc/sc.sub_chunk)
                                                     : ggml_new_tensor_3d(enctx, GGML_TYPE_F32, 3*sc.sub_chunk, sc.sub_chunk, Nenc/sc.sub_chunk);
        ggml_set_input(a_in); ggml_set_input(pos_a); ggml_set_input(mask_a);
        ggml_tensor *pos_a2 = nullptr, *mask_a2 = nullptr;
        if (sc.chunk) {                                       // SAME-S shifted half: pos_a2 used, mask_a2 unused
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
        set_pos(pos_a, Nenc); set_swa_bias(mask_a, sc, Nenc);
        if (sc.chunk) set_pos(pos_a2, Nenc2);                 // mask_a2 unused (block-diagonal)
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

    if (!keep_models) TE.free();   // free T5Gemma before sampling (skip with --keep-models)

    // ---------- DiT ping-pong ----------
    const int S = dc.mem_tokens + T;
    const double t_dit_total = wall_time_s();
    double tp_dit = wall_time_s();
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
    profile_log(prof, "dit_build", wall_time_s() - tp_dit);
    tp_dit = wall_time_s();
    ggml_gallocr_t alloc_dit = ggml_gallocr_new(ggml_backend_get_default_buffer_type(DIT.backend));
    ggml_gallocr_alloc_graph(alloc_dit, gf_dit);
    profile_log(prof, "dit_alloc", wall_time_s() - tp_dit);
    std::vector<int32_t> posb(S); for (int i = 0; i < S; i++) posb[i] = i;
    const float one = 1.0f;
    std::vector<float> tf, vbuf(N);
    double dit_upload = 0.0, dit_compute = 0.0, dit_download_update = 0.0;
    for (int i = 0; i < steps; i++) {
        double ts = wall_time_s();
        ggml_backend_tensor_set(cross, crossb.data(), 0, crossb.size()*sizeof(float));  // re-set each step (gallocr recycle)
        ggml_backend_tensor_set(glob,  globb.data(),  0, globb.size()*sizeof(float));
        ggml_backend_tensor_set(pos_d, posb.data(),   0, posb.size()*sizeof(int32_t));
        ggml_backend_tensor_set(ones,  &one, 0, sizeof(float));
        if (local) ggml_backend_tensor_set(local, localb.data(), 0, localb.size()*sizeof(float));  // re-set (gallocr recycle)
        ggml_backend_tensor_set(x_in,  host_x.data(), 0, N*sizeof(float));
        expo_features(sigmas[i], tf, dc.time_dim, dc.time_min_freq, dc.time_max_freq);
        ggml_backend_tensor_set(tfeat, tf.data(), 0, tf.size()*sizeof(float));
        dit_upload += wall_time_s() - ts;
        ts = wall_time_s();
        ggml_backend_graph_compute(DIT.backend, gf_dit);
        dit_compute += wall_time_s() - ts;
        ts = wall_time_s();
        ggml_backend_tensor_get(vel, vbuf.data(), 0, N*sizeof(float));
        const float tcur = sigmas[i], tnext = sigmas[i+1];
        for (int j = 0; j < N; j++) {
            float denoised = host_x[j] - tcur * vbuf[j];
            host_x[j] = (1.0f - tnext) * denoised + tnext * stepnoise[(size_t)i*N + j];
        }
        dit_download_update += wall_time_s() - ts;
        printf("  step %d/%d  t=%.4f\n", i+1, steps, tcur);
    }
    profile_log(prof, "dit_upload", dit_upload);
    profile_log(prof, "dit_compute", dit_compute);
    profile_log(prof, "dit_get_update", dit_download_update);
    profile_log(prof, "dit_total", wall_time_s() - t_dit_total);
    ggml_gallocr_free(alloc_dit); ggml_free(dctx);
    if (!keep_models) DIT.free();   // free the DiT before the SAME decode (skip with --keep-models)

    // ---------- decode -> WAV ----------
    const double t_dec_total = wall_time_s();
    double dec_build = 0.0, dec_alloc = 0.0, dec_upload = 0.0, dec_compute = 0.0, dec_download = 0.0;

    struct DecodeGraph {
        int T = 0, Ndec = 0, N2 = 0, n_samp = 0, n_ch = 0;
        ggml_context* ctx = nullptr;
        ggml_tensor* z = nullptr;
        ggml_tensor* pos = nullptr;
        ggml_tensor* mask = nullptr;
        ggml_tensor* pos2 = nullptr;
        ggml_tensor* mask2 = nullptr;
        ggml_tensor* audio = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t alloc = nullptr;
    };

    auto set_pos = [&](ggml_tensor* p, int n){
        std::vector<int32_t> b(n);
        for (int i = 0; i < n; i++) b[i] = i;
        ggml_backend_tensor_set(p, b.data(), 0, n*sizeof(int32_t));
    };

    auto build_decode_graph = [&](int run_T) {
        DecodeGraph dg;
        dg.T = run_T;
        dg.Ndec = run_T * sc.sub_chunk;
        dg.N2 = sc.chunk ? dg.Ndec + 2*sc.shift : 0;       // SAME-S needs a shifted 2nd mask
        double ts = wall_time_s();
        ggml_init_params eip = { (size_t)512*1024*1024, nullptr, true };
        dg.ctx = ggml_init(eip);
        dg.z = ggml_new_tensor_2d(dg.ctx, GGML_TYPE_F32, sc.latent, run_T);
        dg.pos = ggml_new_tensor_1d(dg.ctx, GGML_TYPE_I32, dg.Ndec);
        dg.mask = sc.chunk ? ggml_new_tensor_2d(dg.ctx, GGML_TYPE_F32, dg.Ndec, dg.Ndec)
                : same_l_flash_mode == 1 ? ggml_new_tensor_4d(dg.ctx, GGML_TYPE_F16, dg.Ndec, dg.Ndec, 1, 1)
                : same_l_flash_mode == 2 ? ggml_new_tensor_3d(dg.ctx, GGML_TYPE_F16, 3*sc.sub_chunk, sc.sub_chunk, dg.Ndec/sc.sub_chunk)
                                         : ggml_new_tensor_3d(dg.ctx, GGML_TYPE_F32, 3*sc.sub_chunk, sc.sub_chunk, dg.Ndec/sc.sub_chunk);
        ggml_set_input(dg.z); ggml_set_input(dg.pos); ggml_set_input(dg.mask);
        if (sc.chunk) {                                      // SAME-S shifted half: pos2 used, mask2 unused
            dg.pos2  = ggml_new_tensor_1d(dg.ctx, GGML_TYPE_I32, dg.N2);
            dg.mask2 = ggml_new_tensor_2d(dg.ctx, GGML_TYPE_F32, dg.N2, dg.N2);
            ggml_set_input(dg.pos2); ggml_set_input(dg.mask2);
        }
        dg.audio = ggml_cont(dg.ctx, sa3::same_decode(dg.ctx, AE, dg.z, sc, run_T, dg.pos, dg.mask, dg.pos2, dg.mask2).audio);
        ggml_set_output(dg.audio);
        dg.graph = ggml_new_graph_custom(dg.ctx, 32768, false);
        ggml_build_forward_expand(dg.graph, dg.audio);
        dec_build += wall_time_s() - ts;
        ts = wall_time_s();
        dg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(AE.backend));
        ggml_gallocr_alloc_graph(dg.alloc, dg.graph);
        dec_alloc += wall_time_s() - ts;
        dg.n_samp = (int)dg.audio->ne[0];
        dg.n_ch = (int)dg.audio->ne[1];
        return dg;
    };

    auto free_decode_graph = [&](DecodeGraph& dg) {
        ggml_gallocr_free(dg.alloc);
        ggml_free(dg.ctx);
        dg = DecodeGraph{};
    };

    auto run_decode_graph = [&](DecodeGraph& dg, const float* z_src, std::vector<float>& out) {
        double ts = wall_time_s();
        ggml_backend_tensor_set(dg.z, z_src, 0, (size_t)sc.latent*dg.T*sizeof(float));
        set_pos(dg.pos, dg.Ndec);
        set_swa_bias(dg.mask, sc, dg.Ndec);
        if (sc.chunk) set_pos(dg.pos2, dg.N2);               // mask2 unused (block-diagonal)
        dec_upload += wall_time_s() - ts;
        ts = wall_time_s();
        ggml_backend_graph_compute(AE.backend, dg.graph);
        dec_compute += wall_time_s() - ts;
        ts = wall_time_s();
        out.resize((size_t)dg.n_samp*dg.n_ch);
        ggml_backend_tensor_get(dg.audio, out.data(), 0, out.size()*sizeof(float));
        dec_download += wall_time_s() - ts;
    };

    const bool can_chunk_decode = decode_chunk_size > 0 && !sc.chunk && T >= decode_chunk_size;
    if (decode_chunk_size > 0 && sc.chunk)
        fprintf(stderr, "warning: outer --chunked-decode is only enabled for SAME-L; using monolithic SAME-S decode\n");

    const int n_ch = sc.out_channels / sc.patch_size;
    const int n_samp = T * sc.patch_size * sc.output_seg;
    std::vector<float> ab((size_t)n_samp*n_ch, 0.0f);
    if (!can_chunk_decode) {
        DecodeGraph dg = build_decode_graph(T);
        run_decode_graph(dg, host_x.data(), ab);
        free_decode_graph(dg);
    } else {
        const int hop = decode_chunk_size - decode_overlap;
        std::vector<int> starts;
        for (int s = 0; s <= T - decode_chunk_size; s += hop) starts.push_back(s);
        const int final_start = T - decode_chunk_size;
        if (starts.empty() || starts.back() != final_start) starts.push_back(final_start);
        fprintf(stderr, "[sa3] chunked SAME-L decode: %zu chunks, size=%d overlap=%d\n",
                starts.size(), decode_chunk_size, decode_overlap);

        DecodeGraph dg = build_decode_graph(decode_chunk_size);
        std::vector<float> zchunk((size_t)sc.latent * decode_chunk_size);
        std::vector<float> chunk_audio;
        const int chunk_samples = decode_chunk_size * ds;
        const int half_overlap_samples = (decode_overlap / 2) * ds;
        int cursor = 0;
        for (size_t i = 0; i < starts.size(); i++) {
            const int st = starts[i];
            for (int t = 0; t < decode_chunk_size; t++)
                memcpy(&zchunk[(size_t)t*sc.latent], &host_x[(size_t)(st + t)*sc.latent], sc.latent*sizeof(float));
            run_decode_graph(dg, zchunk.data(), chunk_audio);

            const bool first = i == 0, last = i + 1 == starts.size();
            const int out_start = last ? n_samp - chunk_samples : st * ds;
            const int left = first ? 0 : half_overlap_samples;
            const int right = last ? chunk_samples : chunk_samples - half_overlap_samples;
            const int target_start = out_start + left;
            int target_end = out_start + right;
            const int next_start = !last
                ? ((i + 2 == starts.size() ? n_samp - chunk_samples : starts[i + 1] * ds)
                   + half_overlap_samples)
                : target_end;
            target_end = std::min(target_end, next_start);
            const int clipped_start = std::max(target_start, cursor);
            if (target_end > clipped_start) {
                const int copy_left = left + (clipped_start - target_start);
                const int copy_count = target_end - clipped_start;
                for (int c = 0; c < n_ch; c++)
                    memcpy(&ab[(size_t)c*n_samp + clipped_start],
                           &chunk_audio[(size_t)c*chunk_samples + copy_left],
                           (size_t)copy_count*sizeof(float));
                cursor = target_end;
            }
        }
        free_decode_graph(dg);
    }

    profile_log(prof, "dec_build", dec_build);
    profile_log(prof, "dec_alloc", dec_alloc);
    profile_log(prof, "dec_upload", dec_upload);
    profile_log(prof, "dec_compute", dec_compute);
    profile_log(prof, "dec_download", dec_download);
    double tp_dec = wall_time_s();
    sa3::write_wav_planar(wav_p, ab.data(), n_samp, n_ch, 44100);
    profile_log(prof, "write_wav", wall_time_s() - tp_dec);
    profile_log(prof, "dec_total", wall_time_s() - t_dec_total);
    printf("wrote %s  (%.2fs, seed %llu)\n", wav_p, (float)n_samp/44100.0f, (unsigned long long)seed);

    if (keep_models) { TE.free(); DIT.free(); }   // freed early otherwise
    AE.free();
    if (cond_model) { cond_model->free(); delete cond_model; }
    ggml_backend_free(shared_backend);
    profile_log(prof, "total", wall_time_s() - t_total0);
    return 0;
}
