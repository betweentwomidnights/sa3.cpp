// sa3-textmusic: end-to-end ping-pong generation + decode -> WAV (CPU/f32).
// Replays the exact conditioning/schedule/noise dumped by tools/dump_text2music.py
// so the result is directly comparable (latent cossim + ear test) to the PyTorch clip.
#include "gguf_model.h"
#include "dit.h"
#include "same_ae.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string& path, size_t n) {
    std::vector<float> b(n);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path.c_str()); exit(1); }
    if (fread(b.data(), sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    fclose(f);
    return b;
}

// ExpoFourierFeatures(dim, min_freq, max_freq): [cos(t*f*2pi), sin(t*f*2pi)], f log-spaced.
static void expo_features(float t, std::vector<float>& out, int dim, float min_f, float max_f) {
    const int half = dim / 2;
    const float lmin = logf(min_f), lmax = logf(max_f), TWO_PI = 6.28318530717958648f;
    for (int i = 0; i < half; i++) {
        float ramp = half > 1 ? (float)i / (half - 1) : 0.0f;
        float freq = expf(ramp * (lmax - lmin) + lmin);
        float arg = t * freq * TWO_PI;
        out[i] = cosf(arg); out[half + i] = sinf(arg);
    }
}

static int run(int argc, char** argv) {
    const char* dit_path = nullptr; const char* same_path = nullptr;
    const char* dir = "refdata"; const char* outdir = "cppout"; const char* wav_path = "cppout/tm_ggml.wav";
    int frames = 64, steps = 8, ctx_len = 257;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--dit")    && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--same")   && i+1 < argc) same_path = argv[++i];
        else if (!strcmp(argv[i], "--in")     && i+1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")  && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--wav")    && i+1 < argc) wav_path = argv[++i];
    }
    if (!dit_path || !same_path) { fprintf(stderr, "usage: sa3-textmusic --dit <f> --same <f> --in <dir> --frames N --steps N --wav <out.wav>\n"); return 1; }

    sa3::GgufModel DIT = sa3::load_gguf(dit_path);
    sa3::GgufModel AE  = sa3::load_gguf(same_path);
    const sa3::DitConfig  dc = sa3::DitConfig::from(DIT);
    const sa3::SameConfig sc = sa3::SameConfig::from(AE);
    const int T = frames, S = dc.mem_tokens + T, N = (int)T * dc.io;

    // ---------- DiT graph (built once, recomputed per step) ----------
    ggml_init_params dip = { (size_t)512*1024*1024, nullptr, true };
    ggml_context* dctx = ggml_init(dip);
    ggml_tensor* x_in  = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.io, T);
    ggml_tensor* tfeat = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, dc.time_dim);
    ggml_tensor* cross = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.cond_dim, ctx_len);
    ggml_tensor* glob  = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, dc.cond_dim);
    ggml_tensor* pos_d = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, S);
    ggml_tensor* ones  = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, 1);
    for (ggml_tensor* t : {x_in, tfeat, cross, glob, pos_d, ones}) ggml_set_input(t);
    // inpaint: replay tm_local.f32 if the DiT has local-cond weights and the ref exists
    std::string local_path = std::string(dir) + "/tm_local.f32";
    ggml_tensor* local = nullptr;
    if (dc.local_dim > 0) { FILE* lf = fopen(local_path.c_str(), "rb"); if (lf) { fclose(lf);
        local = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.local_dim, T); ggml_set_input(local); } }
    ggml_tensor* vel = ggml_cont(dctx, sa3::dit_forward(dctx, DIT, x_in, tfeat, cross, glob, pos_d, ones, dc, local));
    ggml_set_output(vel);
    ggml_cgraph* gf_dit = ggml_new_graph_custom(dctx, 32768, false);
    ggml_build_forward_expand(gf_dit, vel);
    ggml_gallocr_t alloc_dit = ggml_gallocr_new(ggml_backend_get_default_buffer_type(DIT.backend));
    ggml_gallocr_alloc_graph(alloc_dit, gf_dit);

    // DiT inputs (re-set every step: gallocr may recycle input buffers across recompute)
    std::vector<float> crossb = read_f32(std::string(dir)+"/tm_cross.f32",  (size_t)dc.cond_dim*ctx_len);
    std::vector<float> globb  = read_f32(std::string(dir)+"/tm_global.f32", dc.cond_dim);
    std::vector<float> localb = local ? read_f32(local_path, (size_t)dc.local_dim*T) : std::vector<float>{};
    std::vector<int32_t> posb(S); for (int i = 0; i < S; i++) posb[i] = i;
    const float one = 1.0f;
    auto set_static = [&]{
        ggml_backend_tensor_set(cross, crossb.data(), 0, crossb.size()*sizeof(float));
        ggml_backend_tensor_set(glob,  globb.data(),  0, globb.size()*sizeof(float));
        ggml_backend_tensor_set(pos_d, posb.data(),   0, posb.size()*sizeof(int32_t));
        ggml_backend_tensor_set(ones,  &one, 0, sizeof(float));
        if (local) ggml_backend_tensor_set(local, localb.data(), 0, localb.size()*sizeof(float));
    };

    // ---------- ping-pong loop (host mixing between DiT calls) ----------
    std::vector<float> sigmas    = read_f32(std::string(dir)+"/tm_sigmas.f32", steps+1);
    std::vector<float> host_x    = read_f32(std::string(dir)+"/tm_noise0.f32", N);
    std::vector<float> stepnoise = read_f32(std::string(dir)+"/tm_stepnoise.f32", (size_t)steps*N);
    std::vector<float> tf(dc.time_dim), vbuf(N);
    for (int i = 0; i < steps; i++) {
        set_static();   // re-set every step: gallocr recycles input buffers across recompute
        ggml_backend_tensor_set(x_in, host_x.data(), 0, N*sizeof(float));
        expo_features(sigmas[i], tf, dc.time_dim, dc.time_min_freq, dc.time_max_freq);
        ggml_backend_tensor_set(tfeat, tf.data(), 0, tf.size()*sizeof(float));
        ggml_backend_graph_compute(DIT.backend, gf_dit);
        ggml_backend_tensor_get(vel, vbuf.data(), 0, N*sizeof(float));
        const float tc = sigmas[i], tn = sigmas[i+1];
        for (int j = 0; j < N; j++) {
            float denoised = host_x[j] - tc * vbuf[j];
            host_x[j] = (1.0f - tn) * denoised + tn * stepnoise[(size_t)i*N + j];
        }
        printf("  step %d/%d  t=%.4f -> %.4f\n", i+1, steps, tc, tn);
    }
    { FILE* f = fopen((std::string(outdir)+"/tm_latent.f32").c_str(), "wb");
      fwrite(host_x.data(), sizeof(float), host_x.size(), f); fclose(f); }

    // ---------- decode latent -> audio ----------
    const int Ndec = T * sc.sub_chunk;
    ggml_init_params eip = { (size_t)512*1024*1024, nullptr, true };
    ggml_context* ectx = ggml_init(eip);
    ggml_tensor* z     = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, sc.latent, T);
    ggml_tensor* pos_e = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, Ndec);
    ggml_tensor* mask_e= ggml_new_tensor_3d(ectx, GGML_TYPE_F32, 3*sc.sub_chunk, sc.sub_chunk, Ndec/sc.sub_chunk); // SWA bias (SAME-L)
    for (ggml_tensor* t : {z, pos_e, mask_e}) ggml_set_input(t);
    ggml_tensor* audio = ggml_cont(ectx, sa3::same_decode(ectx, AE, z, sc, T, pos_e, mask_e).audio);
    ggml_set_output(audio);
    ggml_cgraph* gf_dec = ggml_new_graph_custom(ectx, 32768, false);
    ggml_build_forward_expand(gf_dec, audio);
    ggml_gallocr_t alloc_dec = ggml_gallocr_new(ggml_backend_get_default_buffer_type(AE.backend));
    ggml_gallocr_alloc_graph(alloc_dec, gf_dec);

    ggml_backend_tensor_set(z, host_x.data(), 0, N*sizeof(float));   // latent
    std::vector<int32_t> pe(Ndec); for (int i = 0; i < Ndec; i++) pe[i] = i;
    ggml_backend_tensor_set(pos_e, pe.data(), 0, pe.size()*sizeof(int32_t));
    std::vector<float> me = sa3::build_swa_bias(sc, Ndec);
    ggml_backend_tensor_set(mask_e, me.data(), 0, me.size()*sizeof(float));

    ggml_backend_graph_compute(AE.backend, gf_dec);

    const int n_samples = (int)audio->ne[0], n_ch = (int)audio->ne[1];
    std::vector<float> ab((size_t)n_samples*n_ch);
    ggml_backend_tensor_get(audio, ab.data(), 0, ab.size()*sizeof(float));
    sa3::write_wav_planar(wav_path, ab.data(), n_samples, n_ch, 44100);
    printf("wrote %s  (%d samples x %d ch, %.2fs)\n", wav_path, n_samples, n_ch, (float)n_samples/44100.0f);

    ggml_gallocr_free(alloc_dit); ggml_gallocr_free(alloc_dec);
    ggml_free(dctx); ggml_free(ectx); DIT.free(); AE.free();
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
