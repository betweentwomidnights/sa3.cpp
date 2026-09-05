// saos-generate -- experimental classic Stable Audio Open Small GGML runner.
#include "gguf_model.h"
#include "rng.h"
#include "sampling.h"
#include "sat/conditioning.h"
#include "sat/dit.h"
#include "sat/oobleck.h"
#include "sat/t5.h"
#include "sat/tokenizer.h"
#include "wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <numeric>
#include <string>
#include <vector>

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static std::vector<float> read_all_f32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    if (fseek(f, 0, SEEK_END) || ftell(f) < 0) { fclose(f); throw std::runtime_error("cannot size " + path); }
    const long bytes = ftell(f);
    rewind(f);
    if (bytes % (long)sizeof(float)) { fclose(f); throw std::runtime_error("invalid float buffer " + path); }
    std::vector<float> out((size_t)bytes / sizeof(float));
    const size_t got = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    if (got != out.size()) throw std::runtime_error("short read " + path);
    return out;
}

static void write_f32(const std::string& path, const std::vector<float>& data) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    if (fwrite(data.data(), sizeof(float), data.size(), f) != data.size()) {
        fclose(f); throw std::runtime_error("short write " + path);
    }
    fclose(f);
}

static std::vector<float> encode_t5_prompt(const char* path, const std::string& prompt) {
    sa3::sat::UnigramTokenizer tokenizer = sa3::sat::UnigramTokenizer::load(path);
    sa3::GgufModel T5 = sa3::load_gguf(path);
    const sa3::sat::T5EncoderConfig c = sa3::sat::T5EncoderConfig::from(T5);
    const int seq = (int)T5.u32("sat.t5.max_length");
    std::vector<int32_t> ids = tokenizer.encode(prompt, seq);
    std::vector<int32_t> attention((size_t)seq, 0);
    const int valid = (int)ids.size();
    ids.resize((size_t)seq, tokenizer.pad_id);
    std::fill_n(attention.begin(), valid, 1);

    ggml_init_params ip = {(size_t)128 * 1024 * 1024, nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("failed to create T5 graph context");
    ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_tensor* mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_tensor* rel_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)seq * seq);
    for (ggml_tensor* t : {ids_t, mask_t, rel_t}) ggml_set_input(t);
    ggml_tensor* hidden = ggml_cont(ctx, sa3::sat::t5_encode(ctx, T5, ids_t, mask_t, rel_t, c));
    ggml_set_output(hidden);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, hidden);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(T5.backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, graph))
        throw std::runtime_error("failed to allocate T5 graph");
    std::vector<float> mask((size_t)seq * seq);
    for (int q = 0; q < seq; ++q) for (int k = 0; k < seq; ++k)
        mask[(size_t)q * seq + k] = attention[(size_t)k] ? 0.0f : -INFINITY;
    const std::vector<int32_t> rel = sa3::sat::t5_relative_position_buckets(
        seq, c.relative_buckets, c.relative_max_distance);
    ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_tensor_set(rel_t, rel.data(), 0, rel.size() * sizeof(int32_t));
    std::string error;
    if (!sa3::graph_compute_checked(T5.backend, graph, "SAOS T5", error))
        throw std::runtime_error(error);
    std::vector<float> out((size_t)c.dim * seq);
    ggml_backend_tensor_get(hidden, out.data(), 0, out.size() * sizeof(float));
    for (int token = valid; token < seq; ++token)
        std::fill_n(out.data() + (size_t)token * c.dim, c.dim, 0.0f);
    printf("prompt tokens: %d/%d\n", valid, seq);
    ggml_gallocr_free(alloc); ggml_free(ctx); T5.free();
    return out;
}

static int run(int argc, char** argv) {
    const char* dit_path = nullptr;
    const char* ae_path = nullptr;
    const char* cond_prefix = nullptr;
    const char* t5_path = nullptr;
    const char* prompt = nullptr;
    const char* dump_cond_prefix = nullptr;
    const char* wav_path = "saos-ggml.wav";
    const char* latent_path = nullptr;
    const char* initial_path = nullptr;
    const char* step_noise_path = nullptr;
    int frames = 256, steps = 8, output_samples = 485100;
    float seconds = 11.0f;
    uint64_t seed = 1234;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dit") && i + 1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--ae") && i + 1 < argc) ae_path = argv[++i];
        else if (!strcmp(argv[i], "--conditioning") && i + 1 < argc) cond_prefix = argv[++i];
        else if (!strcmp(argv[i], "--t5") && i + 1 < argc) t5_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--dump-conditioning") && i + 1 < argc) dump_cond_prefix = argv[++i];
        else if (!strcmp(argv[i], "--wav") && i + 1 < argc) wav_path = argv[++i];
        else if (!strcmp(argv[i], "--latent") && i + 1 < argc) latent_path = argv[++i];
        else if (!strcmp(argv[i], "--initial-latent") && i + 1 < argc) initial_path = argv[++i];
        else if (!strcmp(argv[i], "--step-noise") && i + 1 < argc) step_noise_path = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc) output_samples = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: saos-generate --dit model.gguf --ae decoder.gguf "
                   "(--t5 t5.gguf --prompt TEXT [--seconds 11] | --conditioning PREFIX) "
                   "[--frames 256] [--steps 8] [--samples 485100] [--seed 1234] [--wav out.wav] "
                   "[--initial-latent x.f32 --step-noise noise.f32]\n");
            return 0;
        }
    }
    const bool native_conditioning = t5_path && prompt;
    if (!dit_path || !ae_path || (!cond_prefix && !native_conditioning) ||
        (cond_prefix && native_conditioning) || frames < 1 || steps < 1)
        throw std::runtime_error("missing/invalid arguments; run --help");

    const double total_start = now_s();
    double conditioning_s = 0.0;
    std::vector<float> prompt_hidden;
    if (native_conditioning) {
        double tc = now_s();
        prompt_hidden = encode_t5_prompt(t5_path, prompt);
        conditioning_s = now_s() - tc;
    }
    double t0 = now_s();
    sa3::GgufModel DIT = sa3::load_gguf(dit_path);
    const double dit_load_s = now_s() - t0;
    const sa3::sat::DitSpec dc = sa3::sat::dit_spec_from(DIT);
    if (DIT.string("sat.diffusion_objective") != "rf_denoiser")
        throw std::runtime_error("this runner requires an rf_denoiser checkpoint");

    std::vector<float> crossb, globb;
    if (cond_prefix) {
        const std::string prefix(cond_prefix);
        crossb = read_all_f32(prefix + ".cross.f32");
        globb = read_all_f32(prefix + ".global.f32");
    } else {
        crossb = std::move(prompt_hidden);
        globb = sa3::sat::seconds_total_condition(DIT, seconds);
        if (crossb.size() != (size_t)64 * dc.cond_token_dim)
            throw std::runtime_error("T5 and DiT conditioning dimensions differ");
        crossb.insert(crossb.end(), globb.begin(), globb.end());
    }
    if (globb.size() != (size_t)dc.global_cond_dim ||
        crossb.size() % (size_t)dc.cond_token_dim)
        throw std::runtime_error("conditioning dimensions do not match the DiT");
    if (dump_cond_prefix) {
        write_f32(std::string(dump_cond_prefix) + ".cross.f32", crossb);
        write_f32(std::string(dump_cond_prefix) + ".global.f32", globb);
    }
    const int cond_tokens = (int)(crossb.size() / (size_t)dc.cond_token_dim);
    const int tokens = frames + 1;
    const size_t latent_n = (size_t)dc.io_channels * frames;

    t0 = now_s();
    ggml_init_params dip = {(size_t)128 * 1024 * 1024, nullptr, true};
    ggml_context* dctx = ggml_init(dip);
    if (!dctx) throw std::runtime_error("failed to create DiT graph context");
    ggml_tensor* x_in = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.io_channels, frames);
    ggml_tensor* time = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, 1);
    ggml_tensor* cross = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.cond_token_dim, cond_tokens);
    ggml_tensor* global = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, dc.global_cond_dim);
    ggml_tensor* pos = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, tokens);
    for (ggml_tensor* t : {x_in, time, cross, global, pos}) ggml_set_input(t);
    ggml_tensor* velocity = ggml_cont(dctx, sa3::sat::classic_dit_forward(
        dctx, DIT, x_in, time, cross, global, pos, dc));
    ggml_set_output(velocity);
    ggml_cgraph* dgraph = ggml_new_graph_custom(dctx, 32768, false);
    ggml_build_forward_expand(dgraph, velocity);
    ggml_gallocr_t dalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(DIT.backend));
    if (!dalloc || !ggml_gallocr_alloc_graph(dalloc, dgraph))
        throw std::runtime_error("failed to allocate DiT graph");
    const double dit_build_s = now_s() - t0;

    std::vector<int32_t> posb((size_t)tokens);
    std::iota(posb.begin(), posb.end(), 0);
    sa3::Rng rng(seed);
    std::vector<float> x(latent_n), noise(latent_n), vel(latent_n);
    if (initial_path) {
        x = read_all_f32(initial_path);
        if (x.size() != latent_n) throw std::runtime_error("initial latent has the wrong size");
    } else {
        rng.fill_normal(x.data(), x.size());
    }
    std::vector<float> supplied_step_noise;
    if (step_noise_path) {
        supplied_step_noise = read_all_f32(step_noise_path);
        if (supplied_step_noise.size() != (size_t)steps * latent_n)
            throw std::runtime_error("step-noise buffer has the wrong size");
    }
    const std::vector<float> schedule = sa3::sampling::make_rf_logsnr_schedule(steps);
    std::vector<double> step_ms;
    step_ms.reserve((size_t)steps);
    std::string compute_error;
    const double denoise_start = now_s();
    for (int i = 0; i < steps; ++i) {
        // Inputs are reset because gallocr is allowed to recycle their storage.
        ggml_backend_tensor_set(x_in, x.data(), 0, x.size() * sizeof(float));
        ggml_backend_tensor_set(time, &schedule[(size_t)i], 0, sizeof(float));
        ggml_backend_tensor_set(cross, crossb.data(), 0, crossb.size() * sizeof(float));
        ggml_backend_tensor_set(global, globb.data(), 0, globb.size() * sizeof(float));
        ggml_backend_tensor_set(pos, posb.data(), 0, posb.size() * sizeof(int32_t));
        const double step_start = now_s();
        if (!sa3::graph_compute_checked(DIT.backend, dgraph, "SAOS DiT", compute_error))
            throw std::runtime_error(compute_error);
        ggml_backend_tensor_get(velocity, vel.data(), 0, vel.size() * sizeof(float));
        const double ms = (now_s() - step_start) * 1000.0;
        step_ms.push_back(ms);
        if (!supplied_step_noise.empty())
            std::copy_n(supplied_step_noise.data() + (size_t)i * latent_n, latent_n, noise.data());
        else
            rng.fill_normal(noise.data(), noise.size());
        sa3::sampling::rf_pingpong_step(x.data(), vel.data(), noise.data(), x.size(),
                                        schedule[(size_t)i], schedule[(size_t)i + 1]);
        printf("step %d/%d  t %.6f -> %.6f  %.1f ms\n", i + 1, steps,
               schedule[(size_t)i], schedule[(size_t)i + 1], ms);
    }
    const double denoise_s = now_s() - denoise_start;
    if (latent_path) write_f32(latent_path, x);

    // Release DiT graph and weights before allocating the long Oobleck decoder graph.
    ggml_gallocr_free(dalloc);
    ggml_free(dctx);
    DIT.free();

    t0 = now_s();
    sa3::GgufModel AE = sa3::load_gguf(ae_path);
    const double ae_load_s = now_s() - t0;
    const sa3::sat::OobleckSpec ac = sa3::sat::oobleck_spec_from(AE);
    if (ac.latent_channels != dc.io_channels)
        throw std::runtime_error("DiT and Oobleck latent widths differ");

    // DiT uses [channel,frame] logical tensors (frame-major storage). Conv1d uses
    // [frame,channel], so transpose to channel-major storage for the decoder input.
    std::vector<float> decoder_z(latent_n);
    for (int f = 0; f < frames; ++f)
        for (int c = 0; c < dc.io_channels; ++c)
            decoder_z[(size_t)c * frames + f] = x[(size_t)f * dc.io_channels + c];

    t0 = now_s();
    ggml_init_params aip = {(size_t)64 * 1024 * 1024, nullptr, true};
    ggml_context* actx = ggml_init(aip);
    if (!actx) throw std::runtime_error("failed to create decoder graph context");
    ggml_tensor* z = ggml_new_tensor_2d(actx, GGML_TYPE_F32, frames, ac.latent_channels);
    ggml_set_input(z);
    ggml_tensor* audio = ggml_cont(actx, sa3::sat::oobleck_decode(actx, AE, z, ac));
    ggml_set_output(audio);
    ggml_cgraph* agraph = ggml_new_graph_custom(actx, 32768, false);
    ggml_build_forward_expand(agraph, audio);
    ggml_gallocr_t aalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(AE.backend));
    if (!aalloc || !ggml_gallocr_alloc_graph(aalloc, agraph))
        throw std::runtime_error("failed to allocate Oobleck graph");
    const double ae_build_s = now_s() - t0;
    ggml_backend_tensor_set(z, decoder_z.data(), 0, decoder_z.size() * sizeof(float));
    t0 = now_s();
    if (!sa3::graph_compute_checked(AE.backend, agraph, "SAOS Oobleck decode", compute_error))
        throw std::runtime_error(compute_error);
    const double decode_s = now_s() - t0;

    const int decoded_samples = (int)audio->ne[0];
    const int channels = (int)audio->ne[1];
    std::vector<float> pcm((size_t)decoded_samples * channels);
    ggml_backend_tensor_get(audio, pcm.data(), 0, pcm.size() * sizeof(float));
    if (output_samples <= 0 || output_samples > decoded_samples) output_samples = decoded_samples;
    sa3::write_wav_planar_strided(wav_path, pcm.data(), output_samples, channels,
                                  (int)AE.u32("sat.sample_rate"), decoded_samples);

    std::vector<double> steady = step_ms;
    if (steady.size() > 1) steady.erase(steady.begin()); // CUDA graph capture/warm-up step
    std::sort(steady.begin(), steady.end());
    const double median_ms = steady[steady.size() / 2];
    const double audio_s = (double)output_samples / (double)AE.u32("sat.sample_rate");
    const double total_s = now_s() - total_start;
    printf("wrote %s  (%d samples, %d ch, %.3f audio sec)\n", wav_path,
           output_samples, channels, audio_s);
    printf("benchmark: conditioning=%.3fs dit_load=%.3fs dit_build_alloc=%.3fs denoise=%.3fs "
           "step_median_warm=%.1fms ae_load=%.3fs ae_build_alloc=%.3fs decode=%.3fs total=%.3fs RTF=%.3f\n",
           conditioning_s, dit_load_s, dit_build_s, denoise_s, median_ms, ae_load_s, ae_build_s,
           decode_s, total_s, total_s / audio_s);

    ggml_gallocr_free(aalloc);
    ggml_free(actx);
    AE.free();
    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
