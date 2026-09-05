// saos-generate -- thin CLI over the optional sa3_saos pipeline component.
#include "sat/pipeline.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static std::vector<float> read_all_f32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    if (fseek(f, 0, SEEK_END) || ftell(f) < 0) {
        fclose(f); throw std::runtime_error("cannot size " + path);
    }
    const long bytes = ftell(f);
    rewind(f);
    if (bytes % (long)sizeof(float)) {
        fclose(f); throw std::runtime_error("invalid float buffer " + path);
    }
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

static int run(int argc, char** argv) {
    sa3::sat::PipelinePaths paths;
    sa3::sat::GenerateParams params;
    const char* cond_prefix = nullptr;
    const char* dump_cond_prefix = nullptr;
    const char* wav_path = "saos-ggml.wav";
    const char* latent_path = nullptr;
    const char* initial_path = nullptr;
    const char* step_noise_path = nullptr;
    bool peak_normalize = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dit") && i + 1 < argc) paths.dit = argv[++i];
        else if (!strcmp(argv[i], "--ae") && i + 1 < argc) paths.autoencoder = argv[++i];
        else if (!strcmp(argv[i], "--conditioning") && i + 1 < argc) cond_prefix = argv[++i];
        else if (!strcmp(argv[i], "--t5") && i + 1 < argc) paths.t5 = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) params.prompt = argv[++i];
        else if (!strcmp(argv[i], "--negative-prompt") && i + 1 < argc) params.negative_prompt = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) params.seconds = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-scale") && i + 1 < argc) params.cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sampler") && i + 1 < argc)
            params.sampler = sa3::sat::parse_sampler(argv[++i]);
        else if (!strcmp(argv[i], "--dump-conditioning") && i + 1 < argc) dump_cond_prefix = argv[++i];
        else if (!strcmp(argv[i], "--wav") && i + 1 < argc) wav_path = argv[++i];
        else if (!strcmp(argv[i], "--latent") && i + 1 < argc) latent_path = argv[++i];
        else if (!strcmp(argv[i], "--initial-latent") && i + 1 < argc) initial_path = argv[++i];
        else if (!strcmp(argv[i], "--step-noise") && i + 1 < argc) step_noise_path = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) params.frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) params.steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc) params.output_samples = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            params.seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--peak-normalize")) peak_normalize = true;
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: saos-generate --dit model.gguf --ae decoder.gguf "
                   "(--t5 t5.gguf --prompt TEXT | --conditioning PREFIX) "
                   "[--negative-prompt TEXT] [--seconds 11] [--sampler auto|euler|dpmpp|pingpong] "
                   "[--cfg-scale N] [--frames 256] [--steps N] [--samples N] [--seed 1234] "
                   "[--peak-normalize] [--wav out.wav] "
                   "[--initial-latent x.f32 --step-noise noise.f32]\n");
            return 0;
        } else {
            throw std::runtime_error(std::string("unknown/incomplete argument: ") + argv[i]);
        }
    }
    const bool native_conditioning = !paths.t5.empty() && !params.prompt.empty();
    if (paths.dit.empty() || paths.autoencoder.empty() ||
        (!cond_prefix && !native_conditioning) || (cond_prefix && native_conditioning))
        throw std::runtime_error("missing/invalid arguments; run --help");
    if (cond_prefix) {
        const std::string prefix(cond_prefix);
        params.cross_conditioning = read_all_f32(prefix + ".cross.f32");
        params.global_conditioning = read_all_f32(prefix + ".global.f32");
    }
    if (initial_path) params.initial_latent = read_all_f32(initial_path);
    if (step_noise_path) params.step_noise = read_all_f32(step_noise_path);
    params.progress = [](const sa3::sat::StepProgress& p) {
        printf("step %d/%d  t %.6f -> %.6f  %.1f ms\n", p.step, p.steps,
               p.t_current, p.t_next, p.milliseconds);
    };

    sa3::sat::Pipeline pipeline(std::move(paths));
    sa3::sat::GenerateResult result = pipeline.generate(params);
    if (result.prompt_tokens)
        printf("prompt tokens: %d/%d\n", result.prompt_tokens, result.max_prompt_tokens);
    if (dump_cond_prefix) {
        write_f32(std::string(dump_cond_prefix) + ".cross.f32", result.cross_conditioning);
        write_f32(std::string(dump_cond_prefix) + ".global.f32", result.global_conditioning);
    }
    if (latent_path) write_f32(latent_path, result.latent);
    if (peak_normalize) {
        float peak = 0.0f;
        for (float sample : result.audio) peak = std::max(peak, std::fabs(sample));
        if (peak > 0.0f)
            for (float& sample : result.audio) sample /= peak;
        printf("peak normalized from %.6f\n", peak);
    }
    sa3::write_wav_planar(wav_path, result.audio.data(), result.samples,
                          result.channels, result.sample_rate);
    const double audio_seconds = (double)result.samples / result.sample_rate;
    printf("wrote %s  (%d samples, %d ch, %.3f audio sec)\n", wav_path,
           result.samples, result.channels, audio_seconds);
    printf("model: objective=%s sampler=%s steps=%d cfg=%.3f\n",
           result.objective.c_str(), sa3::sat::sampler_name(result.sampler),
           result.steps, result.cfg_scale);
    const sa3::sat::GenerateTiming& t = result.timing;
    printf("benchmark: conditioning=%.3fs dit_load=%.3fs dit_build_alloc=%.3fs denoise=%.3fs "
           "step_median_warm=%.1fms ae_load=%.3fs ae_build_alloc=%.3fs decode=%.3fs "
           "total=%.3fs RTF=%.3f\n", t.conditioning_s, t.dit_load_s, t.dit_build_s,
           t.denoise_s, t.warm_step_median_ms, t.ae_load_s, t.ae_build_s, t.decode_s,
           t.total_s, t.total_s / audio_seconds);
    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
