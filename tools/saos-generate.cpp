// saos-generate -- thin CLI over the optional sa3_saos pipeline component.
#include "sat/pipeline.h"
#include "sat/model_paths.h"
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

static void print_help() {
    printf(
        "Usage:\n"
        "  saos-generate --prompt TEXT [options]\n"
        "  saos-generate --dit FILE --t5 FILE --ae FILE --prompt TEXT [options]\n\n"
        "Published SAOS models (from thepatch/stable-audio-open-small-GGUF):\n"
        "  --model NAME          arc (default), kickbass, or jerry-grunge\n"
        "  --models-dir DIR      model directory (default $SA3_MODELS_DIR or ./models)\n"
        "  --encoding TYPE       DiT/T5/Oobleck tier (default q5_k_m)\n"
        "  --t5-encoding TYPE    override the T5 tier\n"
        "  --ae-encoding TYPE    override the Oobleck tier\n\n"
        "Generation:\n"
        "  --prompt TEXT         text prompt (required unless --conditioning is used)\n"
        "  --negative-prompt T   negative text conditioning\n"
        "  --seconds N           output duration, up to 11 seconds (default 11)\n"
        "  --sampler NAME        auto, pingpong, euler, or dpmpp\n"
        "  --steps N             denoising steps\n"
        "  --cfg-scale N         classifier-free guidance scale\n"
        "  --seed N              random seed\n"
        "  --out FILE            output WAV (default saos-ggml.wav; --wav is an alias)\n"
        "  --peak-normalize      normalize the output WAV peak\n\n"
        "Auto defaults come from the GGUF: ARC uses pingpong/8 steps/CFG 1; ordinary\n"
        "SAOS finetunes use Euler/50 steps/CFG 4. Explicit flags override them.\n\n"
        "Low-level/debug:\n"
        "  --dit FILE --t5 FILE --ae FILE    load explicit components\n"
        "  --conditioning PREFIX             read PREFIX.cross.f32/global.f32\n"
        "  --dump-conditioning PREFIX        write conditioning buffers\n"
        "  --latent FILE --initial-latent FILE --step-noise FILE\n"
        "  --frames N --samples N\n\n"
        "Examples:\n"
        "  saos-generate --prompt \"warm dusty chillhop, 90 bpm\" --out loop.wav\n"
        "  saos-generate --model jerry-grunge --prompt \"grunge guitar riff\" --seed 42\n"
        "  saos-generate --model kickbass --sampler dpmpp --steps 40 --cfg-scale 4 \\\n"
        "      --prompt \"hard techno kick and rolling bass loop, 140 BPM\"\n");
}

static int run(int argc, char** argv) {
    sa3::sat::PipelinePaths paths;
    sa3::sat::GenerateParams params;
    const char* cond_prefix = nullptr;
    const char* dump_cond_prefix = nullptr;
    std::string wav_path = "saos-ggml.wav";
    std::string model = sa3::sat::kDefaultSaosVariant;
    std::string encoding = sa3::sat::kDefaultSaosEncoding;
    std::string t5_encoding;
    std::string ae_encoding;
    const char* env_models = std::getenv("SA3_MODELS_DIR");
    std::string models_dir = env_models && *env_models ? env_models : "models";
    const char* latent_path = nullptr;
    const char* initial_path = nullptr;
    const char* step_noise_path = nullptr;
    bool peak_normalize = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dit") && i + 1 < argc) paths.dit = argv[++i];
        else if (!strcmp(argv[i], "--ae") && i + 1 < argc) paths.autoencoder = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--models-dir") && i + 1 < argc) models_dir = argv[++i];
        else if (!strcmp(argv[i], "--encoding") && i + 1 < argc) encoding = argv[++i];
        else if (!strcmp(argv[i], "--t5-encoding") && i + 1 < argc) t5_encoding = argv[++i];
        else if (!strcmp(argv[i], "--ae-encoding") && i + 1 < argc) ae_encoding = argv[++i];
        else if (!strcmp(argv[i], "--conditioning") && i + 1 < argc) cond_prefix = argv[++i];
        else if (!strcmp(argv[i], "--t5") && i + 1 < argc) paths.t5 = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) params.prompt = argv[++i];
        else if (!strcmp(argv[i], "--negative-prompt") && i + 1 < argc) params.negative_prompt = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) params.seconds = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-scale") && i + 1 < argc) params.cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sampler") && i + 1 < argc)
            params.sampler = sa3::sat::parse_sampler(argv[++i]);
        else if (!strcmp(argv[i], "--dump-conditioning") && i + 1 < argc) dump_cond_prefix = argv[++i];
        else if ((!strcmp(argv[i], "--out") || !strcmp(argv[i], "--wav")) && i + 1 < argc)
            wav_path = argv[++i];
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
            print_help();
            return 0;
        } else {
            throw std::runtime_error(std::string("unknown/incomplete argument: ") + argv[i]);
        }
    }
    const int explicit_paths = !paths.dit.empty() + !paths.t5.empty() + !paths.autoencoder.empty();
    if (explicit_paths == 0) {
        if (t5_encoding.empty()) t5_encoding = encoding;
        if (ae_encoding.empty()) ae_encoding = encoding;
        std::string error;
        if (!sa3::sat::resolve_saos_model(models_dir, model, encoding, t5_encoding,
                                          ae_encoding, &paths, &error))
            throw std::runtime_error(error);
    } else if ((cond_prefix && explicit_paths != 2) || (!cond_prefix && explicit_paths != 3)) {
        throw std::runtime_error("explicit model mode requires --dit and --ae, plus --t5 when using --prompt");
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
    printf("wrote %s  (%d samples, %d ch, %.3f audio sec)\n", wav_path.c_str(),
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
