// sat-generate -- family-aware CLI over the optional sa3_sat component.
#include "sat/foundation_prompt.h"
#include "sat/keybed.h"
#include "sat/model_paths.h"
#include "sat/pipeline.h"
#include "sat/profiles.h"
#include "wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

enum class ModelFamily { Saos, Sao1, Foundation, Keybeds };

struct Options {
    sa3::sat::PipelinePaths paths;
    std::string model = sa3::sat::kDefaultSaosVariant;
    std::string models_dir;
    std::string encoding = sa3::sat::kDefaultSatEncoding;
    std::string t5_encoding;
    std::string ae_encoding;
    std::string prompt;
    std::string negative_prompt;
    std::string output;
    std::string conditioning_prefix;
    std::string dump_conditioning_prefix;
    std::string latent_path;
    std::string initial_path;
    std::string step_noise_path;
    std::string foundation_profile = "royalcities";
    std::string randomize_mode = "standard";
    std::string family_hint;
    std::string key_root = "C";
    std::string key_mode = "minor";
    std::optional<float> seconds_start;
    std::optional<float> seconds;
    std::optional<float> seconds_total;
    std::optional<float> cfg_scale;
    std::optional<float> sigma_min;
    std::optional<float> sigma_max;
    std::optional<float> sigma_rho;
    std::optional<float> sde_eta;
    std::optional<sa3::sat::Sampler> sampler;
    std::optional<int> frames;
    std::optional<int> steps;
    std::optional<int> samples;
    std::optional<uint64_t> seed;
    int bars = 4;
    int bpm = 128;
    bool bars_set = false;
    bool bpm_set = false;
    bool key_root_set = false;
    bool key_mode_set = false;
    bool foundation_profile_set = false;
    bool randomize_mode_set = false;
    bool family_hint_set = false;
    bool randomize = false;
    std::string keybed_range = "c2-b5";
    std::string keybed_preview;  // NOTE:COUNT, e.g. C4:6
    std::string out_dir;
    std::vector<std::string> fx;
    bool keybed_range_set = false;
    bool wet = false;
    sa3::LoudnessParams loudness = sa3::loudness_defaults_from_env();
};

void print_help() {
    std::printf(
        "Usage:\n"
        "  sat-generate --model MODEL --prompt TEXT [options]\n"
        "  sat-generate --model foundation-1 --randomize [Foundation options]\n"
        "  sat-generate --model MODEL --dit FILE --t5 FILE --ae FILE --prompt TEXT [options]\n\n"
        "Models (downloaded independently; default: arc):\n"
        "  arc | stable-audio-open-small    SAOS ARC checkpoint\n"
        "  kickbass | jerry-grunge          SAOS full-checkpoint finetunes\n"
        "  stable-audio-open-1.0 | sao1     original 1.1B SAO checkpoint\n"
        "  foundation-1 | foundation        RoyalCities Foundation finetune\n"
        "  foundation-1.2-keybeds | keybeds RoyalCities playable keybed finetune\n\n"
        "Model files:\n"
        "  --models-dir DIR      root directory (default $SA3_MODELS_DIR or ./models)\n"
        "  --encoding TYPE       DiT/T5/Oobleck tier: F16 (default), Q8_0, Q5_K_M, Q4_K_M\n"
        "  --t5-encoding TYPE    override the T5 tier\n"
        "  --ae-encoding TYPE    override the Oobleck tier\n"
        "  --dit/--t5/--ae FILE  load explicit component paths\n\n"
        "Common generation:\n"
        "  --prompt TEXT         text prompt (required unless --randomize or --conditioning)\n"
        "  --negative-prompt T   negative text conditioning\n"
        "  --seconds-start N     timeline offset conditioning\n"
        "  --seconds N           requested output duration (SAOS/SAO 1.0)\n"
        "  --seconds-total N     learned duration conditioning override (advanced)\n"
        "  --sampler NAME        auto, pingpong, euler, dpmpp, dpmpp-2m-sde, dpmpp-3m-sde\n"
        "  --steps N             denoising steps (V-prediction models require at least 2)\n"
        "  --cfg-scale N         classifier-free guidance scale\n"
        "  --sigma-min N         V-prediction minimum sigma\n"
        "  --sigma-max N         V-prediction maximum sigma\n"
        "  --sigma-rho N         V-prediction schedule curvature (default 1)\n"
        "  --sde-eta N           SDE noise strength (default 1)\n"
        "  --seed N              audio seed; also makes --randomize reproducible\n"
        "  --out FILE            output WAV (default MODEL-ggml.wav; --wav is an alias)\n\n"
        "Output loudness (shared with sa3-generate):\n"
        "  --peak-normalize-db N peak target in dBFS (default +2)\n"
        "  --no-peak-normalize   disable peak normalization\n"
        "  --limiter-ceiling-db N  limiter ceiling in dBFS (default -0.3)\n"
        "  --no-limiter          disable the limiter\n"
        "  --limiter-knee N      soft-knee fraction in (0, 1] (default 0.8)\n"
        "Use --no-peak-normalize --no-limiter for raw decoded audio. The same\n"
        "SA3_PEAK_NORMALIZE_DB, SA3_LIMITER_CEILING_DB, and SA3_LIMITER_KNEE\n"
        "environment overrides apply.\n\n"
        "Backend environment:\n"
        "  SA3_DEVICE=metal      select Metal (GPU is selected automatically when available)\n"
        "  SA3_DEVICE=cpu        force CPU; SA3_THREADS controls CPU threads\n\n"
        "Foundation-1 (model-specific):\n"
        "  --bars N              4 or 8 (manual default 4)\n"
        "  --bpm N               100, 110, 120, 128, 130, 140, or 150 (manual default 128)\n"
        "  --key-root NOTE       C..B, including sharps/flats (manual default C)\n"
        "  --key-mode MODE       major or minor (manual default minor)\n"
        "  --foundation-profile P  royalcities (default) or gary\n"
        "  --randomize           generate and render a structured Foundation prompt\n"
        "  --randomize-mode M    standard/M1 (default) or mix/T1\n"
        "  --family NAME         lock --randomize to one Foundation instrument family\n"
        "                        Synth, Keys, Bass, Bowed Strings, Mallet, Wind, Guitar,\n"
        "                        Brass, Vocal, or Plucked Strings (case-insensitive)\n"
        "With --randomize, every omitted Foundation value is randomized; supplied values\n"
        "act as locks. Without it, --prompt is the manual descriptor override.\n"
        "Foundation derives its exact crop, seconds_total, and latent frames from bars/BPM;\n"
        "--seconds, --seconds-total, --frames, and --samples are rejected for this model.\n\n"
        "Foundation-1.2 Keybeds (model-specific):\n"
        "  --prompt TEXT         instrument descriptor, e.g. \"Grand Piano, Warm\"\n"
        "  --randomize           random keybed descriptor (seeded by --seed)\n"
        "  --keybed-range R      c2-b5 (default, 48 notes), c2-f6, or c2-b6 prompt labels\n"
        "  --keybed-preview N:C  preview C = 6, 12, or 24 notes from label N, e.g. C4:6\n"
        "  --wet                 Wet instead of Dry; --fx NAME adds an FX tag (repeatable)\n"
        "  --out-dir DIR         kit folder (default keybed-SEED): chunk WAVs, one WAV per\n"
        "                        sounding note, and kit.sfz\n"
        "Keybeds renders chromatic six-note chunks with one shared seed and resident models,\n"
        "defaults to DPM++ 3M SDE, 80 steps, CFG 6, sigma 0.03-500, and writes raw audio.\n"
        "Rendered pitch sits one octave below the prompt label; files and SFZ keys use the\n"
        "sounding pitch.\n\n"
        "Low-level/debug:\n"
        "  --conditioning PREFIX             read PREFIX.cross.f32/global.f32\n"
        "  --dump-conditioning PREFIX        write conditioning buffers\n"
        "  --latent FILE --initial-latent FILE --step-noise FILE\n"
        "  --frames N --samples N             explicit SAOS/SAO geometry\n\n"
        "Defaults:\n"
        "  SAOS ARC uses pingpong/8 steps/CFG 1; ordinary SAOS finetunes use\n"
        "  Euler/50 steps/CFG 4. SAO 1.0 uses DPM++ 3M SDE, sigma 0.3-500,\n"
        "  100 steps/CFG 7. Foundation's named profile supplies its sampler settings.\n\n"
        "Examples:\n"
        "  sat-generate --model arc --prompt \"warm dusty chillhop, 90 bpm\" --out loop.wav\n"
        "  sat-generate --model jerry-grunge --prompt \"grunge guitar riff\" --seed 42\n"
        "  sat-generate --model foundation-1 --randomize --bars 4 --bpm 128 --seed 42\n"
        "  sat-generate --model foundation-1 --randomize --randomize-mode mix --family Synth\n"
        "\n--randomize prints the exact prompt, seed, variant, timing, and output path.\n");
}

const char* require_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

int parse_int(const char* text, const char* option) {
    try {
        size_t used = 0;
        const long value = std::stol(text, &used, 10);
        if (used != std::strlen(text)) throw std::invalid_argument("suffix");
        return (int)value;
    } catch (...) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + text);
    }
}

uint64_t parse_u64(const char* text, const char* option) {
    try {
        size_t used = 0;
        const unsigned long long value = std::stoull(text, &used, 10);
        if (used != std::strlen(text)) throw std::invalid_argument("suffix");
        return (uint64_t)value;
    } catch (...) {
        throw std::runtime_error(std::string("invalid seed for ") + option + ": " + text);
    }
}

float parse_float(const char* text, const char* option) {
    try {
        size_t used = 0;
        const float value = std::stof(text, &used);
        if (used != std::strlen(text) || !std::isfinite(value)) throw std::invalid_argument("suffix");
        return value;
    } catch (...) {
        throw std::runtime_error(std::string("invalid number for ") + option + ": " + text);
    }
}

Options parse_options(int argc, char** argv) {
    Options o;
    const char* env_models = std::getenv("SA3_MODELS_DIR");
    o.models_dir = env_models && *env_models ? env_models : "models";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_help(); std::exit(0); }
        else if (arg == "--dit") o.paths.dit = require_value(argc, argv, i);
        else if (arg == "--t5") o.paths.t5 = require_value(argc, argv, i);
        else if (arg == "--ae") o.paths.autoencoder = require_value(argc, argv, i);
        else if (arg == "--model") o.model = require_value(argc, argv, i);
        else if (arg == "--models-dir") o.models_dir = require_value(argc, argv, i);
        else if (arg == "--encoding") o.encoding = require_value(argc, argv, i);
        else if (arg == "--t5-encoding") o.t5_encoding = require_value(argc, argv, i);
        else if (arg == "--ae-encoding") o.ae_encoding = require_value(argc, argv, i);
        else if (arg == "--conditioning") o.conditioning_prefix = require_value(argc, argv, i);
        else if (arg == "--prompt") o.prompt = require_value(argc, argv, i);
        else if (arg == "--negative-prompt") o.negative_prompt = require_value(argc, argv, i);
        else if (arg == "--seconds-start") o.seconds_start = parse_float(require_value(argc, argv, i), "--seconds-start");
        else if (arg == "--seconds") o.seconds = parse_float(require_value(argc, argv, i), "--seconds");
        else if (arg == "--seconds-total") o.seconds_total = parse_float(require_value(argc, argv, i), "--seconds-total");
        else if (arg == "--cfg-scale") o.cfg_scale = parse_float(require_value(argc, argv, i), "--cfg-scale");
        else if (arg == "--sigma-min") o.sigma_min = parse_float(require_value(argc, argv, i), "--sigma-min");
        else if (arg == "--sigma-max") o.sigma_max = parse_float(require_value(argc, argv, i), "--sigma-max");
        else if (arg == "--sigma-rho") o.sigma_rho = parse_float(require_value(argc, argv, i), "--sigma-rho");
        else if (arg == "--sde-eta") o.sde_eta = parse_float(require_value(argc, argv, i), "--sde-eta");
        else if (arg == "--sampler") o.sampler = sa3::sat::parse_sampler(require_value(argc, argv, i));
        else if (arg == "--dump-conditioning") o.dump_conditioning_prefix = require_value(argc, argv, i);
        else if (arg == "--out" || arg == "--wav") o.output = require_value(argc, argv, i);
        else if (arg == "--latent") o.latent_path = require_value(argc, argv, i);
        else if (arg == "--initial-latent") o.initial_path = require_value(argc, argv, i);
        else if (arg == "--step-noise") o.step_noise_path = require_value(argc, argv, i);
        else if (arg == "--frames") o.frames = parse_int(require_value(argc, argv, i), "--frames");
        else if (arg == "--steps") o.steps = parse_int(require_value(argc, argv, i), "--steps");
        else if (arg == "--samples") o.samples = parse_int(require_value(argc, argv, i), "--samples");
        else if (arg == "--seed") o.seed = parse_u64(require_value(argc, argv, i), "--seed");
        else if (arg == "--bars") { o.bars = parse_int(require_value(argc, argv, i), "--bars"); o.bars_set = true; }
        else if (arg == "--bpm") { o.bpm = parse_int(require_value(argc, argv, i), "--bpm"); o.bpm_set = true; }
        else if (arg == "--key-root") { o.key_root = require_value(argc, argv, i); o.key_root_set = true; }
        else if (arg == "--key-mode") { o.key_mode = require_value(argc, argv, i); o.key_mode_set = true; }
        else if (arg == "--foundation-profile") { o.foundation_profile = require_value(argc, argv, i); o.foundation_profile_set = true; }
        else if (arg == "--randomize-mode") { o.randomize_mode = require_value(argc, argv, i); o.randomize_mode_set = true; }
        else if (arg == "--family") { o.family_hint = require_value(argc, argv, i); o.family_hint_set = true; }
        else if (arg == "--randomize") o.randomize = true;
        else if (arg == "--keybed-range") { o.keybed_range = require_value(argc, argv, i); o.keybed_range_set = true; }
        else if (arg == "--keybed-preview") o.keybed_preview = require_value(argc, argv, i);
        else if (arg == "--wet") o.wet = true;
        else if (arg == "--fx") o.fx.push_back(require_value(argc, argv, i));
        else if (arg == "--out-dir") o.out_dir = require_value(argc, argv, i);
        else if (arg == "--peak-normalize-db") {
            o.loudness.peak_normalize_enabled = true;
            o.loudness.peak_normalize_db = parse_float(require_value(argc, argv, i), "--peak-normalize-db");
        }
        else if (arg == "--no-peak-normalize") o.loudness.peak_normalize_enabled = false;
        else if (arg == "--limiter-ceiling-db") {
            o.loudness.limiter_enabled = true;
            o.loudness.limiter_ceiling_db = parse_float(require_value(argc, argv, i), "--limiter-ceiling-db");
        }
        else if (arg == "--no-limiter") o.loudness.limiter_enabled = false;
        else if (arg == "--limiter-knee")
            o.loudness.limiter_knee = parse_float(require_value(argc, argv, i), "--limiter-knee");
        else throw std::runtime_error("unknown argument: " + arg + " (run --help)");
    }
    return o;
}

ModelFamily classify_model(const std::string& input, std::string& canonical) {
    const std::string saos = sa3::sat::canonical_saos_variant(input);
    if (saos == "arc" || saos == "kickbass" || saos == "jerry-grunge") {
        canonical = saos;
        return ModelFamily::Saos;
    }
    const std::string large = sa3::sat::canonical_sat_large_model(input);
    if (large == "stable-audio-open-1.0") {
        canonical = large;
        return ModelFamily::Sao1;
    }
    if (large == "foundation-1") {
        canonical = large;
        return ModelFamily::Foundation;
    }
    if (sa3::sat::is_foundation_keybeds_model(large)) {
        canonical = large;
        return ModelFamily::Keybeds;
    }
    throw std::runtime_error("unknown SAT model: " + input + " (run --help)");
}

bool valid_key_root(const std::string& root) {
    static const char* values[] = {
        "C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"
    };
    for (const char* value : values) if (root == value) return true;
    return false;
}

uint64_t entropy_seed() {
    std::random_device device;
    const uint64_t clock = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return ((uint64_t)device() << 32) ^ (uint64_t)device() ^ clock;
}

std::vector<float> read_all_f32(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    if (std::fseek(f, 0, SEEK_END) || std::ftell(f) < 0) {
        std::fclose(f); throw std::runtime_error("cannot size " + path);
    }
    const long bytes = std::ftell(f);
    std::rewind(f);
    if (bytes % (long)sizeof(float)) {
        std::fclose(f); throw std::runtime_error("invalid float buffer " + path);
    }
    std::vector<float> out((size_t)bytes / sizeof(float));
    const size_t got = std::fread(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    if (got != out.size()) throw std::runtime_error("short read " + path);
    return out;
}

void write_f32(const std::string& path, const std::vector<float>& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    if (std::fwrite(data.data(), sizeof(float), data.size(), f) != data.size()) {
        std::fclose(f); throw std::runtime_error("short write " + path);
    }
    std::fclose(f);
}

void validate_family_options(const Options& o, ModelFamily family) {
    const bool keybed_only = o.keybed_range_set || !o.keybed_preview.empty() || o.wet ||
        !o.fx.empty() || !o.out_dir.empty();
    if (family != ModelFamily::Keybeds && keybed_only)
        throw std::runtime_error("--keybed-*/--wet/--fx/--out-dir are Foundation-1.2 Keybeds options");
    if (family == ModelFamily::Keybeds) {
        if (o.bars_set || o.bpm_set || o.key_root_set || o.key_mode_set ||
            o.foundation_profile_set || o.randomize_mode_set || o.family_hint_set)
            throw std::runtime_error("--bars/--bpm/--key-*/--foundation-profile/--randomize-mode/--family are Foundation-1 loop options");
        if (o.seconds || o.seconds_total || o.frames || o.samples || o.seconds_start ||
            !o.output.empty() || !o.conditioning_prefix.empty() ||
            !o.dump_conditioning_prefix.empty() || !o.latent_path.empty() ||
            !o.initial_path.empty() || !o.step_noise_path.empty())
            throw std::runtime_error("Keybeds derives timing per chunk and writes a kit folder; use --out-dir");
        if (o.randomize && !o.prompt.empty())
            throw std::runtime_error("--randomize and --prompt are mutually exclusive");
        if (!o.fx.empty() && !o.wet)
            throw std::runtime_error("--fx requires --wet");
        return;
    }
    const bool foundation_only = o.bars_set || o.bpm_set || o.key_root_set || o.key_mode_set ||
        o.foundation_profile_set || o.randomize || o.randomize_mode_set || o.family_hint_set;
    if (family != ModelFamily::Foundation && foundation_only)
        throw std::runtime_error("--bars/--bpm/--key-*/--foundation-profile/--randomize are Foundation-1 options");
    if (family == ModelFamily::Foundation &&
        (o.seconds || o.seconds_total || o.frames || o.samples))
        throw std::runtime_error("Foundation-1 derives duration and frames from --bars/--bpm; raw duration overrides are not allowed");
    if (family == ModelFamily::Saos && o.seconds && *o.seconds > 11.0f)
        throw std::runtime_error("--seconds must be no greater than 11 for SAOS");
    if (o.randomize && !o.prompt.empty())
        throw std::runtime_error("--randomize and --prompt are mutually exclusive");
    if ((o.randomize_mode_set || o.family_hint_set) && !o.randomize)
        throw std::runtime_error("--randomize-mode and --family require --randomize");
    if (o.randomize && !o.conditioning_prefix.empty())
        throw std::runtime_error("--randomize requires native prompt conditioning, not --conditioning");
}

std::vector<sa3::sat::keybed::Chunk> plan_keybed(const Options& o) {
    namespace kb = sa3::sat::keybed;
    if (!o.keybed_preview.empty()) {
        if (o.keybed_range_set)
            throw std::runtime_error("--keybed-preview and --keybed-range are mutually exclusive");
        const size_t colon = o.keybed_preview.find(':');
        if (colon == std::string::npos)
            throw std::runtime_error("--keybed-preview must look like C4:6");
        const std::optional<int> root = kb::note_name_to_midi(o.keybed_preview.substr(0, colon));
        if (!root) throw std::runtime_error("--keybed-preview must look like C4:6");
        return kb::plan_preview(*root, parse_int(o.keybed_preview.c_str() + colon + 1,
                                                 "--keybed-preview"));
    }
    if (o.keybed_range == "c2-b5") return kb::plan_full_range(kb::FullRange::C2ToB5);
    if (o.keybed_range == "c2-f6") return kb::plan_full_range(kb::FullRange::C2ToF6);
    if (o.keybed_range == "c2-b6") return kb::plan_full_range(kb::FullRange::C2ToB6);
    throw std::runtime_error("--keybed-range must be c2-b5, c2-f6, or c2-b6");
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    FILE* f = std::fopen(path.string().c_str(), "wb");
    const bool ok = f && std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (f) std::fclose(f);
    if (!ok) throw std::runtime_error("cannot write " + path.string());
}

int run_keybed(Options& o, const std::string& model) {
    namespace kb = sa3::sat::keybed;
    namespace fs = std::filesystem;
    const std::vector<kb::Chunk> chunks = plan_keybed(o);
    const uint64_t seed = o.seed ? *o.seed : entropy_seed();
    std::string descriptor = o.prompt;
    if (o.randomize) descriptor = kb::random_descriptor(seed, o.wet).descriptor;
    if (kb::split_descriptor_tokens(descriptor).body.empty())
        throw std::runtime_error("Keybeds needs an instrument descriptor: --prompt or --randomize");

    if (o.paths.dit.empty() && o.paths.t5.empty() && o.paths.autoencoder.empty()) {
        std::string error;
        if (!sa3::sat::resolve_sat_large_model(o.models_dir, model, o.encoding, o.t5_encoding,
                                               o.ae_encoding, &o.paths, &error))
            throw std::runtime_error(error);
    } else if (o.paths.dit.empty() || o.paths.t5.empty() || o.paths.autoencoder.empty()) {
        throw std::runtime_error("explicit Keybeds paths require --dit, --t5, and --ae");
    }

    const kb::SamplerDefaults defaults;
    sa3::sat::GenerateParams params;
    params.seed = seed;
    params.sampler = o.sampler.value_or(sa3::sat::Sampler::Dpmpp3mSde);
    params.steps = o.steps.value_or(defaults.steps);
    params.cfg_scale = o.cfg_scale.value_or(defaults.cfg_scale);
    params.sigma_min = o.sigma_min.value_or(defaults.sigma_min);
    params.sigma_max = o.sigma_max.value_or(defaults.sigma_max);
    if (o.sigma_rho) params.sigma_rho = *o.sigma_rho;
    if (o.sde_eta) params.sde_eta = *o.sde_eta;
    params.negative_prompt = o.negative_prompt;
    params.loudness.peak_normalize_enabled = false; // raw audio keeps chunk levels consistent
    params.loudness.limiter_enabled = false;
    params.keep_models = true;
    if (params.steps < 2) throw std::runtime_error("V-prediction sampling requires at least two steps");

    const fs::path out_dir = o.out_dir.empty() ? fs::path("keybed-" + std::to_string(seed))
                                               : fs::path(o.out_dir);
    fs::create_directories(out_dir / "chunks");
    std::printf("request: model=%s seed=%llu chunks=%zu steps=%d cfg=%.3f\n", model.c_str(),
                (unsigned long long)seed, chunks.size(), params.steps, params.cfg_scale);
    std::printf("descriptor: %s (%s)\n", descriptor.c_str(), o.wet ? "Wet" : "Dry");

    sa3::sat::Pipeline pipeline(std::move(o.paths));
    std::vector<kb::SfzRegion> regions;
    const auto started = std::chrono::steady_clock::now();
    for (size_t index = 0; index < chunks.size(); ++index) {
        const kb::Chunk& chunk = chunks[index];
        params.prompt = kb::build_sequence_prompt(descriptor, chunk.label_midis, o.wet,
                                                  o.fx.empty() ? nullptr : &o.fx);
        params.seconds = (float)chunk.actual_seconds;
        params.seconds_total = (float)chunk.seconds_total;
        params.frames = (int)std::ceil(chunk.seconds_total * 44100.0 / 2048.0);
        std::printf("chunk %zu/%zu: %s\n", index + 1, chunks.size(), params.prompt.c_str());
        const sa3::sat::GenerateResult result = pipeline.generate(params);
        const std::vector<float> audio = kb::finish_chunk_audio(
            result.audio.data(), result.channels, result.samples, result.sample_rate,
            chunk.actual_seconds);
        const int64_t frames = (int64_t)audio.size() / result.channels;
        char chunk_name[80];
        std::snprintf(chunk_name, sizeof(chunk_name), "chunk_%02zu_labels_%s-%s.wav", index + 1,
                      kb::note_filename(chunk.label_midis.front()).c_str(),
                      kb::note_filename(chunk.label_midis.back()).c_str());
        sa3::write_wav_planar((out_dir / "chunks" / chunk_name).string(), audio.data(),
                              (int)frames, result.channels, result.sample_rate);
        for (const kb::NoteSlice& slice :
             kb::note_slices(chunk.label_midis, result.sample_rate, frames)) {
            if (std::any_of(regions.begin(), regions.end(), [&](const kb::SfzRegion& r) {
                    return r.midi == slice.sounding_midi;
                }))
                continue;
            const std::vector<float> note = kb::extract_note_sample(
                audio, result.channels, result.sample_rate, slice);
            const std::string file = kb::note_filename(slice.sounding_midi) + ".wav";
            sa3::write_wav_planar((out_dir / file).string(), note.data(),
                                  (int)(note.size() / (size_t)result.channels),
                                  result.channels, result.sample_rate);
            regions.push_back({file, slice.sounding_midi});
        }
        std::printf("  denoise=%.3fs load=%.3fs total=%.3fs\n", result.timing.denoise_s,
                    result.timing.dit_load_s + result.timing.ae_load_s, result.timing.total_s);
    }
    write_text(out_dir / "kit.sfz", kb::sfz_text(regions));
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    int lo = 127, hi = 0;
    for (const kb::SfzRegion& r : regions) { lo = std::min(lo, r.midi); hi = std::max(hi, r.midi); }
    std::printf("kit: %s (%zu notes, sounding %s-%s, %.1f s)\n", out_dir.string().c_str(),
                regions.size(), kb::midi_to_note_name(lo).c_str(),
                kb::midi_to_note_name(hi).c_str(), elapsed);
    return 0;
}

int run(int argc, char** argv) {
    Options o = parse_options(argc, argv);
    std::string model;
    const ModelFamily family = classify_model(o.model, model);
    validate_family_options(o, family);
    sa3::normalize_loudness_params(o.loudness);
    std::string loudness_error;
    if (!sa3::validate_loudness_params(o.loudness, loudness_error))
        throw std::runtime_error("invalid loudness settings: " + loudness_error);
    if (o.t5_encoding.empty()) o.t5_encoding = o.encoding;
    if (o.ae_encoding.empty()) o.ae_encoding = o.encoding;
    if (family == ModelFamily::Keybeds) return run_keybed(o, model);
    if (o.output.empty()) o.output = model + "-ggml.wav";

    sa3::sat::GenerateParams params;
    params.loudness = o.loudness;
    params.prompt = o.prompt;
    params.negative_prompt = o.negative_prompt;
    if (o.seed) params.seed = *o.seed;
    else if (o.randomize) params.seed = entropy_seed();

    std::optional<sa3::sat::FoundationTiming> foundation_timing;
    std::optional<sa3::sat::FoundationRandomPrompt> randomized;
    if (family == ModelFamily::Foundation) {
        if (o.randomize) {
            const sa3::sat::FoundationRandomControls controls =
                sa3::sat::randomize_foundation_controls(params.seed);
            if (!o.bars_set) o.bars = controls.bars;
            if (!o.bpm_set) o.bpm = controls.bpm;
            if (!o.key_root_set) o.key_root = controls.key_root;
            if (!o.key_mode_set) o.key_mode = controls.key_mode;
        }
        if (!sa3::sat::is_foundation_bar_count(o.bars) || !sa3::sat::is_foundation_bpm(o.bpm))
            (void)sa3::sat::resolve_foundation_timing(o.bars, o.bpm); // standardized diagnostics
        if (!valid_key_root(o.key_root))
            throw std::runtime_error("Foundation key root must be C..B, with an optional sharp or flat");
        if (o.key_mode != "major" && o.key_mode != "minor")
            throw std::runtime_error("Foundation key mode must be major or minor");
        if (o.foundation_profile == "royalcities")
            sa3::sat::apply_foundation_royalcities_sampler(params);
        else if (o.foundation_profile == "gary")
            sa3::sat::apply_foundation_gary_sampler(params);
        else
            throw std::runtime_error("--foundation-profile must be royalcities or gary");
        if (o.randomize) {
            const sa3::sat::FoundationPromptMode mode =
                sa3::sat::parse_foundation_prompt_mode(o.randomize_mode);
            randomized = sa3::sat::randomize_foundation_prompt(params.seed, mode, o.family_hint);
            params.prompt = randomized->description;
        }
        foundation_timing = sa3::sat::apply_foundation_timing(params, o.bars, o.bpm, false);
        params.prompt = sa3::sat::foundation_prompt(params.prompt, o.bars, o.bpm,
                                                    o.key_root, o.key_mode);
    }

    if (o.seconds_start) params.seconds_start = *o.seconds_start;
    if (o.seconds) params.seconds = *o.seconds;
    if (o.seconds_total) params.seconds_total = *o.seconds_total;
    if (o.cfg_scale) params.cfg_scale = *o.cfg_scale;
    if (o.sigma_min) params.sigma_min = *o.sigma_min;
    if (o.sigma_max) params.sigma_max = *o.sigma_max;
    if (o.sigma_rho) params.sigma_rho = *o.sigma_rho;
    if (o.sde_eta) params.sde_eta = *o.sde_eta;
    if (o.sampler) params.sampler = *o.sampler;
    if (o.frames) params.frames = *o.frames;
    if (o.steps) params.steps = *o.steps;
    if (o.samples) params.output_samples = *o.samples;
    if (family != ModelFamily::Saos && o.steps && *o.steps == 1)
        throw std::runtime_error("V-prediction sampling requires at least two steps");
    if (family != ModelFamily::Foundation && !o.frames && (o.seconds || o.samples)) {
        const double wanted_samples = o.samples ? (double)*o.samples : (double)params.seconds * 44100.0;
        params.frames = (int)std::ceil(wanted_samples / 2048.0);
    }

    const int explicit_paths = !o.paths.dit.empty() + !o.paths.t5.empty() + !o.paths.autoencoder.empty();
    if (explicit_paths == 0) {
        std::string error;
        const bool ok = family == ModelFamily::Saos
            ? sa3::sat::resolve_saos_model(o.models_dir, model, o.encoding, o.t5_encoding,
                                           o.ae_encoding, &o.paths, &error)
            : sa3::sat::resolve_sat_large_model(o.models_dir, model, o.encoding, o.t5_encoding,
                                                o.ae_encoding, &o.paths, &error);
        if (!ok) throw std::runtime_error(error);
    } else if ((!o.conditioning_prefix.empty() && explicit_paths != 2) ||
               (o.conditioning_prefix.empty() && explicit_paths != 3)) {
        throw std::runtime_error("explicit model mode requires --dit and --ae, plus --t5 when using --prompt");
    }

    const bool native_conditioning = !o.paths.t5.empty() && !params.prompt.empty();
    if (o.paths.dit.empty() || o.paths.autoencoder.empty() ||
        (o.conditioning_prefix.empty() && !native_conditioning) ||
        (!o.conditioning_prefix.empty() && native_conditioning))
        throw std::runtime_error("missing/invalid conditioning; provide --prompt, --randomize, or --conditioning (run --help)");
    if (!o.conditioning_prefix.empty()) {
        params.cross_conditioning = read_all_f32(o.conditioning_prefix + ".cross.f32");
        params.global_conditioning = read_all_f32(o.conditioning_prefix + ".global.f32");
    }
    if (!o.initial_path.empty()) params.initial_latent = read_all_f32(o.initial_path);
    if (!o.step_noise_path.empty()) params.step_noise = read_all_f32(o.step_noise_path);

    std::printf("request: model=%s seed=%llu\n", model.c_str(),
                (unsigned long long)params.seed);
    if (!params.prompt.empty()) std::printf("prompt: %s\n", params.prompt.c_str());
    if (randomized)
        std::printf("randomize: variant=%s mode=%s family=%s subfamily=%s\n",
                    randomized->variant.c_str(), o.randomize_mode.c_str(),
                    randomized->family.c_str(), randomized->subfamily.c_str());
    if (foundation_timing)
        std::printf("foundation: bars=%d bpm=%d key=%s %s samples=%d seconds_total=%.0f frames=%d profile=%s\n",
                    o.bars, o.bpm, o.key_root.c_str(), o.key_mode.c_str(),
                    foundation_timing->output_samples,
                    foundation_timing->conditioning_seconds_total,
                    foundation_timing->latent_frames, o.foundation_profile.c_str());

    params.progress = [](const sa3::sat::StepProgress& p) {
        std::printf("step %d/%d  t %.6f -> %.6f  %.1f ms\n", p.step, p.steps,
                    p.t_current, p.t_next, p.milliseconds);
    };
    sa3::sat::Pipeline pipeline(std::move(o.paths));
    sa3::sat::GenerateResult result = pipeline.generate(params);
    if (result.prompt_tokens)
        std::printf("prompt tokens: %d/%d\n", result.prompt_tokens, result.max_prompt_tokens);
    if (!o.dump_conditioning_prefix.empty()) {
        write_f32(o.dump_conditioning_prefix + ".cross.f32", result.cross_conditioning);
        write_f32(o.dump_conditioning_prefix + ".global.f32", result.global_conditioning);
    }
    if (!o.latent_path.empty()) write_f32(o.latent_path, result.latent);
    sa3::write_wav_planar(o.output, result.audio.data(), result.samples,
                          result.channels, result.sample_rate);
    const double audio_seconds = (double)result.samples / result.sample_rate;
    std::printf("audio: %s (%d samples, %d ch, %.3f sec)\n", o.output.c_str(),
                result.samples, result.channels, audio_seconds);
    std::printf("model: objective=%s sampler=%s steps=%d cfg=%.3f\n",
                result.objective.c_str(), sa3::sat::sampler_name(result.sampler),
                result.steps, result.cfg_scale);
    if (result.objective == "v")
        std::printf("sigma: min=%.6g max=%.6g rho=%.6g eta=%.6g\n",
                    result.sigma_min, result.sigma_max, result.sigma_rho, result.sde_eta);
    std::printf("loudness: decoded_peak=%.6f final_peak=%.6f",
                result.loudness.decoded_peak, result.loudness.final_peak);
    if (result.loudness.peak_normalize_gain_set)
        std::printf(" peak_gain=%.6f", result.loudness.peak_normalize_gain);
    if (result.loudness.limiter_limited_fraction_set)
        std::printf(" limited=%.4f%%", 100.0f * result.loudness.limiter_limited_fraction);
    std::printf("\n");
    const sa3::sat::GenerateTiming& t = result.timing;
    std::printf("benchmark: conditioning=%.3fs dit_load=%.3fs dit_build_alloc=%.3fs denoise=%.3fs "
                "step_median_warm=%.1fms ae_load=%.3fs ae_build_alloc=%.3fs decode=%.3fs "
                "total=%.3fs RTF=%.3f\n", t.conditioning_s, t.dit_load_s, t.dit_build_s,
                t.denoise_s, t.warm_step_median_ms, t.ae_load_s, t.ae_build_s, t.decode_s,
                t.total_s, t.total_s / audio_seconds);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
