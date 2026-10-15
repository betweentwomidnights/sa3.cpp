// A model-scoped HTTP service for Stable Audio Tools checkpoints. Run separate
// instances for Foundation-1 and SAOS; both use the same packaged runtime. A
// SAOS instance serves the base model and every installed finetune.
#include "sat/foundation_prompt.h"
#include "sat/model_paths.h"
#include "sat/pipeline.h"
#include "sat/profiles.h"
#include "sat/time_stretch.h"
#include "runtime_props.h"
#include "wav.h"

#include "httplib.h"
#include "yyjson.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

enum class Family { Saos, Foundation, Sao1 };

struct Config {
    std::string host = "127.0.0.1";
    int port = 8015;
    std::string model = "foundation-1";
    Family family = Family::Foundation;
    std::string models_dir = "models";
    std::string encoding = "F16";
    std::string t5_encoding;
    std::string ae_encoding;
    std::string device;
};

struct Job {
    std::string status = "queued";
    std::string audio;
    std::string error;
    std::string prompt;
    std::string model;
    int progress = 0;
    int step = 0;
    int total_steps = 0;
    int bars = 0;
    double bpm = 0;
    double host_bpm = 0;
    double seconds = 0;
    uint64_t seed = 0;
    std::string mode;              // "audio2audio", or empty for generation
    double init_noise_level = 0;
    std::chrono::steady_clock::time_point created = std::chrono::steady_clock::now();
};

std::mutex g_mutex;
std::map<std::string, Job> g_jobs;
std::unique_ptr<sa3::sat::Pipeline> g_pipeline;
bool g_busy = false;

std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char buf[7];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else out += (char)c;
    }
    return out;
}

std::string number(double value) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.9g", value);
    return buf;
}

void fail(httplib::Response& res, int status, const std::string& message) {
    res.status = status;
    res.set_content("{\"success\":false,\"error\":\"" + escape(message) + "\"}",
                    "application/json");
}

std::string b64_encode(const std::string& input) {
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve((input.size() + 2) / 3 * 4);
    size_t i = 0;
    while (i + 3 <= input.size()) {
        const unsigned n = (unsigned char)input[i] << 16 |
                           (unsigned char)input[i + 1] << 8 | (unsigned char)input[i + 2];
        output += alphabet[(n >> 18) & 63]; output += alphabet[(n >> 12) & 63];
        output += alphabet[(n >> 6) & 63]; output += alphabet[n & 63];
        i += 3;
    }
    if (i < input.size()) {
        unsigned n = (unsigned char)input[i] << 16;
        if (i + 1 < input.size()) n |= (unsigned char)input[i + 1] << 8;
        output += alphabet[(n >> 18) & 63]; output += alphabet[(n >> 12) & 63];
        output += i + 1 < input.size() ? alphabet[(n >> 6) & 63] : '=';
        output += '=';
    }
    return output;
}

std::string b64_decode(const std::string& input) {
    auto value = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+' || c == '-') return 62;
        if (c == '/' || c == '_') return 63;
        return -1;
    };
    std::string output;
    output.reserve(input.size() / 4 * 3);
    unsigned buffer = 0;
    int bits = 0;
    for (unsigned char c : input) {
        if (c == '=' || std::isspace(c)) continue;
        const int v = value(c);
        if (v < 0) throw std::invalid_argument("audio_data is not valid base64");
        buffer = (buffer << 6) | (unsigned)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            output += (char)((buffer >> bits) & 0xff);
        }
    }
    return output;
}

uint64_t random_seed() {
    std::random_device source;
    return ((uint64_t)source() << 32) | source();
}

std::string session_id() {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", (unsigned long long)random_seed());
    return text;
}

std::string string_field(yyjson_val* root, const char* key, const std::string& fallback = {}) {
    yyjson_val* value = yyjson_obj_get(root, key);
    if (!value || yyjson_is_null(value)) return fallback;
    if (!yyjson_is_str(value)) throw std::invalid_argument(std::string(key) + " must be a string");
    return yyjson_get_str(value);
}

double number_field(yyjson_val* root, const char* key, double fallback) {
    yyjson_val* value = yyjson_obj_get(root, key);
    if (!value || yyjson_is_null(value)) return fallback;
    if (!yyjson_is_num(value)) throw std::invalid_argument(std::string(key) + " must be a number");
    const double n = yyjson_get_num(value);
    if (!std::isfinite(n)) throw std::invalid_argument(std::string(key) + " must be finite");
    return n;
}

int int_field(yyjson_val* root, const char* key, int fallback) {
    const double n = number_field(root, key, fallback);
    if (n != std::floor(n) || n < -1 || n > 1000000)
        throw std::invalid_argument(std::string(key) + " must be an integer");
    return (int)n;
}

uint64_t seed_field(yyjson_val* root) {
    yyjson_val* value = yyjson_obj_get(root, "seed");
    if (!value || yyjson_is_null(value)) return random_seed();
    if (!yyjson_is_int(value)) throw std::invalid_argument("seed must be an integer");
    if (yyjson_is_sint(value) && yyjson_get_sint(value) == -1) return random_seed();
    if (yyjson_is_sint(value) && yyjson_get_sint(value) < 0)
        throw std::invalid_argument("seed must be nonnegative or -1");
    return yyjson_get_uint(value);
}

// The request leaves the seed to the server (omitted, null, or -1).
bool seed_drawn(yyjson_val* root) {
    yyjson_val* given = yyjson_obj_get(root, "seed");
    return !given || yyjson_is_null(given) || (yyjson_is_sint(given) && yyjson_get_sint(given) == -1);
}

bool bool_field(yyjson_val* root, const char* key, bool fallback = false) {
    yyjson_val* value = yyjson_obj_get(root, key);
    if (!value || yyjson_is_null(value)) return fallback;
    if (!yyjson_is_bool(value)) throw std::invalid_argument(std::string(key) + " must be a boolean");
    return yyjson_get_bool(value);
}

bool has_field(yyjson_val* root, const char* key) {
    yyjson_val* value = yyjson_obj_get(root, key);
    return value && !yyjson_is_null(value);
}

bool valid_key_root(const std::string& key) {
    static const char* roots[] = {"C","C#","Db","D","D#","Eb","E","F",
                                "F#","Gb","G","G#","Ab","A","A#","Bb","B"};
    for (const char* root : roots) if (key == root) return true;
    return false;
}

int nearest_foundation_bpm(double host_bpm) {
    int nearest = sa3::sat::kFoundationBpms[0];
    for (int candidate : sa3::sat::kFoundationBpms)
        if (std::abs(candidate - host_bpm) < std::abs(nearest - host_bpm)) nearest = candidate;
    return nearest;
}

double prompt_bpm(const std::string& prompt) {
    static const std::regex tag(R"((\d+(?:\.\d+)?)\s*bpm\b)", std::regex_constants::icase);
    std::smatch match;
    return std::regex_search(prompt, match, tag) ? std::stod(match[1].str()) : 0;
}

std::string foundation_descriptor(yyjson_val* root) {
    std::vector<std::string> parts;
    auto append = [&](const std::string& value) { if (!value.empty()) parts.push_back(value); };
    auto append_array = [&](const char* key) {
        yyjson_val* value = yyjson_obj_get(root, key);
        if (!value || yyjson_is_null(value)) return;
        if (yyjson_is_str(value)) { append(yyjson_get_str(value)); return; }
        if (!yyjson_is_arr(value)) throw std::invalid_argument(std::string(key) + " must be text or an array");
        yyjson_arr_iter iter;
        yyjson_arr_iter_init(value, &iter);
        yyjson_val* item;
        while ((item = yyjson_arr_iter_next(&iter))) {
            if (!yyjson_is_str(item)) throw std::invalid_argument(std::string(key) + " must contain only text");
            append(yyjson_get_str(item));
        }
    };
    for (const char* key : {"family", "subfamily", "descriptor_knob_a", "descriptor_knob_b",
                            "descriptor_knob_c"}) append(string_field(root, key));
    for (const char* key : {"descriptors_extra", "spatial_tags", "band_tags", "wave_tech_tags",
                            "style_tags", "behavior_tags"}) append_array(key);
    for (const auto& field : {std::pair{"reverb_enabled", "reverb_amount"},
                              {"delay_enabled", "delay_type"},
                              {"distortion_enabled", "distortion_amount"},
                              {"phaser_enabled", "phaser_amount"},
                              {"bitcrush_enabled", "bitcrush_amount"}})
        if (bool_field(root, field.first)) append(string_field(root, field.second));
    std::string result;
    for (const auto& part : parts) {
        if (!result.empty()) result += ", ";
        result += part;
    }
    return result;
}

std::string foundation_prompt_with_tags(std::string descriptor, int bars, int bpm,
                                        const std::string& root, const std::string& mode) {
    static const std::regex bars_tag(R"(\b[0-9]+\s+bars\b)", std::regex_constants::icase);
    static const std::regex bpm_tag(R"(\b[0-9]+\s+bpm\b)", std::regex_constants::icase);
    static const std::regex key_tag(R"(\b[A-G](?:#|b)?\s+(?:major|minor)\b)", std::regex_constants::icase);
    const auto has_expected = [&](const std::regex& tag, const std::string& expected) {
        std::smatch match;
        if (!std::regex_search(descriptor, match, tag)) return false;
        std::string actual = match.str();
        std::string wanted = expected;
        std::transform(actual.begin(), actual.end(), actual.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        std::transform(wanted.begin(), wanted.end(), wanted.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (actual != wanted) throw std::invalid_argument("prompt timing or key conflicts with the selected Foundation controls");
        return true;
    };
    if (!has_expected(bars_tag, std::to_string(bars) + " Bars"))
        descriptor += ", " + std::to_string(bars) + " Bars";
    if (!has_expected(bpm_tag, std::to_string(bpm) + " BPM"))
        descriptor += ", " + std::to_string(bpm) + " BPM";
    if (!has_expected(key_tag, root + " " + mode)) descriptor += ", " + root + " " + mode;
    return descriptor;
}

std::string json_strings(const std::vector<std::string>& values) {
    std::string out = "[";
    for (const auto& value : values) {
        if (out.size() > 1) out += ",";
        out += "\"" + escape(value) + "\"";
    }
    return out + "]";
}

// POST /randomize: a whole Foundation preset from the native prompt engine,
// split into the controls a plugin shows. The request and response keep
// gary4local's Python /randomize contract, which gary4juce reads field by
// field: seed (-1 or omitted for random), mode (standard or mix), variant
// (auto, M1 or T1, where T1 means mix) and family_hint.
std::string foundation_randomize_json(yyjson_val* root) {
    // A drawn seed stays within 31 bits, as Python's did: gary4juce reads it
    // back as an int to show in its seed box.
    uint64_t seed = seed_field(root);
    if (seed_drawn(root)) seed &= 0x7fffffffu;
    const auto mode = sa3::sat::parse_foundation_prompt_mode(string_field(root, "mode", "standard"));
    std::string variant = string_field(root, "variant", "auto");
    if (variant != "auto" && variant != "M1" && variant != "T1")
        throw std::invalid_argument("variant must be auto, M1 or T1");
    const bool mix = mode == sa3::sat::FoundationPromptMode::Mix || variant == "T1";
    if (variant == "auto") variant = mix ? "T1" : "M1";
    const auto prompt = sa3::sat::randomize_foundation_prompt(
        seed, mix ? sa3::sat::FoundationPromptMode::Mix : sa3::sat::FoundationPromptMode::Standard,
        string_field(root, "family_hint"));
    const auto parts = sa3::sat::decompose_foundation_prompt(prompt);

    auto knob = [&](size_t i) { return i < parts.descriptors.size() ? parts.descriptors[i] : std::string(); };
    const std::vector<std::string> extras(
        parts.descriptors.begin() + (long)std::min<size_t>(3, parts.descriptors.size()), parts.descriptors.end());
    auto fx = [](const char* name, const char* value_key, const std::string& value) {
        return std::string(",\"") + name + "_enabled\":" + (value.empty() ? "false" : "true") +
            ",\"" + value_key + "\":\"" + escape(value) + "\"";
    };
    return "{\"success\":true,\"seed\":" + std::to_string(seed) +
        ",\"mode\":\"" + sa3::sat::foundation_prompt_mode_name(mode) +
        "\",\"variant\":\"" + variant +
        "\",\"family\":\"" + escape(prompt.family) +
        "\",\"subfamily\":\"" + escape(prompt.subfamily) +
        "\",\"descriptor_knob_a\":\"" + escape(knob(0)) +
        "\",\"descriptor_knob_b\":\"" + escape(knob(1)) +
        "\",\"descriptor_knob_c\":\"" + escape(knob(2)) +
        "\",\"descriptors_extra\":" + json_strings(extras) +
        fx("reverb", "reverb_amount", parts.reverb) +
        fx("delay", "delay_type", parts.delay) +
        fx("distortion", "distortion_amount", parts.distortion) +
        fx("phaser", "phaser_amount", parts.phaser) +
        fx("bitcrush", "bitcrush_amount", parts.bitcrush) +
        ",\"behavior_tags\":" + json_strings(prompt.melody) +
        ",\"speed\":\"" + escape(parts.speed) +
        "\",\"structure\":\"" + escape(parts.structure) +
        "\",\"rhythm\":" + json_strings(parts.rhythm) +
        ",\"contour\":" + json_strings(parts.contour) +
        ",\"density\":" + json_strings(parts.density) +
        ",\"spatial_tags\":" + json_strings(parts.spatial) +
        ",\"band_tags\":" + json_strings(parts.bands) +
        ",\"wave_tech_tags\":" + json_strings(parts.wave_tech) +
        ",\"style_tags\":" + json_strings(parts.styles) +
        ",\"all_tags\":" + json_strings(prompt.tags) +
        ",\"prompt\":\"" + escape(prompt.description) + "\"}";
}

struct Prepared {
    sa3::sat::GenerateParams params;
    int bars = 0;
    double bpm = 0;
    double host_bpm = 0;
    double seconds = 0;
    int stretched_samples = 0;
    std::string mode;
};

// Audio-to-audio keeps gary4local's Python /audio2audio contract: audio_data
// (base64 WAV) and host_bpm are required, bars defaults to 8, and
// init_noise_level (0.01-1, default 0.25) says how far from the input to go.
// The input is stretched from host_bpm to the model BPM before encoding, and
// the result back afterwards, as Python did.
Prepared prepare(const Config& config, yyjson_val* root, bool loop, bool audio2audio = false) {
    Prepared out;
    auto& p = out.params;
    p.prompt = string_field(root, "prompt");
    p.negative_prompt = string_field(root, "negative_prompt");
    p.seed = seed_field(root);
    p.keep_models = true;
    if (audio2audio && string_field(root, "audio_data").empty())
        throw std::invalid_argument("audio_data (base64 WAV) is required");

    const std::string requested_model = string_field(root, "model", config.model);
    if (requested_model != config.model &&
        sa3::sat::canonical_saos_variant(requested_model) != config.model &&
        sa3::sat::canonical_sat_large_model(requested_model) != config.model)
        throw std::invalid_argument("this server is fixed to model " + config.model);
    if (has_field(root, "finetune_repo") || has_field(root, "finetune_checkpoint"))
        throw std::invalid_argument("finetunes are served by a SAOS server (--model saos)");

    if (config.family == Family::Foundation) {
        const bool randomize = bool_field(root, "randomize");
        const auto controls = randomize ? sa3::sat::randomize_foundation_controls(p.seed)
                                        : sa3::sat::FoundationRandomControls{};
        if (audio2audio && !has_field(root, "host_bpm"))
            throw std::invalid_argument("host_bpm is required for audio2audio");
        int bars = int_field(root, "bars", audio2audio ? 8 : randomize ? controls.bars : 4);
        double host_bpm = number_field(root, "host_bpm", 0);
        int bpm = int_field(root, "bpm", host_bpm > 0 ? nearest_foundation_bpm(host_bpm) :
                                      randomize ? controls.bpm : 120);
        if (!sa3::sat::is_foundation_bar_count(bars) || !sa3::sat::is_foundation_bpm(bpm))
            throw std::invalid_argument("Foundation-1 requires 4 or 8 bars and BPM 100, 110, 120, 128, 130, 140, or 150");
        if (has_field(root, "host_bpm") && (host_bpm < 20 || host_bpm > 999))
            throw std::invalid_argument("host_bpm must be between 20 and 999");
        if (!host_bpm) host_bpm = bpm;
        const std::string key_root = string_field(root, "key_root", randomize ? controls.key_root : "C");
        const std::string key_mode = string_field(root, "key_mode", randomize ? controls.key_mode : "minor");
        if (!valid_key_root(key_root) || (key_mode != "major" && key_mode != "minor"))
            throw std::invalid_argument("invalid Foundation key_root or key_mode");
        const std::string profile = string_field(root, "inference_profile", "gary");
        if (profile == "gary" || profile == "gary_fallback") sa3::sat::apply_foundation_gary_sampler(p);
        else if (profile == "royalcities") sa3::sat::apply_foundation_royalcities_sampler(p);
        else throw std::invalid_argument("inference_profile must be gary or royalcities");
        if (randomize) {
            if (!p.prompt.empty()) throw std::invalid_argument("prompt and randomize are mutually exclusive");
            const auto mode = sa3::sat::parse_foundation_prompt_mode(string_field(root, "randomize_mode", "standard"));
            p.prompt = sa3::sat::randomize_foundation_prompt(p.seed, mode,
                string_field(root, "family")).description;
        }
        const std::string custom = string_field(root, "custom_prompt_override");
        if (!custom.empty()) p.prompt = custom;
        if (p.prompt.empty()) p.prompt = foundation_descriptor(root);
        if (p.prompt.empty()) throw std::invalid_argument("Foundation prompt or randomize is required");
        const auto timing = sa3::sat::apply_foundation_timing(p, bars, bpm, false);
        p.prompt = foundation_prompt_with_tags(p.prompt, bars, bpm, key_root, key_mode);
        out.bars = bars;
        out.bpm = bpm;
        out.host_bpm = host_bpm;
        out.seconds = timing.output_seconds;
        out.stretched_samples = (int)std::llround(240.0 * bars * 44100.0 / host_bpm);
    } else {
        if (loop) throw std::invalid_argument("loop generation is supported for SAOS and Foundation-1");
        if (p.prompt.empty()) throw std::invalid_argument("prompt is required");
        const double seconds = number_field(root, "seconds", 11.0);
        if (seconds <= 0 || seconds > 47)
            throw std::invalid_argument("seconds is out of range for this model");
        p.seconds = (float)seconds;
        p.frames = (int)std::ceil(seconds * 44100 / 2048.0);
        p.output_samples = (int)std::llround(seconds * 44100);
        out.seconds = seconds;
    }

    if (has_field(root, "steps")) p.steps = int_field(root, "steps", p.steps);
    if (has_field(root, "cfg_scale")) p.cfg_scale = (float)number_field(root, "cfg_scale", p.cfg_scale);
    if (has_field(root, "guidance_scale")) p.cfg_scale = (float)number_field(root, "guidance_scale", p.cfg_scale);
    if (has_field(root, "sigma_min")) p.sigma_min = (float)number_field(root, "sigma_min", p.sigma_min);
    if (has_field(root, "sigma_max")) p.sigma_max = (float)number_field(root, "sigma_max", p.sigma_max);
    if (has_field(root, "sigma_rho")) p.sigma_rho = (float)number_field(root, "sigma_rho", p.sigma_rho);
    if (has_field(root, "rho")) p.sigma_rho = (float)number_field(root, "rho", p.sigma_rho);
    if (has_field(root, "sde_eta")) p.sde_eta = (float)number_field(root, "sde_eta", p.sde_eta);
    const std::string sampler = string_field(root, "sampler", string_field(root, "sampler_type"));
    if (!sampler.empty()) p.sampler = sa3::sat::parse_sampler(sampler);
    if (p.steps < 0 || p.steps == 1 || (p.cfg_scale < 0 && p.cfg_scale != -1.0f) ||
        p.sigma_rho <= 0 || p.sde_eta < 0)
        throw std::invalid_argument("invalid SAT sampler settings");

    if (audio2audio) {
        if (config.family != Family::Foundation)
            throw std::invalid_argument("audio2audio is a Foundation-1 route");
        const std::string bytes = b64_decode(string_field(root, "audio_data"));
        int samples = 0, channels = 0, rate = 0;
        std::vector<float> audio = sa3::read_wav_planar_bytes(
            (const uint8_t*)bytes.data(), bytes.size(), "audio_data", samples, channels, rate);
        // Played at the model's tempo, host audio lasts host_bpm / bpm times as long.
        const double ratio = out.host_bpm / out.bpm;
        // Only the canvas is encoded, so stretch exactly what it needs (and a
        // second spare): longer input is cropped, and shorter input is padded
        // with the silence the canvas would have added anyway, which also keeps
        // a one-bar recording long enough for the stretcher.
        const int needed = (int)std::ceil((double)p.frames * 2048 / 44100.0 * rate / ratio) + rate;
        if (samples != needed) {
            std::vector<float> fitted((size_t)needed * channels, 0.0f);
            for (int c = 0; c < channels; ++c)
                std::copy_n(audio.data() + (size_t)c * samples, std::min(samples, needed),
                            fitted.data() + (size_t)c * needed);
            audio = std::move(fitted);
            samples = needed;
        }
        if (std::abs(ratio - 1.0) >= 0.001) {
            const int stretched = std::max(1, (int)std::llround(samples * ratio));
            audio = sa3::sat::stretch_planar_exact(audio, channels, samples, stretched, rate);
            samples = stretched;
        }
        p.init_audio = std::move(audio);
        p.init_n_samp = samples;
        p.init_n_ch = channels;
        p.init_sample_rate = rate;
        p.init_noise_level = (float)std::clamp(number_field(root, "init_noise_level", 0.25), 0.01, 1.0);
        // Variations start at the init noise level and must descend from it.
        // The gary profile's sigma_min of 0.5 sits above the lower half of
        // gary4juce's transformation range, and k-diffusion's SDE samplers
        // return NaN on the rising schedule that makes (gary4local's Python
        // service did). Such requests use RoyalCities' 0.01 floor instead (or
        // half the level, below that); an explicit sigma_min at or above the
        // level is refused.
        if (p.sigma_min >= p.init_noise_level) {
            if (has_field(root, "sigma_min"))
                throw std::invalid_argument("sigma_min must be below init_noise_level");
            p.sigma_min = std::min(0.01f, p.init_noise_level / 2);
        }
        out.mode = "audio2audio";
    }
    return out;
}

bool resolve_paths(const Config& config, sa3::sat::PipelinePaths& paths, std::string& error) {
    return config.family == Family::Saos
        ? sa3::sat::resolve_saos_model(config.models_dir, config.model, config.encoding,
                                       config.t5_encoding, config.ae_encoding, &paths, &error)
        : sa3::sat::resolve_sat_large_model(config.models_dir, config.model, config.encoding,
                                            config.t5_encoding, config.ae_encoding, &paths, &error);
}

void prune_jobs() {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = g_jobs.begin(); it != g_jobs.end();) {
        if ((it->second.status == "completed" || it->second.status == "failed") &&
            now - it->second.created > std::chrono::minutes(10)) it = g_jobs.erase(it);
        else ++it;
    }
}

void launch(const Config& config, const std::string& sid, Prepared prepared) {
    std::thread([config, sid, prepared = std::move(prepared)]() mutable {
        auto set_job = [&](auto&& callback) {
            std::lock_guard<std::mutex> guard(g_mutex);
            if (auto it = g_jobs.find(sid); it != g_jobs.end()) callback(it->second);
        };
        try {
            sa3::sat::PipelinePaths paths;
            std::string error;
            if (!resolve_paths(config, paths, error)) throw std::runtime_error(error);
            // A newly downloaded encoder changes the paths; rebuild to pick it up.
            if (!g_pipeline || g_pipeline->paths().encoder != paths.encoder) {
                sa3::sat::PipelineOptions options;
                options.device = config.device;
                auto pipeline = std::make_unique<sa3::sat::Pipeline>(std::move(paths), options);
                std::lock_guard<std::mutex> guard(g_mutex);
                g_pipeline = std::move(pipeline);
            }
            auto& p = prepared.params;
            p.stage = [&](sa3::sat::PipelineStage stage) {
                set_job([&](Job& job) {
                    job.status = stage == sa3::sat::PipelineStage::Loading ? "loading" :
                                 stage == sa3::sat::PipelineStage::Encoding ? "encoding" :
                                 stage == sa3::sat::PipelineStage::Sampling ? "generating" : "decoding";
                    if (stage == sa3::sat::PipelineStage::Decoding) job.progress = 95;
                });
            };
            p.progress = [&](const sa3::sat::StepProgress& step) {
                set_job([&](Job& job) {
                    job.status = "generating";
                    job.step = step.step;
                    job.total_steps = step.steps;
                    job.progress = step.steps > 0 ? 10 + 80 * step.step / step.steps : 10;
                });
            };
            auto result = g_pipeline->generate(p);
            if (prepared.stretched_samples > 0 && prepared.stretched_samples != result.samples) {
                set_job([&](Job& job) { job.status = "stretching"; job.progress = 97; });
                result.audio = sa3::sat::stretch_planar_exact(result.audio, result.channels,
                    result.samples, prepared.stretched_samples, result.sample_rate);
                result.samples = prepared.stretched_samples;
            }
            std::string wav = sa3::wav_planar_bytes(result.audio.data(), result.samples,
                                                     result.channels, result.sample_rate);
            std::string audio = b64_encode(wav);
            set_job([&](Job& job) {
                job.audio = std::move(audio);
                job.status = "completed";
                job.progress = 100;
                job.seconds = (double)result.samples / result.sample_rate;
            });
        } catch (const std::exception& e) {
            set_job([&](Job& job) { job.status = "failed"; job.error = e.what(); });
            std::fprintf(stderr, "[sat-server] job %s failed: %s\n", sid.c_str(), e.what());
        }
        std::lock_guard<std::mutex> guard(g_mutex);
        g_busy = false;
    }).detach();
}

// ---- SAOS ----
// A SAOS server keeps gary4local's Python stable-audio ("jerry") contract. It
// serves every variant installed at its encoding (the base model and the
// published finetunes, which share its T5 and autoencoder) and keeps one DiT
// resident. A request picks a variant as clients picked a checkpoint before:
// model_type "standard", or "finetune" with finetune_repo and
// finetune_checkpoint. Generation is synchronous and answers
// {success, audio_base64, metadata}, or the WAV itself for return_format "file".

using sa3::sat::SaosVariant;

constexpr int kSaosFrames = 256;
constexpr int kSaosCanvasSamples = kSaosFrames * 2048;
constexpr int kSaosSecondsMax = kSaosCanvasSamples / 44100;  // Python's default seconds_total

std::mutex g_generate_mutex;           // one synchronous generation at a time
std::vector<std::string> g_saos_used;  // variant names, most recently used last

bool is_finetune(const SaosVariant& variant) { return variant.checkpoint[0] != '\0'; }

// The cache key and source Python reported for this checkpoint; gary4juce
// stores them as the selection.
std::string saos_key(const SaosVariant& variant) {
    if (!is_finetune(variant)) return "standard_saos";
    auto flat = [](std::string text, bool dots) {
        for (char& c : text) if (c == '/' || (dots && c == '.')) c = '_';
        return text;
    };
    return "finetune_" + flat(variant.repo, false) + "_" + flat(variant.checkpoint, true);
}

std::string saos_source(const SaosVariant& variant) {
    return is_finetune(variant) ? std::string(variant.repo) + "/" + variant.checkpoint : variant.repo;
}

std::string device_name(const Config& config) { return config.device.empty() ? "auto" : config.device; }

bool resolve_saos(const Config& config, const SaosVariant& variant,
                  sa3::sat::PipelinePaths& paths, std::string& error) {
    return sa3::sat::resolve_saos_model(config.models_dir, variant.name, config.encoding,
                                        config.t5_encoding, config.ae_encoding, &paths, &error);
}

std::vector<const SaosVariant*> installed_saos(const Config& config) {
    std::vector<const SaosVariant*> installed;
    for (const auto& variant : sa3::sat::kSaosVariants) {
        sa3::sat::PipelinePaths paths;
        std::string error;
        if (resolve_saos(config, variant, paths, error)) installed.push_back(&variant);
    }
    return installed;
}

// Installed variants, least recently used first, as Python's usage_order: the
// last is the active one (the server's --model until a request picks another).
std::vector<const SaosVariant*> saos_usage_order(const Config& config,
                                                 std::vector<const SaosVariant*> order) {
    auto bump = [&](const std::string& name) {
        auto it = std::find_if(order.begin(), order.end(),
                               [&](const SaosVariant* variant) { return name == variant->name; });
        if (it != order.end()) std::rotate(it, it + 1, order.end());
    };
    bump(config.model);
    std::vector<std::string> used;
    { std::lock_guard<std::mutex> guard(g_mutex); used = g_saos_used; }
    for (const auto& name : used) bump(name);
    return order;
}

const SaosVariant& requested_saos_variant(const Config& config, yyjson_val* root) {
    const std::string type = string_field(root, "model_type");
    const std::string repo = string_field(root, "finetune_repo");
    const std::string checkpoint = string_field(root, "finetune_checkpoint");
    if (type.empty() && repo.empty() && checkpoint.empty()) {
        const std::string name = sa3::sat::canonical_saos_variant(string_field(root, "model", config.model));
        for (const auto& variant : sa3::sat::kSaosVariants)
            if (name == variant.name) return variant;
        throw std::invalid_argument("unknown SAOS model " + name + " (arc, kickbass, or jerry-grunge)");
    }
    if (type == "standard") return sa3::sat::kSaosVariants[0];
    if (!type.empty() && type != "finetune")
        throw std::invalid_argument("model_type must be 'standard' or 'finetune'");
    if (repo.empty() || checkpoint.empty())
        throw std::invalid_argument("finetune_repo and finetune_checkpoint required for finetune models");
    std::string published;
    for (const auto& variant : sa3::sat::kSaosVariants) {
        if (!is_finetune(variant)) continue;
        if (repo == variant.repo && checkpoint == variant.checkpoint) return variant;
        published += (published.empty() ? "" : ", ") + saos_source(variant);
    }
    throw std::invalid_argument("no native SAOS finetune for " + repo + "/" + checkpoint +
                                " (published: " + published + ")");
}

struct SaosRequest {
    const SaosVariant* variant = nullptr;
    sa3::sat::GenerateParams params;
    bool loop = false;
    bool base64 = false;
    std::string original_prompt;
    std::string loop_type;
    double bpm = 0;  // from the prompt's tag (or a bpm field); 0 when there is none
    int bars = 0;
    double seconds_per_bar = 0;
    double calculated_seconds = 0;  // the bars at that BPM, before the canvas crop
};

// Python's request shape, defaults and validation. Steps and CFG default per
// model objective, as Python's /generate did (8 and 1 for the base model, 50
// and 4 for the rectified-flow finetunes); gary4juce always sends both.
SaosRequest prepare_saos(const Config& config, yyjson_val* root, bool loop) {
    SaosRequest out;
    out.loop = loop;
    out.variant = &requested_saos_variant(config, root);
    auto& p = out.params;
    p.prompt = string_field(root, "prompt");
    if (p.prompt.empty()) throw std::invalid_argument("prompt is required");
    out.original_prompt = p.prompt;
    p.negative_prompt = string_field(root, "negative_prompt");
    p.seed = seed_field(root);
    if (seed_drawn(root)) p.seed &= 0xffffffffu;  // Python drew seeds below 2^32
    const std::string format = string_field(root, "return_format", "file");
    if (format != "file" && format != "base64")
        throw std::invalid_argument("return_format must be 'file' or 'base64'");
    out.base64 = format == "base64";
    p.keep_models = true;
    p.frames = kSaosFrames;
    // Python scaled the decoded audio to a full-scale peak and clamped it.
    p.loudness.peak_normalize_enabled = true;
    p.loudness.peak_normalize_db = 0.0f;
    p.loudness.limiter_enabled = false;
    out.bpm = number_field(root, "bpm", prompt_bpm(p.prompt));

    if (loop) {
        if (out.bpm <= 0) throw std::invalid_argument("BPM must be specified in prompt (e.g., '120bpm')");
        if (out.bpm < 20 || out.bpm > 300) throw std::invalid_argument("loop BPM must be between 20 and 300");
        out.seconds_per_bar = 240.0 / out.bpm;
        const double canvas_seconds = (double)kSaosCanvasSamples / 44100;
        int bars = int_field(root, "bars", 0);
        if (bars) {
            if (bars != 1 && bars != 2 && bars != 4 && bars != 8)
                throw std::invalid_argument("bars must be 1, 2, 4, or 8");
            if (out.seconds_per_bar * bars > canvas_seconds)
                throw std::invalid_argument("requested loop exceeds the SAOS canvas");
        } else {
            // The longest count leaving a second of the canvas spare, as Python
            // chose; a single bar longer than the canvas is cropped to it.
            bars = 1;
            for (int candidate : {8, 4, 2, 1})
                if (out.seconds_per_bar * candidate <= canvas_seconds - 1.0) { bars = candidate; break; }
        }
        out.bars = bars;
        out.calculated_seconds = out.seconds_per_bar * bars;
        if (!prompt_bpm(p.prompt)) p.prompt += ", " + number(out.bpm) + " bpm";
        out.loop_type = string_field(root, "loop_type", "auto");
        std::string lower = p.prompt;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        if (out.loop_type == "drums") {
            if (lower.find("drum") == std::string::npos) p.prompt += " drum loop";
            p.negative_prompt = "melody, harmony, pitched instruments, vocals, singing";
        } else if (out.loop_type == "instruments") {
            for (const std::string word : {"drums", "drum"})
                for (size_t at; (at = p.prompt.find(word)) != std::string::npos;) p.prompt.erase(at, word.size());
            p.prompt.erase(0, p.prompt.find_first_not_of(" ,"));
            p.prompt.erase(p.prompt.find_last_not_of(" ,") + 1);
            p.negative_prompt = "drums, percussion, kick, snare, hi-hat";
        } else if (out.loop_type != "auto") {
            throw std::invalid_argument("loop_type must be auto, drums, or instruments");
        }
        p.seconds = p.seconds_total = (float)kSaosSecondsMax;
        p.output_samples = std::min((int)std::floor(out.calculated_seconds * 44100), kSaosCanvasSamples);
    } else {
        // Whole seconds, clamped to the canvas, as Python's seconds_total.
        const double requested = number_field(root, "seconds_total", kSaosSecondsMax);
        if (requested < 1) throw std::invalid_argument("seconds_total must be >= 1");
        const int seconds = std::min((int)std::floor(requested), kSaosSecondsMax);
        p.seconds = p.seconds_total = (float)seconds;
        p.output_samples = seconds * 44100;
    }

    if (has_field(root, "steps")) {
        p.steps = int_field(root, "steps", 0);
        if (p.steps < 1 || p.steps > 250) throw std::invalid_argument("steps must be integer between 1-250");
    }
    if (has_field(root, "cfg_scale")) {
        p.cfg_scale = (float)number_field(root, "cfg_scale", 1.0);
        if (p.cfg_scale < 0 || p.cfg_scale > 20) throw std::invalid_argument("cfg_scale must be number between 0-20");
    }
    // gary4juce sends pingpong for the base model, and euler or dpmpp for a finetune.
    const std::string sampler = string_field(root, "sampler_type", string_field(root, "sampler"));
    if (!sampler.empty()) {
        if (sampler != "euler" && sampler != "dpmpp" && sampler != "pingpong")
            throw std::invalid_argument("sampler_type must be euler, dpmpp, or pingpong");
        p.sampler = sa3::sat::parse_sampler(sampler);
    }
    return out;
}

// Variants share the T5 and autoencoder, so switching between them swaps only
// the DiT.
sa3::sat::GenerateResult generate_saos(const Config& config, const SaosRequest& request,
                                       sa3::sat::PipelinePaths paths) {
    std::lock_guard<std::mutex> one_at_a_time(g_generate_mutex);
    {
        std::lock_guard<std::mutex> guard(g_mutex);
        g_busy = true;
        if (g_pipeline && g_pipeline->paths().t5 == paths.t5 &&
            g_pipeline->paths().autoencoder == paths.autoencoder) {
            g_pipeline->set_dit(paths.dit);
        } else {
            sa3::sat::PipelineOptions options;
            options.device = config.device;
            g_pipeline = std::make_unique<sa3::sat::Pipeline>(std::move(paths), options);
        }
    }
    struct Idle {
        ~Idle() { std::lock_guard<std::mutex> guard(g_mutex); g_busy = false; }
    } idle;
    auto result = g_pipeline->generate(request.params);
    std::lock_guard<std::mutex> guard(g_mutex);
    const std::string name = request.variant->name;
    g_saos_used.erase(std::remove(g_saos_used.begin(), g_saos_used.end(), name), g_saos_used.end());
    g_saos_used.push_back(name);
    return result;
}

// Python's metadata for each route, plus the native variant, sampler and model key.
std::string saos_metadata(const Config& config, const SaosRequest& request,
                          const sa3::sat::GenerateResult& result, double generation_time) {
    auto round2 = [](double value) { return number(std::round(value * 100) / 100); };
    const double duration = (double)result.samples / result.sample_rate;
    const auto& p = request.params;
    std::string body = "{\"prompt\":\"" + escape(p.prompt) + "\"";
    if (request.loop) {
        body += ",\"original_prompt\":\"" + escape(request.original_prompt) +
            "\",\"negative_prompt\":\"" + escape(p.negative_prompt) +
            "\",\"loop_type\":\"" + escape(request.loop_type) +
            "\",\"detected_bpm\":" + number(request.bpm) +
            ",\"bars\":" + std::to_string(request.bars) +
            ",\"loop_duration_seconds\":" + round2(duration) +
            ",\"calculated_duration_seconds\":" + round2(request.calculated_seconds) +
            ",\"available_audio_seconds\":" + round2((double)kSaosCanvasSamples / 44100) +
            ",\"seconds_per_bar\":" + round2(request.seconds_per_bar) +
            ",\"style_transfer\":false,\"style_strength\":null" +
            ",\"model_type\":\"" + (is_finetune(*request.variant) ? "finetune" : "standard") + "\"";
    } else {
        body += ",\"negative_prompt\":" + (p.negative_prompt.empty() ? std::string("null")
                                           : "\"" + escape(p.negative_prompt) + "\"") +
            ",\"duration_seconds\":" + number(p.seconds_total) +
            ",\"detected_bpm\":" + (request.bpm > 0 ? number(request.bpm) : std::string("null"));
    }
    return body + ",\"steps\":" + std::to_string(result.steps) +
        ",\"cfg_scale\":" + number(result.cfg_scale) +
        ",\"seed\":" + std::to_string(p.seed) +
        ",\"sample_rate\":" + std::to_string(result.sample_rate) +
        ",\"generation_time\":" + round2(generation_time) +
        ",\"realtime_factor\":" + round2(duration / std::max(generation_time, 1e-6)) +
        ",\"device\":\"" + escape(device_name(config)) +
        "\",\"model\":\"" + request.variant->name +
        "\",\"model_key\":\"" + escape(saos_key(*request.variant)) +
        "\",\"sampler\":\"" + sa3::sat::sampler_name(result.sampler) + "\"}";
}

// GET /models/status: every installed variant, in Python's shape.
std::string saos_status_json(const Config& config) {
    const auto installed = installed_saos(config);
    std::string loaded, usage, details;
    for (const SaosVariant* variant : installed) {
        const std::string key = "\"" + escape(saos_key(*variant)) + "\"";
        loaded += (loaded.empty() ? "" : ",") + key;
        details += (details.empty() ? "" : ",") + key + ":{\"type\":\"" +
            (is_finetune(*variant) ? "finetune" : "standard") +
            "\",\"source\":\"" + escape(saos_source(*variant)) +
            "\",\"model\":\"" + variant->name +
            "\",\"encoding\":\"" + escape(sa3::sat::normalize_encoding(config.encoding)) +
            "\",\"device\":\"" + escape(device_name(config)) +
            "\",\"sample_rate\":44100,\"sample_size\":" + std::to_string(kSaosCanvasSamples) + "}";
    }
    for (const SaosVariant* variant : saos_usage_order(config, installed))
        usage += (usage.empty() ? "" : ",") + ("\"" + escape(saos_key(*variant)) + "\"");
    const std::string count = std::to_string(installed.size());
    return "{\"cache_status\":{\"loaded_models\":[" + loaded + "],\"usage_order\":[" + usage +
        "],\"cache_utilization\":\"" + count + "/" + count + "\"},\"model_details\":{" + details +
        "},\"max_models\":" + count + "}";
}

// A finetune's prompt bank as compact JSON, or Python's empty bank.
std::string saos_prompt_bank(const Config& config, const SaosVariant& variant) {
    static const std::string empty = R"({"version":1,"dice":{"generic":[],"drums":[],"instrumental":[]}})";
    const std::string relative = sa3::sat::saos_prompts_relative_path(variant.name);
    if (relative.empty()) return empty;
    const std::filesystem::path path = std::filesystem::path(config.models_dir) / relative;
    yyjson_read_err error{};
    yyjson_doc* doc = yyjson_read_file(path.string().c_str(), 0, nullptr, &error);
    if (!doc) {
        if (std::filesystem::exists(path))
            std::fprintf(stderr, "[sat-server] ignoring %s: %s\n", path.string().c_str(), error.msg);
        return empty;
    }
    std::string bank = empty;
    if (yyjson_is_obj(yyjson_doc_get_root(doc))) {
        if (char* text = yyjson_write(doc, 0, nullptr)) { bank = text; std::free(text); }
    }
    yyjson_doc_free(doc);
    return bank;
}

// GET /models/prompts: the bank for ?key=, or ?repo= (and ?checkpoint=), or
// the active variant's, chosen as Python chose.
std::string saos_prompts_json(const Config& config, const httplib::Request& req) {
    const auto order = saos_usage_order(config, installed_saos(config));
    const std::string key = req.get_param_value("key");
    const std::string repo = req.get_param_value("repo");
    const std::string checkpoint = req.get_param_value("checkpoint");
    std::string prefer = req.has_param("prefer") ? req.get_param_value("prefer") : "active";
    std::transform(prefer.begin(), prefer.end(), prefer.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    auto most_recent = [&](auto&& matches) -> const SaosVariant* {
        for (auto it = order.rbegin(); it != order.rend(); ++it) if (matches(**it)) return *it;
        return nullptr;
    };
    const SaosVariant* selected = nullptr;
    if (!key.empty()) {
        selected = most_recent([&](const SaosVariant& v) { return saos_key(v) == key; });
    } else if (!repo.empty()) {
        selected = most_recent([&](const SaosVariant& v) {
            return is_finetune(v) && repo == v.repo && (checkpoint.empty() || checkpoint == v.checkpoint);
        });
    } else {
        if (prefer == "finetune") selected = most_recent([](const SaosVariant& v) { return is_finetune(v); });
        if (!selected && !order.empty()) selected = order.back();
    }
    if (!selected) return R"({"success":true,"prompts":null,"message":"no model selected"})";
    const bool finetune = is_finetune(*selected);
    return "{\"success\":true,\"model_key\":\"" + escape(saos_key(*selected)) +
        "\",\"type\":\"" + (finetune ? "finetune" : "standard") +
        "\",\"source\":\"" + escape(finetune ? selected->repo : saos_source(*selected)) +
        "\",\"checkpoint\":" + (finetune ? "\"" + escape(selected->checkpoint) + "\"" : std::string("null")) +
        ",\"model\":\"" + selected->name +
        "\",\"prompts\":" + saos_prompt_bank(config, *selected) + "}";
}

void serve_saos(const Config& config, const httplib::Request& req, httplib::Response& res, bool loop) {
    yyjson_doc* doc = yyjson_read(req.body.data(), req.body.size(), 0);
    if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
        if (doc) yyjson_doc_free(doc);
        fail(res, 400, "JSON body required");
        return;
    }
    SaosRequest request;
    try {
        request = prepare_saos(config, yyjson_doc_get_root(doc), loop);
    } catch (const std::exception& e) {
        yyjson_doc_free(doc);
        fail(res, 400, e.what());
        return;
    }
    yyjson_doc_free(doc);
    sa3::sat::PipelinePaths paths;
    std::string error;
    if (!resolve_saos(config, *request.variant, paths, error)) { fail(res, 503, error); return; }
    try {
        const auto start = std::chrono::steady_clock::now();
        const auto result = generate_saos(config, request, std::move(paths));
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::string wav = sa3::wav_planar_bytes(result.audio.data(), result.samples, result.channels,
                                                result.sample_rate);
        const std::string metadata = saos_metadata(config, request, result, elapsed);
        std::fprintf(stderr, "[sat-server] %s %s in %.2fs\n", request.variant->name,
                     loop ? "loop" : "generation", elapsed);
        if (request.base64) {
            res.set_content("{\"success\":true,\"audio_base64\":\"" + b64_encode(wav) +
                            "\",\"metadata\":" + metadata + "}", "application/json");
        } else {
            const std::string seed = std::to_string(request.params.seed);
            const std::string name = loop
                ? "loop_" + request.loop_type + "_" + number(request.bpm) + "bpm_" +
                  std::to_string(request.bars) + "bars_" + seed + ".wav"
                : "stable_audio_" + seed + ".wav";
            res.set_header("Content-Disposition", "attachment; filename=\"" + name + "\"");
            res.set_content(std::move(wav), "audio/wav");
        }
    } catch (const std::invalid_argument& e) {
        fail(res, 400, e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[sat-server] %s failed: %s\n", request.variant->name, e.what());
        fail(res, 500, e.what());
    }
}

// A SAOS server is ready with any installed variant; it reports what its
// --model is missing when there are none.
bool service_ready(const Config& config, std::string& error) {
    if (config.family == Family::Saos && !installed_saos(config).empty()) return true;
    sa3::sat::PipelinePaths paths;
    return resolve_paths(config, paths, error);
}

Config parse_args(int argc, char** argv) {
    Config config;
    if (const char* value = std::getenv("SA3_MODELS_DIR")) config.models_dir = value;
    if (const char* value = std::getenv("SA3_PORT")) config.port = std::stoi(value);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--help") {
            std::puts("sat-server --model foundation-1|saos|sao1 [--port N] [--models-dir DIR] [--encoding TIER] [--t5-encoding TIER] [--ae-encoding TIER] [--device DEVICE]\n"
                      "  A SAOS server serves arc and the installed finetunes (kickbass, jerry-grunge);\n"
                      "  --model arc|kickbass|jerry-grunge also starts one, active on that variant.");
            std::exit(0);
        } else if (arg == "--model") config.model = next();
        else if (arg == "--models-dir") config.models_dir = next();
        else if (arg == "--encoding") config.encoding = next();
        else if (arg == "--t5-encoding") config.t5_encoding = next();
        else if (arg == "--ae-encoding") config.ae_encoding = next();
        else if (arg == "--device") config.device = next();
        else if (arg == "--host") config.host = next();
        else if (arg == "--port") config.port = std::stoi(next());
        else throw std::invalid_argument("unknown option " + arg);
    }
    config.model = sa3::sat::canonical_saos_variant(config.model);
    if (config.model == "arc" || config.model == "kickbass" || config.model == "jerry-grunge")
        config.family = Family::Saos;
    else {
        config.model = sa3::sat::canonical_sat_large_model(config.model);
        if (config.model == "foundation-1") config.family = Family::Foundation;
        else if (config.model == "stable-audio-open-1.0") config.family = Family::Sao1;
        else throw std::invalid_argument("unsupported SAT server model " + config.model);
    }
    if (config.t5_encoding.empty()) config.t5_encoding = config.encoding;
    if (config.ae_encoding.empty()) config.ae_encoding = config.encoding;
    if (config.port < 1 || config.port > 65535) throw std::invalid_argument("invalid port");
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--version") {
                std::puts(sa3::runtime_version());
                return 0;
            }
            if (std::string(argv[i]) == "--props") {
                std::puts(sa3::runtime_props_json("sat").c_str());
                return 0;
            }
        }
        const Config config = parse_args(argc, argv);
        httplib::Server server;
        // audio2audio carries a base64 WAV; gary4juce sends up to 240 seconds.
        server.set_payload_max_length((size_t)192 * 1024 * 1024);
        server.Get("/props", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(sa3::runtime_props_json("sat"), "application/json");
        });
        server.Get("/health", [config](const httplib::Request&, httplib::Response& res) {
            std::string error;
            const bool ready = service_ready(config, error);
            bool loaded;
            { std::lock_guard<std::mutex> guard(g_mutex); loaded = (bool)g_pipeline; }
            const std::string family = config.family == Family::Foundation ? "foundation-1" :
                                       config.family == Family::Saos ? "saos" : "sao1";
            std::string body = "{\"status\":\"" + std::string(ready ? "healthy" : "model_missing") +
                "\",\"ready\":" + (ready ? "true" : "false") +
                ",\"model_loaded\":" + (loaded ? "true" : "false") +
                ",\"model\":\"" + config.model + "\",\"family\":\"" + family + "\"";
            if (config.family == Family::Saos) {
                std::vector<std::string> names;
                for (const SaosVariant* variant : installed_saos(config)) names.push_back(variant->name);
                body += ",\"models\":" + json_strings(names);
            }
            if (!ready) body += ",\"error\":\"" + escape(error) + "\"";
            res.set_content(body + "}", "application/json");
            if (!ready) res.status = 503;
        });
        server.Get("/ready", [config](const httplib::Request&, httplib::Response& res) {
            std::string error;
            const bool ready = service_ready(config, error);
            res.set_content(std::string("{\"ready\":") + (ready ? "true}" : "false}"), "application/json");
            if (!ready) res.status = 503;
        });
        auto generate = [config](const httplib::Request& req, httplib::Response& res, bool loop,
                                 bool audio2audio) {
            yyjson_doc* doc = yyjson_read(req.body.data(), req.body.size(), 0);
            if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
                if (doc) yyjson_doc_free(doc);
                fail(res, 400, "JSON object required");
                return;
            }
            try {
                Prepared prepared = prepare(config, yyjson_doc_get_root(doc), loop, audio2audio);
                yyjson_doc_free(doc);
                doc = nullptr;
                sa3::sat::PipelinePaths paths;
                std::string error;
                if (!resolve_paths(config, paths, error)) { fail(res, 503, error); return; }
                if (audio2audio && paths.encoder.empty()) {
                    fail(res, 503, "audio2audio needs " + sa3::sat::sat_oobleck_encoder_relative_path() +
                                   " in " + config.models_dir);
                    return;
                }
                const std::string sid = session_id();
                const uint64_t seed = prepared.params.seed;
                const int bars = prepared.bars;
                const double bpm = prepared.bpm;
                const double host_bpm = prepared.host_bpm;
                const double seconds = prepared.seconds;
                const std::string prompt = prepared.params.prompt;
                const std::string mode = prepared.mode;
                const double init_noise_level = prepared.params.init_noise_level;
                const double sigma_min = prepared.params.sigma_min;
                {
                    std::lock_guard<std::mutex> guard(g_mutex);
                    if (g_busy) { fail(res, 409, "generation already in progress"); return; }
                    prune_jobs();
                    g_busy = true;
                    Job job;
                    job.model = config.model;
                    job.prompt = prompt;
                    job.seed = seed;
                    job.bars = bars;
                    job.bpm = bpm;
                    job.host_bpm = host_bpm;
                    job.seconds = seconds;
                    job.mode = mode;
                    if (audio2audio) job.init_noise_level = init_noise_level;
                    g_jobs.emplace(sid, std::move(job));
                }
                try { launch(config, sid, std::move(prepared)); }
                catch (...) {
                    std::lock_guard<std::mutex> guard(g_mutex);
                    g_busy = false;
                    g_jobs.erase(sid);
                    throw;
                }
                res.set_content("{\"success\":true,\"session_id\":\"" + sid +
                    "\",\"seed\":" + std::to_string(seed) +
                    ",\"model\":\"" + config.model + "\",\"prompt\":\"" + escape(prompt) +
                    "\",\"bars\":" + std::to_string(bars) + ",\"bpm\":" + number(bpm) +
                    ",\"foundation_bpm\":" + number(bpm) +
                    ",\"host_bpm\":" + number(host_bpm) +
                    ",\"stretch_ratio\":" + number(host_bpm > 0 ? host_bpm / bpm : 1) +
                    ",\"gen_duration\":" + number(seconds) +
                    (audio2audio ? ",\"mode\":\"audio2audio\",\"init_noise_level\":" +
                                       number(init_noise_level) + ",\"sigma_min\":" + number(sigma_min) +
                                       ",\"sigma_max\":" + number(init_noise_level) : std::string()) +
                    "}", "application/json");
            } catch (const std::exception& e) {
                if (doc) yyjson_doc_free(doc);
                fail(res, 400, e.what());
            }
        };
        // SAOS answers synchronously, in the Python stable-audio contract.
        server.Post("/generate", [config, generate](const httplib::Request& req, httplib::Response& res) {
            if (config.family == Family::Saos) serve_saos(config, req, res, false);
            else generate(req, res, false, false);
        });
        server.Post("/generate/loop", [config, generate](const httplib::Request& req, httplib::Response& res) {
            if (config.family == Family::Saos) serve_saos(config, req, res, true);
            else generate(req, res, true, false);
        });
        if (config.family == Family::Saos) {
            server.Get("/models/status", [config](const httplib::Request&, httplib::Response& res) {
                res.set_content(saos_status_json(config), "application/json");
            });
            server.Get("/models/prompts", [config](const httplib::Request& req, httplib::Response& res) {
                res.set_content(saos_prompts_json(config, req), "application/json");
            });
        }
        server.Post("/audio2audio", [generate](const httplib::Request& req, httplib::Response& res) {
            generate(req, res, false, true);
        });
        server.Post("/randomize", [config](const httplib::Request& req, httplib::Response& res) {
            if (config.family != Family::Foundation) {
                fail(res, 404, "randomize is a Foundation-1 route");
                return;
            }
            // An empty body asks for a random preset.
            const std::string body = req.body.empty() ? "{}" : req.body;
            yyjson_doc* doc = yyjson_read(body.data(), body.size(), 0);
            if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
                if (doc) yyjson_doc_free(doc);
                fail(res, 400, "JSON object required");
                return;
            }
            try {
                res.set_content(foundation_randomize_json(yyjson_doc_get_root(doc)), "application/json");
            } catch (const std::exception& e) {
                fail(res, 400, e.what());
            }
            yyjson_doc_free(doc);
        });
        server.Get(R"(/poll_status/([a-f0-9]+))", [](const httplib::Request& req, httplib::Response& res) {
            const std::string sid = req.matches[1];
            Job job;
            {
                std::lock_guard<std::mutex> guard(g_mutex);
                prune_jobs();
                auto it = g_jobs.find(sid);
                if (it == g_jobs.end()) { fail(res, 404, "unknown session: " + sid); return; }
                job = it->second;
                if (job.status == "completed" && req.has_param("consume") &&
                    req.get_param_value("consume") != "0") g_jobs.erase(it);
            }
            const bool active = job.status != "completed" && job.status != "failed";
            std::string body = "{\"success\":" + std::string(job.status == "failed" ? "false" : "true") +
                ",\"generation_in_progress\":" + (active ? "true" : "false") +
                ",\"transform_in_progress\":" + (job.status == "stretching" ? "true" : "false") +
                ",\"status\":\"" + job.status +
                "\",\"progress\":" + std::to_string(job.progress) +
                ",\"step\":" + std::to_string(job.step) +
                ",\"total_steps\":" + std::to_string(job.total_steps) +
                ",\"queue_status\":" + (job.status == "queued"
                  ? "{\"status\":\"queued\",\"position\":1}" : active
                  ? "{\"status\":\"ready\"}" : "{}") +
                ",\"meta\":{\"seed\":" + std::to_string(job.seed) +
                ",\"model\":\"" + job.model + "\",\"prompt\":\"" + escape(job.prompt) +
                "\",\"bars\":" + std::to_string(job.bars) + ",\"bpm\":" + number(job.bpm) +
                ",\"foundation_bpm\":" + number(job.bpm) +
                ",\"host_bpm\":" + number(job.host_bpm) +
                ",\"final_duration\":" + number(job.seconds) +
                (job.mode.empty() ? std::string() : ",\"mode\":\"" + job.mode +
                    "\",\"init_noise_level\":" + number(job.init_noise_level)) + "}";
            if (job.status == "completed") body += ",\"audio_data\":\"" + job.audio + "\"";
            if (job.status == "failed") body += ",\"error\":\"" + escape(job.error) + "\"";
            res.set_content(body + "}", "application/json");
        });
        server.Post("/unload", [](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> guard(g_mutex);
            if (g_busy) { fail(res, 409, "generation in progress"); return; }
            if (g_pipeline) g_pipeline->unload();
            g_pipeline.reset();
            res.set_content("{\"status\":\"unloaded\"}", "application/json");
        });
        std::fprintf(stderr, "[sat-server] http://%s:%d model=%s encoding=%s models=%s\n",
                     config.host.c_str(), config.port, config.model.c_str(), config.encoding.c_str(),
                     config.models_dir.c_str());
        if (!server.listen(config.host.c_str(), config.port))
            throw std::runtime_error("cannot bind SAT server port");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sat-server: %s\n", e.what());
        return 1;
    }
}
