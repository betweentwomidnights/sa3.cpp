// A model-scoped HTTP service for Stable Audio Tools checkpoints. Run separate
// instances for Foundation-1 and SAOS; both use the same packaged runtime.
#include "sat/foundation_prompt.h"
#include "sat/model_paths.h"
#include "sat/pipeline.h"
#include "sat/profiles.h"
#include "sat/time_stretch.h"
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

struct Prepared {
    sa3::sat::GenerateParams params;
    int bars = 0;
    double bpm = 0;
    double host_bpm = 0;
    double seconds = 0;
    int stretched_samples = 0;
};

Prepared prepare(const Config& config, yyjson_val* root, bool loop) {
    Prepared out;
    auto& p = out.params;
    p.prompt = string_field(root, "prompt");
    p.negative_prompt = string_field(root, "negative_prompt");
    p.seed = seed_field(root);
    p.keep_models = true;

    const std::string requested_model = string_field(root, "model", config.model);
    if (requested_model != config.model &&
        sa3::sat::canonical_saos_variant(requested_model) != config.model &&
        sa3::sat::canonical_sat_large_model(requested_model) != config.model)
        throw std::invalid_argument("this server is fixed to model " + config.model);
    if (has_field(root, "finetune_repo") || has_field(root, "finetune_checkpoint"))
        throw std::invalid_argument("start a server with --model kickbass or jerry-grunge for that finetune");

    if (config.family == Family::Foundation) {
        const bool randomize = bool_field(root, "randomize");
        const auto controls = randomize ? sa3::sat::randomize_foundation_controls(p.seed)
                                        : sa3::sat::FoundationRandomControls{};
        int bars = int_field(root, "bars", randomize ? controls.bars : 4);
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
        if (loop && config.family != Family::Saos)
            throw std::invalid_argument("loop generation is supported for SAOS and Foundation-1");
        if (p.prompt.empty()) throw std::invalid_argument("prompt is required");
        if (loop) {
            const double bpm = number_field(root, "bpm", prompt_bpm(p.prompt));
            if (bpm < 20 || bpm > 300) throw std::invalid_argument("loop BPM must be between 20 and 300");
            int bars = int_field(root, "bars", 0);
            if (!bars) {
                for (int candidate : {8, 4, 2, 1})
                    if (240.0 * candidate / bpm <= 11.0) { bars = candidate; break; }
            }
            if (bars != 1 && bars != 2 && bars != 4 && bars != 8)
                throw std::invalid_argument("loop bars must be 1, 2, 4, or 8");
            const double seconds = 240.0 * bars / bpm;
            if (seconds > 11.0) throw std::invalid_argument("requested loop exceeds SAOS 11 second canvas");
            if (!prompt_bpm(p.prompt)) p.prompt += ", " + number(bpm) + " bpm";
            const std::string type = string_field(root, "loop_type", "auto");
            if (type == "drums") {
                if (p.prompt.find("drum") == std::string::npos) p.prompt += ", drum loop";
                if (p.negative_prompt.empty()) p.negative_prompt = "melody, harmony, pitched instruments, vocals, singing";
            } else if (type == "instruments") {
                if (p.negative_prompt.empty()) p.negative_prompt = "drums, percussion, kick, snare, hi-hat";
            } else if (type != "auto") throw std::invalid_argument("loop_type must be auto, drums, or instruments");
            p.seconds = 11.0f;
            p.seconds_total = 11.0f;
            p.frames = 256;
            p.output_samples = (int)std::llround(seconds * 44100);
            out.bars = bars;
            out.bpm = bpm;
            out.seconds = seconds;
        } else {
            const double seconds = number_field(root, "seconds", 11.0);
            if (seconds <= 0 || (config.family == Family::Saos && seconds > 11.0) || seconds > 47)
                throw std::invalid_argument("seconds is out of range for this model");
            p.seconds = (float)seconds;
            p.frames = (int)std::ceil(seconds * 44100 / 2048.0);
            p.output_samples = (int)std::llround(seconds * 44100);
            out.seconds = seconds;
        }
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
    if (p.steps < 0 || (config.family != Family::Saos && p.steps == 1) ||
        (p.cfg_scale < 0 && p.cfg_scale != -1.0f) ||
        p.sigma_rho <= 0 || p.sde_eta < 0)
        throw std::invalid_argument("invalid SAT sampler settings");
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
            if (!g_pipeline) {
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

Config parse_args(int argc, char** argv) {
    Config config;
    if (const char* value = std::getenv("SA3_MODELS_DIR")) config.models_dir = value;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--help") {
            std::puts("sat-server --model foundation-1|arc|kickbass|jerry-grunge|sao1 [--port N] [--models-dir DIR] [--encoding TIER] [--t5-encoding TIER] [--ae-encoding TIER] [--device DEVICE]");
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
        const Config config = parse_args(argc, argv);
        httplib::Server server;
        server.set_payload_max_length(64 * 1024);
        server.Get("/health", [config](const httplib::Request&, httplib::Response& res) {
            sa3::sat::PipelinePaths paths;
            std::string error;
            const bool ready = resolve_paths(config, paths, error);
            bool loaded;
            { std::lock_guard<std::mutex> guard(g_mutex); loaded = (bool)g_pipeline; }
            const std::string family = config.family == Family::Foundation ? "foundation-1" :
                                       config.family == Family::Saos ? "saos" : "sao1";
            std::string body = "{\"status\":\"" + std::string(ready ? "healthy" : "model_missing") +
                "\",\"ready\":" + (ready ? "true" : "false") +
                ",\"model_loaded\":" + (loaded ? "true" : "false") +
                ",\"model\":\"" + config.model + "\",\"family\":\"" + family + "\"";
            if (!ready) body += ",\"error\":\"" + escape(error) + "\"";
            res.set_content(body + "}", "application/json");
            if (!ready) res.status = 503;
        });
        server.Get("/ready", [config](const httplib::Request&, httplib::Response& res) {
            sa3::sat::PipelinePaths paths;
            std::string error;
            const bool ready = resolve_paths(config, paths, error);
            res.set_content(std::string("{\"ready\":") + (ready ? "true}" : "false}"), "application/json");
            if (!ready) res.status = 503;
        });
        auto generate = [config](const httplib::Request& req, httplib::Response& res, bool loop) {
            yyjson_doc* doc = yyjson_read(req.body.data(), req.body.size(), 0);
            if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
                if (doc) yyjson_doc_free(doc);
                fail(res, 400, "JSON object required");
                return;
            }
            try {
                Prepared prepared = prepare(config, yyjson_doc_get_root(doc), loop);
                yyjson_doc_free(doc);
                doc = nullptr;
                sa3::sat::PipelinePaths paths;
                std::string error;
                if (!resolve_paths(config, paths, error)) { fail(res, 503, error); return; }
                const std::string sid = session_id();
                const uint64_t seed = prepared.params.seed;
                const int bars = prepared.bars;
                const double bpm = prepared.bpm;
                const double host_bpm = prepared.host_bpm;
                const double seconds = prepared.seconds;
                const std::string prompt = prepared.params.prompt;
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
                    ",\"gen_duration\":" + number(seconds) + "}", "application/json");
            } catch (const std::exception& e) {
                if (doc) yyjson_doc_free(doc);
                fail(res, 400, e.what());
            }
        };
        server.Post("/generate", [generate](const httplib::Request& req, httplib::Response& res) {
            generate(req, res, false);
        });
        server.Post("/generate/loop", [generate](const httplib::Request& req, httplib::Response& res) {
            generate(req, res, true);
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
                ",\"final_duration\":" + number(job.seconds) + "}";
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
