// train_config.h - configuration parsing for native SA3 LoRA training.
#pragma once

#include "yyjson.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace sa3 {

struct TrainConfig {
    std::string model_variant = "medium";
    std::string encoding = "f16";
    std::string models_dir = "models";
    std::string tok_path;
    std::string t5_path;
    std::string cond_path;
    std::string dit_path;
    std::string same_path;
    std::string dataset_dir = "../datasets/mnesia-audio-v1";
    std::string train_split = "train";
    std::string test_split = "test";
    std::string evaluation_split = "evaluation";
    std::string adapter_type = "lora";
    int rank = 8;
    float alpha = 8.0f;
    float learning_rate = 1.0e-4f;
    float weight_decay = 0.0f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps = 1.0e-8f;
    int batch_size = 1;
    int frames = 128;
    float duration_sec = 0.0f;
    std::string precision = "f32";
    unsigned long long seed = 0;
    int checkpoint_every = 1;
    std::string output_dir = "train-runs/sa3-lora";
    std::string resume_adapter;
    std::string optimizer = "adamw";
    std::string svd_bases_path;   // optional GGUF of frozen U/V bases for -xs adapters (exact-parity path)
    std::string prompt_mode = "caption";
    std::string eval_caption;
    int eval_every = 1;
    int generation_steps = 8;
    // Multi-epoch training (Stage 1). 0 = single pass over the dataset (legacy). >0 = loop with
    // per-epoch shuffle until this many optimizer updates. max_epochs caps passes when set.
    int max_steps = 0;
    int max_epochs = 0;
    // Random-crop windowing (Stage 2): pick a random window start per sample instead of the
    // fixed front (start=0) window. Matches the reference pre-encode + random-crop regime.
    bool random_crop = false;
    // Global gradient-norm clip (Stage 3): 0 = off. Reference training uses 1.0.
    float grad_clip = 0.0f;
    // Timestep sampler (Stage 5): "uniform" or "trunc_logit_normal" (the reference default).
    std::string timestep_sampler = "uniform";
    // Training-time distribution shift on sampled timesteps (Stage 10). The reference warps t
    // through the model's distribution_shift_options; for sa3-medium that is type "full" with
    // min_length 256 / max_length 4096 (base/max shift 0.5/1.15). Values map onto
    // sa3_pipeline.h dist_shift_warp with dist_shift_defaults per type. "None" disables.
    std::string dist_shift = "Full";
    // Sequence length fed to the shift. true = reference use_effective_length_for_schedule:
    // ceil(seconds_total * sr / downsampling_ratio) from the FULL source-file duration (the
    // pre-encode metadata is not updated for crops). false = the crop's latent frame count.
    bool dist_shift_effective_length = true;
};

// Canonicalize a dist-shift type name to the sa3_pipeline.h spelling. Accepts any case.
inline bool train_normalize_dist_shift(std::string& v) {
    std::string s;
    for (char ch : v) s += (char)std::tolower((unsigned char)ch);
    if      (s == "none")   v = "None";
    else if (s == "full")   v = "Full";
    else if (s == "flux")   v = "Flux";
    else if (s == "logsnr") v = "LogSNR";
    else return false;
    return true;
}

inline bool train_parse_bool(const std::string& text, bool& out) {
    if (text == "1" || text == "true" || text == "yes" || text == "on")  { out = true;  return true; }
    if (text == "0" || text == "false" || text == "no" || text == "off") { out = false; return true; }
    return false;
}

inline bool train_parse_i32(const std::string& text, int& out) {
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(text.c_str(), &end, 10);
    if (errno || !end || *end != '\0') return false;
    out = (int)v;
    return true;
}

inline bool train_parse_u64(const std::string& text, unsigned long long& out) {
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (errno || !end || *end != '\0') return false;
    out = v;
    return true;
}

inline bool train_parse_f32(const std::string& text, float& out) {
    char* end = nullptr;
    errno = 0;
    float v = std::strtof(text.c_str(), &end);
    if (errno || !end || *end != '\0') return false;
    out = v;
    return true;
}

inline std::string train_read_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open config file: " + path;
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline bool train_set_config_value(TrainConfig& c, const std::string& key, const std::string& value, std::string& err) {
    auto set_i = [&](int& dst) {
        if (!train_parse_i32(value, dst)) { err = "invalid integer for --" + key + ": " + value; return false; }
        return true;
    };
    auto set_u = [&](unsigned long long& dst) {
        if (!train_parse_u64(value, dst)) { err = "invalid unsigned integer for --" + key + ": " + value; return false; }
        return true;
    };
    auto set_f = [&](float& dst) {
        if (!train_parse_f32(value, dst)) { err = "invalid number for --" + key + ": " + value; return false; }
        return true;
    };

    if      (key == "model" || key == "model_variant") c.model_variant = value;
    else if (key == "encoding") c.encoding = value;
    else if (key == "models-dir" || key == "models_dir") c.models_dir = value;
    else if (key == "tok" || key == "tok_path") c.tok_path = value;
    else if (key == "t5" || key == "t5_path") c.t5_path = value;
    else if (key == "cond" || key == "cond_path") c.cond_path = value;
    else if (key == "dit" || key == "dit_path") c.dit_path = value;
    else if (key == "same" || key == "same_path") c.same_path = value;
    else if (key == "dataset" || key == "dataset-dir" || key == "dataset_dir") c.dataset_dir = value;
    else if (key == "train-split" || key == "train_split") c.train_split = value;
    else if (key == "test-split" || key == "test_split") c.test_split = value;
    else if (key == "evaluation-split" || key == "evaluation_split") c.evaluation_split = value;
    else if (key == "adapter-type" || key == "adapter_type") c.adapter_type = value;
    else if (key == "rank") return set_i(c.rank);
    else if (key == "alpha") return set_f(c.alpha);
    else if (key == "learning-rate" || key == "learning_rate" || key == "lr") return set_f(c.learning_rate);
    else if (key == "weight-decay" || key == "weight_decay") return set_f(c.weight_decay);
    else if (key == "adam-beta1" || key == "adam_beta1") return set_f(c.adam_beta1);
    else if (key == "adam-beta2" || key == "adam_beta2") return set_f(c.adam_beta2);
    else if (key == "adam-eps" || key == "adam_eps") return set_f(c.adam_eps);
    else if (key == "batch-size" || key == "batch_size") return set_i(c.batch_size);
    else if (key == "frames") return set_i(c.frames);
    else if (key == "duration" || key == "duration-sec" || key == "duration_sec") return set_f(c.duration_sec);
    else if (key == "precision") c.precision = value;
    else if (key == "seed") return set_u(c.seed);
    else if (key == "checkpoint-every" || key == "checkpoint_every") return set_i(c.checkpoint_every);
    else if (key == "out" || key == "output-dir" || key == "output_dir") c.output_dir = value;
    else if (key == "resume-adapter" || key == "resume_adapter") c.resume_adapter = value;
    else if (key == "optimizer") c.optimizer = value;
    else if (key == "svd-bases" || key == "svd_bases" || key == "svd-bases-path" || key == "svd_bases_path") c.svd_bases_path = value;
    else if (key == "prompt-mode" || key == "prompt_mode") c.prompt_mode = value;
    else if (key == "eval-caption" || key == "eval_caption") c.eval_caption = value;
    else if (key == "eval-every" || key == "eval_every") return set_i(c.eval_every);
    else if (key == "generation-steps" || key == "generation_steps") return set_i(c.generation_steps);
    else if (key == "max-steps" || key == "max_steps") return set_i(c.max_steps);
    else if (key == "max-epochs" || key == "max_epochs" || key == "epochs") return set_i(c.max_epochs);
    else if (key == "grad-clip" || key == "grad_clip" || key == "gradient-clip-val") return set_f(c.grad_clip);
    else if (key == "timestep-sampler" || key == "timestep_sampler") c.timestep_sampler = value;
    else if (key == "dist-shift" || key == "dist_shift") {
        c.dist_shift = value;
        if (!train_normalize_dist_shift(c.dist_shift)) {
            err = "unsupported dist_shift (expected none|full|flux|logsnr): " + value;
            return false;
        }
    }
    else if (key == "dist-shift-effective-length" || key == "dist_shift_effective_length") {
        if (!train_parse_bool(value, c.dist_shift_effective_length)) { err = "invalid boolean for --" + key + ": " + value; return false; }
    }
    else if (key == "random-crop" || key == "random_crop") {
        if (!train_parse_bool(value, c.random_crop)) { err = "invalid boolean for --" + key + ": " + value; return false; }
    }
    else {
        err = "unknown training option: " + key;
        return false;
    }
    return true;
}

inline bool train_apply_json_config(TrainConfig& c, const std::string& path, std::string& err) {
    const std::string js = train_read_file(path, err);
    if (!err.empty()) return false;
    yyjson_doc* doc = yyjson_read(js.data(), js.size(), 0);
    if (!doc) {
        err = "invalid JSON config: " + path;
        return false;
    }
    yyjson_val* root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        err = "config root must be a JSON object: " + path;
        return false;
    }
    yyjson_obj_iter it;
    yyjson_obj_iter_init(root, &it);
    yyjson_val* keyv = nullptr;
    while ((keyv = yyjson_obj_iter_next(&it))) {
        const char* key = yyjson_get_str(keyv);
        yyjson_val* val = yyjson_obj_iter_get_val(keyv);
        std::string text;
        if (yyjson_is_str(val)) {
            text = yyjson_get_str(val);
        } else if (yyjson_is_int(val)) {
            text = std::to_string(yyjson_get_sint(val));
        } else if (yyjson_is_uint(val)) {
            text = std::to_string(yyjson_get_uint(val));
        } else if (yyjson_is_real(val)) {
            std::ostringstream ss;
            ss << yyjson_get_real(val);
            text = ss.str();
        } else {
            yyjson_doc_free(doc);
            err = "unsupported JSON value for key '" + std::string(key) + "'";
            return false;
        }
        if (!train_set_config_value(c, key, text, err)) {
            yyjson_doc_free(doc);
            return false;
        }
    }
    yyjson_doc_free(doc);
    return true;
}

inline bool train_parse_args(int argc, char** argv, TrainConfig& c, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            err = "help requested";
            return false;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) { err = "--config requires a path"; return false; }
            if (!train_apply_json_config(c, argv[++i], err)) return false;
            continue;
        }
        if (arg.rfind("--", 0) != 0) {
            err = "unexpected positional argument: " + arg;
            return false;
        }
        std::string key = arg.substr(2);
        std::string value;
        const size_t eq = key.find('=');
        if (eq != std::string::npos) {
            value = key.substr(eq + 1);
            key = key.substr(0, eq);
        } else {
            if (i + 1 >= argc) { err = arg + " requires a value"; return false; }
            value = argv[++i];
        }
        if (!train_set_config_value(c, key, value, err)) return false;
    }
    return true;
}

inline bool validate_train_config(const TrainConfig& c, std::string& err) {
    if (c.dataset_dir.empty()) { err = "dataset_dir is required"; return false; }
    if (c.output_dir.empty()) { err = "output_dir is required"; return false; }
    if (c.train_split.empty() || c.test_split.empty() || c.evaluation_split.empty()) {
        err = "train/test/evaluation split names are required";
        return false;
    }
    if (c.train_split == c.test_split || c.train_split == c.evaluation_split) {
        err = "train split must be distinct from test/evaluation splits";
        return false;
    }
    if (c.adapter_type != "lora" && c.adapter_type != "dora-rows" &&
        c.adapter_type != "dora-cols" && c.adapter_type != "bora" &&
        c.adapter_type != "lora-xs" && c.adapter_type != "dora-rows-xs" &&
        c.adapter_type != "dora-cols-xs" && c.adapter_type != "bora-xs") {
        err = "unsupported adapter_type: " + c.adapter_type;
        return false;
    }
    if (c.rank <= 0) { err = "rank must be positive"; return false; }
    if (c.alpha <= 0.0f) { err = "alpha must be positive"; return false; }
    if (c.learning_rate <= 0.0f) { err = "learning_rate must be positive"; return false; }
    if (c.batch_size <= 0) { err = "batch_size must be positive"; return false; }
    if (c.frames <= 0 && c.duration_sec <= 0.0f) { err = "frames or duration_sec must be positive"; return false; }
    if (c.optimizer != "adamw") { err = "unsupported optimizer: " + c.optimizer; return false; }
    if (c.prompt_mode != "caption" && c.prompt_mode != "caption-lyrics" && c.prompt_mode != "lyrics") {
        err = "unsupported prompt_mode: " + c.prompt_mode;
        return false;
    }
    if (c.timestep_sampler != "uniform" && c.timestep_sampler != "trunc_logit_normal") {
        err = "unsupported timestep_sampler: " + c.timestep_sampler;
        return false;
    }
    if (c.max_steps < 0 || c.max_epochs < 0) { err = "max_steps/max_epochs must be non-negative"; return false; }
    if (c.grad_clip < 0.0f) { err = "grad_clip must be non-negative (0 = off)"; return false; }
    return true;
}

inline std::string train_config_usage(const char* argv0) {
    std::ostringstream ss;
    ss << "usage: " << argv0 << " [--config train.json] [training options]\n"
       << "core options: --model medium|small-music|small-sfx --models-dir DIR --dataset DIR --out DIR\n"
       << "adapter: --adapter-type lora|dora-rows|dora-cols|bora|*-xs --rank N --alpha F\n"
       << "optimization: --learning-rate F --batch-size N --frames N --duration SEC --seed N\n"
       << "schedule: --timestep-sampler uniform|trunc_logit_normal --dist-shift none|full|flux|logsnr\n"
       << "          --dist-shift-effective-length BOOL (full-file effective length vs crop frames)\n"
       << "conditioning: --prompt-mode caption|caption-lyrics|lyrics\n";
    return ss.str();
}

} // namespace sa3
