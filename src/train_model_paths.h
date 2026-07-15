// train_model_paths.h - resolve training model paths using generation conventions.
#pragma once

#include "sa3_pipeline.h"
#include "train_config.h"

#include <string>

namespace sa3 {

inline bool resolve_train_model_paths(const TrainConfig& cfg, ModelPaths& paths, std::string& err) {
    const bool has_explicit_required =
        !cfg.tok_path.empty() && !cfg.t5_path.empty() && !cfg.dit_path.empty() && !cfg.same_path.empty();
    if (!has_explicit_required) {
        if (!ModelPaths::resolve(cfg.models_dir, cfg.model_variant, cfg.encoding, paths, err)) return false;
    }

    // Stable Audio 3 adapters are trained against the distinct base DiT and applied to the
    // inference-tuned model. This is true for medium, small-music, and small-sfx. The
    // conditioner, tokenizer, encoder, and SAME model still use the inference convention;
    // only the training DiT changes. Never fall back silently to the ARC/post-trained DiT:
    // its weights are close enough to train, but the LoRA then spends capacity compensating
    // for the base-model delta.
    if (cfg.dit_path.empty() &&
        (cfg.model_variant == "medium" || cfg.model_variant == "small-music" ||
         cfg.model_variant == "small-sfx")) {
        const std::string enc = (cfg.encoding == "f32" || cfg.encoding == "F32") ? "F32" : "F16";
        const std::string base_variant = cfg.model_variant + "-base";
        bool ambiguous = false;
        paths.dit = resolve_one(cfg.models_dir, "stable-audio-3-" + base_variant + "-dit-",
                                "-" + enc + ".gguf", &ambiguous);
        if (paths.dit.empty() || ambiguous) {
            err = ambiguous
                ? "multiple " + base_variant + " training DiTs match in " + cfg.models_dir
                : "no " + base_variant + " training DiT (stable-audio-3-" + base_variant +
                  "-dit-*-" + enc +
                  ".gguf) in " + cfg.models_dir;
            err += "; " + cfg.model_variant + " LoRA training requires stabilityai/stable-audio-3-" +
                   base_variant + " (inference still uses " + cfg.model_variant +
                   "). Convert/download that checkpoint or pass --dit explicitly";
            return false;
        }
    }
    if (!cfg.tok_path.empty())  paths.tok = cfg.tok_path;
    if (!cfg.t5_path.empty())   paths.t5 = cfg.t5_path;
    if (!cfg.cond_path.empty()) paths.cond = cfg.cond_path;
    if (!cfg.dit_path.empty())  paths.dit = cfg.dit_path;
    if (!cfg.same_path.empty()) paths.same = cfg.same_path;
    if (paths.tok.empty() || paths.t5.empty() || paths.dit.empty() || paths.same.empty()) {
        err = "training requires tokenizer, T5, DiT, and SAME model paths";
        return false;
    }
    return true;
}

} // namespace sa3
