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
