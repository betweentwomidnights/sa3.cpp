#pragma once

#include "pipeline.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace sa3::sat {

inline constexpr const char* kSaosPublishedRepo = "thepatch/stable-audio-open-small-GGUF";
inline constexpr const char* kDefaultSaosVariant = "arc";
inline constexpr const char* kDefaultSaosEncoding = "Q5_K_M";

inline std::string normalize_encoding(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::toupper(c);
    });
    return value;
}

inline bool is_saos_published_encoding(const std::string& value) {
    const std::string enc = normalize_encoding(value);
    return enc == "F16" || enc == "Q8_0" || enc == "Q5_K_M" || enc == "Q4_K_M";
}

inline std::string canonical_saos_variant(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return c == '_' ? '-' : (char)std::tolower(c);
    });
    if (value == "saos" || value == "stable-audio-open-small") return "arc";
    if (value == "jerry" || value == "jerry-grunge") return "jerry-grunge";
    if (value == "kickbass") return "kickbass";
    return value;
}

inline std::string saos_dit_relative_path(const std::string& variant,
                                          const std::string& encoding) {
    const std::string name = canonical_saos_variant(variant);
    const std::string enc = normalize_encoding(encoding);
    if (!is_saos_published_encoding(enc)) return {};
    if (name == "arc")
        return "stable-audio-open-small-dit-0.3B-v1.0-" + enc + ".gguf";
    if (name == "kickbass")
        return "finetunes/kickbass/kickbass-v1-e257-dit-0.3B-v1.0-" + enc + ".gguf";
    if (name == "jerry-grunge")
        return "finetunes/jerry-grunge/jerry-grunge-bs64-step3000-dit-0.3B-v1.0-" + enc + ".gguf";
    return {};
}

inline std::string saos_t5_relative_path(const std::string& encoding) {
    const std::string enc = normalize_encoding(encoding);
    if (!is_saos_published_encoding(enc)) return {};
    return "t5-base-encoder-0.1B-v1.0-" + enc + ".gguf";
}

inline std::string saos_oobleck_relative_path(const std::string& encoding) {
    const std::string enc = normalize_encoding(encoding);
    if (!is_saos_published_encoding(enc)) return {};
    return "stable-audio-open-small-oobleck-v1.0-" + enc + ".gguf";
}

inline bool resolve_saos_model(const std::string& models_dir,
                                    const std::string& variant,
                                    const std::string& dit_encoding,
                                    const std::string& t5_encoding,
                                    const std::string& ae_encoding,
                                    PipelinePaths* paths,
                                    std::string* error = nullptr) {
    if (!paths) return false;
    const std::string dit = saos_dit_relative_path(variant, dit_encoding);
    const std::string t5 = saos_t5_relative_path(t5_encoding);
    const std::string ae = saos_oobleck_relative_path(ae_encoding);
    if (dit.empty()) {
        if (error) *error = "unknown SAT model or encoding (models: arc, kickbass, jerry-grunge; encodings: f16, q8_0, q5_k_m, q4_k_m)";
        return false;
    }
    const std::filesystem::path root(models_dir);
    paths->dit = (root / std::filesystem::path(dit)).string();
    paths->t5 = (root / std::filesystem::path(t5)).string();
    paths->autoencoder = (root / std::filesystem::path(ae)).string();
    for (const auto& path : {paths->dit, paths->t5, paths->autoencoder}) {
        if (!std::filesystem::is_regular_file(path)) {
            if (error) *error = "missing " + path + " (run: python tools/download_models.py --sat --saos-variant " + canonical_saos_variant(variant) + ")";
            return false;
        }
    }
    return true;
}

} // namespace sa3::sat
