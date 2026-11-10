// Per-LoRA layer filters, as in stable_audio_3's lora_configs: a LoRA is switched off on every
// layer whose name contains one of the filter's comma-separated substrings
// (stable_audio_3/models/lora/utils.py filter_lora_layers; ranges as in lora/model.py _expand).
// Patterns match sa3.cpp's names (dit.3.cross.q) and the original module names relative to
// DiffusionTransformer (transformer.layers.3.cross_attn.to_q), which is what upstream filters on.
// Not wired into the samplers yet.
#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sa3 {
inline std::string lora_filter_trim(std::string s) {
    auto space = [](unsigned char c) { return std::isspace(c) != 0; };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), space));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), space).base(), s.end());
    return s;
}

inline std::vector<std::string> expand_lora_filter(const std::string& token, unsigned depth = 0) {
    if (depth > 64) throw std::invalid_argument("layer_filter has too many bracket ranges");
    for (size_t open = token.find('['); open != std::string::npos; open = token.find('[', open + 1)) {
        size_t dash = open + 1;
        while (dash < token.size() && token[dash] >= '0' && token[dash] <= '9') ++dash;
        if (dash == open + 1 || dash == token.size() || token[dash] != '-') continue;
        size_t close = dash + 1;
        while (close < token.size() && token[close] >= '0' && token[close] <= '9') ++close;
        if (close == dash + 1 || close == token.size() || token[close] != ']') continue;
        uint64_t first, last;
        try {
            first = std::stoull(token.substr(open + 1, dash - open - 1));
            last = std::stoull(token.substr(dash + 1, close - dash - 1));
        } catch (...) { throw std::invalid_argument("layer_filter bracket index is too large"); }
        const uint64_t distance = first < last ? last - first : first - last;
        if (distance >= 4096) throw std::invalid_argument("layer_filter expands beyond 4096 patterns");
        auto suffixes = expand_lora_filter(token.substr(close + 1), depth + 1);
        if (suffixes.size() > 4096 / (distance + 1))
            throw std::invalid_argument("layer_filter expands beyond 4096 patterns");
        std::vector<std::string> result;
        for (uint64_t value = first;; first < last ? ++value : --value) {
            for (const auto& suffix : suffixes)
                result.push_back(token.substr(0, open) + std::to_string(value) + suffix);
            if (value == last) break;
        }
        return result;
    }
    return {token};
}

// Matches stable-audio-tools: comma-separated substrings EXCLUDE layers; bracket
// ranges expand inclusively in either direction. Empty overall text means none.
inline std::vector<std::string> compile_lora_filter(std::string filter) {
    if (filter.size() > 4096) throw std::invalid_argument("layer_filter exceeds 4096 bytes");
    filter = lora_filter_trim(filter);
    std::transform(filter.begin(), filter.end(), filter.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    std::vector<std::string> result;
    for (size_t begin = 0; begin < filter.size();) {
        size_t end = filter.find(',', begin);
        if (end == std::string::npos) end = filter.size();
        if (end != begin) {
            auto expanded = expand_lora_filter(lora_filter_trim(filter.substr(begin, end - begin)));
            if (result.size() + expanded.size() > 4096)
                throw std::invalid_argument("layer_filter expands beyond 4096 patterns");
            result.insert(result.end(), expanded.begin(), expanded.end());
        }
        begin = end + 1;
    }
    return result;
}

// Reverse convert_dit's namespace for substring filters on the original module
// names. Weight suffixes are absent because Python filters named modules. Names are relative to
// DiffusionTransformer, which filter_lora_layers is called on, so they carry no "model." prefix.
inline std::string lora_torch_module(const std::string& stem) {
    static const std::map<std::string, std::string> direct = {
        {"dit.pre_conv", "preprocess_conv"}, {"dit.post_conv", "postprocess_conv"},
        {"dit.cond_embed.0", "to_cond_embed.0"}, {"dit.cond_embed.2", "to_cond_embed.2"},
        {"dit.global_embed.0", "to_global_embed.0"}, {"dit.global_embed.2", "to_global_embed.2"},
        {"dit.time_embed.0", "to_timestep_embed.0"}, {"dit.time_embed.2", "to_timestep_embed.2"},
        {"dit.gce.0", "transformer.global_cond_embedder.0"}, {"dit.gce.2", "transformer.global_cond_embedder.2"},
        {"dit.proj_in", "transformer.project_in"}, {"dit.proj_out", "transformer.project_out"},
    };
    if (auto it = direct.find(stem); it != direct.end()) return it->second;
    if (stem.rfind("dit.", 0) != 0) return "";
    const size_t dot = stem.find('.', 4);
    if (dot == std::string::npos || dot == 4) return "";
    const std::string index = stem.substr(4, dot - 4);
    if (!std::all_of(index.begin(), index.end(), [](char c) { return c >= '0' && c <= '9'; })) return "";
    static const std::map<std::string, std::string> block = {
        {"self.qkv", "self_attn.to_qkv"}, {"self.out", "self_attn.to_out"},
        {"cross.q", "cross_attn.to_q"}, {"cross.kv", "cross_attn.to_kv"}, {"cross.out", "cross_attn.to_out"},
        {"ff.proj", "ff.ff.0.proj"}, {"ff.out", "ff.ff.2"},
        {"local.0", "to_local_embed.0"}, {"local.2", "to_local_embed.2"},
    };
    auto it = block.find(stem.substr(dot + 1));
    return it == block.end() ? "" : "transformer.layers." + index + "." + it->second;
}

inline bool lora_layer_excluded(const std::string& stem, const std::vector<std::string>& patterns) {
    if (patterns.empty()) return false;
    const std::string original = lora_torch_module(stem);
    for (const auto& pattern : patterns)
        if (stem.find(pattern) != std::string::npos || (!original.empty() && original.find(pattern) != std::string::npos))
            return true;
    return false;
}
} // namespace sa3
