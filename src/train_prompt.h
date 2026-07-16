// train_prompt.h - per-sample prompt composition for native SA3 LoRA training (Stage 13).
//
// Faithful port of the dashboard's dataset_processing/prompt_templates.py get_custom_metadata:
// per sample, pick one of the enabled methods (tags / paths / fixed) by `balance` weights, build a
// prompt from that method, then optionally prepend a trigger token. This runs fresh every time a
// track is drawn, so the same track yields different prompts across epochs (caption augmentation).
//
// RNG: the reference uses Python's global `random`; we cannot and need not reproduce its exact
// sequence (the original run's RNG state is unrecoverable). We match the *distribution* using a
// deterministic std::mt19937_64 supplied by the caller, so runs are reproducible from --seed.
#pragma once

#include "yyjson.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace sa3 {

// Display labels for tag keys, matching prompt_templates.py _TAG_DISPLAY (and its key order, which
// is the default tag_keys when the config omits them).
inline const std::vector<std::pair<std::string, std::string>>& prompt_tag_display() {
    static const std::vector<std::pair<std::string, std::string>> m = {
        {"title", "Title"}, {"artist", "Artist"}, {"album", "Album"}, {"genre", "Genre"},
        {"label", "Label"}, {"date", "Year"}, {"composer", "Composer"}, {"bpm", "BPM"},
        {"prompt", "Prompt"},
    };
    return m;
}

inline std::string prompt_tag_label(const std::string& key) {
    for (const auto& kv : prompt_tag_display()) if (kv.first == key) return kv.second;
    return key;
}

struct PromptConfig {
    bool loaded = false;            // false => caller falls back to the raw caption
    bool use_tags = true;
    bool use_paths = false;
    bool use_fixed = false;
    std::string fixed_text;
    std::vector<std::string> tag_keys;   // empty => all _TAG_DISPLAY keys (reference default)
    bool hide_tag_names = false;
    bool hide_commas = false;
    bool split_commas = false;
    bool shuffle = true;
    int balance_tags = 50;
    int balance_paths = 50;
    int balance_fixed = 0;
    bool path_hide_ext = false;
    bool path_hide_dirs = false;
    bool path_hide_topmost = false;
    std::string trigger;
    int trigger_pct = 80;
};

// Per-sample metadata the composer draws from. `tags` holds tag_key -> value (e.g. prompt=caption,
// title=..., bpm=...); missing/empty keys are skipped, exactly like the reference _get.
struct PromptMetadata {
    std::map<std::string, std::string> tags;
    std::string relpath;   // for the path method; the reference used the pre-encoded .npy relpath
    std::string text;      // ultimate fallback (reference metadata["text"])
};

inline std::string prompt_trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (unsigned char)s[b] <= ' ') ++b;
    while (e > b && (unsigned char)s[e - 1] <= ' ') --e;
    return s.substr(b, e - b);
}

inline std::string prompt_get(const PromptMetadata& md, const std::string& key) {
    auto it = md.tags.find(key);
    return it == md.tags.end() ? std::string() : prompt_trim(it->second);
}

// random.randint(a,b): inclusive both ends.
inline int prompt_randint(std::mt19937_64& rng, int a, int b) {
    return std::uniform_int_distribution<int>(a, b)(rng);
}
// random.random(): [0,1).
inline double prompt_rand01(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

// ── Tag-based prompt: port of _build_tag_prompt ──────────────────────────────────────────────
inline std::string prompt_build_tags(const PromptMetadata& md, const PromptConfig& pc,
                                     std::mt19937_64& rng) {
    std::vector<std::string> keys = pc.tag_keys;
    if (keys.empty()) for (const auto& kv : prompt_tag_display()) keys.push_back(kv.first);

    std::vector<std::string> parts;
    for (const std::string& key : keys) {
        const std::string val = prompt_get(md, key);
        if (val.empty()) continue;
        if (key == "prompt") { parts.push_back(val); continue; }
        const std::string label = prompt_tag_label(key);
        if (pc.split_commas && val.find(',') != std::string::npos) {
            size_t pos = 0;
            while (pos <= val.size()) {
                size_t c = val.find(',', pos);
                std::string sub = prompt_trim(val.substr(pos, c == std::string::npos ? std::string::npos : c - pos));
                if (!sub.empty()) parts.push_back(pc.hide_tag_names ? sub : (label + ": " + sub));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        } else {
            parts.push_back(pc.hide_tag_names ? val : (label + ": " + val));
        }
    }
    if (parts.empty()) return "";

    if (pc.shuffle) {
        // 50% shuffle all, 50% random subset of random size (matches the reference).
        if (prompt_rand01(rng) < 0.5) {
            std::shuffle(parts.begin(), parts.end(), rng);
        } else {
            const int k = prompt_randint(rng, 1, (int)parts.size());
            std::shuffle(parts.begin(), parts.end(), rng);
            parts.resize((size_t)k);
        }
    }

    const std::string sep = pc.hide_commas ? " " : ", ";
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) { if (i) out += sep; out += parts[i]; }
    return out;
}

// ── Path-based prompt: port of _build_path_prompt ────────────────────────────────────────────
inline std::string prompt_build_path(const PromptMetadata& md, const PromptConfig& pc) {
    std::string relpath = prompt_trim(md.relpath);
    if (relpath.empty()) return "";
    for (char& ch : relpath) if (ch == '\\') ch = '/';
    std::vector<std::string> parts;
    size_t pos = 0;
    while (true) {
        size_t s = relpath.find('/', pos);
        parts.push_back(relpath.substr(pos, s == std::string::npos ? std::string::npos : s - pos));
        if (s == std::string::npos) break;
        pos = s + 1;
    }
    std::string filename = parts.back();
    std::vector<std::string> dirs(parts.begin(), parts.end() - 1);

    if (pc.path_hide_ext) {
        size_t dot = filename.rfind('.');
        if (dot != std::string::npos && dot > 0) filename = filename.substr(0, dot);
    }
    auto join = [](const std::vector<std::string>& segs, const std::string& fname) {
        std::string out;
        for (const std::string& d : segs) { if (!out.empty()) out += '/'; out += d; }
        if (!out.empty()) out += '/';
        out += fname;
        return out;
    };
    if (pc.path_hide_dirs) {
        if (pc.path_hide_topmost || dirs.empty()) return filename;
        return dirs.front() + "/" + filename;
    }
    if (pc.path_hide_topmost && !dirs.empty()) {
        std::vector<std::string> rest(dirs.begin() + 1, dirs.end());
        return join(rest, filename);
    }
    return join(dirs, filename);
}

// ── Main entry point: port of get_custom_metadata ─────────────────────────────────────────────
inline std::string prompt_compose(const PromptConfig& pc, const PromptMetadata& md,
                                  std::mt19937_64& rng) {
    struct Cand { const char* method; std::string text; int weight; };
    std::vector<Cand> cands;
    if (pc.use_tags)  cands.push_back({"tags",  prompt_build_tags(md, pc, rng), pc.balance_tags});
    if (pc.use_paths) cands.push_back({"paths", prompt_build_path(md, pc),      pc.balance_paths});
    if (pc.use_fixed) cands.push_back({"fixed", pc.fixed_text,                  pc.balance_fixed});

    const char* chosen = nullptr;
    std::string prompt;
    if (cands.empty()) {
        prompt = md.text;
    } else {
        long total = 0;
        for (const Cand& c : cands) total += c.weight;
        if (total <= 0) {
            const Cand& c = cands[(size_t)prompt_randint(rng, 0, (int)cands.size() - 1)];
            chosen = c.method; prompt = c.text;
        } else {
            // random.choices(weights): pick proportional to weight.
            double u = prompt_rand01(rng) * (double)total;
            long acc = 0;
            for (const Cand& c : cands) { acc += c.weight; if (u < (double)acc) { chosen = c.method; prompt = c.text; break; } }
            if (!chosen) { chosen = cands.back().method; prompt = cands.back().text; }
        }
    }

    if (!pc.trigger.empty() && pc.trigger_pct > 0 && prompt_rand01(rng) * 100.0 < (double)pc.trigger_pct) {
        if (!prompt.empty()) {
            const bool use_comma = chosen && std::string(chosen) == "tags" && !pc.hide_commas;
            prompt = pc.trigger + (use_comma ? ", " : " ") + prompt;
        } else {
            prompt = pc.trigger;
        }
    }
    return prompt;
}

// ── Config loading ───────────────────────────────────────────────────────────────────────────
// Accepts a JSON file that is either a bare prompt_config object or a full dataset.json containing
// a "prompt_config" key (so the user can point --prompt-config at the same dataset.json the
// dashboard used). Missing keys keep their reference defaults.
inline bool load_prompt_config(const std::string& path, PromptConfig& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open prompt-config: " + path; return false; }
    std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    yyjson_doc* doc = yyjson_read(js.data(), js.size(), 0);
    if (!doc) { err = "invalid JSON prompt-config: " + path; return false; }
    yyjson_val* root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) { yyjson_doc_free(doc); err = "prompt-config root must be an object"; return false; }
    yyjson_val* pc = yyjson_obj_get(root, "prompt_config");
    if (pc && yyjson_is_obj(pc)) root = pc;   // full dataset.json -> descend into prompt_config

    auto get_bool = [&](const char* k, bool def) {
        yyjson_val* v = yyjson_obj_get(root, k);
        return (v && yyjson_is_bool(v)) ? yyjson_get_bool(v) : def;
    };
    auto get_str = [&](const char* k, const std::string& def) {
        yyjson_val* v = yyjson_obj_get(root, k);
        return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : def;
    };
    auto get_int = [&](yyjson_val* obj, const char* k, int def) {
        yyjson_val* v = obj ? yyjson_obj_get(obj, k) : nullptr;
        return (v && yyjson_is_num(v)) ? (int)yyjson_get_num(v) : def;
    };

    out = PromptConfig{};
    out.loaded = true;
    out.use_tags = get_bool("use_tags", true);
    out.use_paths = get_bool("use_paths", false);
    out.use_fixed = get_bool("use_fixed", false);
    out.fixed_text = get_str("fixed_text", "");
    out.hide_tag_names = get_bool("hide_tag_names", false);
    out.hide_commas = get_bool("hide_commas", false);
    out.split_commas = get_bool("split_commas", false);
    out.shuffle = get_bool("shuffle", true);
    out.trigger = get_str("trigger", "");
    out.trigger_pct = get_int(root, "trigger_pct", 80);

    if (yyjson_val* tk = yyjson_obj_get(root, "tag_keys"); tk && yyjson_is_arr(tk)) {
        size_t i, n; yyjson_val* item;
        yyjson_arr_foreach(tk, i, n, item) if (yyjson_is_str(item)) out.tag_keys.push_back(yyjson_get_str(item));
    }
    if (yyjson_val* bal = yyjson_obj_get(root, "balance"); bal && yyjson_is_obj(bal)) {
        out.balance_tags = get_int(bal, "tags", 50);
        out.balance_paths = get_int(bal, "paths", 50);
        out.balance_fixed = get_int(bal, "fixed", 0);
    }
    if (yyjson_val* po = yyjson_obj_get(root, "path_opts"); po && yyjson_is_obj(po)) {
        auto po_bool = [&](const char* camel, const char* snake) {
            yyjson_val* v = yyjson_obj_get(po, camel); if (!v) v = yyjson_obj_get(po, snake);
            return (v && yyjson_is_bool(v)) ? yyjson_get_bool(v) : false;
        };
        out.path_hide_ext = po_bool("hideExt", "hide_ext");
        out.path_hide_dirs = po_bool("hideDirs", "hide_dirs");
        out.path_hide_topmost = po_bool("hideTopmostDir", "hide_topmost_dir");
    }
    yyjson_doc_free(doc);
    return true;
}

} // namespace sa3
