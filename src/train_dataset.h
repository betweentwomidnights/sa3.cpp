// train_dataset.h - dataset manifest loading for native SA3 LoRA training.
#pragma once

#include "yyjson.h"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace sa3 {

struct TrainDatasetRecord {
    std::string id;
    std::string split;
    std::string audio_path;
    std::string caption_path;
    std::string audio_sha256;
    double duration_seconds = 0.0;
    // Optional prompt-composition metadata (Stage 13). `tags` holds recognized tag keys
    // (title/artist/album/genre/label/date/composer/bpm) if present in metadata.jsonl; `prompt`
    // is sourced from the caption at train time. `relpath` feeds the path-based prompt method.
    std::map<std::string, std::string> tags;
    std::string relpath;
};

struct TrainSplitManifest {
    std::string root_dir;
    std::string split;
    std::vector<std::string> filelist;
    std::vector<TrainDatasetRecord> records;
};

struct TrainAudioCaptionPair {
    std::string id;
    std::string split;
    std::string audio_rel;
    std::string caption_rel;
    std::string audio_path;
    std::string caption_path;
    std::string audio_sha256;
    double duration_seconds = 0.0;
    std::map<std::string, std::string> tags;   // Stage 13 prompt-composition tags (optional)
    std::string relpath;                        // Stage 13 path-method source (optional)
};

inline std::string train_trim(std::string s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    size_t e = s.size();
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
    return s.substr(b, e - b);
}

inline bool train_read_lines(const std::string& path, std::vector<std::string>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        line = train_trim(line);
        if (!line.empty()) out.push_back(line);
    }
    return true;
}

inline std::string train_json_string(yyjson_val* root, const char* key) {
    yyjson_val* v = yyjson_obj_get(root, key);
    return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
}

inline double train_json_number(yyjson_val* root, const char* key) {
    yyjson_val* v = yyjson_obj_get(root, key);
    return (v && yyjson_is_num(v)) ? yyjson_get_num(v) : 0.0;
}

inline bool train_parse_manifest_line(const std::string& line, const std::string& path,
                                      int line_no, TrainDatasetRecord& rec, std::string& err) {
    yyjson_doc* doc = yyjson_read(line.data(), line.size(), 0);
    if (!doc) {
        err = path + ":" + std::to_string(line_no) + ": invalid JSON";
        return false;
    }
    yyjson_val* root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        err = path + ":" + std::to_string(line_no) + ": record must be a JSON object";
        return false;
    }
    rec.id = train_json_string(root, "id");
    rec.split = train_json_string(root, "split");
    rec.audio_path = train_json_string(root, "audio_path");
    rec.caption_path = train_json_string(root, "caption_path");
    rec.audio_sha256 = train_json_string(root, "audio_sha256");
    rec.duration_seconds = train_json_number(root, "duration_seconds");
    rec.relpath = train_json_string(root, "relpath");
    for (const char* key : {"title", "artist", "album", "genre", "label", "date", "composer", "bpm"}) {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (v && yyjson_is_str(v)) { std::string s = yyjson_get_str(v); if (!s.empty()) rec.tags[key] = s; }
        else if (v && yyjson_is_num(v)) rec.tags[key] = std::to_string(yyjson_get_num(v));
    }
    yyjson_doc_free(doc);

    if (rec.id.empty()) {
        err = path + ":" + std::to_string(line_no) + ": missing id";
        return false;
    }
    if (rec.audio_path.empty()) {
        err = path + ":" + std::to_string(line_no) + ": missing audio_path";
        return false;
    }
    return true;
}

inline bool load_train_split_manifest(const std::string& dataset_dir, const std::string& split,
                                      TrainSplitManifest& out, std::string& err) {
    out = TrainSplitManifest{};
    out.root_dir = dataset_dir;
    out.split = split;
    const std::string split_dir = dataset_dir + "/" + split;
    if (!train_read_lines(split_dir + "/filelist.txt", out.filelist, err)) return false;

    std::ifstream f(split_dir + "/metadata.jsonl");
    if (!f) {
        err = "cannot open " + split_dir + "/metadata.jsonl";
        return false;
    }
    std::string line;
    int line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        line = train_trim(line);
        if (line.empty()) continue;
        TrainDatasetRecord rec;
        if (!train_parse_manifest_line(line, split_dir + "/metadata.jsonl", line_no, rec, err)) return false;
        out.records.push_back(std::move(rec));
    }
    return true;
}

inline std::string train_replace_extension(const std::string& path, const std::string& ext) {
    std::filesystem::path p(path);
    p.replace_extension(ext);
    return p.generic_string();
}

inline std::string train_join_path(const std::string& a, const std::string& b) {
    return (std::filesystem::path(a) / std::filesystem::path(b)).string();
}

inline bool resolve_train_pairs(const TrainSplitManifest& m, std::vector<TrainAudioCaptionPair>& out,
                                std::string& err) {
    out.clear();
    std::map<std::string, TrainDatasetRecord> by_audio;
    for (const TrainDatasetRecord& r : m.records) by_audio[r.audio_path] = r;
    const std::string split_dir = train_join_path(m.root_dir, m.split);
    for (const std::string& audio_rel : m.filelist) {
        auto it = by_audio.find(audio_rel);
        TrainDatasetRecord r;
        if (it != by_audio.end()) {
            r = it->second;
        } else {
            r.id = std::filesystem::path(audio_rel).stem().string();
            r.split = m.split;
            r.audio_path = audio_rel;
        }
        TrainAudioCaptionPair p;
        p.id = r.id;
        p.split = r.split.empty() ? m.split : r.split;
        p.audio_rel = r.audio_path.empty() ? audio_rel : r.audio_path;
        p.caption_rel = r.caption_path.empty() ? train_replace_extension(p.audio_rel, ".txt") : r.caption_path;
        p.audio_path = train_join_path(split_dir, p.audio_rel);
        p.caption_path = train_join_path(split_dir, p.caption_rel);
        p.audio_sha256 = r.audio_sha256;
        p.duration_seconds = r.duration_seconds;
        p.tags = r.tags;
        // Path-method source: explicit relpath if given, else the bare audio filename (dir-less,
        // matching how the reference pre-encode stored a bare .npy relpath for this dataset).
        p.relpath = r.relpath.empty() ? std::filesystem::path(p.audio_rel).filename().generic_string()
                                      : r.relpath;
        if (p.id.empty()) {
            err = "cannot resolve empty id for " + p.audio_rel;
            return false;
        }
        if (p.audio_rel.empty() || p.caption_rel.empty()) {
            err = "cannot resolve audio/caption pair for " + p.id;
            return false;
        }
        out.push_back(std::move(p));
    }
    return true;
}

inline bool validate_train_split_pairs(const TrainSplitManifest& m,
                                       const std::vector<TrainAudioCaptionPair>& pairs,
                                       std::string& err) {
    std::set<std::string> ids;
    std::set<std::string> file_entries;
    for (const std::string& rel : m.filelist) {
        if (!file_entries.insert(rel).second) {
            err = m.split + ": duplicate filelist entry: " + rel;
            return false;
        }
    }
    for (const TrainDatasetRecord& r : m.records) {
        if (r.split.empty()) {
            err = m.split + ": metadata record " + r.id + " has empty split";
            return false;
        }
        if (r.split != m.split) {
            err = m.split + ": metadata record " + r.id + " declares split " + r.split;
            return false;
        }
    }
    for (const TrainAudioCaptionPair& p : pairs) {
        if (!ids.insert(p.id).second) {
            err = m.split + ": duplicate id: " + p.id;
            return false;
        }
        if (!std::filesystem::exists(p.audio_path)) {
            err = m.split + ": missing audio file for " + p.id + ": " + p.audio_path;
            return false;
        }
        if (!std::filesystem::exists(p.caption_path)) {
            err = m.split + ": missing caption file for " + p.id + ": " + p.caption_path;
            return false;
        }
    }
    return true;
}

inline std::string train_canonical_or_absolute(const std::string& path) {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::weakly_canonical(path, ec);
    if (ec) p = std::filesystem::absolute(path, ec);
    return ec ? path : p.string();
}

inline bool validate_no_training_contamination(const std::vector<TrainAudioCaptionPair>& train,
                                               const std::vector<TrainAudioCaptionPair>& heldout,
                                               const std::string& heldout_name,
                                               std::string& err) {
    std::set<std::string> train_basenames;
    std::set<std::string> train_paths;
    std::set<std::string> train_hashes;
    for (const TrainAudioCaptionPair& p : train) {
        train_basenames.insert(std::filesystem::path(p.audio_rel).stem().string());
        train_paths.insert(train_canonical_or_absolute(p.audio_path));
        if (!p.audio_sha256.empty()) train_hashes.insert(p.audio_sha256);
    }
    for (const TrainAudioCaptionPair& p : heldout) {
        const std::string base = std::filesystem::path(p.audio_rel).stem().string();
        if (train_basenames.count(base)) {
            err = "train split contaminates " + heldout_name + " by basename: " + base;
            return false;
        }
        const std::string canon = train_canonical_or_absolute(p.audio_path);
        if (train_paths.count(canon)) {
            err = "train split contaminates " + heldout_name + " by path: " + canon;
            return false;
        }
        if (!p.audio_sha256.empty() && train_hashes.count(p.audio_sha256)) {
            err = "train split contaminates " + heldout_name + " by audio_sha256: " + p.audio_sha256;
            return false;
        }
    }
    return true;
}

} // namespace sa3
