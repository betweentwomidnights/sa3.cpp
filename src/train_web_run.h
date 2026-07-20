// train_web_run.h — run state + multi-run registry for sa3-train-web.
//
// A "run" is one invocation of `sa3-train` targeting one output_dir. The web
// companion keeps a persistent registry (survives process restarts) so the UI
// can show history of past trainings and browse their artifacts. Only ONE run
// may be `running` at a time (GPU/VRAM constraint); finished/stopped runs stay
// in the registry as history.
//
// This header is used only by tools/sa3-train-web.cpp. It depends on STL,
// <filesystem>, and the vendored yyjson C API (vendor/yyjson/yyjson.h).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "yyjson.h"

namespace sa3_train_web {

// Status of a single training run.
enum class RunStatus {
    Queued,     // accepted, not yet spawned
    Running,    // child process alive
    Completed,  // child exited 0
    Stopped,    // terminated by user (SIGTERM)
    Failed,     // child exited non-zero or crashed
};

inline const char* run_status_str(RunStatus s) {
    switch (s) {
        case RunStatus::Queued:    return "queued";
        case RunStatus::Running:   return "running";
        case RunStatus::Completed: return "completed";
        case RunStatus::Stopped:   return "stopped";
        case RunStatus::Failed:    return "failed";
    }
    return "unknown";
}

inline RunStatus run_status_from_str(const std::string& s) {
    if (s == "queued")    return RunStatus::Queued;
    if (s == "running")   return RunStatus::Running;
    if (s == "completed") return RunStatus::Completed;
    if (s == "stopped")   return RunStatus::Stopped;
    if (s == "failed")    return RunStatus::Failed;
    return RunStatus::Queued;
}

// One captured progress sample (parsed from metrics.jsonl) for sparklines.
struct MetricSample {
    long long step = 0;
    double lr = 0.0;
    double loss = 0.0;
    double grad_norm = 0.0;
};

// A single training run.
struct TrainRun {
    std::string id;             // unique id (timestamp-based)
    std::string output_dir;     // absolute path to sa3-train output dir
    std::string config_path;    // absolute path to the --config json we wrote
    std::string dataset;        // human-readable label from config
    std::string model;          // model variant
    std::string adapter_type;   // lora / dora-rows / ...
    int max_steps = 0;
    int64_t started_at = 0;     // unix seconds
    int64_t finished_at = 0;    // unix seconds (0 if still running)

    RunStatus status = RunStatus::Queued;

    // Live process handle (only valid while Running / this process owns it).
    pid_t pid = -1;
    int stdout_fd = -1;         // read end of child stdout+stderr pipe

    // Rolling log captured from the child (raw stdout/stderr text).
    std::string log;
    size_t log_committed = 0;   // bytes already flushed to disk snapshot

    // Latest parsed metrics (for quick status without re-reading the file).
    MetricSample latest;

    // Whether a child process is currently owned by THIS process.
    bool owned_by_us() const { return pid > 0; }
};

// ---- small filesystem / JSON helpers ---------------------------------------

inline std::string fs_to_string(const std::filesystem::path& p) { return p.string(); }

inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;
        }
    }
    return out;
}

inline int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// List files in a directory matching any of the given substring patterns.
inline std::vector<std::filesystem::path> list_artifacts(
    const std::string& dir, const std::vector<std::string>& patterns) {
    std::vector<std::filesystem::path> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return out;
    for (auto it = std::filesystem::directory_iterator(dir, ec);
         it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (ec) break;
        const std::string name = it->path().filename().string();
        for (const auto& pat : patterns) {
            if (name.find(pat) != std::string::npos) { out.push_back(it->path()); break; }
        }
    }
    return out;
}

// Parse a metrics.jsonl line into a MetricSample (best-effort). Returns false if
// the line isn't valid JSON or lacks the fields we need.
inline bool parse_metric_line(const std::string& line, MetricSample& out) {
    if (line.empty()) return false;
    yyjson_doc* doc = yyjson_read(line.c_str(), line.size(), 0);
    if (!doc) return false;
    yyjson_val* root = yyjson_doc_get_root(doc);
    bool ok = false;
    if (root && yyjson_is_obj(root)) {
        yyjson_val* v_step = yyjson_obj_get(root, "update");
        yyjson_val* v_lr   = yyjson_obj_get(root, "lr");
        yyjson_val* v_loss = yyjson_obj_get(root, "loss");
        yyjson_val* v_gn   = yyjson_obj_get(root, "grad_norm");
        if (v_step && yyjson_is_num(v_step)) {
            out.step = (long long)yyjson_get_num(v_step);
            if (v_lr)   out.lr = yyjson_get_num(v_lr);
            if (v_loss) out.loss = yyjson_get_num(v_loss);
            if (v_gn)   out.grad_norm = yyjson_get_num(v_gn);
            ok = true;
        }
    }
    yyjson_doc_free(doc);
    return ok;
}

// ---- Registry --------------------------------------------------------------

// Holds all known runs; persists a lightweight index to disk so history
// survives a web-process restart. Not thread-safe by itself — the caller
// serializes with a mutex.
struct RunRegistry {
    std::vector<TrainRun> runs;

    // Add (or replace by id) a run.
    void upsert(TrainRun r) {
        for (auto& e : runs) {
            if (e.id == r.id) { e = std::move(r); return; }
        }
        runs.push_back(std::move(r));
    }

    TrainRun* find(const std::string& id) {
        for (auto& e : runs) if (e.id == id) return &e;
        return nullptr;
    }

    // The single owned, currently-running run (if any).
    TrainRun* active_run() {
        for (auto& e : runs) if (e.status == RunStatus::Running && e.owned_by_us()) return &e;
        return nullptr;
    }

    // Persist the index (metadata only) to `path`.
    bool save(const std::string& path) const {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        std::ofstream f(path, std::ios::trunc);
        if (!f) return false;
        f << "{\"version\":1,\"runs\":[";
        for (size_t i = 0; i < runs.size(); ++i) {
            const auto& r = runs[i];
            if (i) f << ",";
            f << "{\"id\":\"" << json_escape(r.id) << "\""
               << ",\"output_dir\":\"" << json_escape(r.output_dir) << "\""
               << ",\"config_path\":\"" << json_escape(r.config_path) << "\""
               << ",\"dataset\":\"" << json_escape(r.dataset) << "\""
               << ",\"model\":\"" << json_escape(r.model) << "\""
               << ",\"adapter_type\":\"" << json_escape(r.adapter_type) << "\""
               << ",\"max_steps\":" << r.max_steps
               << ",\"started_at\":" << r.started_at
               << ",\"finished_at\":" << r.finished_at
               << ",\"status\":\"" << run_status_str(r.status) << "\"}";
        }
        f << "]}";
        return (bool)f;
    }

    // Load the index from `path`. Existing run dirs are re-adopted as history.
    bool load(const std::string& path) {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) return false;
        std::ifstream f(path);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        std::string buf = ss.str();
        yyjson_doc* doc = yyjson_read(buf.c_str(), buf.size(), 0);
        if (!doc) return false;
        yyjson_val* root = yyjson_doc_get_root(doc);
        bool ok = false;
        if (root && yyjson_is_obj(root)) {
            yyjson_val* arr = yyjson_obj_get(root, "runs");
            if (arr && yyjson_is_arr(arr)) {
                size_t idx, max = yyjson_arr_size(arr);
                yyjson_val* v;
                yyjson_arr_foreach(arr, idx, max, v) {
                    if (!yyjson_is_obj(v)) continue;
                    auto get = [&](const char* k) -> std::string {
                        yyjson_val* x = yyjson_obj_get(v, k);
                        return (x && yyjson_is_str(x)) ? std::string(yyjson_get_str(x)) : std::string();
                    };
                    auto get_i = [&](const char* k, int64_t d) -> int64_t {
                        yyjson_val* x = yyjson_obj_get(v, k);
                        return (x && yyjson_is_num(x)) ? (int64_t)yyjson_get_num(x) : d;
                    };
                    TrainRun r;
                    r.id = get("id");
                    r.output_dir = get("output_dir");
                    r.config_path = get("config_path");
                    r.dataset = get("dataset");
                    r.model = get("model");
                    r.adapter_type = get("adapter_type");
                    r.max_steps = (int)get_i("max_steps", 0);
                    r.started_at = get_i("started_at", 0);
                    r.finished_at = get_i("finished_at", 0);
                    r.status = run_status_from_str(get("status"));
                    // Any run marked running in a stale index (process restarted)
                    // is no longer owned by us — treat as failed/interrupted.
                    if (r.status == RunStatus::Running || r.status == RunStatus::Queued) {
                        r.status = RunStatus::Failed;
                    }
                    r.pid = -1;
                    r.stdout_fd = -1;
                    if (!r.id.empty()) runs.push_back(std::move(r));
                }
                ok = true;
            }
        }
        yyjson_doc_free(doc);
        return ok;
    }
};

}  // namespace sa3_train_web
