// Optional file-based controls for hosts that launch the native trainer.
// Training algorithms and checkpoint ownership remain in run_training.
#pragma once

#include "yyjson.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sa3 {

struct TrainCliControl {
    std::string progress_file;
    std::string cancel_file;
    bool info = false;
    std::string status = "starting";
    std::string phase = "loading";
    std::string message;
    std::string error;
    int step = 0, epoch = 0, max_steps = 0;
    double loss = 0, learning_rate = 0, step_seconds = 0;
    std::string final_adapter, adapter_checkpoint, state_checkpoint;

    static const char* capabilities() {
        return "{\"schema_version\":1,\"cooperative_cancel_file\":true,\"atomic_progress_file\":true}\n";
    }

    static const char* usage() {
        return "  --progress-file PATH   Atomically published JSON progress/result (schema 1)\n"
               "  --cancel-file PATH     Stop at a sample boundary when this file exists\n"
               "  --control-info         Print host-control capabilities without loading models\n";
    }

    bool parse(int argc, char** argv, std::vector<char*>& training_args, std::string& err) {
        training_args.push_back(argv[0]);
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--control-info") { info = true; continue; }
            if (a == "--progress-file" || a == "--cancel-file") {
                if (i + 1 >= argc || !*argv[i + 1] || std::string(argv[i + 1]).find("--") == 0) {
                    err = a + " requires a path"; return false;
                }
                std::string& field = a == "--progress-file" ? progress_file : cancel_file;
                if (!field.empty()) { err = a + " may be specified only once"; return false; }
                field = argv[++i];
            } else training_args.push_back(argv[i]);
        }
        if (!progress_file.empty() && !cancel_file.empty()) {
            namespace fs = std::filesystem;
            std::error_code ec;
            const auto progress = fs::absolute(fs::u8path(progress_file)).lexically_normal();
            const auto cancel = fs::absolute(fs::u8path(cancel_file)).lexically_normal();
            if (progress == cancel || fs::equivalent(progress, cancel, ec)) {
                err = "progress and cancellation files must be different"; return false;
            }
#ifdef _WIN32
            if (_wcsicmp(progress.c_str(), cancel.c_str()) == 0) {
                err = "progress and cancellation files must be different"; return false;
            }
#endif
        }
        return true;
    }

    bool should_cancel() const {
        if (cancel_file.empty()) return false;
        std::error_code ec;
        const bool exists = std::filesystem::exists(std::filesystem::u8path(cancel_file), ec);
        if (ec) throw std::runtime_error("cannot inspect cancellation file: " + ec.message());
        return exists;
    }

    std::string json() const {
        yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
        if (!doc) throw std::bad_alloc();
        yyjson_mut_val* root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_int(doc, root, "schema_version", 1);
        yyjson_mut_obj_add_str(doc, root, "status", status.c_str());
        yyjson_mut_obj_add_str(doc, root, "phase", phase.c_str());
        yyjson_mut_obj_add_str(doc, root, "message", message.c_str());
        yyjson_mut_obj_add_str(doc, root, "error", error.c_str());
        yyjson_mut_obj_add_int(doc, root, "step", step);
        yyjson_mut_obj_add_int(doc, root, "epoch", epoch);
        yyjson_mut_obj_add_int(doc, root, "max_steps", max_steps);
        // Non-finite metrics are not valid JSON. Keep the status readable so
        // hosts can report a failed numerical run rather than a parse error.
        for (const auto& metric : std::vector<std::pair<const char*, double>>{
                 {"loss",loss}, {"learning_rate",learning_rate}, {"step_seconds",step_seconds}}) {
            if (std::isfinite(metric.second)) yyjson_mut_obj_add_real(doc, root, metric.first, metric.second);
            else yyjson_mut_obj_add_null(doc, root, metric.first);
        }
        yyjson_mut_obj_add_str(doc, root, "final_adapter", final_adapter.c_str());
        yyjson_mut_obj_add_str(doc, root, "adapter_checkpoint", adapter_checkpoint.c_str());
        yyjson_mut_obj_add_str(doc, root, "state_checkpoint", state_checkpoint.c_str());
        size_t size = 0;
        char* data = yyjson_mut_write(doc, 0, &size);
        yyjson_mut_doc_free(doc);
        if (!data) throw std::runtime_error("cannot serialize training progress");
        std::string out(data, size);
        std::free(data);
        return out;
    }

    void publish() const {
        if (progress_file.empty()) return;
        namespace fs = std::filesystem;
        const fs::path path = fs::u8path(progress_file);
        if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
#ifdef _WIN32
        const auto pid = GetCurrentProcessId();
#else
        const auto pid = getpid();
#endif
        const fs::path stage = fs::u8path(progress_file + ".tmp-" + std::to_string(pid));
        try {
            const std::string data = json();
            std::ofstream file(stage, std::ios::binary | std::ios::trunc);
            if (!file || !file.write(data.data(), (std::streamsize)data.size()))
                throw std::runtime_error("cannot write training progress staging file");
            file.close();
            if (!file) throw std::runtime_error("cannot flush training progress staging file");
#ifdef _WIN32
            if (!MoveFileExW(stage.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                throw std::runtime_error("cannot publish training progress: Windows error " + std::to_string(GetLastError()));
#else
            fs::rename(stage, path);
#endif
        } catch (...) {
            std::error_code ec; fs::remove(stage, ec);
            throw;
        }
    }
};

} // namespace sa3
