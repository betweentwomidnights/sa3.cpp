#include "train_model_paths.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main() {
    namespace fs = std::filesystem;
    int fails = 0;
    const fs::path root = fs::temp_directory_path() / "sa3_train_model_paths_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const char* names[] = {
        "t5gemma-b-b-ul2-v1.0-vocab.gguf",
        "t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf",
        "stable-audio-3-medium-conditioner-v1.0-F32.gguf",
        "stable-audio-3-medium-dit-1.5B-v1.0-F16.gguf",
        "stable-audio-3-medium-base-dit-1.5B-v1.0-F16.gguf",
        "stable-audio-3-medium-same-1.5B-v1.0-F16.gguf",
        "stable-audio-3-small-music-conditioner-v1.0-F32.gguf",
        "stable-audio-3-small-music-dit-0.5B-v1.0-F16.gguf",
        "stable-audio-3-small-music-base-dit-0.5B-v1.0-F16.gguf",
        "stable-audio-3-small-music-same-s-v1.0-F16.gguf",
    };
    for (const char* n : names) std::ofstream(root / n) << "stub";

    sa3::TrainConfig cfg;
    cfg.models_dir = root.string();
    sa3::ModelPaths p;
    std::string err;
    fails += expect(sa3::resolve_train_model_paths(cfg, p, err), "convention model paths resolve");
    fails += expect(p.tok.find("vocab") != std::string::npos, "tokenizer resolved");
    fails += expect(p.cond.find("conditioner") != std::string::npos, "conditioner resolved");
    fails += expect(p.dit.find("medium-base-dit") != std::string::npos, "medium-base training dit resolved");

    fs::remove(root / "stable-audio-3-medium-base-dit-1.5B-v1.0-F16.gguf");
    err.clear();
    p = sa3::ModelPaths{};
    fails += expect(!sa3::resolve_train_model_paths(cfg, p, err), "inference medium dit rejected for training");
    fails += expect(err.find("medium-base") != std::string::npos, "missing medium-base hint");

    sa3::TrainConfig small_cfg;
    small_cfg.models_dir = root.string();
    small_cfg.model_variant = "small-music";
    err.clear();
    p = sa3::ModelPaths{};
    fails += expect(sa3::resolve_train_model_paths(small_cfg, p, err), "small convention paths resolve");
    fails += expect(p.dit.find("small-music-base-dit") != std::string::npos,
                    "small-music-base training dit resolved");

    fs::remove(root / "stable-audio-3-small-music-base-dit-0.5B-v1.0-F16.gguf");
    err.clear();
    p = sa3::ModelPaths{};
    fails += expect(!sa3::resolve_train_model_paths(small_cfg, p, err),
                    "inference small-music dit rejected for training");
    fails += expect(err.find("small-music-base") != std::string::npos, "missing small-base hint");

    sa3::TrainConfig explicit_cfg;
    explicit_cfg.tok_path = "tok.gguf";
    explicit_cfg.t5_path = "t5.gguf";
    explicit_cfg.dit_path = "dit.gguf";
    explicit_cfg.same_path = "same.gguf";
    err.clear();
    p = sa3::ModelPaths{};
    fails += expect(sa3::resolve_train_model_paths(explicit_cfg, p, err), "explicit paths resolve");
    fails += expect(p.tok == "tok.gguf" && p.same == "same.gguf", "explicit paths retained");

    sa3::TrainConfig missing;
    missing.models_dir = (root / "none").string();
    err.clear();
    p = sa3::ModelPaths{};
    fails += expect(!sa3::resolve_train_model_paths(missing, p, err), "missing convention paths rejected");
    fails += expect(err.find("run: python tools/download_models.py") != std::string::npos, "missing model hint");

    fs::remove_all(root);
    if (fails) return 1;
    std::printf("train_model_paths_test: ok\n");
    return 0;
}
