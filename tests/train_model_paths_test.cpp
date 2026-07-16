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

static bool write_metadata_stub(const std::filesystem::path& path, const char* finetune,
                                bool include_marker, bool training_base) {
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "sa3-dit");
    if (finetune) gguf_set_val_str(g, "general.finetune", finetune);
    if (include_marker) gguf_set_val_bool(g, "dit.training_base", training_base);
    const bool ok = gguf_write_to_file(g, path.string().c_str(), false);
    gguf_free(g);
    return ok;
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

    const fs::path marked = root / "marked-medium-base.gguf";
    const fs::path unmarked = root / "unmarked-medium.gguf";
    const fs::path false_marker = root / "false-medium-base.gguf";
    const fs::path wrong_family = root / "marked-small-base.gguf";
    fails += expect(write_metadata_stub(marked, "medium-base", true, true), "write marked metadata stub");
    fails += expect(write_metadata_stub(unmarked, "medium-base", false, false), "write unmarked metadata stub");
    fails += expect(write_metadata_stub(false_marker, "medium-base", true, false), "write false marker stub");
    fails += expect(write_metadata_stub(wrong_family, "small-music-base", true, true), "write wrong-family stub");
    err.clear();
    fails += expect(sa3::validate_training_base_dit_metadata(cfg, marked.string(), err),
                    "marked medium-base metadata accepted");
    err.clear();
    fails += expect(!sa3::validate_training_base_dit_metadata(cfg, unmarked.string(), err),
                    "missing training-base marker rejected");
    fails += expect(err.find("dit.training_base=true") != std::string::npos, "missing marker guidance");
    err.clear();
    fails += expect(!sa3::validate_training_base_dit_metadata(cfg, false_marker.string(), err),
                    "false training-base marker rejected");
    err.clear();
    fails += expect(!sa3::validate_training_base_dit_metadata(cfg, wrong_family.string(), err),
                    "wrong training-base family rejected");
    fails += expect(err.find("medium-base") != std::string::npos, "wrong-family guidance");

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
