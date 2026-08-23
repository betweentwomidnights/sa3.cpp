// sa3-train: native ggML LoRA training for Stable Audio 3 DiT adapters.
//
// The run itself lives in train_job.h so libsa3 can drive the same code; this is argv, printing,
// and the exit code.
#include "env.h"
#include "train_config.h"
#include "train_job.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    sa3::load_dotenv();
    sa3::TrainConfig cfg;
    // Match the rest of the CLI: env.cmd/env.ps1 sets this once, while an explicit flag wins.
    if (const char* models = std::getenv("SA3_MODELS_DIR"); models && *models) cfg.models_dir = models;
    std::string err;
    if (!sa3::train_parse_args(argc, argv, cfg, err)) {
        std::fprintf(stderr, "%s", sa3::train_config_usage(argv[0]).c_str());
        if (err != "help requested") std::fprintf(stderr, "error: %s\n", err.c_str());
        return err == "help requested" ? 0 : 2;
    }
    if (cfg.cpu_threads == 0) cfg.cpu_threads = sa3::cpu_threads_from_env();
    sa3::train_finalize_defaults(cfg);

    sa3::TrainHooks hooks;
    hooks.log = [](const std::string& line) { std::fputs(line.c_str(), stdout); std::fflush(stdout); };
    for (int i = 0; i < argc; ++i) {
        if (i) hooks.command_line += ' ';
        hooks.command_line += argv[i];
    }

    sa3::TrainResult result;
    if (!sa3::run_training(cfg, hooks, result, err)) {
        std::fprintf(stderr, "sa3-train: %s\n", err.c_str());
        return 1;
    }
    if (result.cancelled) std::printf("\n[train] cancelled at step %d\n", result.steps);
    std::printf("\n[train] try your adapter now:\n%s\n", result.preview_command.c_str());
    return 0;
}
