// sa3-train: native ggML LoRA training for Stable Audio 3 DiT adapters.
//
// The run itself lives in train_job.h so libsa3 can drive the same code; this is argv, printing,
// and the exit code.
#include "env.h"
#include "train_config.h"
#include "train_job.h"
#include "train_cli_control.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    sa3::load_dotenv();
    sa3::TrainConfig cfg;
    // Match the rest of the CLI: env.cmd/env.ps1 sets this once, while an explicit flag wins.
    if (const char* models = std::getenv("SA3_MODELS_DIR"); models && *models) cfg.models_dir = models;
    std::string err;
    sa3::TrainCliControl control;
    std::vector<char*> training_args;
    if (!control.parse(argc, argv, training_args, err)) {
        std::fprintf(stderr, "error: %s\n%s", err.c_str(), control.usage());
        return 2;
    }
    if (control.info) { std::fputs(control.capabilities(), stdout); return 0; }
    if (!sa3::train_parse_args((int)training_args.size(), training_args.data(), cfg, err)) {
        std::fprintf(stderr, "%s", sa3::train_config_usage(argv[0]).c_str());
        std::fprintf(stderr, "%s", control.usage());
        if (err != "help requested") std::fprintf(stderr, "error: %s\n", err.c_str());
        return err == "help requested" ? 0 : 2;
    }
    if (cfg.cpu_threads == 0) cfg.cpu_threads = sa3::cpu_threads_from_env();
    sa3::train_finalize_defaults(cfg);
    control.max_steps = cfg.max_steps;
    try {
        control.publish();
        if (control.should_cancel()) {
            control.status = control.phase = "cancelled";
            control.message = "Cancelled before loading models.";
            control.publish();
            std::puts("[train] cancelled before loading models");
            return 0;
        }
    }
    catch (const std::exception& e) { std::fprintf(stderr, "sa3-train: %s\n", e.what()); return 1; }

    sa3::TrainHooks hooks;
    hooks.log = [&control](const std::string& line) {
        std::fputs(line.c_str(), stdout); std::fflush(stdout);
        control.message = line;
        if (line.find("[pre-encode]") == 0) control.phase = "pre-encode";
        else if (line.find("epoch ") == 0) control.phase = "training";
        else if (line.find("checkpoint:") == 0 || line.find("final checkpoint:") == 0) control.phase = "saving";
        control.status = "running";
        control.publish();
    };
    hooks.should_cancel = [&control]() { return control.should_cancel(); };
    hooks.on_step = [&control](const sa3::TrainStepReport& r) {
        control.step = r.step; control.epoch = r.epoch; control.max_steps = r.max_steps;
        control.loss = r.loss; control.learning_rate = r.learning_rate; control.step_seconds = r.step_seconds;
        control.phase = "training"; control.status = "running";
        control.publish();
    };
    for (int i = 0; i < argc; ++i) {
        if (i) hooks.command_line += ' ';
        hooks.command_line += argv[i];
    }

    sa3::TrainResult result;
    const bool ok = sa3::run_training(cfg, hooks, result, err);
    control.status = !ok ? "failed" : result.cancelled ? "cancelled" : "completed";
    control.phase = control.status;
    control.error = ok ? "" : err;
    control.step = result.steps > 0 ? result.steps : control.step;
    control.final_adapter = result.final_adapter;
    control.adapter_checkpoint = result.last_adapter_checkpoint;
    control.state_checkpoint = result.last_state_checkpoint;
    try { control.publish(); }
    catch (const std::exception& e) { std::fprintf(stderr, "sa3-train: %s\n", e.what()); return 1; }
    if (!ok) {
        std::fprintf(stderr, "sa3-train: %s\n", err.c_str());
        return 1;
    }
    if (result.cancelled) std::printf("\n[train] cancelled at step %d\n", result.steps);
    // A cancel during pre-encode stops before any adapter exists, so there is nothing to preview.
    // Cancelling mid-training still writes one, which is why this is a check and not an else.
    if (result.preview_command.empty())
        std::printf("\n[train] no adapter was written (cancelled before the first step)\n");
    else
        std::printf("\n[train] try your adapter now:\n%s\n", result.preview_command.c_str());
    return 0;
}
