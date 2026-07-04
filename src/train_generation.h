// train_generation.h - held-out generation command helper for trained adapters.
#pragma once

#include "train_audio.h"
#include "train_config.h"
#include "train_model_paths.h"

#include <cstdlib>
#include <string>

namespace sa3 {

inline std::string build_sa3_generate_lora_command(const std::string& exe, const ModelPaths& paths,
                                                   const std::string& prompt,
                                                   const std::string& lora_path,
                                                   const std::string& out_wav,
                                                   int frames, int steps, unsigned long long seed) {
    std::string cmd = shell_quote_single(exe) +
        " --tok " + shell_quote_single(paths.tok) +
        " --t5 " + shell_quote_single(paths.t5) +
        (paths.cond.empty() ? "" : " --cond " + shell_quote_single(paths.cond)) +
        " --dit " + shell_quote_single(paths.dit) +
        " --same " + shell_quote_single(paths.same) +
        " --prompt " + shell_quote_single(prompt) +
        " --lora " + shell_quote_single(lora_path) +
        " --frames " + std::to_string(frames) +
        " --steps " + std::to_string(steps) +
        " --seed " + std::to_string(seed) +
        " --out " + shell_quote_single(out_wav);
    return cmd;
}

inline bool run_sa3_generate_lora_command(const std::string& cmd, std::string& err) {
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        err = "sa3-generate failed with exit code " + std::to_string(rc);
        return false;
    }
    return true;
}

} // namespace sa3
