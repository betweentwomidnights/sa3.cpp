#include "train_generation.h"

#include <cstdio>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main() {
    sa3::ModelPaths p;
    p.tok = "tok.gguf";
    p.t5 = "t5.gguf";
    p.cond = "cond.gguf";
    p.dit = "dit.gguf";
    p.same = "same.gguf";
    std::string cmd = sa3::build_sa3_generate_lora_command("sa3-generate", p, "hello world", "adapter.gguf", "out.wav", 8, 2, 42);
    int fails = 0;
    fails += expect(cmd.find("--lora") != std::string::npos && cmd.find("adapter.gguf") != std::string::npos, "lora arg");
    fails += expect(cmd.find("--prompt") != std::string::npos && cmd.find("hello world") != std::string::npos, "prompt arg");
    fails += expect(cmd.find("--seed 42") != std::string::npos, "seed arg");
    if (fails) return 1;
    std::printf("train_generation_test: ok\n");
    return 0;
}
