#include "lora.h"
#include "train_checkpoint.h"

#include <cstdio>
#include <filesystem>
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
    sa3::TrainLoraState st;
    st.adapter_type = "lora";
    st.rank = 2;
    st.alpha = 4.0f;
    sa3::TrainLoraParam p;
    p.target.weight_name = "dit.0.self.qkv.weight";
    p.target.stem = "dit.0.self.qkv";
    p.target.in = 3;
    p.target.out = 5;
    p.lora_A = {0, 1, 2, 3, 4, 5};
    p.lora_B = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    st.params.push_back(p);
    const fs::path out = fs::temp_directory_path() / "sa3_train_checkpoint_test.gguf";
    std::string err;
    fails += expect(sa3::write_train_lora_gguf(st, out.string(), err), err.c_str());
    sa3::LoraAdapter loaded = sa3::load_lora(out.string().c_str(), 0.75f);
    fails += expect(loaded.type == "lora", "loaded type");
    fails += expect(loaded.rank == 2, "loaded rank");
    fails += expect(loaded.alpha > 3.99f && loaded.alpha < 4.01f, "loaded alpha");
    fails += expect(loaded.strength > 0.74f && loaded.strength < 0.76f, "loaded strength");
    ggml_tensor* A = loaded.gguf.get("dit.0.self.qkv.lora_A");
    ggml_tensor* B = loaded.gguf.get("dit.0.self.qkv.lora_B");
    fails += expect(A->ne[0] == 3 && A->ne[1] == 2, "A shape");
    fails += expect(B->ne[0] == 2 && B->ne[1] == 5, "B shape");
    loaded.gguf.free();
    sa3::TrainLoraState resumed;
    err.clear();
    fails += expect(sa3::load_train_lora_gguf(out.string(), resumed, err), err.c_str());
    fails += expect(resumed.adapter_type == "lora", "resumed type");
    fails += expect(resumed.rank == 2 && resumed.params.size() == 1, "resumed shape metadata");
    fails += expect(resumed.params[0].target.stem == "dit.0.self.qkv", "resumed stem");
    fails += expect(resumed.params[0].lora_A == p.lora_A, "resumed A values");
    fails += expect(resumed.params[0].lora_B == p.lora_B, "resumed B values");
    fs::remove(out);
    if (fails) return 1;
    std::printf("train_checkpoint_test: ok\n");
    return 0;
}
