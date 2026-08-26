// sa3-lora-convert: native C++ equivalent of tools/convert_lora.py (no Python).
// Converts an exported LoRA (safetensors + json from lora_ckpt_export.py) into a sa3.cpp LoRA gguf.
//
//   sa3-lora-convert --in loras/kev --out models/lora-kev-f32.gguf
//   sa3-lora-convert --safetensors a.safetensors --json a.json --out a.gguf
#include "lora_convert.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    std::string safetensors, json, out, in_base;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) in_base = argv[++i];
        else if (!strcmp(argv[i], "--safetensors") && i + 1 < argc) safetensors = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    }
    if (!in_base.empty()) {   // --in <basename> => <basename>.safetensors [+ <basename>.json]
        if (safetensors.empty()) safetensors = in_base + ".safetensors";
        // The sidecar is optional now: autoencoder adapters carry their config in the
        // safetensors' own __metadata__ and ship as a single file, so only pair one up if present.
        if (json.empty() && std::ifstream(in_base + ".json").good()) json = in_base + ".json";
    }
    if (safetensors.empty() || out.empty()) {
        fprintf(stderr, "usage: sa3-lora-convert (--in <basename> | --safetensors <f> [--json <f>]) --out <gguf>\n"
                        "  --json is the sidecar a DiT export needs; omit it for an adapter carrying its\n"
                        "  own lora_config metadata (decoder/encoder adapters do).\n");
        return 2;
    }

    std::string err;
    if (!sa3::convert_lora_safetensors(safetensors, json, out, err)) {
        fprintf(stderr, "sa3-lora-convert: %s\n", err.c_str());
        return 1;
    }
    printf("wrote %s\n", out.c_str());
    return 0;
}
