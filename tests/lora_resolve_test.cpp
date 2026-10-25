// A bare adapter name resolves to the file /loras lists under that name, not to whichever
// lora-<name>-*.gguf sorts first: "koan" once picked lora-koan-27-2-f32.gguf.
#include "sa3_pipeline.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static int fails = 0;

static void expect_file(const std::string& dir, const std::string& name, const std::string& want) {
    const std::string got = sa3::resolve_lora_file(dir, name);
    const std::string got_name = got.empty() ? "" : std::filesystem::path(got).filename().string();
    if (got_name != want) {
        std::fprintf(stderr, "FAIL: '%s' resolved to '%s', want '%s'\n", name.c_str(), got_name.c_str(), want.c_str());
        ++fails;
    }
}

int main() {
    std::printf("lora_resolve_test\n");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "sa3_lora_resolve_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    for (const char* f : {"lora-koan-27-2-f32.gguf", "lora-koan-f32.gguf", "lora-koanx-f32.gguf",
                          "lora-kev-v2-f32.gguf", "lora-solo-take3-f32.gguf", "lora-bare.gguf"})
        std::ofstream(dir / f) << "x";
    const std::string d = dir.string();

    expect_file(d, "koan", "lora-koan-f32.gguf");            // not lora-koan-27-2-f32.gguf
    expect_file(d, "koan-27-2", "lora-koan-27-2-f32.gguf");
    expect_file(d, "koanx", "lora-koanx-f32.gguf");
    expect_file(d, "kev", "lora-kev-v2-f32.gguf");           // a version token is not part of the name
    expect_file(d, "bare", "lora-bare.gguf");
    expect_file(d, "solo", "lora-solo-take3-f32.gguf");      // no exact file: the old prefix match
    expect_file(d, "missing", "");

    fs::remove_all(dir);
    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("ok\n");
    return 0;
}
