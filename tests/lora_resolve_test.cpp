// A bare adapter name resolves to the file /loras lists under that name, not to whichever
// lora-<name>-*.gguf sorts first: "koan" once picked lora-koan-27-2-f32.gguf.
#include "sa3_pipeline.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

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

    // ABI paths are UTF-8. The snowman is outside every Western Windows code page.
    const std::string accented = sa3::path_utf8(dir) + "/caf\xC3\xA9 \xE2\x98\x83";
    fs::create_directories(sa3::utf8_path(accented));
    const std::string file = "lora-caf\xC3\xA9-f32.gguf";
    std::ofstream(sa3::utf8_path(accented + "/" + file)) << "x";
    const std::string got = sa3::resolve_lora_file(accented, "caf\xC3\xA9");
    if (got.empty() || !fs::exists(sa3::utf8_path(got)) || sa3::path_utf8(sa3::utf8_path(got).filename()) != file) {
        std::fprintf(stderr, "FAIL: accented adapter resolved to '%s'\n", got.c_str());
        ++fails;
    }
    const std::map<std::string, std::vector<float>> norms{{"block", {1.f, 2.f}}};
    sa3::normcache::save(got, norms);
    std::map<std::string, std::vector<float>> loaded;
    if (!sa3::normcache::load(got, loaded) || loaded != norms) {
        std::fprintf(stderr, "FAIL: norm cache did not round trip beside an accented adapter\n");
        ++fails;
    }

    fs::remove_all(dir);
    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("ok\n");
    return 0;
}
