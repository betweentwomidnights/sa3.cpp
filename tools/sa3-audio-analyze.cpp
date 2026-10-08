// CPU-only audio metadata CLI. No models, backend DLLs or Python required.
#include "audio_analysis.h"
#include "yyjson.h"
#include <cstdlib>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

static int run(const std::vector<std::string>& args) {
    if (args.size() == 2 && args[1] == "--control-info") {
        puts("{\"schema_version\":1,\"cpu_only\":true,\"wav_input\":true}");
        return 0;
    }
    yyjson_mut_doc* doc = yyjson_mut_doc_new(nullptr);
    auto* obj = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, obj);
    int status = 0;
    try {
        if (args.size() != 3 || args[1] != "--in")
            throw std::runtime_error("usage: sa3-audio-analyze --in <audio.wav>");
        const auto result = sa3::analysis::analyze_wav(args[2]);
        yyjson_mut_obj_add_bool(doc, obj, "ok", true);
        yyjson_mut_obj_add_uint(doc, obj, "schema_version", 1);
        if (result.bpm) yyjson_mut_obj_add_int(doc, obj, "bpm", *result.bpm);
        else yyjson_mut_obj_add_null(doc, obj, "bpm");
        yyjson_mut_obj_add_strcpy(doc, obj, "keyscale", result.keyscale.c_str());
        std::string suggestion;
        if (result.bpm) suggestion = std::to_string(*result.bpm) + " bpm";
        if (!result.keyscale.empty()) suggestion += (suggestion.empty() ? "" : ", ") + result.keyscale;
        yyjson_mut_obj_add_strcpy(doc, obj, "suggestion", suggestion.c_str());
        yyjson_mut_obj_add_str(doc, obj, "bpm_source", result.bpm ? "local" : "missing");
        yyjson_mut_obj_add_str(doc, obj, "key_source", result.keyscale.empty() ? "missing" : "local");
        auto confidence = [&](const char* key, const std::optional<double>& value) {
            if (value) yyjson_mut_obj_add_real(doc, obj, key, std::nearbyint(*value*10000)/10000);
            else yyjson_mut_obj_add_null(doc, obj, key);
        };
        confidence("bpm_confidence", result.bpm_confidence);
        confidence("key_confidence", result.key_confidence);
    } catch (const std::exception& error) {
        status = 1;
        yyjson_mut_obj_add_bool(doc, obj, "ok", false);
        yyjson_mut_obj_add_strcpy(doc, obj, "error", error.what());
    }
    char* json = yyjson_mut_write(doc, 0, nullptr);
    if (!json) { yyjson_mut_doc_free(doc); return 1; }
    puts(json);
    free(json);
    yyjson_mut_doc_free(doc);
    return status;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
        if (size <= 0) return 1;
        std::string text(size, '\0');
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, text.data(), size, nullptr, nullptr);
        text.pop_back();
        args.push_back(std::move(text));
    }
    return run(args);
}
#else
int main(int argc, char** argv) { return run(std::vector<std::string>(argv, argv + argc)); }
#endif
