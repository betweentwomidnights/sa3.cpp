#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <string>

#ifndef SA3_VERSION_STRING
#define SA3_VERSION_STRING "0.0.0"
#endif

namespace sa3 {

inline const char* runtime_version() { return SA3_VERSION_STRING; }

inline std::string runtime_json_quote(const char* input) {
    std::string out = "\"";
    for (const unsigned char* p = (const unsigned char*)(input ? input : ""); *p; ++p) {
        const unsigned char c = *p;
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            constexpr char hex[] = "0123456789abcdef";
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else out += (char)c;
    }
    return out + '"';
}

// Model-free host contract. Keep /health and --control-info on one capability
// definition. This probe must not initialize ggml, read .env, or bind a port.
inline const char* sa3_server_capabilities_json() {
    return "{\"fixed_prefix\":true,\"request_splice\":true,"
           "\"conditioning_duration\":true,\"model_lifecycle\":true}";
}

inline std::string sa3_server_control_info_json() {
    return std::string("{\"schema_version\":1,\"service\":\"sa3\",\"version\":") +
        runtime_json_quote(runtime_version()) + ",\"capabilities\":" +
        sa3_server_capabilities_json() + "}";
}

// Gary4local checks this before installing a model or binding a service port.
// Use the registry name (CUDA/Vulkan/CPU), not the numbered device name.
inline std::string runtime_props_json(const char* service) {
    ggml_backend_load_all();
    std::string body = "{\"success\":true,\"service\":" + runtime_json_quote(service) +
        ",\"version\":" + runtime_json_quote(runtime_version()) + ",\"devices\":[";
    bool first = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const auto kind = ggml_backend_dev_type(dev);
        if (kind != GGML_BACKEND_DEVICE_TYPE_CPU && kind != GGML_BACKEND_DEVICE_TYPE_GPU &&
            kind != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        if (!first) body += ',';
        first = false;
        body += "{\"name\":" + runtime_json_quote(ggml_backend_dev_name(dev)) +
            ",\"description\":" + runtime_json_quote(ggml_backend_dev_description(dev)) +
            ",\"backend\":" + runtime_json_quote(reg ? ggml_backend_reg_name(reg) : "") +
            ",\"type\":" + runtime_json_quote(kind == GGML_BACKEND_DEVICE_TYPE_CPU ? "cpu" :
                kind == GGML_BACKEND_DEVICE_TYPE_IGPU ? "integrated_gpu" : "gpu") +
            ",\"memory_free_bytes\":" + std::to_string(free_bytes) +
            ",\"memory_total_bytes\":" + std::to_string(total_bytes) + "}";
    }
    return body + "]}";
}

}  // namespace sa3
