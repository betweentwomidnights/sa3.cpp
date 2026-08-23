// encoding.h - the --encoding vocabulary, shared by inference and training.
//
// Split out of sa3_pipeline.h so the training config can validate an encoding without pulling in
// ggml: there must be exactly one table of what an encoding is called and what it resolves to.
#pragma once

#include <string>
#include <vector>

namespace sa3 {

// Canonical file-suffix label for an --encoding value, or "" if unrecognised.
// Suffixes follow the Encoding field in docs/DISTRIBUTION.md (gguf-spec naming, same
// spelling llama.cpp uses): Q4_K_M / Q5_K_M, not Q4_KM. The compact spellings are
// accepted as input aliases so older locally-quantized files and commands still work,
// but only the canonical form is ever written or resolved.
inline std::string encoding_suffix(const std::string& encoding) {
    if (encoding == "f32"     || encoding == "F32")     return "F32";
    if (encoding == "f16"     || encoding == "F16")     return "F16";
    if (encoding == "q4_k_m"  || encoding == "Q4_K_M")  return "Q4_K_M";
    if (encoding == "q5_k_m"  || encoding == "Q5_K_M")  return "Q5_K_M";
    if (encoding == "q5_k"    || encoding == "Q5_K")    return "Q5_K";
    if (encoding == "q8_0"    || encoding == "Q8_0")    return "Q8_0";
    if (encoding == "q4_km"   || encoding == "Q4_KM")   return "Q4_K_M";   // alias
    if (encoding == "q5_km"   || encoding == "Q5_KM")   return "Q5_K_M";   // alias
    return "";
}

// Every canonical encoding label, for "what else is available" diagnostics and CLI help.
inline const std::vector<std::string>& encoding_labels() {
    static const std::vector<std::string> v = {"F16", "F32", "Q4_K_M", "Q5_K_M", "Q5_K", "Q8_0"};
    return v;
}

} // namespace sa3
