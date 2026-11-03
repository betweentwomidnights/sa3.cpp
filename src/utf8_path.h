// Paths crossing the C ABI are UTF-8 on every platform. On Windows a narrow fopen, std::ifstream or
// std::filesystem::path reads a std::string in the system code page instead, so anything outside
// ASCII opens the wrong file or none. Convert at the point of use with these. (ggml_fopen and the
// gguf readers already take UTF-8.)
#pragma once

#include <filesystem>
#include <string>

namespace sa3 {

inline std::filesystem::path utf8_path(const std::string& path) { return std::filesystem::u8path(path); }

inline std::string path_utf8(const std::filesystem::path& path) { return path.u8string(); }

} // namespace sa3
