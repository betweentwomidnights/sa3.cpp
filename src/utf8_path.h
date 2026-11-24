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

// Replace every byte sequence that is not valid UTF-8 with U+FFFD. Text bound for JSON goes through
// this, because yyjson refuses invalid UTF-8 and one stray system-code-page byte (a localized OS error
// message, a name on a Windows older than 1903) would otherwise lose the whole document.
inline std::string utf8_sanitize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    const auto* s = reinterpret_cast<const unsigned char*>(text.data());
    const size_t n = text.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c = s[i];
        size_t len = 0;
        unsigned int lo = 0x80, hi = 0xBF;   // allowed range of the first continuation byte
        if (c < 0x80) len = 1;
        else if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c == 0xE0) { len = 3; lo = 0xA0; }
        else if (c >= 0xE1 && c <= 0xEC) len = 3;
        else if (c == 0xED) { len = 3; hi = 0x9F; }   // no UTF-16 surrogates
        else if (c >= 0xEE && c <= 0xEF) len = 3;
        else if (c == 0xF0) { len = 4; lo = 0x90; }
        else if (c >= 0xF1 && c <= 0xF3) len = 4;
        else if (c == 0xF4) { len = 4; hi = 0x8F; }   // nothing above U+10FFFF
        bool ok = len > 0 && i + len <= n;
        for (size_t k = 1; ok && k < len; ++k) {
            const unsigned char b = s[i + k];
            ok = k == 1 ? (b >= lo && b <= hi) : (b >= 0x80 && b <= 0xBF);
        }
        if (ok) { out.append(text, i, len); i += len; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

} // namespace sa3
