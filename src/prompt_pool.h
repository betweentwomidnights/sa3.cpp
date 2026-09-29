#pragma once

#include <cctype>
#include <regex>
#include <string>

namespace sa3 {

// Captions can carry a tempo tag for training, but Studio supplies tempo from its BPM control.
inline std::string strip_prompt_bpm(const std::string& prompt) {
    static const std::regex bpm_tag(
        R"(\b(?:[0-9]+(?:\.[0-9]+)?\s*bpm|bpm\s*[:=]?\s*[0-9]+(?:\.[0-9]+)?)\b)",
        std::regex_constants::icase);
    if (!std::regex_search(prompt, bpm_tag)) return prompt;
    auto trim = [](const std::string& value) {
        size_t first = 0, last = value.size();
        while (first < last && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
        while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) --last;
        return value.substr(first, last - first);
    };

    std::string cleaned;
    size_t start = 0;
    while (start < prompt.size()) {
        const size_t end = prompt.find_first_of(",;", start);
        const std::string part = trim(prompt.substr(start, end == std::string::npos ? end : end - start));
        const std::string without_bpm = trim(std::regex_replace(part, bpm_tag, ""));
        if (!without_bpm.empty()) {
            if (!cleaned.empty()) cleaned += ", ";
            cleaned += without_bpm;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return cleaned;
}

}  // namespace sa3
