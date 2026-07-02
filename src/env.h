#pragma once

#include <cstdlib>
#include <fstream>
#include <string>

namespace sa3 {

inline std::string trim_ascii(std::string s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    const size_t b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

// Load KEY=VALUE lines from ./.env (or $SA3_ENV_FILE) into the environment without overriding
// real environment variables. Lines may start with `export `, use # comments, and quote values.
inline void load_dotenv() {
    const char* ef = std::getenv("SA3_ENV_FILE");
    std::ifstream f(ef && *ef ? ef : ".env");
    if (!f) return;

    std::string line;
    while (std::getline(f, line)) {
        std::string l = trim_ascii(line);
        if (l.empty() || l[0] == '#') continue;
        if (l.rfind("export ", 0) == 0) l = trim_ascii(l.substr(7));
        const size_t eq = l.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim_ascii(l.substr(0, eq));
        std::string v = trim_ascii(l.substr(eq + 1));
        if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front())
            v = v.substr(1, v.size() - 2);
        if (k.empty() || std::getenv(k.c_str())) continue;
#ifdef _WIN32
        _putenv_s(k.c_str(), v.c_str());
#else
        setenv(k.c_str(), v.c_str(), 1);
#endif
    }
}

} // namespace sa3
