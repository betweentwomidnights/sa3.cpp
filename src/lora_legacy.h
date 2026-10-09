#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sa3 {

struct LegacyDoraShape {
    int64_t in, out, magnitude_elements;
    int axis = -1; // original PyTorch norm dimension: 0=columns, 1=rows; -1=flattened
};

// Gary's previous PyTorch trainer wrote the legacy name "dora". Resolve its
// normalization axis before conversion/dispatch, rather than treating it as a
// new family. Original (out,1)/(1,in) magnitudes carry the axis explicitly;
// flattened magnitudes can identify it on rectangular weights. Square-only
// flattened adapters retain the Python loader's paper-correct rows default.
inline std::string resolve_legacy_dora(const std::vector<LegacyDoraShape>& shapes) {
    if (shapes.empty()) throw std::runtime_error("[lora] legacy dora has no paired magnitude/A/B tensors");
    int axis = -1;
    for (const auto& s : shapes) {
        int current = s.axis;
        if (s.magnitude_elements != s.in && s.magnitude_elements != s.out)
            throw std::runtime_error("[lora] legacy dora magnitude does not match input/output dimensions");
        if (current == -1 && s.in != s.out)
            current = s.magnitude_elements == s.out ? 1 : 0;
        if (current != -1) {
            if (s.magnitude_elements != (current == 1 ? s.out : s.in))
                throw std::runtime_error("[lora] legacy dora magnitude shape conflicts with its normalization axis");
            if (axis != -1 && axis != current)
                throw std::runtime_error("[lora] legacy dora mixes row and column magnitude shapes");
            axis = current;
        }
    }
    return axis == 0 ? "dora-cols" : "dora-rows";
}

} // namespace sa3
