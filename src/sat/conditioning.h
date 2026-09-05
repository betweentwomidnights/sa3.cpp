// sat/conditioning.h -- host-side classic stable-audio-tools conditioners.
#pragma once

#include "gguf_model.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace sa3::sat {

inline std::vector<float> number_condition(const GgufModel& W, const std::string& prefix,
                                           float value, float min_value, float max_value) {
    if (!(max_value > min_value)) throw gguf_error("invalid number-conditioner range");
    ggml_tensor* fw = W.get(prefix + "fourier.weight");
    ggml_tensor* lw = W.get(prefix + "linear.weight");
    ggml_tensor* lb = W.get(prefix + "linear.bias");
    const size_t half = (size_t)fw->ne[0];
    const size_t features = 1 + 2 * half;
    const size_t output = (size_t)lw->ne[1];
    if ((size_t)lw->ne[0] != features || (size_t)lb->ne[0] != output)
        throw gguf_error("invalid number-conditioner tensor geometry");
    std::vector<float> freq(half), weight(features * output), bias(output);
    ggml_backend_tensor_get(fw, freq.data(), 0, freq.size() * sizeof(float));
    ggml_backend_tensor_get(lw, weight.data(), 0, weight.size() * sizeof(float));
    ggml_backend_tensor_get(lb, bias.data(), 0, bias.size() * sizeof(float));
    const float normalized = (std::clamp(value, min_value, max_value) - min_value) /
                             (max_value - min_value);
    std::vector<float> input(features);
    input[0] = normalized;
    for (size_t i = 0; i < half; ++i) {
        const float phase = normalized * freq[i] * 6.2831853071795864769f;
        input[1 + i] = std::sin(phase);
        input[1 + half + i] = std::cos(phase);
    }
    std::vector<float> out(output);
    for (size_t o = 0; o < output; ++o) {
        float sum = bias[o];
        const float* row = weight.data() + o * features;
        for (size_t i = 0; i < features; ++i) sum += row[i] * input[i];
        out[o] = sum;
    }
    return out;
}

inline std::vector<float> seconds_total_condition(const GgufModel& W, float seconds) {
    return number_condition(W, "conditioner.seconds_total.", seconds,
                            W.f32("sat.conditioner.seconds_total.min"),
                            W.f32("sat.conditioner.seconds_total.max"));
}

inline std::vector<float> seconds_start_condition(const GgufModel& W, float seconds) {
    return number_condition(W, "conditioner.seconds_start.", seconds,
                            W.f32("sat.conditioner.seconds_start.min"),
                            W.f32("sat.conditioner.seconds_start.max"));
}

} // namespace sa3::sat
