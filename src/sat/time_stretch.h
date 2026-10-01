#pragma once

#include "signalsmith-stretch/signalsmith-stretch.h"

#include <stdexcept>
#include <vector>

namespace sa3::sat {

// Offline, pitch-preserving stretch of tightly packed planar audio. The exact()
// helper handles pre-roll and flush so the returned buffer has no latency tail.
inline std::vector<float> stretch_planar_exact(const std::vector<float>& audio,
                                               int channels, int input_samples,
                                               int output_samples, int sample_rate) {
    if (channels < 1 || input_samples < 1 || output_samples < 1 || sample_rate < 1 ||
        audio.size() != (size_t)channels * input_samples)
        throw std::invalid_argument("invalid SAT time-stretch geometry");
    if (input_samples == output_samples) return audio;

    signalsmith::stretch::SignalsmithStretch<float> stretch(0);
    stretch.presetDefault(channels, (float)sample_rate);
    std::vector<float> output((size_t)channels * output_samples);
    std::vector<const float*> input_planes(channels);
    std::vector<float*> output_planes(channels);
    for (int channel = 0; channel < channels; ++channel) {
        input_planes[channel] = audio.data() + (size_t)channel * input_samples;
        output_planes[channel] = output.data() + (size_t)channel * output_samples;
    }
    if (!stretch.exact(input_planes, input_samples, output_planes, output_samples))
        throw std::runtime_error("SAT time-stretch source is too short for the selected tempo");
    return output;
}

}  // namespace sa3::sat
