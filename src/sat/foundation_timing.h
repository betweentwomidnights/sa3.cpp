// sat/foundation_timing.h -- Foundation-1's trained musical-duration policy.
//
// This is deliberately a small, model-profile helper rather than part of the DiT or
// Oobleck graphs.  Applications can share RoyalCities' exact 4/8-bar geometry without
// taking on a tempo-stretching dependency.
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace sa3::sat {

inline constexpr std::array<int, 7> kFoundationBpms = {
    100, 110, 120, 128, 130, 140, 150,
};

struct FoundationTiming {
    int bars = 0;
    int bpm = 0;
    int output_samples = 0;              // exact musical crop
    double output_seconds = 0.0;
    float conditioning_seconds_total = 0.0f; // RoyalCities: whole-second ceiling
    int latent_frames = 0;               // conditioned canvas, padded to Oobleck stride
};

inline bool is_foundation_bpm(int bpm) {
    for (int supported : kFoundationBpms)
        if (bpm == supported) return true;
    return false;
}

inline bool is_foundation_bar_count(int bars) {
    return bars == 4 || bars == 8;
}

inline FoundationTiming resolve_foundation_timing(int bars, int bpm,
                                                   int sample_rate = 44100,
                                                   int downsampling_ratio = 2048) {
    if (!is_foundation_bar_count(bars))
        throw std::invalid_argument("Foundation-1 supports exactly 4 or 8 bars");
    if (!is_foundation_bpm(bpm))
        throw std::invalid_argument(
            "Foundation-1 BPM must be 100, 110, 120, 128, 130, 140, or 150");
    if (sample_rate <= 0 || downsampling_ratio <= 0)
        throw std::invalid_argument("invalid Foundation-1 audio geometry");

    FoundationTiming result;
    result.bars = bars;
    result.bpm = bpm;
    result.output_seconds = 60.0 * 4.0 * bars / bpm;
    result.output_samples = (int)std::llround(result.output_seconds * sample_rate);
    result.conditioning_seconds_total =
        (float)std::ceil((double)result.output_samples / sample_rate);
    const int canvas_samples =
        (int)std::llround(result.conditioning_seconds_total * sample_rate);
    result.latent_frames =
        (canvas_samples + downsampling_ratio - 1) / downsampling_ratio;
    return result;
}

} // namespace sa3::sat
