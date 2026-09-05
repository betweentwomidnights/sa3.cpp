// sat/profiles.h -- optional application policy for named SAT checkpoints.
#pragma once

#include "sat/foundation_timing.h"
#include "sat/pipeline.h"

#include <string>

namespace sa3::sat {

inline std::string foundation_prompt(const std::string& description, int bars, int bpm) {
    const std::string suffix = std::to_string(bars) + " Bars, " +
                               std::to_string(bpm) + " BPM";
    return description.empty() ? suffix : description + ", " + suffix;
}

// Apply only Foundation's trained duration contract. Sampler choice remains a named
// application profile because the RoyalCities and Gary defaults are both useful.
inline FoundationTiming apply_foundation_timing(GenerateParams& params, int bars, int bpm,
                                                 bool append_prompt_timing = true) {
    const FoundationTiming timing = resolve_foundation_timing(bars, bpm);
    params.seconds = (float)timing.output_seconds;
    params.seconds_total = timing.conditioning_seconds_total;
    params.output_samples = timing.output_samples;
    params.frames = timing.latent_frames;
    if (append_prompt_timing)
        params.prompt = foundation_prompt(params.prompt, bars, bpm);
    return timing;
}

inline void apply_foundation_royalcities_sampler(GenerateParams& params) {
    params.sampler = Sampler::Dpmpp3mSde;
    params.sigma_min = 0.01f;
    params.sigma_max = 100.0f;
    params.steps = 100;
    params.cfg_scale = 7.0f;
}

inline void apply_foundation_gary_sampler(GenerateParams& params) {
    params.sampler = Sampler::Dpmpp2mSde;
    params.sigma_min = 0.5f;
    params.sigma_max = 50.0f;
    params.steps = 100;
    params.cfg_scale = 7.0f;
}

} // namespace sa3::sat
