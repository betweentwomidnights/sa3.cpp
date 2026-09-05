// sampling.h -- model-agnostic host-side sampling primitives.
//
// Keep schedule construction and state updates out of model pipelines.  A pipeline owns
// graph execution and CFG; these helpers only describe the numerical integration contract.
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sa3::sampling {

// stable-audio-tools sample_rf schedule:
//   logsnr = linspace(logsnr_start, logsnr_end, steps + 1)
//   t      = sigmoid(-logsnr)
// with exact endpoint anchoring.  SAOS uses sigma_max=1, logsnr_start=-6 and
// logsnr_end=2.  Keeping this separate from SA3's distribution-shift schedule avoids
// accumulating model-family conditionals in sa3_pipeline.h.
inline std::vector<float> make_rf_logsnr_schedule(int steps,
                                                   float sigma_max = 1.0f,
                                                   float logsnr_end = 2.0f) {
    if (steps < 1) throw std::invalid_argument("sampling steps must be positive");
    if (!(sigma_max > 0.0f && sigma_max <= 1.0f))
        throw std::invalid_argument("sigma_max must be in (0, 1]");

    const float logsnr_start = sigma_max < 1.0f
        ? std::log(((1.0f - sigma_max) / sigma_max) + 1.0e-6f)
        : -6.0f;
    std::vector<float> schedule((size_t)steps + 1);
    for (int i = 0; i <= steps; ++i) {
        const float u = (float)i / (float)steps;
        const float logsnr = logsnr_start + u * (logsnr_end - logsnr_start);
        schedule[(size_t)i] = 1.0f / (1.0f + std::exp(logsnr));
    }
    schedule.front() = sigma_max;
    schedule.back() = 0.0f;
    return schedule;
}

// One stable-audio-tools RF ping-pong update.  `velocity` is the model prediction at
// t_current and `noise` is a fresh normal sample for this step.  The caller deliberately
// supplies noise even for t_next=0 so RNG consumption can match the reference sampler.
inline void rf_pingpong_step(float* x,
                             const float* velocity,
                             const float* noise,
                             size_t n,
                             float t_current,
                             float t_next) {
    if (!x || !velocity || !noise) throw std::invalid_argument("null ping-pong buffer");
    if (t_current < 0.0f || t_current > 1.0f || t_next < 0.0f || t_next > t_current)
        throw std::invalid_argument("ping-pong timesteps must satisfy 0 <= next <= current <= 1");
    for (size_t i = 0; i < n; ++i) {
        const float denoised = x[i] - t_current * velocity[i];
        x[i] = (1.0f - t_next) * denoised + t_next * noise[i];
    }
}

} // namespace sa3::sampling
