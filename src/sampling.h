// sampling.h -- model-agnostic host-side sampling primitives.
//
// Keep schedule construction and state updates out of model pipelines.  A pipeline owns
// graph execution and CFG; these helpers only describe the numerical integration contract.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
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

// First-order ODE integration used by stable-audio-tools' rectified-flow Euler
// sampler. `dt` is negative because inference traverses t=1 -> 0.
inline void rf_euler_step(float* x,
                          const float* velocity,
                          size_t n,
                          float t_current,
                          float t_next) {
    if (!x || !velocity) throw std::invalid_argument("null Euler buffer");
    if (t_current < 0.0f || t_current > 1.0f || t_next < 0.0f || t_next > t_current)
        throw std::invalid_argument("Euler timesteps must satisfy 0 <= next <= current <= 1");
    const float dt = t_next - t_current;
    for (size_t i = 0; i < n; ++i) x[i] += dt * velocity[i];
}

// State carried by the second-order DPM-Solver++ update used for ordinary
// rectified-flow checkpoints. It stores x0 from the preceding evaluation.
struct RfDpmppState {
    std::vector<float> old_denoised;
    float previous_t = 0.0f;
};

inline float rf_log_snr(float t) {
    if (t <= 0.0f) return INFINITY;
    if (t >= 1.0f) return -INFINITY;
    return std::log((1.0f - t) / t);
}

// Exact host translation of stable-audio-tools sample_flow_dpmpp. The first
// update is first order; subsequent updates use the previous denoised estimate.
inline void rf_dpmpp_step(float* x,
                          const float* velocity,
                          size_t n,
                          float t_current,
                          float t_next,
                          RfDpmppState& state) {
    if (!x || !velocity) throw std::invalid_argument("null DPM++ buffer");
    if (t_current <= 0.0f || t_current > 1.0f || t_next < 0.0f || t_next >= t_current)
        throw std::invalid_argument("DPM++ timesteps must satisfy 0 <= next < current <= 1");

    std::vector<float> denoised(n);
    for (size_t i = 0; i < n; ++i) denoised[i] = x[i] - t_current * velocity[i];

    if (t_next == 0.0f) {
        std::copy(denoised.begin(), denoised.end(), x);
    } else {
        const float h = rf_log_snr(t_next) - rf_log_snr(t_current);
        const float x_scale = t_next / t_current;
        const float denoised_scale = -(1.0f - t_next) * std::expm1(-h);
        if (state.old_denoised.size() != n) {
            for (size_t i = 0; i < n; ++i)
                x[i] = x_scale * x[i] + denoised_scale * denoised[i];
        } else {
            const float h_last = rf_log_snr(t_current) - rf_log_snr(state.previous_t);
            const float r = h_last / h;
            const float old_mix = 1.0f / (2.0f * r);
            for (size_t i = 0; i < n; ++i) {
                const float denoised_d = (1.0f + old_mix) * denoised[i]
                                       - old_mix * state.old_denoised[i];
                x[i] = x_scale * x[i] + denoised_scale * denoised_d;
            }
        }
    }
    state.old_denoised = std::move(denoised);
    state.previous_t = t_current;
}

// k-diffusion's polynomial-in-log-sigma schedule. With rho=1 (the Stable Audio
// default) this is a geometric progression from sigma_max to sigma_min followed
// by the exact denoised endpoint at zero.
inline std::vector<float> make_sigma_polyexponential_schedule(int steps,
                                                               float sigma_min,
                                                               float sigma_max,
                                                               float rho = 1.0f) {
    if (steps < 2) throw std::invalid_argument("V-prediction sampling requires at least two steps");
    if (!(sigma_min > 0.0f && sigma_max > sigma_min && rho > 0.0f))
        throw std::invalid_argument("invalid V-prediction sigma schedule");
    std::vector<float> sigmas((size_t)steps + 1);
    const float lo = std::log(sigma_min);
    const float range = std::log(sigma_max) - lo;
    for (int i = 0; i < steps; ++i) {
        const float ramp = std::pow(1.0f - (float)i / (float)(steps - 1), rho);
        sigmas[(size_t)i] = std::exp(ramp * range + lo);
    }
    sigmas.back() = 0.0f;
    return sigmas;
}

struct VPredictionScalings {
    float skip;
    float output;
    float input;
    float timestep;
};

// k_diffusion.external.VDenoiser with sigma_data=1.
inline VPredictionScalings v_prediction_scalings(float sigma) {
    if (!(sigma >= 0.0f) || !std::isfinite(sigma))
        throw std::invalid_argument("V-prediction sigma must be finite and non-negative");
    const float denom = std::sqrt(sigma * sigma + 1.0f);
    return {1.0f / (sigma * sigma + 1.0f), -sigma / denom, 1.0f / denom,
            std::atan(sigma) * (2.0f / 3.14159265358979323846f)};
}

inline void v_prediction_to_denoised(const float* x,
                                      const float* velocity,
                                      float* denoised,
                                      size_t n,
                                      float sigma) {
    if (!x || !velocity || !denoised) throw std::invalid_argument("null V-prediction buffer");
    const VPredictionScalings c = v_prediction_scalings(sigma);
    for (size_t i = 0; i < n; ++i)
        denoised[i] = velocity[i] * c.output + x[i] * c.skip;
}

struct VPredictionDpmppState {
    std::vector<float> denoised_1;
    std::vector<float> denoised_2;
    float h_1 = 0.0f;
    float h_2 = 0.0f;
};

// One DPM-Solver++(2M) SDE midpoint update. `noise` is a normalized Brownian
// increment; independent standard-normal increments have the same distribution
// for a fixed monotone schedule and keep RNG policy outside this primitive.
inline void v_dpmpp_2m_sde_step(float* x,
                                const float* denoised,
                                const float* noise,
                                size_t n,
                                float sigma,
                                float sigma_next,
                                VPredictionDpmppState& state,
                                float eta = 1.0f) {
    if (!x || !denoised || (!noise && sigma_next > 0.0f && eta != 0.0f))
        throw std::invalid_argument("null V-prediction DPM++ buffer");
    if (!(sigma > sigma_next && sigma_next >= 0.0f && eta >= 0.0f))
        throw std::invalid_argument("V-prediction sigmas must descend to zero");
    if (sigma_next == 0.0f) {
        std::copy_n(denoised, n, x);
    } else {
        const float h = std::log(sigma / sigma_next);
        const float eta_h = eta * h;
        const float denoised_scale = -std::expm1(-h - eta_h);
        const float x_scale = (sigma_next / sigma) * std::exp(-eta_h);
        for (size_t i = 0; i < n; ++i)
            x[i] = x_scale * x[i] + denoised_scale * denoised[i];
        if (state.denoised_1.size() == n) {
            const float r = state.h_1 / h;
            const float correction = 0.5f * denoised_scale / r;
            for (size_t i = 0; i < n; ++i)
                x[i] += correction * (denoised[i] - state.denoised_1[i]);
        }
        if (eta != 0.0f) {
            const float noise_scale = sigma_next * std::sqrt(-std::expm1(-2.0f * eta_h));
            for (size_t i = 0; i < n; ++i) x[i] += noise_scale * noise[i];
        }
        state.h_1 = h;
    }
    state.denoised_1.assign(denoised, denoised + n);
}

// One DPM-Solver++(3M) SDE update, translated from k-diffusion.
inline void v_dpmpp_3m_sde_step(float* x,
                                const float* denoised,
                                const float* noise,
                                size_t n,
                                float sigma,
                                float sigma_next,
                                VPredictionDpmppState& state,
                                float eta = 1.0f) {
    if (!x || !denoised || (!noise && sigma_next > 0.0f && eta != 0.0f))
        throw std::invalid_argument("null V-prediction DPM++ buffer");
    if (!(sigma > sigma_next && sigma_next >= 0.0f && eta >= 0.0f))
        throw std::invalid_argument("V-prediction sigmas must descend to zero");
    float h = state.h_1;
    if (sigma_next == 0.0f) {
        std::copy_n(denoised, n, x);
    } else {
        h = std::log(sigma / sigma_next);
        const float h_eta = h * (eta + 1.0f);
        const float denoised_scale = -std::expm1(-h_eta);
        for (size_t i = 0; i < n; ++i)
            x[i] = std::exp(-h_eta) * x[i] + denoised_scale * denoised[i];
        if (state.denoised_2.size() == n) {
            const float r0 = state.h_1 / h;
            const float r1 = state.h_2 / h;
            const float phi2 = std::expm1(-h_eta) / h_eta + 1.0f;
            const float phi3 = phi2 / h_eta - 0.5f;
            for (size_t i = 0; i < n; ++i) {
                const float d10 = (denoised[i] - state.denoised_1[i]) / r0;
                const float d11 = (state.denoised_1[i] - state.denoised_2[i]) / r1;
                const float d2 = (d10 - d11) / (r0 + r1);
                const float d1 = d10 + d2 * r0;
                x[i] += phi2 * d1 - phi3 * d2;
            }
        } else if (state.denoised_1.size() == n) {
            const float r = state.h_1 / h;
            const float phi2 = std::expm1(-h_eta) / h_eta + 1.0f;
            for (size_t i = 0; i < n; ++i)
                x[i] += phi2 * (denoised[i] - state.denoised_1[i]) / r;
        }
        if (eta != 0.0f) {
            const float noise_scale = sigma_next *
                std::sqrt(-std::expm1(-2.0f * h * eta));
            for (size_t i = 0; i < n; ++i) x[i] += noise_scale * noise[i];
        }
    }
    state.denoised_2 = std::move(state.denoised_1);
    state.denoised_1.assign(denoised, denoised + n);
    state.h_2 = state.h_1;
    state.h_1 = h;
}

} // namespace sa3::sampling
