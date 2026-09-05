// sat/pipeline.h -- reusable Stable Audio Tools text-to-audio orchestration.
#pragma once

#include "sat/model_spec.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace sa3::sat {

const char* sampler_name(Sampler sampler);
Sampler parse_sampler(const std::string& name);

struct PipelinePaths {
    std::string dit;
    std::string autoencoder;
    std::string t5;
};

struct StepProgress {
    int step = 0;
    int steps = 0;
    float t_current = 0.0f;
    float t_next = 0.0f;
    double milliseconds = 0.0;
};

struct GenerateParams {
    std::string prompt;
    std::string negative_prompt;
    float seconds_start = 0.0f;
    float seconds = 11.0f;
    int frames = 256;
    int steps = 0;                 // 0: objective-specific default
    int output_samples = 0;        // 0: seconds * model sample rate
    float cfg_scale = -1.0f;       // <0: objective-specific default
    float sigma_min = -1.0f;       // <0: objective-specific default
    float sigma_max = -1.0f;       // <0: objective-specific default
    float sigma_rho = 1.0f;
    float sde_eta = 1.0f;
    uint64_t seed = 1234;
    Sampler sampler = Sampler::Auto;

    // Optional deterministic/parity inputs. Supplying cross/global bypasses T5.
    std::vector<float> cross_conditioning;
    std::vector<float> global_conditioning;
    std::vector<float> initial_latent;
    std::vector<float> step_noise;
    std::function<void(const StepProgress&)> progress;
};

struct GenerateTiming {
    double conditioning_s = 0.0;
    double dit_load_s = 0.0;
    double dit_build_s = 0.0;
    double denoise_s = 0.0;
    double warm_step_median_ms = 0.0;
    double ae_load_s = 0.0;
    double ae_build_s = 0.0;
    double decode_s = 0.0;
    double total_s = 0.0;
};

struct GenerateResult {
    std::vector<float> audio;      // tightly packed planar channels
    std::vector<float> latent;     // frame-major [frame, channel]
    std::vector<float> cross_conditioning;
    std::vector<float> global_conditioning;
    int samples = 0;
    int channels = 0;
    int sample_rate = 0;
    int prompt_tokens = 0;
    int max_prompt_tokens = 0;
    int steps = 0;
    float cfg_scale = 1.0f;
    float sigma_min = 0.0f;
    float sigma_max = 0.0f;
    float sigma_rho = 0.0f;
    float sde_eta = 0.0f;
    Sampler sampler = Sampler::Auto;
    std::string objective;
    GenerateTiming timing;
};

// The pipeline intentionally stages T5, DiT, and Oobleck rather than retaining all
// graphs at once. This keeps the component usable on modest GPUs and leaves residency
// policy to a future application-facing cache layer.
class Pipeline {
public:
    Pipeline() = default;
    explicit Pipeline(PipelinePaths paths) { load(std::move(paths)); }

    void load(PipelinePaths paths);
    const PipelinePaths& paths() const { return paths_; }
    GenerateResult generate(const GenerateParams& params) const;

private:
    PipelinePaths paths_;
};

} // namespace sa3::sat
