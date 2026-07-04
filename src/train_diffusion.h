// train_diffusion.h - RF/flow-matching timestep and target sampling for SA3 training.
#pragma once

#include <random>
#include <string>
#include <vector>

namespace sa3 {

struct TrainDiffusionSample {
    float t = 0.0f;
    std::vector<float> noise;
    std::vector<float> x_t;
    std::vector<float> velocity_target;
};

class TrainDiffusionSampler {
public:
    explicit TrainDiffusionSampler(unsigned long long seed) : rng_(seed) {}

    bool sample(const std::vector<float>& z, TrainDiffusionSample& out, std::string& err) {
        if (z.empty()) {
            err = "latent vector is empty";
            return false;
        }
        out.t = uniform_(rng_);
        out.noise.resize(z.size());
        out.x_t.resize(z.size());
        out.velocity_target.resize(z.size());
        for (size_t i = 0; i < z.size(); ++i) {
            const float n = normal_(rng_);
            out.noise[i] = n;
            out.x_t[i] = (1.0f - out.t) * z[i] + out.t * n;
            out.velocity_target[i] = n - z[i];
        }
        return true;
    }

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> uniform_{0.0f, 1.0f};
    std::normal_distribution<float> normal_{0.0f, 1.0f};
};

} // namespace sa3
