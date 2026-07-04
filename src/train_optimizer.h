// train_optimizer.h - host AdamW updates for native SA3 LoRA training.
#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace sa3 {

struct TrainAdamWParams {
    float learning_rate = 1.0e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1.0e-8f;
    float weight_decay = 0.0f;
};

struct TrainAdamWTensorState {
    std::vector<float> m;
    std::vector<float> v;
};

inline bool adamw_update_vector(std::vector<float>& param, const std::vector<float>& grad,
                                TrainAdamWTensorState& state, const TrainAdamWParams& hp,
                                int step, std::string& err) {
    if (param.size() != grad.size()) {
        err = "AdamW parameter/gradient size mismatch";
        return false;
    }
    if (step <= 0) {
        err = "AdamW step must be positive";
        return false;
    }
    if (state.m.size() != param.size()) state.m.assign(param.size(), 0.0f);
    if (state.v.size() != param.size()) state.v.assign(param.size(), 0.0f);
    const float b1_corr = 1.0f - std::pow(hp.beta1, (float)step);
    const float b2_corr = 1.0f - std::pow(hp.beta2, (float)step);
    for (size_t i = 0; i < param.size(); ++i) {
        const float g = grad[i];
        state.m[i] = hp.beta1 * state.m[i] + (1.0f - hp.beta1) * g;
        state.v[i] = hp.beta2 * state.v[i] + (1.0f - hp.beta2) * g * g;
        const float mh = state.m[i] / b1_corr;
        const float vh = state.v[i] / b2_corr;
        param[i] -= hp.learning_rate * (mh / (std::sqrt(vh) + hp.eps) + hp.weight_decay * param[i]);
    }
    return true;
}

} // namespace sa3
