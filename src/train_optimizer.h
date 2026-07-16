// train_optimizer.h - host AdamW updates for native SA3 LoRA training.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace sa3 {

// InverseLR schedule (Stage 11): closed-form port of stable_audio_3/training/utils.py
// InverseLR._get_closed_form_lr. `step0` is the 0-indexed optimizer step, i.e. PyTorch
// _LRScheduler.last_epoch: 0 at the first update, 1 at the second, ... (the reference config uses
// interval "step", so last_epoch == the number of optimizer updates already applied).
//   warmup_factor = 1 - warmup^(step0+1)                 (exponential warmup; 0 disables)
//   lr_mult       = (1 + step0/inv_gamma)^(-power)
//   lr(step0)     = warmup_factor * max(final_lr, base_lr * lr_mult)
// Reference ratatat-2 config: inv_gamma 1e6, power 0.5, warmup 0.995, final_lr 0.
inline float inverse_lr(float base_lr, int step0, float inv_gamma, float power,
                        float warmup, float final_lr) {
    const double n = (double)step0;
    const double wf = (warmup > 0.0f) ? (1.0 - std::pow((double)warmup, n + 1.0)) : 1.0;
    const double lr_mult = std::pow(1.0 + n / (double)inv_gamma, -(double)power);
    return (float)(wf * std::max((double)final_lr, (double)base_lr * lr_mult));
}

struct TrainAdamWParams {
    float learning_rate = 1.0e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1.0e-8f;
    float weight_decay = 0.0f;
    float grad_clip = 0.0f;   // global gradient-norm clip; 0 = off (reference training uses 1.0)
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
