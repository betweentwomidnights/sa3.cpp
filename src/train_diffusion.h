// train_diffusion.h - RF/flow-matching timestep and target sampling for SA3 training.
#pragma once

#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace sa3 {

struct TrainDiffusionSample {
    float t = 0.0f;
    std::vector<float> noise;
    std::vector<float> x_t;
    std::vector<float> velocity_target;
};

// Standard normal CDF via erfc.
inline double sa3_norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.7071067811865476); }

// Inverse standard normal CDF (Acklam's rational approximation, ~1e-9 accuracy).
inline double sa3_norm_icdf(double p) {
    if (p <= 0.0) return -1e9;
    if (p >= 1.0) return 1e9;
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                                -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - plow;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    if (p > phigh) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    const double q = p - 0.5, r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
}

class TrainDiffusionSampler {
public:
    explicit TrainDiffusionSampler(unsigned long long seed, const std::string& mode = "uniform")
        : rng_(seed), mode_(mode) {}

    bool sample(const std::vector<float>& z, TrainDiffusionSample& out, std::string& err) {
        return sample_at(z, draw_t(), out, err);
    }

    // Noise the latent at an externally supplied t. Lets the caller warp the drawn t
    // (training dist-shift) between draw_t() and noising, matching the reference order
    // (timestep_sampler -> dist_shift.shift -> noise).
    bool sample_at(const std::vector<float>& z, float t, TrainDiffusionSample& out, std::string& err) {
        if (z.empty()) {
            err = "latent vector is empty";
            return false;
        }
        out.t = t;
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

    // Timestep draw. "uniform" = U(0,1). "trunc_logit_normal" ports the reference
    // truncated_logistic_normal_rescaled(left_trunc=0.075) then flips (t = 1 - t),
    // matching stable_audio_3/training/diffusion.py + inference/sampling.py.
    float draw_t() {
        if (mode_ == "trunc_logit_normal") {
            const double left = 0.075;
            const double lb = sa3_norm_cdf(std::log(left / (1.0 - left)));  // Phi(logit(left))
            const double u = sa3_norm_cdf((double)normal_(rng_));           // Phi(z), z~N(0,1)
            double tc = lb + (1.0 - lb) * u;                                // uniform in [lb, 1]
            if (tc < 1e-7) tc = 1e-7; else if (tc > 1.0 - 1e-7) tc = 1.0 - 1e-7;
            const double ts = 1.0 / (1.0 + std::exp(-sa3_norm_icdf(tc)));   // sigmoid(icdf(tc))
            const double rescaled = (ts - left) / (1.0 - left);
            return (float)(1.0 - rescaled);
        }
        return uniform_(rng_);
    }

    // std::normal_distribution may retain a cached second variate, so exact continuation must
    // persist the distributions as well as the engine. The standard stream operators capture both.
    std::string serialize_state() const {
        std::ostringstream out;
        out << rng_ << '\n' << uniform_ << '\n' << normal_;
        return out.str();
    }

    bool restore_state(const std::string& state, std::string& err) {
        std::istringstream in(state);
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> uniform;
        std::normal_distribution<float> normal;
        if (!(in >> rng >> uniform >> normal)) {
            err = "invalid diffusion sampler state";
            return false;
        }
        rng_ = rng;
        uniform_ = uniform;
        normal_ = normal;
        return true;
    }

private:
    std::mt19937_64 rng_;
    std::string mode_;
    std::uniform_real_distribution<float> uniform_{0.0f, 1.0f};
    std::normal_distribution<float> normal_{0.0f, 1.0f};
};

} // namespace sa3
