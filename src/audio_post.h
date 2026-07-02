#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace sa3 {

// Linear-interpolation resampler for PLANAR audio (samples[c*n_samples + s]). Adequate for a2a init
// audio (a conditioning source); the official SA3 uses a torchaudio sinc resampler. Returns the
// resampled planar buffer and sets out_samples. src_rate==dst_rate returns the input unchanged.
inline std::vector<float> resample_planar_linear(const std::vector<float>& input,
                                                 int n_samples, int n_ch,
                                                 int src_rate, int dst_rate, int& out_samples) {
    if (n_samples <= 0 || n_ch <= 0) { out_samples = 0; return {}; }
    if (src_rate <= 0 || dst_rate <= 0) throw std::runtime_error("invalid sample rate for resampling");
    if (src_rate == dst_rate) { out_samples = n_samples; return input; }

    out_samples = std::max(1, (int)std::llround((double)n_samples * (double)dst_rate / (double)src_rate));
    std::vector<float> out((size_t)out_samples * n_ch);
    const double src_step = (double)src_rate / (double)dst_rate;
    for (int c = 0; c < n_ch; c++) {
        const float* in_ch = input.data() + (size_t)c * n_samples;
        float* out_ch = out.data() + (size_t)c * out_samples;
        for (int s = 0; s < out_samples; s++) {
            const double pos = (double)s * src_step;
            int i0 = (int)std::floor(pos);
            if (i0 >= n_samples - 1) { out_ch[s] = in_ch[n_samples - 1]; continue; }
            const float frac = (float)(pos - (double)i0);
            out_ch[s] = in_ch[i0] + (in_ch[i0 + 1] - in_ch[i0]) * frac;
        }
    }
    return out;
}

struct LoudnessParams {
    float latent_rescale = 1.0f;
    float latent_shift = 0.0f;
    bool  latent_target_std_enabled = false;
    float latent_target_std = 0.0f;
    float latent_adapt_min = 0.9f;
    float latent_adapt_max = 1.0f;

    bool  peak_normalize_enabled = true;
    float peak_normalize_db = 2.0f;

    bool  limiter_enabled = true;
    float limiter_ceiling_db = -0.3f;
    float limiter_knee = 0.8f;
};

struct LoudnessMeta {
    LoudnessParams params;
    bool  latent_std_set = false;
    float latent_std = 0.0f;
    float latent_factor = 1.0f;
    float decoded_peak = 0.0f;
    bool  peak_normalize_gain_set = false;
    float peak_normalize_gain = 1.0f;
    bool  limiter_limited_fraction_set = false;
    float limiter_limited_fraction = 0.0f;
    bool  safety_gain_set = false;
    float safety_gain = 1.0f;
    float final_peak = 0.0f;
};

inline std::string lower_ascii_copy(std::string s) {
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

inline bool parse_float_text(const char* text, float& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    const float v = std::strtof(text, &end);
    if (end == text) return false;
    while (end && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) end++;
    if (end && *end) return false;
    out = v;
    return std::isfinite(out);
}

inline bool text_disables_optional_float(const char* text) {
    if (!text) return true;
    std::string s = lower_ascii_copy(text);
    const size_t a = s.find_first_not_of(" \t\r\n");
    const size_t b = s.find_last_not_of(" \t\r\n");
    s = a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
    return s.empty() || s == "off" || s == "none" || s == "null" || s == "false" || s == "disabled";
}

inline float env_float(const char* name, float fallback) {
    float v = fallback;
    const char* text = std::getenv(name);
    return parse_float_text(text, v) ? v : fallback;
}

inline void env_optional_float(const char* name, bool& enabled, float& value, bool positive_disables = false) {
    const char* text = std::getenv(name);
    if (!text) return;
    if (text_disables_optional_float(text)) { enabled = false; return; }
    float v = value;
    if (!parse_float_text(text, v)) return;
    if (positive_disables && v > 0.0f) { enabled = false; return; }
    enabled = true;
    value = v;
}

inline LoudnessParams loudness_defaults_from_env() {
    LoudnessParams p;
    p.latent_rescale = env_float("SA3_LATENT_RESCALE", p.latent_rescale);
    p.latent_shift = env_float("SA3_LATENT_SHIFT", p.latent_shift);
    env_optional_float("SA3_LATENT_TARGET_STD", p.latent_target_std_enabled, p.latent_target_std);
    p.latent_adapt_min = env_float("SA3_LATENT_ADAPT_MIN", p.latent_adapt_min);
    p.latent_adapt_max = env_float("SA3_LATENT_ADAPT_MAX", p.latent_adapt_max);
    env_optional_float("SA3_PEAK_NORMALIZE_DB", p.peak_normalize_enabled, p.peak_normalize_db);
    env_optional_float("SA3_LIMITER_CEILING_DB", p.limiter_enabled, p.limiter_ceiling_db, true);
    p.limiter_knee = env_float("SA3_LIMITER_KNEE", p.limiter_knee);
    return p;
}

inline void normalize_loudness_params(LoudnessParams& p) {
    if (p.limiter_enabled && p.limiter_ceiling_db > 0.0f) p.limiter_enabled = false;
}

inline bool validate_loudness_params(const LoudnessParams& p, std::string& err) {
    auto finite = [](float v) { return std::isfinite(v); };
    if (!finite(p.latent_rescale) || p.latent_rescale < 0.0f) {
        err = "latent_rescale must be finite and >= 0";
        return false;
    }
    if (!finite(p.latent_shift)) {
        err = "latent_shift must be finite";
        return false;
    }
    if (p.latent_target_std_enabled && (!finite(p.latent_target_std) || p.latent_target_std <= 0.0f)) {
        err = "latent_target_std must be finite and > 0";
        return false;
    }
    if (!finite(p.latent_adapt_min) || !finite(p.latent_adapt_max) ||
        p.latent_adapt_min < 0.0f || p.latent_adapt_max < p.latent_adapt_min) {
        err = "latent_adapt_min/max must be finite, min >= 0, and max >= min";
        return false;
    }
    if (p.peak_normalize_enabled && !finite(p.peak_normalize_db)) {
        err = "peak_normalize_db must be finite";
        return false;
    }
    if (p.limiter_enabled && !finite(p.limiter_ceiling_db)) {
        err = "limiter_ceiling_db must be finite";
        return false;
    }
    if (!finite(p.limiter_knee) || p.limiter_knee <= 0.0f || p.limiter_knee > 1.0f) {
        err = "limiter_knee must be finite and in (0, 1]";
        return false;
    }
    return true;
}

inline float db_to_linear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

inline float max_abs(const std::vector<float>& x) {
    float m = 0.0f;
    for (float v : x) m = std::max(m, std::fabs(v));
    return m;
}

inline float sample_stddev(const std::vector<float>& x) {
    if (x.size() < 2) return 0.0f;
    double mean = 0.0;
    for (float v : x) mean += (double)v;
    mean /= (double)x.size();
    double ss = 0.0;
    for (float v : x) {
        const double d = (double)v - mean;
        ss += d * d;
    }
    return (float)std::sqrt(ss / (double)(x.size() - 1));
}

inline LoudnessMeta make_loudness_meta(const LoudnessParams& params) {
    LoudnessMeta meta;
    meta.params = params;
    return meta;
}

inline void apply_latent_loudness(std::vector<float>& latents, const LoudnessParams& params, LoudnessMeta& meta) {
    float factor = params.latent_rescale;
    if (params.latent_target_std_enabled) {
        const float std = sample_stddev(latents);
        meta.latent_std = std;
        meta.latent_std_set = true;
        if (std > 1e-6f) {
            factor = params.latent_target_std / std;
            factor = std::max(params.latent_adapt_min, std::min(params.latent_adapt_max, factor));
        } else {
            factor = 1.0f;
        }
    }
    meta.latent_factor = factor;
    if (factor == 1.0f && params.latent_shift == 0.0f) return;
    for (float& v : latents) v = v * factor + params.latent_shift;
}

inline void apply_audio_loudness(std::vector<float>& audio, const LoudnessParams& params, LoudnessMeta& meta) {
    meta.decoded_peak = max_abs(audio);
    if (params.peak_normalize_enabled && meta.decoded_peak > 1e-6f) {
        const float gain = db_to_linear(params.peak_normalize_db) / meta.decoded_peak;
        for (float& v : audio) v *= gain;
        meta.peak_normalize_gain = gain;
        meta.peak_normalize_gain_set = true;
    }

    if (params.limiter_enabled) {
        const float ceiling = db_to_linear(params.limiter_ceiling_db);
        const float knee_fraction = std::max(1e-6f, std::min(1.0f, params.limiter_knee));
        const float knee = ceiling * knee_fraction;
        size_t limited = 0;
        for (float& v : audio) {
            const float mag = std::fabs(v);
            if (mag <= knee) continue;
            limited++;
            const float limited_mag = knee >= ceiling
                ? std::min(mag, ceiling)
                : knee + (ceiling - knee) * std::tanh((mag - knee) / (ceiling - knee));
            v = (v < 0.0f ? -limited_mag : limited_mag);
        }
        meta.limiter_limited_fraction = audio.empty() ? 0.0f : (float)((double)limited / (double)audio.size());
        meta.limiter_limited_fraction_set = true;
    }

    // True-peak safety: when peak-normalize is active, never leave the output above 0 dBFS — otherwise
    // a positive peak_normalize_db (default +2 dB) with the limiter disabled would hard-clip on the
    // 16-bit WAV write. No-op in the normal limiter-on path (peak already <= the ceiling < 1.0). Gated
    // on peak_normalize so a fully-raw request (both off) stays byte-for-byte untouched.
    const float peak = max_abs(audio);
    if (params.peak_normalize_enabled && peak > 1.0f) {
        const float g = 1.0f / peak;
        for (float& v : audio) v *= g;
        meta.safety_gain = g;
        meta.safety_gain_set = true;
        meta.final_peak = 1.0f;
    } else {
        meta.final_peak = peak;
    }
}

} // namespace sa3
