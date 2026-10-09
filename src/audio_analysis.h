// CPU metadata suggestions for dataset sidecars. RMS-onset autocorrelation
// and Krumhansl-Kessler chroma profiles, matching Gary's reference estimators.
#pragma once
#include "wav.h"
#include "signalsmith-linear/fft.h"
#include <array>
#include <complex>
#include <limits>
#include <numeric>
#include <optional>

namespace sa3 {
namespace analysis {
constexpr int rate = 11025;
constexpr double pi = 3.14159265358979323846;
struct Result {
    std::optional<int> bpm;
    std::string keyscale;
    std::optional<double> bpm_confidence, key_confidence;
};

inline double bessel_i0(double x) {
    double sum = 1, term = 1;
    for (int k = 1; k < 50; ++k) {
        term *= x*x/(4.0*k*k);
        sum += term;
        if (term < sum*1e-16) break;
    }
    return sum;
}

// Centered zero-padded polyphase FIR: Kaiser beta=5, 10 samples on each
// side at the upsampled rate. Equal rates bypass filtering, as in resample_poly.
inline std::vector<float> resample(const std::vector<float>& input, int source_rate) {
    if (source_rate < 1000 || source_rate > 384000)
        throw std::runtime_error("analysis sample rate must be between 1000 and 384000 Hz");
    if (source_rate == rate || input.empty()) return input;
    const int gcd = std::gcd(source_rate, rate), up = rate/gcd, down = source_rate/gcd;
    const int scale = std::max(up, down), half = 10*scale;
    std::vector<double> coefficients(2*half + 1);
    double sum = 0;
    for (int i = -half; i <= half; ++i) {
        const double x = double(i)/scale;
        const double sinc = i == 0 ? 1 : std::sin(pi*x)/(pi*x);
        const double window = bessel_i0(5*std::sqrt(std::max(0.0, 1-double(i)*i/(double(half)*half))))/bessel_i0(5);
        sum += coefficients[i + half] = sinc*window/scale;
    }
    std::vector<float> taps(coefficients.size());
    for (size_t i = 0; i < taps.size(); ++i) taps[i] = float(coefficients[i]/sum)*up;
    std::vector<float> output((input.size()*up + down - 1)/down);
    for (size_t i = 0; i < output.size(); ++i) {
        const int64_t center = int64_t(i)*down;
        const int64_t start = std::max<int64_t>(0, (center - half + up - 1)/up);
        const int64_t end = std::min<int64_t>(input.size() - 1, (center + half)/up);
        float value = 0;
        for (int64_t sample = start; sample <= end; ++sample)
            value += input[size_t(sample)]*taps[size_t(half + center - sample*up)];
        output[i] = value;
    }
    return output;
}

inline void estimate_bpm(const std::vector<float>& mono, Result& result) {
    constexpr size_t frame = 1024, hop = 512;
    if (mono.size() < frame + 2*hop) return;
    const size_t count = 1 + (mono.size() - frame)/hop;
    std::vector<float> onset(count);
    float previous = 0;
    double mean = 0;
    for (size_t i = 0; i < count; ++i) {
        double energy = 0;
        for (size_t j = 0; j < frame; ++j) energy += double(mono[i*hop + j])*mono[i*hop + j];
        const float rms = float(std::sqrt(energy/frame + 1e-12));
        mean += onset[i] = i == 0 ? 0 : std::max(0.f, rms - previous);
        previous = rms;
    }
    const float onset_mean = float(mean/count);
    bool nonzero = false;
    for (float& value : onset) { value -= onset_mean; nonzero |= value != 0; }
    if (!nonzero) return;
    constexpr double fps = double(rate)/hop;
    const int min_lag = std::max(1, int(std::nearbyint(fps*60/220)));
    const int max_lag = std::min<int>(int(count) - 1, int(std::nearbyint(fps*60/60)));
    if (max_lag <= min_lag) return;
    std::vector<double> values(max_lag - min_lag + 1);
    for (int lag = min_lag; lag <= max_lag; ++lag) {
        double value = 0;
        for (size_t i = size_t(lag); i < count; ++i) value += double(onset[i])*onset[i - lag];
        values[lag - min_lag] = value;
    }
    std::vector<int> peaks;
    for (int i = 1; i + 1 < int(values.size()); ++i) {
        if (values[i] <= values[i - 1]) continue;
        int end = i;
        while (end + 1 < int(values.size()) && values[end + 1] == values[i]) ++end;
        if (end + 1 < int(values.size()) && values[end + 1] < values[i]) peaks.push_back((i + end)/2);
        i = end;
    }
    // SciPy peak-distance pruning considers the strongest peaks first.
    std::sort(peaks.begin(), peaks.end(), [&](int a, int b) { return values[a] == values[b] ? a > b : values[a] > values[b]; });
    std::vector<int> chosen;
    for (int peak : peaks) {
        bool near = false;
        for (int other : chosen) near |= std::abs(peak - other) < std::max(1, min_lag/2);
        if (!near) chosen.push_back(peak);
    }
    chosen.erase(std::remove_if(chosen.begin(), chosen.end(), [&](int peak) { return values[peak] <= 0; }), chosen.end());
    if (chosen.empty()) return;
    const double bpm = 60*fps/(chosen[0] + min_lag);
    result.bpm = int(std::nearbyint(bpm));
    result.bpm_confidence = values[chosen[0]]/std::max(1e-9, chosen.size() > 1 ? std::abs(values[chosen[1]]) : 1e-9);
}

inline std::array<double,12> normalize(std::array<double,12> values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0)/12;
    double norm = 0;
    for (double& value : values) { value -= mean; norm += value*value; }
    norm = std::sqrt(norm);
    if (norm > 0) for (double& value : values) value /= norm;
    return values;
}

inline void estimate_key(const std::vector<float>& mono, Result& result) {
    constexpr size_t frame = 8192, hop = 2048;
    if (mono.size() < frame) return;
    signalsmith::linear::RealFFT<float> fft(frame);
    std::vector<float> time(frame), window(frame), magnitude(frame/2 + 1);
    std::vector<std::complex<float>> frequency(frame/2);
    for (size_t i = 0; i < frame; ++i) window[i] = float(0.5 - 0.5*std::cos(2*pi*i/frame));
    std::array<double,12> chroma{};
    for (size_t start = 0; start + frame <= mono.size(); start += hop) {
        for (size_t i = 0; i < frame; ++i) time[i] = mono[start + i]*window[i];
        fft.fft(time.data(), frequency.data());
        for (const auto& value : frequency) if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            throw std::runtime_error("audio exceeds finite analysis range");
        magnitude[0] = std::abs(frequency[0].real());
        magnitude[frame/2] = std::abs(frequency[0].imag());
        for (size_t i = 1; i < frame/2; ++i) magnitude[i] = std::abs(frequency[i]);
        const float threshold = *std::max_element(magnitude.begin(), magnitude.end())*float(std::pow(10.0, -55.0/20));
        for (size_t i = 1; i < frame/2; ++i) {
            const double hz = double(i)*rate/frame;
            if (hz < 55 || hz > 2500 || magnitude[i] < threshold) continue;
            const int midi = int(std::nearbyint(69 + 12*std::log2(hz/440)));
            chroma[size_t((midi%12 + 12)%12)] += magnitude[i]/std::sqrt(hz);
        }
    }
    const double total = std::accumulate(chroma.begin(), chroma.end(), 0.0);
    if (total <= 0) return;
    for (double& value : chroma) value /= total;
    chroma = normalize(chroma);
    constexpr std::array<double,12> major{6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88};
    constexpr std::array<double,12> minor{6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17};
    const char* notes[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    struct Candidate { std::string key; double score; };
    std::vector<Candidate> candidates;
    for (int tonic = 0; tonic < 12; ++tonic) {
        for (int mode = 0; mode < 2; ++mode) {
            std::array<double,12> shifted{};
            for (int i = 0; i < 12; ++i) shifted[i] = (mode == 0 ? major : minor)[(i - tonic + 12)%12];
            shifted = normalize(shifted);
            double score = 0;
            for (int i = 0; i < 12; ++i) score += chroma[i]*shifted[i];
            candidates.push_back({std::string(notes[tonic]) + (mode == 0 ? " major" : " minor"), score});
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    result.keyscale = candidates[0].key;
    result.key_confidence = candidates[0].score - candidates[1].score;
}

inline Result analyze(const std::vector<float>& mono, int source_rate) {
    for (float value : mono) if (!std::isfinite(value)) throw std::runtime_error("audio contains non-finite samples");
    const auto samples = resample(mono, source_rate);
    for (float value : samples) if (!std::isfinite(value)) throw std::runtime_error("resampled audio contains non-finite samples");
    Result result;
    estimate_bpm(samples, result);
    estimate_key(samples, result);
    return result;
}

inline Result analyze_wav(const std::string& path) {
    int count = 0, channels = 0, source_rate = 0;
    const auto samples = read_wav_planar(path, count, channels, source_rate, false);
    std::vector<float> mono(count);
    for (int i = 0; i < count; ++i) {
        double sum = 0;
        for (int ch = 0; ch < channels; ++ch) {
            const float value = samples[size_t(ch)*count + i];
            if (!std::isfinite(value)) throw std::runtime_error("audio contains non-finite samples");
            sum += value;
        }
        mono[i] = float(sum/channels);
    }
    return analyze(mono, source_rate);
}
} // namespace analysis
} // namespace sa3
