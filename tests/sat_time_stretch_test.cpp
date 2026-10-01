#include "sat/time_stretch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

double tone_power(const float* audio, int start, int count, double frequency, int sample_rate) {
    constexpr double pi = 3.14159265358979323846;
    double real = 0, imaginary = 0;
    for (int i = 0; i < count; ++i) {
        const double phase = 2 * pi * frequency * i / sample_rate;
        real += audio[start + i] * std::cos(phase);
        imaginary += audio[start + i] * std::sin(phase);
    }
    return real * real + imaginary * imaginary;
}

bool check(int output_samples) {
    constexpr int sample_rate = 44100;
    constexpr int input_samples = sample_rate * 2;
    constexpr double pi = 3.14159265358979323846;
    std::vector<float> input(2 * input_samples);
    for (int i = 0; i < input_samples; ++i) {
        const float sample = (float)(0.5 * std::sin(2 * pi * 440 * i / sample_rate));
        input[i] = input[input_samples + i] = sample;
    }
    const auto output = sa3::sat::stretch_planar_exact(input, 2, input_samples,
                                                        output_samples, sample_rate);
    if (output.size() != (size_t)2 * output_samples) return false;
    const int start = sample_rate / 4;
    const int count = std::min(sample_rate, output_samples - start * 2);
    const double desired = tone_power(output.data(), start, count, 440, sample_rate);
    const double changed_pitch = tone_power(output.data(), start, count,
                                             440.0 * input_samples / output_samples, sample_rate);
    std::printf("samples=%d 440Hz=%.2f changed_pitch=%.2f\n",
                output_samples, desired, changed_pitch);
    return desired > 1000 && desired > changed_pitch * 10;
}

}  // namespace

int main() {
    return check(88200 * 11 / 10) && check(88200 * 4 / 5) ? 0 : 1;
}
