// train_audio.h - audio loading helpers for native SA3 LoRA training.
#pragma once

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sa3 {

struct TrainAudio {
    std::vector<float> samples; // planar [channel][sample]
    int n_samples = 0;
    int n_channels = 0;
    int sample_rate = 0;
};

// popen/pclose spelling differs on MSVC; wrap so the decode path is portable.
inline FILE* sa3_popen(const char* cmd, const char* mode) {
#ifdef _WIN32
    return _popen(cmd, mode);
#else
    const char posix_mode[2] = {mode[0], '\0'};
    return popen(cmd, posix_mode);
#endif
}

inline int sa3_pclose(FILE* f) {
#ifdef _WIN32
    return _pclose(f);
#else
    return pclose(f);
#endif
}

inline std::string shell_quote_path(const std::string& s) {
#ifdef _WIN32
    // _popen runs the command through cmd.exe, where single quotes are not special;
    // wrap the path in double quotes instead (file paths do not contain '"').
    return "\"" + s + "\"";
#else
    // POSIX sh: single-quote and escape any embedded single quotes.
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
#endif
}

inline bool decode_mp3_planar_ffmpeg(const std::string& path, int target_sample_rate, int target_channels,
                                     TrainAudio& out, std::string& err) {
    if (target_sample_rate <= 0) {
        err = "target sample rate must be positive";
        return false;
    }
    if (target_channels <= 0) {
        err = "target channel count must be positive";
        return false;
    }
    const std::string cmd = "ffmpeg -v error -i " + shell_quote_path(path) +
        " -f f32le -acodec pcm_f32le -ac " + std::to_string(target_channels) +
        " -ar " + std::to_string(target_sample_rate) + " -";
    FILE* pipe = sa3_popen(cmd.c_str(), "rb");   // "rb": binary mode matters on Windows
    if (!pipe) {
        err = "failed to start ffmpeg for " + path;
        return false;
    }
    std::vector<float> interleaved;
    std::array<unsigned char, 1 << 15> buf{};
    while (true) {
        const size_t n = fread(buf.data(), 1, buf.size(), pipe);
        if (n > 0) {
            const size_t old = interleaved.size();
            interleaved.resize(old + n / sizeof(float));
            std::memcpy(interleaved.data() + old, buf.data(), (n / sizeof(float)) * sizeof(float));
        }
        if (n < buf.size()) {
            if (feof(pipe)) break;
            if (ferror(pipe)) {
                sa3_pclose(pipe);
                err = "error reading decoded audio from ffmpeg for " + path;
                return false;
            }
        }
    }
    const int rc = sa3_pclose(pipe);
    if (rc != 0) {
        err = "ffmpeg failed while decoding " + path;
        return false;
    }
    if (interleaved.empty() || interleaved.size() % (size_t)target_channels != 0) {
        err = "decoded audio has invalid sample count for " + path;
        return false;
    }
    const int n_samples = (int)(interleaved.size() / (size_t)target_channels);
    out.samples.assign((size_t)n_samples * target_channels, 0.0f);
    for (int i = 0; i < n_samples; ++i) {
        for (int ch = 0; ch < target_channels; ++ch) {
            out.samples[(size_t)ch * n_samples + i] = interleaved[(size_t)i * target_channels + ch];
        }
    }
    out.n_samples = n_samples;
    out.n_channels = target_channels;
    out.sample_rate = target_sample_rate;
    return true;
}

inline bool prepare_train_audio_window(const TrainAudio& in, int target_samples, int start_sample,
                                       TrainAudio& out, std::string& err) {
    if (in.n_samples <= 0 || in.n_channels <= 0 || in.sample_rate <= 0) {
        err = "input audio is empty";
        return false;
    }
    if (target_samples <= 0) {
        out = in;
        return true;
    }
    if (start_sample < 0) {
        err = "start_sample must be non-negative";
        return false;
    }
    out.n_samples = target_samples;
    out.n_channels = in.n_channels;
    out.sample_rate = in.sample_rate;
    out.samples.assign((size_t)target_samples * in.n_channels, 0.0f);
    for (int ch = 0; ch < in.n_channels; ++ch) {
        const float* src = in.samples.data() + (size_t)ch * in.n_samples;
        float* dst = out.samples.data() + (size_t)ch * target_samples;
        for (int i = 0; i < target_samples; ++i) {
            const int si = start_sample + i;
            if (si >= 0 && si < in.n_samples) dst[i] = src[si];
        }
    }
    return true;
}

} // namespace sa3
