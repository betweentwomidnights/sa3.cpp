// train_audio.h - audio loading helpers for native SA3 LoRA training.
#pragma once

#include "wav.h"

#include <array>
#include <cctype>
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

// True for a path ending in ".wav" (case-insensitive).
inline bool train_audio_path_is_wav(const std::string& path) {
    if (path.size() < 4) return false;
    std::string tail = path.substr(path.size() - 4);
    for (char& c : tail) c = (char)std::tolower((unsigned char)c);
    return tail == ".wav";
}

// Read a WAV straight into the training layout, with no subprocess. Declines (returns false, err
// set) whenever the file does not already match the requested rate/channel layout, because the
// only resampler in this tree is linear (audio_post.h) -- adequate for a2a conditioning, but not
// what the reference training data was resampled with, and training data is not the place to
// silently swap in a worse one. Those files fall back to ffmpeg, which resamples properly.
inline bool decode_wav_planar_native(const std::string& path, int target_sample_rate, int target_channels,
                                     TrainAudio& out, std::string& err) {
    int n_samples = 0, n_ch = 0, sample_rate = 0;
    std::vector<float> planar;
    try {
        planar = read_wav_planar(path, n_samples, n_ch, sample_rate);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (n_samples <= 0 || n_ch <= 0) {
        err = "decoded audio has invalid sample count for " + path;
        return false;
    }
    if (sample_rate != target_sample_rate) {
        err = path + " is " + std::to_string(sample_rate) + " Hz, not " +
              std::to_string(target_sample_rate) + " Hz";
        return false;
    }
    if (n_ch == target_channels) {
        out.samples = std::move(planar);
    } else if (n_ch == 1) {
        // ffmpeg -ac N on a mono source copies the channel; match that exactly.
        out.samples.assign((size_t)n_samples * target_channels, 0.0f);
        for (int ch = 0; ch < target_channels; ++ch)
            std::memcpy(out.samples.data() + (size_t)ch * n_samples, planar.data(),
                        (size_t)n_samples * sizeof(float));
    } else {
        err = path + " has " + std::to_string(n_ch) + " channels, not " +
              std::to_string(target_channels);
        return false;
    }
    out.n_samples = n_samples;
    out.n_channels = target_channels;
    out.sample_rate = target_sample_rate;
    return true;
}

// Decode one dataset file to planar float at the training rate. WAV files already in the training
// layout are read natively; everything else shells out to ffmpeg. An embedded host that cannot
// spawn processes should supply samples through TrainAudioSource instead of going through here.
inline bool decode_train_audio_file(const std::string& path, int target_sample_rate, int target_channels,
                                    TrainAudio& out, std::string& err) {
    if (train_audio_path_is_wav(path)) {
        std::string native_err;
        if (decode_wav_planar_native(path, target_sample_rate, target_channels, out, native_err))
            return true;
    }
    return decode_mp3_planar_ffmpeg(path, target_sample_rate, target_channels, out, err);
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
