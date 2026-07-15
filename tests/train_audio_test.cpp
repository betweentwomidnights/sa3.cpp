#include "train_audio.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    sa3::TrainAudio audio;
    sa3::TrainAudio win;
    std::string err;
    int fails = 0;
    if (argc > 1) {
        const std::string mp3 = argv[1];
        if (!std::filesystem::exists(mp3)) {
            std::fprintf(stderr, "FAIL: mp3 fixture not found: %s\n", mp3.c_str());
            return 1;
        }
        fails += expect(sa3::decode_mp3_planar_ffmpeg(mp3, 44100, 2, audio, err), err.c_str());
        fails += expect(audio.sample_rate == 44100, "sample rate");
        fails += expect(audio.n_channels == 2, "channels");
        fails += expect(audio.n_samples > 44100, "decoded more than one second");
        double energy = 0.0;
        const int n = audio.n_samples < 44100 ? audio.n_samples : 44100;
        for (int i = 0; i < n; ++i) energy += std::fabs(audio.samples[(size_t)i]);
        fails += expect(energy > 1.0, "decoded non-silent audio");
        err.clear();
        fails += expect(sa3::prepare_train_audio_window(audio, 4096, 100, win, err), "window prepare");
        fails += expect(win.n_samples == 4096 && win.n_channels == 2, "window shape");
        fails += expect(win.samples[0] == audio.samples[100], "window crop channel 0");
        fails += expect(win.samples[(size_t)4096] == audio.samples[(size_t)audio.n_samples + 100], "window crop channel 1");
    }
    sa3::TrainAudio tiny;
    tiny.n_samples = 2;
    tiny.n_channels = 1;
    tiny.sample_rate = 44100;
    tiny.samples = {0.25f, -0.5f};
    err.clear();
    fails += expect(sa3::prepare_train_audio_window(tiny, 4, 0, win, err), "window pad");
    fails += expect(win.samples.size() == 4 && win.samples[0] == 0.25f && win.samples[1] == -0.5f &&
                    win.samples[2] == 0.0f && win.samples[3] == 0.0f, "window pad values");
    if (fails) return 1;
    if (argc > 1) {
        std::printf("train_audio_test: ok (%d samples, %d ch)\n", audio.n_samples, audio.n_channels);
    } else {
        std::printf("train_audio_test: ok (window tests; pass an MP3 path to test ffmpeg decode)\n");
    }
    return 0;
}
