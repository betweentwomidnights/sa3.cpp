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

        // The dispatcher must not change the data for a WAV fixture: whichever route it takes,
        // the samples have to match what ffmpeg produced above.
        if (sa3::train_audio_path_is_wav(mp3)) {
            sa3::TrainAudio via_dispatch;
            err.clear();
            fails += expect(sa3::decode_train_audio_file(mp3, 44100, 2, via_dispatch, err), err.c_str());
            bool identical = via_dispatch.n_samples == audio.n_samples &&
                             via_dispatch.n_channels == audio.n_channels &&
                             via_dispatch.samples.size() == audio.samples.size();
            for (size_t i = 0; identical && i < audio.samples.size(); ++i)
                identical = via_dispatch.samples[i] == audio.samples[i];
            fails += expect(identical, "native wav decode matches ffmpeg exactly");
        }
    }
    {   // Native WAV decode: no ffmpeg, and byte-equal to what the ffmpeg path would produce.
        const std::filesystem::path tmp =
            std::filesystem::temp_directory_path() / "sa3-train-audio-native.wav";
        // Values exactly representable in the 16-bit PCM the writer emits (k/32768).
        const int n = 8;
        std::vector<float> stereo((size_t)n * 2);
        for (int i = 0; i < n; ++i) {
            stereo[(size_t)i]     = (float)(i * 256) / 32768.0f;
            stereo[(size_t)n + i] = (float)(-i * 256) / 32768.0f;
        }
        sa3::write_wav_planar(tmp.string(), stereo.data(), n, 2, 44100);

        sa3::TrainAudio nat;
        err.clear();
        fails += expect(sa3::decode_wav_planar_native(tmp.string(), 44100, 2, nat, err), err.c_str());
        fails += expect(nat.n_samples == n && nat.n_channels == 2 && nat.sample_rate == 44100,
                        "native wav shape");
        // write_wav_planar scales by 32767, read_wav_planar divides by 32768, so the round trip
        // is not the identity -- compare against what that pair actually yields.
        auto through_s16 = [](float v) { return (float)(int16_t)(v * 32767.0f) / 32768.0f; };
        bool same = nat.samples.size() == stereo.size();
        for (size_t i = 0; same && i < stereo.size(); ++i)
            same = nat.samples[i] == through_s16(stereo[i]);
        fails += expect(same, "native wav round-trip values");

        // A rate the caller did not ask for must decline rather than resample.
        sa3::TrainAudio wrong_rate;
        err.clear();
        fails += expect(!sa3::decode_wav_planar_native(tmp.string(), 48000, 2, wrong_rate, err),
                        "native wav declines a rate mismatch");

        // Mono source fanned out to N channels, matching ffmpeg -ac N.
        const std::filesystem::path mono_path =
            std::filesystem::temp_directory_path() / "sa3-train-audio-native-mono.wav";
        std::vector<float> mono((size_t)n);
        for (int i = 0; i < n; ++i) mono[(size_t)i] = (float)(i * 128) / 32768.0f;
        sa3::write_wav_planar(mono_path.string(), mono.data(), n, 1, 44100);
        sa3::TrainAudio fanned;
        err.clear();
        fails += expect(sa3::decode_wav_planar_native(mono_path.string(), 44100, 2, fanned, err),
                        err.c_str());
        bool duplicated = fanned.n_channels == 2 && fanned.n_samples == n;
        for (int i = 0; duplicated && i < n; ++i)
            duplicated = fanned.samples[(size_t)i] == through_s16(mono[(size_t)i]) &&
                         fanned.samples[(size_t)n + i] == through_s16(mono[(size_t)i]);
        fails += expect(duplicated, "native wav mono fan-out");

        std::filesystem::remove(tmp);
        std::filesystem::remove(mono_path);
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
