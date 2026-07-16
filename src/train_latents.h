// train_latents.h - pre-encoded latent cache for native SA3 LoRA training.
//
// The reference pipeline (gary4local services/sa3/dataset_processing/pre_encode.py +
// stable-audio-3 PreEncodedDataset) never encodes audio crops during training: each source file
// is encoded ONCE, full-length, and training random-crops in LATENT space. Per Zach (SA3's
// creator) pre-encoding is required for training to behave; mechanically it also means every
// latent frame sees its full receptive-field context instead of crop-boundary artifacts, and it
// lets the trainer drop the autoencoder from VRAM after startup.
//
// Faithful reproduction of the reference method:
//  - audio is laid on the reference's padded grid: max_samples = (600s * sr) // ds * ds
//    (26,456,064 samples / 6459 latent frames at 44.1kHz, ds=4096); longer files are cropped
//  - encoded CHUNKED exactly like encode_audio(chunked=True): chunk 128 latents, overlap 32,
//    hop 96, final chunk anchored to the padded end, half-overlap trimmed at inner edges
//    (chunks that only cover padding are skipped: training crops never read them)
//  - n_valid = ceil(actual_samples / ds) — identical to the reference's nearest-interpolated
//    padding mask; crops are drawn from [0, last_valid_ix - crop] like PreEncodedDataset
//  - optional per-track latent-RMS loudness fix (pre_encode.py --per-track-target-latent-rms,
//    0.9 for the ratatat reference runs): iteratively rescale the audio until the encoded
//    latent RMS over the valid region hits the target (<= 4 correction rounds, 3% tolerance)
//  - seconds_total = round(actual_samples / sr, 3) — fractional, like the .json sidecars; it
//    feeds both the seconds conditioning and the dist-shift effective length
//
// Alternatively train_load_latent_dir() reads the .npy/.json sidecars a gary4local job already
// produced (encoded/latents/<model>/), so a parity run can train on BIT-IDENTICAL data to the
// PyTorch reference run.
//
// Known deviations (accepted): the reference encoded with a fp16 autoencoder (--half); we run
// the normal sa3.cpp encode (f16 weights, f32 activations). Files shorter than the crop length
// are an error here instead of silence/zero-padded (the reference pads via silence.npy, which
// gary4local's pre-encode does not produce anyway).
#pragma once

#include "train_audio.h"
#include "train_same.h"

#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace sa3 {

constexpr double kPreEncodePadSeconds = 600.0;  // gary4local pre_encode.py default max duration
constexpr int    kPreEncodeChunkLat   = 128;    // encode_audio(chunked=True) defaults
constexpr int    kPreEncodeOverlapLat = 32;
constexpr int    kPreEncodeNormIters  = 4;      // loudness-fix defaults (--norm-iters/--norm-tol)
constexpr float  kPreEncodeNormTol    = 0.03f;

struct TrainLatentEntry {
    std::vector<float> z;        // [latent, n_valid] in ggml memory order (latent fastest)
    int latent = 0;
    int n_valid = 0;             // valid latent frames = ceil(actual_samples / ds)
    double seconds_total = 0.0;  // round(actual_samples / sr, 3), fractional like the sidecars
    float gain = 1.0f;           // audio gain applied by the loudness fix (1.0 = off/converged at 1)
    float rms_pre = 0.0f;        // latent RMS before/after the loudness fix
    float rms_achieved = 0.0f;
    int norm_rounds = 0;
};

using TrainLatentCache = std::map<std::string, TrainLatentEntry>;

// One full-file encode pass on the reference chunk grid at a given audio gain.
// z is filled as [latent, total_latents]; only chunks whose kept frames intersect
// [0, n_valid) are encoded (the rest is padding that training crops never read).
inline bool train_pre_encode_stitched(GgufModel& ae, const SameConfig& sc, const TrainAudio& full,
                                      int64_t actual_samples, int n_valid, int total_latents,
                                      float gain, std::vector<float>& z, std::string& err) {
    const int ds = sc.patch_size * sc.output_seg;
    const int ch = sc.out_channels / sc.patch_size;
    const int chunk_lat = kPreEncodeChunkLat;
    const int hop_lat = chunk_lat - kPreEncodeOverlapLat;
    const int half = kPreEncodeOverlapLat / 2;
    const int64_t chunk_samples = (int64_t)chunk_lat * ds;

    // reference grid: range(0, total - chunk + 1, hop), plus a final chunk anchored to the end
    std::vector<int> starts;
    for (int s = 0; s + chunk_lat <= total_latents; s += hop_lat) starts.push_back(s);
    if (starts.empty() || starts.back() != total_latents - chunk_lat)
        starts.push_back(total_latents - chunk_lat);
    const int n_chunks = (int)starts.size();

    z.assign((size_t)sc.latent * (size_t)total_latents, 0.0f);
    TrainAudio win;
    win.n_channels = ch;
    win.sample_rate = full.sample_rate;
    win.n_samples = (int)chunk_samples;
    win.samples.resize((size_t)ch * (size_t)chunk_samples);

    for (int i = 0; i < n_chunks; ++i) {
        const bool first = i == 0;
        const bool last = i == n_chunks - 1;
        const int out_start = last ? total_latents - chunk_lat : starts[i];
        const int left = first ? 0 : half;
        const int right = last ? chunk_lat : chunk_lat - half;
        if (out_start + left >= n_valid) continue;  // kept frames are pure padding

        const int64_t s0 = (int64_t)starts[i] * ds;
        for (int c = 0; c < ch; ++c) {
            const float* src = full.samples.data() + (size_t)c * (size_t)full.n_samples;
            float* dst = win.samples.data() + (size_t)c * (size_t)chunk_samples;
            for (int64_t s = 0; s < chunk_samples; ++s) {
                const int64_t idx = s0 + s;
                dst[s] = idx < actual_samples ? src[idx] * gain : 0.0f;  // zero pad past the signal
            }
        }
        TrainLatents lat;
        if (!encode_train_audio_to_latents(ae, sc, win, lat, err)) return false;
        if (lat.latent != sc.latent || lat.frames != chunk_lat) {
            err = "pre-encode chunk produced unexpected latent shape";
            return false;
        }
        for (int f = left; f < right; ++f)
            std::memcpy(z.data() + (size_t)(out_start + f) * sc.latent,
                        lat.z.data() + (size_t)f * sc.latent, sizeof(float) * (size_t)sc.latent);
    }
    return true;
}

// Full-file pre-encode with the optional per-track latent-RMS loudness fix
// (target_rms <= 0 disables it, matching pre_encode.py's flag default).
inline bool train_pre_encode_file(GgufModel& ae, const SameConfig& sc, const TrainAudio& full,
                                  float target_rms, TrainLatentEntry& out, std::string& err) {
    const int ds = sc.patch_size * sc.output_seg;
    const int64_t max_samples = (int64_t)(kPreEncodePadSeconds * full.sample_rate) / ds * ds;
    const int total_latents = (int)(max_samples / ds);
    const int64_t actual = std::min<int64_t>(full.n_samples, max_samples);
    if (actual <= 0) {
        err = "pre-encode got empty audio";
        return false;
    }
    const int n_valid = std::min(total_latents, (int)std::ceil((double)actual / (double)ds));

    float gain = 1.0f;
    float measured = 0.0f, pre = 0.0f;
    int rounds = 0;
    std::vector<float> z;
    const int iters = target_rms > 0.0f ? kPreEncodeNormIters : 0;
    for (int it = 0; it <= iters; ++it) {
        if (!train_pre_encode_stitched(ae, sc, full, actual, n_valid, total_latents, gain, z, err))
            return false;
        double ss = 0.0;
        const size_t n = (size_t)sc.latent * (size_t)n_valid;
        for (size_t k = 0; k < n; ++k) ss += (double)z[k] * z[k];
        measured = std::max((float)std::sqrt(ss / (double)n), 1e-6f);
        if (it == 0) pre = measured;
        rounds = it;
        if (target_rms <= 0.0f) break;
        if (std::fabs(measured - target_rms) / target_rms <= kPreEncodeNormTol || it == iters) break;
        gain *= target_rms / measured;
    }

    out.latent = sc.latent;
    out.n_valid = n_valid;
    out.z.assign(z.begin(), z.begin() + (size_t)sc.latent * (size_t)n_valid);
    out.seconds_total = std::round((double)actual / (double)full.sample_rate * 1000.0) / 1000.0;
    out.gain = gain;
    out.rms_pre = pre;
    out.rms_achieved = measured;
    out.norm_rounds = rounds;
    return true;
}

// Minimal .npy reader for the sidecar latents: v1/v2 header, little-endian f32, C-order 2D.
inline bool train_load_npy_f32_2d(const std::string& path, int64_t& d0, int64_t& d1,
                                  std::vector<float>& data, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) { err = "not a .npy file: " + path; return false; }
    const int ver = (unsigned char)magic[6];
    uint32_t hlen = 0;
    if (ver == 1) {
        uint16_t h16 = 0;
        f.read((char*)&h16, 2);
        hlen = h16;
    } else {
        f.read((char*)&hlen, 4);
    }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    if (!f) { err = "truncated .npy header: " + path; return false; }
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("'fortran_order': False") == std::string::npos) {
        err = "unsupported .npy layout (need C-order '<f4'): " + path;
        return false;
    }
    const size_t sp = hdr.find("'shape':");
    const size_t po = sp == std::string::npos ? std::string::npos : hdr.find('(', sp);
    long long a = 0, b = 0;
    if (po == std::string::npos || std::sscanf(hdr.c_str() + po, "(%lld, %lld", &a, &b) != 2 || a <= 0 || b <= 0) {
        err = "cannot parse .npy shape: " + path;
        return false;
    }
    d0 = a;
    d1 = b;
    data.resize((size_t)a * (size_t)b);
    f.read((char*)data.data(), (std::streamsize)(data.size() * sizeof(float)));
    if (!f) { err = "truncated .npy data: " + path; return false; }
    return true;
}

// Load a gary4local pre-encode output dir (encoded/latents/<model>/): every .npy + .json pair
// becomes a cache entry keyed by file stem. Latents are transposed from the npy's [C, T]
// C-order into ggml order and trimmed to the padding mask's valid region.
inline bool train_load_latent_dir(const std::string& dir, int expect_latent,
                                  TrainLatentCache& cache, std::string& err) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(dir)) { err = "latents dir not found: " + dir; return false; }
    for (const auto& de : fs::recursive_directory_iterator(dir)) {
        if (!de.is_regular_file() || de.path().extension() != ".npy") continue;
        const std::string stem = de.path().stem().string();
        if (stem == "silence" || stem.empty() || stem[0] == '.') continue;
        fs::path jp = de.path();
        jp.replace_extension(".json");
        if (!fs::exists(jp)) continue;

        int64_t C = 0, T = 0;
        std::vector<float> raw;
        if (!train_load_npy_f32_2d(de.path().string(), C, T, raw, err)) return false;
        if (expect_latent > 0 && C != expect_latent) {
            err = "latent dim mismatch in " + de.path().string() + " (got " + std::to_string(C) +
                  ", model wants " + std::to_string(expect_latent) + ")";
            return false;
        }

        std::ifstream jf(jp);
        std::string js((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
        yyjson_doc* doc = yyjson_read(js.data(), js.size(), 0);
        if (!doc) { err = "cannot parse " + jp.string(); return false; }
        yyjson_val* root = yyjson_doc_get_root(doc);
        int n_valid = (int)T;
        yyjson_val* pm = yyjson_obj_get(root, "padding_mask");
        if (pm && yyjson_is_arr(pm)) {
            // valid length = index of the last 1 in the mask, + 1 (PreEncodedDataset's last_ix)
            int last_one = -1;
            size_t idx, max;
            yyjson_val* v;
            yyjson_arr_foreach(pm, idx, max, v) {
                if (yyjson_get_int(v) == 1) last_one = (int)idx;
            }
            if (last_one >= 0) n_valid = std::min((int)T, last_one + 1);
        }
        TrainLatentEntry e;
        e.latent = (int)C;
        e.n_valid = n_valid;
        yyjson_val* st = yyjson_obj_get(root, "seconds_total");
        e.seconds_total = st ? yyjson_get_num(st) : 0.0;
        yyjson_val* ga = yyjson_obj_get(root, "audio_gain_applied");
        if (ga) e.gain = (float)yyjson_get_num(ga);
        yyjson_val* ra = yyjson_obj_get(root, "latent_rms_achieved");
        if (ra) e.rms_achieved = (float)yyjson_get_num(ra);
        yyjson_doc_free(doc);

        e.z.resize((size_t)C * (size_t)n_valid);
        for (int64_t t = 0; t < n_valid; ++t)
            for (int64_t c = 0; c < C; ++c)
                e.z[(size_t)t * C + c] = raw[(size_t)c * T + t];
        cache[stem] = std::move(e);
    }
    if (cache.empty()) { err = "no .npy/.json latent pairs found in " + dir; return false; }
    return true;
}

// Latent-space crop: [latent, frames] starting at latent frame `start` (contiguous in ggml order).
inline void train_crop_latents(const TrainLatentEntry& e, int start, int frames, TrainLatents& out) {
    out.latent = e.latent;
    out.frames = frames;
    out.z.assign(e.z.begin() + (size_t)start * e.latent,
                 e.z.begin() + (size_t)(start + frames) * e.latent);
}

} // namespace sa3
