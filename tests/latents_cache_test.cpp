// latents_cache_test: the pre-encode cache round-trips, and refuses to serve a stale entry.
//
// A false miss costs an encode. A false hit trains on the wrong latents and says nothing, so the
// key checks below are the point of the test, not the round-trip.
//
// The npy is also asserted against the NumPy format spec rather than against our own reader,
// because the format exists so that gary4local / underfit / the official PyTorch pre-encode can
// read what we write and we can read what they write.
#include "train_latents.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int fails = 0;

static void expect(bool ok, const std::string& msg) {
    if (!ok) {
        std::printf("  FAIL %s\n", msg.c_str());
        fails++;
    }
}

static sa3::TrainLatentEntry make_entry(int latent, int frames) {
    sa3::TrainLatentEntry e;
    e.latent = latent;
    e.n_valid = frames;
    e.z.resize((size_t)latent * (size_t)frames);
    for (int t = 0; t < frames; ++t)
        for (int c = 0; c < latent; ++c)
            e.z[(size_t)t * latent + c] = 0.001f * (float)(t * latent + c) - 0.5f;
    e.seconds_total = 12.345;
    e.gain = 0.5539f;
    e.rms_pre = 1.2232f;
    e.rms_achieved = 0.9248f;
    e.norm_rounds = 4;
    return e;
}

static sa3::TrainLatentKey make_key() {
    sa3::TrainLatentKey k;
    k.autoencoder = "stable-audio-3-medium-same-l-v1.0-Q4_K_M.gguf";
    k.source_fingerprint = "0123456789abcdef";
    k.target_latent_rms = 0.9f;
    return k;
}

// The parts of the NumPy spec a foreign reader depends on.
static void check_npy_spec(const fs::path& p, int64_t d0, int64_t d1) {
    std::ifstream f(p, std::ios::binary);
    expect((bool)f, "npy opens");
    if (!f) return;
    std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    expect(all.size() > 10, "npy is not truncated");
    if (all.size() <= 10) return;

    expect(std::memcmp(all.data(), "\x93NUMPY", 6) == 0, "npy magic");
    expect((unsigned char)all[6] == 1 && (unsigned char)all[7] == 0, "npy version is 1.0");
    uint16_t hlen = 0;
    std::memcpy(&hlen, all.data() + 8, 2);
    expect((10 + hlen) % 64 == 0, "npy header is 64-byte aligned");
    const std::string hdr = all.substr(10, hlen);
    expect(!hdr.empty() && hdr.back() == '\n', "npy header ends with a newline");
    expect(hdr.find("'descr': '<f4'") != std::string::npos, "npy descr is <f4");
    expect(hdr.find("'fortran_order': False") != std::string::npos, "npy is C-order");
    const std::string shape = "'shape': (" + std::to_string(d0) + ", " + std::to_string(d1) + ")";
    expect(hdr.find(shape) != std::string::npos, "npy shape is " + shape + " (channels first)");
    expect(all.size() - 10 - hlen == (size_t)d0 * (size_t)d1 * sizeof(float),
           "npy data length matches its shape");
}

int main() {
    std::printf("latents_cache_test\n");
    const fs::path dir = fs::temp_directory_path() / "sa3_latents_cache_test";
    std::error_code ec;
    fs::remove_all(dir, ec);

    const int latent = 8, frames = 37;
    const sa3::TrainLatentEntry src = make_entry(latent, frames);
    const sa3::TrainLatentKey key = make_key();
    const std::string name = sa3::train_latent_cache_name("a track [with brackets]");
    std::string err;

    expect(sa3::train_write_latent_entry(dir.string(), name, src, key, err),
           "write succeeds: " + err);
    check_npy_spec(dir / (name + ".npy"), latent, frames);
    expect(!fs::exists(dir / (name + ".npy.part")), "no .part left behind");

    // round-trip: every value and every scalar the trainer reads back out
    sa3::TrainLatentEntry got;
    expect(sa3::train_read_latent_entry(dir.string(), name, latent, key, got), "matching key hits");
    expect(got.latent == src.latent && got.n_valid == src.n_valid, "shape survives");
    expect(got.z.size() == src.z.size(), "element count survives");
    double worst = 0.0;
    for (size_t i = 0; i < std::min(got.z.size(), src.z.size()); ++i)
        worst = std::max(worst, (double)std::fabs(got.z[i] - src.z[i]));
    expect(worst == 0.0, "latents are bit-identical, not merely close");
    expect(got.seconds_total == src.seconds_total, "seconds_total survives");
    expect(got.gain == src.gain && got.rms_pre == src.rms_pre &&
           got.rms_achieved == src.rms_achieved && got.norm_rounds == src.norm_rounds,
           "loudness-fix results survive");

    // a filename is not a key: each of these must miss
    sa3::TrainLatentEntry ignored;
    sa3::TrainLatentKey k = key;
    k.source_fingerprint = "ffffffffffffffff";
    expect(!sa3::train_read_latent_entry(dir.string(), name, latent, k, ignored),
           "different audio misses");
    k = key;
    k.target_latent_rms = 0.5f;
    expect(!sa3::train_read_latent_entry(dir.string(), name, latent, k, ignored),
           "different target_latent_rms misses");
    k = key;
    k.autoencoder = "stable-audio-3-medium-same-l-v1.0-Q8_0.gguf";
    expect(!sa3::train_read_latent_entry(dir.string(), name, latent, k, ignored),
           "different autoencoder misses");
    k = key;
    k.version = key.version + 1;
    expect(!sa3::train_read_latent_entry(dir.string(), name, latent, k, ignored),
           "different cache version misses");
    expect(!sa3::train_read_latent_entry(dir.string(), name, latent + 1, key, ignored),
           "different latent width misses");
    expect(!sa3::train_read_latent_entry(dir.string(), "no such entry", latent, key, ignored),
           "absent entry misses");

    // train_load_latent_dir is how a gary4local/underfit output is consumed; a cache directory
    // has to be readable by it too, or the format claim is empty
    sa3::TrainLatentCache cache;
    expect(sa3::train_load_latent_dir(dir.string(), latent, cache, err),
           "the gary4local reader accepts a cache directory: " + err);
    expect(cache.size() == 1, "one entry seen by the gary4local reader");
    auto it = cache.find(name);
    expect(it != cache.end(), "keyed by plain stem, no fingerprint suffix");
    if (it != cache.end()) {
        expect(it->second.n_valid == frames, "gary4local reader agrees on n_valid");
        double w = 0.0;
        for (size_t i = 0; i < std::min(it->second.z.size(), src.z.size()); ++i)
            w = std::max(w, (double)std::fabs(it->second.z[i] - src.z[i]));
        expect(w == 0.0, "gary4local reader recovers the same latents");
    }

    // a sidecar without key fields is a gary4local file, not a cache entry: it must not be served
    // as one, or --latents-dir data would silently satisfy a cache lookup for different audio
    {
        std::ofstream j(dir / (name + ".json"));
        j << "{\"seconds_total\": 12.345, \"padding_mask\": [";
        for (int i = 0; i < frames; ++i) j << (i ? "," : "") << "1";
        j << "]}\n";
    }
    expect(!sa3::train_read_latent_entry(dir.string(), name, latent, key, ignored),
           "an unkeyed gary4local sidecar is not treated as a cache hit");

    fs::remove_all(dir, ec);
    std::printf(fails ? "FAILED (%d)\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
