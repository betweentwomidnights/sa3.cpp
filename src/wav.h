// wav.h — minimal 16-bit PCM WAV writer.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace sa3 {

namespace wav_detail {

struct FileHandle {
    FILE* f = nullptr;
    ~FileHandle() { if (f) fclose(f); }
};

inline uint16_t u16le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

inline uint32_t u32le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline int32_t s24le(const uint8_t* p) {
    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    if (v & 0x800000u) v |= 0xff000000u;
    return (int32_t)v;
}

inline int32_t s32le(const uint8_t* p) {
    uint32_t u = u32le(p);
    int32_t v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

inline void read_exact(FILE* f, void* dst, size_t bytes, const std::string& what) {
    if (bytes && fread(dst, 1, bytes, f) != bytes) {
        throw std::runtime_error("unexpected end of file while reading " + what);
    }
}

inline std::vector<int16_t> interleave_planar_i16(const float* data, int n_samples,
                                                  int n_ch, int channel_stride) {
    if (!data || n_samples < 0 || n_ch < 1 || channel_stride < n_samples)
        throw std::invalid_argument("invalid planar audio geometry");
    auto clip = [](float v) {
        v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
        return (int16_t)(v * 32767.0f);
    };
    std::vector<int16_t> inter((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; ++s)
        for (int c = 0; c < n_ch; ++c)
            inter[(size_t)s*n_ch + c] = clip(data[(size_t)c*channel_stride + s]);
    return inter;
}

} // namespace wav_detail

// Write interleaved 16-bit WAV from planar f32 channels whose source planes may be
// longer than the requested output. channel_stride is the distance between channel
// starts; this matters when cropping a decoded buffer without first repacking it.
inline void write_wav_planar_strided(const std::string& path, const float* data, int n_samples,
                                     int n_ch, int sample_rate, int channel_stride) {
    std::vector<int16_t> inter = wav_detail::interleave_planar_i16(
        data, n_samples, n_ch, channel_stride);

    const uint32_t data_bytes = (uint32_t)inter.size() * sizeof(int16_t);
    const uint32_t byte_rate  = sample_rate * n_ch * 2;
    FILE* f = fopen(path.c_str(), "wb");
    auto u32 = [&](uint32_t v){ fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v){ fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16((uint16_t)n_ch);
    u32(sample_rate); u32(byte_rate); u16((uint16_t)(n_ch*2)); u16(16);
    fwrite("data", 1, 4, f); u32(data_bytes);
    fwrite(inter.data(), 1, data_bytes, f);
    fclose(f);
}

// Write a complete tightly-packed planar buffer.
inline void write_wav_planar(const std::string& path, const float* data, int n_samples,
                             int n_ch, int sample_rate) {
    write_wav_planar_strided(path, data, n_samples, n_ch, sample_rate, n_samples);
}

// Same WAV bytes in memory (for an HTTP response body, etc.) instead of a file.
inline std::string wav_planar_bytes(const float* data, int n_samples, int n_ch, int sample_rate) {
    auto clip = [](float v){ v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v); return (int16_t)(v * 32767.0f); };
    std::vector<int16_t> inter((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; s++)
        for (int c = 0; c < n_ch; c++)
            inter[(size_t)s*n_ch + c] = clip(data[(size_t)c*n_samples + s]);
    const uint32_t data_bytes = (uint32_t)inter.size() * sizeof(int16_t);
    const uint32_t byte_rate  = sample_rate * n_ch * 2;
    std::string out;
    auto bytes = [&](const void* p, size_t n){ out.append((const char*)p, n); };
    auto u32 = [&](uint32_t v){ bytes(&v, 4); };
    auto u16 = [&](uint16_t v){ bytes(&v, 2); };
    bytes("RIFF", 4); u32(36 + data_bytes); bytes("WAVE", 4);
    bytes("fmt ", 4); u32(16); u16(1); u16((uint16_t)n_ch);
    u32(sample_rate); u32(byte_rate); u16((uint16_t)(n_ch*2)); u16(16);
    bytes("data", 4); u32(data_bytes);
    bytes(inter.data(), data_bytes);
    return out;
}

// Read a PCM/float WAV held in memory into planar f32 channels (ch0[0..n-1],
// ch1[0..n-1], ...), the same layout write_wav_planar consumes and same_encode
// expects ([L, ch], L fastest). `what` names the source in error messages.
inline std::vector<float> read_wav_planar_bytes(const uint8_t* bytes, size_t size,
                                                const std::string& what, int& n_samples,
                                                int& n_ch, int& sample_rate, bool clamp = true) {
    using namespace wav_detail;
    const std::string& path = what;
    if (size < 12 || memcmp(bytes, "RIFF", 4) || memcmp(bytes + 8, "WAVE", 4)) {
        throw std::runtime_error(path + " is not a RIFF/WAVE file");
    }

    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data_start = nullptr;
    size_t data_size = 0;

    size_t pos = 12;
    while (size - pos >= 8) {
        const uint8_t* id = bytes + pos;
        const uint32_t sz = u32le(bytes + pos + 4);
        pos += 8;
        if (sz > size - pos) {
            // A truncated fmt or data chunk is an error; a truncated trailing
            // chunk of any other kind ends the file, as seeking past it did.
            if (!memcmp(id, "data", 4) || !memcmp(id, "fmt ", 4))
                throw std::runtime_error("unexpected end of file while reading " +
                                         std::string(!memcmp(id, "data", 4) ? "data chunk" : "fmt chunk"));
            break;
        }
        if (!memcmp(id, "fmt ", 4)) {
            if (sz < 16) throw std::runtime_error(path + ": invalid fmt chunk");
            const uint8_t* fmt_data = bytes + pos;
            fmt = u16le(fmt_data);
            ch = u16le(fmt_data + 2);
            rate = u32le(fmt_data + 4);
            bits = u16le(fmt_data + 14);

            // WAVE_FORMAT_EXTENSIBLE stores the real codec in the SubFormat GUID.
            if (fmt == 0xfffe && sz >= 40) {
                uint16_t sub_format = u16le(fmt_data + 24);
                if (sub_format == 1 || sub_format == 3) fmt = sub_format;
            }
        } else if (!memcmp(id, "data", 4)) {
            data_start = bytes + pos;
            data_size = sz;
        }
        // Chunks are word-aligned; unknown ones are skipped.
        pos += sz + (sz & 1);
        if (pos > size) break;
    }
    const std::vector<uint8_t> data(data_start, data_start + data_size);

    if ((fmt != 1 && fmt != 3) || ch == 0 || rate == 0 || data.empty()) {
        throw std::runtime_error(path + ": unsupported or empty WAV (fmt=" + std::to_string(fmt) +
                                 " bits=" + std::to_string(bits) + " ch=" + std::to_string(ch) + ")");
    }
    if (!((fmt == 1 && bits == 8) || bits == 16 || bits == 24 || bits == 32 || (fmt == 3 && bits == 64))) {
        throw std::runtime_error(path + ": unsupported WAV bit depth " + std::to_string(bits));
    }
    if (fmt == 3 && !(bits == 32 || bits == 64)) {
        throw std::runtime_error(path + ": unsupported float WAV bit depth " + std::to_string(bits));
    }

    const size_t bytes_per_sample = bits / 8;
    const size_t frame_bytes = bytes_per_sample * ch;
    if (frame_bytes == 0 || data.size() < frame_bytes) {
        throw std::runtime_error(path + ": WAV data chunk is too small");
    }

    n_ch = ch; sample_rate = (int)rate; n_samples = (int)(data.size() / frame_bytes);
    std::vector<float> planar((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; s++) {
        for (int c = 0; c < n_ch; c++) {
            const uint8_t* p = data.data() + ((size_t)s * n_ch + c) * bytes_per_sample;
            float v = 0.0f;
            if (fmt == 3) {
                if (bits == 32) {
                    std::memcpy(&v, p, sizeof(float));
                } else {
                    double d = 0.0;
                    std::memcpy(&d, p, sizeof(double));
                    v = (float)d;
                }
            } else if (bits == 8) {
                v = (int(p[0]) - 128) / 128.0f;
            } else if (bits == 16) {
                int16_t sample = (int16_t)u16le(p);
                v = sample / 32768.0f;
            } else if (bits == 24) {
                v = s24le(p) / 8388608.0f;
            } else {
                v = s32le(p) / 2147483648.0f;
            }
            planar[(size_t)c*n_samples + s] = clamp ? std::max(-1.0f, std::min(1.0f, v)) : v;
        }
    }
    return planar;
}

// Read a PCM/float WAV file into planar f32 channels; see read_wav_planar_bytes.
inline std::vector<float> read_wav_planar(const std::string& path, int& n_samples,
                                          int& n_ch, int& sample_rate, bool clamp = true) {
    using namespace wav_detail;

#ifdef _WIN32
    FileHandle fh{_wfopen(std::filesystem::u8path(path).c_str(), L"rb")};
#else
    FileHandle fh{fopen(path.c_str(), "rb")};
#endif
    FILE* f = fh.f;
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> bytes;
    uint8_t chunk[1 << 16];
    size_t got;
    while ((got = fread(chunk, 1, sizeof(chunk), f)) > 0) bytes.insert(bytes.end(), chunk, chunk + got);
    return read_wav_planar_bytes(bytes.data(), bytes.size(), path, n_samples, n_ch, sample_rate, clamp);
}

} // namespace sa3
