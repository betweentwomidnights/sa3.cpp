// train_same.h - SAME autoencoder helpers for native SA3 LoRA training.
#pragma once

#include "gguf_model.h"
#include "same_ae.h"
#include "train_audio.h"

#include <cstring>
#include <string>
#include <vector>

namespace sa3 {

struct TrainLatents {
    std::vector<float> z; // [latent, T] in ggml memory order
    int latent = 0;
    int frames = 0;
};

inline void train_set_positions(ggml_tensor* p, int64_t n) {
    std::vector<int32_t> b((size_t)n);
    for (int64_t i = 0; i < n; ++i) b[(size_t)i] = (int32_t)i;
    ggml_backend_tensor_set(p, b.data(), 0, b.size() * sizeof(int32_t));
}

inline void train_set_same_mask(ggml_tensor* mask, const SameConfig& c, int64_t n) {
    if (!mask || !mask->buffer || c.chunk) return;
    std::vector<float> mb = build_swa_bias(c, n);
    ggml_backend_tensor_set(mask, mb.data(), 0, mb.size() * sizeof(float));
}

// One encode graph, kept across calls. Every input except the audio depends only on the frame
// count, so a caller encoding many windows of the same length builds once and then only uploads
// samples. Pre-encode is exactly that: a fixed 128-latent chunk, ~29 chunks a file, up to 5
// passes for the loudness fix.
//
// The alternative -- building and tearing down per call -- allocates a backend buffer each time,
// and Metal keeps freed buffers wired for keep_alive_s (180 s by default), so on a small device
// the run accumulates residency it is not using until the GPU cannot allocate and resets.
struct TrainSameEncoder {
    ggml_context*  ctx   = nullptr;
    ggml_cgraph*   graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor*   in    = nullptr;
    ggml_tensor*   z     = nullptr;
    ggml_tensor*   pos   = nullptr;
    ggml_tensor*   mask  = nullptr;
    ggml_tensor*   pos2  = nullptr;
    // Host copies of the inputs that depend only on the frame count. They are computed once but
    // re-uploaded before every compute: the graph's buffer is shared across runs, so a leaf left
    // from the previous run is not something to rely on.
    std::vector<int32_t> pos_host, pos2_host;
    std::vector<float>   mask_host;
    int frames = 0;          // what the graph was built for; a different length rebuilds
    int n_samples = 0;

    TrainSameEncoder() = default;
    ~TrainSameEncoder() { free(); }
    TrainSameEncoder(const TrainSameEncoder&) = delete;
    TrainSameEncoder& operator=(const TrainSameEncoder&) = delete;

    void free() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
        alloc = nullptr;
        ctx = nullptr;
        graph = nullptr;
        in = z = pos = mask = pos2 = nullptr;
        pos_host.clear();
        pos2_host.clear();
        mask_host.clear();
        frames = n_samples = 0;
    }
};

inline bool train_same_encoder_build(TrainSameEncoder& e, GgufModel& ae, const SameConfig& c,
                                     int T, int n_samples, std::string& err) {
    e.free();
    const int64_t N = (int64_t)T * c.sub_chunk;
    const int64_t N2 = c.chunk ? N + 2 * c.shift : 0;
    const int ch = c.out_channels / c.patch_size;

    ggml_init_params ip = { (size_t)512 * 1024 * 1024, nullptr, true };
    e.ctx = ggml_init(ip);
    if (!e.ctx) { err = "ggml_init failed for SAME encode graph"; return false; }

    e.in = ggml_new_tensor_2d(e.ctx, GGML_TYPE_F32, n_samples, ch);
    e.pos = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I32, N);
    e.mask = c.chunk ? ggml_new_tensor_2d(e.ctx, GGML_TYPE_F32, N, N)
                     : ggml_new_tensor_3d(e.ctx, GGML_TYPE_F32, 3 * c.sub_chunk, c.sub_chunk, N / c.sub_chunk);
    ggml_set_input(e.in);
    ggml_set_input(e.pos);
    ggml_set_input(e.mask);
    ggml_tensor* mask2 = nullptr;
    if (c.chunk) {
        e.pos2 = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I32, N2);
        mask2 = ggml_new_tensor_2d(e.ctx, GGML_TYPE_F32, N2, N2);
        ggml_set_input(e.pos2);
        ggml_set_input(mask2);
    }
    e.z = ggml_cont(e.ctx, same_encode(e.ctx, ae, e.in, c, T, e.pos, e.mask, e.pos2, mask2).z);
    ggml_set_output(e.z);
    e.graph = ggml_new_graph_custom(e.ctx, 32768, false);
    ggml_build_forward_expand(e.graph, e.z);
    e.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ae.backend));
    if (!e.alloc || !ggml_gallocr_alloc_graph(e.alloc, e.graph)) {
        err = "failed to allocate the SAME encode graph (out of device memory)";
        e.free();
        return false;
    }

    // computed once, uploaded per run
    e.pos_host.resize((size_t)N);
    for (int64_t i = 0; i < N; ++i) e.pos_host[(size_t)i] = (int32_t)i;
    if (!c.chunk) e.mask_host = build_swa_bias(c, N);
    if (c.chunk) {
        e.pos2_host.resize((size_t)N2);
        for (int64_t i = 0; i < N2; ++i) e.pos2_host[(size_t)i] = (int32_t)i;
    }

    e.frames = T;
    e.n_samples = n_samples;
    return true;
}

inline bool train_same_encoder_run(TrainSameEncoder& e, GgufModel& ae, const SameConfig& c,
                                   const TrainAudio& audio, TrainLatents& out, std::string& err) {
    const int ch = c.out_channels / c.patch_size;
    const int ds = c.patch_size * c.output_seg;
    if (audio.n_channels != ch) {
        err = "SAME encoder expected " + std::to_string(ch) + " channel audio, got " + std::to_string(audio.n_channels);
        return false;
    }
    if (audio.n_samples <= 0 || audio.n_samples % ds != 0) {
        err = "audio sample count must be a positive multiple of " + std::to_string(ds);
        return false;
    }
    const int T = audio.n_samples / ds;
    if (!e.ctx || e.frames != T || e.n_samples != audio.n_samples) {
        if (!train_same_encoder_build(e, ae, c, T, audio.n_samples, err)) return false;
    }

    ggml_backend_tensor_set(e.in, audio.samples.data(), 0, audio.samples.size() * sizeof(float));
    ggml_backend_tensor_set(e.pos, e.pos_host.data(), 0, e.pos_host.size() * sizeof(int32_t));
    if (!e.mask_host.empty())
        ggml_backend_tensor_set(e.mask, e.mask_host.data(), 0, e.mask_host.size() * sizeof(float));
    if (e.pos2)
        ggml_backend_tensor_set(e.pos2, e.pos2_host.data(), 0, e.pos2_host.size() * sizeof(int32_t));
    if (!graph_compute_checked(ae.backend, e.graph, "SAME encode", err)) return false;

    out.latent = c.latent;
    out.frames = T;
    out.z.resize((size_t)c.latent * T);
    ggml_backend_tensor_get(e.z, out.z.data(), 0, out.z.size() * sizeof(float));
    return true;
}

// One-shot encode for callers with nothing to reuse a graph across.
inline bool encode_train_audio_to_latents(GgufModel& ae, const SameConfig& c, const TrainAudio& audio,
                                          TrainLatents& out, std::string& err) {
    TrainSameEncoder e;
    return train_same_encoder_run(e, ae, c, audio, out, err);
}

} // namespace sa3
