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

inline bool encode_train_audio_to_latents(GgufModel& ae, const SameConfig& c, const TrainAudio& audio,
                                          TrainLatents& out, std::string& err) {
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
    const int64_t N = (int64_t)T * c.sub_chunk;
    const int64_t N2 = c.chunk ? N + 2 * c.shift : 0;

    ggml_init_params ip = { (size_t)512 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        err = "ggml_init failed for SAME encode graph";
        return false;
    }
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, audio.n_samples, ch);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor* mask = c.chunk ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N)
                                : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * c.sub_chunk, c.sub_chunk, N / c.sub_chunk);
    ggml_set_input(in);
    ggml_set_input(pos);
    ggml_set_input(mask);
    ggml_tensor* pos2 = nullptr;
    ggml_tensor* mask2 = nullptr;
    if (c.chunk) {
        pos2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N2);
        mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N2, N2);
        ggml_set_input(pos2);
        ggml_set_input(mask2);
    }
    ggml_tensor* z = ggml_cont(ctx, same_encode(ctx, ae, in, c, T, pos, mask, pos2, mask2).z);
    ggml_set_output(z);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
    ggml_build_forward_expand(gf, z);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ae.backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(in, audio.samples.data(), 0, audio.samples.size() * sizeof(float));
    train_set_positions(pos, N);
    train_set_same_mask(mask, c, N);
    if (c.chunk) train_set_positions(pos2, N2);
    ggml_backend_graph_compute(ae.backend, gf);

    out.latent = c.latent;
    out.frames = T;
    out.z.resize((size_t)c.latent * T);
    ggml_backend_tensor_get(z, out.z.data(), 0, out.z.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

} // namespace sa3
