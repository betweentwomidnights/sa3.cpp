// sat/oobleck.h -- config-driven Oobleck decoder graph for classic Stable Audio.
//
// The convolution/col2im formulation comes from the working acestep.cpp Oobleck port,
// but this version is stateless over sa3::GgufModel and derives every channel/stride from
// ModelSpec.  The converter is responsible for fusing weight normalization and storing
// ConvTranspose1d weights in the documented col2im layout.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "sat/model_spec.h"

#include <string>

namespace sa3::sat {

inline OobleckSpec oobleck_spec_from(const GgufModel& m) {
    if (m.string("sat.architecture") != "stable-audio-tools")
        throw gguf_error("not a stable-audio-tools GGUF");
    if (m.u32("sat.format_version") != 1)
        throw gguf_error("unsupported stable-audio-tools GGUF version");
    const std::vector<int32_t> multipliers = m.i32s("sat.ae.channel_multipliers");
    const std::vector<int32_t> strides = m.i32s("sat.ae.encoder_strides");
    if (multipliers.size() != 5 || strides.size() != 5)
        throw gguf_error("Oobleck GGUF must describe five stages");
    OobleckSpec c;
    c.audio_channels = (int)m.u32("sat.ae.audio_channels");
    c.base_channels = (int)m.u32("sat.ae.base_channels");
    for (size_t i = 0; i < 5; ++i) {
        c.channel_multipliers[i] = multipliers[i];
        c.encoder_strides[i] = strides[i];
    }
    c.encoder_out_channels = (int)m.u32("sat.ae.encoder_out_channels");
    c.latent_channels = (int)m.u32("sat.ae.latent_channels");
    c.use_snake = m.boolean("sat.ae.use_snake");
    c.final_tanh = m.boolean("sat.ae.final_tanh");
    const int declared_ratio = (int)m.u32("sat.ae.downsampling_ratio");
    if (c.downsampling_ratio() != declared_ratio)
        throw gguf_error("Oobleck GGUF stride product disagrees with downsampling ratio");
    return c;
}

// Canonical converted-GGUF tensor contract:
//   ae.decoder.in.{weight,bias}
//   ae.decoder.blocks.I.snake.{alpha_exp,beta_recip}
//   ae.decoder.blocks.I.up.{weight_col2im,bias}
//   ae.decoder.blocks.I.res.R.{snake1,snake2}.{alpha_exp,beta_recip}
//   ae.decoder.blocks.I.res.R.{conv1,conv2}.{weight,bias}
//   ae.decoder.out_snake.{alpha_exp,beta_recip}
//   ae.decoder.out.weight
//
// Regular Conv1d weight: [kernel, in_channels, out_channels].
// ConvTranspose col2im weight: [in_channels, kernel * out_channels].

inline ggml_tensor* oobleck_snake(ggml_context* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* alpha_exp,
                                  ggml_tensor* beta_recip) {
    // Audio tensors are [time, channels], so channel parameters must be [1, C]
    // (a bare [C] would try to broadcast across the time dimension).
    ggml_tensor* a = ggml_reshape_2d(ctx, alpha_exp, 1, alpha_exp->ne[0]);
    ggml_tensor* b = ggml_reshape_2d(ctx, beta_recip, 1, beta_recip->ne[0]);
    ggml_tensor* s = ggml_sin(ctx, ggml_mul(ctx, x, a));
    return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), b));
}

inline ggml_tensor* oobleck_conv1d(ggml_context* ctx,
                                   ggml_tensor* weight,
                                   ggml_tensor* bias,
                                   ggml_tensor* x,
                                   int stride,
                                   int padding,
                                   int dilation) {
    ggml_tensor* y = ggml_conv_1d(ctx, weight, x, stride, padding, dilation);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (bias) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
    return y;
}

inline ggml_tensor* oobleck_conv_transpose1d(ggml_context* ctx,
                                             ggml_tensor* weight_col2im,
                                             ggml_tensor* bias,
                                             ggml_tensor* x,
                                             int stride,
                                             int padding,
                                             int out_channels) {
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* columns = ggml_mul_mat(ctx, weight_col2im, xt);
    ggml_tensor* y = ggml_col2im_1d(ctx, columns, stride, out_channels, padding);
    if (bias) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
    return y;
}

inline ggml_tensor* oobleck_residual_unit(ggml_context* ctx,
                                          const GgufModel& W,
                                          const std::string& prefix,
                                          ggml_tensor* x,
                                          int dilation) {
    ggml_tensor* residual = x;
    x = oobleck_snake(ctx, x,
                      W.get(prefix + "snake1.alpha_exp"),
                      W.get(prefix + "snake1.beta_recip"));
    x = oobleck_conv1d(ctx,
                       W.get(prefix + "conv1.weight"),
                       W.get(prefix + "conv1.bias"),
                       x, 1, 3 * dilation, dilation);
    x = oobleck_snake(ctx, x,
                      W.get(prefix + "snake2.alpha_exp"),
                      W.get(prefix + "snake2.beta_recip"));
    x = oobleck_conv1d(ctx,
                       W.get(prefix + "conv2.weight"),
                       W.get(prefix + "conv2.bias"),
                       x, 1, 0, 1);
    return ggml_add(ctx, residual, x);
}

// latent is time-major [frames, latent_channels]; returns [samples, audio_channels].
inline ggml_tensor* oobleck_decode(ggml_context* ctx,
                                   const GgufModel& W,
                                   ggml_tensor* latent,
                                   const OobleckSpec& c) {
    constexpr int dilations[3] = {1, 3, 9};
    const int stages = (int)c.channel_multipliers.size();
    ggml_tensor* x = oobleck_conv1d(ctx,
                                    W.get("ae.decoder.in.weight"),
                                    W.get("ae.decoder.in.bias"),
                                    latent, 1, 3, 1);
    for (int block = 0; block < stages; ++block) {
        const int source_stage = stages - 1 - block;
        const int stride = c.encoder_strides[(size_t)source_stage];
        const int out_multiplier = source_stage > 0
            ? c.channel_multipliers[(size_t)source_stage - 1]
            : 1;
        const int out_channels = c.base_channels * out_multiplier;
        const int kernel = 2 * stride;
        const std::string p = "ae.decoder.blocks." + std::to_string(block) + ".";
        x = oobleck_snake(ctx, x,
                          W.get(p + "snake.alpha_exp"),
                          W.get(p + "snake.beta_recip"));
        x = oobleck_conv_transpose1d(ctx,
                                     W.get(p + "up.weight_col2im"),
                                     W.get(p + "up.bias"),
                                     x, stride, (kernel - stride) / 2, out_channels);
        for (int r = 0; r < 3; ++r) {
            x = oobleck_residual_unit(ctx, W,
                                      p + "res." + std::to_string(r) + ".",
                                      x, dilations[r]);
        }
    }
    x = oobleck_snake(ctx, x,
                      W.get("ae.decoder.out_snake.alpha_exp"),
                      W.get("ae.decoder.out_snake.beta_recip"));
    x = oobleck_conv1d(ctx, W.get("ae.decoder.out.weight"), nullptr, x, 1, 3, 1);
    return c.final_tanh ? ggml_tanh(ctx, x) : x;
}

} // namespace sa3::sat
