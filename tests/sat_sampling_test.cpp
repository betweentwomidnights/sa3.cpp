#include "sampling.h"
#include "sat/dit.h"
#include "sat/model_spec.h"
#include "sat/oobleck.h"
#include "sat/t5.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

static bool near(float a, float b, float eps = 1.0e-6f) {
    return std::fabs(a - b) <= eps;
}

static int test_t5_graph_shape() {
    int fails = 0;
    sa3::sat::T5EncoderConfig c;
    c.dim = 8;
    c.layers = 2;
    c.heads = 2;
    c.head_dim = 4;
    c.intermediate = 16;
    c.vocab = 32;
    c.relative_buckets = 8;
    c.relative_max_distance = 16;

    sa3::GgufModel weights;
    ggml_init_params wp = {ggml_tensor_overhead() * 64, nullptr, true};
    weights.ctx = ggml_init(wp);
    auto w1 = [&](const std::string& name, int64_t n0) {
        ggml_tensor* t = ggml_new_tensor_1d(weights.ctx, GGML_TYPE_F32, n0);
        ggml_set_name(t, name.c_str());
        weights.tensors[name] = t;
    };
    auto w2 = [&](const std::string& name, int64_t n0, int64_t n1) {
        ggml_tensor* t = ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name.c_str());
        weights.tensors[name] = t;
    };
    w2("te.embed.weight", c.dim, c.vocab);
    w2("te.relative_attention_bias.weight", c.heads, c.relative_buckets);
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "te." + std::to_string(l) + ".";
        w1(p + "attn_norm.weight", c.dim);
        w2(p + "q.weight", c.dim, c.heads * c.head_dim);
        w2(p + "k.weight", c.dim, c.heads * c.head_dim);
        w2(p + "v.weight", c.dim, c.heads * c.head_dim);
        w2(p + "o.weight", c.heads * c.head_dim, c.dim);
        w1(p + "ffn_norm.weight", c.dim);
        w2(p + "wi.weight", c.dim, c.intermediate);
        w2(p + "wo.weight", c.intermediate, c.dim);
    }
    w1("te.norm.weight", c.dim);

    constexpr int seq = 4;
    ggml_init_params gp = {ggml_tensor_overhead() * 512, nullptr, true};
    ggml_context* ctx = ggml_init(gp);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_tensor* rel = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq * seq);
    ggml_tensor* out = sa3::sat::t5_encode(ctx, weights, ids, mask, rel, c);
    fails += expect(out->ne[0] == c.dim && out->ne[1] == seq,
                    "classic T5 graph output shape");
    ggml_free(ctx);
    return fails;
}

static int test_oobleck_decoder_graph_shape() {
    int fails = 0;
    sa3::sat::OobleckSpec c;
    c.base_channels = 2;
    c.channel_multipliers = {1, 2, 4, 8, 16};
    c.encoder_strides = {2, 2, 2, 2, 2};
    c.encoder_out_channels = 4;
    c.latent_channels = 2;

    sa3::GgufModel weights;
    ggml_init_params wp = {ggml_tensor_overhead() * 512, nullptr, true};
    weights.ctx = ggml_init(wp);
    auto w1 = [&](const std::string& name, int64_t n0) {
        ggml_tensor* t = ggml_new_tensor_1d(weights.ctx, GGML_TYPE_F32, n0);
        ggml_set_name(t, name.c_str());
        weights.tensors[name] = t;
    };
    auto w2 = [&](const std::string& name, int64_t n0, int64_t n1) {
        ggml_tensor* t = ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name.c_str());
        weights.tensors[name] = t;
    };
    auto w3 = [&](const std::string& name, int64_t n0, int64_t n1, int64_t n2) {
        ggml_tensor* t = ggml_new_tensor_3d(weights.ctx, GGML_TYPE_F32, n0, n1, n2);
        ggml_set_name(t, name.c_str());
        weights.tensors[name] = t;
    };
    auto snake = [&](const std::string& p, int channels) {
        w1(p + "alpha_exp", channels);
        w1(p + "beta_recip", channels);
    };

    const int stages = (int)c.channel_multipliers.size();
    int in_channels = c.base_channels * c.channel_multipliers.back();
    w3("ae.decoder.in.weight", 7, c.latent_channels, in_channels);
    w1("ae.decoder.in.bias", in_channels);
    for (int block = 0; block < stages; ++block) {
        const int source_stage = stages - 1 - block;
        const int stride = c.encoder_strides[(size_t)source_stage];
        const int out_multiplier = source_stage > 0
            ? c.channel_multipliers[(size_t)source_stage - 1]
            : 1;
        const int out_channels = c.base_channels * out_multiplier;
        const std::string p = "ae.decoder.blocks." + std::to_string(block) + ".";
        snake(p + "snake.", in_channels);
        w2(p + "up.weight_col2im", in_channels, 2 * stride * out_channels);
        w1(p + "up.bias", out_channels);
        for (int r = 0; r < 3; ++r) {
            const std::string rp = p + "res." + std::to_string(r) + ".";
            snake(rp + "snake1.", out_channels);
            w3(rp + "conv1.weight", 7, out_channels, out_channels);
            w1(rp + "conv1.bias", out_channels);
            snake(rp + "snake2.", out_channels);
            w3(rp + "conv2.weight", 1, out_channels, out_channels);
            w1(rp + "conv2.bias", out_channels);
        }
        in_channels = out_channels;
    }
    snake("ae.decoder.out_snake.", c.base_channels);
    w3("ae.decoder.out.weight", 7, c.base_channels, c.audio_channels);

    constexpr int frames = 3;
    ggml_init_params gp = {ggml_tensor_overhead() * 2048, nullptr, true};
    ggml_context* ctx = ggml_init(gp);
    ggml_tensor* latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, frames, c.latent_channels);
    ggml_tensor* audio = sa3::sat::oobleck_decode(ctx, weights, latent, c);
    fails += expect(audio->ne[0] == frames * c.downsampling_ratio() &&
                    audio->ne[1] == c.audio_channels,
                    "config-driven Oobleck decoder output shape");
    ggml_free(ctx);
    return fails;
}

static int test_classic_dit_graph_shape() {
    int fails = 0;
    sa3::sat::DitSpec c;
    c.io_channels = 4; c.embed_dim = 64; c.depth = 1; c.num_heads = 1;
    c.cond_token_dim = 8; c.global_cond_dim = 8;
    sa3::GgufModel weights;
    weights.gguf = gguf_init_empty();
    gguf_set_val_u32(weights.gguf, "sat.dit.rotary_dims", 32);
    ggml_init_params wp = {ggml_tensor_overhead() * 128, nullptr, true};
    weights.ctx = ggml_init(wp);
    auto w1 = [&](const std::string& name, int64_t n0) {
        ggml_tensor* t = ggml_new_tensor_1d(weights.ctx, GGML_TYPE_F32, n0);
        weights.tensors[name] = t;
    };
    auto w2 = [&](const std::string& name, int64_t n0, int64_t n1) {
        ggml_tensor* t = ggml_new_tensor_2d(weights.ctx, GGML_TYPE_F32, n0, n1);
        weights.tensors[name] = t;
    };
    w2("dit.pre.weight", 4, 4); w2("dit.post.weight", 4, 4);
    w2("dit.time_fourier.weight", 1, 128);
    w2("dit.time.0.weight", 256, 64); w1("dit.time.0.bias", 64);
    w2("dit.time.2.weight", 64, 64); w1("dit.time.2.bias", 64);
    w2("dit.cond.0.weight", 8, 64); w2("dit.cond.2.weight", 64, 64);
    w2("dit.global.0.weight", 8, 64); w2("dit.global.2.weight", 64, 64);
    w2("dit.in.weight", 4, 64); w2("dit.out.weight", 64, 4);
    const std::string p = "dit.blocks.0.";
    for (const char* norm : {"pre_norm", "cross_norm", "ff_norm"}) {
        w1(p + norm + ".gamma", 64); w1(p + norm + ".beta", 64);
    }
    w2(p + "self.qkv.weight", 64, 192); w2(p + "self.out.weight", 64, 64);
    w2(p + "cross.q.weight", 64, 64); w2(p + "cross.kv.weight", 64, 128);
    w2(p + "cross.out.weight", 64, 64);
    for (const char* attn : {"self", "cross"}) for (const char* qk : {"q", "k"}) {
        w1(p + attn + "." + qk + "_norm.weight", 64);
        w1(p + attn + "." + qk + "_norm.bias", 64);
    }
    w2(p + "ff.in.weight", 64, 512); w1(p + "ff.in.bias", 512);
    w2(p + "ff.out.weight", 256, 64); w1(p + "ff.out.bias", 64);

    constexpr int frames = 3, cond_tokens = 5;
    ggml_init_params gp = {ggml_tensor_overhead() * 2048, nullptr, true};
    ggml_context* ctx = ggml_init(gp);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, frames);
    ggml_tensor* time = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor* cross = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, cond_tokens);
    ggml_tensor* global = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, frames + 1);
    ggml_tensor* out = sa3::sat::classic_dit_forward(ctx, weights, x, time, cross, global, pos, c);
    fails += expect(out->ne[0] == c.io_channels && out->ne[1] == frames,
                    "classic DiT graph output shape");
    ggml_free(ctx);
    return fails;
}

static int test_oobleck_metadata() {
    int fails = 0;
    sa3::GgufModel metadata;
    metadata.gguf = gguf_init_empty();
    gguf_set_val_str(metadata.gguf, "sat.architecture", "stable-audio-tools");
    gguf_set_val_u32(metadata.gguf, "sat.format_version", 1);
    gguf_set_val_u32(metadata.gguf, "sat.ae.audio_channels", 2);
    gguf_set_val_u32(metadata.gguf, "sat.ae.base_channels", 128);
    const int32_t multipliers[] = {1, 2, 4, 8, 16};
    const int32_t strides[] = {2, 4, 4, 8, 8};
    gguf_set_arr_data(metadata.gguf, "sat.ae.channel_multipliers",
                      GGUF_TYPE_INT32, multipliers, 5);
    gguf_set_arr_data(metadata.gguf, "sat.ae.encoder_strides",
                      GGUF_TYPE_INT32, strides, 5);
    gguf_set_val_u32(metadata.gguf, "sat.ae.encoder_out_channels", 128);
    gguf_set_val_u32(metadata.gguf, "sat.ae.latent_channels", 64);
    gguf_set_val_u32(metadata.gguf, "sat.ae.downsampling_ratio", 2048);
    gguf_set_val_bool(metadata.gguf, "sat.ae.use_snake", true);
    gguf_set_val_bool(metadata.gguf, "sat.ae.final_tanh", false);
    const sa3::sat::OobleckSpec parsed = sa3::sat::oobleck_spec_from(metadata);
    fails += expect(parsed.audio_channels == 2 && parsed.base_channels == 128,
                    "Oobleck GGUF scalar metadata");
    fails += expect(parsed.channel_multipliers[4] == 16 &&
                    parsed.encoder_strides[4] == 8 &&
                    parsed.downsampling_ratio() == 2048,
                    "Oobleck GGUF array metadata");
    fails += expect(parsed.latent_channels == 64 && parsed.use_snake && !parsed.final_tanh,
                    "Oobleck GGUF activation metadata");
    return fails;
}

int main() {
    int fails = 0;

    const sa3::sat::ModelSpec saos = sa3::sat::stable_audio_open_small();
    std::string why;
    fails += expect(sa3::sat::validate(saos, &why), why.c_str());
    fails += expect(saos.oobleck.downsampling_ratio() == 2048, "SAOS Oobleck ratio");
    fails += expect(saos.sample_size / saos.oobleck.downsampling_ratio() == 256,
                    "SAOS latent frame count");
    fails += expect(saos.dit.embed_dim == 1024 && saos.dit.depth == 16,
                    "SAOS DiT topology");

    // A finetune may change inference defaults without changing any loadable tensor shape.
    sa3::sat::ModelSpec finetune = saos;
    finetune.sample_size = 262144;
    finetune.objective = sa3::sat::DiffusionObjective::RectifiedFlow;
    finetune.default_steps = 50;
    finetune.default_cfg_scale = 4.0f;
    fails += expect(sa3::sat::weight_topology_compatible(saos, finetune, &why),
                    "inference-only finetune remains weight compatible");

    // SAO 1.0 / Foundation use the larger classic DiT.  Even with an identical VAE this
    // must not enter the SAOS weight loader.
    sa3::sat::ModelSpec larger = saos;
    larger.dit.embed_dim = 1536;
    larger.dit.depth = 24;
    larger.dit.num_heads = 24;
    larger.dit.qk_layer_norm = false;
    fails += expect(!sa3::sat::weight_topology_compatible(saos, larger, &why),
                    "larger classic DiT rejected by SAOS topology check");
    fails += expect(why == "DiT topology differs", "topology rejection is diagnostic");

    fails += expect(sa3::sat::t5_relative_position_bucket(0) == 0,
                    "T5 relative bucket zero");
    fails += expect(sa3::sat::t5_relative_position_bucket(-1) == 1,
                    "T5 past-key bucket");
    fails += expect(sa3::sat::t5_relative_position_bucket(1) == 17,
                    "T5 future-key bucket");
    fails += expect(sa3::sat::t5_relative_position_bucket(-10000) == 15 &&
                    sa3::sat::t5_relative_position_bucket(10000) == 31,
                    "T5 distant buckets saturate");
    const std::vector<int32_t> buckets = sa3::sat::t5_relative_position_buckets(3);
    fails += expect(buckets.size() == 9 && buckets[1] == 17 && buckets[3] == 1,
                    "T5 bucket matrix uses key-query orientation");
    fails += test_t5_graph_shape();
    fails += test_classic_dit_graph_shape();
    fails += test_oobleck_decoder_graph_shape();
    fails += test_oobleck_metadata();

    // Cropping a longer planar decode must keep the original plane stride. The
    // historical bug used crop_samples here, shifting the right channel earlier.
    const float planar[] = {0.1f, 0.2f, 0.3f, 0.4f, -0.1f, -0.2f, -0.3f, -0.4f};
    const std::vector<int16_t> inter = sa3::wav_detail::interleave_planar_i16(
        planar, /*n_samples=*/2, /*n_ch=*/2, /*channel_stride=*/4);
    fails += expect(inter.size() == 4 && inter[0] > 0 && inter[1] < 0 &&
                    inter[2] > inter[0] && inter[3] < inter[1],
                    "cropped planar WAV preserves source channel stride");

    const std::vector<float> t = sa3::sampling::make_rf_logsnr_schedule(8);
    fails += expect(t.size() == 9, "8-step schedule has 9 endpoints");
    fails += expect(t.front() == 1.0f && t.back() == 0.0f, "schedule endpoints are exact");
    for (size_t i = 1; i < t.size(); ++i)
        fails += expect(t[i] <= t[i - 1], "schedule is monotone");
    // Reference: sigmoid(-linspace(-6, 2, 9))[1] = sigmoid(5).
    fails += expect(near(t[1], 1.0f / (1.0f + std::exp(-5.0f))),
                    "SAOS logSNR schedule matches stable-audio-tools");

    std::vector<float> x = {2.0f, -1.0f};
    const float velocity[] = {0.5f, -2.0f};
    const float noise[] = {-0.25f, 0.75f};
    sa3::sampling::rf_pingpong_step(x.data(), velocity, noise, x.size(), 0.8f, 0.25f);
    fails += expect(near(x[0], 1.1375f) && near(x[1], 0.6375f),
                    "ping-pong update matches reference algebra");

    bool threw = false;
    try { (void)sa3::sampling::make_rf_logsnr_schedule(0); }
    catch (const std::invalid_argument&) { threw = true; }
    fails += expect(threw, "invalid schedule rejected");

    if (fails) return 1;
    std::printf("sat_sampling_test: ok\n");
    return 0;
}
