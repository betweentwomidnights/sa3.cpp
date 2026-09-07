// sat/model_spec.h -- weight-topology and inference contracts for classic
// stable-audio-tools models.  This family is intentionally separate from the SA3 model
// structs: it uses Oobleck, T5-base, and the older ContinuousTransformer DiT.
#pragma once

#include <array>
#include <cstdint>
#include <numeric>
#include <string>

namespace sa3::sat {

enum class DiffusionObjective {
    RectifiedFlow,
    RFDenoiser,
    VPrediction,
};

enum class Sampler {
    Auto,
    Euler,
    Dpmpp,
    PingPong,
    Dpmpp2mSde,
    Dpmpp3mSde,
};

struct OobleckSpec {
    int audio_channels = 2;
    int base_channels = 128;
    std::array<int, 5> channel_multipliers = {1, 2, 4, 8, 16};
    std::array<int, 5> encoder_strides = {2, 4, 4, 8, 8};
    int encoder_out_channels = 128; // VAE moments before the bottleneck split
    int latent_channels = 64;
    bool use_snake = true;
    bool final_tanh = false;

    int downsampling_ratio() const {
        return std::accumulate(encoder_strides.begin(), encoder_strides.end(), 1,
                               [](int a, int b) { return a * b; });
    }
};

struct T5Spec {
    int hidden_size = 768;
    int max_length = 64;
    bool sentencepiece_unigram = true;
};

struct ConditionerSpec {
    bool prompt = true;
    bool seconds_start = false;
    bool seconds_total = true;
    int output_dim = 768;
};

struct DitSpec {
    int io_channels = 64;
    int embed_dim = 1024;
    int depth = 16;
    int num_heads = 8;
    int cond_token_dim = 768;
    int global_cond_dim = 768;
    bool qk_layer_norm = true;
};

struct ModelSpec {
    std::string architecture;
    uint32_t format_version = 1;
    int sample_rate = 44100;
    int audio_channels = 2;
    int sample_size = 524288;
    OobleckSpec oobleck;
    T5Spec text_encoder;
    ConditionerSpec conditioner;
    DitSpec dit;
    DiffusionObjective objective = DiffusionObjective::RFDenoiser;
    Sampler default_sampler = Sampler::PingPong;
    int default_steps = 8;
    float default_cfg_scale = 1.0f;
};

inline ModelSpec stable_audio_open_small() {
    ModelSpec spec;
    spec.architecture = "stable-audio-open-small";
    return spec;
}

inline bool validate(const ModelSpec& s, std::string* why = nullptr) {
    auto fail = [&](const char* message) {
        if (why) *why = message;
        return false;
    };
    if (s.architecture.empty()) return fail("architecture is empty");
    if (s.sample_rate <= 0 || s.audio_channels <= 0 || s.sample_size <= 0)
        return fail("invalid audio geometry");
    if (s.oobleck.audio_channels != s.audio_channels)
        return fail("Oobleck audio channels do not match the model");
    if (s.oobleck.downsampling_ratio() <= 0 ||
        s.sample_size % s.oobleck.downsampling_ratio() != 0)
        return fail("sample size is not divisible by the Oobleck downsampling ratio");
    if (s.oobleck.latent_channels != s.dit.io_channels)
        return fail("Oobleck latent channels do not match DiT I/O channels");
    if (!s.conditioner.prompt || !s.conditioner.seconds_total)
        return fail("SAOS requires prompt and seconds_total conditioners");
    if (s.text_encoder.hidden_size != s.conditioner.output_dim ||
        s.dit.cond_token_dim != s.conditioner.output_dim)
        return fail("conditioning dimensions disagree");
    if (s.dit.embed_dim <= 0 || s.dit.depth <= 0 || s.dit.num_heads <= 0 ||
        s.dit.embed_dim % s.dit.num_heads != 0)
        return fail("invalid DiT geometry");
    if (s.default_steps <= 0 || s.default_cfg_scale < 0.0f)
        return fail("invalid inference defaults");
    if (why) why->clear();
    return true;
}

// True means the same converted weight tensors can be loaded.  Inference-only fields such
// as sample_size, objective, sampler, step count and CFG are intentionally excluded: a
// topology-compatible finetune may legitimately change those.  This is stricter than merely
// calling two checkpoints "Stable Audio" and prevents loading an SAO-1.0/Foundation-sized
// DiT into the smaller SAOS graph.
inline bool weight_topology_compatible(const ModelSpec& a,
                                       const ModelSpec& b,
                                       std::string* why = nullptr) {
    auto mismatch = [&](const char* message) {
        if (why) *why = message;
        return false;
    };
    const auto& av = a.oobleck;
    const auto& bv = b.oobleck;
    if (av.audio_channels != bv.audio_channels || av.base_channels != bv.base_channels ||
        av.channel_multipliers != bv.channel_multipliers ||
        av.encoder_strides != bv.encoder_strides ||
        av.encoder_out_channels != bv.encoder_out_channels ||
        av.latent_channels != bv.latent_channels || av.use_snake != bv.use_snake ||
        av.final_tanh != bv.final_tanh)
        return mismatch("Oobleck topology differs");
    if (a.text_encoder.hidden_size != b.text_encoder.hidden_size ||
        a.text_encoder.sentencepiece_unigram != b.text_encoder.sentencepiece_unigram)
        return mismatch("text encoder topology differs");
    if (a.conditioner.prompt != b.conditioner.prompt ||
        a.conditioner.seconds_start != b.conditioner.seconds_start ||
        a.conditioner.seconds_total != b.conditioner.seconds_total ||
        a.conditioner.output_dim != b.conditioner.output_dim)
        return mismatch("conditioner topology differs");
    const auto& ad = a.dit;
    const auto& bd = b.dit;
    if (ad.io_channels != bd.io_channels || ad.embed_dim != bd.embed_dim ||
        ad.depth != bd.depth || ad.num_heads != bd.num_heads ||
        ad.cond_token_dim != bd.cond_token_dim ||
        ad.global_cond_dim != bd.global_cond_dim ||
        ad.qk_layer_norm != bd.qk_layer_norm)
        return mismatch("DiT topology differs");
    if (why) why->clear();
    return true;
}

} // namespace sa3::sat
