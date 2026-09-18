#include "sampling.h"
#include "sat/dit.h"
#include "sat/foundation_prompt.h"
#include "sat/foundation_timing.h"
#include "sat/keybed.h"
#include "sat/model_spec.h"
#include "sat/model_paths.h"
#include "sat/oobleck.h"
#include "sat/pipeline.h"
#include "sat/profiles.h"
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

static int test_shared_loudness() {
    int fails = 0;
    sa3::sat::GenerateParams defaults;
    fails += expect(defaults.loudness.peak_normalize_enabled && defaults.loudness.limiter_enabled,
                    "SAT generation shares the SA3 loudness defaults");

    std::vector<float> audio{0.25f, -0.5f, 0.125f};
    sa3::LoudnessMeta meta = sa3::make_loudness_meta(defaults.loudness);
    sa3::apply_audio_loudness(audio, defaults.loudness, meta);
    fails += expect(near(meta.decoded_peak, 0.5f) && meta.peak_normalize_gain_set,
                    "shared loudness records decoded peak and normalization gain");
    fails += expect(meta.limiter_limited_fraction_set &&
                    meta.final_peak <= sa3::db_to_linear(defaults.loudness.limiter_ceiling_db) + 1.0e-6f,
                    "shared limiter keeps SAT audio under its ceiling");

    sa3::LoudnessParams raw;
    raw.peak_normalize_enabled = false;
    raw.limiter_enabled = false;
    std::vector<float> raw_audio{2.0f, -0.5f};
    sa3::LoudnessMeta raw_meta = sa3::make_loudness_meta(raw);
    sa3::apply_audio_loudness(raw_audio, raw, raw_meta);
    fails += expect(near(raw_audio[0], 2.0f) && near(raw_meta.final_peak, 2.0f) &&
                    !raw_meta.peak_normalize_gain_set && !raw_meta.limiter_limited_fraction_set,
                    "disabling both stages preserves exact raw SAT audio");
    return fails;
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

static int test_keybed_prompts() {
    namespace kb = sa3::sat::keybed;
    int fails = 0;
    // Expected strings captured from RC-stable-audio-tools 43dcbb4b keybed_prompts.py.
    fails += expect(kb::build_sequence_prompt("Grand Piano, Warm, Gritty", {60, 61, 62, 63, 64, 65}, false) ==
                    "Keybed, Sequence, Timbre Profile, Grand Piano, Warm, Gritty, Dry, Chromatic Chunk, Note Sequence, C4, C#4, D4, D#4, E4, F4",
                    "keybed documented Grand Piano sequence prompt");
    fails += expect(kb::build_sequence_prompt("Synth Lead, Bright, Medium Reverb, Ping Pong Delay", {58, 59}, true) ==
                    "Keybed, Sequence, Timbre Profile, Synth Lead, Bright, Wet, Medium Reverb, Ping Pong Delay, Chromatic Chunk, Note Sequence, A#3, B3",
                    "keybed wet prompt keeps descriptor FX after Wet");
    fails += expect(kb::build_sequence_prompt("Synth Lead, Bright, Medium Reverb", {36, 37}, false) ==
                    "Keybed, Sequence, Timbre Profile, Synth Lead, Bright, Dry, Chromatic Chunk, Note Sequence, C2, C#2",
                    "keybed dry prompt drops FX");
    fails += expect(kb::build_sequence_prompt("Pad, Warm", {104}, true) ==
                    "Keybed, Sequence, Timbre Profile, Pad, Warm, Wet, Chromatic Chunk, Note Sequence, G#7",
                    "keybed wet prompt without FX is just Wet");
    const std::vector<std::string> fx{"High Delay"};
    fails += expect(kb::build_sequence_prompt("Pad, Warm", {kb::note_name_to_midi("Bb3").value_or(0),
                                                            kb::note_name_to_midi("Db4").value_or(0)}, true, &fx) ==
                    "Keybed, Sequence, Timbre Profile, Pad, Warm, Wet, High Delay, Chromatic Chunk, Note Sequence, A#3, C#4",
                    "keybed explicit FX and flat note names");
    fails += expect(kb::build_sequence_prompt("Keybed, Sequence, Timbre Profile, Rhodes Piano, Warm, Warm, Wet, Low Reverb, Chromatic Chunk, Note Sequence, C4, C#4", {28, 29}, true) ==
                    "Keybed, Sequence, Timbre Profile, Rhodes Piano, Warm, Wet, Low Reverb, Chromatic Chunk, Note Sequence, E1, F1",
                    "keybed pasted sequence prompt is cleaned and rebuilt");
    fails += expect(kb::build_sequence_prompt("Keybed, Timbre Profile, Marimba, Target Note, C4, keybed_pos_049, Medium Register, Dry, Bright", {72, 72}, false) ==
                    "Keybed, Sequence, Timbre Profile, Marimba, Bright, Dry, Chromatic Chunk, Note Sequence, C5, C5",
                    "keybed strips target, position, and register grammar without de-duplicating notes");
    fails += expect(kb::build_single_note_prompt("Cello, Rich, Plate Reverb", 54, true) ==
                    "Keybed, Timbre Profile, Cello, Rich, Wet, Plate Reverb, Target Note, F#3",
                    "keybed single-note prompt");
    const kb::DescriptorTokens split = kb::split_descriptor_tokens(
        "Keybed, Target Position, keybed_pos_010, Bass, 808, Dry, High Distortion, b3, Top Register");
    fails += expect(split.body == std::vector<std::string>{"Bass", "808"} &&
                    split.fx == std::vector<std::string>{"High Distortion"},
                    "keybed descriptor split separates body and FX");

    const auto midi = [](const char* name) { return kb::note_name_to_midi(name).value_or(-999); };
    fails += expect(midi("C4") == 60 && midi("c#4") == 61 && midi("Db4") == 61 &&
                    midi("bb3") == 58 && midi("B3") == 59 && midi("C-1") == 0 && midi("G#7") == 104,
                    "keybed note names follow C4 = 60");
    fails += expect(!kb::note_name_to_midi("E#3") && !kb::note_name_to_midi("Cb4") &&
                    !kb::note_name_to_midi("H2") && !kb::note_name_to_midi("C"),
                    "keybed rejects note names outside RC's table");
    fails += expect(kb::midi_to_note_name(61) == "C#4" && kb::midi_to_note_name(0) == "C-1" &&
                    kb::note_filename(49) == "Csharp3",
                    "keybed MIDI note names use sharps");
    fails += expect(kb::label_to_sounding_midi(60) == 48,
                    "keybed labels sound one octave below their names");

    const std::vector<kb::Chunk> full = kb::plan_full_range(kb::FullRange::C2ToB5);
    fails += expect(full.size() == 8 && full.front().label_midis.front() == 36 &&
                    full.back().label_midis.back() == 83 && full.front().seconds_total == 20 &&
                    near((float)full.front().actual_seconds, 19.25f),
                    "keybed compact range is 8 six-note 20 s chunks");
    fails += expect(kb::plan_full_range(kb::FullRange::C2ToF6).size() == 9 &&
                    kb::plan_full_range(kb::FullRange::C2ToB6).size() == 10,
                    "keybed extended and wide ranges");
    fails += expect(kb::chunk_labels({60, 61, 62, 63, 64, 65, 66}).size() == 1,
                    "keybed drops a trailing one-note chunk");
    const std::vector<kb::Chunk> tail = kb::chunk_labels({60, 61, 62, 63, 64, 65, 66, 67});
    fails += expect(tail.size() == 2 && tail[1].seconds_total == 7 &&
                    near((float)tail[1].actual_seconds, 6.25f),
                    "keybed keeps a trailing two-note chunk with its own duration");
    fails += expect(kb::clamp_preview_root(24, 6) == 36 && kb::clamp_preview_root(100, 6) == 84 &&
                    kb::clamp_preview_root(90, 24) == 81 && kb::plan_preview(60, 24).size() == 4,
                    "keybed preview roots clamp to C2-C6 and G#7");
    return fails;
}

static int test_keybed_audio() {
    namespace kb = sa3::sat::keybed;
    int fails = 0;
    // Reference values from RC's _apply_short_fade, _note_slices_for_sequence,
    // trim_tail_conservative, and 120 ms terminal fade on the same synthetic chunk.
    const int sr = 44100;
    const double pi = 3.14159265358979323846;
    const int64_t decoded = (int64_t)std::lround(kb::sequence_actual_seconds(2) * sr) + 2048;
    std::vector<float> planar((size_t)(decoded * 2), 0.0f);
    const double freq[] = {220.0, 330.0}, decay[] = {12.0, 0.0}, amp[] = {0.9, 1.3};
    for (int64_t k = 0; k < decoded; ++k) {
        const double t = (double)k / sr;
        for (int i = 0; i < 2; ++i) {
            const double start = i * 3.25;
            if (t < start || t >= start + 3.0) continue;
            const double lt = t - start;
            const float v = (float)(amp[i] * std::exp(-decay[i] * lt) * std::sin(2.0 * pi * freq[i] * lt));
            planar[(size_t)k] = v;
            planar[(size_t)(decoded + k)] = 0.5f * v;
        }
    }
    const std::vector<float> chunk = kb::finish_chunk_audio(planar.data(), 2, decoded, sr,
                                                            kb::sequence_actual_seconds(2));
    const int64_t frames = (int64_t)chunk.size() / 2;
    double abs_sum = 0.0;
    for (float v : chunk) abs_sum += std::fabs(v);
    fails += expect(frames == 275625 && std::fabs(abs_sum - 155465.704661) < 0.05 &&
                    near(chunk[100], 0.001772515f, 1e-6f) &&
                    near(chunk[(size_t)frames - 100], 0.28125f, 1e-6f),
                    "keybed chunk crop, clamp, and fades match RC");

    const std::vector<kb::NoteSlice> slices = kb::note_slices({57, 64}, sr, frames);
    fails += expect(slices.size() == 2 && slices[0].start_frame == 0 &&
                    slices[0].end_frame == 132300 && slices[1].start_frame == 143325 &&
                    slices[1].end_frame == 275625 && slices[0].sounding_midi == 45,
                    "keybed slices sit on the 3.25 s grid");

    struct Expected { int64_t frames; double abs_sum; float right_500; };
    const Expected expected[] = {
        {28665, 3009.790682, 0.013986669f},
        {132300, 149598.730714, -0.649072468f},
    };
    for (int i = 0; i < 2; ++i) {
        const std::vector<float> note = kb::extract_note_sample(chunk, 2, sr, slices[(size_t)i]);
        double sum = 0.0;
        for (float v : note) sum += std::fabs(v);
        const int64_t n = (int64_t)note.size() / 2;
        fails += expect(n == expected[i].frames && std::fabs(sum - expected[i].abs_sum) < 0.05 &&
                        near(note[(size_t)(n + 500)], expected[i].right_500, 1e-6f) &&
                        std::fabs(note[(size_t)(n - 2)]) < 1e-6f,
                        i == 0 ? "keybed decaying note is tail-trimmed like RC"
                               : "keybed sustained note is not trimmed");
    }

    const std::string sfz = kb::sfz_text({{"E3.wav", 52}, {"A2.wav", 45}});
    fails += expect(sfz.find("ampeg_attack=0.0050\nampeg_decay=0.0000\nampeg_sustain=100\n"
                             "ampeg_release=0.2500\n") != std::string::npos &&
                    sfz.find("<region> sample=A2.wav lokey=45 hikey=45 pitch_keycenter=45\n"
                             "<region> sample=E3.wav") != std::string::npos,
                    "keybed SFZ is sorted with RC's envelope header");

    // Wet FX chains: RC picks a reverb-or-delay primary and a second category about a quarter
    // of the time, never a second reverb.
    {
        int fails_local = 0, two = 0;
        const std::vector<std::string>& all = kb::vocab::fx_choices();
        for (uint64_t seed = 0; seed < 400; ++seed)
        {
            const std::vector<std::string> chain = kb::random_fx_chain(seed);
            if (chain.empty() || chain.size() > 2) { ++fails_local; continue; }
            two += chain.size() == 2;
            for (const std::string& tag : chain)
                if (std::find(all.begin(), all.end(), tag) == all.end() || !kb::is_fx_token(tag))
                    ++fails_local;
            const bool primary_is_space = chain[0].find("Reverb") != std::string::npos ||
                                          chain[0].find("Delay") != std::string::npos;
            if (!primary_is_space) ++fails_local;
            if (chain.size() == 2 && chain[0].find("Reverb") != std::string::npos &&
                chain[1].find("Reverb") != std::string::npos)
                ++fails_local;
        }
        fails += expect(fails_local == 0 && two > 60 && two < 160,
                        "keybed wet FX chains follow RC's category weights");
        fails += expect(kb::random_fx_chain(7) == kb::random_fx_chain(7) &&
                        kb::random_fx_chain(7, false).size() == 1,
                        "keybed FX chains are seed-stable and can be limited to one tag");
        fails += expect(all.size() == 22 && all.front() == "Low Reverb" && all.back() == "High Bitcrush",
                        "keybed FX choices flatten every category");
    }

    // Structured sounds: free text sorts onto controls, unknown words survive as extras.
    {
        const kb::SoundSpec s = kb::classify_descriptor(
            "keys, rhodes piano, Warm, soft, Staccato, Sine, tape wobble, Plate Reverb, Ping Pong Delay");
        fails += expect(s.family == "Keys" && s.subfamily == "Rhodes Piano" &&
                        s.character == std::vector<std::string>{"Warm", "Soft"} &&
                        s.articulation == "Staccato" && s.oscillator == "Sine" &&
                        s.extras == std::vector<std::string>{"tape wobble"} && s.wet &&
                        s.fx == std::vector<std::string>{"Plate Reverb", "Ping Pong Delay"},
                        "keybed classifier sorts a descriptor onto its controls");
        fails += expect(kb::descriptor_of(s) == "Keys, Rhodes Piano, Warm, Soft, Staccato, Sine, tape wobble",
                        "keybed descriptor rebuilds in RC's order");
        const kb::SoundSpec pasted = kb::classify_descriptor(
            "Keybed, Sequence, Timbre Profile, Marimba, Bright, Dry, Chromatic Chunk, Note Sequence, C5, C#5");
        fails += expect(pasted.family == "Mallet" && pasted.subfamily == "Marimba" && !pasted.wet &&
                        pasted.character == std::vector<std::string>{"Bright"} && pasted.extras.empty(),
                        "keybed classifier infers the family and strips pasted grammar");
        kb::SoundSpec fx = s;
        kb::set_fx(fx, "High Reverb");
        fails += expect(fx.fx == std::vector<std::string>{"High Reverb", "Ping Pong Delay"},
                        "keybed FX slots hold one tag per category");
        int mismatches = 0;
        for (uint64_t seed = 0; seed < 200; ++seed) {
            const kb::SoundSpec r = kb::random_sound(seed, seed % 2 == 0);
            const kb::SoundSpec again = kb::classify_descriptor(kb::descriptor_of(r));
            if (again.family != r.family || again.subfamily != r.subfamily ||
                again.character != r.character || again.articulation != r.articulation ||
                again.oscillator != r.oscillator || !again.extras.empty() ||
                r.wet != !r.fx.empty() || kb::sequence_prompt_of(r, {60, 61}).empty())
                ++mismatches;
        }
        fails += expect(mismatches == 0, "keybed random sounds round-trip through the classifier");
    }

    // Layered keybeds: SHA-1 and RC's seed derivation, pinned to Python hashlib outputs.
    fails += expect(kb::detail::sha1_hex("") == "da39a3ee5e6b4b0d3255bfef95601890afd80709" &&
                    kb::detail::sha1_hex("abc") == "a9993e364706816aba3e25717850c26c9cd0d89d" &&
                    kb::detail::sha1_hex(std::string(1000, 'a')) == "291e9a6c66994949b57ba5e650361e98fc36b1ba",
                    "keybed SHA-1 matches the FIPS vectors");
    fails += expect(kb::layer_seed(42, 0) == 8162322u && kb::layer_seed(42, 1) == 1324442250u &&
                    kb::layer_seed(42, 2) == 2901690929u && kb::layer_seed(0, 0) == 4024071094u &&
                    kb::layer_seed(2147483647, 2) == 3749525202u,
                    "keybed layer seeds match RC's _layer_generation_seeds");
    fails += expect(kb::layer_prompt_seed(42, 0) == 2848088190u && kb::layer_prompt_seed(42, 2) == 1775213216u &&
                    kb::layer_prompt_seed(123456789, 1) == 846938297u,
                    "keybed layer prompt seeds match RC's _prompt_seed_for_layer");

    const kb::RandomDescriptor a = kb::random_descriptor(42), b = kb::random_descriptor(42);
    fails += expect(a.descriptor == b.descriptor && !a.family.empty() &&
                    a.descriptor.rfind(a.family, 0) == 0 &&
                    kb::split_descriptor_tokens(a.descriptor).fx.empty(),
                    "keybed random descriptors are seed-stable and FX-free");
    return fails;
}

int main() {
    int fails = 0;

    fails += test_shared_loudness();
    fails += test_keybed_prompts();
    fails += test_keybed_audio();
    fails += expect(std::string(sa3::sat::kDefaultSatEncoding) == "F16",
                    "SAT catalogs default to the reference F16 bundles");
    fails += expect(sa3::sat::saos_dit_relative_path("arc", "q5_k_m") ==
                    "stable-audio-open-small-dit-0.3B-v1.0-Q5_K_M.gguf",
                    "canonical SAOS ARC filename");
    fails += expect(sa3::sat::saos_dit_relative_path("kickbass", "q4_k_m") ==
                    "finetunes/kickbass/kickbass-v1-e257-dit-0.3B-v1.0-Q4_K_M.gguf",
                    "canonical nested KickBass filename");
    fails += expect(sa3::sat::saos_dit_relative_path("jerry_grunge", "F16") ==
                    "finetunes/jerry-grunge/jerry-grunge-bs64-step3000-dit-0.3B-v1.0-F16.gguf",
                    "SAOS variant aliases normalize");
    fails += expect(sa3::sat::saos_t5_relative_path("q8_0") ==
                    "t5-base-encoder-0.1B-v1.0-Q8_0.gguf" &&
                    sa3::sat::saos_oobleck_relative_path("q8_0") ==
                    "stable-audio-open-small-oobleck-v1.0-Q8_0.gguf",
                    "canonical shared SAOS component filenames");
    fails += expect(sa3::sat::sat_large_dit_relative_path("sao1", "q5_k_m") ==
                        "stable-audio-open-1.0-dit-1.1B-v1.0-Q5_K_M.gguf" &&
                    sa3::sat::sat_large_dit_relative_path("foundation", "q4_k_m") ==
                        "foundation-1-dit-1.1B-v1.0-Q4_K_M.gguf",
                    "canonical large SAT DiT filenames");
    fails += expect(sa3::sat::sat_large_dit_relative_path("keybeds", "F16") ==
                        "foundation-1.2-keybeds-dit-1.1B-v1.0-F16.gguf" &&
                    sa3::sat::is_sat_large_model("foundation-1.2-samples") &&
                    sa3::sat::is_foundation_keybeds_model("Foundation_1.2-Keybeds") &&
                    !sa3::sat::is_sat_large_model("medium") &&
                    !sa3::sat::is_sat_large_model("arc"),
                    "Foundation-1.2 variants route as large SAT models");
    {
        const sa3::sat::ModelSpec keybeds = sa3::sat::foundation_1_2_keybeds();
        std::string why;
        fails += expect(sa3::sat::validate(keybeds, &why) &&
                        sa3::sat::weight_topology_compatible(sa3::sat::foundation_1(), keybeds, &why) &&
                        keybeds.default_steps == 80 && keybeds.default_cfg_scale == 6.0f,
                        "Foundation-1.2 Keybeds shares Foundation-1 topology");
    }
    fails += expect(sa3::sat::sat_t5_128_relative_path("F16") ==
                        "t5-base-encoder-128tok-0.1B-v1.0-F16.gguf" &&
                    sa3::sat::sat_oobleck_relative_path("Q8_0") ==
                        "stable-audio-open-oobleck-v1.0-Q8_0.gguf",
                    "canonical shared large SAT component filenames");

    const sa3::sat::ModelSpec saos = sa3::sat::stable_audio_open_small();
    std::string why;
    fails += expect(sa3::sat::validate(saos, &why), why.c_str());
    fails += expect(saos.oobleck.downsampling_ratio() == 2048, "SAOS Oobleck ratio");
    fails += expect(saos.sample_size / saos.oobleck.downsampling_ratio() == 256,
                    "SAOS latent frame count");
    fails += expect(saos.dit.embed_dim == 1024 && saos.dit.depth == 16,
                    "SAOS DiT topology");

    const sa3::sat::ModelSpec sao1 = sa3::sat::stable_audio_open_1();
    const sa3::sat::ModelSpec foundation = sa3::sat::foundation_1();
    fails += expect(sa3::sat::validate(sao1, &why) &&
                    sa3::sat::validate(foundation, &why),
                    "SAO 1.0 family specs validate");
    fails += expect(sao1.dit.embed_dim == 1536 && sao1.dit.depth == 24 &&
                    sao1.dit.num_heads == 24 && !sao1.dit.qk_layer_norm,
                    "SAO 1.0 classic DiT topology");
    fails += expect(sao1.conditioner.seconds_start &&
                    sao1.dit.global_cond_dim == 1536 &&
                    sao1.text_encoder.max_length == 128,
                    "SAO 1.0 timing and text conditioning topology");
    fails += expect(sao1.sample_size == 2097152 && foundation.sample_size == 882000,
                    "Foundation changes only the published sample window");
    fails += expect(foundation.sample_size / foundation.oobleck.downsampling_ratio() == 430,
                    "Foundation follows stable-audio-tools floor division for latent frames");
    fails += expect(sa3::sat::weight_topology_compatible(sao1, foundation, &why),
                    "Foundation is weight-topology compatible with SAO 1.0");

    // Match RoyalCities' Foundation UI exactly: the audio crop follows the musical
    // duration, while seconds_total and the generated canvas round up independently.
    struct TimingCase { int bars, bpm, samples, seconds_total, frames; };
    constexpr TimingCase timing_cases[] = {
        {4, 100, 423360, 10, 216}, {4, 110, 384873, 9, 194},
        {4, 120, 352800, 8, 173},  {4, 128, 330750, 8, 173},
        {4, 130, 325662, 8, 173},  {4, 140, 302400, 7, 151},
        {4, 150, 282240, 7, 151},  {8, 100, 846720, 20, 431},
        {8, 110, 769745, 18, 388}, {8, 120, 705600, 16, 345},
        {8, 128, 661500, 15, 323}, {8, 130, 651323, 15, 323},
        {8, 140, 604800, 14, 302}, {8, 150, 564480, 13, 280},
    };
    for (const TimingCase& expected : timing_cases) {
        const sa3::sat::FoundationTiming timing =
            sa3::sat::resolve_foundation_timing(expected.bars, expected.bpm);
        fails += expect(timing.output_samples == expected.samples &&
                        timing.conditioning_seconds_total == expected.seconds_total &&
                        timing.latent_frames == expected.frames,
                        "Foundation trained BPM/bar timing table");
    }
    fails += expect(sa3::sat::kFoundationBpms.size() == 7 &&
                    sa3::sat::kFoundationBpms.back() == 150,
                    "Foundation trained BPM set includes 150");
    bool rejected_bad_bars = false, rejected_bad_bpm = false;
    try { (void)sa3::sat::resolve_foundation_timing(6, 128); }
    catch (const std::invalid_argument&) { rejected_bad_bars = true; }
    try { (void)sa3::sat::resolve_foundation_timing(4, 125); }
    catch (const std::invalid_argument&) { rejected_bad_bpm = true; }
    fails += expect(rejected_bad_bars && rejected_bad_bpm,
                    "Foundation timing rejects untrained geometry");
    sa3::sat::GenerateParams foundation_params;
    foundation_params.prompt = "warm tape-saturated breakbeat loop";
    const sa3::sat::FoundationTiming applied =
        sa3::sat::apply_foundation_timing(foundation_params, 4, 128);
    sa3::sat::apply_foundation_royalcities_sampler(foundation_params);
    fails += expect(foundation_params.prompt ==
                        "warm tape-saturated breakbeat loop, 4 Bars, 128 BPM" &&
                    foundation_params.output_samples == applied.output_samples &&
                    foundation_params.frames == 173 && foundation_params.seconds_total == 8.0f,
                    "Foundation profile applies prompt and independent generation geometry");
    fails += expect(foundation_params.sampler == sa3::sat::Sampler::Dpmpp3mSde &&
                    foundation_params.sigma_min == 0.01f &&
                    foundation_params.sigma_max == 100.0f,
                    "Foundation RoyalCities sampler profile");
    fails += expect(sa3::sat::foundation_prompt("warm pad", 8, 120, "F#", "minor") ==
                        "warm pad, 8 Bars, 120 BPM, F# minor",
                    "Foundation prompt includes timing and key conditioning");

    const sa3::sat::FoundationRandomPrompt random_a =
        sa3::sat::randomize_foundation_prompt(42, sa3::sat::FoundationPromptMode::Standard);
    const sa3::sat::FoundationRandomPrompt random_b =
        sa3::sat::randomize_foundation_prompt(42, sa3::sat::FoundationPromptMode::Standard);
    fails += expect(!random_a.description.empty() && random_a.description == random_b.description &&
                    random_a.variant == "M1",
                    "Foundation random prompts are deterministic from the audio seed");
    const sa3::sat::FoundationRandomControls controls_a =
        sa3::sat::randomize_foundation_controls(42);
    const sa3::sat::FoundationRandomControls controls_b =
        sa3::sat::randomize_foundation_controls(42);
    fails += expect(controls_a.bars == controls_b.bars && controls_a.bpm == controls_b.bpm &&
                    controls_a.key_root == controls_b.key_root &&
                    controls_a.key_mode == controls_b.key_mode &&
                    sa3::sat::is_foundation_bar_count(controls_a.bars) &&
                    sa3::sat::is_foundation_bpm(controls_a.bpm),
                    "Foundation omitted controls randomize deterministically on the trained grid");
    const sa3::sat::FoundationRandomPrompt synth_mix =
        sa3::sat::randomize_foundation_prompt(7, sa3::sat::FoundationPromptMode::Mix, "synth");
    fails += expect(synth_mix.family == "Synth" && synth_mix.variant == "T1" &&
                    synth_mix.description.find("Synth") != std::string::npos,
                    "Foundation T1 randomizer honors a family lock");
    bool rejected_family = false;
    try {
        (void)sa3::sat::randomize_foundation_prompt(
            1, sa3::sat::FoundationPromptMode::Standard, "Drums");
    } catch (const std::invalid_argument&) { rejected_family = true; }
    fails += expect(rejected_family, "Foundation randomizer rejects unknown family locks");

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

    float xe[] = {1.0f, -2.0f};
    const float ve[] = {2.0f, 4.0f};
    sa3::sampling::rf_euler_step(xe, ve, 2, 1.0f, 0.5f);
    fails += expect(near(xe[0], 0.0f) && near(xe[1], -4.0f),
                    "RF Euler update matches reference algebra");

    float xd[] = {1.0f, -2.0f};
    sa3::sampling::RfDpmppState dpmpp;
    sa3::sampling::rf_dpmpp_step(xd, ve, 2, 1.0f, 0.75f, dpmpp);
    fails += expect(near(xd[0], 0.5f) && near(xd[1], -3.0f),
                    "RF DPM++ first update matches reference algebra");
    const float vd[] = {0.25f, -0.5f};
    sa3::sampling::rf_dpmpp_step(xd, vd, 2, 0.75f, 0.5f, dpmpp);
    fails += expect(near(xd[0], 0.4375f) && near(xd[1], -2.875f),
                    "RF DPM++ multistep update matches reference algebra");
    const float vd2[] = {-0.1f, 0.2f};
    sa3::sampling::rf_dpmpp_step(xd, vd2, 2, 0.5f, 0.25f, dpmpp);
    fails += expect(near(xd[0], 0.50625f) && near(xd[1], -3.0125f),
                    "RF DPM++ finite-history correction matches reference algebra");
    const float before0 = xd[0], before1 = xd[1];
    sa3::sampling::rf_dpmpp_step(xd, vd2, 2, 0.25f, 0.0f, dpmpp);
    fails += expect(near(xd[0], before0 - 0.25f * vd2[0]) &&
                    near(xd[1], before1 - 0.25f * vd2[1]),
                    "RF DPM++ terminal update returns the denoised estimate");

    fails += expect(sa3::sat::parse_sampler("dpmpp") == sa3::sat::Sampler::Dpmpp &&
                    std::string(sa3::sat::sampler_name(sa3::sat::Sampler::PingPong)) == "pingpong",
                    "SAT sampler names round-trip");

    const std::vector<float> sigmas =
        sa3::sampling::make_sigma_polyexponential_schedule(3, 0.5f, 50.0f);
    fails += expect(sigmas.size() == 4 && near(sigmas[0], 50.0f, 1.0e-4f) &&
                    near(sigmas[1], 5.0f) && near(sigmas[2], 0.5f) && sigmas[3] == 0.0f,
                    "V-prediction polyexponential schedule matches k-diffusion");
    const sa3::sampling::VPredictionScalings vc =
        sa3::sampling::v_prediction_scalings(1.0f);
    fails += expect(near(vc.skip, 0.5f) && near(vc.output, -std::sqrt(0.5f)) &&
                    near(vc.input, std::sqrt(0.5f)) && near(vc.timestep, 0.5f),
                    "V-prediction preconditioning matches k-diffusion");
    float xv[] = {2.0f, -1.0f};
    const float vd0[] = {0.5f, -0.25f};
    const float vn0[] = {0.1f, -0.2f};
    const float vd1[] = {0.4f, -0.1f};
    const float vn1[] = {-0.3f, 0.4f};
    const float vd2s[] = {0.3f, 0.05f};
    const float vn2[] = {0.2f, 0.1f};
    const float vd3[] = {0.2f, 0.2f};
    sa3::sampling::VPredictionDpmppState vs;
    sa3::sampling::v_dpmpp_3m_sde_step(xv, vd0, vn0, 2, 50.0f, 5.0f, vs);
    fails += expect(near(xv[0], 1.0124937f) && near(xv[1], -1.2524874f),
                    "V-prediction DPM++ 3M first update matches reference algebra");
    sa3::sampling::v_dpmpp_3m_sde_step(xv, vd1, vn1, 2, 5.0f, 0.5f, vs);
    fails += expect(near(xv[0], 0.1783744f) && near(xv[1], 0.20522625f),
                    "V-prediction DPM++ 3M second-order update matches reference algebra");
    sa3::sampling::v_dpmpp_3m_sde_step(xv, vd2s, vn2, 2, 0.5f, 0.05f, vs);
    fails += expect(near(xv[0], 0.2302312f) && near(xv[1], 0.17428084f),
                    "V-prediction DPM++ 3M third-order update matches reference algebra");
    sa3::sampling::v_dpmpp_3m_sde_step(xv, vd3, nullptr, 2, 0.05f, 0.0f, vs);
    fails += expect(near(xv[0], 0.2f) && near(xv[1], 0.2f),
                    "V-prediction DPM++ terminal update returns denoised estimate");

    float xv2[] = {2.0f, -1.0f};
    sa3::sampling::VPredictionDpmppState vs2;
    sa3::sampling::v_dpmpp_2m_sde_step(xv2, vd0, vn0, 2, 50.0f, 5.0f, vs2);
    fails += expect(near(xv2[0], 1.0124937f) && near(xv2[1], -1.2524874f),
                    "V-prediction DPM++ 2M first update matches reference algebra");
    sa3::sampling::v_dpmpp_2m_sde_step(xv2, vd1, vn1, 2, 5.0f, 0.5f, vs2);
    fails += expect(near(xv2[0], 0.20737682f) && near(xv2[1], 0.16172261f),
                    "V-prediction DPM++ 2M multistep update matches reference algebra");
    sa3::sampling::v_dpmpp_2m_sde_step(xv2, vd3, nullptr, 2, 0.5f, 0.0f, vs2);
    fails += expect(near(xv2[0], 0.2f) && near(xv2[1], 0.2f),
                    "V-prediction DPM++ 2M terminal update matches reference algebra");

    fails += expect(sa3::sat::parse_sampler("dpmpp-2m-sde") ==
                        sa3::sat::Sampler::Dpmpp2mSde &&
                    sa3::sat::parse_sampler("dpmpp-3m-sde") ==
                        sa3::sat::Sampler::Dpmpp3mSde,
                    "V-prediction SDE sampler names round-trip");

    bool threw = false;
    try { (void)sa3::sampling::make_rf_logsnr_schedule(0); }
    catch (const std::invalid_argument&) { threw = true; }
    fails += expect(threw, "invalid schedule rejected");

    if (fails) return 1;
    std::printf("sat_sampling_test: ok\n");
    return 0;
}
