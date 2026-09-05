#include "sat/pipeline.h"

#include "gguf_model.h"
#include "rng.h"
#include "sampling.h"
#include "sat/conditioning.h"
#include "sat/dit.h"
#include "sat/oobleck.h"
#include "sat/t5.h"
#include "sat/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sa3::sat {
namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct GraphArena {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ~GraphArena() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

struct PromptEncoding {
    std::vector<float> hidden;
    int valid_tokens = 0;
    int max_tokens = 0;
};

std::vector<PromptEncoding> encode_t5_prompts(const std::string& path,
                                               const std::vector<std::string>& prompts) {
    UnigramTokenizer tokenizer = UnigramTokenizer::load(path.c_str());
    GgufModel t5 = load_gguf(path.c_str());
    const T5EncoderConfig c = T5EncoderConfig::from(t5);
    const int seq = (int)t5.u32("sat.t5.max_length");

    GraphArena arena;
    ggml_init_params ip = {(size_t)128 * 1024 * 1024, nullptr, true};
    arena.ctx = ggml_init(ip);
    if (!arena.ctx) throw std::runtime_error("failed to create T5 graph context");
    ggml_tensor* ids_t = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, seq);
    ggml_tensor* mask_t = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32, seq, seq);
    ggml_tensor* rel_t = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, (int64_t)seq * seq);
    for (ggml_tensor* t : {ids_t, mask_t, rel_t}) ggml_set_input(t);
    ggml_tensor* hidden = ggml_cont(arena.ctx, t5_encode(arena.ctx, t5, ids_t, mask_t, rel_t, c));
    ggml_set_output(hidden);
    ggml_cgraph* graph = ggml_new_graph_custom(arena.ctx, 8192, false);
    ggml_build_forward_expand(graph, hidden);
    arena.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t5.backend));
    if (!arena.alloc || !ggml_gallocr_alloc_graph(arena.alloc, graph))
        throw std::runtime_error("failed to allocate T5 graph");

    const std::vector<int32_t> rel = t5_relative_position_buckets(
        seq, c.relative_buckets, c.relative_max_distance);
    std::vector<PromptEncoding> results;
    results.reserve(prompts.size());
    for (const std::string& prompt : prompts) {
        std::vector<int32_t> ids = tokenizer.encode(prompt, seq);
        const int valid = (int)ids.size();
        ids.resize((size_t)seq, tokenizer.pad_id);
        std::vector<float> mask((size_t)seq * seq, -INFINITY);
        for (int q = 0; q < seq; ++q)
            std::fill_n(mask.data() + (size_t)q * seq, valid, 0.0f);
        ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
        // gallocr may recycle input storage after every execution.
        ggml_backend_tensor_set(rel_t, rel.data(), 0, rel.size() * sizeof(int32_t));
        std::string error;
        if (!graph_compute_checked(t5.backend, graph, "SAT T5", error))
            throw std::runtime_error(error);
        PromptEncoding result;
        result.hidden.resize((size_t)c.dim * seq);
        ggml_backend_tensor_get(hidden, result.hidden.data(), 0,
                                result.hidden.size() * sizeof(float));
        for (int token = valid; token < seq; ++token)
            std::fill_n(result.hidden.data() + (size_t)token * c.dim, c.dim, 0.0f);
        result.valid_tokens = valid;
        result.max_tokens = seq;
        results.push_back(std::move(result));
    }
    return results;
}

Sampler default_sampler(const std::string& objective) {
    if (objective == "rf_denoiser") return Sampler::PingPong;
    if (objective == "v") return Sampler::Dpmpp3mSde;
    return Sampler::Euler;
}

} // namespace

const char* sampler_name(Sampler sampler) {
    switch (sampler) {
        case Sampler::Auto: return "auto";
        case Sampler::Euler: return "euler";
        case Sampler::Dpmpp: return "dpmpp";
        case Sampler::PingPong: return "pingpong";
        case Sampler::Dpmpp2mSde: return "dpmpp-2m-sde";
        case Sampler::Dpmpp3mSde: return "dpmpp-3m-sde";
    }
    return "unknown";
}

Sampler parse_sampler(const std::string& name) {
    if (name == "auto") return Sampler::Auto;
    if (name == "euler") return Sampler::Euler;
    if (name == "dpmpp") return Sampler::Dpmpp;
    if (name == "pingpong") return Sampler::PingPong;
    if (name == "dpmpp-2m-sde") return Sampler::Dpmpp2mSde;
    if (name == "dpmpp-3m-sde") return Sampler::Dpmpp3mSde;
    throw std::invalid_argument("unknown SAT sampler: " + name);
}

void Pipeline::load(PipelinePaths paths) {
    if (paths.dit.empty() || paths.autoencoder.empty())
        throw std::invalid_argument("SAT pipeline requires DiT and autoencoder GGUF paths");
    paths_ = std::move(paths);
}

GenerateResult Pipeline::generate(const GenerateParams& params) const {
    if (paths_.dit.empty() || paths_.autoencoder.empty())
        throw std::runtime_error("SAT pipeline is not loaded");
    if (params.frames < 1 || params.seconds <= 0.0f || params.steps < 0 ||
        params.output_samples < 0 || !std::isfinite(params.cfg_scale) ||
        !std::isfinite(params.seconds_start) || !std::isfinite(params.sigma_min) ||
        !std::isfinite(params.sigma_max) || !std::isfinite(params.sigma_rho) ||
        !std::isfinite(params.sde_eta) || !(params.sigma_rho > 0.0f) ||
        !(params.sde_eta >= 0.0f))
        throw std::invalid_argument("invalid SAT generation geometry");
    const bool conditioning_override = !params.cross_conditioning.empty() ||
                                       !params.global_conditioning.empty();
    if (conditioning_override && (params.cross_conditioning.empty() ||
                                  params.global_conditioning.empty()))
        throw std::invalid_argument("cross and global conditioning overrides must be supplied together");
    if (!conditioning_override && (paths_.t5.empty() || params.prompt.empty()))
        throw std::invalid_argument("native SAT conditioning requires T5 and a prompt");

    GenerateResult result;
    const double total_start = now_s();
    std::vector<PromptEncoding> encoded;
    if (!conditioning_override) {
        const double start = now_s();
        std::vector<std::string> prompts{params.prompt};
        if (!params.negative_prompt.empty()) prompts.push_back(params.negative_prompt);
        encoded = encode_t5_prompts(paths_.t5, prompts);
        result.timing.conditioning_s = now_s() - start;
        result.prompt_tokens = encoded.front().valid_tokens;
        result.max_prompt_tokens = encoded.front().max_tokens;
    }

    DitSpec dc;
    std::vector<float> x;
    {
        double start = now_s();
        GgufModel dit = load_gguf(paths_.dit.c_str());
        result.timing.dit_load_s = now_s() - start;
        dc = dit_spec_from(dit);
        result.objective = dit.string("sat.diffusion_objective");
        if (result.objective != "rf_denoiser" && result.objective != "rectified_flow" &&
            result.objective != "v")
            throw std::runtime_error("unsupported SAT diffusion objective");
        result.sampler = params.sampler == Sampler::Auto
            ? default_sampler(result.objective) : params.sampler;
        const bool v_prediction = result.objective == "v";
        const bool sde_sampler = result.sampler == Sampler::Dpmpp2mSde ||
                                 result.sampler == Sampler::Dpmpp3mSde;
        if (!v_prediction && sde_sampler)
            throw std::invalid_argument("k-diffusion SDE samplers are not supported by the RF pipeline");
        if (v_prediction && !sde_sampler)
            throw std::invalid_argument("V-prediction currently requires dpmpp-2m-sde or dpmpp-3m-sde");
        result.steps = params.steps > 0 ? params.steps
            : (result.objective == "rf_denoiser" ? 8 : (v_prediction ? 100 : 50));
        result.cfg_scale = params.cfg_scale >= 0.0f ? params.cfg_scale
            : (result.objective == "rf_denoiser" ? 1.0f : (v_prediction ? 7.0f : 4.0f));
        if (v_prediction) {
            result.sigma_min = params.sigma_min > 0.0f ? params.sigma_min : 0.3f;
            result.sigma_max = params.sigma_max > 0.0f ? params.sigma_max : 500.0f;
            result.sigma_rho = params.sigma_rho;
            result.sde_eta = params.sde_eta;
        }

        if (conditioning_override) {
            result.cross_conditioning = params.cross_conditioning;
            result.global_conditioning = params.global_conditioning;
        } else {
            result.cross_conditioning = std::move(encoded[0].hidden);
            std::vector<float> timing_cross;
            if (dit.has("conditioner.seconds_start.fourier.weight")) {
                std::vector<float> start_condition =
                    seconds_start_condition(dit, params.seconds_start);
                result.global_conditioning.insert(result.global_conditioning.end(),
                                                   start_condition.begin(),
                                                   start_condition.end());
                timing_cross.insert(timing_cross.end(), start_condition.begin(),
                                    start_condition.end());
            }
            std::vector<float> total_condition = seconds_total_condition(dit, params.seconds);
            result.global_conditioning.insert(result.global_conditioning.end(),
                                               total_condition.begin(),
                                               total_condition.end());
            timing_cross.insert(timing_cross.end(), total_condition.begin(),
                                total_condition.end());
            if (result.cross_conditioning.size() !=
                (size_t)encoded[0].max_tokens * dc.cond_token_dim)
                throw std::runtime_error("T5 and DiT conditioning dimensions differ");
            result.cross_conditioning.insert(result.cross_conditioning.end(),
                                              timing_cross.begin(), timing_cross.end());
        }
        if (result.global_conditioning.size() != (size_t)dc.global_cond_dim ||
            result.cross_conditioning.size() % (size_t)dc.cond_token_dim)
            throw std::runtime_error("conditioning dimensions do not match the DiT");

        std::vector<float> negative_cross(result.cross_conditioning.size(), 0.0f);
        if (!params.negative_prompt.empty()) {
            if (conditioning_override)
                throw std::invalid_argument("negative prompt cannot accompany conditioning overrides");
            negative_cross = std::move(encoded[1].hidden);
            const size_t timing_dims = result.global_conditioning.size();
            negative_cross.insert(negative_cross.end(),
                                  result.cross_conditioning.end() - timing_dims,
                                  result.cross_conditioning.end());
        }

        const int cond_tokens = (int)(result.cross_conditioning.size() /
                                      (size_t)dc.cond_token_dim);
        const int tokens = params.frames + 1;
        const size_t latent_n = (size_t)dc.io_channels * params.frames;
        if (!params.initial_latent.empty() && params.initial_latent.size() != latent_n)
            throw std::invalid_argument("initial latent has the wrong size");
        if (!params.step_noise.empty() &&
            (result.sampler != Sampler::PingPong && !sde_sampler))
            throw std::invalid_argument("step noise requires a stochastic sampler");
        if (!params.step_noise.empty() &&
            params.step_noise.size() != (size_t)result.steps * latent_n)
            throw std::invalid_argument("step noise requires one buffer per step");

        start = now_s();
        GraphArena arena;
        ggml_init_params dip = {(size_t)128 * 1024 * 1024, nullptr, true};
        arena.ctx = ggml_init(dip);
        if (!arena.ctx) throw std::runtime_error("failed to create DiT graph context");
        ggml_tensor* x_in = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32,
                                                dc.io_channels, params.frames);
        ggml_tensor* time = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_F32, 1);
        ggml_tensor* cross = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32,
                                                 dc.cond_token_dim, cond_tokens);
        ggml_tensor* global = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_F32,
                                                  dc.global_cond_dim);
        ggml_tensor* pos = ggml_new_tensor_1d(arena.ctx, GGML_TYPE_I32, tokens);
        for (ggml_tensor* t : {x_in, time, cross, global, pos}) ggml_set_input(t);
        ggml_tensor* velocity = ggml_cont(arena.ctx, classic_dit_forward(
            arena.ctx, dit, x_in, time, cross, global, pos, dc));
        ggml_set_output(velocity);
        ggml_cgraph* graph = ggml_new_graph_custom(arena.ctx, 32768, false);
        ggml_build_forward_expand(graph, velocity);
        arena.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dit.backend));
        if (!arena.alloc || !ggml_gallocr_alloc_graph(arena.alloc, graph))
            throw std::runtime_error("failed to allocate DiT graph");
        result.timing.dit_build_s = now_s() - start;

        std::vector<int32_t> positions((size_t)tokens);
        std::iota(positions.begin(), positions.end(), 0);
        Rng rng(params.seed);
        x.resize(latent_n);
        if (!params.initial_latent.empty()) x = params.initial_latent;
        else rng.fill_normal(x.data(), x.size());
        std::vector<float> velocity_cond(latent_n), velocity_uncond(latent_n), noise(latent_n);
        std::vector<float> model_input(latent_n), denoised(latent_n);
        const std::vector<float> schedule = v_prediction
            ? sampling::make_sigma_polyexponential_schedule(result.steps, result.sigma_min,
                                                              result.sigma_max, params.sigma_rho)
            : sampling::make_rf_logsnr_schedule(result.steps);
        if (v_prediction)
            for (float& value : x) value *= schedule.front();
        sampling::RfDpmppState dpmpp;
        sampling::VPredictionDpmppState v_dpmpp;
        std::vector<double> step_ms;
        step_ms.reserve((size_t)result.steps);
        std::string compute_error;

        auto evaluate = [&](const std::vector<float>& state,
                            const std::vector<float>& condition, float t,
                            std::vector<float>& output) {
            ggml_backend_tensor_set(x_in, state.data(), 0, state.size() * sizeof(float));
            ggml_backend_tensor_set(time, &t, 0, sizeof(float));
            ggml_backend_tensor_set(cross, condition.data(), 0,
                                    condition.size() * sizeof(float));
            ggml_backend_tensor_set(global, result.global_conditioning.data(), 0,
                                    result.global_conditioning.size() * sizeof(float));
            ggml_backend_tensor_set(pos, positions.data(), 0,
                                    positions.size() * sizeof(int32_t));
            if (!graph_compute_checked(dit.backend, graph, "SAT DiT", compute_error))
                throw std::runtime_error(compute_error);
            ggml_backend_tensor_get(velocity, output.data(), 0,
                                    output.size() * sizeof(float));
        };

        const double denoise_start = now_s();
        for (int i = 0; i < result.steps; ++i) {
            const double step_start = now_s();
            const float current = schedule[(size_t)i];
            const float next = schedule[(size_t)i + 1];
            float model_t = current;
            if (v_prediction) {
                const sampling::VPredictionScalings c =
                    sampling::v_prediction_scalings(current);
                model_t = c.timestep;
                for (size_t j = 0; j < latent_n; ++j) model_input[j] = x[j] * c.input;
            } else {
                model_input = x;
            }
            evaluate(model_input, result.cross_conditioning, model_t, velocity_cond);
            if (result.cfg_scale != 1.0f) {
                evaluate(model_input, negative_cross, model_t, velocity_uncond);
                for (size_t j = 0; j < latent_n; ++j)
                    velocity_cond[j] = velocity_uncond[j] + result.cfg_scale *
                        (velocity_cond[j] - velocity_uncond[j]);
            }
            const double milliseconds = (now_s() - step_start) * 1000.0;
            step_ms.push_back(milliseconds);
            if (v_prediction) {
                sampling::v_prediction_to_denoised(x.data(), velocity_cond.data(),
                                                     denoised.data(), latent_n, current);
                if (next > 0.0f && params.sde_eta != 0.0f) {
                    if (!params.step_noise.empty())
                        std::copy_n(params.step_noise.data() + (size_t)i * latent_n,
                                    latent_n, noise.data());
                    else
                        rng.fill_normal(noise.data(), noise.size());
                }
                if (result.sampler == Sampler::Dpmpp2mSde)
                    sampling::v_dpmpp_2m_sde_step(x.data(), denoised.data(), noise.data(),
                                                   latent_n, current, next, v_dpmpp,
                                                   params.sde_eta);
                else
                    sampling::v_dpmpp_3m_sde_step(x.data(), denoised.data(), noise.data(),
                                                   latent_n, current, next, v_dpmpp,
                                                   params.sde_eta);
            } else switch (result.sampler) {
                case Sampler::Euler:
                    sampling::rf_euler_step(x.data(), velocity_cond.data(), latent_n,
                                             current, next);
                    break;
                case Sampler::Dpmpp:
                    sampling::rf_dpmpp_step(x.data(), velocity_cond.data(), latent_n,
                                             current, next, dpmpp);
                    break;
                case Sampler::PingPong:
                    if (!params.step_noise.empty())
                        std::copy_n(params.step_noise.data() + (size_t)i * latent_n,
                                    latent_n, noise.data());
                    else
                        rng.fill_normal(noise.data(), noise.size());
                    sampling::rf_pingpong_step(x.data(), velocity_cond.data(), noise.data(),
                                                latent_n, current, next);
                    break;
                case Sampler::Auto:
                    throw std::logic_error("unresolved SAT sampler");
                default:
                    throw std::logic_error("unsupported SAT sampler");
            }
            if (params.progress) params.progress({i + 1, result.steps, current, next,
                                                  milliseconds});
        }
        result.timing.denoise_s = now_s() - denoise_start;
        std::vector<double> steady = step_ms;
        if (steady.size() > 1) steady.erase(steady.begin());
        std::sort(steady.begin(), steady.end());
        result.timing.warm_step_median_ms = steady[steady.size() / 2];
    }

    result.latent = x;
    double start = now_s();
    GgufModel ae = load_gguf(paths_.autoencoder.c_str());
    result.timing.ae_load_s = now_s() - start;
    const OobleckSpec ac = oobleck_spec_from(ae);
    if (ac.latent_channels != dc.io_channels)
        throw std::runtime_error("DiT and Oobleck latent widths differ");
    const size_t latent_n = (size_t)dc.io_channels * params.frames;
    std::vector<float> decoder_z(latent_n);
    for (int frame = 0; frame < params.frames; ++frame)
        for (int channel = 0; channel < dc.io_channels; ++channel)
            decoder_z[(size_t)channel * params.frames + frame] =
                x[(size_t)frame * dc.io_channels + channel];

    start = now_s();
    GraphArena arena;
    ggml_init_params aip = {(size_t)64 * 1024 * 1024, nullptr, true};
    arena.ctx = ggml_init(aip);
    if (!arena.ctx) throw std::runtime_error("failed to create decoder graph context");
    ggml_tensor* z = ggml_new_tensor_2d(arena.ctx, GGML_TYPE_F32,
                                         params.frames, ac.latent_channels);
    ggml_set_input(z);
    ggml_tensor* audio = ggml_cont(arena.ctx, oobleck_decode(arena.ctx, ae, z, ac));
    ggml_set_output(audio);
    ggml_cgraph* graph = ggml_new_graph_custom(arena.ctx, 32768, false);
    ggml_build_forward_expand(graph, audio);
    arena.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ae.backend));
    if (!arena.alloc || !ggml_gallocr_alloc_graph(arena.alloc, graph))
        throw std::runtime_error("failed to allocate Oobleck graph");
    result.timing.ae_build_s = now_s() - start;
    ggml_backend_tensor_set(z, decoder_z.data(), 0, decoder_z.size() * sizeof(float));
    std::string compute_error;
    start = now_s();
    if (!graph_compute_checked(ae.backend, graph, "SAT Oobleck decode", compute_error))
        throw std::runtime_error(compute_error);
    result.timing.decode_s = now_s() - start;

    const int decoded_samples = (int)audio->ne[0];
    result.channels = (int)audio->ne[1];
    result.sample_rate = (int)ae.u32("sat.sample_rate");
    result.samples = params.output_samples > 0 ? params.output_samples
        : (int)std::lround(params.seconds * result.sample_rate);
    result.samples = std::min(result.samples, decoded_samples);
    std::vector<float> decoded((size_t)decoded_samples * result.channels);
    ggml_backend_tensor_get(audio, decoded.data(), 0, decoded.size() * sizeof(float));
    result.audio.resize((size_t)result.samples * result.channels);
    for (int channel = 0; channel < result.channels; ++channel)
        std::copy_n(decoded.data() + (size_t)channel * decoded_samples, result.samples,
                    result.audio.data() + (size_t)channel * result.samples);
    result.timing.total_s = now_s() - total_start;
    return result;
}

} // namespace sa3::sat
