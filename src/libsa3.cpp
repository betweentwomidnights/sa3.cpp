// libsa3 — the V1 C ABI over sa3::Pipeline. See libsa3_v1.h and docs/C_ABI_V1.md for the
// contract, and libsa3_training_v1.h for the independently versioned training capability.
#define SA3_BUILD_DLL
#include "libsa3_v1.h"
#include "libsa3_training_v1.h"
#include "sa3_pipeline.h"
#include "lora_convert.h"
#include "train_job.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

// The context owns the pipeline via a unique_ptr so context_unload() can drop the models (and
// their VRAM) while keeping the context alive; the next generate lazily reloads from `paths`.
// It holds no per-call state: the result carries everything a generation reports.
struct sa3_context {
    std::unique_ptr<sa3::Pipeline> pipe;
    sa3::ModelPaths paths;
    std::string adapters_dir;
    int cpu_threads = 0;
    std::string device;  // remembered so the frugal-reload path keeps the same backend
};





extern "C" {
}

namespace {

template <typename T>
void init_v1_struct(T* value) {
    if (!value) return;
    const uint32_t caller_size = value->size;
    if (caller_size < sizeof(uint32_t)) return;
    std::memset(value, 0, std::min<size_t>(caller_size, sizeof(T)));
    value->size = caller_size;
}

template <typename T>
bool has_v1_size(const T* value, uint32_t minimum_size) {
    return value && value->size >= minimum_size;
}

/// Both tables publish this. Bumped when the exported surface changes, which is now only ever a
/// V1 table growing -- the legacy entry points it used to sit beside are gone.
const char* SA3_CALL v1_runtime_version(void) { return "sa3.cpp libsa3 7 (ABI 1)"; }

void SA3_CALL v1_error_init(sa3_error_v1* error) {
    init_v1_struct(error);
}

void set_v1_error(sa3_error_v1* error, sa3_status_v1 code, const std::string& message) {
    if (!error || error->size < sizeof(uint32_t)) return;
    const uint32_t caller_size = error->size;
    std::memset(error, 0, std::min<size_t>(caller_size, sizeof(*error)));
    error->size = caller_size;
    if (caller_size >= offsetof(sa3_error_v1, code) + sizeof(error->code)) error->code = code;
    if (caller_size > offsetof(sa3_error_v1, message)) {
        const size_t cap = std::min<size_t>(sizeof(error->message),
                                            caller_size - offsetof(sa3_error_v1, message));
        if (cap > 0) {
            std::strncpy(error->message, message.c_str(), cap - 1);
            error->message[cap - 1] = '\0';
        }
    }
}

void clear_v1_error(sa3_error_v1* error) {
    set_v1_error(error, SA3_STATUS_OK_V1, "");
}

void SA3_CALL v1_context_config_init(sa3_context_config_v1* config) {
    init_v1_struct(config);
}

void SA3_CALL v1_audio_view_init(sa3_audio_view_v1* audio) {
    init_v1_struct(audio);
    if (audio && audio->size >= SA3_AUDIO_VIEW_V1_MIN_SIZE) audio->layout = SA3_AUDIO_PLANAR_V1;
}

void SA3_CALL v1_request_init(sa3_request_v1* request) {
    init_v1_struct(request);
    if (!request || request->size < SA3_REQUEST_V1_MIN_SIZE) return;
    request->operation = SA3_OPERATION_GENERATE_V1;
    request->duration_seconds = 12.0;
    request->steps = 8;
    request->seed = -1;
    request->cfg_scale = 1.0f;
    request->distribution_shift = SA3_DISTRIBUTION_LOGSNR_V1;
    request->residency = SA3_RESIDENCY_RESIDENT_V1;
    request->input_audio.size = sizeof(request->input_audio);
    request->input_audio.layout = SA3_AUDIO_PLANAR_V1;
    request->transform_noise_level = 0.85f;
    request->generation_tail_padding_seconds = 6.0f;
    request->continuation_tail_padding_seconds = 6.0f;
    request->adapter_stride = sizeof(sa3_adapter_v1);
    request->loudness.size = sizeof(request->loudness);
    request->loudness.peak_normalize = 1;
    request->loudness.peak_normalize_db = 2.0f;
    request->loudness.limiter = 1;
    request->loudness.limiter_ceiling_db = -0.3f;
    request->loudness.limiter_knee = 0.8f;
    request->loudness.latent_rescale = 1.0f;
    request->continuation.size = sizeof(request->continuation);
    request->continuation.splice_source = 1;
    request->continuation.mask_overlap_seconds = 0.2f;
    request->continuation.crossfade_seconds = 0.03f;
    request->continuation.gain_match = 1;
    request->encode_overlap = 32;
    request->decode_overlap = 32;
}

void SA3_CALL v1_adapter_init(sa3_adapter_v1* adapter) {
    init_v1_struct(adapter);
    if (adapter && adapter->size >= SA3_ADAPTER_V1_MIN_SIZE) adapter->strength = 1.0f;
}

void SA3_CALL v1_loudness_init(sa3_loudness_v1* loudness) {
    init_v1_struct(loudness);
    if (!loudness || loudness->size < SA3_LOUDNESS_V1_MIN_SIZE) return;
    loudness->peak_normalize = 1;
    loudness->peak_normalize_db = 2.0f;
    loudness->limiter = 1;
    loudness->limiter_ceiling_db = -0.3f;
    loudness->limiter_knee = 0.8f;
    loudness->latent_rescale = 1.0f;
}

void SA3_CALL v1_continuation_init(sa3_continuation_v1* continuation) {
    init_v1_struct(continuation);
    if (!continuation || continuation->size < SA3_CONTINUATION_V1_MIN_SIZE) return;
    continuation->splice_source = 1;
    continuation->mask_overlap_seconds = 0.2f;
    continuation->crossfade_seconds = 0.03f;
    continuation->gain_match = 1;
}

void SA3_CALL v1_result_init(sa3_result_v1* result) {
    init_v1_struct(result);
    if (result && result->size >= SA3_RESULT_V1_MIN_SIZE) result->layout = SA3_AUDIO_PLANAR_V1;
}

void SA3_CALL v1_lora_convert_init(sa3_lora_convert_v1* options) {
    init_v1_struct(options);
}

sa3_progress_stage_v1 v1_progress_stage(const char* stage) {
    if (!stage) return SA3_PROGRESS_OTHER_V1;
    if (std::strcmp(stage, "loading") == 0) return SA3_PROGRESS_LOADING_V1;
    if (std::strcmp(stage, "encoding") == 0) return SA3_PROGRESS_ENCODING_V1;
    if (std::strcmp(stage, "sampling") == 0) return SA3_PROGRESS_SAMPLING_V1;
    if (std::strcmp(stage, "decoding") == 0) return SA3_PROGRESS_DECODING_V1;
    if (std::strcmp(stage, "done") == 0) return SA3_PROGRESS_DONE_V1;
    return SA3_PROGRESS_OTHER_V1;
}

struct V1CallbackBridge {
    const sa3_request_v1* request = nullptr;
};

void v1_progress_bridge(void* user, const char* stage, int step, int total, float fraction) {
    const auto* bridge = static_cast<const V1CallbackBridge*>(user);
    if (!bridge || !bridge->request || !bridge->request->on_progress) return;
    sa3_progress_v1 progress{};
    progress.size = sizeof(progress);
    progress.stage = v1_progress_stage(stage);
    progress.stage_name = stage;
    progress.step = step;
    progress.total = total;
    progress.fraction = fraction;
    bridge->request->on_progress(bridge->request->callback_user, &progress);
}

int v1_cancel_bridge(void* user) {
    const auto* bridge = static_cast<const V1CallbackBridge*>(user);
    return bridge && bridge->request && bridge->request->should_cancel
         ? bridge->request->should_cancel(bridge->request->callback_user) : 0;
}

bool seconds_to_samples(double seconds, int& samples) {
    if (!std::isfinite(seconds) || seconds <= 0.0) return false;
    const double rounded = std::round(seconds * 44100.0);
    if (rounded < 1.0 || rounded > (double)std::numeric_limits<int>::max()) return false;
    samples = (int)rounded;
    return true;
}

const char* v1_distribution_name(sa3_distribution_shift_v1 shift) {
    switch (shift) {
        case SA3_DISTRIBUTION_LOGSNR_V1: return "LogSNR";
        case SA3_DISTRIBUTION_FLUX_V1: return "Flux";
        case SA3_DISTRIBUTION_FULL_V1: return "Full";
        case SA3_DISTRIBUTION_NONE_V1: return "None";
        default: return nullptr;
    }
}


sa3_status_v1 fail_v1(sa3_error_v1* error, sa3_status_v1 status, const std::string& message) {
    set_v1_error(error, status, message);
    return status;
}

sa3_status_v1 SA3_CALL v1_context_create(const sa3_context_config_v1* config,
                                          sa3_context** out_context,
                                          sa3_error_v1* error) {
    clear_v1_error(error);
    if (!has_v1_size(config, SA3_CONTEXT_CONFIG_V1_MIN_SIZE) || !out_context)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "context config is too small or out_context is null");
    *out_context = nullptr;
    if (config->cpu_threads < 0)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "cpu_threads must be non-negative");

    try {
        // A null or empty string means "unset" and falls back; everything else is taken as given.
        auto text = [](const char* value) { return value && *value ? std::string(value) : std::string(); };
        std::string models_dir = text(config->models_dir);
        if (models_dir.empty()) {
            const char* from_env = std::getenv("SA3_MODELS_DIR");
            models_dir = (from_env && *from_env) ? from_env : "models";
        }
        std::string variant = text(config->variant);
        if (variant.empty()) variant = "medium";
        std::string dit = text(config->dit_encoding);
        if (dit.empty()) dit = "f16";
        std::string adapters_dir = text(config->adapters_dir);
        if (adapters_dir.empty()) adapters_dir = models_dir;
        const std::string device = text(config->device);

        sa3::ModelPaths paths;
        std::string why;
        if (!sa3::ModelPaths::resolve(models_dir, variant, dit, text(config->text_encoder_encoding),
                                      text(config->autoencoder_encoding), paths, why))
            return fail_v1(error, SA3_STATUS_IO_ERROR_V1, why);

        auto created = std::make_unique<sa3_context>();
        created->paths = paths;
        created->adapters_dir = adapters_dir;
        created->cpu_threads = config->cpu_threads;
        created->device = device;
        created->pipe = std::make_unique<sa3::Pipeline>();
        created->pipe->load(paths, created->cpu_threads, device.empty() ? nullptr : device.c_str());
        *out_context = created.release();
        return SA3_STATUS_OK_V1;
    } catch (const std::bad_alloc&) {
        return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory creating context");
    } catch (const std::exception& e) {
        return fail_v1(error, SA3_STATUS_MODEL_ERROR_V1, e.what());
    } catch (...) {
        return fail_v1(error, SA3_STATUS_INTERNAL_ERROR_V1, "unknown error creating context");
    }
}

void SA3_CALL v1_context_unload(sa3_context* context) {
    if (context) context->pipe.reset();
}

void SA3_CALL v1_context_destroy(sa3_context* context) {
    delete context;
}

void SA3_CALL v1_result_free(sa3_result_v1* result) {
    if (!result || result->size < sizeof(uint32_t)) return;
    const uint32_t caller_size = result->size;
    if (caller_size >= offsetof(sa3_result_v1, samples) + sizeof(result->samples) && result->samples)
        std::free(result->samples);
    std::memset(result, 0, std::min<size_t>(caller_size, sizeof(*result)));
    result->size = caller_size;
    if (caller_size >= offsetof(sa3_result_v1, layout) + sizeof(result->layout))
        result->layout = SA3_AUDIO_PLANAR_V1;
}

/// Runs one prepared request and publishes the result.
///
/// This is where a C++ exception becomes a V1 status, and the only place that happens on the
/// generation path. The pipeline signals a cooperative stop by throwing, which is a cancellation
/// rather than a failure -- callers get CANCELLED and an untouched result.
sa3_status_v1 run_generation(sa3_context* context, sa3::GenParams& params,
                             sa3_result_v1* result, sa3_error_v1* error) {
    try {
        if (!context->pipe) {   // reload after context_unload()
            context->pipe = std::make_unique<sa3::Pipeline>();
            context->pipe->load(context->paths, context->cpu_threads,
                                context->device.empty() ? nullptr : context->device.c_str());
        }
        const sa3::GenResult r = context->pipe->generate(params);
        const size_t n = (size_t)r.n_samp * r.n_ch;
        float* samples = (float*)std::malloc(n * sizeof(float));
        if (!samples)
            return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory allocating result audio");
        std::memcpy(samples, r.samples.data(), n * sizeof(float));

        const uint32_t result_size = result->size;
        std::memset(result, 0, std::min<size_t>(result_size, sizeof(*result)));
        result->size = result_size;
        result->samples = samples;
        result->n_samples = (uint64_t)r.n_samp;
        result->n_channels = (uint32_t)r.n_ch;
        result->sample_rate = (uint32_t)r.sample_rate;
        result->layout = SA3_AUDIO_PLANAR_V1;
        result->seed = params.seed;
        // Straight off the pipeline's own metadata. It used to be parked on the context between
        // the generate call and a separate getter, because the legacy sa3_audio had nowhere to
        // carry it; sa3_result_v1 does, so the context no longer holds state between calls.
        result->decoded_peak = r.loudness.decoded_peak;
        result->peak_normalize_gain_set = r.loudness.peak_normalize_gain_set ? 1 : 0;
        result->peak_normalize_gain = r.loudness.peak_normalize_gain;
        result->limiter_limited_fraction_set = r.loudness.limiter_limited_fraction_set ? 1 : 0;
        result->limiter_limited_fraction = r.loudness.limiter_limited_fraction;
        result->safety_gain_set = r.loudness.safety_gain_set ? 1 : 0;
        result->safety_gain = r.loudness.safety_gain;
        result->final_peak = r.loudness.final_peak;
        result->latent_factor = r.loudness.latent_factor;
        result->splice_applied = r.splice.applied ? 1 : 0;
        result->splice_end_seconds = r.splice.splice_end_seconds;
        result->splice_crossfade_seconds = r.splice.xfade_applied;
        result->splice_gain = r.splice.gain;
        result->mask_start_seconds = r.splice.mask_start_seconds;
        result->mask_overlap_seconds = r.splice.mask_overlap_applied;
        return SA3_STATUS_OK_V1;
    } catch (const std::bad_alloc&) {
        return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory during generation");
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("cancelled") != std::string::npos)
            return fail_v1(error, SA3_STATUS_CANCELLED_V1, what);
        return fail_v1(error, SA3_STATUS_MODEL_ERROR_V1, what);
    } catch (...) {
        return fail_v1(error, SA3_STATUS_INTERNAL_ERROR_V1, "unknown error during generation");
    }
}

sa3_status_v1 SA3_CALL v1_generate(sa3_context* context,
                                    const sa3_request_v1* request,
                                    sa3_result_v1* result,
                                    sa3_error_v1* error) {
    clear_v1_error(error);
    if (!context || !has_v1_size(request, SA3_REQUEST_V1_MIN_SIZE) ||
        !has_v1_size(result, SA3_RESULT_V1_MIN_SIZE))
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "context, full V1 request, and full V1 result are required");
    if (result->samples)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "result already owns audio; call result_free before reusing it");
    if (request->steps <= 0 || !std::isfinite(request->cfg_scale) ||
        !std::isfinite(request->generation_tail_padding_seconds) ||
        !std::isfinite(request->continuation_tail_padding_seconds) ||
        request->generation_tail_padding_seconds < 0.0f ||
        request->continuation_tail_padding_seconds < 0.0f)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid steps, CFG, or tail padding");
    const char* distribution = v1_distribution_name(request->distribution_shift);
    if (!distribution)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "unknown distribution shift");
    for (float value : request->distribution_shift_params)
        if (!std::isfinite(value))
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                           "distribution shift parameters must be finite");
    float distribution_params[4];
    std::copy(std::begin(request->distribution_shift_params),
              std::end(request->distribution_shift_params),
              std::begin(distribution_params));
    if (std::all_of(std::begin(distribution_params), std::end(distribution_params),
                    [](float value) { return value == 0.0f; }))
        sa3::dist_shift_defaults(distribution, distribution_params[0], distribution_params[1],
                                distribution_params[2], distribution_params[3]);
    if (request->residency != SA3_RESIDENCY_FRUGAL_V1 &&
        request->residency != SA3_RESIDENCY_RESIDENT_V1)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "unknown residency mode");

    int target_samples = 0;
    int source_samples_44100 = 0;
    int frames = 1;
    const bool needs_input = request->operation == SA3_OPERATION_TRANSFORM_V1 ||
                             request->operation == SA3_OPERATION_CONTINUE_V1;
    if (request->operation == SA3_OPERATION_GENERATE_V1) {
        if (!seconds_to_samples(request->duration_seconds, target_samples))
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "generation duration is out of range");
        frames = std::max(1, (target_samples + 4095) / 4096);
        if (frames & 1) ++frames;
    } else if (needs_input) {
        if (!has_v1_size(&request->input_audio, SA3_AUDIO_VIEW_V1_MIN_SIZE) || !request->input_audio.samples ||
            request->input_audio.n_samples == 0 || request->input_audio.n_channels == 0 ||
            request->input_audio.sample_rate == 0 ||
            request->input_audio.n_samples > (uint64_t)std::numeric_limits<int>::max() ||
            request->input_audio.n_channels > (uint32_t)std::numeric_limits<int>::max())
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "valid input audio is required");
        if (request->input_audio.layout != SA3_AUDIO_PLANAR_V1 &&
            request->input_audio.layout != SA3_AUDIO_INTERLEAVED_V1)
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "unknown input audio layout");
        if (!std::isfinite(request->transform_noise_level) ||
            request->transform_noise_level < 0.0f || request->transform_noise_level > 1.0f)
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                           "transform noise level must be finite and in [0, 1]");
        const double resampled = std::round((double)request->input_audio.n_samples * 44100.0 /
                                            (double)request->input_audio.sample_rate);
        if (resampled < 1.0 || resampled > (double)std::numeric_limits<int>::max())
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "input audio duration is out of range");
        source_samples_44100 = (int)resampled;
        target_samples = source_samples_44100;
        if (request->operation == SA3_OPERATION_CONTINUE_V1) {
            int added_samples = 0;
            if (!seconds_to_samples(request->duration_seconds, added_samples) ||
                added_samples > std::numeric_limits<int>::max() - source_samples_44100)
                return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "continuation duration is out of range");
            target_samples += added_samples;
        }
    } else {
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "unknown generation operation");
    }

    if (request->encode_chunk_size < 0 || request->encode_overlap < 0 ||
        (request->encode_chunk_size > 0 && request->encode_overlap >= request->encode_chunk_size) ||
        request->decode_chunk_size < 0 || request->decode_overlap < 0 ||
        (request->decode_chunk_size > 0 && request->decode_overlap >= request->decode_chunk_size))
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid encode/decode chunk settings");

    if (!has_v1_size(&request->loudness, SA3_LOUDNESS_V1_MIN_SIZE) ||
        !has_v1_size(&request->continuation, SA3_CONTINUATION_V1_MIN_SIZE))
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "embedded loudness or continuation options are too small");
    // Every V1 value is explicit, so these map straight across: there is no "did the caller mean
    // zero or mean default" question left to answer, because request_init already answered it.
    sa3::LoudnessParams loudness;
    loudness.peak_normalize_enabled = request->loudness.peak_normalize != 0;
    loudness.peak_normalize_db      = request->loudness.peak_normalize_db;
    loudness.limiter_enabled        = request->loudness.limiter != 0;
    loudness.limiter_ceiling_db     = request->loudness.limiter_ceiling_db;
    loudness.limiter_knee           = request->loudness.limiter_knee;
    loudness.latent_rescale         = request->loudness.latent_rescale;
    loudness.latent_shift           = request->loudness.latent_shift;
    sa3::normalize_loudness_params(loudness);
    {
        std::string why;
        if (!sa3::validate_loudness_params(loudness, why))
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "loudness: " + why);
    }

    // Continuation options describe the splice, which only runs for Continue. Other operations
    // leave the library defaults in place rather than validating values they will never read.
    sa3::SpliceParams splice = sa3::splice_defaults_from_env();
    if (request->operation == SA3_OPERATION_CONTINUE_V1) {
        splice.enabled      = request->continuation.splice_source != 0;
        splice.mask_overlap = request->continuation.mask_overlap_seconds;
        splice.xfade        = request->continuation.crossfade_seconds;
        splice.gain_match   = request->continuation.gain_match != 0;
        std::string why;
        if (!sa3::validate_splice_params(splice, why))
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "splice: " + why);
    }

    std::vector<float> planar_input;
    const float* input_samples = needs_input ? request->input_audio.samples : nullptr;
    if (needs_input && request->input_audio.layout == SA3_AUDIO_INTERLEAVED_V1) {
        const size_t ns = (size_t)request->input_audio.n_samples;
        const size_t nc = (size_t)request->input_audio.n_channels;
        if (nc > std::numeric_limits<size_t>::max() / ns)
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "input audio is too large");
        try {
            planar_input.resize(ns * nc);
        } catch (const std::bad_alloc&) {
            return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory converting input audio");
        }
        for (size_t s = 0; s < ns; ++s)
            for (size_t c = 0; c < nc; ++c)
                planar_input[c * ns + s] = request->input_audio.samples[s * nc + c];
        input_samples = planar_input.data();
    }

    std::vector<const char*> adapter_names;
    std::vector<float> adapter_strengths;
    if (request->adapter_count > 0) {
        if (!request->adapters || request->adapter_stride < SA3_ADAPTER_V1_MIN_SIZE ||
            request->adapter_count > (uint32_t)std::numeric_limits<int>::max() ||
            request->adapter_count > std::numeric_limits<size_t>::max() / request->adapter_stride)
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid adapter array");
        try {
            adapter_names.reserve(request->adapter_count);
            adapter_strengths.reserve(request->adapter_count);
            const auto* bytes = reinterpret_cast<const unsigned char*>(request->adapters);
            for (uint32_t i = 0; i < request->adapter_count; ++i) {
                sa3_adapter_v1 adapter{};
                std::memcpy(&adapter, bytes + (size_t)i * request->adapter_stride,
                            std::min<size_t>(request->adapter_stride, sizeof(adapter)));
                if (adapter.size < SA3_ADAPTER_V1_MIN_SIZE || adapter.size > request->adapter_stride)
                    return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                                   "adapter entry " + std::to_string(i) + " has an invalid size");
                if (!adapter.path_or_name || !*adapter.path_or_name)
                    return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                                   "adapter entry " + std::to_string(i) + " has an empty path or name");
                if (!std::isfinite(adapter.strength))
                    return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                                   "adapter entry " + std::to_string(i) + " has a non-finite strength");
                adapter_names.push_back(adapter.path_or_name);
                adapter_strengths.push_back(adapter.strength);
            }
        } catch (const std::bad_alloc&) {
            return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory preparing adapters");
        }
    }

    V1CallbackBridge bridge{request};
    sa3::GenParams p;
    p.prompt = request->prompt ? request->prompt : "";
    if (request->negative_prompt) p.negative_prompt = request->negative_prompt;
    p.frames = frames;
    p.steps = request->steps;
    p.seed = sa3::pick_seed((long long)request->seed);
    p.cfg_scale = request->cfg_scale;
    // Tail headroom is the generation schedule's; a Continue gets its headroom through the
    // inpaint canvas below instead, so the sampler is told the piece does not end there.
    p.duration_padding_sec = request->operation == SA3_OPERATION_GENERATE_V1
        ? request->generation_tail_padding_seconds : 0.0f;
    p.target_n_samp = target_samples;
    p.keep_models = request->residency == SA3_RESIDENCY_RESIDENT_V1;
    p.dist_shift = distribution;
    p.ds_p1 = distribution_params[0];
    p.ds_p2 = distribution_params[1];
    p.ds_p3 = distribution_params[2];
    p.ds_p4 = distribution_params[3];
    p.loudness = loudness;
    p.splice = splice;
    if (request->encode_chunk_size > 0) {
        p.encode_chunk_size = request->encode_chunk_size;
        p.encode_overlap = request->encode_overlap;
    }
    if (request->decode_chunk_size > 0) {
        p.decode_chunk_size = request->decode_chunk_size;
        p.decode_overlap = request->decode_overlap;
    }

    if (needs_input) {
        const size_t total = (size_t)request->input_audio.n_samples *
                             (size_t)request->input_audio.n_channels;
        try {
            p.init_audio.assign(input_samples, input_samples + total);
        } catch (const std::bad_alloc&) {
            return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory copying input audio");
        }
        p.init_n_samp = (int)request->input_audio.n_samples;
        p.init_n_ch = (int)request->input_audio.n_channels;
        p.init_sample_rate = (int)request->input_audio.sample_rate;
        p.init_noise_level = request->transform_noise_level;
        if (request->operation == SA3_OPERATION_CONTINUE_V1) {
            // The window opens where the source ends; the splice pulls it back from there. The
            // canvas runs past the promised length by the continuation headroom so the model has
            // somewhere to go, and target_n_samp trims it back afterwards.
            p.inpaint_start = (float)((double)source_samples_44100 / 44100.0);
            const double canvas_samples = (double)target_samples +
                std::round((double)request->continuation_tail_padding_seconds * 44100.0);
            if (canvas_samples > (double)std::numeric_limits<int>::max())
                return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "continuation canvas is out of range");
            p.inpaint_end = (float)(canvas_samples / 44100.0);
        }
    }

    // A bare name resolves against adapters_dir; an existing path is taken as given.
    try {
        for (size_t i = 0; i < adapter_names.size(); ++i) {
            const std::string name = adapter_names[i];
            std::string path = std::filesystem::exists(name)
                             ? name
                             : sa3::resolve_one(context->adapters_dir, "lora-" + name + "-", ".gguf");
            if (path.empty())
                return fail_v1(error, SA3_STATUS_IO_ERROR_V1, "unknown adapter '" + name + "'");
            p.loras.push_back({path, adapter_strengths[i]});
        }
    } catch (const std::exception& e) {
        return fail_v1(error, SA3_STATUS_IO_ERROR_V1, e.what());
    }

    if (request->on_progress) {
        V1CallbackBridge* b = &bridge;
        p.on_progress = [b](const sa3::Progress& pr) {
            v1_progress_bridge(b, pr.stage, pr.step, pr.total, pr.fraction);
        };
    }
    if (request->should_cancel) {
        V1CallbackBridge* b = &bridge;
        p.should_cancel = [b]() { return v1_cancel_bridge(b) != 0; };
    }

    return run_generation(context, p, result, error);
}

sa3_status_v1 SA3_CALL v1_convert_lora(const sa3_lora_convert_v1* options,
                                        sa3_error_v1* error) {
    clear_v1_error(error);
    if (!has_v1_size(options, SA3_LORA_CONVERT_V1_MIN_SIZE) ||
        !options->safetensors_path || !options->output_gguf_path)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid LoRA conversion options");
    try {
        std::string why;
        // A null or empty json_path means the config is in the safetensors' own __metadata__,
        // which is where autoencoder adapters keep it -- they ship as one self-describing file.
        if (!sa3::convert_lora_safetensors(options->safetensors_path,
                                           options->json_path ? options->json_path : "",
                                           options->output_gguf_path, why))
            return fail_v1(error, SA3_STATUS_IO_ERROR_V1, why.empty() ? "LoRA conversion failed" : why);
        return SA3_STATUS_OK_V1;
    } catch (const std::exception& e) {
        return fail_v1(error, SA3_STATUS_IO_ERROR_V1, e.what());
    } catch (...) {
        return fail_v1(error, SA3_STATUS_INTERNAL_ERROR_V1, "unknown error converting LoRA");
    }
}

static void copy_field(char* dst, size_t cap, const std::string& src) {
    if (!dst || cap == 0) return;
    const size_t n = src.size() < cap - 1 ? src.size() : cap - 1;
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

void SA3_CALL v1_training_config_init(sa3_training_config_v1* config) {
    init_v1_struct(config);
    if (!config || config->size < SA3_TRAINING_CONFIG_V1_MIN_SIZE) return;
    config->steps = 10000;
    config->rank = 16;
    config->alpha = 0.0f;
    config->learning_rate = 1.0e-4f;
    config->frames = 512;
    config->duration_seconds = 0.0f;
    config->batch_size = 1;
    config->checkpoint_every = 500;
    config->cpu_threads = 0;
    config->pre_encode = 1;
    config->seed = 42;
    config->evict_text_encoder = 0;
    config->latents_cache = 1;
}

void SA3_CALL v1_training_callbacks_init(sa3_training_callbacks_v1* callbacks) {
    init_v1_struct(callbacks);
}

void SA3_CALL v1_training_result_init(sa3_training_result_v1* result) {
    init_v1_struct(result);
}

/// Carries the V1 callbacks into the trainer's own hook types, and remembers why host audio
/// failed. The error is recorded rather than thrown because the trainer reports failures through
/// its own return path; v1_training_run reads it back once the run unwinds.
struct V1TrainingBridge {
    const sa3_training_callbacks_v1* callbacks = nullptr;
    std::vector<float> audio_copy;
    std::string audio_error;
    sa3_status_v1 audio_status = SA3_STATUS_OK_V1;

    void fail(std::string message, sa3_status_v1 status, std::string& why) {
        audio_status = status;
        audio_error = std::move(message);
        why = audio_error;
    }

    /// False lets libsa3 read the file itself. That is both the NOT_HANDLED answer and what an
    /// error degrades to, after recording what went wrong.
    bool load_audio(const std::string& path, int sample_rate, int channels,
                    sa3::TrainAudio& out, std::string& why);
};

sa3_status_v1 v1_training_audio_error_status(sa3_status_v1 status) {
    switch (status) {
        case SA3_STATUS_INVALID_ARGUMENT_V1:
        case SA3_STATUS_UNSUPPORTED_ABI_V1:
        case SA3_STATUS_CANCELLED_V1:
        case SA3_STATUS_MODEL_ERROR_V1:
        case SA3_STATUS_IO_ERROR_V1:
        case SA3_STATUS_OUT_OF_MEMORY_V1:
        case SA3_STATUS_INTERNAL_ERROR_V1:
            return status;
        default:
            return SA3_STATUS_IO_ERROR_V1;
    }
}

bool V1TrainingBridge::load_audio(const std::string& path, int sample_rate, int channels,
                                  sa3::TrainAudio& out, std::string& why) {
    if (!callbacks || !callbacks->load_audio) return false;
    sa3_training_audio_buffer_v1 buffer{};
    buffer.size = sizeof(buffer);
    buffer.audio.size = sizeof(buffer.audio);
    v1_audio_view_init(&buffer.audio);
    sa3_error_v1 callback_error{};
    callback_error.size = sizeof(callback_error);
    v1_error_init(&callback_error);

    const sa3_training_audio_status_v1 supplied = callbacks->load_audio(
        callbacks->user, path.c_str(), (uint32_t)sample_rate, (uint32_t)channels,
        &buffer, &callback_error);
    if (supplied == SA3_TRAINING_AUDIO_NOT_HANDLED_V1) return false;
    if (supplied == SA3_TRAINING_AUDIO_ERROR_V1) {
        const char* begin = callback_error.message;
        const char* end = std::find(begin, begin + sizeof(callback_error.message), '\0');
        fail(begin != end ? std::string(begin, end) : "host failed to decode training audio",
             v1_training_audio_error_status(callback_error.code), why);
        return false;   // ERROR does not transfer ownership, so nothing to release
    }
    if (supplied != SA3_TRAINING_AUDIO_READY_V1) {
        fail("training audio callback returned an unknown status",
             SA3_STATUS_INVALID_ARGUMENT_V1, why);
        return false;
    }

    // READY transferred ownership, so release runs exactly once from here on -- including on
    // every validation and allocation failure below.
    const auto release = [&]() { callbacks->release_audio(callbacks->user, buffer.owner); };
    const sa3_audio_view_v1& audio = buffer.audio;
    if (!has_v1_size(&buffer, SA3_TRAINING_AUDIO_BUFFER_V1_MIN_SIZE) ||
        !has_v1_size(&audio, SA3_AUDIO_VIEW_V1_MIN_SIZE) || !audio.samples || audio.n_samples == 0 ||
        audio.n_samples > (uint64_t)std::numeric_limits<int>::max() ||
        audio.n_channels != (uint32_t)channels || audio.sample_rate != (uint32_t)sample_rate ||
        (audio.layout != SA3_AUDIO_PLANAR_V1 && audio.layout != SA3_AUDIO_INTERLEAVED_V1)) {
        fail("training audio callback returned an invalid or mismatched audio view",
             SA3_STATUS_INVALID_ARGUMENT_V1, why);
        release();
        return false;
    }
    const size_t ns = (size_t)audio.n_samples;
    const size_t nc = (size_t)audio.n_channels;
    if (nc > std::numeric_limits<size_t>::max() / ns) {
        fail("training audio callback returned too many samples",
             SA3_STATUS_INVALID_ARGUMENT_V1, why);
        release();
        return false;
    }
    try {
        audio_copy.resize(ns * nc);
    } catch (const std::bad_alloc&) {
        fail("out of memory copying training callback audio", SA3_STATUS_OUT_OF_MEMORY_V1, why);
        release();
        return false;
    }
    if (audio.layout == SA3_AUDIO_PLANAR_V1) {
        std::copy(audio.samples, audio.samples + ns * nc, audio_copy.begin());
    } else {
        for (size_t s = 0; s < ns; ++s)
            for (size_t c = 0; c < nc; ++c)
                audio_copy[c * ns + s] = audio.samples[s * nc + c];
    }
    release();

    out.samples = audio_copy;
    out.n_samples = (int)ns;
    out.n_channels = channels;
    out.sample_rate = sample_rate;
    return true;
}

sa3_status_v1 training_failure_status(int code, const char* message) {
    if (code == 1) return SA3_STATUS_INVALID_ARGUMENT_V1;
    if (message && (std::strstr(message, "required") || std::strstr(message, "invalid") ||
                    std::strstr(message, "must be")))
        return SA3_STATUS_INVALID_ARGUMENT_V1;
    if (message && (std::strstr(message, "dataset") || std::strstr(message, "file") ||
                    std::strstr(message, "path") || std::strstr(message, "directory")))
        return SA3_STATUS_IO_ERROR_V1;
    return SA3_STATUS_MODEL_ERROR_V1;
}

sa3_status_v1 SA3_CALL v1_training_run(const sa3_training_config_v1* config,
                                        const sa3_training_callbacks_v1* callbacks,
                                        sa3_training_result_v1* result,
                                        sa3_error_v1* error) {
    clear_v1_error(error);
    if (!has_v1_size(config, SA3_TRAINING_CONFIG_V1_MIN_SIZE) ||
        !has_v1_size(result, SA3_TRAINING_RESULT_V1_MIN_SIZE) ||
        (callbacks && !has_v1_size(callbacks, SA3_TRAINING_CALLBACKS_V1_MIN_SIZE)))
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "full V1 training config and result are required");
    if (callbacks && ((callbacks->load_audio == nullptr) !=
                      (callbacks->release_audio == nullptr)))
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "training load_audio and release_audio must be installed together");
    if (!config->dataset_dir || !*config->dataset_dir)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "training dataset_dir is required");
    if (config->steps <= 0 || config->rank <= 0 ||
        !std::isfinite(config->alpha) || config->alpha < 0.0f ||
        !std::isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
        config->frames <= 0 || !std::isfinite(config->duration_seconds) ||
        config->duration_seconds < 0.0f || config->batch_size <= 0 ||
        config->checkpoint_every < 0 || config->cpu_threads < 0 || config->seed < 0)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid V1 training configuration");

    const uint32_t result_size = result->size;
    std::memset(result, 0, std::min<size_t>(result_size, sizeof(*result)));
    result->size = result_size;

    try {
        sa3::TrainConfig tc;
        if (const char* models = std::getenv("SA3_MODELS_DIR"); models && *models) tc.models_dir = models;
        // The JSON config is applied first so the fields below win over it -- it is the escape
        // hatch for trainer options V1 does not name, not an override of the ones it does.
        if (config->config_path && *config->config_path) {
            std::string why;
            if (!sa3::train_apply_json_config(tc, config->config_path, why))
                return fail_v1(error, SA3_STATUS_IO_ERROR_V1, why);
        }
        const auto str = [](const char* value, std::string& into) {
            if (value && *value) into = value;
        };
        str(config->models_dir,             tc.models_dir);
        str(config->variant,                tc.model_variant);
        str(config->dit_encoding,           tc.encoding);
        str(config->text_encoder_encoding,  tc.text_encoding);
        str(config->autoencoder_encoding,   tc.ae_encoding);
        str(config->dataset_dir,            tc.dataset_dir);
        str(config->output_dir,             tc.output_dir);
        str(config->latents_dir,            tc.latents_dir);
        str(config->latents_cache_dir,      tc.latents_cache_dir);
        str(config->prompt_config_path,     tc.prompt_config_path);
        str(config->resume_path,            tc.resume_path);
        str(config->adapter_type,           tc.adapter_type);
        str(config->lora_scope,             tc.lora_scope);
        str(config->device,                 tc.device);

        // Every scalar is explicit in V1: config_init established the defaults, so a zero here is
        // the caller's zero and is copied through as one.
        tc.max_steps = config->steps;
        tc.rank = config->rank;
        tc.alpha = config->alpha;
        tc.learning_rate = config->learning_rate;
        tc.frames = config->frames;
        tc.duration_sec = config->duration_seconds;
        tc.batch_size = config->batch_size;
        tc.checkpoint_every = config->checkpoint_every;
        tc.cpu_threads = config->cpu_threads;
        tc.pre_encode = config->pre_encode != 0;
        tc.evict_text_encoder = config->evict_text_encoder != 0;
        tc.latents_cache = config->latents_cache != 0;
        tc.seed = (unsigned long long)config->seed;

        if (tc.dataset_dir.empty())
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "dataset_dir is required");
        if (tc.cpu_threads == 0) tc.cpu_threads = sa3::cpu_threads_from_env();
        sa3::train_finalize_defaults(tc);

        V1TrainingBridge bridge{};
        bridge.callbacks = callbacks;
        sa3::TrainHooks th;
        if (callbacks) {
            if (callbacks->on_log) {
                th.log = [&bridge](const std::string& line) {
                    bridge.callbacks->on_log(bridge.callbacks->user, line.c_str());
                };
            }
            if (callbacks->on_step) {
                th.on_step = [&bridge](const sa3::TrainStepReport& r) {
                    sa3_training_step_v1 step{};
                    step.size = sizeof(step);
                    step.epoch = r.epoch;
                    step.step = r.step;
                    step.max_steps = r.max_steps;
                    step.id = r.id.c_str();
                    step.prompt = r.prompt.c_str();
                    step.mask = r.mask.c_str();
                    step.timestep = r.t;
                    step.learning_rate = r.learning_rate;
                    step.loss = r.loss;
                    step.gradient_norm = r.grad_norm;
                    step.step_seconds = r.step_seconds;
                    step.cfg_dropped = r.cfg_dropped ? 1 : 0;
                    step.updated = r.updated ? 1 : 0;
                    step.generated_frames = r.n_gen;
                    step.context_frames = r.n_ctx;
                    bridge.callbacks->on_step(bridge.callbacks->user, &step);
                };
            }
            if (callbacks->should_cancel) {
                th.should_cancel = [&bridge]() {
                    return bridge.callbacks->should_cancel(bridge.callbacks->user) != 0;
                };
            }
            if (callbacks->load_audio) {
                th.load_audio = [&bridge](const sa3::TrainAudioCaptionPair& pair, int sr, int ch,
                                          sa3::TrainAudio& audio, std::string& why) {
                    return bridge.load_audio(pair.audio_path, sr, ch, audio, why);
                };
            }
        }
        if (config->command_line) th.command_line = config->command_line;

        sa3::TrainResult run;
        std::string why;
        const bool ok = sa3::run_training(tc, th, run, why);
        // A host decode failure is the more specific cause, so it outranks whatever the trainer
        // reported on its way out.
        if (!bridge.audio_error.empty())
            return fail_v1(error, bridge.audio_status, bridge.audio_error);
        if (!ok)
            return fail_v1(error, training_failure_status(3, why.c_str()),
                           why.empty() ? "training failed" : why);

        result->completed_steps = run.steps;
        result->cancelled = run.cancelled ? 1 : 0;
        result->mean_step_seconds = run.mean_step_seconds;
        copy_field(result->final_adapter, sizeof(result->final_adapter), run.final_adapter);
        copy_field(result->last_checkpoint, sizeof(result->last_checkpoint), run.last_adapter_checkpoint);
        copy_field(result->preview_command, sizeof(result->preview_command), run.preview_command);
        // CANCELLED still carries a valid result; the caller inspects final_adapter rather than
        // discarding it.
        if (result->cancelled)
            return fail_v1(error, SA3_STATUS_CANCELLED_V1, "training cancelled");
        return SA3_STATUS_OK_V1;
    } catch (const std::bad_alloc&) {
        return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory during training");
    } catch (const std::exception& e) {
        return fail_v1(error, SA3_STATUS_MODEL_ERROR_V1, e.what());
    } catch (...) {
        return fail_v1(error, SA3_STATUS_INTERNAL_ERROR_V1, "unknown error during training");
    }
}

const sa3_api_v1 k_api_v1 = {
    sizeof(sa3_api_v1),
    SA3_ABI_VERSION_1,
    v1_runtime_version,
    v1_error_init,
    v1_context_config_init,
    v1_request_init,
    v1_audio_view_init,
    v1_adapter_init,
    v1_loudness_init,
    v1_continuation_init,
    v1_result_init,
    v1_lora_convert_init,
    v1_context_create,
    v1_context_unload,
    v1_context_destroy,
    v1_generate,
    v1_result_free,
    v1_convert_lora,
    {nullptr}
};

const sa3_training_api_v1 k_training_api_v1 = {
    sizeof(sa3_training_api_v1),
    SA3_TRAINING_ABI_VERSION_1,
    v1_runtime_version,
    v1_error_init,
    v1_training_config_init,
    v1_training_callbacks_init,
    v1_training_result_init,
    v1_training_run,
    {nullptr}
};

} // namespace

extern "C" {

SA3_API const sa3_api_v1* SA3_CALL sa3_get_api(uint32_t abi_version) {
    return abi_version == SA3_ABI_VERSION_1 ? &k_api_v1 : nullptr;
}

SA3_API const sa3_training_api_v1* SA3_CALL sa3_get_training_api(uint32_t abi_version) {
    return abi_version == SA3_TRAINING_ABI_VERSION_1 ? &k_training_api_v1 : nullptr;
}

} // extern "C"
