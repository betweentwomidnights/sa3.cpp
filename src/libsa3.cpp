// libsa3 — C ABI implementation over sa3::Pipeline. See libsa3.h for the contract.
#define SA3_BUILD_DLL
#include "libsa3.h"
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

// The context owns the pipeline via a unique_ptr so sa3_unload() can drop the models (and their
// VRAM) while keeping the context alive; the next sa3_generate() lazily reloads from `paths`.
struct sa3_context {
    std::unique_ptr<sa3::Pipeline> pipe;
    sa3::ModelPaths paths;
    std::string adapters_dir;
    int cpu_threads = 0;
    std::string device;  // remembered so the frugal-reload path keeps the same backend
    // Everything computed after the sampler, for sa3_last_meta(). size stays 0 until the first
    // successful generate, which is how the getter knows there is nothing to report yet.
    sa3_meta last_meta{};
};

static void set_err(char* err, int n, const std::string& m) {
    if (err && n > 0) { std::strncpy(err, m.c_str(), (size_t)n - 1); err[n - 1] = '\0'; }
}

static bool apply_init_audio(const sa3_init_audio& in, sa3::GenParams& p, std::string& err,
                             bool explicit_values = false) {
    if (in.mode == SA3_INIT_AUDIO_NONE) return true;
    if (in.mode != SA3_INIT_AUDIO_A2A && in.mode != SA3_INIT_AUDIO_INPAINT) {
        err = "init_audio.mode must be SA3_INIT_AUDIO_NONE, SA3_INIT_AUDIO_A2A, or SA3_INIT_AUDIO_INPAINT";
        return false;
    }
    if (!in.samples || in.n_samp <= 0 || in.n_ch <= 0 || in.sample_rate <= 0) {
        err = "init_audio requires non-null planar samples, n_samp > 0, n_ch > 0, and sample_rate > 0";
        return false;
    }
    if (!std::isfinite(in.init_noise_level) || !std::isfinite(in.inpaint_start) || !std::isfinite(in.inpaint_end)) {
        err = "init_audio fields must be finite";
        return false;
    }
    const int64_t total = (int64_t)in.n_samp * (int64_t)in.n_ch;
    if (total <= 0 || (uint64_t)total > (uint64_t)(std::numeric_limits<size_t>::max() / sizeof(float))) {
        err = "init_audio sample count is too large";
        return false;
    }

    p.init_audio.assign(in.samples, in.samples + (size_t)total);
    p.init_n_samp = in.n_samp;
    p.init_n_ch = in.n_ch;
    p.init_sample_rate = in.sample_rate;
    p.init_noise_level = explicit_values ? in.init_noise_level
                                         : (in.init_noise_level > 0.0f ? in.init_noise_level : 0.85f);
    if (in.mode == SA3_INIT_AUDIO_INPAINT) {
        p.inpaint_start = in.inpaint_start;
        p.inpaint_end = in.inpaint_end;
    }
    return true;
}

static int sa3_generate_impl(sa3_context* ctx, const sa3_request* req, const sa3_request_ex* req_ex,
                             const sa3_splice* splice_override, sa3_audio* out,
                             char* err, int err_len, int target_n_samp = 0,
                             bool explicit_values = false) {
    if (!ctx || !req || !out) { set_err(err, err_len, "null argument"); return 1; }
    out->samples = nullptr; out->n_samp = 0; out->n_ch = 0; out->sample_rate = 0; out->seed = 0;
    try {
        if (!ctx->pipe) {   // reload after sa3_unload()
            ctx->pipe = std::make_unique<sa3::Pipeline>();
            ctx->pipe->load(ctx->paths, ctx->cpu_threads, ctx->device.empty() ? nullptr : ctx->device.c_str());
        }
        sa3::GenParams p;
        p.prompt = req->prompt ? req->prompt : "";
        if (req->negative_prompt) p.negative_prompt = req->negative_prompt;
        p.frames = explicit_values ? req->frames : (req->frames > 0 ? req->frames : 128);
        p.steps  = explicit_values ? req->steps  : (req->steps > 0 ? req->steps : 8);
        p.seed   = sa3::pick_seed((long long)req->seed);
        p.cfg_scale = explicit_values ? req->cfg_scale
                                      : (req->cfg_scale != 0.0f ? req->cfg_scale : 1.0f);
        p.duration_padding_sec = explicit_values ? req->duration_padding_sec
                                                 : (req->duration_padding_sec >= 0.0f ? req->duration_padding_sec : 6.0f);
        p.target_n_samp = target_n_samp;
        p.keep_models = req->keep_models != 0;

        // distribution shift: NULL/"" -> "LogSNR"; per-type defaults, overridden by dist_shift_params if any set.
        p.dist_shift = (req->dist_shift && *req->dist_shift) ? req->dist_shift : "LogSNR";
        sa3::dist_shift_defaults(p.dist_shift, p.ds_p1, p.ds_p2, p.ds_p3, p.ds_p4);
        {
            const float* dp = req->dist_shift_params;
            if (explicit_values || dp[0] != 0.0f || dp[1] != 0.0f || dp[2] != 0.0f || dp[3] != 0.0f) {
                p.ds_p1 = dp[0]; p.ds_p2 = dp[1]; p.ds_p3 = dp[2]; p.ds_p4 = dp[3];
            }
        }
        if (req->loudness.set) {              // per-request loudness (incl. raw: peak_normalize=0, limiter=0)
            sa3::LoudnessParams lp;
            lp.peak_normalize_enabled = req->loudness.peak_normalize != 0;
            lp.peak_normalize_db      = req->loudness.peak_normalize_db;
            lp.limiter_enabled        = req->loudness.limiter != 0;
            lp.limiter_ceiling_db     = req->loudness.limiter_ceiling_db;
            lp.limiter_knee           = explicit_values ? req->loudness.limiter_knee
                                                        : (req->loudness.limiter_knee > 0.0f ? req->loudness.limiter_knee : lp.limiter_knee);
            lp.latent_rescale         = explicit_values ? req->loudness.latent_rescale
                                                        : (req->loudness.latent_rescale > 0.0f ? req->loudness.latent_rescale : 1.0f);
            lp.latent_shift           = req->loudness.latent_shift;
            sa3::normalize_loudness_params(lp);
            std::string lerr;
            if (!sa3::validate_loudness_params(lp, lerr)) { set_err(err, err_len, "loudness: " + lerr); return 4; }
            p.loudness = lp;
        } else {
            p.loudness = sa3::loudness_defaults_from_env();   // gary4local defaults + SA3_* env overrides
        }
        p.splice = sa3::splice_defaults_from_env();           // ditto, SA3_CONTINUE_* (req_ex may override)

        if (req_ex) {
            std::string ierr;
            if (!apply_init_audio(req_ex->init_audio, p, ierr, explicit_values)) { set_err(err, err_len, ierr); return 5; }
            if (req_ex->encode_chunk_size > 0) {
                p.encode_chunk_size = req_ex->encode_chunk_size;
                p.encode_overlap = explicit_values ? req_ex->encode_overlap
                                                   : (req_ex->encode_overlap > 0 ? req_ex->encode_overlap : 32);
            }
            if (req_ex->decode_chunk_size > 0) {
                p.decode_chunk_size = req_ex->decode_chunk_size;
                p.decode_overlap = explicit_values ? req_ex->decode_overlap
                                                   : (req_ex->decode_overlap > 0 ? req_ex->decode_overlap : 32);
            }
            if (splice_override && splice_override->set) { // per-request splice (incl. old behaviour: splice=0)
                sa3::SpliceParams sp;
                sp.enabled    = splice_override->splice != 0;
                sp.gain_match = splice_override->gain_match != 0;
                // 0 is a meaningful value for both (no pullback / no crossfade), so only a negative
                // falls back to the default rather than the usual 0-means-default convention.
                sp.mask_overlap = splice_override->mask_overlap >= 0.0f ? splice_override->mask_overlap : sp.mask_overlap;
                sp.xfade        = splice_override->xfade        >= 0.0f ? splice_override->xfade        : sp.xfade;
                std::string serr;
                if (!sa3::validate_splice_params(sp, serr)) { set_err(err, err_len, "splice: " + serr); return 6; }
                p.splice = sp;
            }
            if (req_ex->should_cancel) {
                const sa3_cancel_cb cb = req_ex->should_cancel;
                void* user = req_ex->cancel_user;
                p.should_cancel = [cb, user]() { return cb(user) != 0; };
            }
        }

        for (int i = 0; i < req->n_loras; i++) {
            const std::string name = (req->lora_names && req->lora_names[i]) ? req->lora_names[i] : "";
            if (name.empty()) continue;
            std::string path = std::filesystem::exists(name)
                             ? name : sa3::resolve_one(ctx->adapters_dir, "lora-" + name + "-", ".gguf");
            if (path.empty()) { set_err(err, err_len, "unknown lora '" + name + "'"); return 2; }
            const float s = req->lora_strengths ? req->lora_strengths[i] : 1.0f;
            p.loras.push_back({path, s});
        }
        if (req->on_progress) {
            const sa3_progress_cb cb = req->on_progress; void* user = req->user;
            p.on_progress = [cb, user](const sa3::Progress& pr) { cb(user, pr.stage, pr.step, pr.total, pr.fraction); };
        }

        sa3::GenResult r = ctx->pipe->generate(p);
        const size_t n = (size_t)r.n_samp * r.n_ch;
        out->samples = (float*)std::malloc(n * sizeof(float));
        if (!out->samples) { set_err(err, err_len, "out of memory"); return 3; }
        std::memcpy(out->samples, r.samples.data(), n * sizeof(float));
        out->n_samp = r.n_samp; out->n_ch = r.n_ch; out->sample_rate = r.sample_rate;
        out->seed = p.seed;

        sa3_meta m{};
        m.size = (uint32_t)sizeof(sa3_meta);
        m.seed = p.seed;
        m.decoded_peak                = r.loudness.decoded_peak;
        m.peak_normalize_gain_set     = r.loudness.peak_normalize_gain_set ? 1 : 0;
        m.peak_normalize_gain         = r.loudness.peak_normalize_gain;
        m.limiter_limited_fraction_set= r.loudness.limiter_limited_fraction_set ? 1 : 0;
        m.limiter_limited_fraction    = r.loudness.limiter_limited_fraction;
        m.safety_gain_set             = r.loudness.safety_gain_set ? 1 : 0;
        m.safety_gain                 = r.loudness.safety_gain;
        m.final_peak                  = r.loudness.final_peak;
        m.latent_factor               = r.loudness.latent_factor;
        m.splice_applied              = r.splice.applied ? 1 : 0;
        m.splice_end_seconds          = r.splice.splice_end_seconds;
        m.splice_xfade_applied        = r.splice.xfade_applied;
        m.splice_gain                 = r.splice.gain;
        m.mask_start_seconds          = r.splice.mask_start_seconds;
        m.mask_overlap_applied        = r.splice.mask_overlap_applied;
        ctx->last_meta = m;
        return 0;
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return 10; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return 10; }
}

static sa3_context* sa3_init_impl(const sa3_config* cfg, int cpu_threads, const char* device,
                                  const char* text_encoding, const char* ae_encoding,
                                  char* err, int err_len) {
    try {
        if (cpu_threads < 0) { set_err(err, err_len, "cpu_threads must be positive"); return nullptr; }
        std::string models_dir = cfg && cfg->models_dir ? cfg->models_dir : "";
        if (models_dir.empty()) { const char* e = std::getenv("SA3_MODELS_DIR"); models_dir = (e && *e) ? e : "models"; }
        const std::string variant  = cfg && cfg->variant  ? cfg->variant  : "medium";
        const std::string encoding = cfg && cfg->encoding ? cfg->encoding : "f16";
        const std::string adir     = cfg && cfg->adapters_dir ? cfg->adapters_dir : models_dir;

        const std::string tenc = text_encoding ? text_encoding : "";
        const std::string aenc = ae_encoding ? ae_encoding : "";

        sa3::ModelPaths mp; std::string rerr;
        if (!sa3::ModelPaths::resolve(models_dir, variant, encoding, tenc, aenc, mp, rerr)) { set_err(err, err_len, rerr); return nullptr; }

        auto ctx = std::make_unique<sa3_context>();
        ctx->paths = mp;
        ctx->adapters_dir = adir;
        ctx->cpu_threads = cpu_threads;
        if (device) ctx->device = device;
        ctx->pipe = std::make_unique<sa3::Pipeline>();
        ctx->pipe->load(mp, ctx->cpu_threads, device);
        return ctx.release();
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return nullptr; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return nullptr; }
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
bool has_v1_size(const T* value) {
    return value && value->size >= sizeof(T);
}

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
    if (audio && audio->size >= sizeof(*audio)) audio->layout = SA3_AUDIO_PLANAR_V1;
}

void SA3_CALL v1_request_init(sa3_request_v1* request) {
    init_v1_struct(request);
    if (!request || request->size < sizeof(*request)) return;
    request->operation = SA3_OPERATION_GENERATE_V1;
    request->duration_seconds = 12.0;
    request->steps = 8;
    request->seed = -1;
    request->cfg_scale = 1.0f;
    request->distribution_shift = SA3_DISTRIBUTION_LOGSNR_V1;
    request->distribution_shift_params[0] = 2000.0f;
    request->distribution_shift_params[1] = -6.2f;
    request->distribution_shift_params[2] = 0.0f;
    request->distribution_shift_params[3] = 2.0f;
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
    if (adapter && adapter->size >= sizeof(*adapter)) adapter->strength = 1.0f;
}

void SA3_CALL v1_loudness_init(sa3_loudness_v1* loudness) {
    init_v1_struct(loudness);
    if (!loudness || loudness->size < sizeof(*loudness)) return;
    loudness->peak_normalize = 1;
    loudness->peak_normalize_db = 2.0f;
    loudness->limiter = 1;
    loudness->limiter_ceiling_db = -0.3f;
    loudness->limiter_knee = 0.8f;
    loudness->latent_rescale = 1.0f;
}

void SA3_CALL v1_continuation_init(sa3_continuation_v1* continuation) {
    init_v1_struct(continuation);
    if (!continuation || continuation->size < sizeof(*continuation)) return;
    continuation->splice_source = 1;
    continuation->mask_overlap_seconds = 0.2f;
    continuation->crossfade_seconds = 0.03f;
    continuation->gain_match = 1;
}

void SA3_CALL v1_result_init(sa3_result_v1* result) {
    init_v1_struct(result);
    if (result && result->size >= sizeof(*result)) result->layout = SA3_AUDIO_PLANAR_V1;
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

sa3_status_v1 status_from_legacy(int code, const char* message) {
    if (code == 0) return SA3_STATUS_OK_V1;
    if (message && std::strstr(message, "cancelled")) return SA3_STATUS_CANCELLED_V1;
    if (code == 1 || code == 4 || code == 5 || code == 6) return SA3_STATUS_INVALID_ARGUMENT_V1;
    if (code == 2) return SA3_STATUS_IO_ERROR_V1;
    if (code == 3) return SA3_STATUS_OUT_OF_MEMORY_V1;
    return SA3_STATUS_MODEL_ERROR_V1;
}

sa3_status_v1 fail_v1(sa3_error_v1* error, sa3_status_v1 status, const std::string& message) {
    set_v1_error(error, status, message);
    return status;
}

sa3_status_v1 SA3_CALL v1_context_create(const sa3_context_config_v1* config,
                                          sa3_context** out_context,
                                          sa3_error_v1* error) {
    clear_v1_error(error);
    if (!has_v1_size(config) || !out_context)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "context config is too small or out_context is null");
    *out_context = nullptr;
    if (config->cpu_threads < 0)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "cpu_threads must be non-negative");

    sa3_config_ex legacy{};
    legacy.config.models_dir = config->models_dir;
    legacy.config.adapters_dir = config->adapters_dir;
    legacy.config.variant = config->variant;
    legacy.config.encoding = config->dit_encoding;
    legacy.cpu_threads = config->cpu_threads;
    legacy.device = config->device;
    legacy.text_encoder_encoding = config->text_encoder_encoding;
    legacy.autoencoder_encoding = config->autoencoder_encoding;
    char message[1024]{};
    *out_context = sa3_init_impl(&legacy.config, legacy.cpu_threads, legacy.device,
                                 legacy.text_encoder_encoding, legacy.autoencoder_encoding,
                                 message, (int)sizeof(message));
    if (!*out_context)
        return fail_v1(error, SA3_STATUS_MODEL_ERROR_V1,
                       message[0] ? message : "failed to create context");
    return SA3_STATUS_OK_V1;
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

sa3_status_v1 SA3_CALL v1_generate(sa3_context* context,
                                    const sa3_request_v1* request,
                                    sa3_result_v1* result,
                                    sa3_error_v1* error) {
    clear_v1_error(error);
    if (!context || !has_v1_size(request) || !has_v1_size(result))
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
        if (!has_v1_size(&request->input_audio) || !request->input_audio.samples ||
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

    if (!has_v1_size(&request->loudness) || !has_v1_size(&request->continuation))
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1,
                       "embedded loudness or continuation options are too small");
    sa3_loudness loudness{};
    loudness.set = 1;
    loudness.peak_normalize = request->loudness.peak_normalize;
    loudness.peak_normalize_db = request->loudness.peak_normalize_db;
    loudness.limiter = request->loudness.limiter;
    loudness.limiter_ceiling_db = request->loudness.limiter_ceiling_db;
    loudness.limiter_knee = request->loudness.limiter_knee;
    loudness.latent_rescale = request->loudness.latent_rescale;
    loudness.latent_shift = request->loudness.latent_shift;

    sa3_splice splice{};
    splice.set = 1;
    splice.splice = request->continuation.splice_source;
    splice.mask_overlap = request->continuation.mask_overlap_seconds;
    splice.xfade = request->continuation.crossfade_seconds;
    splice.gain_match = request->continuation.gain_match;

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
        if (!request->adapters || request->adapter_stride < sizeof(sa3_adapter_v1) ||
            request->adapter_count > (uint32_t)std::numeric_limits<int>::max())
            return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid adapter array");
        try {
            adapter_names.reserve(request->adapter_count);
            adapter_strengths.reserve(request->adapter_count);
            const auto* bytes = reinterpret_cast<const unsigned char*>(request->adapters);
            for (uint32_t i = 0; i < request->adapter_count; ++i) {
                sa3_adapter_v1 adapter{};
                std::memcpy(&adapter, bytes + (size_t)i * request->adapter_stride, sizeof(adapter));
                if (adapter.size < sizeof(adapter) || !adapter.path_or_name || !*adapter.path_or_name ||
                    !std::isfinite(adapter.strength))
                    return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid adapter entry");
                adapter_names.push_back(adapter.path_or_name);
                adapter_strengths.push_back(adapter.strength);
            }
        } catch (const std::bad_alloc&) {
            return fail_v1(error, SA3_STATUS_OUT_OF_MEMORY_V1, "out of memory preparing adapters");
        }
    }

    V1CallbackBridge bridge{request};
    sa3_request_v2 legacy{};
    legacy.size = sizeof(legacy);
    legacy.request.request.prompt = request->prompt;
    legacy.request.request.negative_prompt = request->negative_prompt;
    legacy.request.request.frames = frames;
    legacy.request.request.steps = request->steps;
    legacy.request.request.seed = request->seed;
    legacy.request.request.cfg_scale = request->cfg_scale;
    legacy.request.request.duration_padding_sec = request->operation == SA3_OPERATION_GENERATE_V1
        ? request->generation_tail_padding_seconds : 0.0f;
    legacy.request.request.keep_models = request->residency == SA3_RESIDENCY_RESIDENT_V1;
    legacy.request.request.loudness = loudness;
    legacy.request.request.n_loras = (int)adapter_names.size();
    legacy.request.request.lora_names = adapter_names.empty() ? nullptr : adapter_names.data();
    legacy.request.request.lora_strengths = adapter_strengths.empty() ? nullptr : adapter_strengths.data();
    legacy.request.request.dist_shift = distribution;
    std::copy(std::begin(request->distribution_shift_params),
              std::end(request->distribution_shift_params),
              std::begin(legacy.request.request.dist_shift_params));
    if (request->on_progress) {
        legacy.request.request.on_progress = v1_progress_bridge;
        legacy.request.request.user = &bridge;
    }
    if (needs_input) {
        legacy.request.init_audio.mode = request->operation == SA3_OPERATION_TRANSFORM_V1
            ? SA3_INIT_AUDIO_A2A : SA3_INIT_AUDIO_INPAINT;
        legacy.request.init_audio.samples = input_samples;
        legacy.request.init_audio.n_samp = (int)request->input_audio.n_samples;
        legacy.request.init_audio.n_ch = (int)request->input_audio.n_channels;
        legacy.request.init_audio.sample_rate = (int)request->input_audio.sample_rate;
        legacy.request.init_audio.init_noise_level = request->transform_noise_level;
        if (request->operation == SA3_OPERATION_CONTINUE_V1) {
            legacy.request.init_audio.inpaint_start = (float)((double)source_samples_44100 / 44100.0);
            const double canvas_samples = (double)target_samples +
                std::round((double)request->continuation_tail_padding_seconds * 44100.0);
            if (canvas_samples > (double)std::numeric_limits<int>::max())
                return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "continuation canvas is out of range");
            legacy.request.init_audio.inpaint_end = (float)(canvas_samples / 44100.0);
            legacy.splice = splice;
        }
    }
    legacy.request.encode_chunk_size = request->encode_chunk_size;
    legacy.request.encode_overlap = request->encode_overlap;
    legacy.request.decode_chunk_size = request->decode_chunk_size;
    legacy.request.decode_overlap = request->decode_overlap;
    if (request->should_cancel) {
        legacy.request.should_cancel = v1_cancel_bridge;
        legacy.request.cancel_user = &bridge;
    }

    sa3_audio audio{};
    char message[1024]{};
    const int rc = sa3_generate_impl(context, &legacy.request.request, &legacy.request,
                                     request->operation == SA3_OPERATION_CONTINUE_V1 ? &legacy.splice : nullptr,
                                     &audio, message, (int)sizeof(message), target_samples, true);
    if (rc != 0) {
        const sa3_status_v1 status = status_from_legacy(rc, message);
        return fail_v1(error, status, message[0] ? message : "generation failed");
    }

    const uint32_t result_size = result->size;
    std::memset(result, 0, sizeof(*result));
    result->size = result_size;
    result->samples = audio.samples;
    result->n_samples = (uint64_t)audio.n_samp;
    result->n_channels = (uint32_t)audio.n_ch;
    result->sample_rate = (uint32_t)audio.sample_rate;
    result->layout = SA3_AUDIO_PLANAR_V1;
    result->seed = audio.seed;
    const sa3_meta& meta = context->last_meta;
    result->decoded_peak = meta.decoded_peak;
    result->peak_normalize_gain_set = meta.peak_normalize_gain_set;
    result->peak_normalize_gain = meta.peak_normalize_gain;
    result->limiter_limited_fraction_set = meta.limiter_limited_fraction_set;
    result->limiter_limited_fraction = meta.limiter_limited_fraction;
    result->safety_gain_set = meta.safety_gain_set;
    result->safety_gain = meta.safety_gain;
    result->final_peak = meta.final_peak;
    result->latent_factor = meta.latent_factor;
    result->splice_applied = meta.splice_applied;
    result->splice_end_seconds = meta.splice_end_seconds;
    result->splice_crossfade_seconds = meta.splice_xfade_applied;
    result->splice_gain = meta.splice_gain;
    result->mask_start_seconds = meta.mask_start_seconds;
    result->mask_overlap_seconds = meta.mask_overlap_applied;
    return SA3_STATUS_OK_V1;
}

sa3_status_v1 SA3_CALL v1_convert_lora(const sa3_lora_convert_v1* options,
                                        sa3_error_v1* error) {
    clear_v1_error(error);
    if (!has_v1_size(options) || !options->safetensors_path || !options->output_gguf_path)
        return fail_v1(error, SA3_STATUS_INVALID_ARGUMENT_V1, "invalid LoRA conversion options");
    char message[1024]{};
    const int rc = sa3_convert_lora(options->safetensors_path, options->json_path,
                                    options->output_gguf_path, message, (int)sizeof(message));
    if (rc != 0) {
        const sa3_status_v1 status = status_from_legacy(rc, message);
        return fail_v1(error, status, message[0] ? message : "LoRA conversion failed");
    }
    return SA3_STATUS_OK_V1;
}

const sa3_api_v1 k_api_v1 = {
    sizeof(sa3_api_v1),
    SA3_ABI_VERSION_1,
    sa3_version,
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

} // namespace

extern "C" {

SA3_API const sa3_api_v1* SA3_CALL sa3_get_api(uint32_t abi_version) {
    return abi_version == SA3_ABI_VERSION_1 ? &k_api_v1 : nullptr;
}

SA3_API sa3_context* sa3_init(const sa3_config* cfg, char* err, int err_len) {
    return sa3_init_impl(cfg, 0, nullptr, nullptr, nullptr, err, err_len);
}

SA3_API sa3_context* sa3_init_ex(const sa3_config_ex* cfg, char* err, int err_len) {
    return sa3_init_impl(cfg ? &cfg->config : nullptr, cfg ? cfg->cpu_threads : 0,
                         cfg ? cfg->device : nullptr,
                         cfg ? cfg->text_encoder_encoding : nullptr,
                         cfg ? cfg->autoencoder_encoding : nullptr, err, err_len);
}

SA3_API int sa3_generate(sa3_context* ctx, const sa3_request* req, sa3_audio* out, char* err, int err_len) {
    return sa3_generate_impl(ctx, req, nullptr, nullptr, out, err, err_len);
}

SA3_API int sa3_generate_ex(sa3_context* ctx, const sa3_request_ex* req, sa3_audio* out, char* err, int err_len) {
    if (!req) { set_err(err, err_len, "null argument"); return 1; }
    return sa3_generate_impl(ctx, &req->request, req, nullptr, out, err, err_len);
}

SA3_API int sa3_generate_v2(sa3_context* ctx, const sa3_request_v2* req, sa3_audio* out, char* err, int err_len) {
    if (!req) { set_err(err, err_len, "null v2 request"); return 1; }
    const size_t required = offsetof(sa3_request_v2, splice) + sizeof(sa3_splice);
    if (req->size < required) { set_err(err, err_len, "v2 request size is too small"); return 1; }
    return sa3_generate_impl(ctx, &req->request.request, &req->request, &req->splice,
                             out, err, err_len);
}

SA3_API void sa3_free_audio(sa3_audio* a) {
    if (a && a->samples) { std::free(a->samples); a->samples = nullptr; a->n_samp = 0; }
}

// Copy back only what BOTH sides know about: min(the caller's size, ours). A caller on an older
// header gets the prefix it understands; one on a newer header sees a smaller size written back and
// knows the tail it declared is untouched. This is what lets sa3_meta grow forever without an ABI
// break, and without a new entry point per revision.
SA3_API int sa3_last_meta(sa3_context* ctx, sa3_meta* out) {
    if (!ctx || !out) return 1;
    const uint32_t want = out->size;
    if (want < sizeof(uint32_t)) return 2;              // too small to even carry the size back
    if (ctx->last_meta.size == 0) return 3;             // no generation has completed on this ctx
    const uint32_t n = want < sizeof(sa3_meta) ? want : (uint32_t)sizeof(sa3_meta);
    std::memcpy(out, &ctx->last_meta, n);
    out->size = n;
    return 0;
}

SA3_API void sa3_unload(sa3_context* ctx) { if (ctx) ctx->pipe.reset(); }   // drop models; keep ctx

SA3_API void sa3_free(sa3_context* ctx) { delete ctx; }

SA3_API const char* sa3_version(void) { return "sa3.cpp libsa3 6 (ABI 1)"; }

SA3_API int sa3_convert_lora(const char* safetensors_path, const char* json_path,
                             const char* out_gguf_path, char* err, int err_len) {
    if (!safetensors_path || !out_gguf_path) { set_err(err, err_len, "null argument"); return 1; }
    try {
        std::string e;
        // NULL/"" json_path means "the config is in the safetensors' own __metadata__", which is
        // where save_lora_safetensors puts it for autoencoder adapters -- they ship as one file.
        if (!sa3::convert_lora_safetensors(safetensors_path, json_path ? json_path : "",
                                           out_gguf_path, e)) {
            set_err(err, err_len, e); return 2;
        }
        return 0;
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return 10; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return 10; }
}

static void copy_field(char* dst, size_t cap, const std::string& src) {
    if (!dst || cap == 0) return;
    const size_t n = src.size() < cap - 1 ? src.size() : cap - 1;
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

SA3_API int sa3_train(const sa3_train_config* cfg, const sa3_train_hooks* hooks,
                      sa3_train_result* out, char* err, int err_len) {
    if (!cfg) { set_err(err, err_len, "null argument"); return 1; }
    if (out) std::memset(out, 0, sizeof(*out));
    try {
        sa3::TrainConfig tc;
        if (const char* models = std::getenv("SA3_MODELS_DIR"); models && *models) tc.models_dir = models;

        // the config file first, so the explicit fields below win over it
        if (cfg->config_path && *cfg->config_path) {
            std::string jerr;
            if (!sa3::train_apply_json_config(tc, cfg->config_path, jerr)) { set_err(err, err_len, jerr); return 2; }
        }

        auto str = [](const char* v, std::string& dst) { if (v && *v) dst = v; };
        str(cfg->models_dir,     tc.models_dir);
        str(cfg->variant,        tc.model_variant);
        str(cfg->encoding,       tc.encoding);
        str(cfg->text_encoder_encoding, tc.text_encoding);
        str(cfg->autoencoder_encoding,  tc.ae_encoding);
        str(cfg->dataset_dir,    tc.dataset_dir);
        str(cfg->output_dir,     tc.output_dir);
        str(cfg->latents_dir,    tc.latents_dir);
        str(cfg->prompt_config_path, tc.prompt_config_path);
        str(cfg->resume_path,    tc.resume_path);
        str(cfg->adapter_type,   tc.adapter_type);
        str(cfg->lora_scope,     tc.lora_scope);
        str(cfg->latents_cache_dir, tc.latents_cache_dir);

        if (cfg->steps > 0)            tc.max_steps = cfg->steps;
        if (cfg->rank > 0)             tc.rank = cfg->rank;
        if (cfg->alpha > 0.0f)         tc.alpha = cfg->alpha;
        if (cfg->learning_rate > 0.0f) tc.learning_rate = cfg->learning_rate;
        if (cfg->frames > 0)           tc.frames = cfg->frames;
        if (cfg->duration_sec > 0.0f)  tc.duration_sec = cfg->duration_sec;
        if (cfg->batch_size > 0)       tc.batch_size = cfg->batch_size;
        if (cfg->checkpoint_every > 0) tc.checkpoint_every = cfg->checkpoint_every;
        if (cfg->checkpoint_every < 0) tc.checkpoint_every = 0;   // negative = no intermediate writes
        if (cfg->cpu_threads > 0)      tc.cpu_threads = cfg->cpu_threads;
        if (cfg->pre_encode)           tc.pre_encode = true;
        if (cfg->evict_text_encoder)   tc.evict_text_encoder = true;
        if (cfg->latents_cache < 0)    tc.latents_cache = false;
        if (cfg->seed != 0)            tc.seed = (unsigned long long)cfg->seed;

        if (tc.dataset_dir.empty()) { set_err(err, err_len, "dataset_dir is required"); return 2; }
        str(cfg->device, tc.device);
        if (tc.cpu_threads == 0) tc.cpu_threads = sa3::cpu_threads_from_env();
        sa3::train_finalize_defaults(tc);

        sa3::TrainHooks th;
        if (hooks) {
            void* user = hooks->user;
            if (hooks->on_log) {
                const sa3_train_log_cb cb = hooks->on_log;
                th.log = [cb, user](const std::string& line) { cb(user, line.c_str()); };
            }
            if (hooks->on_step) {
                const sa3_train_step_cb cb = hooks->on_step;
                th.on_step = [cb, user](const sa3::TrainStepReport& r) {
                    sa3_train_step s;
                    std::memset(&s, 0, sizeof(s));
                    s.epoch = r.epoch;
                    s.step = r.step;
                    s.max_steps = r.max_steps;
                    s.id = r.id.c_str();
                    s.prompt = r.prompt.c_str();
                    s.mask = r.mask.c_str();
                    s.t = r.t;
                    s.learning_rate = r.learning_rate;
                    s.loss = r.loss;
                    s.grad_norm = r.grad_norm;
                    s.step_seconds = r.step_seconds;
                    s.cfg_dropped = r.cfg_dropped ? 1 : 0;
                    s.updated = r.updated ? 1 : 0;
                    s.n_gen = r.n_gen;
                    s.n_ctx = r.n_ctx;
                    cb(user, &s);
                };
            }
            if (hooks->should_cancel) {
                const sa3_train_cancel_cb cb = hooks->should_cancel;
                th.should_cancel = [cb, user]() { return cb(user) != 0; };
            }
            if (hooks->load_audio) {
                const sa3_train_audio_cb cb = hooks->load_audio;
                th.load_audio = [cb, user](const sa3::TrainAudioCaptionPair& pair, int sr, int ch,
                                           sa3::TrainAudio& audio, std::string& lerr) {
                    const float* samples = nullptr;
                    int n_samp = 0;
                    if (cb(user, pair.audio_path.c_str(), sr, ch, &samples, &n_samp) == 0) return false;
                    if (!samples || n_samp <= 0) {
                        lerr = "load_audio returned no samples for " + pair.audio_path;
                        return false;
                    }
                    audio.samples.assign(samples, samples + (size_t)n_samp * (size_t)ch);
                    audio.n_samples = n_samp;
                    audio.n_channels = ch;
                    audio.sample_rate = sr;
                    return true;
                };
            }
            if (hooks->command_line) th.command_line = hooks->command_line;
        }

        sa3::TrainResult result;
        std::string terr;
        if (!sa3::run_training(tc, th, result, terr)) { set_err(err, err_len, terr); return 3; }
        if (out) {
            out->steps = result.steps;
            out->cancelled = result.cancelled ? 1 : 0;
            out->mean_step_seconds = result.mean_step_seconds;
            copy_field(out->final_adapter, sizeof(out->final_adapter), result.final_adapter);
            copy_field(out->last_checkpoint, sizeof(out->last_checkpoint), result.last_adapter_checkpoint);
            copy_field(out->preview_command, sizeof(out->preview_command), result.preview_command);
        }
        return 0;
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return 10; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return 10; }
}

} // extern "C"
