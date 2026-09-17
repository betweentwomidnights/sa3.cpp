/* libsa3 V1 — stable C ABI for embedding sa3.cpp.
 *
 * Resolve the single sa3_get_api symbol, request SA3_ABI_VERSION_1, initialize
 * option structs through the returned table, and keep all library-owned audio
 * paired with result_free. Public data structs are size tagged. Top-level structs,
 * callback payloads, and strided array entries may grow by appending fields; types
 * embedded by value in another public struct are frozen for ABI V1.
 */
#ifndef LIBSA3_V1_H
#define LIBSA3_V1_H

#include <stddef.h>
#include <stdint.h>

#ifndef SA3_API
#  if defined(_WIN32) && defined(SA3_BUILD_DLL)
#    define SA3_API __declspec(dllexport)
#  else
#    define SA3_API
#  endif
#endif

#if defined(_WIN32)
#  define SA3_CALL __cdecl
#else
#  define SA3_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SA3_CONTEXT_DECLARED
#define SA3_CONTEXT_DECLARED
typedef struct sa3_context sa3_context;
#endif

#define SA3_ABI_VERSION_1 1u

typedef int32_t sa3_status_v1;
enum {
    SA3_STATUS_OK_V1                  = 0,
    SA3_STATUS_INVALID_ARGUMENT_V1    = 1,
    SA3_STATUS_UNSUPPORTED_ABI_V1     = 2,
    SA3_STATUS_CANCELLED_V1           = 3,
    SA3_STATUS_MODEL_ERROR_V1         = 4,
    SA3_STATUS_IO_ERROR_V1            = 5,
    SA3_STATUS_OUT_OF_MEMORY_V1       = 6,
    SA3_STATUS_INTERNAL_ERROR_V1      = 7
};

typedef int32_t sa3_operation_v1;
enum {
    SA3_OPERATION_GENERATE_V1  = 0,
    SA3_OPERATION_TRANSFORM_V1 = 1,
    SA3_OPERATION_CONTINUE_V1  = 2
};

typedef int32_t sa3_distribution_shift_v1;
enum {
    SA3_DISTRIBUTION_LOGSNR_V1 = 0,
    SA3_DISTRIBUTION_FLUX_V1   = 1,
    SA3_DISTRIBUTION_FULL_V1   = 2,
    SA3_DISTRIBUTION_NONE_V1   = 3
};

/* Samplers for SAT variants (stable-audio-tools checkpoints such as Foundation). SA3 variants
 * ignore this field. AUTO selects the variant's profile sampler. */
typedef int32_t sa3_sampler_v1;
enum {
    SA3_SAMPLER_AUTO_V1         = 0,
    SA3_SAMPLER_DPMPP_2M_SDE_V1 = 1,
    SA3_SAMPLER_DPMPP_3M_SDE_V1 = 2
};

typedef int32_t sa3_residency_v1;
enum {
    SA3_RESIDENCY_FRUGAL_V1   = 0,
    SA3_RESIDENCY_RESIDENT_V1 = 1
};

typedef int32_t sa3_audio_layout_v1;
enum {
    SA3_AUDIO_PLANAR_V1     = 0,
    SA3_AUDIO_INTERLEAVED_V1 = 1
};

typedef int32_t sa3_progress_stage_v1;
enum {
    SA3_PROGRESS_LOADING_V1  = 0,
    SA3_PROGRESS_ENCODING_V1 = 1,
    SA3_PROGRESS_SAMPLING_V1 = 2,
    SA3_PROGRESS_DECODING_V1 = 3,
    SA3_PROGRESS_DONE_V1     = 4,
    SA3_PROGRESS_OTHER_V1    = 5
};

typedef struct {
    uint32_t size;
    sa3_status_v1 code;
    char message[1024];
} sa3_error_v1;

typedef struct {
    uint32_t size;
    const char* models_dir;
    const char* adapters_dir;
    const char* variant;
    const char* dit_encoding;
    const char* text_encoder_encoding;
    const char* autoencoder_encoding;
    const char* device;
    int32_t cpu_threads;
} sa3_context_config_v1;

/* Non-owning input view. n_samples is per channel. Planar data is laid out as
 * samples[channel * n_samples + sample]. Interleaved input is also accepted. */
typedef struct {
    uint32_t size;
    const float* samples;
    uint64_t n_samples;
    uint32_t n_channels;
    uint32_t sample_rate;
    sa3_audio_layout_v1 layout;
    uint32_t reserved;
} sa3_audio_view_v1;

/* One adapter entry. request.adapter_stride must be sizeof(sa3_adapter_v1),
 * allowing a later ABI revision to append fields without changing array indexing. */
typedef struct {
    uint32_t size;
    const char* path_or_name;
    float strength;
} sa3_adapter_v1;

typedef struct {
    uint32_t size;
    int32_t peak_normalize;
    float peak_normalize_db;
    int32_t limiter;
    float limiter_ceiling_db;
    float limiter_knee;
    float latent_rescale;
    float latent_shift;
} sa3_loudness_v1;

typedef struct {
    uint32_t size;
    int32_t splice_source;
    float mask_overlap_seconds;
    float crossfade_seconds;
    int32_t gain_match;
} sa3_continuation_v1;

typedef struct {
    uint32_t size;
    sa3_progress_stage_v1 stage;
    const char* stage_name; /* valid only during the callback */
    int32_t step;
    int32_t total;
    float fraction;
} sa3_progress_v1;

typedef void (SA3_CALL *sa3_progress_callback_v1)(void* user, const sa3_progress_v1* progress);
typedef int32_t (SA3_CALL *sa3_cancel_callback_v1)(void* user);
typedef void (SA3_CALL *sa3_reserved_function_v1)(void);

/* duration_seconds means exact output duration for Generate and seconds to add
 * for Continue. Transform follows the input duration. libsa3 owns model-frame
 * rounding, SAME-S constraints, continuation headroom, splicing, and trimming.
 *
 * SAT variants (context variant "foundation-1", "foundation-1.2-keybeds", ...) support
 * Generate only, without adapters; distribution shift, tail padding, and chunk sizes do
 * not apply. They read the fields appended after callback_user when request.size covers
 * them. Set steps and cfg_scale explicitly: request_init's values are SA3 defaults. */
typedef struct {
    uint32_t size;
    sa3_operation_v1 operation;
    const char* prompt;
    const char* negative_prompt;
    double duration_seconds;
    int32_t steps;
    int64_t seed;
    float cfg_scale;
    sa3_distribution_shift_v1 distribution_shift;
    /* All zeroes select the built-in defaults for distribution_shift. Set all
     * four values to override that shift's defaults explicitly. */
    float distribution_shift_params[4];
    sa3_residency_v1 residency;

    /* These by-value option types are frozen for V1; extend sa3_request_v1 itself instead. */
    sa3_audio_view_v1 input_audio;
    float transform_noise_level;
    float generation_tail_padding_seconds;
    float continuation_tail_padding_seconds;

    const sa3_adapter_v1* adapters;
    uint32_t adapter_count;
    uint32_t adapter_stride;

    sa3_loudness_v1 loudness;
    sa3_continuation_v1 continuation;

    int32_t encode_chunk_size;
    int32_t encode_overlap;
    int32_t decode_chunk_size;
    int32_t decode_overlap;

    sa3_progress_callback_v1 on_progress;
    sa3_cancel_callback_v1 should_cancel;
    void* callback_user;

    /* Appended after the frozen V1 prefix; SAT variants only. Zero selects the profile
     * default for sampler and sigmas. conditioning_seconds_total zero means
     * ceil(duration_seconds); a larger value conditions a longer canvas that is then
     * cropped to duration_seconds (Foundation keybed chunks use this). */
    sa3_sampler_v1 sampler;
    float sigma_min;
    float sigma_max;
    double conditioning_seconds_start;
    double conditioning_seconds_total;
} sa3_request_v1;

/* Library-owned planar float audio plus generation metadata. Set size before
 * generate. result_free is safe on a zero-initialized or already-freed result. */
typedef struct {
    uint32_t size;
    float* samples;
    uint64_t n_samples;
    uint32_t n_channels;
    uint32_t sample_rate;
    sa3_audio_layout_v1 layout;
    uint64_t seed;

    float decoded_peak;
    int32_t peak_normalize_gain_set;
    float peak_normalize_gain;
    int32_t limiter_limited_fraction_set;
    float limiter_limited_fraction;
    int32_t safety_gain_set;
    float safety_gain;
    float final_peak;
    float latent_factor;

    int32_t splice_applied;
    float splice_end_seconds;
    float splice_crossfade_seconds;
    float splice_gain;
    float mask_start_seconds;
    float mask_overlap_seconds;
} sa3_result_v1;

typedef struct {
    uint32_t size;
    const char* safetensors_path;
    const char* json_path;
    const char* output_gguf_path;
} sa3_lora_convert_v1;

typedef struct sa3_api_v1 {
    uint32_t size;
    uint32_t abi_version;
    const char* (SA3_CALL *runtime_version)(void);

    void (SA3_CALL *error_init)(sa3_error_v1* error);
    void (SA3_CALL *context_config_init)(sa3_context_config_v1* config);
    void (SA3_CALL *request_init)(sa3_request_v1* request);
    void (SA3_CALL *audio_view_init)(sa3_audio_view_v1* audio);
    void (SA3_CALL *adapter_init)(sa3_adapter_v1* adapter);
    void (SA3_CALL *loudness_init)(sa3_loudness_v1* loudness);
    void (SA3_CALL *continuation_init)(sa3_continuation_v1* continuation);
    void (SA3_CALL *result_init)(sa3_result_v1* result);
    void (SA3_CALL *lora_convert_init)(sa3_lora_convert_v1* options);

    sa3_status_v1 (SA3_CALL *context_create)(const sa3_context_config_v1* config,
                                              sa3_context** out_context,
                                              sa3_error_v1* error);
    void (SA3_CALL *context_unload)(sa3_context* context);
    void (SA3_CALL *context_destroy)(sa3_context* context);
    sa3_status_v1 (SA3_CALL *generate)(sa3_context* context,
                                       const sa3_request_v1* request,
                                       sa3_result_v1* result,
                                       sa3_error_v1* error);
    void (SA3_CALL *result_free)(sa3_result_v1* result);
    sa3_status_v1 (SA3_CALL *convert_lora)(const sa3_lora_convert_v1* options,
                                           sa3_error_v1* error);

    sa3_reserved_function_v1 reserved[16];
} sa3_api_v1;

/* Frozen V1 prefixes. Where a type is appendable, these expressions must continue to name the
 * final field published by the initial V1 contract. Libraries validate against these values,
 * never against a future sizeof(struct). By-value types remain exactly this shape for ABI V1. */
#define SA3_ERROR_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_error_v1, message) + sizeof(((sa3_error_v1*)0)->message)))
#define SA3_CONTEXT_CONFIG_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_context_config_v1, cpu_threads) + sizeof(((sa3_context_config_v1*)0)->cpu_threads)))
#define SA3_AUDIO_VIEW_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_audio_view_v1, reserved) + sizeof(((sa3_audio_view_v1*)0)->reserved)))
#define SA3_ADAPTER_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_adapter_v1, strength) + sizeof(((sa3_adapter_v1*)0)->strength)))
#define SA3_LOUDNESS_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_loudness_v1, latent_shift) + sizeof(((sa3_loudness_v1*)0)->latent_shift)))
#define SA3_CONTINUATION_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_continuation_v1, gain_match) + sizeof(((sa3_continuation_v1*)0)->gain_match)))
#define SA3_PROGRESS_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_progress_v1, fraction) + sizeof(((sa3_progress_v1*)0)->fraction)))
#define SA3_REQUEST_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_request_v1, callback_user) + sizeof(((sa3_request_v1*)0)->callback_user)))
/* A request at least this large carries the SAT sampler fields. */
#define SA3_REQUEST_V1_SAT_SIZE \
    ((uint32_t)(offsetof(sa3_request_v1, conditioning_seconds_total) + \
                sizeof(((sa3_request_v1*)0)->conditioning_seconds_total)))
#define SA3_RESULT_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_result_v1, mask_overlap_seconds) + sizeof(((sa3_result_v1*)0)->mask_overlap_seconds)))
#define SA3_LORA_CONVERT_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_lora_convert_v1, output_gguf_path) + sizeof(((sa3_lora_convert_v1*)0)->output_gguf_path)))
#define SA3_API_V1_MIN_SIZE \
    ((uint32_t)(offsetof(sa3_api_v1, convert_lora) + sizeof(((sa3_api_v1*)0)->convert_lora)))

/* The only entry point a dynamically loaded V1 consumer needs to resolve.
 * Returns NULL for an unsupported ABI major. The returned table is static and
 * owned by libsa3 for the lifetime of the loaded module. */
SA3_API const sa3_api_v1* SA3_CALL sa3_get_api(uint32_t abi_version);

#ifdef __cplusplus
}
#endif

#endif /* LIBSA3_V1_H */
