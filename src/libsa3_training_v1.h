/* libsa3 Training V1 — stable C ABI for in-process adapter training.
 *
 * Training is a separate capability table from inference because its options and
 * callbacks evolve independently. A host may use either table or both.
 */
#ifndef LIBSA3_TRAINING_V1_H
#define LIBSA3_TRAINING_V1_H

#include "libsa3_v1.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SA3_TRAINING_ABI_VERSION_1 1u

typedef struct {
    uint32_t size;

    const char* models_dir;
    const char* dataset_dir;
    const char* output_dir;
    const char* config_path;
    const char* latents_dir;
    const char* latents_cache_dir;
    const char* prompt_config_path;
    const char* resume_path;

    const char* variant;
    const char* dit_encoding;
    const char* text_encoder_encoding;
    const char* autoencoder_encoding;
    const char* adapter_type;
    const char* lora_scope;
    const char* device;

    int32_t steps;
    int32_t rank;
    float alpha;
    float learning_rate;
    int32_t frames;
    float duration_seconds;
    int32_t batch_size;
    int32_t checkpoint_every;
    int32_t cpu_threads;
    int32_t pre_encode;
    int64_t seed;
    int32_t evict_text_encoder;
    int32_t latents_cache;
} sa3_training_config_v1;

/* Borrowed for the duration of on_step. String pointers have the same lifetime. */
typedef struct {
    uint32_t size;
    int32_t epoch;
    int32_t step;
    int32_t max_steps;
    const char* id;
    const char* prompt;
    const char* mask;
    float timestep;
    float learning_rate;
    float loss;
    double gradient_norm;
    double step_seconds;
    int32_t cfg_dropped;
    int32_t updated;
    int32_t generated_frames;
    int32_t context_frames;
} sa3_training_step_v1;

typedef void (SA3_CALL *sa3_training_log_callback_v1)(void* user, const char* line);
typedef void (SA3_CALL *sa3_training_step_callback_v1)(void* user,
                                                        const sa3_training_step_v1* step);
typedef int32_t (SA3_CALL *sa3_training_cancel_callback_v1)(void* user);

/* Optional sandboxed-host decoder. Return non-zero with a complete borrowed audio view to supply
 * audio, or zero to let libsa3 read audio_path. The requested sample rate and channel count must
 * be returned unchanged. Planar and interleaved views are accepted and copied before return. */
typedef int32_t (SA3_CALL *sa3_training_audio_callback_v1)(
    void* user,
    const char* audio_path,
    uint32_t requested_sample_rate,
    uint32_t requested_channels,
    sa3_audio_view_v1* out_audio);

typedef struct {
    uint32_t size;
    sa3_training_log_callback_v1 on_log;
    sa3_training_step_callback_v1 on_step;
    sa3_training_cancel_callback_v1 should_cancel;
    sa3_training_audio_callback_v1 load_audio;
    void* user;
    const char* command_line;
} sa3_training_callbacks_v1;

/* Caller-owned fixed-size result. A cancelled run may still return a final adapter and checkpoint.
 * SA3_STATUS_CANCELLED_V1 means this result is valid and must be inspected. */
typedef struct {
    uint32_t size;
    int32_t completed_steps;
    int32_t cancelled;
    double mean_step_seconds;
    char final_adapter[1024];
    char last_checkpoint[1024];
    char preview_command[2048];
} sa3_training_result_v1;

typedef struct sa3_training_api_v1 {
    uint32_t size;
    uint32_t abi_version;
    const char* (SA3_CALL *runtime_version)(void);

    void (SA3_CALL *config_init)(sa3_training_config_v1* config);
    void (SA3_CALL *callbacks_init)(sa3_training_callbacks_v1* callbacks);
    void (SA3_CALL *result_init)(sa3_training_result_v1* result);

    /* Blocks until completion, cancellation, or failure. Serialize training jobs in a process.
     * On OK or CANCELLED, result is valid. Error is optional and uses the inference V1 type. */
    sa3_status_v1 (SA3_CALL *run)(const sa3_training_config_v1* config,
                                  const sa3_training_callbacks_v1* callbacks,
                                  sa3_training_result_v1* result,
                                  sa3_error_v1* error);

    void* reserved[16];
} sa3_training_api_v1;

SA3_API const sa3_training_api_v1* SA3_CALL sa3_get_training_api(uint32_t abi_version);

#ifdef __cplusplus
}
#endif

#endif /* LIBSA3_TRAINING_V1_H */
