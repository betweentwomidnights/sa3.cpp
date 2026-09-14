#include "libsa3_training_v1.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do {                                                     \
    if (!(condition)) {                                                           \
        fprintf(stderr, "contract check failed at %s:%d: %s\n",                 \
                __FILE__, __LINE__, #condition);                                  \
        return 1;                                                                 \
    }                                                                             \
} while (0)

_Static_assert(offsetof(sa3_error_v1, size) == 0, "size must lead error");
_Static_assert(offsetof(sa3_context_config_v1, size) == 0, "size must lead config");
_Static_assert(offsetof(sa3_audio_view_v1, size) == 0, "size must lead audio view");
_Static_assert(offsetof(sa3_request_v1, size) == 0, "size must lead request");
_Static_assert(offsetof(sa3_result_v1, size) == 0, "size must lead result");
_Static_assert(offsetof(sa3_api_v1, size) == 0, "size must lead API table");
_Static_assert(offsetof(sa3_training_config_v1, size) == 0, "size must lead training config");
_Static_assert(offsetof(sa3_training_step_v1, size) == 0, "size must lead training step");
_Static_assert(offsetof(sa3_training_audio_buffer_v1, size) == 0,
               "size must lead training audio buffer");
_Static_assert(offsetof(sa3_training_result_v1, size) == 0, "size must lead training result");

static sa3_training_audio_status_v1 SA3_CALL test_training_audio_loader(
    void* user, const char* path, uint32_t sample_rate, uint32_t channels,
    sa3_training_audio_buffer_v1* audio, sa3_error_v1* error) {
    (void)user; (void)path; (void)sample_rate; (void)channels; (void)audio; (void)error;
    return SA3_TRAINING_AUDIO_NOT_HANDLED_V1;
}

int main(void) {
    const sa3_api_v1* api = sa3_get_api(SA3_ABI_VERSION_1);
    CHECK(api != NULL);
    CHECK(sa3_get_api(0) == NULL);
    CHECK(sa3_get_api(SA3_ABI_VERSION_1 + 1) == NULL);
    CHECK(api->size >= SA3_API_V1_MIN_SIZE);
    CHECK(api->abi_version == SA3_ABI_VERSION_1);
    CHECK(api->runtime_version != NULL);
    CHECK(strstr(api->runtime_version(), "ABI 1") != NULL);

    sa3_context_config_v1 config;
    memset(&config, 0xA5, sizeof(config));
    config.size = sizeof(config);
    api->context_config_init(&config);
    CHECK(config.size == sizeof(config));
    CHECK(config.models_dir == NULL);
    CHECK(config.cpu_threads == 0);

    sa3_request_v1 request;
    memset(&request, 0xA5, sizeof(request));
    request.size = sizeof(request);
    api->request_init(&request);
    CHECK(request.size == sizeof(request));
    CHECK(request.operation == SA3_OPERATION_GENERATE_V1);
    CHECK(request.duration_seconds == 12.0);
    CHECK(request.steps == 8);
    CHECK(request.seed == -1);
    CHECK(request.cfg_scale == 1.0f);
    CHECK(request.distribution_shift == SA3_DISTRIBUTION_LOGSNR_V1);
    CHECK(request.residency == SA3_RESIDENCY_RESIDENT_V1);
    CHECK(request.input_audio.size == sizeof(sa3_audio_view_v1));
    CHECK(request.input_audio.layout == SA3_AUDIO_PLANAR_V1);
    CHECK(request.transform_noise_level == 0.85f);
    CHECK(request.generation_tail_padding_seconds == 6.0f);
    CHECK(request.continuation_tail_padding_seconds == 6.0f);
    CHECK(request.adapter_stride == sizeof(sa3_adapter_v1));
    CHECK(request.loudness.size == sizeof(sa3_loudness_v1));
    CHECK(request.loudness.peak_normalize == 1);
    CHECK(request.continuation.size == sizeof(sa3_continuation_v1));
    CHECK(request.continuation.splice_source == 1);

    struct future_request {
        sa3_request_v1 known;
        uint64_t future_tail;
    } future_request;
    memset(&future_request, 0xA5, sizeof(future_request));
    future_request.known.size = sizeof(future_request);
    api->request_init(&future_request.known);
    CHECK(future_request.known.size == sizeof(future_request));
    CHECK(future_request.known.steps == 8);
    CHECK(future_request.future_tail == UINT64_C(0xA5A5A5A5A5A5A5A5));

    sa3_loudness_v1 loudness;
    memset(&loudness, 0xA5, sizeof(loudness));
    loudness.size = sizeof(loudness);
    api->loudness_init(&loudness);
    CHECK(loudness.peak_normalize == 1);
    CHECK(loudness.peak_normalize_db == 2.0f);
    CHECK(loudness.limiter == 1);
    CHECK(loudness.limiter_ceiling_db == -0.3f);
    CHECK(loudness.latent_rescale == 1.0f);

    sa3_continuation_v1 continuation;
    memset(&continuation, 0xA5, sizeof(continuation));
    continuation.size = sizeof(continuation);
    api->continuation_init(&continuation);
    CHECK(continuation.splice_source == 1);
    CHECK(continuation.mask_overlap_seconds == 0.2f);
    CHECK(continuation.crossfade_seconds == 0.03f);
    CHECK(continuation.gain_match == 1);

    sa3_error_v1 error;
    memset(&error, 0, sizeof(error));
    error.size = sizeof(error);
    api->error_init(&error);
    CHECK(api->context_create(&config, NULL, &error) == SA3_STATUS_INVALID_ARGUMENT_V1);
    CHECK(error.code == SA3_STATUS_INVALID_ARGUMENT_V1);
    CHECK(error.message[0] != '\0');

    sa3_result_v1 result;
    memset(&result, 0, sizeof(result));
    result.size = sizeof(result);
    api->result_init(&result);
    CHECK(api->generate(NULL, &request, &result, &error) == SA3_STATUS_INVALID_ARGUMENT_V1);
    CHECK(result.samples == NULL);
    api->result_free(&result);
    api->result_free(&result);
    CHECK(result.size == sizeof(result));
    CHECK(result.samples == NULL);

    struct future_result {
        sa3_result_v1 known;
        uint64_t future_tail;
    } future_result;
    memset(&future_result, 0xA5, sizeof(future_result));
    future_result.known.size = sizeof(future_result);
    future_result.known.samples = NULL;
    future_result.future_tail = UINT64_C(0x123456789ABCDEF0);
    api->result_free(&future_result.known);
    CHECK(future_result.known.size == sizeof(future_result));
    CHECK(future_result.future_tail == UINT64_C(0x123456789ABCDEF0));

    const sa3_training_api_v1* training = sa3_get_training_api(SA3_TRAINING_ABI_VERSION_1);
    CHECK(training != NULL);
    CHECK(sa3_get_training_api(0) == NULL);
    CHECK(sa3_get_training_api(SA3_TRAINING_ABI_VERSION_1 + 1) == NULL);
    CHECK(training->size >= SA3_TRAINING_API_V1_MIN_SIZE);
    CHECK(training->abi_version == SA3_TRAINING_ABI_VERSION_1);

    sa3_training_config_v1 training_config;
    memset(&training_config, 0xA5, sizeof(training_config));
    training_config.size = sizeof(training_config);
    training->config_init(&training_config);
    CHECK(training_config.steps == 10000);
    CHECK(training_config.rank == 16);
    CHECK(training_config.learning_rate == 1.0e-4f);
    CHECK(training_config.frames == 512);
    CHECK(training_config.batch_size == 1);
    CHECK(training_config.checkpoint_every == 500);
    CHECK(training_config.pre_encode == 1);
    CHECK(training_config.seed == 42);
    CHECK(training_config.latents_cache == 1);

    struct future_training_config {
        sa3_training_config_v1 known;
        uint64_t future_tail;
    } future_training_config;
    memset(&future_training_config, 0xA5, sizeof(future_training_config));
    future_training_config.known.size = sizeof(future_training_config);
    future_training_config.future_tail = UINT64_C(0x123456789ABCDEF0);
    training->config_init(&future_training_config.known);
    CHECK(future_training_config.known.size == sizeof(future_training_config));
    CHECK(future_training_config.known.steps == 10000);
    CHECK(future_training_config.future_tail == UINT64_C(0x123456789ABCDEF0));

    sa3_training_callbacks_v1 training_callbacks;
    memset(&training_callbacks, 0xA5, sizeof(training_callbacks));
    training_callbacks.size = sizeof(training_callbacks);
    training->callbacks_init(&training_callbacks);
    CHECK(training_callbacks.on_log == NULL);
    CHECK(training_callbacks.user == NULL);

    sa3_training_result_v1 training_result;
    memset(&training_result, 0xA5, sizeof(training_result));
    training_result.size = sizeof(training_result);
    training->result_init(&training_result);
    CHECK(training_result.completed_steps == 0);

    struct future_training_result {
        sa3_training_result_v1 known;
        uint64_t future_tail;
    } future_training_result;
    memset(&future_training_result, 0xA5, sizeof(future_training_result));
    future_training_result.known.size = sizeof(future_training_result);
    future_training_result.future_tail = UINT64_C(0x123456789ABCDEF0);
    training->result_init(&future_training_result.known);
    CHECK(future_training_result.known.size == sizeof(future_training_result));
    CHECK(future_training_result.known.completed_steps == 0);
    CHECK(future_training_result.future_tail == UINT64_C(0x123456789ABCDEF0));
    CHECK(training->run(&training_config, &training_callbacks, &training_result, &error)
          == SA3_STATUS_INVALID_ARGUMENT_V1);
    CHECK(strstr(error.message, "dataset_dir") != NULL);

    training_callbacks.load_audio = test_training_audio_loader;
    CHECK(training->run(&training_config, &training_callbacks, &training_result, &error)
          == SA3_STATUS_INVALID_ARGUMENT_V1);
    CHECK(strstr(error.message, "release_audio") != NULL);

    puts("libsa3 V1 C ABI contract passed");
    return 0;
}
