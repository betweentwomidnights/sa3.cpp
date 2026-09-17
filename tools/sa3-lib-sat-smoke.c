/* sa3-lib-sat-smoke -- drive a SAT variant (Foundation-1.2 Keybeds) through the libsa3 V1 C ABI.
 *
 * Usage: sa3-lib-sat-smoke MODELS_DIR [VARIANT] [STEPS]
 *
 * Renders a two-note keybed chunk (the same request shape an instrument host sends), checks the
 * result geometry and that it is not silent, then cancels a second render mid-sampling and checks
 * for CANCELLED. Needs the model files, so it is a tool rather than a CTest. */
#include "libsa3_v1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int sampling_steps;
    int cancel_after;
} smoke_state;

static void SA3_CALL on_progress(void* user, const sa3_progress_v1* progress) {
    smoke_state* state = (smoke_state*)user;
    if (progress->stage == SA3_PROGRESS_SAMPLING_V1 && progress->step > 0)
        state->sampling_steps = progress->step;
    if (progress->stage != SA3_PROGRESS_SAMPLING_V1 || progress->step == 1 ||
        progress->step == progress->total)
        printf("  progress %-9s %3d/%-3d %5.1f%%\n", progress->stage_name, progress->step,
               progress->total, 100.0f * progress->fraction);
}

static int32_t SA3_CALL should_cancel(void* user) {
    const smoke_state* state = (const smoke_state*)user;
    return state->cancel_after > 0 && state->sampling_steps >= state->cancel_after;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODELS_DIR [VARIANT] [STEPS]\n", argv[0]);
        return 2;
    }
    const char* variant = argc > 2 ? argv[2] : "foundation-1.2-keybeds";
    const int steps = argc > 3 ? atoi(argv[3]) : 8;

    const sa3_api_v1* api = sa3_get_api(SA3_ABI_VERSION_1);
    if (!api) { fprintf(stderr, "no V1 table\n"); return 1; }
    printf("%s\n", api->runtime_version());

    sa3_error_v1 error; memset(&error, 0, sizeof error); error.size = sizeof error;
    sa3_context_config_v1 config; memset(&config, 0, sizeof config); config.size = sizeof config;
    api->context_config_init(&config);
    config.models_dir = argv[1];
    config.variant = variant;
    sa3_context* context = NULL;
    if (api->context_create(&config, &context, &error) != SA3_STATUS_OK_V1) {
        fprintf(stderr, "context_create: %s\n", error.message);
        return 1;
    }

    /* Two notes: 2*3.0 + 0.25 = 6.25 s cropped from a 7 s conditioned canvas. */
    smoke_state state = {0, 0};
    sa3_request_v1 request; memset(&request, 0, sizeof request); request.size = sizeof request;
    api->request_init(&request);
    request.prompt = "Keybed, Sequence, Timbre Profile, Grand Piano, Warm, Dry, "
                     "Chromatic Chunk, Note Sequence, C4, C#4";
    request.duration_seconds = 6.25;
    request.conditioning_seconds_total = 7.0;
    request.steps = steps;
    request.cfg_scale = 6.0f;
    request.seed = 42;
    request.sampler = SA3_SAMPLER_DPMPP_3M_SDE_V1;
    request.loudness.peak_normalize = 0;
    request.loudness.limiter = 0;
    request.on_progress = on_progress;
    request.should_cancel = should_cancel;
    request.callback_user = &state;

    sa3_result_v1 result; memset(&result, 0, sizeof result); result.size = sizeof result;
    api->result_init(&result);
    int failures = 0;
    sa3_status_v1 status = api->generate(context, &request, &result, &error);
    if (status != SA3_STATUS_OK_V1) {
        fprintf(stderr, "generate: status %d: %s\n", status, error.message);
        api->context_destroy(context);
        return 1;
    }
    double energy = 0.0;
    for (uint64_t i = 0; i < result.n_samples * result.n_channels; ++i)
        energy += (double)result.samples[i] * result.samples[i];
    const double rms = sqrt(energy / (double)(result.n_samples * result.n_channels));
    printf("audio: %llu samples, %u ch, %u Hz, seed %llu, rms %.4f, peak %.4f\n",
           (unsigned long long)result.n_samples, result.n_channels, result.sample_rate,
           (unsigned long long)result.seed, rms, result.final_peak);
    if (result.n_samples != 275625 || result.n_channels != 2 || result.sample_rate != 44100) {
        fprintf(stderr, "FAIL: unexpected geometry\n"); ++failures;
    }
    if (!(rms > 1.0e-3)) { fprintf(stderr, "FAIL: output is silent\n"); ++failures; }
    if (result.seed != 42) { fprintf(stderr, "FAIL: seed not reported\n"); ++failures; }
    if (result.peak_normalize_gain_set || result.limiter_limited_fraction_set) {
        fprintf(stderr, "FAIL: raw loudness request was post-processed\n"); ++failures;
    }
    api->result_free(&result);

    /* Unsupported operation on a SAT context. */
    request.operation = SA3_OPERATION_CONTINUE_V1;
    request.input_audio.samples = (const float*)&energy;
    request.input_audio.n_samples = 1;
    request.input_audio.n_channels = 1;
    request.input_audio.sample_rate = 44100;
    status = api->generate(context, &request, &result, &error);
    if (status != SA3_STATUS_INVALID_ARGUMENT_V1 || !strstr(error.message, "generate operation")) {
        fprintf(stderr, "FAIL: continue on a SAT variant returned %d: %s\n", status, error.message);
        ++failures;
    }
    request.operation = SA3_OPERATION_GENERATE_V1;
    memset(&request.input_audio, 0, sizeof request.input_audio);
    request.input_audio.size = sizeof request.input_audio;

    /* Cooperative cancel after three sampling steps, with resident weights still warm. */
    state.sampling_steps = 0;
    state.cancel_after = 3;
    request.steps = steps > 4 ? steps : 5;
    status = api->generate(context, &request, &result, &error);
    printf("cancel: status %d after %d steps: %s\n", status, state.sampling_steps, error.message);
    if (status != SA3_STATUS_CANCELLED_V1 || result.samples != NULL) {
        fprintf(stderr, "FAIL: expected CANCELLED with no audio\n"); ++failures;
    }

    api->context_unload(context);
    api->context_destroy(context);
    if (failures) return 1;
    puts("sa3-lib-sat-smoke: ok");
    return 0;
}
