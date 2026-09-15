/* sa3-libcancel — V1 cooperative-cancel smoke test for embedded hosts.
 *
 * Uses the same knobs the IPlug2 demo relies on for long renders: frugal residency plus chunked
 * decode. By default the cancel callback fires on the first poll, so this validates the callback
 * plumbing and the cleanup after it without paying for a full generation.
 *
 * A cancelled generate must report SA3_STATUS_CANCELLED_V1 and leave the result owning nothing.
 *
 * Usage:
 *   sa3-libcancel [models_dir] [cancel_after_polls]
 */
#include "libsa3_v1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct cancel_state {
    int polls;
    int cancel_after_polls;
    int progress_calls;
} cancel_state;

static int32_t SA3_CALL should_cancel(void* user) {
    cancel_state* state = (cancel_state*)user;
    if (!state) return 1;
    state->polls += 1;
    return state->polls > state->cancel_after_polls;
}

static void SA3_CALL on_progress(void* user, const sa3_progress_v1* progress) {
    cancel_state* state = (cancel_state*)user;
    if (state) state->progress_calls += 1;
    if (!progress) return;
    printf("  [%3.0f%%] %s %d/%d\n", progress->fraction * 100.0f,
           progress->stage_name ? progress->stage_name : "render",
           progress->step, progress->total);
    fflush(stdout);
}

int main(int argc, char** argv) {
    const char* models_dir = argc > 1 ? argv[1] : NULL;
    cancel_state state;
    memset(&state, 0, sizeof state);
    state.cancel_after_polls = argc > 2 ? atoi(argv[2]) : 0;
    if (state.cancel_after_polls < 0) state.cancel_after_polls = 0;

    const sa3_api_v1* api = sa3_get_api(SA3_ABI_VERSION_1);
    if (!api) { fprintf(stderr, "libsa3 does not provide ABI v1\n"); return 1; }
    printf("%s\n", api->runtime_version());
    printf("cancel_after_polls=%d\n", state.cancel_after_polls);

    sa3_error_v1 err;
    memset(&err, 0, sizeof err);
    err.size = sizeof err;
    api->error_init(&err);

    sa3_context_config_v1 cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.size = sizeof cfg;
    api->context_config_init(&cfg);
    cfg.models_dir = models_dir;
    cfg.variant = "medium";
    cfg.dit_encoding = "f16";

    sa3_context* ctx = NULL;
    if (api->context_create(&cfg, &ctx, &err) != SA3_STATUS_OK_V1) {
        fprintf(stderr, "context_create failed: %s\n", err.message);
        return 1;
    }

    sa3_request_v1 req;
    memset(&req, 0, sizeof req);
    req.size = sizeof req;
    api->request_init(&req);
    req.operation = SA3_OPERATION_GENERATE_V1;
    req.prompt = "warm analog cancellation smoke test";
    req.duration_seconds = 12.0;
    req.residency = SA3_RESIDENCY_FRUGAL_V1;
    req.decode_chunk_size = 128;
    req.decode_overlap = 32;
    req.on_progress = on_progress;
    req.should_cancel = should_cancel;
    req.callback_user = &state;

    sa3_result_v1 result;
    memset(&result, 0, sizeof result);
    result.size = sizeof result;
    api->result_init(&result);

    const sa3_status_v1 status = api->generate(ctx, &req, &result, &err);
    int rc = 0;
    if (status == SA3_STATUS_OK_V1) {
        fprintf(stderr, "expected cancellation, but generate succeeded\n");
        rc = 2;
    } else if (status != SA3_STATUS_CANCELLED_V1) {
        fprintf(stderr, "expected SA3_STATUS_CANCELLED_V1, got %d: %s\n", status, err.message);
        rc = 3;
    } else if (result.samples != NULL || result.n_samples != 0 || result.n_channels != 0) {
        fprintf(stderr, "cancelled request left the result owning audio\n");
        rc = 4;
    } else {
        printf("cancelled successfully after %d poll(s), progress callbacks=%d\n",
               state.polls, state.progress_calls);
    }

    api->result_free(&result);
    api->context_destroy(ctx);
    return rc;
}
