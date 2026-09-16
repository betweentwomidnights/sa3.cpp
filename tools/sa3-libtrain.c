/* sa3-libtrain — pure-C smoke test and usage example for the libsa3 Training V1 ABI.
 *
 * This is the call sequence an embedded host (iOS app, plugin) uses to train an adapter
 * in-process, including the callbacks it drives the run with. Training resolves its own table:
 *
 *   sa3_get_training_api -> config_init -> callbacks_init -> run -> inspect result
 *
 * It is versioned apart from the inference table so training can gain a major without forcing
 * inference-only hosts to migrate, and it initializes the shared error type itself so a
 * training-only host never has to resolve the inference table at all.
 *
 *   usage: sa3-libtrain <dataset-dir> <out-dir> [steps] [variant] [latents-dir] [latents-cache-dir]
 *
 * latents-cache-dir is the one a sandboxed host usually has to set: the pre-encode cache defaults
 * to a directory under dataset_dir, which an app that cannot write there needs to redirect. Pass
 * "-" to turn the cache off instead.
 *
 * The adapter it writes is byte-identical to one produced by `sa3-train` with the same config.
 */
#include "libsa3_training_v1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct host {
    int steps_seen;
    int cancel_after;   /* 0 = never cancel */
};

static void SA3_CALL on_log(void* user, const char* line) {
    (void)user;
    if (!line) return;
    fputs(line, stdout);
    fflush(stdout);
}

static void SA3_CALL on_step(void* user, const sa3_training_step_v1* s) {
    struct host* h = (struct host*)user;
    if (!h || !s) return;
    h->steps_seen++;
    printf("  [step %d/%d] loss=%.6f gnorm=%.4f lr=%.3e %.2fs id=%s\n",
           s->step, s->max_steps, s->loss, s->gradient_norm, s->learning_rate,
           s->step_seconds, s->id ? s->id : "");
    fflush(stdout);
}

static int32_t SA3_CALL should_cancel(void* user) {
    struct host* h = (struct host*)user;
    return h && h->cancel_after > 0 && h->steps_seen >= h->cancel_after;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <dataset-dir> <out-dir> [steps] [variant] [latents-dir] [latents-cache-dir]\n",
                argv[0]);
        return 2;
    }

    const sa3_training_api_v1* api = sa3_get_training_api(SA3_TRAINING_ABI_VERSION_1);
    if (!api) { fprintf(stderr, "libsa3 does not provide training ABI v1\n"); return 1; }

    sa3_training_config_v1 cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.size = sizeof cfg;
    api->config_init(&cfg);
    cfg.dataset_dir      = argv[1];
    cfg.output_dir       = argv[2];
    cfg.steps            = argc > 3 ? atoi(argv[3]) : 4;
    cfg.variant          = argc > 4 ? argv[4] : "small-music";
    cfg.latents_dir      = argc > 5 ? argv[5] : NULL;
    if (argc > 6) {
        if (strcmp(argv[6], "-") == 0) cfg.latents_cache = 0;
        else                           cfg.latents_cache_dir = argv[6];
    }
    cfg.frames           = 128;
    cfg.checkpoint_every = 0;               /* no intermediate checkpoints for a smoke test */
    cfg.seed             = 42;
    cfg.command_line     = "sa3-libtrain";

    struct host h;
    memset(&h, 0, sizeof h);
    if (getenv("SA3_LIBTRAIN_CANCEL_AFTER")) h.cancel_after = atoi(getenv("SA3_LIBTRAIN_CANCEL_AFTER"));

    sa3_training_callbacks_v1 hooks;
    memset(&hooks, 0, sizeof hooks);
    hooks.size = sizeof hooks;
    api->callbacks_init(&hooks);
    hooks.on_log        = on_log;
    hooks.on_step       = on_step;
    hooks.should_cancel = should_cancel;
    hooks.user          = &h;
    /* load_audio/release_audio stay null: this host lets libsa3 read the dataset itself. They are
       for sandboxes that cannot, and must be installed as a pair. */

    sa3_training_result_v1 res;
    memset(&res, 0, sizeof res);
    res.size = sizeof res;
    api->result_init(&res);

    sa3_error_v1 err;
    memset(&err, 0, sizeof err);
    err.size = sizeof err;
    api->error_init(&err);

    printf("libsa3 %s: training %s -> %s\n", api->runtime_version(), cfg.dataset_dir, cfg.output_dir);
    const sa3_status_v1 status = api->run(&cfg, &hooks, &res, &err);
    /* Both OK and CANCELLED return a valid result -- a cancelled training phase has usually still
       written a final adapter, so it is inspected rather than discarded. */
    if (status != SA3_STATUS_OK_V1 && status != SA3_STATUS_CANCELLED_V1) {
        fprintf(stderr, "sa3-libtrain: %s (status=%d)\n", err.message, status);
        return 1;
    }

    printf("\nsteps=%d cancelled=%d mean=%.3fs/step\n",
           res.completed_steps, res.cancelled, res.mean_step_seconds);
    printf("final adapter: %s\n", res.final_adapter);
    if (res.last_checkpoint[0]) printf("last checkpoint: %s\n", res.last_checkpoint);
    if (res.preview_command[0]) printf("try it: %s\n", res.preview_command);

    if (res.completed_steps <= 0) { fprintf(stderr, "sa3-libtrain: no optimizer update ran\n"); return 1; }
    if (h.steps_seen <= 0) { fprintf(stderr, "sa3-libtrain: on_step never fired\n"); return 1; }
    if (!res.final_adapter[0]) { fprintf(stderr, "sa3-libtrain: no final adapter path\n"); return 1; }
    printf("sa3-libtrain: ok\n");
    return 0;
}
