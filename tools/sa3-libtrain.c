/* sa3-libtrain — pure-C smoke test + usage example for the libsa3 training API (see src/libsa3.h).
 * This is the call sequence an embedded host (iOS app, plugin) would use to train an adapter
 * in-process, including the callbacks it drives the run with.
 *
 *   usage: sa3-libtrain <dataset-dir> <out-dir> [steps] [variant] [latents-dir] [latents-cache-dir]
 *
 * latents-cache-dir is the one a sandboxed host usually has to set: the pre-encode cache defaults
 * to a directory under dataset_dir, which an app that cannot write there needs to redirect. Pass
 * "-" to turn the cache off instead.
 *
 * The adapter it writes is byte-identical to one produced by `sa3-train` with the same config.
 */
#include "libsa3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct host {
    int steps_seen;
    int cancel_after;   /* 0 = never cancel */
};

static void on_log(void* user, const char* line) {
    (void)user;
    fputs(line, stdout);
    fflush(stdout);
}

static void on_step(void* user, const sa3_train_step* s) {
    struct host* h = (struct host*)user;
    h->steps_seen++;
    printf("  [step %d/%d] loss=%.6f gnorm=%.4f lr=%.3e %.2fs id=%s\n",
           s->step, s->max_steps, s->loss, s->grad_norm, s->learning_rate, s->step_seconds, s->id);
    fflush(stdout);
}

static int should_cancel(void* user) {
    struct host* h = (struct host*)user;
    return h->cancel_after > 0 && h->steps_seen >= h->cancel_after;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <dataset-dir> <out-dir> [steps] [variant] [latents-dir] [latents-cache-dir]\n",
                argv[0]);
        return 2;
    }

    sa3_train_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.dataset_dir      = argv[1];
    cfg.output_dir       = argv[2];
    cfg.steps            = argc > 3 ? atoi(argv[3]) : 4;
    cfg.variant          = argc > 4 ? argv[4] : "small-music";
    cfg.latents_dir      = argc > 5 ? argv[5] : NULL;
    if (argc > 6) {
        if (strcmp(argv[6], "-") == 0) cfg.latents_cache = -1;   /* negative disables, like checkpoint_every */
        else                           cfg.latents_cache_dir = argv[6];
    }
    cfg.frames           = 128;
    cfg.checkpoint_every = -1;              /* no intermediate checkpoints for a smoke test */
    cfg.seed             = 42;

    struct host h;
    memset(&h, 0, sizeof h);
    if (getenv("SA3_LIBTRAIN_CANCEL_AFTER")) h.cancel_after = atoi(getenv("SA3_LIBTRAIN_CANCEL_AFTER"));

    sa3_train_hooks hooks;
    memset(&hooks, 0, sizeof hooks);
    hooks.on_log        = on_log;
    hooks.on_step       = on_step;
    hooks.should_cancel = should_cancel;
    hooks.user          = &h;
    hooks.command_line  = "sa3-libtrain";

    sa3_train_result res;
    char err[1024] = {0};
    printf("libsa3 %s: training %s -> %s\n", sa3_version(), cfg.dataset_dir, cfg.output_dir);
    const int rc = sa3_train(&cfg, &hooks, &res, err, (int)sizeof err);
    if (rc != 0) {
        fprintf(stderr, "sa3-libtrain: %s (rc=%d)\n", err, rc);
        return 1;
    }

    printf("\nsteps=%d cancelled=%d mean=%.3fs/step\n", res.steps, res.cancelled, res.mean_step_seconds);
    printf("final adapter: %s\n", res.final_adapter);
    if (res.last_checkpoint[0]) printf("last checkpoint: %s\n", res.last_checkpoint);
    printf("try it: %s\n", res.preview_command);

    if (res.steps <= 0) { fprintf(stderr, "sa3-libtrain: no optimizer update ran\n"); return 1; }
    if (h.steps_seen <= 0) { fprintf(stderr, "sa3-libtrain: on_step never fired\n"); return 1; }
    if (!res.final_adapter[0]) { fprintf(stderr, "sa3-libtrain: no final adapter path\n"); return 1; }
    printf("sa3-libtrain: ok\n");
    return 0;
}
