/* sa3-libtest — a pure-C smoke test + minimal usage example for libsa3 (see src/libsa3.h).
 * This is exactly the call sequence a JUCE / IPlug2 host would use:
 *   sa3_init -> sa3_generate (with a progress callback) -> use samples -> sa3_free_audio -> sa3_free.
 *   usage: sa3-libtest ["prompt"] [out.wav] [cpu_threads]
 */
#include "libsa3.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void on_progress(void* user, const char* stage, int step, int total, float frac) {
    (void)user;
    printf("  [%3.0f%%] %s %d/%d\n", frac * 100.0f, stage, step, total);
    fflush(stdout);
}

/* Write PLANAR float samples (samples[c*n_samp+s]) as a 16-bit interleaved WAV. */
static void write_wav(const char* path, const float* planar, int n_samp, int n_ch, int sr) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return; }
    const uint32_t data_bytes = (uint32_t)n_samp * n_ch * 2;
    const uint16_t block_align = (uint16_t)(n_ch * 2), bits = 16, fmt = 1, ch = (uint16_t)n_ch;
    const uint32_t chunk = 36 + data_bytes, fmtlen = 16, srate = (uint32_t)sr, byte_rate = (uint32_t)sr * n_ch * 2;
    fwrite("RIFF", 1, 4, f); fwrite(&chunk, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmtlen, 4, 1, f); fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&srate, 4, 1, f); fwrite(&byte_rate, 4, 1, f); fwrite(&block_align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_bytes, 4, 1, f);
    for (int s = 0; s < n_samp; s++)
        for (int c = 0; c < n_ch; c++) {
            float v = planar[(size_t)c * n_samp + s];
            v = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
            int16_t iv = (int16_t)(v * 32767.0f);
            fwrite(&iv, 2, 1, f);
        }
    fclose(f);
}

int main(int argc, char** argv) {
    const char* prompt = argc > 1 ? argv[1] : "warm analog house groove";
    const char* out    = argc > 2 ? argv[2] : "libsa3_test.wav";
    const int cpu_threads = argc > 3 ? atoi(argv[3]) : 0;
    char err[512] = {0};

    printf("%s\n", sa3_version());

    sa3_config_ex cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.config.variant = "medium";
    cfg.config.encoding = "f16";
    cfg.cpu_threads = cpu_threads;
    sa3_context* ctx = sa3_init_ex(&cfg, err, (int)sizeof err);
    if (!ctx) { fprintf(stderr, "sa3_init failed: %s\n", err); return 1; }

    sa3_request req;
    memset(&req, 0, sizeof req);
    req.prompt = prompt;
    req.frames = 128;                 /* ~12 s */
    req.steps = 8;
    req.seed = -1;                    /* random */
    req.cfg_scale = 1.0f;
    req.duration_padding_sec = 6.0f;
    req.keep_models = 1;
    req.on_progress = on_progress;
    /* optional LoRA:
       const char* names[] = { "kev" }; const float strengths[] = { 1.0f };
       req.n_loras = 1; req.lora_names = names; req.lora_strengths = strengths; */

    sa3_audio audio;
    memset(&audio, 0, sizeof audio);
    int rc = sa3_generate(ctx, &req, &audio, err, (int)sizeof err);
    if (rc != 0) { fprintf(stderr, "sa3_generate failed (%d): %s\n", rc, err); sa3_free(ctx); return 1; }

    printf("generated %.2fs, %dch @ %dHz, seed %llu\n",
           (double)audio.n_samp / audio.sample_rate, audio.n_ch, audio.sample_rate,
           (unsigned long long)audio.seed);
    write_wav(out, audio.samples, audio.n_samp, audio.n_ch, audio.sample_rate);
    printf("wrote %s\n", out);

    sa3_free_audio(&audio);
    sa3_free(ctx);
    return 0;
}
