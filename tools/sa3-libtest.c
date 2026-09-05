/* sa3-libtest — a pure-C smoke test + minimal usage example for libsa3 (see src/libsa3.h).
 * This is exactly the call sequence a JUCE / IPlug2 host would use:
 *   sa3_init -> sa3_generate (with a progress callback) -> use samples -> sa3_free_audio -> sa3_free.
 *   usage: sa3-libtest ["prompt"] [out.wav] [cpu_threads]
 *
 * It then continues that clip through sa3_generate_ex to exercise the continuation splice, and
 * reads sa3_last_meta twice — once with a full struct, once with a deliberately undersized one,
 * which is the only thing that tests the `size` contract at all.
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

    /* Everything the pipeline computed after the sampler. Zero the struct and declare its size;
       what the library does not fill stays zero. */
    sa3_meta meta;
    memset(&meta, 0, sizeof meta);
    meta.size = (uint32_t)sizeof meta;
    if (sa3_last_meta(ctx, &meta) != 0) { fprintf(stderr, "sa3_last_meta failed\n"); sa3_free_audio(&audio); sa3_free(ctx); return 1; }
    printf("meta: filled %u/%u bytes, decoded peak %.4f, normalize gain %.4f, limited %.4f%%, final peak %.4f\n",
           meta.size, (unsigned)sizeof meta, meta.decoded_peak, meta.peak_normalize_gain,
           meta.limiter_limited_fraction * 100.0f, meta.final_peak);

    /* The size contract, from the other side: a caller built against an older header declares a
       SHORTER struct. It must get the prefix it understands and be told how much that was — never
       a write past the end of what it allocated. Casting a truncated struct is exactly what such a
       caller does at the ABI level, which is why the test does it here. */
    {
        struct { uint32_t size; uint64_t seed; float decoded_peak; } old_client;
        memset(&old_client, 0, sizeof old_client);
        old_client.size = (uint32_t)sizeof old_client;
        if (sa3_last_meta(ctx, (sa3_meta*)&old_client) != 0) {
            fprintf(stderr, "sa3_last_meta rejected an undersized struct\n");
            sa3_free_audio(&audio); sa3_free(ctx); return 1;
        }
        if (old_client.size != (uint32_t)sizeof old_client || old_client.decoded_peak != meta.decoded_peak) {
            fprintf(stderr, "undersized sa3_meta came back wrong: size %u, peak %.6f\n",
                    old_client.size, old_client.decoded_peak);
            sa3_free_audio(&audio); sa3_free(ctx); return 1;
        }
        printf("meta: an old client asking for %u bytes got %u back, prefix intact\n",
               (unsigned)sizeof old_client, old_client.size);
    }

    /* ---- continuation: feed the clip back in and extend it, with the source splice on ----
       inpaint_start is where the source ends; the library pulls the sampler's mask back into it by
       mask_overlap and pastes the original samples over everything up to that same point. A
       zero-initialized sa3_splice (set = 0) asks for gary4local's tuned defaults. */
    const int src_n = audio.n_samp, src_ch = audio.n_ch, src_sr = audio.sample_rate;
    float* src = (float*)malloc((size_t)src_n * src_ch * sizeof(float));
    if (!src) { fprintf(stderr, "out of memory\n"); sa3_free_audio(&audio); sa3_free(ctx); return 1; }
    memcpy(src, audio.samples, (size_t)src_n * src_ch * sizeof(float));
    sa3_free_audio(&audio);

    const float src_seconds = (float)src_n / (float)src_sr;
    sa3_request_ex rx;
    memset(&rx, 0, sizeof rx);
    rx.request = req;
    rx.init_audio.mode = SA3_INIT_AUDIO_INPAINT;
    rx.init_audio.samples = src;
    rx.init_audio.n_samp = src_n;
    rx.init_audio.n_ch = src_ch;
    rx.init_audio.sample_rate = src_sr;
    rx.init_audio.inpaint_start = src_seconds;
    rx.init_audio.inpaint_end = src_seconds + 6.0f;   /* six more seconds of music */

    sa3_audio cont;
    memset(&cont, 0, sizeof cont);
    rc = sa3_generate_ex(ctx, &rx, &cont, err, (int)sizeof err);
    if (rc != 0) { fprintf(stderr, "sa3_generate_ex failed (%d): %s\n", rc, err); free(src); sa3_free(ctx); return 1; }

    memset(&meta, 0, sizeof meta);
    meta.size = (uint32_t)sizeof meta;
    if (sa3_last_meta(ctx, &meta) != 0) { fprintf(stderr, "sa3_last_meta failed\n"); free(src); sa3_free_audio(&cont); sa3_free(ctx); return 1; }
    printf("continued to %.2fs; splice %s, kept 0-%.3fs, xfade %.0fms, gain %.3f, mask start %.3fs (pullback %.3fs)\n",
           (double)cont.n_samp / cont.sample_rate, meta.splice_applied ? "on" : "off",
           meta.splice_end_seconds, meta.splice_xfade_applied * 1000.0f, meta.splice_gain,
           meta.mask_start_seconds, meta.mask_overlap_applied);
    if (!meta.splice_applied) { fprintf(stderr, "splice did not run on a continuation\n"); free(src); sa3_free_audio(&cont); sa3_free(ctx); return 1; }

    {
        char cont_out[1024];
        snprintf(cont_out, sizeof cont_out, "%s.continued.wav", out);
        write_wav(cont_out, cont.samples, cont.n_samp, cont.n_ch, cont.sample_rate);
        printf("wrote %s\n", cont_out);
    }

    free(src);
    sa3_free_audio(&cont);
    sa3_free(ctx);
    return 0;
}
