/* sa3-libtest — pure-C smoke test and minimal usage example for the libsa3 V1 ABI.
 *
 * This is the whole call sequence an embedded host (JUCE / IPlug2 plugin, iOS app) uses:
 *
 *   sa3_get_api -> context_create -> request_init -> generate -> use samples
 *                                 -> result_free  -> context_destroy
 *
 * Note the shape every V1 struct shares: zero it, set `size`, then let the library's initializer
 * write the defaults in. That last step is not optional — a meaningful zero in V1 means zero, so
 * the initializer is the only thing that knows what a default is.
 *
 * It generates, then continues what it generated, which is also the shortest demonstration that
 * the library owns duration planning: Generate returns exactly what you asked for, and Continue
 * returns the source plus exactly what you asked to add.
 *
 *   usage: sa3-libtest ["prompt"] [out.wav] [cpu_threads]
 */
#include "libsa3_v1.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void SA3_CALL on_progress(void* user, const sa3_progress_v1* progress) {
    (void)user;
    if (!progress) return;
    printf("  [%3.0f%%] %s %d/%d\n", progress->fraction * 100.0f,
           progress->stage_name ? progress->stage_name : "working",
           progress->step, progress->total);
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

    const sa3_api_v1* api = sa3_get_api(SA3_ABI_VERSION_1);
    if (!api) { fprintf(stderr, "libsa3 does not provide ABI v1\n"); return 1; }
    printf("%s (abi %u)\n", api->runtime_version(), api->abi_version);

    sa3_error_v1 err;
    memset(&err, 0, sizeof err);
    err.size = sizeof err;
    api->error_init(&err);

    sa3_context_config_v1 cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.size = sizeof cfg;
    api->context_config_init(&cfg);
    cfg.variant = "medium";
    cfg.dit_encoding = "f16";
    cfg.cpu_threads = cpu_threads;

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
    req.prompt = prompt;
    req.duration_seconds = 12.0;     /* exactly what comes back */
    req.on_progress = on_progress;
    /* optional adapters, applied over the base for this call only:
       sa3_adapter_v1 a; memset(&a, 0, sizeof a); a.size = sizeof a;
       api->adapter_init(&a); a.path_or_name = "kev"; a.strength = 1.0f;
       req.adapters = &a; req.adapter_count = 1;
       req.adapter_stride = (uint32_t)sizeof(sa3_adapter_v1); */

    sa3_result_v1 result;
    memset(&result, 0, sizeof result);
    result.size = sizeof result;
    api->result_init(&result);

    if (api->generate(ctx, &req, &result, &err) != SA3_STATUS_OK_V1) {
        fprintf(stderr, "generate failed: %s\n", err.message);
        api->context_destroy(ctx);
        return 1;
    }
    printf("generated %.2fs, %uch @ %uHz, seed %llu, peak %.4f\n",
           (double)result.n_samples / result.sample_rate, result.n_channels,
           result.sample_rate, (unsigned long long)result.seed, result.final_peak);
    write_wav(out, result.samples, (int)result.n_samples, (int)result.n_channels,
              (int)result.sample_rate);
    printf("wrote %s\n", out);

    /* Keep a copy as the continuation source, then hand the library's buffer back. Audio is
       library-owned: result_free is the only correct way to release it, DLL boundary included. */
    const uint64_t src_n = result.n_samples;
    const uint32_t src_ch = result.n_channels;
    const uint32_t src_rate = result.sample_rate;
    float* src = (float*)malloc((size_t)src_n * src_ch * sizeof(float));
    if (!src) { fprintf(stderr, "out of memory\n"); api->result_free(&result); api->context_destroy(ctx); return 1; }
    memcpy(src, result.samples, (size_t)src_n * src_ch * sizeof(float));
    api->result_free(&result);

    /* Continue: duration_seconds is what gets ADDED. The library places the regeneration window,
       splices the original source back over the head, and trims to source + added. */
    sa3_request_v1 cont;
    memset(&cont, 0, sizeof cont);
    cont.size = sizeof cont;
    api->request_init(&cont);
    cont.operation = SA3_OPERATION_CONTINUE_V1;
    cont.prompt = prompt;
    cont.duration_seconds = 6.0;
    cont.on_progress = on_progress;
    cont.input_audio.size = sizeof cont.input_audio;
    cont.input_audio.samples = src;
    cont.input_audio.n_samples = src_n;
    cont.input_audio.n_channels = src_ch;
    cont.input_audio.sample_rate = src_rate;
    cont.input_audio.layout = SA3_AUDIO_PLANAR_V1;

    memset(&result, 0, sizeof result);
    result.size = sizeof result;
    api->result_init(&result);
    if (api->generate(ctx, &cont, &result, &err) != SA3_STATUS_OK_V1) {
        fprintf(stderr, "continue failed: %s\n", err.message);
        free(src);
        api->context_destroy(ctx);
        return 1;
    }
    printf("continued to %.2fs; splice %s, kept 0-%.3fs, xfade %.0fms, gain %.3f\n",
           (double)result.n_samples / result.sample_rate,
           result.splice_applied ? "on" : "off", result.splice_end_seconds,
           result.splice_crossfade_seconds * 1000.0f, result.splice_gain);

    {
        char continued[1024];
        snprintf(continued, sizeof continued, "%s.continued.wav", out);
        write_wav(continued, result.samples, (int)result.n_samples, (int)result.n_channels,
                  (int)result.sample_rate);
        printf("wrote %s\n", continued);
    }

    free(src);
    api->result_free(&result);
    api->context_destroy(ctx);
    return 0;
}
