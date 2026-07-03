/* libsa3 — a tiny C ABI over the sa3 generation pipeline, for embedding SA3 directly in a host
 * (e.g. a JUCE / IPlug2 plugin) without spawning the CLI or the HTTP server.
 *
 * Lifecycle:   sa3_init() -> sa3_generate()/sa3_generate_ex() [* N] -> sa3_free()
 * Ownership:   sa3_generate()/sa3_generate_ex() allocate sa3_audio.samples; release it with
 *              sa3_free_audio() (same
 *              allocator/CRT as the library — do NOT free() it yourself across a DLL boundary).
 * Threading:   a context is NOT reentrant — serialize sa3_generate() calls per context.
 * Errors:      no C++ exceptions cross this boundary; failures return NULL / non-zero and fill `err`.
 *
 * Runtime deps (ship alongside the host): the sa3 shared lib (sa3.dll / libsa3.so / libsa3.dylib),
 * the ggml backend libs it links (ggml*.dll etc.), and the model gguf set in the models dir.
 */
#ifndef LIBSA3_H
#define LIBSA3_H

#include <stdint.h>

#if defined(_WIN32) && defined(SA3_BUILD_DLL)
#  define SA3_API __declspec(dllexport)
#else
#  define SA3_API   /* consumers link the import lib; dllimport is optional on MSVC */
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sa3_context sa3_context;

/* Model set + backend selection. Any NULL string falls back to a default:
 *   models_dir   -> $SA3_MODELS_DIR, else "models"
 *   adapters_dir -> models_dir
 *   variant      -> "medium"   ("medium" | "small-music" | "small-sfx")
 *   encoding     -> "f16"      ("f16" | "f32")                                                    */
typedef struct {
    const char* models_dir;
    const char* adapters_dir;
    const char* variant;
    const char* encoding;
} sa3_config;

/* Optional progress callback. fraction is overall 0..1 (UI does *100); stage is
 * "sampling" | "decoding" | "done". Called on the sa3_generate() thread. */
typedef void (*sa3_progress_cb)(void* user, const char* stage, int step, int total, float fraction);

/* Optional cooperative cancel callback for embedded hosts. Return non-zero to ask generation to stop
 * at the next safe boundary. An in-flight backend graph may still run until it reaches that boundary. */
typedef int (*sa3_cancel_cb)(void* user);

/* Loudness post-processing. Defaults mirror gary4local (peak-normalize +2 dB, limiter -0.3 dB) — see
 * docs/LOUDNESS.md. Leave `set = 0` (the zero-initialized default) to use those defaults plus any
 * SA3_* env overrides. Set `set = 1` to drive it per-request from the fields below, including RAW,
 * unprocessed output (`peak_normalize = 0` and `limiter = 0`). A final true-peak safety always
 * prevents >0 dBFS output regardless of these values. */
typedef struct {
    int   set;                  /* 0 = library defaults; 1 = use the fields below */
    int   peak_normalize;       /* bool: normalize the decoded peak */
    float peak_normalize_db;    /* target peak, dBFS (e.g. 2.0) */
    int   limiter;              /* bool: soft-knee limiter after normalize */
    float limiter_ceiling_db;   /* limiter ceiling, dBFS (e.g. -0.3) */
    float limiter_knee;         /* knee fraction (0,1]; <=0 -> 0.8 */
    float latent_rescale;       /* advanced: latents = latents*rescale + shift before decode; <=0 -> 1.0 */
    float latent_shift;
} sa3_loudness;

/* One generation request. Zero-initialize it (memset 0), then set what you need — a 0/NULL field
 * means "use the default" shown below. */
typedef struct {
    const char* prompt;             /* NULL/"" = unprompted */
    const char* negative_prompt;    /* NULL/"" = none (only affects output when cfg_scale != 1) */
    int      frames;                /* output latent frames; 0 -> 128 (~12 s; ~10.77 frames/s) */
    int      steps;                 /* 0 -> 8 */
    int64_t  seed;                  /* < 0 -> random; the seed actually used is returned in sa3_audio */
    float    cfg_scale;             /* 0 -> 1.0 (CFG off, single pass). >1 or <1 enables CFG (~2x/step) */
    float    duration_padding_sec;  /* < 0 -> 6.0 headroom (no ending); 0 lets the model end the piece */
    int      keep_models;           /* 1 = keep resident after this call (lower latency next time) */
    sa3_loudness loudness;          /* loudness post-processing; set=0 -> gary4local defaults */

    /* LoRA adapters, parallel arrays of length n_loras. A name resolves to lora-<name>-*.gguf in the
     * adapters dir; a full path is used as-is. lora_strengths may be NULL (all 1.0). */
    int                 n_loras;
    const char* const*  lora_names;
    const float*        lora_strengths;

    sa3_progress_cb on_progress;    /* NULL = none */
    void*           user;           /* passed back to on_progress */

    /* Distribution shift (sampler schedule warp), matching the CLI/server. dist_shift is one of
     * "LogSNR" (default) | "Flux" | "Full" | "None"; NULL/"" -> "LogSNR". dist_shift_params overrides the
     * per-type defaults only when any element is non-zero (all-zero -> use the type's defaults). Per-type
     * meaning follows sa3_pipeline.h dist_shift_warp (e.g. LogSNR = anchor_length, anchor_logsnr, rate, logsnr_end). */
    const char* dist_shift;
    float       dist_shift_params[4];
} sa3_request;

enum {
    SA3_INIT_AUDIO_NONE    = 0,
    SA3_INIT_AUDIO_A2A     = 1,
    SA3_INIT_AUDIO_INPAINT = 2
};

/* Optional audio input for sa3_generate_ex(). `samples` is PLANAR float audio laid out as
 * samples[ch * n_samp + sample] at sample_rate Hz. The pipeline resamples to 44.1 kHz internally.
 *
 * mode = SA3_INIT_AUDIO_A2A:
 *   Audio-to-audio. Output length follows the input audio; init_noise_level <= 0 uses 0.85.
 *
 * mode = SA3_INIT_AUDIO_INPAINT:
 *   Inpainting/continuation. Requires a local-conditioning DiT. The window [inpaint_start,
 *   inpaint_end) is in seconds; audio outside the window is kept, and inpaint_end can extend the
 *   total output past the supplied source audio.
 */
typedef struct {
    int          mode;
    const float* samples;
    int          n_samp;
    int          n_ch;
    int          sample_rate;
    float        init_noise_level;
    float        inpaint_start;
    float        inpaint_end;
} sa3_init_audio;

/* Extended generation request. Keeps the original sa3_request layout intact for ABI stability while
 * exposing the embedded-audio path used by plugin hosts. Zero-initialize this struct, fill `request`
 * like a normal sa3_request, then set init_audio if needed.
 *
 * encode/decode chunk sizes are optional advanced SAME-L controls for long init audio. Leave them 0
 * for the monolithic default; when a chunk size is set, overlap <= 0 falls back to 32.
 *
 * should_cancel/cancel_user are optional and allow plugin hosts to abort long renders during UI/DAW
 * shutdown without unloading the library while a worker thread is still inside libsa3. */
typedef struct {
    sa3_request request;
    sa3_init_audio init_audio;
    int encode_chunk_size;
    int encode_overlap;
    int decode_chunk_size;
    int decode_overlap;
    sa3_cancel_cb should_cancel;
    void* cancel_user;
} sa3_request_ex;

/* Decoded audio. samples is PLANAR by channel: samples[c * n_samp + s]. Free with sa3_free_audio(). */
typedef struct {
    float*   samples;
    int      n_samp;
    int      n_ch;
    int      sample_rate;
    uint64_t seed;                  /* the seed actually used (resolved if the request's seed was < 0) */
} sa3_audio;

/* Load the models. Returns NULL on failure, writing a message into err (if err && err_len > 0). */
SA3_API sa3_context* sa3_init(const sa3_config* cfg, char* err, int err_len);

/* Generate. Returns 0 on success (out filled), non-zero on failure (err filled). Not reentrant. */
SA3_API int sa3_generate(sa3_context* ctx, const sa3_request* req, sa3_audio* out, char* err, int err_len);

/* Extended generate with raw init audio for audio2audio and inpaint/continuation. */
SA3_API int sa3_generate_ex(sa3_context* ctx, const sa3_request_ex* req, sa3_audio* out, char* err, int err_len);

/* Release the sample buffer allocated by sa3_generate(). */
SA3_API void sa3_free_audio(sa3_audio* audio);

/* Free the loaded models (VRAM) but keep the context; the next sa3_generate() reloads them. */
SA3_API void sa3_unload(sa3_context* ctx);

/* Destroy the context (also frees any loaded models). */
SA3_API void sa3_free(sa3_context* ctx);

/* Static version string. */
SA3_API const char* sa3_version(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBSA3_H */
