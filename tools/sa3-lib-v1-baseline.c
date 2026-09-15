/* sa3-lib-v1-baseline — a behavioural fingerprint of the V1 ABI, for proving a refactor changed
 * nothing.
 *
 * The contract test covers struct sizes, initializers and error paths, all without models. This is
 * the other half: it runs real generations and prints what came back. Anything that re-points how a
 * request reaches the pipeline — retiring an ABI, collapsing a mapping layer, "tidying" a default —
 * is a change no compiler can check, because a wrong field is a different take rather than a build
 * error. So this measures OUTPUT: exact sample counts, every reported metadata field, and an
 * FNV-1a hash of the decoded audio.
 *
 * Capture it before a change, capture it after, diff the two. Identical or it drifted.
 *
 *   sa3-lib-v1-baseline [adapter.gguf] > before.txt
 *   ... make the change, rebuild ...
 *   sa3-lib-v1-baseline [adapter.gguf] > after.txt
 *   diff before.txt after.txt
 *
 * RUN IT ON CPU. GPU backends are not necessarily run-to-run reproducible — on an AMD Radeon Pro
 * 5300M via Metal the same binary and the same seed hash differently on every run, because a
 * float-ordering difference in the first sampling step compounds into a different take. On that
 * hardware a GPU comparison reports drift that is not there. Set SA3_BASELINE_DEVICE=cpu.
 *
 * Environment:
 *   SA3_MODELS_DIR         where the ggufs live (required)
 *   SA3_BASELINE_DEVICE    "cpu" for a reproducible run; unset uses the default backend
 *   SA3_BASELINE_SECONDS   output length, default 4.0 — 1.0 keeps a CPU run tolerable
 *   SA3_BASELINE_STEPS     sampling steps, default 4 — 2 likewise
 *
 * The optional adapter argument should be a gguf for the medium/SAME-L family; pass none to skip
 * that scenario. The whole set runs in about 100s on CPU at SECONDS=1.0 STEPS=2.
 *
 * Twelve scenarios: all three operations, non-default loudness, frugal residency, every
 * distribution shift on its resolved defaults, explicit shift parameters, an adapter array,
 * cancellation, and two invalid requests.
 */
#include "libsa3_v1.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t hash_audio(const sa3_result_v1* r) {
    if (!r->samples) return 0;
    const unsigned char* p = (const unsigned char*)r->samples;
    size_t n = (size_t)r->n_samples * r->n_channels * sizeof(float);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static void report(const char* label, sa3_status_v1 st, const sa3_result_v1* r) {
    printf("[%s]\n", label);
    printf("  status            %d\n", st);
    if (st != SA3_STATUS_OK_V1) return;
    printf("  n_samples         %llu\n", (unsigned long long)r->n_samples);
    printf("  n_channels        %u\n", r->n_channels);
    printf("  sample_rate       %u\n", r->sample_rate);
    printf("  seed              %llu\n", (unsigned long long)r->seed);
    printf("  decoded_peak      %.6f\n", r->decoded_peak);
    printf("  pk_norm_set/gain  %d %.6f\n", r->peak_normalize_gain_set, r->peak_normalize_gain);
    printf("  lim_set/fraction  %d %.6f\n", r->limiter_limited_fraction_set, r->limiter_limited_fraction);
    printf("  safety_set/gain   %d %.6f\n", r->safety_gain_set, r->safety_gain);
    printf("  final_peak        %.6f\n", r->final_peak);
    printf("  latent_factor     %.6f\n", r->latent_factor);
    printf("  splice_applied    %d\n", r->splice_applied);
    printf("  splice_end_s      %.6f\n", r->splice_end_seconds);
    printf("  splice_xfade_s    %.6f\n", r->splice_crossfade_seconds);
    printf("  splice_gain       %.6f\n", r->splice_gain);
    printf("  mask_start_s      %.6f\n", r->mask_start_seconds);
    printf("  mask_overlap_s    %.6f\n", r->mask_overlap_seconds);
    printf("  audio_fnv1a       %016llx\n", (unsigned long long)hash_audio(r));
}

static int cancel_now(void* u) { (void)u; return 1; }

static const sa3_api_v1* api;

static sa3_context* open_ctx(const char* variant, const char* enc, const char* ae) {
    sa3_error_v1 err; memset(&err, 0, sizeof err); err.size = sizeof err; api->error_init(&err);
    sa3_context_config_v1 c; memset(&c, 0, sizeof c); c.size = sizeof c; api->context_config_init(&c);
    c.models_dir = getenv("SA3_MODELS_DIR");
    c.variant = variant; c.dit_encoding = enc; c.autoencoder_encoding = ae;
    /* Metal on this box is not run-to-run deterministic, so equivalence is checked on CPU. */
    c.device = getenv("SA3_BASELINE_DEVICE");
    sa3_context* ctx = NULL;
    if (api->context_create(&c, &ctx, &err) != SA3_STATUS_OK_V1) {
        printf("context_create(%s) failed: %s\n", variant, err.message); exit(1);
    }
    return ctx;
}

static double g_seconds = 4.0;
static int32_t g_steps = 4;

int main(int argc, char** argv) {
    const char* adapter = argc > 1 ? argv[1] : NULL;
    if (getenv("SA3_BASELINE_SECONDS")) g_seconds = atof(getenv("SA3_BASELINE_SECONDS"));
    if (getenv("SA3_BASELINE_STEPS"))   g_steps   = atoi(getenv("SA3_BASELINE_STEPS"));
    api = sa3_get_api(SA3_ABI_VERSION_1);
    if (!api) { printf("no V1 api\n"); return 1; }
    printf("runtime %s, abi %u\n\n", api->runtime_version(), api->abi_version);

    sa3_error_v1 err; memset(&err, 0, sizeof err); err.size = sizeof err; api->error_init(&err);
    sa3_result_v1 out;
    sa3_context* ctx = open_ctx("small-music", "q4_k_m", NULL);

    /* ---- Generate ---- */
    sa3_request_v1 g; memset(&g, 0, sizeof g); g.size = sizeof g; api->request_init(&g);
    g.operation = SA3_OPERATION_GENERATE_V1;
    g.prompt = "dry beatbox loop"; g.negative_prompt = "vocals";
    g.duration_seconds = g_seconds; g.steps = g_steps; g.seed = 7; g.cfg_scale = 1.0f;
    memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
    report("generate", api->generate(ctx, &g, &out, &err), &out);

    const uint64_t src_n = out.n_samples;
    const uint32_t src_ch = out.n_channels;
    float* src = malloc((size_t)src_n * src_ch * sizeof(float));
    memcpy(src, out.samples, (size_t)src_n * src_ch * sizeof(float));
    api->result_free(&out);

    sa3_audio_view_v1 view; memset(&view, 0, sizeof view);
    view.size = sizeof view; view.samples = src; view.n_samples = src_n;
    view.n_channels = src_ch; view.sample_rate = 44100; view.layout = SA3_AUDIO_PLANAR_V1;

    /* ---- Continue (exercises splice + continuation padding mapping) ---- */
    sa3_request_v1 c; memset(&c, 0, sizeof c); c.size = sizeof c; api->request_init(&c);
    c.operation = SA3_OPERATION_CONTINUE_V1;
    c.prompt = "dry beatbox loop"; c.duration_seconds = g_seconds * 0.75; c.steps = g_steps; c.seed = 7;
    c.input_audio = view;
    memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
    report("continue", api->generate(ctx, &c, &out, &err), &out);
    api->result_free(&out);

    /* ---- Transform (exercises noise level + ignored duration) ---- */
    sa3_request_v1 t; memset(&t, 0, sizeof t); t.size = sizeof t; api->request_init(&t);
    t.operation = SA3_OPERATION_TRANSFORM_V1;
    t.prompt = "dry beatbox loop"; t.duration_seconds = 99.0; t.steps = g_steps; t.seed = 7;
    t.transform_noise_level = 0.6f; t.input_audio = view;
    memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
    report("transform", api->generate(ctx, &t, &out, &err), &out);
    api->result_free(&out);

    /* ---- Non-default loudness + frugal residency + explicit shift params ---- */
    sa3_request_v1 o; memset(&o, 0, sizeof o); o.size = sizeof o; api->request_init(&o);
    o.operation = SA3_OPERATION_GENERATE_V1;
    o.prompt = "dry beatbox loop"; o.duration_seconds = g_seconds; o.steps = g_steps; o.seed = 21;
    o.residency = SA3_RESIDENCY_FRUGAL_V1;
    o.loudness.peak_normalize = 1; o.loudness.peak_normalize_db = -1.0f;
    o.loudness.limiter = 1; o.loudness.limiter_ceiling_db = -2.0f; o.loudness.limiter_knee = 0.5f;
    o.loudness.latent_rescale = 0.98f; o.loudness.latent_shift = 0.01f;
    o.distribution_shift = SA3_DISTRIBUTION_FLUX_V1;
    o.distribution_shift_params[0] = 256.0f; o.distribution_shift_params[1] = 4096.0f;
    o.distribution_shift_params[2] = 0.5f;   o.distribution_shift_params[3] = 1.15f;
    o.decode_chunk_size = 128; o.decode_overlap = 32;
    memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
    report("loudness+frugal+flux", api->generate(ctx, &o, &out, &err), &out);
    api->result_free(&out);

    /* ---- Every distribution shift on its resolved defaults ---- */
    for (int sh = 0; sh <= 3; sh++) {
        sa3_request_v1 d; memset(&d, 0, sizeof d); d.size = sizeof d; api->request_init(&d);
        d.operation = SA3_OPERATION_GENERATE_V1;
        d.prompt = "dry beatbox loop"; d.duration_seconds = g_seconds; d.steps = g_steps; d.seed = 11;
        d.distribution_shift = (sa3_distribution_shift_v1)sh;
        memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
        char label[32]; snprintf(label, sizeof label, "shift-%d", sh);
        report(label, api->generate(ctx, &d, &out, &err), &out);
        api->result_free(&out);
    }

    /* ---- Cancellation ---- */
    sa3_request_v1 x; memset(&x, 0, sizeof x); x.size = sizeof x; api->request_init(&x);
    x.operation = SA3_OPERATION_GENERATE_V1;
    x.prompt = "dry beatbox loop"; x.duration_seconds = g_seconds; x.steps = g_steps; x.seed = 7;
    x.should_cancel = cancel_now;
    memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
    report("cancelled", api->generate(ctx, &x, &out, &err), &out);
    printf("  message           %s\n", err.message);
    api->result_free(&out);

    /* ---- Invalid input: the error paths are part of the contract too ---- */
    sa3_request_v1 bad; memset(&bad, 0, sizeof bad); bad.size = sizeof bad; api->request_init(&bad);
    bad.operation = SA3_OPERATION_CONTINUE_V1; bad.duration_seconds = 1.0;
    memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
    printf("[missing-input] status %d: %s\n", api->generate(ctx, &bad, &out, &err), err.message);

    bad.input_audio = view; bad.transform_noise_level = 5.0f;
    printf("[bad-noise]     status %d: %s\n", api->generate(ctx, &bad, &out, &err), err.message);

    api->context_destroy(ctx);

    /* ---- Adapter array, on the family the decoder adapter belongs to ---- */
    if (adapter) {
        ctx = open_ctx("medium", "q4_k_m", "q4_k_m");
        sa3_request_v1 a; memset(&a, 0, sizeof a); a.size = sizeof a; api->request_init(&a);
        a.operation = SA3_OPERATION_GENERATE_V1;
        a.prompt = "warm analog house groove"; a.duration_seconds = g_seconds; a.steps = g_steps; a.seed = 3;
        sa3_adapter_v1 entries[1]; memset(entries, 0, sizeof entries);
        entries[0].size = sizeof entries[0]; api->adapter_init(&entries[0]);
        entries[0].path_or_name = adapter; entries[0].strength = 0.75f;
        a.adapters = entries; a.adapter_count = 1;
        a.adapter_stride = (uint32_t)sizeof(sa3_adapter_v1);
        memset(&out, 0, sizeof out); out.size = sizeof out; api->result_init(&out);
        report("adapter-0.75", api->generate(ctx, &a, &out, &err), &out);
        api->result_free(&out);
        api->context_destroy(ctx);
    }
    return 0;
}
