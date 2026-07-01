// sa3_pipeline.h — the reusable SA3 generation pipeline.
//
// Load the four nets + conditioner ONCE, then call generate() per request. This is the single
// code path behind all three surfaces, so they share behavior + the config/resolution vocabulary:
//   - sa3-generate (CLI)   : parse args -> Pipeline::load -> generate -> write_wav
//   - sa3-server (HTTP)    : load once -> generate() per POST /generate
//   - libsa3 (C ABI)       : sa3_init -> sa3_generate -> sa3_free, wrapping a Pipeline
//
// Residency is a per-request policy (GenParams::keep_models), NOT a surface-specific thing — the
// server and the C API both expose it:
//   - keep_models=true  (resident): nothing freed between requests -> lowest latency, but peak VRAM
//                                    caps gen length on small cards. Best for back-to-back ~12s clips.
//   - keep_models=false (frugal):   free T5 before sampling + DiT before decode (the CLI's trick), then
//                                    reload them at the next generate() (~0.5-1.5s). Costs a reload per
//                                    request but fits long-form on 8 GB (e.g. gary4local). generate()
//                                    lazily (re)loads whatever a prior frugal call freed, so the flag
//                                    can vary per call. Reloading the DiT also re-bases it for that
//                                    request's LoRAs.
//
// Threading: generate() runs one ggml graph at a time and mutates the DiT's LoRA overrides, so a
// single Pipeline is NOT safe to call concurrently — the server serializes requests (a queue).
//
// Style: header-only (inline) to match the rest of src/ — the CLI, server, and C API just #include
// this. (Open question for the docs: revisit a sa3_pipeline.cpp if incremental build time bites.)
#pragma once

#include "gguf_model.h"
#include "tokenizer.h"
#include "t5gemma.h"
#include "dit.h"
#include "same_ae.h"
#include "lora.h"
#include "rng.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sa3 {

// ---- small shared helpers (used by generate(); also handy to the server/CLI) ----

// ExpoFourierFeatures(dim, min, max): [cos, sin](t * f * 2pi), f log-spaced.
inline void expo_features(float t, std::vector<float>& out, int dim, float min_f, float max_f) {
    const int half = dim / 2;
    const float lmin = logf(min_f), lmax = logf(max_f), TWO_PI = 6.28318530717958648f;
    out.resize(dim);
    for (int i = 0; i < half; i++) {
        float ramp = half > 1 ? (float)i / (half - 1) : 0.0f;
        float freq = expf(ramp * (lmax - lmin) + lmin);
        float arg = t * freq * TWO_PI;
        out[i] = cosf(arg); out[half + i] = sinf(arg);
    }
}

// SA3 sampling-schedule distribution shift. Warps ONE linear-schedule t in (0,1) to the shifted
// t, matching stable_audio_3/inference/distribution_shift.py for the scalar (batch=1, sigma=1,
// use_sine=False) case we always hit. Endpoints (t=0/1) are re-anchored by the caller, so this is
// only ever called for interior t (no div-by-zero in Full, no log(0)). seq_len = latent frames T.
//   type:  "LogSNR" | "Flux" | "Full" | "None"   (anything else => identity)
//   LogSNR: p=(anchor_length, anchor_logsnr, rate, logsnr_end)
//   Flux:   p=(min_length, max_length, alpha_min, alpha_max)
//   Full:   p=(base_shift, max_shift, min_length, max_length)
inline float dist_shift_warp(const std::string& type, float t, int seq_len,
                             float p1, float p2, float p3, float p4) {
    if (type == "LogSNR") {
        const float anchor_length = p1, anchor_logsnr = p2, rate = p3, logsnr_end = p4;
        const float logsnr_start = anchor_logsnr - rate * std::log2((float)seq_len / anchor_length);
        const float logsnr = logsnr_end - t * (logsnr_end - logsnr_start);
        return 1.0f / (1.0f + std::exp(logsnr));            // sigmoid(-logsnr)
    }
    if (type == "Flux") {
        const float min_length = p1, max_length = p2, alpha_min = p3, alpha_max = p4;
        const float sl = std::min(std::max((float)seq_len, min_length), max_length);
        const float lmin = std::log(min_length);
        float lmax = std::log(max_length);
        if (lmax == lmin) lmax += 1e-8f;                    // constant-alpha guard (upstream)
        const float frac = (std::log(sl) - lmin) / (lmax - lmin);
        const float log_amin = std::log(std::max(alpha_min, 1e-8f));
        const float log_amax = std::log(std::max(alpha_max, 1e-8f));
        const float alpha = std::exp(log_amin + frac * (log_amax - log_amin));
        return alpha * t / (1.0f + (alpha - 1.0f) * t);
    }
    if (type == "Full") {
        const float base_shift = p1, max_shift = p2, min_length = p3, max_length = p4;
        const float sl = std::min(std::max((float)seq_len, min_length), max_length);
        const float mu = -(base_shift + (max_shift - base_shift) * (sl - min_length) / (max_length - min_length));
        const float em = std::exp(mu);
        return 1.0f - em / (em + t / (1.0f - t));           // (1/(1-t)-1) == t/(1-t), sigma=1
    }
    return t;                                               // "None" / unknown => identity
}

// Per-type defaults for the 4 dist-shift params, from the official SA3 configs / paper. The
// CLI/server call this when the user selects a type without explicit params (mirrors the gradio,
// where each type's sliders have their own defaults). LogSNR = the medium model default.
inline void dist_shift_defaults(const std::string& type, float& p1, float& p2, float& p3, float& p4) {
    if      (type == "Flux") { p1 = 256.0f; p2 = 4096.0f; p3 = 6.93f;  p4 = 6.93f;  } // Self-Flow audio sampleshift
    else if (type == "Full") { p1 = 0.5f;   p2 = 1.15f;   p3 = 256.0f; p4 = 4096.0f; } // HF SA3 config (all variants)
    else if (type == "None") { p1 = p2 = p3 = p4 = 0.0f; }
    else                     { p1 = 2000.0f; p2 = -6.2f;  p3 = 0.0f;   p4 = 2.0f;   } // LogSNR (medium, rate=0)
}

// Host classifier-free-guidance combine for the rf_denoiser objective, matching models/dit.py
// 579-619 (the scalar/no-padding-mask case). v_cond/v_uncond are the two velocity predictions and
// x is the current latent; sigma is the step's t. Writes the guided velocity to v_out. Layout is
// [io, T] flattened as idx = t*io + c (channel fastest). Callers skip this when cfg_scale == 1.0.
//   apg_scale: 1.0 = full APG (orthogonal only), 0.0 = vanilla CFG (full diff), else blend.
//   cfg_norm_threshold: >0 clamps the guidance-delta L2 norm. scale_phi: CFG-rescale toward cond std.
inline void cfg_combine(std::vector<float>& v_out,
                        const std::vector<float>& v_cond, const std::vector<float>& v_uncond,
                        const std::vector<float>& x, int N, int io, int T, float sigma,
                        float cfg_scale, float apg_scale, float scale_phi, float cfg_norm_threshold) {
    std::vector<float> cond_den(N), diff(N);
    for (int j = 0; j < N; j++) {
        cond_den[j] = x[j] - v_cond[j] * sigma;
        diff[j]     = (v_uncond[j] - v_cond[j]) * sigma;      // == cond_denoised - uncond_denoised
    }
    if (cfg_norm_threshold > 0.0f) {                          // clamp ||diff|| to the threshold
        double s = 0; for (int j = 0; j < N; j++) s += (double)diff[j]*diff[j];
        float dn = (float)std::sqrt(s);
        if (dn > cfg_norm_threshold) { float sf = cfg_norm_threshold / dn; for (int j = 0; j < N; j++) diff[j] *= sf; }
    }
    std::vector<float> cfg_diff(N);
    if (apg_scale == 0.0f) {
        cfg_diff = diff;                                      // vanilla CFG
    } else {                                                 // APG: project diff off the cond_denoised direction
        double cn2 = 0; for (int j = 0; j < N; j++) cn2 += (double)cond_den[j]*cond_den[j];
        double dot = 0; for (int j = 0; j < N; j++) dot += (double)diff[j]*cond_den[j];
        float proj = cn2 > 1e-16 ? (float)(dot / cn2) : 0.0f; // <diff,cond_hat>/||cond_den|| coefficient
        for (int j = 0; j < N; j++) {
            float orth = diff[j] - proj * cond_den[j];
            cfg_diff[j] = (apg_scale == 1.0f) ? orth : apg_scale*orth + (1.0f - apg_scale)*diff[j];
        }
    }
    for (int j = 0; j < N; j++) {
        float cfg_den = cond_den[j] + (cfg_scale - 1.0f) * cfg_diff[j];
        v_out[j] = (x[j] - cfg_den) / sigma;
    }
    if (scale_phi != 0.0f) {                                 // CFG rescale, per-time std over channels
        for (int t = 0; t < T; t++) {
            const int base = t*io;
            double c1=0,c2=0,o1=0,o2=0;
            for (int c = 0; c < io; c++) { float a=v_cond[base+c], b=v_out[base+c]; c1+=a; c2+=(double)a*a; o1+=b; o2+=(double)b*b; }
            float cstd = (float)std::sqrt(std::max(0.0, c2/io - (c1/io)*(c1/io)));
            float ostd = (float)std::sqrt(std::max(0.0, o2/io - (o1/io)*(o1/io)));
            float r = ostd > 1e-8f ? cstd/ostd : 1.0f;
            for (int c = 0; c < io; c++) { float b=v_out[base+c]; v_out[base+c] = scale_phi*(b*r) + (1.0f - scale_phi)*b; }
        }
    }
}

// Resolve a requested seed to a concrete one: any negative value (the -1 "random" sentinel, matching
// the official SA3 / gary convention) draws a fresh seed. The CLI/server should print the returned
// value so a good result stays reproducible.
inline uint64_t pick_seed(long long requested) {
    if (requested >= 0) return (uint64_t)requested;
    std::random_device rd;
    return ((uint64_t)rd() << 32) ^ (uint64_t)rd()
         ^ (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline double wall_time_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline bool profile_enabled() {
    const char* p = getenv("SA3_PROFILE");
    return p && strcmp(p, "0") != 0;
}

inline void profile_log(bool enabled, const char* label, double seconds) {
    if (enabled) fprintf(stderr, "[sa3-profile] %-18s %8.3f ms\n", label, seconds * 1000.0);
}

inline std::vector<float> tensor_to_host(const GgufModel& M, const std::string& name) {
    ggml_tensor* t = M.get(name);
    std::vector<float> b(ggml_nelements(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size() * sizeof(float));
    return b;
}

// SAME-L sliding-window bias upload; no-op when the mask tensor is unused (SAME-S block-diagonal).
inline void set_swa_bias(ggml_tensor* mt, const SameConfig& sc, int64_t N) {
    if (!mt->buffer || sc.chunk) return;
    if (mt->type == GGML_TYPE_F16) {
        const bool compact = mt->ne[0] == 3*sc.sub_chunk && mt->ne[1] == sc.sub_chunk;
        std::vector<ggml_fp16_t> bias = compact ? build_swa_bias_f16(sc, N) : build_swa_full_bias_f16(sc, N);
        ggml_backend_tensor_set(mt, bias.data(), 0, bias.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<float> bias = build_swa_bias(sc, N);
        ggml_backend_tensor_set(mt, bias.data(), 0, bias.size() * sizeof(float));
    }
}

// Find the one file in `dir` whose name starts with `prefix` and ends with `suffix`. "" if none;
// returns "" + sets ambiguous=true if >1 match. Resolves --model / --lora by the naming convention.
inline std::string resolve_one(const std::string& dir, const std::string& prefix,
                               const std::string& suffix, bool* ambiguous = nullptr) {
    namespace fs = std::filesystem;
    std::string found; std::error_code ec;
    if (ambiguous) *ambiguous = false;
    if (!fs::is_directory(dir, ec)) return found;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.size() >= prefix.size() + suffix.size()
            && n.compare(0, prefix.size(), prefix) == 0
            && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
            if (!found.empty()) { if (ambiguous) *ambiguous = true; return found; }
            found = e.path().string();
        }
    }
    return found;
}

// The five gguf paths a generation needs. Resolve by the naming convention (--model / download_models.py)
// or set explicitly (CLI --tok/--dit/... overrides). resolve() globs <models_dir> by prefix+suffix so the
// size label (1.5B/0.5B) and exact encoder name aren't hardcoded.
struct ModelPaths {
    std::string tok;     // t5gemma-b-b-ul2-v1.0-vocab.gguf
    std::string t5;      // t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf  (shared)
    std::string cond;    // stable-audio-3-<variant>-conditioner-v1.0-F32.gguf  (empty => read from t5)
    std::string dit;     // stable-audio-3-<variant>-dit-*-<ENC>.gguf
    std::string same;    // stable-audio-3-<variant>-same-*-<ENC>.gguf

    // variant in {medium, small-music, small-sfx}; encoding in {f16, f32}. Returns false + a message
    // in `err` if a required file is missing (so the caller can surface a friendly hint).
    static bool resolve(const std::string& models_dir, const std::string& variant,
                        const std::string& encoding, ModelPaths& out, std::string& err);
};

inline bool ModelPaths::resolve(const std::string& md, const std::string& variant,
                                const std::string& encoding, ModelPaths& out, std::string& err) {
    const std::string ENC = (encoding == "f32" || encoding == "F32") ? "F32" : "F16";
    auto one = [&](const std::string& prefix, const std::string& suffix, const char* what) {
        std::string p = resolve_one(md, prefix, suffix);
        if (p.empty())
            err += (err.empty() ? "" : "; ") + ("no " + std::string(what) + " (" + prefix + "*" + suffix + ")");
        return p;
    };
    out.tok  = one("t5gemma-b-b-ul2-v1.0-vocab",             ".gguf",            "tokenizer");
    out.t5   = one("t5gemma-b-b-ul2-encoder-",               ".gguf",            "encoder");
    out.cond = one("stable-audio-3-" + variant + "-conditioner-", ".gguf",       "conditioner");
    out.dit  = one("stable-audio-3-" + variant + "-dit-",  "-" + ENC + ".gguf",  "DiT");
    out.same = one("stable-audio-3-" + variant + "-same-", "-" + ENC + ".gguf",  "SAME");
    if (!err.empty()) err += " in " + md + "/ (run: python tools/download_models.py --variant " + variant + ")";
    return err.empty();
}

// Progress tick passed to GenParams::on_progress. `fraction` is a ready-to-use overall [0,1] (UIs do
// fraction*100) so a consumer needs no knowledge of the internal stages; step/total/stage give detail.
// Curve mirrors gary's poll_status: sampling spans 0..0.9, decode 0.9..1.0, then a final "done" at 1.0.
struct Progress {
    const char* stage;   // "sampling" | "decoding" | "done"
    int   step;          // completed units in this stage
    int   total;         // total units in this stage
    float fraction;      // overall progress in [0,1]
};

// One generation request. Defaults reproduce the CLI's text2music defaults.
struct GenParams {
    std::string prompt;
    int      frames = 128;             // output length in latent frames (EVEN for SAME-S); 128 ~= 12 s
    int      steps  = 8;
    uint64_t seed   = 0;

    // audio2audio / inpaint (optional). init_audio is planar [init_n_samp, init_n_ch] @ 44.1 kHz;
    // empty => text2music. Length-derivation (T from the source, padding) happens inside generate().
    std::vector<float> init_audio;
    int   init_n_samp = 0, init_n_ch = 0;
    float init_noise_level = 0.85f;    // sigma_max for a2a (1.0 == text2music)
    float inpaint_start = -1.0f;       // inpaint region in seconds; needs init_audio + a local-cond DiT
    float inpaint_end   = -1.0f;       // also the TOTAL output duration (a short clip can extend)

    // Adapters applied (in order) for THIS request, then reset. Paths are full gguf paths
    // (the CLI/server resolve names -> paths before building GenParams).
    std::vector<std::pair<std::string, float>> loras;   // (gguf path, strength)

    // Sampling-schedule distribution shift — warps the linear t schedule. Mirrors the official
    // SA3 gradio selector {LogSNR, Flux, Full, None}; the 4 params' meaning is per-type (see
    // sa3::dist_shift_warp / dist_shift_defaults). The defaults below are the medium model's
    // LogSNR with rate=0, i.e. byte-identical to the previously-hardcoded schedule.
    std::string dist_shift = "LogSNR";   // "LogSNR" | "Flux" | "Full" | "None"
    float ds_p1 = 2000.0f;  // LogSNR:anchor_length  Flux:min_length  Full:base_shift
    float ds_p2 = -6.2f;    // LogSNR:anchor_logsnr  Flux:max_length  Full:max_shift
    float ds_p3 = 0.0f;     // LogSNR:rate           Flux:alpha_min   Full:min_length
    float ds_p4 = 2.0f;     // LogSNR:logsnr_end     Flux:alpha_max   Full:max_length

    // Schedule headroom (text2music only). Generate a (frames + duration_padding) canvas, condition
    // seconds_total + the dist-shift schedule on the REQUESTED length, then truncate back to frames.
    // Matches the official SA3 default (6.0). 0 lets the model "end" the piece (silence/fade tail);
    // >0 leaves room so the kept region has no ending — the basis for continuation / loop generation.
    float duration_padding_sec = 6.0f;

    // Classifier-free guidance (matches dit.py:479-619). cfg_scale==1.0 => a single conditioned pass
    // (no CFG, byte-identical to before). Otherwise each step also runs an UNconditioned pass (negative
    // prompt if given, else null) and guides. The knobs only bite when cfg_scale != 1.0.
    std::string negative_prompt;
    float cfg_scale        = 1.0f;    // guidance strength; 1.0 = off
    float cfg_rescale      = 0.0f;    // scale_phi: rescale the guided output toward the conditioned std
    float cfg_interval_min = 0.0f;    // only apply CFG when the step t is within [min, max]
    float cfg_interval_max = 1.0f;
    float apg_scale        = 1.0f;    // 1.0 = full APG (orthogonal), 0.0 = vanilla CFG, else blend
    float cfg_norm_threshold = 0.0f;  // >0 clamps the guidance-delta L2 norm

    // Long-audio decode tiling; 0 = monolithic (the sliding-window decoder is already linear).
    int decode_chunk_size = 0;
    int decode_overlap    = 32;

    // Residency for THIS request (see the header banner). Default true = resident (keep models loaded
    // for the next request — the right default when a server/lib is reused). Set false for frugal
    // early-free: fits long-form on small VRAM at the cost of a reload next request. The one-shot CLI
    // sets this false by default (it frees before exit anyway).
    bool keep_models = true;

    // Optional progress hook (transport-agnostic: a server can poll it or push SSE). Called on the
    // generate() thread, one tick per sampling step + decode; see Progress for the ready fraction.
    std::function<void(const Progress&)> on_progress;
};

struct GenResult {
    std::vector<float> samples;        // planar [n_samp, n_ch]
    int n_samp = 0;
    int n_ch = 2;
    int sample_rate = 44100;
};

// Holds the loaded models for the life of the process. Move-only (owns the backend + buffers).
class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Load all nets onto one shared backend (the GPU if available; SA3_DEVICE=cpu forces CPU).
    // Throws std::runtime_error on a missing/!@#$ file. Idempotent guard: load() once per Pipeline.
    void load(const ModelPaths& paths);
    bool loaded() const { return loaded_; }

    // Run one generation. At entry it (re)loads any net a prior frugal (keep_models=false) call freed,
    // so keep_models can vary per call. Applies params.loras over the base DiT for this call and resets
    // after, so consecutive calls may use different adapters. Serialize calls per Pipeline.
    GenResult generate(const GenParams& params);

    // Cheap accessors a server/health-check wants.
    const SameConfig& same_config() const { return sc_; }
    const DitConfig&  dit_config()  const { return dc_; }

private:
    void ensure_nets_loaded();         // (re)load T5/DiT/SAME/cond from paths_ if a frugal call freed them

    bool loaded_ = false;
    bool nets_resident_ = false;       // are the heavy nets currently loaded? (false after a frugal gen)
    ModelPaths paths_;                 // kept so frugal mode can reload T5/DiT freed mid-request
    ggml_backend_t backend_ = nullptr;
    Tokenizer tok_;
    GgufModel TE_, DIT_, AE_;
    std::unique_ptr<GgufModel> CD_;    // per-variant conditioner sidecar, or null => use TE_
    std::vector<std::pair<std::string,float>> dit_loras_;  // adapters currently baked into the live DiT (in-place)
    T5GemmaConfig tc_{};
    DitConfig     dc_{};
    SameConfig    sc_{};
};

// ---- Pipeline implementation (header-only) ----

inline void Pipeline::ensure_nets_loaded() {
    if (nets_resident_) return;
    TE_  = load_gguf(paths_.t5.c_str(),   backend_);
    DIT_ = load_gguf(paths_.dit.c_str(),  backend_);
    AE_  = load_gguf(paths_.same.c_str(), backend_);
    CD_  = paths_.cond.empty() ? nullptr
                               : std::make_unique<GgufModel>(load_gguf(paths_.cond.c_str(), backend_));
    dit_loras_.clear();        // a freshly loaded DiT carries no adapters
    nets_resident_ = true;
}

inline void Pipeline::load(const ModelPaths& paths) {
    if (loaded_) return;
    paths_ = paths;
    backend_ = make_backend();
    tok_ = Tokenizer::load(paths_.tok.c_str());
    ensure_nets_loaded();
    tc_ = T5GemmaConfig::from(TE_);
    dc_ = DitConfig::from(DIT_);
    sc_ = SameConfig::from(AE_);
    loaded_ = true;
}

inline Pipeline::~Pipeline() {
    if (backend_) {
        if (nets_resident_) { TE_.free(); DIT_.free(); AE_.free(); if (CD_) CD_->free(); }
        ggml_backend_free(backend_);
    }
}

inline Pipeline::Pipeline(Pipeline&& o) noexcept { *this = std::move(o); }
inline Pipeline& Pipeline::operator=(Pipeline&& o) noexcept {
    if (this != &o) {
        loaded_ = o.loaded_; nets_resident_ = o.nets_resident_;
        paths_ = std::move(o.paths_); backend_ = o.backend_;
        tok_ = std::move(o.tok_); TE_ = std::move(o.TE_); DIT_ = std::move(o.DIT_); AE_ = std::move(o.AE_);
        CD_ = std::move(o.CD_); tc_ = o.tc_; dc_ = o.dc_; sc_ = o.sc_;
        o.backend_ = nullptr; o.loaded_ = false; o.nets_resident_ = false;
    }
    return *this;
}

inline GenResult Pipeline::generate(const GenParams& params) {
    const bool prof = profile_enabled();
    ensure_nets_loaded();   // reload anything a prior frugal (keep_models=false) call freed

    // aliases so the moved body reads exactly like the original CLI generation code
    GgufModel& TE = TE_; GgufModel& DIT = DIT_; GgufModel& AE = AE_;
    const GgufModel& CD = CD_ ? *CD_ : TE_;
    const T5GemmaConfig& tc = tc_;
    const DitConfig& dc = dc_;
    const SameConfig& sc = sc_;
    ggml_backend_t shared_backend = backend_;

    const std::string& prompt = params.prompt;
    int frames = params.frames;
    const int steps = params.steps;
    const uint64_t seed = params.seed;
    const bool keep_models = params.keep_models;
    const float init_noise_level = params.init_noise_level;
    const float inpaint_start = params.inpaint_start, inpaint_end = params.inpaint_end;
    const int decode_chunk_size = params.decode_chunk_size, decode_overlap = params.decode_overlap;
    const std::string& dist_shift = params.dist_shift;
    const float ds_p1 = params.ds_p1, ds_p2 = params.ds_p2, ds_p3 = params.ds_p3, ds_p4 = params.ds_p4;
    const float duration_padding_sec = params.duration_padding_sec;
    const std::string& negative_prompt = params.negative_prompt;
    const float cfg_scale = params.cfg_scale, cfg_rescale = params.cfg_rescale;
    const float cfg_interval_min = params.cfg_interval_min, cfg_interval_max = params.cfg_interval_max;
    const float apg_scale = params.apg_scale, cfg_norm_threshold = params.cfg_norm_threshold;
    const bool do_cfg = (cfg_scale != 1.0f);
    const bool inpaint = (inpaint_start >= 0.0f || inpaint_end >= 0.0f);
    const bool has_init = !params.init_audio.empty();

    int same_l_flash_mode = sc.chunk ? 0 : nn::same_flash_attn_mode();
    if (same_l_flash_mode == 2) {
        const char* bn = ggml_backend_name(shared_backend);
        if (bn && (strstr(bn, "CUDA") || strstr(bn, "ROCm") || strstr(bn, "HIP"))) {
            fprintf(stderr, "[sa3] SA3_SAME_FLASH_ATTN=local is unsupported on %s; falling back to full\n", bn);
            same_l_flash_mode = 1;
        }
    }
    const bool same_l_flash_attn = same_l_flash_mode != 0;
    const int ds = sc.patch_size * sc.output_seg;

    // ---------- LoRA/DoRA adapters (chained in order) ----------
    // apply_loras is IN-PLACE over the DiT base, so changing adapters/strength needs a clean base.
    // Only reload + re-apply when this request's set differs from what's already baked into the DiT
    // (so resident back-to-back requests with the SAME adapters skip the work; a change pays a reload).
    if (dit_loras_ != params.loras) {
        if (!dit_loras_.empty()) {           // DiT carries a prior request's adapters -> reload a clean base
            DIT.free();
            DIT_ = load_gguf(paths_.dit.c_str(), backend_);
            dit_loras_.clear();
        }
        if (!params.loras.empty()) {
            std::vector<sa3::LoraAdapter> adapters;
            for (auto& ls : params.loras) adapters.push_back(sa3::load_lora(ls.first.c_str(), ls.second, DIT.backend));
            sa3::apply_loras(DIT, adapters);
            printf("lora: applied %zu adapter(s):\n", adapters.size());
            for (size_t i = 0; i < adapters.size(); i++)
                printf("  [%zu] %s  type=%s strength=%.2f\n", i, params.loras[i].first.c_str(),
                       adapters[i].type.c_str(), adapters[i].strength);
        }
        dit_loras_ = params.loras;
    }

    // ---------- init audio: pad + derive output T (overrides params.frames) ----------
    std::vector<float> init_audio; int init_L = 0;
    if (has_init) {
        const int n_samp = params.init_n_samp, n_ch = params.init_n_ch;
        const std::vector<float>& raw = params.init_audio;
        if (n_ch != sc.out_channels / sc.patch_size)
            throw std::runtime_error("init audio must be " + std::to_string(sc.out_channels / sc.patch_size) + "-channel");
        int want = n_samp;
        if (inpaint && inpaint_end > 0.0f) want = std::max(want, (int)(inpaint_end * 44100.0f));
        const int mult = sc.chunk ? 2 * ds : ds;
        init_L = ((want + mult - 1) / mult) * mult;
        init_audio.assign((size_t)init_L * n_ch, 0.0f);
        const int copy = std::min(n_samp, init_L);
        for (int c = 0; c < n_ch; c++)
            memcpy(&init_audio[(size_t)c*init_L], &raw[(size_t)c*n_samp], copy * sizeof(float));
        frames = init_L / ds;
        printf("%s: init %.2fs -> output T=%d (%.2fs)\n",
               inpaint ? "inpaint" : "audio2audio", (float)n_samp/44100.0f, frames, (float)init_L/44100.0f);
    }

    // text2music: generate a (frames + duration_padding) canvas so the model isn't forced to "end"
    // the piece in the kept region, then truncate to `frames` at the very end. eff_frames (= the
    // requested length) drives seconds_total conditioning + the dist-shift schedule (upstream's
    // use_effective_length_for_schedule); T (the canvas) drives all latent/DiT/decode sizing. a2a/
    // inpaint derive frames from the init audio, so they keep pad=0.
    const int max_len = (int)TE.u32("t5g.max_length");
    const int eff_frames = frames;
    int pad_frames = 0;
    if (!has_init && duration_padding_sec > 0.0f) {
        const int per = sc.patch_size * sc.output_seg;          // audio samples per latent frame
        const int mult = sc.chunk ? 2 : 1;                      // SAME-S needs an even latent length
        int pf = (int)(duration_padding_sec * 44100.0f / (float)per + 0.5f);
        pad_frames = ((pf + mult - 1) / mult) * mult;
    }
    const int T = eff_frames + pad_frames;
    const int cond_dim = tc.dim, ctx_len = max_len + 1;
    const int N = T * dc.io;
    if (same_l_flash_attn) {
        const int64_t n_same = (int64_t)T * sc.sub_chunk;
        const int64_t mask_elems = same_l_flash_mode == 2 ? (int64_t)3 * sc.sub_chunk * n_same : n_same * n_same;
        const double mask_mib = (double)mask_elems * sizeof(ggml_fp16_t) / (1024.0 * 1024.0);
        fprintf(stderr, "[sa3] SAME-L %s flash attention enabled: F16 band mask %.1f MiB\n",
                same_l_flash_mode == 2 ? "local" : "full", mask_mib);
    }
    const float sigma_max = (has_init && !inpaint) ? init_noise_level : 1.0f;
    if (inpaint && (!has_init || dc.local_dim <= 0))
        throw std::runtime_error("inpaint needs init audio + a DiT with local-cond weights (dit.local_dim>0)");

    // ---------- tokenize + T5Gemma encode (positive, and negative for CFG) ----------
    auto tokenize = [&](const std::string& p, std::vector<int32_t>& ids, std::vector<int32_t>& attn) {
        std::vector<int32_t> e = tok_.encode(p);
        const int l = std::min((int)e.size(), max_len);
        ids.assign(max_len, tok_.pad_id); attn.assign(max_len, 0);
        for (int i = 0; i < l; i++) { ids[i] = e[i]; attn[i] = 1; }
        return l;
    };
    std::vector<int32_t> ids, attn;
    const int L = tokenize(prompt, ids, attn);
    printf("prompt: \"%s\"  (%d tokens, ~%.2fs)\n", prompt.c_str(), L, (float)eff_frames * (sc.patch_size * sc.output_seg) / 44100.0f);
    if (pad_frames) printf("  + %.2fs schedule headroom (canvas T=%d, truncated back to %d)\n",
                           (float)pad_frames * (sc.patch_size * sc.output_seg) / 44100.0f, T, eff_frames);

    // Encode one prompt's token ids -> T5Gemma hidden [cond_dim*max_len].
    auto t5_encode = [&](const std::vector<int32_t>& ids_v, const std::vector<int32_t>& attn_v) {
        std::vector<float> hidden((size_t)cond_dim*max_len);
        const double t_t5_total = wall_time_s();
        double tp = wall_time_s();
        ggml_init_params ip = { (size_t)256*1024*1024, nullptr, true };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* ids_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len);
        ggml_tensor* pos_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_len);
        ggml_tensor* mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, max_len, max_len);
        for (ggml_tensor* t : {ids_t, pos_t, mask_t}) ggml_set_input(t);
        ggml_tensor* h = ggml_cont(ctx, sa3::t5gemma_encode(ctx, TE, ids_t, pos_t, mask_t, tc));
        ggml_set_output(h);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(gf, h);
        profile_log(prof, "t5_build", wall_time_s() - tp);
        tp = wall_time_s();
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(TE.backend));
        ggml_gallocr_alloc_graph(alloc, gf);
        profile_log(prof, "t5_alloc", wall_time_s() - tp);
        tp = wall_time_s();
        ggml_backend_tensor_set(ids_t, ids_v.data(), 0, max_len*sizeof(int32_t));
        std::vector<int32_t> pos(max_len); for (int i = 0; i < max_len; i++) pos[i] = i;
        ggml_backend_tensor_set(pos_t, pos.data(), 0, max_len*sizeof(int32_t));
        std::vector<float> mb((size_t)max_len*max_len);
        for (int q = 0; q < max_len; q++) for (int k = 0; k < max_len; k++)
            mb[(size_t)q*max_len+k] = attn_v[k] ? 0.0f : -INFINITY;
        ggml_backend_tensor_set(mask_t, mb.data(), 0, mb.size()*sizeof(float));
        profile_log(prof, "t5_upload", wall_time_s() - tp);
        tp = wall_time_s();
        ggml_backend_graph_compute(TE.backend, gf);
        profile_log(prof, "t5_compute", wall_time_s() - tp);
        tp = wall_time_s();
        ggml_backend_tensor_get(h, hidden.data(), 0, hidden.size()*sizeof(float));
        profile_log(prof, "t5_download", wall_time_s() - tp);
        ggml_gallocr_free(alloc); ggml_free(ctx);
        profile_log(prof, "t5_total", wall_time_s() - t_t5_total);
        return hidden;
    };

    // ---------- conditioning assembly (host): [T5 hidden (pad-substituted) | secs embed] ----------
    double t0 = wall_time_s();
    std::vector<float> pad_emb = tensor_to_host(CD, "te.padding_embedding");
    const float secs = (float)eff_frames * (sc.patch_size * sc.output_seg) / 44100.0f;
    const float smin = CD.f32("t5g.secs_min"), smax = CD.f32("t5g.secs_max");
    const int sdim = (int)CD.u32("t5g.secs_dim");
    float sclamp = secs < smin ? smin : (secs > smax ? smax : secs);
    float snorm = (sclamp - smin) / (smax - smin);
    std::vector<float> ef; expo_features(snorm, ef, sdim, 0.5f, 10000.0f);
    std::vector<float> sw = tensor_to_host(CD, "te.secs.weight");
    std::vector<float> sb = tensor_to_host(CD, "te.secs.bias");
    std::vector<float> secs_embed(cond_dim);
    for (int d = 0; d < cond_dim; d++) {
        float acc = sb[d];
        for (int i = 0; i < sdim; i++) acc += ef[i] * sw[(size_t)d*sdim + i];
        secs_embed[d] = acc;
    }
    // hidden (pad-substituted) + appended secs token -> cross-attn conditioning [cond_dim*ctx_len].
    auto assemble_cross = [&](std::vector<float> hidden, const std::vector<int32_t>& attn_v) {
        for (int p = 0; p < max_len; p++)
            if (!attn_v[p]) memcpy(&hidden[(size_t)p*cond_dim], pad_emb.data(), cond_dim*sizeof(float));
        std::vector<float> cb((size_t)cond_dim*ctx_len);
        memcpy(cb.data(), hidden.data(), (size_t)cond_dim*max_len*sizeof(float));
        memcpy(&cb[(size_t)cond_dim*max_len], secs_embed.data(), cond_dim*sizeof(float));
        return cb;
    };
    std::vector<float> crossb = assemble_cross(t5_encode(ids, attn), attn);
    std::vector<float>& globb = secs_embed;
    // CFG: unconditioned cross-attn = the negative prompt (if given), else zeros (dit.py null_embed).
    std::vector<float> uncond_crossb;
    if (do_cfg) {
        if (!negative_prompt.empty()) {
            std::vector<int32_t> nids, nattn; tokenize(negative_prompt, nids, nattn);
            uncond_crossb = assemble_cross(t5_encode(nids, nattn), nattn);
        } else {
            uncond_crossb.assign((size_t)cond_dim*ctx_len, 0.0f);
        }
    }
    if (const char* dc_dir = getenv("SA3_DUMP_COND")) {
        FILE* f1 = fopen((std::string(dc_dir)+"/gen_cross.f32").c_str(), "wb");
        fwrite(crossb.data(), sizeof(float), crossb.size(), f1); fclose(f1);
        FILE* f2 = fopen((std::string(dc_dir)+"/gen_global.f32").c_str(), "wb");
        fwrite(globb.data(), sizeof(float), globb.size(), f2); fclose(f2);
    }
    profile_log(prof, "conditioning", wall_time_s() - t0);

    // ---------- schedule (SA3 distribution shift; default = LogSNR rate=0) ----------
    // Linear t = sigma_max*(1 - i/steps), warped by the selected dist-shift, with endpoints
    // re-anchored (t[0]=sigma_max, t[steps]=0) exactly as upstream build_schedule does.
    std::vector<float> sigmas(steps+1);
    for (int i = 0; i <= steps; i++) {
        float t_in = sigma_max * (1.0f - (float)i / steps);
        sigmas[i] = (i == 0) ? sigma_max
                  : (i == steps) ? 0.0f
                  : sa3::dist_shift_warp(dist_shift, t_in, eff_frames, ds_p1, ds_p2, ds_p3, ds_p4);
    }

    // ---------- audio2audio: encode init audio -> latent z_init [latent, T] ----------
    std::vector<float> z_init;
    if (has_init) {
        z_init.resize(N);
        const int Nenc  = T * sc.sub_chunk;
        const int Nenc2 = sc.chunk ? Nenc + 2*sc.shift : 0;
        ggml_init_params encp = { (size_t)512*1024*1024, nullptr, true };
        ggml_context* enctx = ggml_init(encp);
        ggml_tensor* a_in   = ggml_new_tensor_2d(enctx, GGML_TYPE_F32, init_L, sc.out_channels / sc.patch_size);
        ggml_tensor* pos_a  = ggml_new_tensor_1d(enctx, GGML_TYPE_I32, Nenc);
        ggml_tensor* mask_a = sc.chunk ? ggml_new_tensor_2d(enctx, GGML_TYPE_F32, Nenc, Nenc)
                            : same_l_flash_mode == 1 ? ggml_new_tensor_4d(enctx, GGML_TYPE_F16, Nenc, Nenc, 1, 1)
                            : same_l_flash_mode == 2 ? ggml_new_tensor_3d(enctx, GGML_TYPE_F16, 3*sc.sub_chunk, sc.sub_chunk, Nenc/sc.sub_chunk)
                                                     : ggml_new_tensor_3d(enctx, GGML_TYPE_F32, 3*sc.sub_chunk, sc.sub_chunk, Nenc/sc.sub_chunk);
        ggml_set_input(a_in); ggml_set_input(pos_a); ggml_set_input(mask_a);
        ggml_tensor *pos_a2 = nullptr, *mask_a2 = nullptr;
        if (sc.chunk) {
            pos_a2  = ggml_new_tensor_1d(enctx, GGML_TYPE_I32, Nenc2);
            mask_a2 = ggml_new_tensor_2d(enctx, GGML_TYPE_F32, Nenc2, Nenc2);
            ggml_set_input(pos_a2); ggml_set_input(mask_a2);
        }
        ggml_tensor* zt = ggml_cont(enctx, sa3::same_encode(enctx, AE, a_in, sc, T, pos_a, mask_a, pos_a2, mask_a2).z);
        ggml_set_output(zt);
        ggml_cgraph* gf_enc = ggml_new_graph_custom(enctx, 32768, false);
        ggml_build_forward_expand(gf_enc, zt);
        ggml_gallocr_t alloc_enc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(AE.backend));
        ggml_gallocr_alloc_graph(alloc_enc, gf_enc);
        ggml_backend_tensor_set(a_in, init_audio.data(), 0, init_audio.size()*sizeof(float));
        auto set_pos  = [&](ggml_tensor* p, int n){ std::vector<int32_t> b(n); for (int i=0;i<n;i++) b[i]=i; ggml_backend_tensor_set(p, b.data(), 0, n*sizeof(int32_t)); };
        set_pos(pos_a, Nenc); set_swa_bias(mask_a, sc, Nenc);
        if (sc.chunk) set_pos(pos_a2, Nenc2);
        ggml_backend_graph_compute(AE.backend, gf_enc);
        ggml_backend_tensor_get(zt, z_init.data(), 0, N*sizeof(float));
        ggml_gallocr_free(alloc_enc); ggml_free(enctx);
    }

    // ---------- noise (audio2audio: noise = init*(1-sigma_max) + noise*sigma_max) ----------
    sa3::Rng rng(seed);
    std::vector<float> host_x(N); rng.fill_normal(host_x.data(), N);
    std::vector<float> stepnoise((size_t)steps*N); rng.fill_normal(stepnoise.data(), stepnoise.size());
    if (has_init && !inpaint)
        for (int j = 0; j < N; j++) host_x[j] = z_init[j]*(1.0f - sigma_max) + host_x[j]*sigma_max;

    // ---------- inpaint: build local_add_cond = [mask(1) | z_init*mask(256)] ----------
    std::vector<float> localb;
    if (inpaint) {
        auto ceil_div = [](int a, int b){ return (a + b - 1) / b; };
        const int sa = (int)(inpaint_start * 44100.0f);
        const int ea = inpaint_end < 0 ? T*ds : (int)(inpaint_end * 44100.0f);
        const int f0 = std::max(0, std::min(T, ceil_div(sa, ds)));
        const int f1 = std::max(f0, std::min(T, ceil_div(ea, ds)));
        localb.assign((size_t)dc.local_dim * T, 0.0f);
        for (int t = 0; t < T; t++) {
            float m = (t >= f0 && t < f1) ? 0.0f : 1.0f;
            localb[(size_t)t*dc.local_dim + 0] = m;
            for (int c = 0; c < dc.io; c++)
                localb[(size_t)t*dc.local_dim + 1 + c] = z_init[(size_t)t*dc.io + c] * m;
        }
        printf("inpaint: regenerating frames [%d,%d) of %d (%.2f-%.2fs), keeping the rest\n",
               f0, f1, T, f0 * (float)ds / 44100.0f, f1 * (float)ds / 44100.0f);
    }

    if (!keep_models) TE.free();

    // ---------- DiT ping-pong ----------
    const int S = dc.mem_tokens + T;
    const double t_dit_total = wall_time_s();
    double tp_dit = wall_time_s();
    ggml_init_params dip = { (size_t)512*1024*1024, nullptr, true };
    ggml_context* dctx = ggml_init(dip);
    ggml_tensor* x_in  = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.io, T);
    ggml_tensor* tfeat = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, dc.time_dim);
    ggml_tensor* cross = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, cond_dim, ctx_len);
    ggml_tensor* glob  = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, cond_dim);
    ggml_tensor* pos_d = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, S);
    ggml_tensor* ones  = ggml_new_tensor_1d(dctx, GGML_TYPE_F32, 1);
    for (ggml_tensor* t : {x_in, tfeat, cross, glob, pos_d, ones}) ggml_set_input(t);
    ggml_tensor* local = nullptr;
    if (inpaint) { local = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, dc.local_dim, T); ggml_set_input(local); }
    ggml_tensor* vel = ggml_cont(dctx, sa3::dit_forward(dctx, DIT, x_in, tfeat, cross, glob, pos_d, ones, dc, local));
    ggml_set_output(vel);
    ggml_cgraph* gf_dit = ggml_new_graph_custom(dctx, 32768, false);
    ggml_build_forward_expand(gf_dit, vel);
    profile_log(prof, "dit_build", wall_time_s() - tp_dit);
    tp_dit = wall_time_s();
    ggml_gallocr_t alloc_dit = ggml_gallocr_new(ggml_backend_get_default_buffer_type(DIT.backend));
    ggml_gallocr_alloc_graph(alloc_dit, gf_dit);
    profile_log(prof, "dit_alloc", wall_time_s() - tp_dit);
    std::vector<int32_t> posb(S); for (int i = 0; i < S; i++) posb[i] = i;
    const float one = 1.0f;
    std::vector<float> tf, vbuf(N), vbuf_unc, vcfg;   // vbuf = conditioned velocity; vcfg = guided (CFG)
    double dit_upload = 0.0, dit_compute = 0.0, dit_download_update = 0.0;
    for (int i = 0; i < steps; i++) {
        double ts = wall_time_s();
        ggml_backend_tensor_set(cross, crossb.data(), 0, crossb.size()*sizeof(float));
        ggml_backend_tensor_set(glob,  globb.data(),  0, globb.size()*sizeof(float));
        ggml_backend_tensor_set(pos_d, posb.data(),   0, posb.size()*sizeof(int32_t));
        ggml_backend_tensor_set(ones,  &one, 0, sizeof(float));
        if (local) ggml_backend_tensor_set(local, localb.data(), 0, localb.size()*sizeof(float));
        ggml_backend_tensor_set(x_in,  host_x.data(), 0, N*sizeof(float));
        expo_features(sigmas[i], tf, dc.time_dim, dc.time_min_freq, dc.time_max_freq);
        ggml_backend_tensor_set(tfeat, tf.data(), 0, tf.size()*sizeof(float));
        dit_upload += wall_time_s() - ts;
        ts = wall_time_s();
        ggml_backend_graph_compute(DIT.backend, gf_dit);
        dit_compute += wall_time_s() - ts;
        ts = wall_time_s();
        ggml_backend_tensor_get(vel, vbuf.data(), 0, N*sizeof(float));   // conditioned velocity
        const float tcur = sigmas[i], tnext = sigmas[i+1];
        const float* v_use = vbuf.data();
        // classifier-free guidance: run the unconditioned pass and combine (gated to the cfg interval)
        if (do_cfg && tcur >= cfg_interval_min && tcur <= cfg_interval_max) {
            dit_download_update += wall_time_s() - ts;
            ts = wall_time_s();
            // re-upload every input (the graph may consume/reuse input buffers between computes),
            // swapping only the cross-attn cond to the unconditioned branch.
            ggml_backend_tensor_set(x_in,  host_x.data(), 0, N*sizeof(float));
            ggml_backend_tensor_set(tfeat, tf.data(), 0, tf.size()*sizeof(float));
            ggml_backend_tensor_set(glob,  globb.data(), 0, globb.size()*sizeof(float));
            ggml_backend_tensor_set(pos_d, posb.data(), 0, posb.size()*sizeof(int32_t));
            ggml_backend_tensor_set(ones,  &one, 0, sizeof(float));
            if (local) ggml_backend_tensor_set(local, localb.data(), 0, localb.size()*sizeof(float));
            ggml_backend_tensor_set(cross, uncond_crossb.data(), 0, uncond_crossb.size()*sizeof(float));
            ggml_backend_graph_compute(DIT.backend, gf_dit);
            dit_compute += wall_time_s() - ts;
            ts = wall_time_s();
            vbuf_unc.resize(N); vcfg.resize(N);
            ggml_backend_tensor_get(vel, vbuf_unc.data(), 0, N*sizeof(float));
            sa3::cfg_combine(vcfg, vbuf, vbuf_unc, host_x, N, dc.io, T, tcur,
                             cfg_scale, apg_scale, cfg_rescale, cfg_norm_threshold);
            v_use = vcfg.data();
        }
        for (int j = 0; j < N; j++) {
            float denoised = host_x[j] - tcur * v_use[j];
            host_x[j] = (1.0f - tnext) * denoised + tnext * stepnoise[(size_t)i*N + j];
        }
        dit_download_update += wall_time_s() - ts;
        if (params.on_progress)   // sampling spans 0..0.9 of the overall bar; callback overrides the printf
            params.on_progress({"sampling", i+1, steps, 0.9f * (float)(i+1) / (float)steps});
        else
            printf("  step %d/%d  t=%.4f%s\n", i+1, steps, tcur,
                   (do_cfg && tcur >= cfg_interval_min && tcur <= cfg_interval_max) ? "  (cfg)" : "");
    }
    profile_log(prof, "dit_upload", dit_upload);
    profile_log(prof, "dit_compute", dit_compute);
    profile_log(prof, "dit_get_update", dit_download_update);
    profile_log(prof, "dit_total", wall_time_s() - t_dit_total);
    ggml_gallocr_free(alloc_dit); ggml_free(dctx);
    if (!keep_models) { DIT.free(); dit_loras_.clear(); }   // DiT gone -> next gen reloads a clean base

    // ---------- decode -> samples ----------
    const double t_dec_total = wall_time_s();
    double dec_build = 0.0, dec_alloc = 0.0, dec_upload = 0.0, dec_compute = 0.0, dec_download = 0.0;

    struct DecodeGraph {
        int T = 0, Ndec = 0, N2 = 0, n_samp = 0, n_ch = 0;
        ggml_context* ctx = nullptr;
        ggml_tensor* z = nullptr;
        ggml_tensor* pos = nullptr;
        ggml_tensor* mask = nullptr;
        ggml_tensor* pos2 = nullptr;
        ggml_tensor* mask2 = nullptr;
        ggml_tensor* audio = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t alloc = nullptr;
    };
    auto set_pos = [&](ggml_tensor* p, int n){
        std::vector<int32_t> b(n);
        for (int i = 0; i < n; i++) b[i] = i;
        ggml_backend_tensor_set(p, b.data(), 0, n*sizeof(int32_t));
    };
    auto build_decode_graph = [&](int run_T) {
        DecodeGraph dg;
        dg.T = run_T;
        dg.Ndec = run_T * sc.sub_chunk;
        dg.N2 = sc.chunk ? dg.Ndec + 2*sc.shift : 0;
        double ts = wall_time_s();
        ggml_init_params eip = { (size_t)512*1024*1024, nullptr, true };
        dg.ctx = ggml_init(eip);
        dg.z = ggml_new_tensor_2d(dg.ctx, GGML_TYPE_F32, sc.latent, run_T);
        dg.pos = ggml_new_tensor_1d(dg.ctx, GGML_TYPE_I32, dg.Ndec);
        dg.mask = sc.chunk ? ggml_new_tensor_2d(dg.ctx, GGML_TYPE_F32, dg.Ndec, dg.Ndec)
                : same_l_flash_mode == 1 ? ggml_new_tensor_4d(dg.ctx, GGML_TYPE_F16, dg.Ndec, dg.Ndec, 1, 1)
                : same_l_flash_mode == 2 ? ggml_new_tensor_3d(dg.ctx, GGML_TYPE_F16, 3*sc.sub_chunk, sc.sub_chunk, dg.Ndec/sc.sub_chunk)
                                         : ggml_new_tensor_3d(dg.ctx, GGML_TYPE_F32, 3*sc.sub_chunk, sc.sub_chunk, dg.Ndec/sc.sub_chunk);
        ggml_set_input(dg.z); ggml_set_input(dg.pos); ggml_set_input(dg.mask);
        if (sc.chunk) {
            dg.pos2  = ggml_new_tensor_1d(dg.ctx, GGML_TYPE_I32, dg.N2);
            dg.mask2 = ggml_new_tensor_2d(dg.ctx, GGML_TYPE_F32, dg.N2, dg.N2);
            ggml_set_input(dg.pos2); ggml_set_input(dg.mask2);
        }
        dg.audio = ggml_cont(dg.ctx, sa3::same_decode(dg.ctx, AE, dg.z, sc, run_T, dg.pos, dg.mask, dg.pos2, dg.mask2).audio);
        ggml_set_output(dg.audio);
        dg.graph = ggml_new_graph_custom(dg.ctx, 32768, false);
        ggml_build_forward_expand(dg.graph, dg.audio);
        dec_build += wall_time_s() - ts;
        ts = wall_time_s();
        dg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(AE.backend));
        ggml_gallocr_alloc_graph(dg.alloc, dg.graph);
        dec_alloc += wall_time_s() - ts;
        dg.n_samp = (int)dg.audio->ne[0];
        dg.n_ch = (int)dg.audio->ne[1];
        return dg;
    };
    auto free_decode_graph = [&](DecodeGraph& dg) {
        ggml_gallocr_free(dg.alloc);
        ggml_free(dg.ctx);
        dg = DecodeGraph{};
    };
    auto run_decode_graph = [&](DecodeGraph& dg, const float* z_src, std::vector<float>& out) {
        double ts = wall_time_s();
        ggml_backend_tensor_set(dg.z, z_src, 0, (size_t)sc.latent*dg.T*sizeof(float));
        set_pos(dg.pos, dg.Ndec);
        set_swa_bias(dg.mask, sc, dg.Ndec);
        if (sc.chunk) set_pos(dg.pos2, dg.N2);
        dec_upload += wall_time_s() - ts;
        ts = wall_time_s();
        ggml_backend_graph_compute(AE.backend, dg.graph);
        dec_compute += wall_time_s() - ts;
        ts = wall_time_s();
        out.resize((size_t)dg.n_samp*dg.n_ch);
        ggml_backend_tensor_get(dg.audio, out.data(), 0, out.size()*sizeof(float));
        dec_download += wall_time_s() - ts;
    };

    const bool can_chunk_decode = decode_chunk_size > 0 && !sc.chunk && T >= decode_chunk_size;
    if (decode_chunk_size > 0 && sc.chunk)
        fprintf(stderr, "warning: outer --chunked-decode is only enabled for SAME-L; using monolithic SAME-S decode\n");

    const int n_ch = sc.out_channels / sc.patch_size;
    const int n_samp = T * sc.patch_size * sc.output_seg;
    std::vector<float> ab((size_t)n_samp*n_ch, 0.0f);
    if (params.on_progress) params.on_progress({"decoding", 0, 1, 0.9f});   // decode spans 0.9..1.0
    if (!can_chunk_decode) {
        DecodeGraph dg = build_decode_graph(T);
        run_decode_graph(dg, host_x.data(), ab);
        free_decode_graph(dg);
    } else {
        const int hop = decode_chunk_size - decode_overlap;
        std::vector<int> starts;
        for (int s = 0; s <= T - decode_chunk_size; s += hop) starts.push_back(s);
        const int final_start = T - decode_chunk_size;
        if (starts.empty() || starts.back() != final_start) starts.push_back(final_start);
        fprintf(stderr, "[sa3] chunked SAME-L decode: %zu chunks, size=%d overlap=%d\n",
                starts.size(), decode_chunk_size, decode_overlap);
        DecodeGraph dg = build_decode_graph(decode_chunk_size);
        std::vector<float> zchunk((size_t)sc.latent * decode_chunk_size);
        std::vector<float> chunk_audio;
        const int chunk_samples = decode_chunk_size * ds;
        const int half_overlap_samples = (decode_overlap / 2) * ds;
        int cursor = 0;
        for (size_t i = 0; i < starts.size(); i++) {
            const int st = starts[i];
            for (int t = 0; t < decode_chunk_size; t++)
                memcpy(&zchunk[(size_t)t*sc.latent], &host_x[(size_t)(st + t)*sc.latent], sc.latent*sizeof(float));
            run_decode_graph(dg, zchunk.data(), chunk_audio);
            if (params.on_progress)
                params.on_progress({"decoding", (int)(i+1), (int)starts.size(),
                                    0.9f + 0.1f * (float)(i+1) / (float)starts.size()});
            const bool first = i == 0, last = i + 1 == starts.size();
            const int out_start = last ? n_samp - chunk_samples : st * ds;
            const int left = first ? 0 : half_overlap_samples;
            const int right = last ? chunk_samples : chunk_samples - half_overlap_samples;
            const int target_start = out_start + left;
            int target_end = out_start + right;
            const int next_start = !last
                ? ((i + 2 == starts.size() ? n_samp - chunk_samples : starts[i + 1] * ds)
                   + half_overlap_samples)
                : target_end;
            target_end = std::min(target_end, next_start);
            const int clipped_start = std::max(target_start, cursor);
            if (target_end > clipped_start) {
                const int copy_left = left + (clipped_start - target_start);
                const int copy_count = target_end - clipped_start;
                for (int c = 0; c < n_ch; c++)
                    memcpy(&ab[(size_t)c*n_samp + clipped_start],
                           &chunk_audio[(size_t)c*chunk_samples + copy_left],
                           (size_t)copy_count*sizeof(float));
                cursor = target_end;
            }
        }
        free_decode_graph(dg);
    }
    profile_log(prof, "dec_build", dec_build);
    profile_log(prof, "dec_alloc", dec_alloc);
    profile_log(prof, "dec_upload", dec_upload);
    profile_log(prof, "dec_compute", dec_compute);
    profile_log(prof, "dec_download", dec_download);
    profile_log(prof, "dec_total", wall_time_s() - t_dec_total);

    if (!keep_models) { AE.free(); nets_resident_ = false; }   // frugal: free everything -> reload next call

    // truncate the padded canvas back to the requested length (planar -> compact each channel)
    const int out_n_samp = eff_frames * sc.patch_size * sc.output_seg;
    if (out_n_samp < n_samp) {
        std::vector<float> tr((size_t)out_n_samp * n_ch);
        for (int c = 0; c < n_ch; c++)
            memcpy(&tr[(size_t)c*out_n_samp], &ab[(size_t)c*n_samp], (size_t)out_n_samp*sizeof(float));
        ab = std::move(tr);
    }

    if (params.on_progress) params.on_progress({"done", steps, steps, 1.0f});

    GenResult r;
    r.samples = std::move(ab);
    r.n_samp = out_n_samp < n_samp ? out_n_samp : n_samp;
    r.n_ch = n_ch;
    r.sample_rate = 44100;
    return r;
}

} // namespace sa3
