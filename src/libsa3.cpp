// libsa3 — C ABI implementation over sa3::Pipeline. See libsa3.h for the contract.
#define SA3_BUILD_DLL
#include "libsa3.h"
#include "sa3_pipeline.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

// The context owns the pipeline via a unique_ptr so sa3_unload() can drop the models (and their
// VRAM) while keeping the context alive; the next sa3_generate() lazily reloads from `paths`.
struct sa3_context {
    std::unique_ptr<sa3::Pipeline> pipe;
    sa3::ModelPaths paths;
    std::string adapters_dir;
};

static void set_err(char* err, int n, const std::string& m) {
    if (err && n > 0) { std::strncpy(err, m.c_str(), (size_t)n - 1); err[n - 1] = '\0'; }
}

extern "C" {

SA3_API sa3_context* sa3_init(const sa3_config* cfg, char* err, int err_len) {
    try {
        std::string models_dir = cfg && cfg->models_dir ? cfg->models_dir : "";
        if (models_dir.empty()) { const char* e = std::getenv("SA3_MODELS_DIR"); models_dir = (e && *e) ? e : "models"; }
        const std::string variant  = cfg && cfg->variant  ? cfg->variant  : "medium";
        const std::string encoding = cfg && cfg->encoding ? cfg->encoding : "f16";
        const std::string adir     = cfg && cfg->adapters_dir ? cfg->adapters_dir : models_dir;

        sa3::ModelPaths mp; std::string rerr;
        if (!sa3::ModelPaths::resolve(models_dir, variant, encoding, mp, rerr)) { set_err(err, err_len, rerr); return nullptr; }

        auto ctx = std::make_unique<sa3_context>();
        ctx->paths = mp;
        ctx->adapters_dir = adir;
        ctx->pipe = std::make_unique<sa3::Pipeline>();
        ctx->pipe->load(mp);
        return ctx.release();
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return nullptr; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return nullptr; }
}

SA3_API int sa3_generate(sa3_context* ctx, const sa3_request* req, sa3_audio* out, char* err, int err_len) {
    if (!ctx || !req || !out) { set_err(err, err_len, "null argument"); return 1; }
    try {
        if (!ctx->pipe) {   // reload after sa3_unload()
            ctx->pipe = std::make_unique<sa3::Pipeline>();
            ctx->pipe->load(ctx->paths);
        }
        sa3::GenParams p;
        p.prompt = req->prompt ? req->prompt : "";
        if (req->negative_prompt) p.negative_prompt = req->negative_prompt;
        p.frames = req->frames > 0 ? req->frames : 128;
        p.steps  = req->steps  > 0 ? req->steps  : 8;
        p.seed   = sa3::pick_seed((long long)req->seed);
        p.cfg_scale = req->cfg_scale != 0.0f ? req->cfg_scale : 1.0f;
        p.duration_padding_sec = req->duration_padding_sec >= 0.0f ? req->duration_padding_sec : 6.0f;
        p.keep_models = req->keep_models != 0;
        if (req->loudness.set) {              // per-request loudness (incl. raw: peak_normalize=0, limiter=0)
            sa3::LoudnessParams lp;
            lp.peak_normalize_enabled = req->loudness.peak_normalize != 0;
            lp.peak_normalize_db      = req->loudness.peak_normalize_db;
            lp.limiter_enabled        = req->loudness.limiter != 0;
            lp.limiter_ceiling_db     = req->loudness.limiter_ceiling_db;
            lp.limiter_knee           = req->loudness.limiter_knee > 0.0f ? req->loudness.limiter_knee : lp.limiter_knee;
            lp.latent_rescale         = req->loudness.latent_rescale > 0.0f ? req->loudness.latent_rescale : 1.0f;
            lp.latent_shift           = req->loudness.latent_shift;
            sa3::normalize_loudness_params(lp);
            std::string lerr;
            if (!sa3::validate_loudness_params(lp, lerr)) { set_err(err, err_len, "loudness: " + lerr); return 4; }
            p.loudness = lp;
        } else {
            p.loudness = sa3::loudness_defaults_from_env();   // gary4local defaults + SA3_* env overrides
        }

        for (int i = 0; i < req->n_loras; i++) {
            const std::string name = (req->lora_names && req->lora_names[i]) ? req->lora_names[i] : "";
            if (name.empty()) continue;
            std::string path = std::filesystem::exists(name)
                             ? name : sa3::resolve_one(ctx->adapters_dir, "lora-" + name + "-", ".gguf");
            if (path.empty()) { set_err(err, err_len, "unknown lora '" + name + "'"); return 2; }
            const float s = req->lora_strengths ? req->lora_strengths[i] : 1.0f;
            p.loras.push_back({path, s});
        }
        if (req->on_progress) {
            const sa3_progress_cb cb = req->on_progress; void* user = req->user;
            p.on_progress = [cb, user](const sa3::Progress& pr) { cb(user, pr.stage, pr.step, pr.total, pr.fraction); };
        }

        sa3::GenResult r = ctx->pipe->generate(p);
        const size_t n = (size_t)r.n_samp * r.n_ch;
        out->samples = (float*)std::malloc(n * sizeof(float));
        if (!out->samples) { set_err(err, err_len, "out of memory"); return 3; }
        std::memcpy(out->samples, r.samples.data(), n * sizeof(float));
        out->n_samp = r.n_samp; out->n_ch = r.n_ch; out->sample_rate = r.sample_rate;
        out->seed = p.seed;
        return 0;
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return 10; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return 10; }
}

SA3_API void sa3_free_audio(sa3_audio* a) {
    if (a && a->samples) { std::free(a->samples); a->samples = nullptr; a->n_samp = 0; }
}

SA3_API void sa3_unload(sa3_context* ctx) { if (ctx) ctx->pipe.reset(); }   // drop models; keep ctx

SA3_API void sa3_free(sa3_context* ctx) { delete ctx; }

SA3_API const char* sa3_version(void) { return "sa3.cpp libsa3 1"; }

} // extern "C"
