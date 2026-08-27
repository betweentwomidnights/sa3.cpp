// libsa3 — C ABI implementation over sa3::Pipeline. See libsa3.h for the contract.
#define SA3_BUILD_DLL
#include "libsa3.h"
#include "sa3_pipeline.h"
#include "lora_convert.h"
#include "train_job.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

// The context owns the pipeline via a unique_ptr so sa3_unload() can drop the models (and their
// VRAM) while keeping the context alive; the next sa3_generate() lazily reloads from `paths`.
struct sa3_context {
    std::unique_ptr<sa3::Pipeline> pipe;
    sa3::ModelPaths paths;
    std::string adapters_dir;
    int cpu_threads = 0;
    std::string device;  // remembered so the frugal-reload path keeps the same backend
};

static void set_err(char* err, int n, const std::string& m) {
    if (err && n > 0) { std::strncpy(err, m.c_str(), (size_t)n - 1); err[n - 1] = '\0'; }
}

static bool apply_init_audio(const sa3_init_audio& in, sa3::GenParams& p, std::string& err) {
    if (in.mode == SA3_INIT_AUDIO_NONE) return true;
    if (in.mode != SA3_INIT_AUDIO_A2A && in.mode != SA3_INIT_AUDIO_INPAINT) {
        err = "init_audio.mode must be SA3_INIT_AUDIO_NONE, SA3_INIT_AUDIO_A2A, or SA3_INIT_AUDIO_INPAINT";
        return false;
    }
    if (!in.samples || in.n_samp <= 0 || in.n_ch <= 0 || in.sample_rate <= 0) {
        err = "init_audio requires non-null planar samples, n_samp > 0, n_ch > 0, and sample_rate > 0";
        return false;
    }
    if (!std::isfinite(in.init_noise_level) || !std::isfinite(in.inpaint_start) || !std::isfinite(in.inpaint_end)) {
        err = "init_audio fields must be finite";
        return false;
    }
    const int64_t total = (int64_t)in.n_samp * (int64_t)in.n_ch;
    if (total <= 0 || (uint64_t)total > (uint64_t)(std::numeric_limits<size_t>::max() / sizeof(float))) {
        err = "init_audio sample count is too large";
        return false;
    }

    p.init_audio.assign(in.samples, in.samples + (size_t)total);
    p.init_n_samp = in.n_samp;
    p.init_n_ch = in.n_ch;
    p.init_sample_rate = in.sample_rate;
    p.init_noise_level = in.init_noise_level > 0.0f ? in.init_noise_level : 0.85f;
    if (in.mode == SA3_INIT_AUDIO_INPAINT) {
        p.inpaint_start = in.inpaint_start;
        p.inpaint_end = in.inpaint_end;
    }
    return true;
}

static int sa3_generate_impl(sa3_context* ctx, const sa3_request* req, const sa3_request_ex* req_ex,
                             sa3_audio* out, char* err, int err_len) {
    if (!ctx || !req || !out) { set_err(err, err_len, "null argument"); return 1; }
    out->samples = nullptr; out->n_samp = 0; out->n_ch = 0; out->sample_rate = 0; out->seed = 0;
    try {
        if (!ctx->pipe) {   // reload after sa3_unload()
            ctx->pipe = std::make_unique<sa3::Pipeline>();
            ctx->pipe->load(ctx->paths, ctx->cpu_threads, ctx->device.empty() ? nullptr : ctx->device.c_str());
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

        // distribution shift: NULL/"" -> "LogSNR"; per-type defaults, overridden by dist_shift_params if any set.
        p.dist_shift = (req->dist_shift && *req->dist_shift) ? req->dist_shift : "LogSNR";
        sa3::dist_shift_defaults(p.dist_shift, p.ds_p1, p.ds_p2, p.ds_p3, p.ds_p4);
        {
            const float* dp = req->dist_shift_params;
            if (dp[0] != 0.0f || dp[1] != 0.0f || dp[2] != 0.0f || dp[3] != 0.0f) {
                p.ds_p1 = dp[0]; p.ds_p2 = dp[1]; p.ds_p3 = dp[2]; p.ds_p4 = dp[3];
            }
        }
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

        if (req_ex) {
            std::string ierr;
            if (!apply_init_audio(req_ex->init_audio, p, ierr)) { set_err(err, err_len, ierr); return 5; }
            if (req_ex->encode_chunk_size > 0) {
                p.encode_chunk_size = req_ex->encode_chunk_size;
                p.encode_overlap = req_ex->encode_overlap > 0 ? req_ex->encode_overlap : 32;
            }
            if (req_ex->decode_chunk_size > 0) {
                p.decode_chunk_size = req_ex->decode_chunk_size;
                p.decode_overlap = req_ex->decode_overlap > 0 ? req_ex->decode_overlap : 32;
            }
            if (req_ex->should_cancel) {
                const sa3_cancel_cb cb = req_ex->should_cancel;
                void* user = req_ex->cancel_user;
                p.should_cancel = [cb, user]() { return cb(user) != 0; };
            }
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

static sa3_context* sa3_init_impl(const sa3_config* cfg, int cpu_threads, const char* device,
                                  const char* text_encoding, const char* ae_encoding,
                                  char* err, int err_len) {
    try {
        if (cpu_threads < 0) { set_err(err, err_len, "cpu_threads must be positive"); return nullptr; }
        std::string models_dir = cfg && cfg->models_dir ? cfg->models_dir : "";
        if (models_dir.empty()) { const char* e = std::getenv("SA3_MODELS_DIR"); models_dir = (e && *e) ? e : "models"; }
        const std::string variant  = cfg && cfg->variant  ? cfg->variant  : "medium";
        const std::string encoding = cfg && cfg->encoding ? cfg->encoding : "f16";
        const std::string adir     = cfg && cfg->adapters_dir ? cfg->adapters_dir : models_dir;

        const std::string tenc = text_encoding ? text_encoding : "";
        const std::string aenc = ae_encoding ? ae_encoding : "";

        sa3::ModelPaths mp; std::string rerr;
        if (!sa3::ModelPaths::resolve(models_dir, variant, encoding, tenc, aenc, mp, rerr)) { set_err(err, err_len, rerr); return nullptr; }

        auto ctx = std::make_unique<sa3_context>();
        ctx->paths = mp;
        ctx->adapters_dir = adir;
        ctx->cpu_threads = cpu_threads;
        if (device) ctx->device = device;
        ctx->pipe = std::make_unique<sa3::Pipeline>();
        ctx->pipe->load(mp, ctx->cpu_threads, device);
        return ctx.release();
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return nullptr; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return nullptr; }
}

extern "C" {

SA3_API sa3_context* sa3_init(const sa3_config* cfg, char* err, int err_len) {
    return sa3_init_impl(cfg, 0, nullptr, nullptr, nullptr, err, err_len);
}

SA3_API sa3_context* sa3_init_ex(const sa3_config_ex* cfg, char* err, int err_len) {
    return sa3_init_impl(cfg ? &cfg->config : nullptr, cfg ? cfg->cpu_threads : 0,
                         cfg ? cfg->device : nullptr,
                         cfg ? cfg->text_encoder_encoding : nullptr,
                         cfg ? cfg->autoencoder_encoding : nullptr, err, err_len);
}

SA3_API int sa3_generate(sa3_context* ctx, const sa3_request* req, sa3_audio* out, char* err, int err_len) {
    return sa3_generate_impl(ctx, req, nullptr, out, err, err_len);
}

SA3_API int sa3_generate_ex(sa3_context* ctx, const sa3_request_ex* req, sa3_audio* out, char* err, int err_len) {
    if (!req) { set_err(err, err_len, "null argument"); return 1; }
    return sa3_generate_impl(ctx, &req->request, req, out, err, err_len);
}

SA3_API void sa3_free_audio(sa3_audio* a) {
    if (a && a->samples) { std::free(a->samples); a->samples = nullptr; a->n_samp = 0; }
}

SA3_API void sa3_unload(sa3_context* ctx) { if (ctx) ctx->pipe.reset(); }   // drop models; keep ctx

SA3_API void sa3_free(sa3_context* ctx) { delete ctx; }

SA3_API const char* sa3_version(void) { return "sa3.cpp libsa3 4"; }

SA3_API int sa3_convert_lora(const char* safetensors_path, const char* json_path,
                             const char* out_gguf_path, char* err, int err_len) {
    if (!safetensors_path || !out_gguf_path) { set_err(err, err_len, "null argument"); return 1; }
    try {
        std::string e;
        // NULL/"" json_path means "the config is in the safetensors' own __metadata__", which is
        // where save_lora_safetensors puts it for autoencoder adapters -- they ship as one file.
        if (!sa3::convert_lora_safetensors(safetensors_path, json_path ? json_path : "",
                                           out_gguf_path, e)) {
            set_err(err, err_len, e); return 2;
        }
        return 0;
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return 10; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return 10; }
}

static void copy_field(char* dst, size_t cap, const std::string& src) {
    if (!dst || cap == 0) return;
    const size_t n = src.size() < cap - 1 ? src.size() : cap - 1;
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

SA3_API int sa3_train(const sa3_train_config* cfg, const sa3_train_hooks* hooks,
                      sa3_train_result* out, char* err, int err_len) {
    if (!cfg) { set_err(err, err_len, "null argument"); return 1; }
    if (out) std::memset(out, 0, sizeof(*out));
    try {
        sa3::TrainConfig tc;
        if (const char* models = std::getenv("SA3_MODELS_DIR"); models && *models) tc.models_dir = models;

        // config_json first, so the explicit fields below win over it.
        if (cfg->config_json && *cfg->config_json) {
            std::string jerr;
            if (!sa3::train_apply_json_config(tc, cfg->config_json, jerr)) { set_err(err, err_len, jerr); return 2; }
        }

        auto str = [](const char* v, std::string& dst) { if (v && *v) dst = v; };
        str(cfg->models_dir,     tc.models_dir);
        str(cfg->variant,        tc.model_variant);
        str(cfg->encoding,       tc.encoding);
        str(cfg->text_encoder_encoding, tc.text_encoding);
        str(cfg->autoencoder_encoding,  tc.ae_encoding);
        str(cfg->dataset_dir,    tc.dataset_dir);
        str(cfg->output_dir,     tc.output_dir);
        str(cfg->latents_dir,    tc.latents_dir);
        str(cfg->prompt_config,  tc.prompt_config_path);
        str(cfg->resume_path,    tc.resume_path);
        str(cfg->adapter_type,   tc.adapter_type);
        str(cfg->lora_scope,     tc.lora_scope);

        if (cfg->steps > 0)            tc.max_steps = cfg->steps;
        if (cfg->rank > 0)             tc.rank = cfg->rank;
        if (cfg->alpha > 0.0f)         tc.alpha = cfg->alpha;
        if (cfg->learning_rate > 0.0f) tc.learning_rate = cfg->learning_rate;
        if (cfg->frames > 0)           tc.frames = cfg->frames;
        if (cfg->duration_sec > 0.0f)  tc.duration_sec = cfg->duration_sec;
        if (cfg->batch_size > 0)       tc.batch_size = cfg->batch_size;
        if (cfg->checkpoint_every > 0) tc.checkpoint_every = cfg->checkpoint_every;
        if (cfg->checkpoint_every < 0) tc.checkpoint_every = 0;   // negative = no intermediate writes
        if (cfg->cpu_threads > 0)      tc.cpu_threads = cfg->cpu_threads;
        if (cfg->pre_encode)           tc.pre_encode = true;
        if (cfg->evict_text_encoder)   tc.evict_text_encoder = true;
        if (cfg->seed != 0)            tc.seed = (unsigned long long)cfg->seed;

        if (tc.dataset_dir.empty()) { set_err(err, err_len, "dataset_dir is required"); return 2; }
        str(cfg->device, tc.device);
        if (tc.cpu_threads == 0) tc.cpu_threads = sa3::cpu_threads_from_env();
        sa3::train_finalize_defaults(tc);

        sa3::TrainHooks th;
        if (hooks) {
            void* user = hooks->user;
            if (hooks->on_log) {
                const sa3_train_log_cb cb = hooks->on_log;
                th.log = [cb, user](const std::string& line) { cb(user, line.c_str()); };
            }
            if (hooks->on_step) {
                const sa3_train_step_cb cb = hooks->on_step;
                th.on_step = [cb, user](const sa3::TrainStepReport& r) {
                    sa3_train_step s;
                    std::memset(&s, 0, sizeof(s));
                    s.epoch = r.epoch;
                    s.step = r.step;
                    s.max_steps = r.max_steps;
                    s.id = r.id.c_str();
                    s.prompt = r.prompt.c_str();
                    s.mask = r.mask.c_str();
                    s.t = r.t;
                    s.learning_rate = r.learning_rate;
                    s.loss = r.loss;
                    s.grad_norm = r.grad_norm;
                    s.step_seconds = r.step_seconds;
                    s.cfg_dropped = r.cfg_dropped ? 1 : 0;
                    s.updated = r.updated ? 1 : 0;
                    s.n_gen = r.n_gen;
                    s.n_ctx = r.n_ctx;
                    cb(user, &s);
                };
            }
            if (hooks->should_cancel) {
                const sa3_train_cancel_cb cb = hooks->should_cancel;
                th.should_cancel = [cb, user]() { return cb(user) != 0; };
            }
            if (hooks->load_audio) {
                const sa3_train_audio_cb cb = hooks->load_audio;
                th.load_audio = [cb, user](const sa3::TrainAudioCaptionPair& pair, int sr, int ch,
                                           sa3::TrainAudio& audio, std::string& lerr) {
                    const float* samples = nullptr;
                    int n_samp = 0;
                    if (cb(user, pair.audio_path.c_str(), sr, ch, &samples, &n_samp) == 0) return false;
                    if (!samples || n_samp <= 0) {
                        lerr = "load_audio returned no samples for " + pair.audio_path;
                        return false;
                    }
                    audio.samples.assign(samples, samples + (size_t)n_samp * (size_t)ch);
                    audio.n_samples = n_samp;
                    audio.n_channels = ch;
                    audio.sample_rate = sr;
                    return true;
                };
            }
            if (hooks->command_line) th.command_line = hooks->command_line;
        }

        sa3::TrainResult result;
        std::string terr;
        if (!sa3::run_training(tc, th, result, terr)) { set_err(err, err_len, terr); return 3; }
        if (out) {
            out->steps = result.steps;
            out->cancelled = result.cancelled ? 1 : 0;
            out->mean_step_seconds = result.mean_step_seconds;
            copy_field(out->final_adapter, sizeof(out->final_adapter), result.final_adapter);
            copy_field(out->last_checkpoint, sizeof(out->last_checkpoint), result.last_adapter_checkpoint);
            copy_field(out->preview_command, sizeof(out->preview_command), result.preview_command);
        }
        return 0;
    } catch (const std::exception& e) { set_err(err, err_len, e.what()); return 10; }
      catch (...)                     { set_err(err, err_len, "unknown error"); return 10; }
}

} // extern "C"
