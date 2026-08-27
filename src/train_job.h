// train_job.h - the whole LoRA training run, minus the CLI.
//
// sa3-train's main() used to hold every decision a run makes: resume validation, the pre-encode
// cache, graph rebuilds, the epoch loop, checkpointing. None of it was reachable from libsa3, so an
// embedded host could generate but not train. This is that orchestration with the argv parsing and
// the printing lifted out, so the CLI and the C API run the same code.
//
// Hosts that cannot touch the filesystem or spawn ffmpeg supply audio through TrainHooks::load_audio
// and get progress through TrainHooks::on_step.
#pragma once

#include "env.h"
#include "gguf_model.h"
#include "sa3_pipeline.h"
#include "tokenizer.h"
#include "train_audio.h"
#include "train_checkpoint.h"
#include "train_conditioning.h"
#include "train_config.h"
#include "train_dataset.h"
#include "train_diffusion.h"
#include "train_dit.h"
#include "train_inpaint.h"
#include "train_latents.h"
#include "train_loop.h"
#include "train_model_paths.h"
#include "train_prompt.h"
#include "train_resume.h"
#include "train_same.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sa3 {

// One optimizer update, as reported to the host.
struct TrainStepReport {
    int epoch = 0;
    int step = 0;               // 1-based update index
    int max_steps = 0;          // 0 when the run is not step-bounded
    std::string id;             // dataset item id
    std::string prompt;         // the composed caption this step trained on
    float t = 0.0f;             // sampled (and dist-shifted) timestep
    float learning_rate = 0.0f;
    float loss = 0.0f;
    double grad_norm = 0.0;     // pre-clip global norm
    double step_seconds = 0.0;
    bool cfg_dropped = false;
    bool updated = false;       // false while accumulating toward batch_size
    std::string mask;           // inpaint mask type, "" when not inpainting
    int n_gen = 0;
    int n_ctx = 0;
};

// Everything the host can plug into a run. All members are optional.
struct TrainHooks {
    // Human-readable progress, one line at a time, already formatted and newline-terminated.
    std::function<void(const std::string&)> log;
    // Per-update metrics.
    std::function<void(const TrainStepReport&)> on_step;
    // Return true to stop at the next sample boundary. The run still checkpoints and writes a
    // final adapter, so a cancelled run is resumable rather than lost.
    std::function<bool()> should_cancel;
    // Supply decoded audio instead of reading pair.audio_path. Return false to fall back to the
    // file path. This is the hook that lets a sandboxed host train without ffmpeg or a filesystem.
    std::function<bool(const TrainAudioCaptionPair& pair, int sample_rate, int channels,
                       TrainAudio& out, std::string& err)> load_audio;
    // Recorded verbatim in the run's command.txt; skipped when empty.
    std::string command_line;
};

struct TrainResult {
    int steps = 0;                          // last completed update
    double mean_step_seconds = 0.0;
    std::string final_adapter;              // adapter-final.gguf
    std::string last_adapter_checkpoint;    // adapter-step-N.gguf, "" if none was written
    std::string last_state_checkpoint;      // trainer-state-step-N.gguf
    std::string preview_command;            // ready-to-run sa3-generate line for the CLI to print
    bool cancelled = false;
};

namespace train_job_detail {

inline std::string vformat(const char* fmt, va_list ap) {
    va_list copy;
    va_copy(copy, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    if (n <= 0) return std::string();
    std::string out((size_t)n, '\0');
    std::vsnprintf(&out[0], (size_t)n + 1, fmt, ap);
    return out;
}

} // namespace train_job_detail

inline std::string train_job_format(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::string s = train_job_detail::vformat(fmt, ap);
    va_end(ap);
    return s;
}

inline void train_job_log(const TrainHooks& hooks, const std::string& line) {
    if (hooks.log) hooks.log(line);
}

inline void train_job_logf(const TrainHooks& hooks, const char* fmt, ...) {
    if (!hooks.log) return;
    va_list ap;
    va_start(ap, fmt);
    std::string s = train_job_detail::vformat(fmt, ap);
    va_end(ap);
    hooks.log(s);
}

// Host-supplied audio wins; otherwise decode the file. A host hook that declines (returns false)
// falls through to the file, so supplying audio for only part of a dataset works.
inline bool train_job_load_audio(const TrainHooks& hooks, const TrainAudioCaptionPair& pair,
                                 int sample_rate, int channels, TrainAudio& out, std::string& err) {
    if (hooks.load_audio) {
        std::string hook_err;
        if (hooks.load_audio(pair, sample_rate, channels, out, hook_err)) return true;
        if (!hook_err.empty()) { err = hook_err; return false; }
    }
    return decode_train_audio_file(pair.audio_path, sample_rate, channels, out, err);
}

inline std::string train_job_read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read caption " + path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // gary4local reads captions with .strip() (pre_encode.py extract_tags) before they become the
    // prompt tag; an unstripped trailing newline perturbs every T5 position (bidirectional attn).
    const char* ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Keep the final preview command friendly to both cmd.exe and PowerShell for ordinary paths and
// prompts. Windows paths cannot contain a double quote; prompt quotes are rendered as apostrophes.
inline std::string train_job_preview_quote(std::string text) {
    for (char& ch : text) {
        if (ch == '\r' || ch == '\n' || ch == '\t') ch = ' ';
        else if (ch == '"') ch = '\'';
    }
    while (text.find("  ") != std::string::npos) text.replace(text.find("  "), 2, " ");
    return "\"" + text + "\"";
}

// Stage 13: build the training prompt. With a prompt-config loaded, compose per sample from
// tags/paths/fixed (the caption feeds the `prompt` tag); otherwise use the caption directly.
// `prompt_rng` supplies the per-sample randomness (shuffle/subset/method choice).
inline std::string train_job_build_prompt(const PromptConfig& pcfg, const TrainAudioCaptionPair& pair,
                                          std::mt19937_64& prompt_rng) {
    if (!pcfg.loaded) return train_job_read_text_file(pair.caption_path);
    PromptMetadata md;
    md.tags = pair.tags;
    md.tags["prompt"] = train_job_read_text_file(pair.caption_path);   // caption == the `prompt` tag
    md.relpath = pair.relpath;
    md.text = md.tags["prompt"];
    return prompt_compose(pcfg, md, prompt_rng);
}

inline bool train_job_load_pairs(const TrainConfig& cfg, const std::string& split,
                                 TrainSplitManifest& manifest,
                                 std::vector<TrainAudioCaptionPair>& pairs, std::string& err) {
    if (!load_train_split_manifest(cfg.dataset_dir, split, manifest, err)) return false;
    if (!resolve_train_pairs(manifest, pairs, err)) return false;
    if (!validate_train_split_pairs(manifest, pairs, err)) return false;
    return true;
}

// Does this backend actually support the LoRA backward pass? The Metal out_prod kernels stage
// their tiles through simdgroup matrices with no scalar fallback, so a GPU below Apple7 (A13 and
// older, and the iOS simulator, which reports Apple2) cannot run them. sa3 computes graphs on one
// backend directly rather than through ggml_backend_sched, so nothing reroutes an unsupported op to
// the CPU -- it would dispatch and trap. Ask before training instead of finding out mid-step.
inline bool train_backend_supports_backward(ggml_backend_t backend) {
    struct ggml_init_params ip = { 16*1024, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) return true;   // cannot probe; let the run proceed rather than block on the probe
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 48, 32);
    const bool ok = ggml_backend_supports_op(backend, ggml_out_prod(ctx, a, b));
    ggml_free(ctx);
    return ok;
}

// Run a training job to completion. `cfg` must already have been through train_finalize_defaults().
// Returns false with `err` set on any failure; no exception escapes.
inline bool run_training(const TrainConfig& cfg, const TrainHooks& hooks,
                         TrainResult& out, std::string& err) {
    try {
        if (!sa3::validate_train_config(cfg, err)) {
            return false;
        }
        if (!cfg.resume_path.empty()) {
            std::string adapter_probe, state_probe;
            if (!sa3::train_resolve_resume_pair(cfg.resume_path, adapter_probe, state_probe, err)) {
                return false;
            }
            const std::filesystem::path output_abs =
                std::filesystem::absolute(cfg.output_dir).lexically_normal();
            const std::filesystem::path state_dir_abs =
                std::filesystem::absolute(std::filesystem::path(state_probe).parent_path()).lexically_normal();
            if (output_abs == state_dir_abs) {
                const int requested =
                    sa3::train_checkpoint_step_from_name(std::filesystem::path(state_probe).filename().string());
                const int latest = sa3::train_latest_checkpoint_step(cfg.output_dir);
                if (requested >= 0 && latest > requested) {
                    err = "step " + std::to_string(requested) + " is not the latest checkpoint in " +
                          cfg.output_dir + " (step " + std::to_string(latest) + " exists); " +
                          "resume the latest checkpoint or choose a different output directory";
                    return false;
                }
            }
        } else {
            // A fresh run into a directory that already holds checkpoints is refused by
            // write_train_checkpoint_pair, which keeps step checkpoints immutable -- but that only
            // runs at checkpoint time, so the failure landed after --checkpoint-every steps of
            // training (or, if checkpoint_every exceeds max_steps, not until the very end). Check it
            // here instead, before anything is loaded or encoded.
            const int latest = sa3::train_latest_checkpoint_step(cfg.output_dir);
            if (latest >= 0) {
                const std::string step = std::to_string(latest);
                err = cfg.output_dir + " already contains training checkpoints (latest step " + step +
                      "). Step checkpoints are immutable, so this run would fail on its first write. "
                      "Resume " + (std::filesystem::path(cfg.output_dir) /
                                   ("trainer-state-step-" + step + ".gguf")).string() +
                      ", or choose a different output directory";
                return false;
            }
        }
        sa3::ModelPaths paths;
        if (!sa3::resolve_train_model_paths(cfg, paths, err)) {
            return false;
        }
        if (!sa3::validate_training_base_dit_metadata(cfg, paths.dit, err)) {
            return false;
        }
        sa3::PromptConfig prompt_cfg;   // Stage 13: prompt tag-composition (empty => raw caption)
        if (!cfg.prompt_config_path.empty() && !sa3::load_prompt_config(cfg.prompt_config_path, prompt_cfg, err)) {
            return false;
        }

        sa3::TrainSplitManifest train_m, test_m, eval_m;
        std::vector<sa3::TrainAudioCaptionPair> train_pairs, test_pairs, eval_pairs;
        if (!train_job_load_pairs(cfg, cfg.train_split, train_m, train_pairs, err) ||
            !train_job_load_pairs(cfg, cfg.test_split, test_m, test_pairs, err) ||
            !train_job_load_pairs(cfg, cfg.evaluation_split, eval_m, eval_pairs, err)) {
            return false;
        }
        if (!sa3::validate_no_training_contamination(train_pairs, test_pairs, cfg.test_split, err) ||
            !sa3::validate_no_training_contamination(train_pairs, eval_pairs, cfg.evaluation_split, err)) {
            return false;
        }

        std::filesystem::create_directories(cfg.output_dir);
        std::ofstream metrics(std::filesystem::path(cfg.output_dir) / "metrics.jsonl", std::ios::app);
        if (!metrics) throw std::runtime_error("cannot write metrics in " + cfg.output_dir);
        {
            const char* snapshot_name = cfg.resume_path.empty()
                ? "config.snapshot.txt" : "config.resume.snapshot.txt";
            std::ofstream snap(std::filesystem::path(cfg.output_dir) / snapshot_name);
            snap << "model_variant=" << cfg.model_variant << "\n";
            snap << "encoding=" << cfg.encoding << "\n";
            snap << "models_dir=" << cfg.models_dir << "\n";
            snap << "resolved_tok=" << paths.tok << "\n";
            snap << "resolved_t5=" << paths.t5 << "\n";
            snap << "resolved_cond=" << paths.cond << "\n";
            snap << "resolved_dit=" << paths.dit << "\n";
            snap << "resolved_same=" << paths.same << "\n";
            snap << "dataset_dir=" << cfg.dataset_dir << "\n";
            snap << "train_split=" << cfg.train_split << "\n";
            snap << "test_split=" << cfg.test_split << "\n";
            snap << "evaluation_split=" << cfg.evaluation_split << "\n";
            snap << "adapter_type=" << cfg.adapter_type << "\n";
            snap << "checkpoint_backward=" << (cfg.ckpt_backward ? "true" : "false") << "\n";
            snap << "pre_encode=" << (cfg.pre_encode ? "true" : "false") << "\n";
            snap << "latents_dir=" << (cfg.latents_dir.empty() ? "(none)" : cfg.latents_dir) << "\n";
            snap << "target_latent_rms=" << cfg.target_latent_rms << "\n";
            snap << "rank=" << cfg.rank << "\n";
            snap << "lora_scope=" << cfg.lora_scope << "\n";
            {   // resolved target count is logged to stderr; the snapshot records the request
                auto csv = [](const std::vector<std::string>& v) {
                    std::string o;
                    for (size_t i = 0; i < v.size(); i++) { if (i) o += ","; o += v[i]; }
                    return o;
                };
                snap << "lora_include=" << csv(cfg.lora_include) << "\n";
                snap << "lora_exclude=" << csv(cfg.lora_exclude) << "\n";
            }
            snap << "alpha=" << cfg.alpha << "\n";
            snap << "learning_rate=" << cfg.learning_rate << "\n";
            snap << "weight_decay=" << cfg.weight_decay << "\n";
            snap << "adam_beta1=" << cfg.adam_beta1 << "\n";
            snap << "adam_beta2=" << cfg.adam_beta2 << "\n";
            snap << "adam_eps=" << cfg.adam_eps << "\n";
            snap << "batch_size=" << cfg.batch_size << "\n";
            snap << "cpu_threads=" << cfg.cpu_threads << "\n";
            snap << "frames=" << cfg.frames << "\n";
            snap << "duration_sec=" << cfg.duration_sec << "\n";
            snap << "seed=" << cfg.seed << "\n";
            snap << "checkpoint_every=" << cfg.checkpoint_every << "\n";
            snap << "output_dir=" << cfg.output_dir << "\n";
            snap << "resume=" << (cfg.resume_path.empty() ? "(none)" : cfg.resume_path) << "\n";
            snap << "prompt_config=" << (cfg.prompt_config_path.empty() ? "(none)" : cfg.prompt_config_path) << "\n";
            snap << "max_steps=" << cfg.max_steps << "\n";
            snap << "max_epochs=" << cfg.max_epochs << "\n";
            snap << "random_crop=" << (cfg.random_crop ? "true" : "false") << "\n";
            snap << "grad_clip=" << cfg.grad_clip << "\n";
            snap << "timestep_sampler=" << cfg.timestep_sampler << "\n";
            snap << "dist_shift=" << cfg.dist_shift << "\n";
            snap << "dist_shift_effective_length=" << (cfg.dist_shift_effective_length ? "true" : "false") << "\n";
            snap << "cfg_dropout_prob=" << cfg.cfg_dropout_prob << "\n";
            snap << "inpainting=" << (cfg.inpainting ? "true" : "false") << "\n";
            if (cfg.inpainting) {
                snap << "inpaint_mask_probs=" << cfg.inpaint_mask_probs << "\n";
                snap << "mask_loss_weight=" << cfg.mask_loss_weight << "\n";
                snap << "mask_padding_attention=" << (cfg.mask_padding_attention ? "true" : "false") << "\n";
            }
            snap << "lr_scheduler=" << cfg.lr_scheduler << "\n";
            if (cfg.lr_scheduler != "constant") {
                snap << "lr_inv_gamma=" << cfg.lr_inv_gamma << "\n";
                snap << "lr_power=" << cfg.lr_power << "\n";
                snap << "lr_warmup=" << cfg.lr_warmup << "\n";
                snap << "lr_final=" << cfg.lr_final << "\n";
            }
        }
        if (!hooks.command_line.empty()) {
            std::ofstream cmd(std::filesystem::path(cfg.output_dir) / "command.txt",
                              cfg.resume_path.empty() ? std::ios::out : (std::ios::out | std::ios::app));
            if (!cfg.resume_path.empty()) cmd << "resume: ";
            cmd << hooks.command_line << "\n";
        }

        train_job_logf(hooks, "[train] output: %s\n", cfg.output_dir.c_str());
        if (!cfg.prompt_config_path.empty())
            train_job_logf(hooks, "[train] prompt config: %s\n", cfg.prompt_config_path.c_str());
        train_job_logf(hooks, "[train] base DiT: %s\n", paths.dit.c_str());

        // Owned so that every way out of this function releases it. The models below are RAII
        // (GgufModel::~GgufModel calls free()), but the backend is a raw handle that only the
        // success path freed -- so any throw, and now any cancel, leaked it. That is invisible to
        // the CLI, which exits immediately after, and not at all invisible to an embedded host
        // that calls sa3_train() again.
        struct BackendHandle {
            ggml_backend_t h = nullptr;
            ~BackendHandle() { if (h) ggml_backend_free(h); }
            operator ggml_backend_t() const { return h; }
        } backend_owner{ sa3::make_backend(cfg.cpu_threads,
                                           cfg.device.empty() ? nullptr : cfg.device.c_str()) };
        ggml_backend_t backend = backend_owner;
        if (!train_backend_supports_backward(backend)) {
            const std::string name = ggml_backend_name(backend) ? ggml_backend_name(backend) : "(unknown)";
            err = "backend '" + name + "' cannot run the LoRA backward pass (OUT_PROD): its GPU is "
                  "below Apple7 / lacks simdgroup matrix support. Train with --device cpu, or use a "
                  "device with an A14 or newer GPU. Inference is unaffected";
            return false;
        }
        sa3::Tokenizer tok = sa3::Tokenizer::load(paths.tok.c_str());
        sa3::GgufModel te = sa3::load_gguf(paths.t5.c_str(), backend);
        sa3::GgufModel dit = sa3::load_gguf(paths.dit.c_str(), backend);
        sa3::GgufModel ae = sa3::load_gguf(paths.same.c_str(), backend);
        sa3::GgufModel cond = paths.cond.empty() ? sa3::load_gguf(paths.t5.c_str(), backend)
                                                 : sa3::load_gguf(paths.cond.c_str(), backend);
        sa3::T5GemmaConfig tc = sa3::T5GemmaConfig::from(te);
        sa3::DitConfig dc = sa3::DitConfig::from(dit);
        sa3::SameConfig sc = sa3::SameConfig::from(ae);
        sa3::TrainLoraScope lora_scope;
        lora_scope.name    = cfg.lora_scope;
        lora_scope.include = cfg.lora_include;
        lora_scope.exclude = cfg.lora_exclude;
        std::vector<sa3::TrainLoraTarget> targets = sa3::enumerate_train_lora_targets(dit, lora_scope);
        if (!cfg.inpainting) {
            // The dit.*.local.* weights are only exercised by the inpainting local-cond path. Without
            // it they'd be dead LoRA targets (never in the forward, so no gradient and no buffer from
            // the graph allocator -> upload would fail). Drop them when inpainting is off.
            targets.erase(std::remove_if(targets.begin(), targets.end(),
                [](const sa3::TrainLoraTarget& t) { return t.stem.find(".local.") != std::string::npos; }),
                targets.end());
        }
        if (targets.empty()) throw std::runtime_error("no DiT LoRA targets found");

        // Say out loud what is being adapted. The scope and the filters change the parameter
        // count silently otherwise, and the count is the thing you want to eyeball against the
        // scope you asked for (core=168, full=228 on medium).
        {
            size_t block_targets = 0;
            for (const sa3::TrainLoraTarget& t : targets)
                if (sa3::train_lora_is_block_weight(t.weight_name)) block_targets++;
            std::string scope_line = train_job_format(
                "[train] lora scope: %s -> %zu targets (%zu per-block, %zu elsewhere)"
                "  rank %d alpha %.4g scale %.4g",
                cfg.lora_scope.c_str(), targets.size(), block_targets, targets.size() - block_targets,
                cfg.rank, (double)cfg.alpha, (double)cfg.alpha / (double)cfg.rank);
            if (!cfg.lora_include.empty() || !cfg.lora_exclude.empty()) {
                scope_line += "  [filters:";
                for (const std::string& p : cfg.lora_include) scope_line += " +" + p;
                for (const std::string& p : cfg.lora_exclude) scope_line += " -" + p;
                scope_line += "]";
            }
            train_job_log(hooks, scope_line + "\n");
        }

        sa3::GgufModel svd_bases;
        const bool have_bases = !cfg.svd_bases_path.empty();
        if (have_bases) svd_bases = sa3::load_gguf(cfg.svd_bases_path.c_str(), backend);

        sa3::TrainLoraState lora;
        if (!sa3::init_train_lora_state(dit, targets, cfg.adapter_type, cfg.rank, cfg.alpha, cfg.seed, lora, err,
                                       have_bases ? &svd_bases : nullptr)) {
            throw std::runtime_error(err);
        }
        if (have_bases) svd_bases.free();

        const std::string resume_compatibility =
            sa3::train_resume_compatibility(cfg, paths.dit, targets, train_pairs);
        const bool resuming = !cfg.resume_path.empty();
        std::string resume_adapter_path, resume_state_path;
        sa3::TrainLoopState loop;
        sa3::TrainResumeProgress loaded_progress;
        int resume_start_step = 0;
        if (resuming) {
            if (!sa3::train_resolve_resume_pair(cfg.resume_path, resume_adapter_path, resume_state_path, err))
                throw std::runtime_error(err);
            sa3::TrainLoraState loaded_lora;
            if (!sa3::load_train_lora_gguf(resume_adapter_path, loaded_lora, err) ||
                !sa3::train_restore_adapter_values(lora, loaded_lora, err) ||
                !sa3::load_train_state_gguf(resume_state_path, lora, loop, loaded_progress, err)) {
                throw std::runtime_error(err);
            }
            if (!sa3::train_validate_resume_adapter_fingerprint(
                    resume_adapter_path, loaded_progress.adapter_fingerprint, err))
                throw std::runtime_error(err);
            if (!sa3::train_validate_resume_compatibility(loaded_progress.compatibility,
                                                          resume_compatibility, err))
                throw std::runtime_error(err);
            if (loaded_progress.adapter_file != std::filesystem::path(resume_adapter_path).filename().string())
                throw std::runtime_error("trainer state points to a different adapter checkpoint");
            if (loaded_progress.order.size() != train_pairs.size())
                throw std::runtime_error("resume dataset order length does not match the training split");
            std::vector<bool> seen(train_pairs.size(), false);
            for (uint64_t index : loaded_progress.order) {
                if (index >= train_pairs.size() || seen[(size_t)index])
                    throw std::runtime_error("resume dataset order is not a valid permutation");
                seen[(size_t)index] = true;
            }
            resume_start_step = loop.step;
            if (cfg.max_steps > 0 && loop.step > cfg.max_steps)
                throw std::runtime_error("resume step exceeds --steps (which is the total target step)");
            train_job_logf(hooks, "[resume] step %d from %s\n", loop.step, resume_state_path.c_str());
        }

        const int target_samples = cfg.duration_sec > 0.0f ? (int)(cfg.duration_sec * 44100.0f + 0.5f)
                                                           : cfg.frames * sc.patch_size * sc.output_seg;

        // Pre-encoded latents (the reference training method, train_latents.h): every file is
        // encoded ONCE full-length here — or loaded from a gary4local pre-encode output — and the
        // per-step path random-crops in latent space. The autoencoder is freed afterwards.
        const bool use_latents = cfg.pre_encode || !cfg.latents_dir.empty();
        const int crop_frames = target_samples / (sc.patch_size * sc.output_seg);
        sa3::TrainLatentCache lat_cache;

        // Native pre-encode needs the autoencoder and nothing else, but T5 / the DiT / the
        // conditioner were loaded above for their configs and the LoRA target enumeration, and
        // holding all four through a full-corpus encode costs ~1.8 GB before the first latent on
        // medium-q4. That does not fit a 4 GB device. Drop them for the duration and reload after
        // -- the same one-model-at-a-time discipline Pipeline::load already uses for inference.
        //
        // Safe because nothing that survives points into them: TrainLoraTarget carries names and
        // dims only, TrainLoraParam keeps host std::vector<float> copies, and every config was
        // already extracted by value. load_gguf(path, backend) leaves owns_backend false, so the
        // shared backend outlives the free.
        const bool native_pre_encode = use_latents && cfg.latents_dir.empty();
        if (native_pre_encode) {
            te.free();
            dit.free();
            cond.free();
        }
        if (use_latents) {
            if (!cfg.latents_dir.empty()) {
                if (!sa3::train_load_latent_dir(cfg.latents_dir, sc.latent, lat_cache, err))
                    throw std::runtime_error(err);
                train_job_logf(hooks, "[pre-encode] loaded %zu latent files from %s\n",
                             lat_cache.size(), cfg.latents_dir.c_str());
            } else {
                for (const auto& pair : train_pairs) {
                    // Checked per FILE, because that is the granularity available: one iteration
                    // decodes a whole track and encodes it full-length, which on a phone is
                    // seconds to minutes. Without this the host's cancel was simply not observed
                    // until the first training step, so cancelling during pre-encode looked like
                    // a hang -- and on a large corpus it is the longest stretch of the run.
                    if (hooks.should_cancel && hooks.should_cancel()) {
                        out.cancelled = true;
                        train_job_logf(hooks, "[pre-encode] cancelled after %zu file(s)\n",
                                       lat_cache.size());
                        return true;
                    }
                    const std::string stem = std::filesystem::path(pair.audio_path).stem().string();
                    if (lat_cache.count(stem)) continue;
                    sa3::TrainAudio decoded;
                    if (!train_job_load_audio(hooks, pair, 44100, sc.out_channels / sc.patch_size, decoded, err))
                        throw std::runtime_error(err);
                    sa3::TrainLatentEntry e;
                    if (!sa3::train_pre_encode_file(ae, sc, decoded, cfg.target_latent_rms, e, err))
                        throw std::runtime_error(err);
                    train_job_logf(hooks, "[pre-encode] %s: %.1fs -> %d frames, gain %.4f, rms %.4f -> %.4f (%d rounds)\n",
                                 stem.c_str(), e.seconds_total, e.n_valid, e.gain, e.rms_pre, e.rms_achieved, e.norm_rounds);
                    lat_cache[stem] = std::move(e);
                }
            }
            for (const auto& pair : train_pairs) {
                const std::string stem = std::filesystem::path(pair.audio_path).stem().string();
                auto it = lat_cache.find(stem);
                if (it == lat_cache.end())
                    throw std::runtime_error("no pre-encoded latents for " + stem);
                if (it->second.n_valid < crop_frames)
                    throw std::runtime_error(stem + " has only " + std::to_string(it->second.n_valid) +
                                             " valid latent frames (< --frames " + std::to_string(crop_frames) +
                                             "); shorten --frames or drop the file");
            }
            ae.free();  // the autoencoder is no longer needed; frees ~1.7 GB of VRAM
            if (native_pre_encode) {   // bring back what the training loop needs
                te   = sa3::load_gguf(paths.t5.c_str(),  backend);
                dit  = sa3::load_gguf(paths.dit.c_str(), backend);
                cond = paths.cond.empty() ? sa3::load_gguf(paths.t5.c_str(), backend)
                                          : sa3::load_gguf(paths.cond.c_str(), backend);
            }
        }
        sa3::TrainDitGraph graph;
        sa3::TrainDitCkpt ck;
        // Checkpointed (per-block) backward: peak activation memory is one block's working set,
        // so the step stays VRAM-resident. Every adapter family supports it -- the families
        // dit_lin cannot apply functionally materialize their W_eff inside each segment's own
        // graph instead of once for the whole step (see install_overrides in train_ckpt.h).
        const bool use_ckpt = cfg.ckpt_backward;
        int graph_frames = 0, graph_cond_dim = 0, graph_ctx_len = 0;
        sa3::TrainAdamWParams opt;
        opt.learning_rate = cfg.learning_rate;
        opt.beta1 = cfg.adam_beta1;
        opt.beta2 = cfg.adam_beta2;
        opt.eps = cfg.adam_eps;
        opt.weight_decay = cfg.weight_decay;
        opt.grad_clip = cfg.grad_clip;
        sa3::TrainDiffusionSampler sampler(cfg.seed, cfg.timestep_sampler);
        sa3::TrainLoraGradAccum accum;

        // Stage 11: per-step LR schedule. train_apply_accumulated_adamw increments loop.step at
        // entry, so at the call site loop.step is the 0-indexed update index (PyTorch last_epoch).
        auto scheduled_opt = [&]() {
            sa3::TrainAdamWParams o = opt;
            if (cfg.lr_scheduler != "constant")
                o.learning_rate = sa3::inverse_lr(opt.learning_rate, loop.step, cfg.lr_inv_gamma,
                                                  cfg.lr_power, cfg.lr_warmup, cfg.lr_final);
            return o;
        };

        // Stage 10: training-time dist-shift params (per-type defaults; sa3-medium trains with
        // "Full" base_shift 0.5 / max_shift 1.15 / min 256 / max 4096).
        float ds_p1 = 0.0f, ds_p2 = 0.0f, ds_p3 = 0.0f, ds_p4 = 0.0f;
        sa3::dist_shift_defaults(cfg.dist_shift, ds_p1, ds_p2, ds_p3, ds_p4);
        const int downsampling_ratio = sc.patch_size * sc.output_seg;  // audio samples per latent frame

        // Stage 1: multi-epoch loop. max_steps>0 or max_epochs>0 enables per-epoch shuffle and
        // stops at max_steps optimizer updates; both 0 keeps the legacy single pass.
        const bool multi_epoch = cfg.max_steps > 0 || cfg.max_epochs > 0;
        std::mt19937_64 shuffle_rng(cfg.seed ^ 0x9e3779b97f4a7c15ULL);
        std::mt19937_64 crop_rng(cfg.seed ^ 0xd1b54a32d192ed03ULL);
        std::mt19937_64 cfg_rng(cfg.seed ^ 0xa0761d6478bd642fULL);   // Stage 9: cfg-dropout stream
        std::uniform_real_distribution<float> cfg_drop_dist(0.0f, 1.0f);
        std::mt19937_64 prompt_rng(cfg.seed ^ 0x2545f4914f6cdd1dULL); // Stage 13: prompt-composition stream
        std::mt19937_64 inpaint_rng(cfg.seed ^ 0x14057b7ef767814fULL); // Stage 12: inpaint-mask stream
        std::vector<double> inpaint_probs;
        if (cfg.inpainting && !sa3::train_parse_probs(cfg.inpaint_mask_probs, inpaint_probs, err))
            throw std::runtime_error(err);
        std::vector<size_t> order(train_pairs.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        int epoch = 0;
        size_t first_oi = 0;
        bool restored_epoch_order = false;
        if (resuming) {
            if (loaded_progress.epoch > (uint64_t)std::numeric_limits<int>::max())
                throw std::runtime_error("resume epoch exceeds the supported range");
            epoch = (int)loaded_progress.epoch;
            first_oi = (size_t)loaded_progress.next_sample;
            for (size_t i = 0; i < order.size(); ++i) order[i] = (size_t)loaded_progress.order[i];
            if (!sa3::train_restore_random_state(loaded_progress.shuffle_rng, shuffle_rng, "shuffle", err) ||
                !sa3::train_restore_random_state(loaded_progress.crop_rng, crop_rng, "crop", err) ||
                !sa3::train_restore_random_state(loaded_progress.cfg_rng, cfg_rng, "CFG dropout", err) ||
                !sa3::train_restore_random_state(loaded_progress.prompt_rng, prompt_rng, "prompt", err) ||
                !sa3::train_restore_random_state(loaded_progress.inpaint_rng, inpaint_rng, "inpainting", err) ||
                !sampler.restore_state(loaded_progress.diffusion_rng, err)) throw std::runtime_error(err);
            restored_epoch_order = true;
        }

        auto capture_progress = [&](int checkpoint_epoch, size_t next_sample) {
            sa3::TrainResumeProgress progress;
            progress.epoch = (uint64_t)checkpoint_epoch;
            progress.next_sample = (uint64_t)next_sample;
            progress.compatibility = resume_compatibility;
            for (size_t index : order) progress.order.push_back((uint64_t)index);
            progress.shuffle_rng = sa3::train_serialize_random_state(shuffle_rng);
            progress.crop_rng = sa3::train_serialize_random_state(crop_rng);
            progress.cfg_rng = sa3::train_serialize_random_state(cfg_rng);
            progress.prompt_rng = sa3::train_serialize_random_state(prompt_rng);
            progress.inpaint_rng = sa3::train_serialize_random_state(inpaint_rng);
            progress.diffusion_rng = sampler.serialize_state();
            return progress;
        };

        int last_checkpoint_step = -1;
        auto do_checkpoint = [&](int checkpoint_epoch, size_t next_sample) {
            const std::string suffix = "-step-" + std::to_string(loop.step) + ".gguf";
            const std::string adapter = (std::filesystem::path(cfg.output_dir) / ("adapter" + suffix)).string();
            const std::string state = (std::filesystem::path(cfg.output_dir) / ("trainer-state" + suffix)).string();
            if (!sa3::write_train_checkpoint_pair(lora, loop, capture_progress(checkpoint_epoch, next_sample),
                                                  adapter, state, err)) throw std::runtime_error(err);
            last_checkpoint_step = loop.step;
            out.last_adapter_checkpoint = adapter;
            out.last_state_checkpoint = state;
            train_job_logf(hooks, "checkpoint: %s\ntrainer state: %s\n", adapter.c_str(), state.c_str());
        };

        int cursor_epoch = epoch;
        size_t cursor_next_sample = first_oi;
        bool stop = cfg.max_steps > 0 && loop.step >= cfg.max_steps;
        if (cfg.max_epochs > 0 && epoch >= cfg.max_epochs) stop = true;
        // Per-step wall time. Started here so model load and pre-encode are excluded -- those are
        // one-off and identical across configurations, and folding them in would mask the thing
        // worth measuring (what a target scope or adapter family costs per step).
        double t_prev_step = sa3::wall_time_s();
        const double t_train_begin = t_prev_step;
        double step_seconds_total = 0.0;
        int step_seconds_count = 0;
        double step_recent_total = 0.0;   // rolling window for the console line
        int step_recent_count = 0;
        while (!stop) {
            size_t oi_begin = 0;
            if (restored_epoch_order) {
                oi_begin = first_oi;
                restored_epoch_order = false;
            } else if (multi_epoch) {
                std::shuffle(order.begin(), order.end(), shuffle_rng);
            }
            for (size_t oi = oi_begin; oi < order.size() && !stop; ++oi) {
                if (hooks.should_cancel && hooks.should_cancel()) {
                    out.cancelled = true;
                    stop = true;
                    break;
                }
                const auto& pair = train_pairs[order[oi]];
                cursor_epoch = epoch;
                cursor_next_sample = oi + 1;
                // Per-phase profiling (SA3_TRAIN_PROFILE=1): decode/AE-encode/T5-cond/DiT/AdamW ms.
                const bool prof = getenv("SA3_TRAIN_PROFILE") != nullptr;
                auto tnow = [] { return std::chrono::steady_clock::now(); };
                auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                auto p0 = tnow();
                auto p1 = p0;   // decode/encode boundary (legacy mode); crop is instant in latents mode
                sa3::TrainLatents latents;
                double seconds_total = 0.0;   // full-file duration; fractional in latents mode
                if (use_latents) {
                    // PreEncodedDataset crop semantics: start = randint(0, last_ix - crop)
                    // (inclusive), drawn only when the last valid frame index exceeds the crop.
                    const sa3::TrainLatentEntry& e =
                        lat_cache.at(std::filesystem::path(pair.audio_path).stem().string());
                    int crop_start = 0;
                    const int last_ix = e.n_valid - 1;
                    if (cfg.random_crop && last_ix > crop_frames) {
                        std::uniform_int_distribution<int> sd(0, last_ix - crop_frames);
                        crop_start = sd(crop_rng);
                    }
                    sa3::train_crop_latents(e, crop_start, crop_frames, latents);
                    // round(actual_samples/sr, 3) like the reference sidecars — it feeds the
                    // seconds conditioning and the dist-shift effective length un-ceiled.
                    seconds_total = e.seconds_total;
                } else {
                    sa3::TrainAudio decoded, windowed;
                    if (!train_job_load_audio(hooks, pair, 44100, sc.out_channels / sc.patch_size, decoded, err))
                        throw std::runtime_error(err);
                    // Stage 2: random-crop window start (fixed length -> the training graph is unchanged).
                    int crop_start = 0;
                    if (cfg.random_crop && decoded.n_samples > target_samples) {
                        std::uniform_int_distribution<int> sd(0, decoded.n_samples - target_samples);
                        crop_start = sd(crop_rng);
                    }
                    if (!sa3::prepare_train_audio_window(decoded, target_samples, crop_start, windowed, err))
                        throw std::runtime_error(err);
                    p1 = tnow();
                    if (!sa3::encode_train_audio_to_latents(ae, sc, windowed, latents, err))
                        throw std::runtime_error(err);
                    // Legacy behavior: full-file duration, ceil'd to whole seconds.
                    seconds_total = std::ceil((double)decoded.n_samples / 44100.0);
                }
                auto p2 = tnow();
                const std::string caption = train_job_build_prompt(prompt_cfg, pair, prompt_rng);
                sa3::TrainConditioning conditioning;
                if (!sa3::encode_train_caption_conditioning(tok, te, cond, tc, caption, (float)seconds_total, conditioning, err))
                    throw std::runtime_error(err);
                // Debug hook mirroring sa3_pipeline's SA3_DUMP_COND: dump the step's conditioning so
                // the training-side encoder can be diffed against the inference pipeline's.
                if (const char* dc_dir = getenv("SA3_DUMP_COND")) {
                    FILE* f1 = fopen((std::string(dc_dir) + "/train_cross.f32").c_str(), "wb");
                    fwrite(conditioning.cross.data(), sizeof(float), conditioning.cross.size(), f1);
                    fclose(f1);
                    FILE* f2 = fopen((std::string(dc_dir) + "/train_global.f32").c_str(), "wb");
                    fwrite(conditioning.global.data(), sizeof(float), conditioning.global.size(), f2);
                    fclose(f2);
                    std::fprintf(stderr, "[dump] conditioning for prompt \"%s\" secs %.3f -> %s\n",
                                 caption.c_str(), seconds_total, dc_dir);
                }
                // Stage 9: cfg-dropout. With prob cfg_dropout_prob, replace the cross-attention
                // conditioning (prompt tokens + appended seconds token) with zeros, matching
                // dit.py null_embed = zeros_like(cross_attn_cond); the global seconds embedding is
                // kept. Draw per sample so batch_size>1 gets a per-element decision like bernoulli.
                bool cfg_dropped = false;
                if (cfg.cfg_dropout_prob > 0.0f && cfg_drop_dist(cfg_rng) < cfg.cfg_dropout_prob) {
                    std::fill(conditioning.cross.begin(), conditioning.cross.end(), 0.0f);
                    cfg_dropped = true;
                }
                auto p3 = tnow();
                const bool graphs_built = use_ckpt ? ck.pctx != nullptr : graph.ctx != nullptr;
                if (!graphs_built || graph_frames != latents.frames ||
                    graph_cond_dim != conditioning.cond_dim || graph_ctx_len != conditioning.ctx_len) {
                    if (use_ckpt) {
                        sa3::free_train_dit_ckpt(ck);
                        if (!sa3::build_train_dit_ckpt(dit, dc, lora, latents.frames,
                                                       conditioning.cond_dim, conditioning.ctx_len, ck, err,
                                                       cfg.inpainting))
                            throw std::runtime_error(err);
                    } else {
                        sa3::free_train_dit_graph(graph);
                        if (!sa3::build_train_dit_forward_graph(dit, dc, lora, latents.frames,
                                                                conditioning.cond_dim, conditioning.ctx_len, graph, err,
                                                                cfg.inpainting))
                            throw std::runtime_error(err);
                        // build_train_dit_forward_graph now allocates the graph internally (gallocr).
                    }
                    graph_frames = latents.frames;
                    graph_cond_dim = conditioning.cond_dim;
                    graph_ctx_len = conditioning.ctx_len;
                }
                sa3::TrainDiffusionSample sample;
                // Stage 10: warp the drawn t like DiffusionCondTrainingWrapper (dist_shift.shift
                // with use_effective_length_for_schedule: effective length from seconds_total).
                float t = sampler.draw_t();
                if (cfg.dist_shift != "None") {
                    // Reference (underfit loop.py): ceil(int(seconds_total * sr) / ratio) — the
                    // sample count is truncated to int before the divide. Identical to the old
                    // formula when seconds_total is a whole number (legacy mode).
                    const int eff_len = cfg.dist_shift_effective_length
                        ? (int)std::ceil((double)(int64_t)(seconds_total * 44100.0) / (double)downsampling_ratio)
                        : latents.frames;
                    t = sa3::dist_shift_warp(cfg.dist_shift, t, eff_len, ds_p1, ds_p2, ds_p3, ds_p4);
                }
                if (!sampler.sample_at(latents.z, t, sample, err)) throw std::runtime_error(err);
                // Stage 12: inpainting objective. Generate a per-sample mask (type by inpaint_probs),
                // build [mask | latent*mask] local-add cond + the inpaint-aware loss weight. All crop
                // frames are real (no padding), so real_len == latents.frames.
                sa3::TrainInpaint inpaint;
                bool have_inpaint = false;
                if (cfg.inpainting) {
                    sa3::InpaintMaskType mtype;
                    std::vector<float> mask = sa3::generate_inpaint_mask(latents.frames, latents.frames,
                                                                         inpaint_probs, inpaint_rng, mtype);
                    inpaint = sa3::build_train_inpaint(latents.z, mask, dc.io, latents.frames, dc.local_dim,
                                                       cfg.mask_loss_weight, cfg.mask_padding_attention);
                    inpaint.type = mtype;
                    have_inpaint = true;
                }
                auto p4 = tnow();   // p3->p4 = graph build (first step) + sampling + inpaint gen (host)
                float loss = 0.0f;
                if (use_ckpt) {
                    if (!sa3::run_train_dit_accumulate_ckpt(dit.backend, ck, lora, accum, sample, conditioning, dc, loss, err,
                                                            have_inpaint ? &inpaint : nullptr))
                        throw std::runtime_error(err);
                } else if (!sa3::run_train_dit_accumulate(dit.backend, graph, lora, accum, sample, conditioning, dc, loss, err,
                                                          have_inpaint ? &inpaint : nullptr)) {
                    throw std::runtime_error(err);
                }
                auto p5 = tnow();   // p4->p5 = DiT fwd+bwd+grad-read (synced by loss/grad tensor_get)
                // Deterministic cross-framework replay bundle. The native step is the source of
                // truth for every tensor that enters the DiT; a PyTorch harness can consume these
                // files without reproducing any crop/RNG/conditioning decisions. This is separate
                // from SA3_DUMP_GRADS so forward-only and gradient comparisons can share a bundle.
                if (const char* ds = getenv("SA3_DUMP_STEP")) {
                    std::filesystem::create_directories(ds);
                    auto wr = [&](const std::string& n, const std::vector<float>& v) {
                        if (v.empty()) return;
                        FILE* f = fopen((std::string(ds) + "/" + n + ".f32").c_str(), "wb");
                        if (f) { fwrite(v.data(), sizeof(float), v.size(), f); fclose(f); }
                    };
                    wr("latent", latents.z);
                    wr("noise", sample.noise);
                    wr("x_t", sample.x_t);
                    wr("target", sample.velocity_target);
                    wr("cross", conditioning.cross);
                    wr("global", conditioning.global);
                    if (have_inpaint) {
                        wr("local", inpaint.local);
                        wr("loss_weight", inpaint.loss_weight);
                    }
                    std::vector<float> velocity(sample.x_t.size());
                    ggml_tensor* velocity_t = use_ckpt ? ck.velocity : graph.velocity;
                    if (velocity_t)
                        ggml_backend_tensor_get(velocity_t, velocity.data(), 0,
                                                velocity.size() * sizeof(float));
                    wr("velocity_cpp", velocity);
                    if (use_ckpt) {
                        auto wr_tensor = [&](const std::string& n, ggml_tensor* tensor) {
                            if (!tensor) return;
                            std::vector<float> data((size_t)ggml_nelements(tensor));
                            ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
                            wr(n, data);
                        };
                        wr_tensor("context_cpp", ck.context_p);
                        wr_tensor("gcond_cpp", ck.gcond_p);
                        for (size_t i = 0; i < ck.xb.size(); ++i)
                            wr_tensor("block_" + std::to_string(i) + "_cpp", ck.xb[i]);
                    }
                    std::ofstream meta(std::filesystem::path(ds) / "meta.json");
                    meta << "{\n"
                         << "  \"t\": " << sample.t << ",\n"
                         << "  \"loss_cpp\": " << loss << ",\n"
                         << "  \"frames\": " << latents.frames << ",\n"
                         << "  \"io\": " << dc.io << ",\n"
                         << "  \"cond_dim\": " << conditioning.cond_dim << ",\n"
                         << "  \"ctx_len\": " << conditioning.ctx_len << ",\n"
                         << "  \"local_dim\": " << (have_inpaint ? dc.local_dim : 0) << ",\n"
                         << "  \"cfg_drop\": " << (cfg_dropped ? "true" : "false") << ",\n"
                         << "  \"target_count\": " << lora.params.size() << "\n"
                         << "}\n";
                    std::ofstream manifest(std::filesystem::path(ds) / "targets.txt");
                    for (const auto& p : lora.params) manifest << p.target.stem << "\n";
                    std::fprintf(stderr, "[dump] replay step %d -> %s\n", loop.step + 1, ds);
                }
                // Debug hook: dump the raw accumulated gradients (pre-Adam) for cross-backend /
                // cross-framework comparison, then exit after the first step.
                if (const char* gd = getenv("SA3_DUMP_GRADS")) {
                    std::filesystem::create_directories(gd);
                    auto wr = [&](const std::string& n, const std::vector<float>& v) {
                        if (v.empty()) return;
                        FILE* f = fopen((std::string(gd) + "/" + n + ".f32").c_str(), "wb");
                        if (f) { fwrite(v.data(), sizeof(float), v.size(), f); fclose(f); }
                    };
                    for (size_t i = 0; i < lora.params.size(); ++i) {
                        const std::string& stem = lora.params[i].target.stem;
                        if (i < accum.A.size()) wr(stem + ".gA", accum.A[i]);
                        if (i < accum.B.size()) wr(stem + ".gB", accum.B[i]);
                        if (i < accum.mag.size()) wr(stem + ".gmag", accum.mag[i]);
                    }
                    std::fprintf(stderr, "[dump] gradients for step %d -> %s\n", loop.step + 1, gd);
                }
                bool updated = false;
                float applied_lr = opt.learning_rate;
                double grad_norm = 0.0;   // pre-clip global grad norm (reference train/grad_norm)
                if (accum.count >= cfg.batch_size) {
                    sa3::TrainAdamWParams step_opt = scheduled_opt();
                    applied_lr = step_opt.learning_rate;
                    if (!sa3::train_apply_accumulated_adamw(lora, accum, loop, step_opt, err, &grad_norm)) throw std::runtime_error(err);
                    updated = true;
                }
                auto p6 = tnow();
                if (prof)
                    std::fprintf(stderr, "[prof] step %d: decode=%.0f ae_encode=%.0f t5_cond=%.0f prep=%.0f dit=%.0f adamw=%.0f total=%.0f ms\n",
                                 loop.step, ms(p0,p1), ms(p1,p2), ms(p2,p3), ms(p3,p4), ms(p4,p5), ms(p5,p6), ms(p0,p6));
                // Truncated composed prompt, so the caption/path mix is visible when eyeballing runs.
                std::string prompt_preview = caption.substr(0, 48);
                for (char& ch : prompt_preview) if (ch == '\n' || ch == '\r') ch = ' ';
                static const char* kMaskNames[] = {"segments", "full", "causal", "spans"};
                std::string inpaint_tag;
                if (have_inpaint)
                    inpaint_tag = std::string(" mask=") + kMaskNames[(int)inpaint.type] +
                                  "(" + std::to_string(inpaint.n_gen) + "gen/" + std::to_string(inpaint.n_ctx) + "ctx)";
                const double t_now_step = sa3::wall_time_s();
                const double step_s = t_now_step - t_prev_step;
                t_prev_step = t_now_step;
                step_seconds_total += step_s;  step_seconds_count++;
                step_recent_total  += step_s;  step_recent_count++;
                const double step_recent_avg = step_recent_total / (double)step_recent_count;
                if (step_recent_count >= 50) { step_recent_total = 0.0; step_recent_count = 0; }
                train_job_logf(hooks, "epoch %d step %d id=%s t=%.4f%s%s lr=%.3e loss=%.6f gnorm=%.4f %.2fs/step prompt=\"%s%s\"\n",
                            epoch, loop.step, pair.id.c_str(), sample.t, cfg_dropped ? " cfg_drop" : "",
                            inpaint_tag.c_str(), applied_lr, loss, grad_norm, step_recent_avg,
                            prompt_preview.c_str(), caption.size() > 48 ? "..." : "");
                if (hooks.on_step) {
                    TrainStepReport r;
                    r.epoch = epoch;
                    r.step = loop.step;
                    r.max_steps = cfg.max_steps;
                    r.id = pair.id;
                    r.prompt = caption;
                    r.t = sample.t;
                    r.learning_rate = applied_lr;
                    r.loss = loss;
                    r.grad_norm = grad_norm;
                    r.step_seconds = step_s;
                    r.cfg_dropped = cfg_dropped;
                    r.updated = updated;
                    r.mask = have_inpaint ? kMaskNames[(int)inpaint.type] : "";
                    r.n_gen = have_inpaint ? inpaint.n_gen : 0;
                    r.n_ctx = have_inpaint ? inpaint.n_ctx : 0;
                    hooks.on_step(r);
                }
                metrics << "{\"epoch\":" << epoch << ",\"update\":" << loop.step
                        << ",\"split\":\"train\",\"id\":\"" << pair.id
                        << "\",\"t\":" << sample.t << ",\"cfg_drop\":" << (cfg_dropped ? 1 : 0);
                if (have_inpaint)
                    metrics << ",\"mask\":\"" << kMaskNames[(int)inpaint.type] << "\""
                            << ",\"n_gen\":" << inpaint.n_gen << ",\"n_ctx\":" << inpaint.n_ctx;
                metrics << ",\"lr\":" << applied_lr << ",\"loss\":" << loss
                        << ",\"grad_norm\":" << grad_norm << ",\"step_s\":" << step_s << "}\n";
                metrics.flush();
                if (updated && cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0)
                    do_checkpoint(epoch, oi + 1);
                if (cfg.max_steps > 0 && loop.step >= cfg.max_steps) stop = true;
            }
            ++epoch;
            if (!multi_epoch) break;
            if (cfg.max_epochs > 0 && epoch >= cfg.max_epochs) stop = true;
        }
        if (accum.count > 0) {
            sa3::TrainAdamWParams step_opt = scheduled_opt();
            if (!sa3::train_apply_accumulated_adamw(lora, accum, loop, step_opt, err)) throw std::runtime_error(err);
            if (cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0)
                do_checkpoint(cursor_epoch, cursor_next_sample);
        }

        if (loop.step <= 0) throw std::runtime_error("training completed without an optimizer update");
        out.steps = loop.step;
        if (step_seconds_count > 0) {
            const double mean_step = step_seconds_total / (double)step_seconds_count;
            out.mean_step_seconds = mean_step;
            const double wall = sa3::wall_time_s() - t_train_begin;
            train_job_logf(hooks, "[train] %d steps in %.1fs, mean %.3fs/step (%.2f steps/s)\n",
                        step_seconds_count, wall, mean_step, mean_step > 0.0 ? 1.0 / mean_step : 0.0);
        }
        if (last_checkpoint_step != loop.step) {
            const std::string current_adapter = (std::filesystem::path(cfg.output_dir) /
                ("adapter-step-" + std::to_string(loop.step) + ".gguf")).string();
            const bool same_resume_pair = resuming && loop.step == resume_start_step &&
                std::filesystem::absolute(current_adapter).lexically_normal() ==
                std::filesystem::absolute(resume_adapter_path).lexically_normal();
            if (!same_resume_pair) do_checkpoint(cursor_epoch, cursor_next_sample);
        }
        const std::string final_ckpt = (std::filesystem::path(cfg.output_dir) / "adapter-final.gguf").string();
        if (!sa3::write_train_final_adapter(lora, final_ckpt, err)) throw std::runtime_error(err);
        out.final_adapter = final_ckpt;
        train_job_logf(hooks, "final checkpoint: %s\n", final_ckpt.c_str());

        std::string preview_prompt = cfg.eval_caption;
        if (preview_prompt.empty() && !eval_pairs.empty()) {
            try { preview_prompt = train_job_read_text_file(eval_pairs.front().caption_path); }
            catch (...) { /* A preview hint must never turn a successful run into a failure. */ }
        }
        if (preview_prompt.empty()) preview_prompt = "describe the music you want to hear";
        if (preview_prompt.size() > 240) preview_prompt.resize(240);

        const std::string final_abs = std::filesystem::absolute(final_ckpt).lexically_normal().string();
        const std::string models_abs = std::filesystem::absolute(cfg.models_dir).lexically_normal().string();
        const std::string preview_wav = std::filesystem::absolute(
            std::filesystem::path(cfg.output_dir) / "preview.wav").lexically_normal().string();
        std::ostringstream preview;
        preview << "sa3-generate --model " << cfg.model_variant
                << " --models-dir " << train_job_preview_quote(models_abs)
                << " --lora " << train_job_preview_quote(final_abs)
                << " --prompt " << train_job_preview_quote(preview_prompt);
        if (cfg.cpu_threads > 0) preview << " --threads " << cfg.cpu_threads;
        preview << " --out " << train_job_preview_quote(preview_wav);

        sa3::free_train_dit_ckpt(ck);
        sa3::free_train_dit_graph(graph);
        cond.free(); ae.free(); dit.free(); te.free();
        // No ggml_backend_free here: backend_owner owns the handle and frees it on the way out.
        // Freeing it twice took down every successful run at teardown -- after the final adapter
        // was already on disk, so it read as "crashed before saving" (CUDA: "invalid resource
        // handle" in ~ggml_backend_cuda_context, exit 127).
        out.preview_command = preview.str();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

} // namespace sa3
