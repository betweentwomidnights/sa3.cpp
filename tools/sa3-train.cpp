// sa3-train: native ggML LoRA training for Stable Audio 3 DiT adapters.
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
#include "train_latents.h"
#include "train_loop.h"
#include "train_model_paths.h"
#include "train_prompt.h"
#include "train_same.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static std::string read_text_file(const std::string& path) {
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

static std::string read_optional_text_file(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream f(path);
    if (!f) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::string compose_train_prompt(const sa3::TrainConfig& cfg,
                                        const sa3::TrainAudioCaptionPair& pair) {
    const std::string caption = read_text_file(pair.caption_path);
    const std::string lyrics = read_optional_text_file(pair.lyrics_path);
    if (cfg.prompt_mode == "lyrics") {
        return lyrics.empty() ? caption : ("Lyrics:\n" + lyrics);
    }
    if (cfg.prompt_mode == "caption-lyrics" && !lyrics.empty()) {
        return caption + "\nLyrics:\n" + lyrics;
    }
    return caption;
}

// Stage 13: build the training prompt. With a prompt-config loaded, compose per sample from
// tags/paths/fixed (the caption feeds the `prompt` tag); otherwise fall back to the caption/lyrics
// modes. `prompt_rng` supplies the per-sample randomness (shuffle/subset/method choice).
static std::string build_train_prompt(const sa3::TrainConfig& cfg, const sa3::PromptConfig& pcfg,
                                      const sa3::TrainAudioCaptionPair& pair, std::mt19937_64& prompt_rng) {
    if (!pcfg.loaded) return compose_train_prompt(cfg, pair);
    sa3::PromptMetadata md;
    md.tags = pair.tags;
    md.tags["prompt"] = read_text_file(pair.caption_path);   // caption == the `prompt` tag
    md.relpath = pair.relpath;
    md.text = md.tags["prompt"];
    return sa3::prompt_compose(pcfg, md, prompt_rng);
}

static bool load_pairs_checked(const sa3::TrainConfig& cfg, const std::string& split,
                               sa3::TrainSplitManifest& manifest,
                               std::vector<sa3::TrainAudioCaptionPair>& pairs,
                               std::string& err) {
    if (!sa3::load_train_split_manifest(cfg.dataset_dir, split, manifest, err)) return false;
    if (!sa3::resolve_train_pairs(manifest, pairs, err)) return false;
    if (!sa3::validate_train_split_pairs(manifest, pairs, err)) return false;
    return true;
}

int main(int argc, char** argv) {
    try {
        sa3::load_dotenv();
        sa3::TrainConfig cfg;
        std::string err;
        if (!sa3::train_parse_args(argc, argv, cfg, err)) {
            std::fprintf(stderr, "%s", sa3::train_config_usage(argv[0]).c_str());
            if (err != "help requested") std::fprintf(stderr, "error: %s\n", err.c_str());
            return err == "help requested" ? 0 : 2;
        }
        if (!sa3::validate_train_config(cfg, err)) {
            std::fprintf(stderr, "sa3-train: %s\n", err.c_str());
            return 2;
        }
        sa3::ModelPaths paths;
        if (!sa3::resolve_train_model_paths(cfg, paths, err)) {
            std::fprintf(stderr, "sa3-train: %s\n", err.c_str());
            return 1;
        }
        sa3::PromptConfig prompt_cfg;   // Stage 13: prompt tag-composition (empty => raw caption)
        if (!cfg.prompt_config_path.empty() && !sa3::load_prompt_config(cfg.prompt_config_path, prompt_cfg, err)) {
            std::fprintf(stderr, "sa3-train: %s\n", err.c_str());
            return 1;
        }

        sa3::TrainSplitManifest train_m, test_m, eval_m;
        std::vector<sa3::TrainAudioCaptionPair> train_pairs, test_pairs, eval_pairs;
        if (!load_pairs_checked(cfg, cfg.train_split, train_m, train_pairs, err) ||
            !load_pairs_checked(cfg, cfg.test_split, test_m, test_pairs, err) ||
            !load_pairs_checked(cfg, cfg.evaluation_split, eval_m, eval_pairs, err)) {
            std::fprintf(stderr, "sa3-train: %s\n", err.c_str());
            return 1;
        }
        if (!sa3::validate_no_training_contamination(train_pairs, test_pairs, cfg.test_split, err) ||
            !sa3::validate_no_training_contamination(train_pairs, eval_pairs, cfg.evaluation_split, err)) {
            std::fprintf(stderr, "sa3-train: %s\n", err.c_str());
            return 1;
        }

        std::filesystem::create_directories(cfg.output_dir);
        std::ofstream metrics(std::filesystem::path(cfg.output_dir) / "metrics.jsonl", std::ios::app);
        if (!metrics) throw std::runtime_error("cannot write metrics in " + cfg.output_dir);
        {
            std::ofstream snap(std::filesystem::path(cfg.output_dir) / "config.snapshot.txt");
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
            snap << "alpha=" << cfg.alpha << "\n";
            snap << "learning_rate=" << cfg.learning_rate << "\n";
            snap << "batch_size=" << cfg.batch_size << "\n";
            snap << "frames=" << cfg.frames << "\n";
            snap << "duration_sec=" << cfg.duration_sec << "\n";
            snap << "seed=" << cfg.seed << "\n";
            snap << "output_dir=" << cfg.output_dir << "\n";
            snap << "prompt_mode=" << cfg.prompt_mode << "\n";
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
        {
            std::ofstream cmd(std::filesystem::path(cfg.output_dir) / "command.txt");
            for (int i = 0; i < argc; ++i) {
                if (i) cmd << ' ';
                cmd << argv[i];
            }
            cmd << "\n";
        }

        std::fprintf(stderr, "[train] base DiT: %s\n", paths.dit.c_str());

        ggml_backend_t backend = sa3::make_backend();
        sa3::Tokenizer tok = sa3::Tokenizer::load(paths.tok.c_str());
        sa3::GgufModel te = sa3::load_gguf(paths.t5.c_str(), backend);
        sa3::GgufModel dit = sa3::load_gguf(paths.dit.c_str(), backend);
        sa3::GgufModel ae = sa3::load_gguf(paths.same.c_str(), backend);
        sa3::GgufModel cond = paths.cond.empty() ? sa3::load_gguf(paths.t5.c_str(), backend)
                                                 : sa3::load_gguf(paths.cond.c_str(), backend);
        sa3::T5GemmaConfig tc = sa3::T5GemmaConfig::from(te);
        sa3::DitConfig dc = sa3::DitConfig::from(dit);
        sa3::SameConfig sc = sa3::SameConfig::from(ae);
        std::vector<sa3::TrainLoraTarget> targets = sa3::enumerate_train_lora_targets(dit);
        if (!cfg.inpainting) {
            // The dit.*.local.* weights are only exercised by the inpainting local-cond path. Without
            // it they'd be dead LoRA targets (never in the forward, so no gradient and no buffer from
            // the graph allocator -> upload would fail). Drop them when inpainting is off.
            targets.erase(std::remove_if(targets.begin(), targets.end(),
                [](const sa3::TrainLoraTarget& t) { return t.stem.find(".local.") != std::string::npos; }),
                targets.end());
        }
        if (targets.empty()) throw std::runtime_error("no DiT LoRA targets found");

        sa3::GgufModel svd_bases;
        const bool have_bases = !cfg.svd_bases_path.empty();
        if (have_bases) svd_bases = sa3::load_gguf(cfg.svd_bases_path.c_str(), backend);

        sa3::TrainLoraState lora;
        if (!cfg.resume_adapter.empty()) {
            if (!sa3::load_train_lora_gguf(cfg.resume_adapter, lora, err)) throw std::runtime_error(err);
        } else if (!sa3::init_train_lora_state(dit, targets, cfg.adapter_type, cfg.rank, cfg.alpha, cfg.seed, lora, err,
                                               have_bases ? &svd_bases : nullptr)) {
            throw std::runtime_error(err);
        }
        if (have_bases) svd_bases.free();

        const int target_samples = cfg.duration_sec > 0.0f ? (int)(cfg.duration_sec * 44100.0f + 0.5f)
                                                           : cfg.frames * sc.patch_size * sc.output_seg;

        // Pre-encoded latents (the reference training method, train_latents.h): every file is
        // encoded ONCE full-length here — or loaded from a gary4local pre-encode output — and the
        // per-step path random-crops in latent space. The autoencoder is freed afterwards.
        const bool use_latents = cfg.pre_encode || !cfg.latents_dir.empty();
        const int crop_frames = target_samples / (sc.patch_size * sc.output_seg);
        sa3::TrainLatentCache lat_cache;
        if (use_latents) {
            if (!cfg.latents_dir.empty()) {
                if (!sa3::train_load_latent_dir(cfg.latents_dir, sc.latent, lat_cache, err))
                    throw std::runtime_error(err);
                std::fprintf(stderr, "[pre-encode] loaded %zu latent files from %s\n",
                             lat_cache.size(), cfg.latents_dir.c_str());
            } else {
                for (const auto& pair : train_pairs) {
                    const std::string stem = std::filesystem::path(pair.audio_path).stem().string();
                    if (lat_cache.count(stem)) continue;
                    sa3::TrainAudio decoded;
                    if (!sa3::decode_mp3_planar_ffmpeg(pair.audio_path, 44100, sc.out_channels / sc.patch_size, decoded, err))
                        throw std::runtime_error(err);
                    sa3::TrainLatentEntry e;
                    if (!sa3::train_pre_encode_file(ae, sc, decoded, cfg.target_latent_rms, e, err))
                        throw std::runtime_error(err);
                    std::fprintf(stderr, "[pre-encode] %s: %.1fs -> %d frames, gain %.4f, rms %.4f -> %.4f (%d rounds)\n",
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
        }
        sa3::TrainDitGraph graph;
        sa3::TrainDitCkpt ck;
        // Checkpointed (per-block) backward: peak activation memory is one block's working set,
        // so the step stays VRAM-resident. Functional families only; others use the monolithic graph.
        const bool use_ckpt = cfg.ckpt_backward &&
                              (cfg.adapter_type == "lora" || cfg.adapter_type == "dora-rows");
        int graph_frames = 0, graph_cond_dim = 0, graph_ctx_len = 0;
        sa3::TrainLoopState loop;
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

        auto do_checkpoint = [&]() {
            const std::string ckpt = (std::filesystem::path(cfg.output_dir) /
                ("adapter-step-" + std::to_string(loop.step) + ".gguf")).string();
            if (!sa3::write_train_lora_gguf(lora, ckpt, err)) throw std::runtime_error(err);
            std::printf("checkpoint: %s\n", ckpt.c_str());
        };

        int epoch = 0;
        bool stop = false;
        while (!stop) {
            if (multi_epoch) std::shuffle(order.begin(), order.end(), shuffle_rng);
            for (size_t oi = 0; oi < order.size() && !stop; ++oi) {
                const auto& pair = train_pairs[order[oi]];
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
                    if (!sa3::decode_mp3_planar_ffmpeg(pair.audio_path, 44100, sc.out_channels / sc.patch_size, decoded, err))
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
                const std::string caption = build_train_prompt(cfg, prompt_cfg, pair, prompt_rng);
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
                std::printf("epoch %d step %d id=%s t=%.4f%s%s lr=%.3e loss=%.6f gnorm=%.4f prompt=\"%s%s\"\n",
                            epoch, loop.step, pair.id.c_str(), sample.t, cfg_dropped ? " cfg_drop" : "",
                            inpaint_tag.c_str(), applied_lr, loss, grad_norm, prompt_preview.c_str(), caption.size() > 48 ? "..." : "");
                metrics << "{\"epoch\":" << epoch << ",\"update\":" << loop.step
                        << ",\"split\":\"train\",\"id\":\"" << pair.id
                        << "\",\"t\":" << sample.t << ",\"cfg_drop\":" << (cfg_dropped ? 1 : 0);
                if (have_inpaint)
                    metrics << ",\"mask\":\"" << kMaskNames[(int)inpaint.type] << "\""
                            << ",\"n_gen\":" << inpaint.n_gen << ",\"n_ctx\":" << inpaint.n_ctx;
                metrics << ",\"lr\":" << applied_lr << ",\"loss\":" << loss
                        << ",\"grad_norm\":" << grad_norm << "}\n";
                metrics.flush();
                if (updated && cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0) do_checkpoint();
                if (cfg.max_steps > 0 && loop.step >= cfg.max_steps) stop = true;
            }
            ++epoch;
            if (!multi_epoch) break;
            if (cfg.max_epochs > 0 && epoch >= cfg.max_epochs) stop = true;
        }
        if (accum.count > 0) {
            sa3::TrainAdamWParams step_opt = scheduled_opt();
            if (!sa3::train_apply_accumulated_adamw(lora, accum, loop, step_opt, err)) throw std::runtime_error(err);
            if (cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0) do_checkpoint();
        }

        const std::string final_ckpt = (std::filesystem::path(cfg.output_dir) / "adapter-final.gguf").string();
        if (!sa3::write_train_lora_gguf(lora, final_ckpt, err)) throw std::runtime_error(err);
        std::printf("final checkpoint: %s\n", final_ckpt.c_str());

        sa3::free_train_dit_ckpt(ck);
        sa3::free_train_dit_graph(graph);
        cond.free(); ae.free(); dit.free(); te.free();
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sa3-train: %s\n", e.what());
        return 1;
    }
}
