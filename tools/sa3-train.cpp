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
#include "train_loop.h"
#include "train_model_paths.h"
#include "train_same.h"

#include <algorithm>
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
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
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
            snap << "dataset_dir=" << cfg.dataset_dir << "\n";
            snap << "train_split=" << cfg.train_split << "\n";
            snap << "test_split=" << cfg.test_split << "\n";
            snap << "evaluation_split=" << cfg.evaluation_split << "\n";
            snap << "adapter_type=" << cfg.adapter_type << "\n";
            snap << "rank=" << cfg.rank << "\n";
            snap << "alpha=" << cfg.alpha << "\n";
            snap << "learning_rate=" << cfg.learning_rate << "\n";
            snap << "batch_size=" << cfg.batch_size << "\n";
            snap << "frames=" << cfg.frames << "\n";
            snap << "duration_sec=" << cfg.duration_sec << "\n";
            snap << "seed=" << cfg.seed << "\n";
            snap << "output_dir=" << cfg.output_dir << "\n";
            snap << "prompt_mode=" << cfg.prompt_mode << "\n";
            snap << "max_steps=" << cfg.max_steps << "\n";
            snap << "max_epochs=" << cfg.max_epochs << "\n";
            snap << "random_crop=" << (cfg.random_crop ? "true" : "false") << "\n";
            snap << "grad_clip=" << cfg.grad_clip << "\n";
            snap << "timestep_sampler=" << cfg.timestep_sampler << "\n";
            snap << "dist_shift=" << cfg.dist_shift << "\n";
            snap << "dist_shift_effective_length=" << (cfg.dist_shift_effective_length ? "true" : "false") << "\n";
        }
        {
            std::ofstream cmd(std::filesystem::path(cfg.output_dir) / "command.txt");
            for (int i = 0; i < argc; ++i) {
                if (i) cmd << ' ';
                cmd << argv[i];
            }
            cmd << "\n";
        }

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
        sa3::TrainDitGraph graph;
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
                sa3::TrainLatents latents;
                if (!sa3::encode_train_audio_to_latents(ae, sc, windowed, latents, err))
                    throw std::runtime_error(err);
                const std::string caption = compose_train_prompt(cfg, pair);
                sa3::TrainConditioning conditioning;
                // Reference parity: seconds_total is the FULL source-file duration (ceil'd at
                // pre-encode) and is not updated for the crop window. It feeds both the
                // seconds_total conditioning and the dist-shift effective length.
                const int seconds_total = (int)std::ceil((double)decoded.n_samples / 44100.0);
                if (!sa3::encode_train_caption_conditioning(tok, te, cond, tc, caption, (float)seconds_total, conditioning, err))
                    throw std::runtime_error(err);
                if (!graph.ctx || graph_frames != latents.frames ||
                    graph_cond_dim != conditioning.cond_dim || graph_ctx_len != conditioning.ctx_len) {
                    sa3::free_train_dit_graph(graph);
                    if (!sa3::build_train_dit_forward_graph(dit, dc, lora, latents.frames,
                                                            conditioning.cond_dim, conditioning.ctx_len, graph, err))
                        throw std::runtime_error(err);
                    graph.ctx_buf = ggml_backend_alloc_ctx_tensors(graph.ctx, dit.backend);
                    if (!graph.ctx_buf) throw std::runtime_error("failed to allocate DiT training graph tensors");
                    graph_frames = latents.frames;
                    graph_cond_dim = conditioning.cond_dim;
                    graph_ctx_len = conditioning.ctx_len;
                }
                sa3::TrainDiffusionSample sample;
                // Stage 10: warp the drawn t like DiffusionCondTrainingWrapper (dist_shift.shift
                // with use_effective_length_for_schedule: effective length from seconds_total).
                float t = sampler.draw_t();
                if (cfg.dist_shift != "None") {
                    const int eff_len = cfg.dist_shift_effective_length
                        ? (int)std::ceil((double)seconds_total * 44100.0 / (double)downsampling_ratio)
                        : latents.frames;
                    t = sa3::dist_shift_warp(cfg.dist_shift, t, eff_len, ds_p1, ds_p2, ds_p3, ds_p4);
                }
                if (!sampler.sample_at(latents.z, t, sample, err)) throw std::runtime_error(err);
                float loss = 0.0f;
                if (!sa3::run_train_dit_accumulate(dit.backend, graph, lora, accum, sample, conditioning, dc, loss, err))
                    throw std::runtime_error(err);
                bool updated = false;
                if (accum.count >= cfg.batch_size) {
                    if (!sa3::train_apply_accumulated_adamw(lora, accum, loop, opt, err)) throw std::runtime_error(err);
                    updated = true;
                }
                std::printf("epoch %d step %d id=%s t=%.4f loss=%.6f\n", epoch, loop.step, pair.id.c_str(), sample.t, loss);
                metrics << "{\"epoch\":" << epoch << ",\"update\":" << loop.step
                        << ",\"split\":\"train\",\"id\":\"" << pair.id
                        << "\",\"t\":" << sample.t << ",\"loss\":" << loss << "}\n";
                metrics.flush();
                if (updated && cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0) do_checkpoint();
                if (cfg.max_steps > 0 && loop.step >= cfg.max_steps) stop = true;
            }
            ++epoch;
            if (!multi_epoch) break;
            if (cfg.max_epochs > 0 && epoch >= cfg.max_epochs) stop = true;
        }
        if (accum.count > 0) {
            if (!sa3::train_apply_accumulated_adamw(lora, accum, loop, opt, err)) throw std::runtime_error(err);
            if (cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0) do_checkpoint();
        }

        const std::string final_ckpt = (std::filesystem::path(cfg.output_dir) / "adapter-final.gguf").string();
        if (!sa3::write_train_lora_gguf(lora, final_ckpt, err)) throw std::runtime_error(err);
        std::printf("final checkpoint: %s\n", final_ckpt.c_str());

        sa3::free_train_dit_graph(graph);
        cond.free(); ae.free(); dit.free(); te.free();
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sa3-train: %s\n", e.what());
        return 1;
    }
}
