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

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read caption " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
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

        sa3::TrainLoraState lora;
        if (!cfg.resume_adapter.empty()) {
            if (!sa3::load_train_lora_gguf(cfg.resume_adapter, lora, err)) throw std::runtime_error(err);
        } else if (!sa3::init_train_lora_state(dit, targets, cfg.adapter_type, cfg.rank, cfg.alpha, cfg.seed, lora, err)) {
            throw std::runtime_error(err);
        }

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
        sa3::TrainDiffusionSampler sampler(cfg.seed);
        sa3::TrainLoraGradAccum accum;

        for (size_t i = 0; i < train_pairs.size(); ++i) {
            const auto& pair = train_pairs[i];
            sa3::TrainAudio decoded, windowed;
            if (!sa3::decode_mp3_planar_ffmpeg(pair.audio_path, 44100, sc.out_channels / sc.patch_size, decoded, err))
                throw std::runtime_error(err);
            if (!sa3::prepare_train_audio_window(decoded, target_samples, 0, windowed, err))
                throw std::runtime_error(err);
            sa3::TrainLatents latents;
            if (!sa3::encode_train_audio_to_latents(ae, sc, windowed, latents, err))
                throw std::runtime_error(err);
            const std::string caption = read_text_file(pair.caption_path);
            sa3::TrainConditioning conditioning;
            const float seconds = (float)windowed.n_samples / 44100.0f;
            if (!sa3::encode_train_caption_conditioning(tok, te, cond, tc, caption, seconds, conditioning, err))
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
            if (!sampler.sample(latents.z, sample, err)) throw std::runtime_error(err);
            float loss = 0.0f;
            if (!sa3::run_train_dit_accumulate(dit.backend, graph, lora, accum, sample, conditioning, dc, loss, err))
                throw std::runtime_error(err);
            bool updated = false;
            if (accum.count >= cfg.batch_size) {
                if (!sa3::train_apply_accumulated_adamw(lora, accum, loop, opt, err)) throw std::runtime_error(err);
                updated = true;
            }
            std::printf("train %zu/%zu update=%d id=%s loss=%.6f\n",
                        i + 1, train_pairs.size(), loop.step, pair.id.c_str(), loss);
            metrics << "{\"sample\":" << (i + 1) << ",\"update\":" << loop.step
                    << ",\"split\":\"train\",\"id\":\"" << pair.id
                    << "\",\"loss\":" << loss << "}\n";
            metrics.flush();
            if (updated && cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0) {
                const std::string ckpt = (std::filesystem::path(cfg.output_dir) /
                    ("adapter-step-" + std::to_string(loop.step) + ".gguf")).string();
                if (!sa3::write_train_lora_gguf(lora, ckpt, err)) throw std::runtime_error(err);
                std::printf("checkpoint: %s\n", ckpt.c_str());
            }
        }
        if (accum.count > 0) {
            if (!sa3::train_apply_accumulated_adamw(lora, accum, loop, opt, err)) throw std::runtime_error(err);
            if (cfg.checkpoint_every > 0 && (loop.step % cfg.checkpoint_every) == 0) {
                const std::string ckpt = (std::filesystem::path(cfg.output_dir) /
                    ("adapter-step-" + std::to_string(loop.step) + ".gguf")).string();
                if (!sa3::write_train_lora_gguf(lora, ckpt, err)) throw std::runtime_error(err);
                std::printf("checkpoint: %s\n", ckpt.c_str());
            }
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
