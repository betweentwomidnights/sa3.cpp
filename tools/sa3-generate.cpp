// sa3-generate: standalone text2music. prompt string -> WAV, no PyTorch in the loop.
// tokenize -> T5Gemma encode -> conditioning assembly (learned padding + seconds)
// -> ping-pong sampler over the DiT -> SAME-L decode -> WAV.
#include "gguf_model.h"
#include "tokenizer.h"
#include "t5gemma.h"
#include "dit.h"
#include "same_ae.h"
#include "lora.h"
#include "sa3_pipeline.h"
#include "env.h"
#include "rng.h"
#include "wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// (expo_features / wall_time_s / profile_* / tensor_to_host / set_swa_bias moved to sa3_pipeline.h)

int main(int argc, char** argv) {
    sa3::load_dotenv();   // ./.env -> SA3_* defaults (real env wins); must precede every getenv below
    const bool prof = sa3::profile_enabled();
    const double t_total0 = sa3::wall_time_s();
    const char* tok_p = nullptr; const char* t5_p = nullptr; const char* dit_p = nullptr; const char* same_p = nullptr;
    const char* cond_p = nullptr;            // per-variant conditioner sidecar gguf (optional; falls back to --t5 if bundled)
    const char* model_variant = nullptr;     // --model: resolve the 5 base ggufs by naming convention
    const char* encoding = "f16";            // --encoding f16|f32 (which DiT/SAME precision --model picks)
    const char* env_md = getenv("SA3_MODELS_DIR");
    const char* models_dir = (env_md && *env_md) ? env_md : "models";  // --models-dir / SA3_MODELS_DIR
    const char* env_ad = getenv("SA3_ADAPTERS_DIR");
    const char* adapters_dir = (env_ad && *env_ad) ? env_ad : nullptr; // --adapters-dir / SA3_ADAPTERS_DIR (else = models_dir)
    std::string prompt = "Upbeat funk groove with slap bass, bright horns, tight drums";
    const char* wav_p = "song.wav";
    const char* init_p = nullptr;            // audio2audio / inpaint: source WAV (encoded to z_init)
    float init_noise_level = 0.85f;          // sigma_max for audio2audio (1.0 == text2music)
    float inpaint_start = -1.0f, inpaint_end = -1.0f;   // inpaint: regenerate this [start,end] sec region
    std::vector<std::pair<std::string,float>> lora_specs;   // (gguf, strength) applied in flag order
    bool keep_models = false;        // --keep-models: don't free TE/DIT early (keep all resident)
    int encode_chunk_size = 0, encode_overlap = 32;     // outer SAME-L encode tiling; 0 = monolithic
    int decode_chunk_size = 0, decode_overlap = 32;     // outer SAME-L decode tiling; 0 = monolithic
    int frames = 128, steps = 8; long long seed = 0;   // seed < 0 (e.g. -1) => random (resolved below)
    std::string dist_shift = "LogSNR";                  // schedule warp: LogSNR|Flux|Full|None
    float ds_p1 = 2000.0f, ds_p2 = -6.2f, ds_p3 = 0.0f, ds_p4 = 2.0f;   // LogSNR defaults (per-type, see --dist-shift)
    float duration_padding_sec = 6.0f;                  // text2music schedule headroom (0 = let the model end the piece)
    std::string negative_prompt;                        // CFG negative prompt (only used when cfg_scale != 1)
    float cfg_scale = 1.0f, cfg_rescale = 0.0f, apg_scale = 1.0f, cfg_norm_threshold = 0.0f;
    float cfg_interval_min = 0.0f, cfg_interval_max = 1.0f;
    sa3::LoudnessParams loudness = sa3::loudness_defaults_from_env();
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")  && i+1 < argc) model_variant = argv[++i];
        else if (!strcmp(argv[i], "--encoding") && i+1 < argc) encoding = argv[++i];
        else if (!strcmp(argv[i], "--models-dir") && i+1 < argc) models_dir = argv[++i];
        else if (!strcmp(argv[i], "--adapters-dir") && i+1 < argc) adapters_dir = argv[++i];
        else if (!strcmp(argv[i], "--tok")    && i+1 < argc) tok_p = argv[++i];
        else if (!strcmp(argv[i], "--t5")     && i+1 < argc) t5_p = argv[++i];
        else if (!strcmp(argv[i], "--dit")    && i+1 < argc) dit_p = argv[++i];
        else if (!strcmp(argv[i], "--same")   && i+1 < argc) same_p = argv[++i];
        else if (!strcmp(argv[i], "--cond")   && i+1 < argc) cond_p = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--frames") && i+1 < argc) frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")  && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")   && i+1 < argc) seed = strtoll(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--out")    && i+1 < argc) wav_p = argv[++i];
        else if (!strcmp(argv[i], "--init")   && i+1 < argc) init_p = argv[++i];
        else if (!strcmp(argv[i], "--init-noise-level") && i+1 < argc) init_noise_level = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--inpaint-start") && i+1 < argc) inpaint_start = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--inpaint-end")   && i+1 < argc) inpaint_end   = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lora")   && i+1 < argc) lora_specs.push_back({argv[++i], 1.0f});
        else if (!strcmp(argv[i], "--lora-strength") && i+1 < argc) {   // sets the most recent --lora
            if (lora_specs.empty()) { fprintf(stderr, "--lora-strength must follow a --lora\n"); return 1; }
            lora_specs.back().second = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--dist-shift") && i+1 < argc) {   // selects type + its default params
            dist_shift = argv[++i];
            sa3::dist_shift_defaults(dist_shift, ds_p1, ds_p2, ds_p3, ds_p4);
        }
        else if (!strcmp(argv[i], "--dist-shift-params") && i+1 < argc) {   // override the 4 params (follows --dist-shift)
            if (sscanf(argv[++i], "%f,%f,%f,%f", &ds_p1, &ds_p2, &ds_p3, &ds_p4) != 4) {
                fprintf(stderr, "--dist-shift-params expects p1,p2,p3,p4 (meaning depends on --dist-shift)\n");
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--duration-padding") && i+1 < argc) duration_padding_sec = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--negative-prompt") && i+1 < argc) negative_prompt = argv[++i];
        else if (!strcmp(argv[i], "--cfg-scale") && i+1 < argc) cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-rescale") && i+1 < argc) cfg_rescale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--apg-scale") && i+1 < argc) apg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-norm-threshold") && i+1 < argc) cfg_norm_threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-interval") && i+1 < argc) {
            if (sscanf(argv[++i], "%f,%f", &cfg_interval_min, &cfg_interval_max) != 2) {
                fprintf(stderr, "--cfg-interval expects min,max\n"); return 1;
            }
        }
        else if (!strcmp(argv[i], "--chunked-decode")) decode_chunk_size = 128;
        else if (!strcmp(argv[i], "--chunked-encode")) encode_chunk_size = 128;
        else if (!strcmp(argv[i], "--encode-chunk-size") && i+1 < argc) encode_chunk_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--encode-overlap") && i+1 < argc) encode_overlap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-chunk-size") && i+1 < argc) decode_chunk_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-overlap") && i+1 < argc) decode_overlap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keep-models")) keep_models = true;   // disable progressive VRAM frees
        else if (!strcmp(argv[i], "--latent-rescale") && i+1 < argc) loudness.latent_rescale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--latent-shift") && i+1 < argc) loudness.latent_shift = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--latent-target-std") && i+1 < argc) {
            loudness.latent_target_std_enabled = true;
            loudness.latent_target_std = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--no-latent-target-std")) loudness.latent_target_std_enabled = false;
        else if (!strcmp(argv[i], "--latent-adapt-min") && i+1 < argc) loudness.latent_adapt_min = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--latent-adapt-max") && i+1 < argc) loudness.latent_adapt_max = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--peak-normalize-db") && i+1 < argc) {
            loudness.peak_normalize_enabled = true;
            loudness.peak_normalize_db = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--no-peak-normalize")) loudness.peak_normalize_enabled = false;
        else if (!strcmp(argv[i], "--limiter-ceiling-db") && i+1 < argc) {
            loudness.limiter_enabled = true;
            loudness.limiter_ceiling_db = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--no-limiter")) loudness.limiter_enabled = false;
        else if (!strcmp(argv[i], "--limiter-knee") && i+1 < argc) loudness.limiter_knee = (float)atof(argv[++i]);
    }
    // --model <variant>: fill the five base ggufs from <models-dir> by the naming convention.
    // Explicit --tok/--t5/--cond/--dit/--same still win (override per-slot).
    std::vector<std::string> resolved; resolved.reserve(5);   // keep resolved paths alive for the char* slots
    if (model_variant) {
        const std::string mv = model_variant, md = models_dir;
        const std::string ENC = (strcmp(encoding, "f32") == 0) ? "F32" : "F16";
        auto fill = [&](const char*& slot, const std::string& prefix, const std::string& suffix) {
            if (slot) return;                                  // explicit flag overrides the convention
            std::string p = sa3::resolve_one(md, prefix, suffix);
            if (p.empty()) {
                fprintf(stderr, "[sa3] --model %s: no %s*%s in %s/ (run: python tools/download_models.py --variant %s)\n",
                        mv.c_str(), prefix.c_str(), suffix.c_str(), md.c_str(), mv.c_str());
                exit(1);
            }
            resolved.push_back(std::move(p)); slot = resolved.back().c_str();
        };
        fill(tok_p,  "t5gemma-b-b-ul2-v1.0-vocab",             ".gguf");
        fill(t5_p,   "t5gemma-b-b-ul2-encoder-",               ".gguf");        // shared encoder (F32)
        fill(cond_p, "stable-audio-3-" + mv + "-conditioner-", ".gguf");        // F32
        fill(dit_p,  "stable-audio-3-" + mv + "-dit-",  "-" + ENC + ".gguf");
        fill(same_p, "stable-audio-3-" + mv + "-same-", "-" + ENC + ".gguf");
    }
    // --lora <name|path>: a bare name resolves to <adapters-dir>/lora-<name>-*.gguf (adapters can live
    // anywhere via SA3_ADAPTERS_DIR/--adapters-dir; defaults to the models dir). A real path is used as-is.
    const char* adir = adapters_dir ? adapters_dir : models_dir;
    for (auto& spec : lora_specs) {
        if (std::filesystem::exists(spec.first)) continue;
        std::string p = sa3::resolve_one(adir, "lora-" + spec.first + "-", ".gguf");
        if (p.empty()) {
            fprintf(stderr, "[sa3] --lora %s: not a file and no lora-%s-*.gguf in %s/\n",
                    spec.first.c_str(), spec.first.c_str(), adir);
            exit(1);
        }
        spec.first = std::move(p);
    }

    const bool inpaint = (inpaint_start >= 0.0f || inpaint_end >= 0.0f);   // inpaint mode (needs --init source)
    if (!tok_p || !t5_p || !dit_p || !same_p) {
        fprintf(stderr, "usage: sa3-generate (--model medium|small-music|small-sfx [--encoding f16|f32] [--models-dir DIR]\n"
                        "                     | --tok <f> --t5 <f> --cond <f> --dit <f> --same <f>)\n"
                        "                     --prompt \"...\" [--lora NAME|PATH [--lora-strength S]]... [--frames N] [--steps N] [--seed S]\n"
                        "                     [--dist-shift LogSNR|Flux|Full|None [--dist-shift-params p1,p2,p3,p4]] [--duration-padding SEC]\n"
                        "                     [--cfg-scale S [--negative-prompt \"...\"] [--cfg-rescale R] [--cfg-interval min,max] [--apg-scale A] [--cfg-norm-threshold T]] [--out song.wav]\n");
        return 1;
    }
    if (encode_chunk_size < 0 || encode_overlap < 0 ||
        (encode_chunk_size > 0 && encode_overlap >= encode_chunk_size) ||
        decode_chunk_size < 0 || decode_overlap < 0 ||
        (decode_chunk_size > 0 && decode_overlap >= decode_chunk_size)) {
        fprintf(stderr, "invalid encode/decode chunk size or overlap (overlap must be >= 0 and < chunk size)\n");
        return 1;
    }
    sa3::normalize_loudness_params(loudness);
    std::string loudness_err;
    if (!sa3::validate_loudness_params(loudness, loudness_err)) {
        fprintf(stderr, "invalid loudness settings: %s\n", loudness_err.c_str());
        return 1;
    }

    // ---------- build model paths + the request, then run the shared pipeline ----------
    sa3::ModelPaths paths;
    paths.tok = tok_p; paths.t5 = t5_p; paths.dit = dit_p; paths.same = same_p;
    if (cond_p) paths.cond = cond_p;

    sa3::GenParams params;
    params.prompt            = prompt;
    params.frames            = frames;
    params.steps             = steps;
    const uint64_t seed_resolved = sa3::pick_seed(seed);   // -1 => random
    params.seed              = seed_resolved;
    params.init_noise_level  = init_noise_level;
    params.inpaint_start     = inpaint_start;
    params.inpaint_end       = inpaint_end;
    params.encode_chunk_size = encode_chunk_size;
    params.encode_overlap    = encode_overlap;
    params.decode_chunk_size = decode_chunk_size;
    params.decode_overlap    = decode_overlap;
    params.loudness          = loudness;
    params.dist_shift        = dist_shift;
    params.ds_p1 = ds_p1; params.ds_p2 = ds_p2; params.ds_p3 = ds_p3; params.ds_p4 = ds_p4;
    params.duration_padding_sec = duration_padding_sec;
    params.negative_prompt   = negative_prompt;
    params.cfg_scale = cfg_scale; params.cfg_rescale = cfg_rescale; params.apg_scale = apg_scale;
    params.cfg_norm_threshold = cfg_norm_threshold;
    params.cfg_interval_min = cfg_interval_min; params.cfg_interval_max = cfg_interval_max;
    params.keep_models       = keep_models;   // no on_progress -> the pipeline prints the step lines itself
    for (auto& ls : lora_specs) params.loras.push_back(ls);

    if (init_p) {   // audio2audio / inpaint source WAV -> raw planar samples in the request
        int n_samp = 0, n_ch = 0, sr = 0;
        params.init_audio = sa3::read_wav_planar(init_p, n_samp, n_ch, sr);
        if (sr != 44100) {
            int resampled_n_samp = 0;
            params.init_audio = sa3::resample_planar_linear(params.init_audio, n_samp, n_ch, sr, 44100, resampled_n_samp);
            fprintf(stderr, "resampled init WAV from %d Hz/%d samples to 44100 Hz/%d samples\n",
                    sr, n_samp, resampled_n_samp);
            n_samp = resampled_n_samp;
        }
        params.init_n_samp = n_samp; params.init_n_ch = n_ch;
    }

    try {
        sa3::Pipeline pipe;
        pipe.load(paths);
        sa3::GenResult r = pipe.generate(params);
        double tp = sa3::wall_time_s();
        sa3::write_wav_planar(wav_p, r.samples.data(), r.n_samp, r.n_ch, r.sample_rate);
        sa3::profile_log(prof, "write_wav", sa3::wall_time_s() - tp);
        printf("wrote %s  (%.2fs, seed %llu)\n", wav_p, (float)r.n_samp / r.sample_rate, (unsigned long long)seed_resolved);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    sa3::profile_log(prof, "total", sa3::wall_time_s() - t_total0);
    return 0;
}
