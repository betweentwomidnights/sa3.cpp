// train_resume.h - exact native trainer continuation state and immutable checkpoint pairs.
#pragma once

#include "train_checkpoint.h"
#include "train_config.h"
#include "train_dataset.h"
#include "train_loop.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace sa3 {

struct TrainResumeProgress {
    uint64_t epoch = 0;
    uint64_t next_sample = 0;
    std::vector<uint64_t> order;
    std::string adapter_file;
    std::string adapter_fingerprint;
    std::string compatibility;
    std::string shuffle_rng;
    std::string crop_rng;
    std::string cfg_rng;
    std::string prompt_rng;
    std::string inpaint_rng;
    std::string diffusion_rng;
};

template <typename T>
inline std::string train_serialize_random_state(const T& value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

template <typename T>
inline bool train_restore_random_state(const std::string& text, T& value, const char* name,
                                       std::string& err) {
    std::istringstream in(text);
    T restored;
    if (!(in >> restored)) {
        err = std::string("invalid ") + name + " random state";
        return false;
    }
    value = restored;
    return true;
}

inline bool train_resolve_resume_pair(const std::string& input, std::string& adapter_path,
                                      std::string& state_path, std::string& err) {
    namespace fs = std::filesystem;
    const fs::path p(input);
    const std::string name = p.filename().string();
    static const std::string adapter_prefix = "adapter-step-";
    static const std::string state_prefix = "trainer-state-step-";
    if (name.rfind(adapter_prefix, 0) == 0 && p.extension() == ".gguf") {
        adapter_path = p.string();
        state_path = (p.parent_path() / (state_prefix + name.substr(adapter_prefix.size()))).string();
        return true;
    }
    if (name.rfind(state_prefix, 0) == 0 && p.extension() == ".gguf") {
        state_path = p.string();
        adapter_path = (p.parent_path() / (adapter_prefix + name.substr(state_prefix.size()))).string();
        return true;
    }
    err = "--resume expects adapter-step-N.gguf or trainer-state-step-N.gguf";
    return false;
}

inline int train_checkpoint_step_from_name(const std::string& name) {
    static const std::string prefix = "trainer-state-step-";
    static const std::string suffix = ".gguf";
    if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size() + suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) return -1;
    const std::string number = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
    int step = 0;
    for (char ch : number) {
        if (ch < '0' || ch > '9' || step > (std::numeric_limits<int>::max() - (ch - '0')) / 10) return -1;
        step = step * 10 + (ch - '0');
    }
    return step;
}

inline int train_latest_checkpoint_step(const std::string& directory) {
    namespace fs = std::filesystem;
    std::error_code ec;
    int latest = -1;
    for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        latest = std::max(latest, train_checkpoint_step_from_name(it->path().filename().string()));
    }
    return latest;
}

inline bool train_restore_adapter_values(TrainLoraState& expected, const TrainLoraState& loaded,
                                         std::string& err) {
    if (loaded.adapter_type != expected.adapter_type || loaded.rank != expected.rank ||
        loaded.alpha != expected.alpha) {
        err = "resume adapter type/rank/alpha does not match the current training configuration";
        return false;
    }
    std::map<std::string, const TrainLoraParam*> by_stem;
    for (const TrainLoraParam& p : loaded.params) {
        if (!by_stem.emplace(p.target.stem, &p).second) {
            err = "duplicate resume adapter target: " + p.target.stem;
            return false;
        }
    }
    if (by_stem.size() != expected.params.size()) {
        err = "resume adapter target count does not match the current model";
        return false;
    }
    auto copy_checked = [&](std::vector<float>& dst, const std::vector<float>& src,
                            const std::string& stem, const char* kind) {
        if (dst.size() != src.size()) {
            err = "resume adapter tensor shape mismatch: " + stem + "." + kind;
            return false;
        }
        dst = src;
        return true;
    };
    for (TrainLoraParam& dst : expected.params) {
        auto it = by_stem.find(dst.target.stem);
        if (it == by_stem.end()) {
            err = "resume adapter is missing target: " + dst.target.stem;
            return false;
        }
        const TrainLoraParam& src = *it->second;
        if (!copy_checked(dst.lora_A, src.lora_A, dst.target.stem, "lora_A") ||
            !copy_checked(dst.lora_B, src.lora_B, dst.target.stem, "lora_B") ||
            !copy_checked(dst.U, src.U, dst.target.stem, "U") ||
            !copy_checked(dst.V, src.V, dst.target.stem, "V") ||
            !copy_checked(dst.M_xs, src.M_xs, dst.target.stem, "M_xs") ||
            !copy_checked(dst.magnitude, src.magnitude, dst.target.stem, "magnitude") ||
            !copy_checked(dst.magnitude_r, src.magnitude_r, dst.target.stem, "magnitude_r") ||
            !copy_checked(dst.magnitude_c, src.magnitude_c, dst.target.stem, "magnitude_c")) return false;
    }
    return true;
}

inline uint64_t train_resume_fnv1a(uint64_t hash, const void* data, size_t size) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

inline void train_resume_hash_text(uint64_t& hash, const std::string& text) {
    hash = train_resume_fnv1a(hash, text.data(), text.size());
    const unsigned char separator = 0xff;
    hash = train_resume_fnv1a(hash, &separator, 1);
}

inline std::string train_resume_file_fingerprint(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot fingerprint checkpoint file: " + path; return {}; }
    uint64_t hash = UINT64_C(14695981039346656037);
    std::vector<char> buffer(1u << 20);
    while (f) {
        f.read(buffer.data(), (std::streamsize)buffer.size());
        const std::streamsize got = f.gcount();
        if (got > 0) hash = train_resume_fnv1a(hash, buffer.data(), (size_t)got);
    }
    if (!f.eof()) { err = "failed while fingerprinting checkpoint file: " + path; return {}; }
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

inline bool train_validate_resume_adapter_fingerprint(const std::string& path,
                                                      const std::string& expected,
                                                      std::string& err) {
    const std::string actual = train_resume_file_fingerprint(path, err);
    if (actual.empty()) return false;
    if (actual != expected) {
        err = "resume adapter does not match its trainer-state sidecar";
        return false;
    }
    return true;
}

inline std::string train_resume_file_identity(const std::string& path, bool include_contents) {
    namespace fs = std::filesystem;
    if (path.empty()) return "(none)";
    std::error_code ec;
    fs::path p = fs::weakly_canonical(fs::path(path), ec);
    if (ec) p = fs::absolute(fs::path(path), ec).lexically_normal();
    std::ostringstream out;
    out << p.generic_string();
    if (fs::is_directory(path, ec)) {
        std::vector<std::string> entries;
        ec.clear();
        for (fs::recursive_directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            const fs::path rel = fs::relative(it->path(), path, ec);
            if (ec) break;
            const uintmax_t entry_size = it->file_size(ec);
            if (ec) break;
            const auto stamp = it->last_write_time(ec).time_since_epoch().count();
            if (ec) break;
            entries.push_back(rel.generic_string() + "|" + std::to_string(entry_size) + "|" +
                              std::to_string(stamp));
        }
        std::sort(entries.begin(), entries.end());
        for (const std::string& entry : entries) out << "|entry=" << entry;
        return out.str();
    }
    ec.clear();
    const uintmax_t size = fs::file_size(path, ec);
    if (!ec) out << "|size=" << size;
    ec.clear();
    const auto stamp = fs::last_write_time(path, ec).time_since_epoch().count();
    if (!ec) out << "|mtime=" << stamp;
    if (include_contents) {
        std::ifstream f(path, std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        uint64_t h = UINT64_C(14695981039346656037);
        h = train_resume_fnv1a(h, contents.data(), contents.size());
        out << "|fnv=" << std::hex << std::setw(16) << std::setfill('0') << h;
    }
    return out.str();
}

// Stable compatibility fingerprint for every setting/input that can change the update trajectory.
// max_steps, checkpoint cadence, output path, and evaluation-only settings are intentionally
// excluded so a resumed run can be extended or moved without changing its math.
inline std::string train_resume_compatibility(const TrainConfig& c, const std::string& dit_path,
                                              const std::vector<TrainLoraTarget>& targets,
                                              const std::vector<TrainAudioCaptionPair>& pairs) {
    std::ostringstream cfg;
    cfg << std::setprecision(std::numeric_limits<float>::max_digits10)
        << c.model_variant << '|' << c.encoding << '|' << c.adapter_type << '|' << c.rank << '|'
        << c.alpha << '|' << c.learning_rate << '|' << c.weight_decay << '|' << c.adam_beta1 << '|'
        << c.adam_beta2 << '|' << c.adam_eps << '|' << c.batch_size << '|' << c.cpu_threads << '|'
        << c.frames << '|'
        << c.duration_sec << '|' << c.seed << '|' << c.random_crop << '|' << c.ckpt_backward << '|'
        << c.pre_encode << '|' << c.target_latent_rms << '|' << c.grad_clip << '|'
        << c.timestep_sampler << '|' << c.dist_shift << '|' << c.dist_shift_effective_length << '|'
        << c.cfg_dropout_prob << '|' << c.inpainting << '|' << c.inpaint_mask_probs << '|'
        << c.mask_loss_weight << '|' << c.mask_padding_attention << '|' << c.lr_scheduler << '|'
        << c.lr_inv_gamma << '|' << c.lr_power << '|' << c.lr_warmup << '|' << c.lr_final << '|'
        << c.train_split << '|' << train_resume_file_identity(dit_path, false) << '|'
        << train_resume_file_identity(c.latents_dir, false) << '|'
        << train_resume_file_identity(c.prompt_config_path, true);

    uint64_t hash = UINT64_C(14695981039346656037);
    train_resume_hash_text(hash, cfg.str());
    for (const TrainLoraTarget& target : targets) {
        train_resume_hash_text(hash, target.stem);
        train_resume_hash_text(hash, std::to_string(target.in));
        train_resume_hash_text(hash, std::to_string(target.out));
    }
    for (const TrainAudioCaptionPair& pair : pairs) {
        train_resume_hash_text(hash, pair.id);
        train_resume_hash_text(hash, pair.audio_rel);
        train_resume_hash_text(hash, pair.caption_rel);
        train_resume_hash_text(hash, pair.audio_sha256.empty()
            ? train_resume_file_identity(pair.audio_path, false) : pair.audio_sha256);
        train_resume_hash_text(hash, train_resume_file_identity(pair.caption_path, true));
        for (const auto& tag : pair.tags) {
            train_resume_hash_text(hash, tag.first);
            train_resume_hash_text(hash, tag.second);
        }
    }
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

inline bool train_validate_resume_compatibility(const std::string& saved, const std::string& current,
                                                std::string& err) {
    if (saved == current) return true;
    err = "resume checkpoint is incompatible with the current model, dataset, or training settings";
    return false;
}

struct TrainResumeTensorRef {
    std::string name;
    const std::vector<float>* data = nullptr;
};

inline bool train_collect_optimizer_tensors(const TrainLoraState& lora, const TrainLoopState& loop,
                                            std::vector<TrainResumeTensorRef>& tensors,
                                            std::string& err) {
    const size_t n = lora.params.size();
    auto collect = [&](size_t i, const char* kind, const std::vector<float>& param,
                       const std::vector<TrainAdamWTensorState>& states) {
        if (param.empty()) return true;
        if (states.size() != n || states[i].m.size() != param.size() || states[i].v.size() != param.size()) {
            err = "optimizer state is incomplete for " + lora.params[i].target.stem + "." + kind;
            return false;
        }
        const std::string base = "optimizer." + lora.params[i].target.stem + "." + kind;
        tensors.push_back({base + ".m", &states[i].m});
        tensors.push_back({base + ".v", &states[i].v});
        return true;
    };
    for (size_t i = 0; i < n; ++i) {
        const TrainLoraParam& p = lora.params[i];
        if (!collect(i, "lora_A", p.lora_A, loop.A_state) ||
            !collect(i, "lora_B", p.lora_B, loop.B_state) ||
            !collect(i, "M_xs", p.M_xs, loop.mxs_state) ||
            !collect(i, "magnitude", p.magnitude, loop.mag_state) ||
            !collect(i, "magnitude_r", p.magnitude_r, loop.mag_r_state) ||
            !collect(i, "magnitude_c", p.magnitude_c, loop.mag_c_state)) return false;
    }
    return true;
}

inline bool write_train_state_gguf(const TrainLoraState& lora, const TrainLoopState& loop,
                                   const TrainResumeProgress& progress, const std::string& out_path,
                                   std::string& err) {
    if (loop.step <= 0 || progress.order.empty() || progress.next_sample > progress.order.size()) {
        err = "cannot write invalid trainer continuation state";
        return false;
    }
    std::vector<TrainResumeTensorRef> tensors;
    if (!train_collect_optimizer_tensors(lora, loop, tensors, err)) return false;
    size_t data_bytes = 0;
    for (const TrainResumeTensorRef& tensor : tensors) data_bytes += tensor.data->size() * sizeof(float);
    ggml_init_params ip = {data_bytes + tensors.size() * 2 * (ggml_tensor_overhead() + 64) + (1u << 20),
                           nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) { err = "ggml_init failed while writing trainer state"; return false; }
    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "sa3-trainer-state");
    gguf_set_val_str(g, "general.name", "sa3 native trainer continuation state");
    gguf_set_val_u32(g, "trainer.state_version", 1);
    gguf_set_val_u64(g, "trainer.step", (uint64_t)loop.step);
    gguf_set_val_u64(g, "trainer.epoch", progress.epoch);
    gguf_set_val_u64(g, "trainer.next_sample", progress.next_sample);
    gguf_set_val_u32(g, "trainer.n_targets", (uint32_t)lora.params.size());
    gguf_set_val_str(g, "trainer.adapter_file", progress.adapter_file.c_str());
    gguf_set_val_str(g, "trainer.adapter_fingerprint", progress.adapter_fingerprint.c_str());
    gguf_set_val_str(g, "trainer.compatibility", progress.compatibility.c_str());
    gguf_set_val_str(g, "trainer.rng.shuffle", progress.shuffle_rng.c_str());
    gguf_set_val_str(g, "trainer.rng.crop", progress.crop_rng.c_str());
    gguf_set_val_str(g, "trainer.rng.cfg", progress.cfg_rng.c_str());
    gguf_set_val_str(g, "trainer.rng.prompt", progress.prompt_rng.c_str());
    gguf_set_val_str(g, "trainer.rng.inpaint", progress.inpaint_rng.c_str());
    gguf_set_val_str(g, "trainer.rng.diffusion", progress.diffusion_rng.c_str());
    gguf_set_arr_data(g, "trainer.order", GGUF_TYPE_UINT64, progress.order.data(), progress.order.size());
    std::vector<const char*> target_names;
    for (const TrainLoraParam& p : lora.params) target_names.push_back(p.target.stem.c_str());
    gguf_set_arr_str(g, "trainer.targets", target_names.data(), target_names.size());
    for (const TrainResumeTensorRef& tensor : tensors) {
        const int64_t ne[1] = {(int64_t)tensor.data->size()};
        train_checkpoint_add_tensor(ctx, g, tensor.name, *tensor.data, 1, ne);
    }
    const bool ok = gguf_write_to_file(g, out_path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) { err = "failed to write " + out_path; return false; }
    return true;
}

inline bool train_read_state_string(const GgufModel& g, const char* key, std::string& value,
                                    std::string& err) {
    const int i = gguf_find_key(g.gguf, key);
    if (i < 0 || gguf_get_kv_type(g.gguf, i) != GGUF_TYPE_STRING) {
        err = "trainer state is missing string metadata: " + std::string(key);
        return false;
    }
    value = gguf_get_val_str(g.gguf, i);
    return true;
}

inline bool load_train_state_gguf(const std::string& path, const TrainLoraState& lora,
                                  TrainLoopState& loop, TrainResumeProgress& progress,
    std::string& err) {
    try {
        TrainCheckpointCpuBackend cpu;
        GgufModel g = load_gguf(path.c_str(), cpu.backend);
        auto u64 = [&](const char* key, uint64_t& value) {
            const int i = gguf_find_key(g.gguf, key);
            if (i < 0 || gguf_get_kv_type(g.gguf, i) != GGUF_TYPE_UINT64) {
                err = "trainer state is missing integer metadata: " + std::string(key);
                return false;
            }
            value = gguf_get_val_u64(g.gguf, i);
            return true;
        };
        std::string architecture;
        if (!train_read_state_string(g, "general.architecture", architecture, err) ||
            architecture != "sa3-trainer-state") {
            if (err.empty()) err = "not an sa3 trainer-state GGUF: " + path;
            return false;
        }
        const int version_key = gguf_find_key(g.gguf, "trainer.state_version");
        if (version_key < 0 || gguf_get_kv_type(g.gguf, version_key) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(g.gguf, version_key) != 1) {
            err = "unsupported trainer-state version";
            return false;
        }
        uint64_t step = 0;
        if (!u64("trainer.step", step) || !u64("trainer.epoch", progress.epoch) ||
            !u64("trainer.next_sample", progress.next_sample) || step > (uint64_t)std::numeric_limits<int>::max())
            return false;
        loop = TrainLoopState{};
        loop.step = (int)step;
        if (!train_read_state_string(g, "trainer.adapter_file", progress.adapter_file, err) ||
            !train_read_state_string(g, "trainer.adapter_fingerprint", progress.adapter_fingerprint, err) ||
            !train_read_state_string(g, "trainer.compatibility", progress.compatibility, err) ||
            !train_read_state_string(g, "trainer.rng.shuffle", progress.shuffle_rng, err) ||
            !train_read_state_string(g, "trainer.rng.crop", progress.crop_rng, err) ||
            !train_read_state_string(g, "trainer.rng.cfg", progress.cfg_rng, err) ||
            !train_read_state_string(g, "trainer.rng.prompt", progress.prompt_rng, err) ||
            !train_read_state_string(g, "trainer.rng.inpaint", progress.inpaint_rng, err) ||
            !train_read_state_string(g, "trainer.rng.diffusion", progress.diffusion_rng, err)) return false;

        const int order_key = gguf_find_key(g.gguf, "trainer.order");
        if (order_key < 0 || gguf_get_kv_type(g.gguf, order_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g.gguf, order_key) != GGUF_TYPE_UINT64) {
            err = "trainer state is missing the dataset order";
            return false;
        }
        const size_t order_n = gguf_get_arr_n(g.gguf, order_key);
        const uint64_t* order = static_cast<const uint64_t*>(gguf_get_arr_data(g.gguf, order_key));
        progress.order.assign(order, order + order_n);
        if (progress.order.empty() || progress.next_sample > progress.order.size()) {
            err = "trainer state has an invalid dataset cursor";
            return false;
        }

        const int targets_key = gguf_find_key(g.gguf, "trainer.targets");
        if (targets_key < 0 || gguf_get_kv_type(g.gguf, targets_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g.gguf, targets_key) != GGUF_TYPE_STRING ||
            gguf_get_arr_n(g.gguf, targets_key) != lora.params.size()) {
            err = "trainer state target inventory does not match the adapter";
            return false;
        }
        for (size_t i = 0; i < lora.params.size(); ++i) {
            if (lora.params[i].target.stem != gguf_get_arr_str(g.gguf, targets_key, i)) {
                err = "trainer state target order does not match the current model";
                return false;
            }
        }

        const size_t n = lora.params.size();
        loop.A_state.resize(n); loop.B_state.resize(n); loop.mxs_state.resize(n);
        loop.mag_state.resize(n); loop.mag_r_state.resize(n); loop.mag_c_state.resize(n);
        auto load_component = [&](size_t i, const char* kind, const std::vector<float>& param,
                                  std::vector<TrainAdamWTensorState>& states) {
            if (param.empty()) return true;
            const std::string base = "optimizer." + lora.params[i].target.stem + "." + kind;
            const std::string mn = base + ".m", vn = base + ".v";
            if (!g.has(mn) || !g.has(vn)) {
                err = "trainer state is missing optimizer tensors for " + base;
                return false;
            }
            train_checkpoint_tensor_to_f32(g.get(mn), states[i].m);
            train_checkpoint_tensor_to_f32(g.get(vn), states[i].v);
            if (states[i].m.size() != param.size() || states[i].v.size() != param.size()) {
                err = "trainer optimizer tensor shape mismatch for " + base;
                return false;
            }
            return true;
        };
        for (size_t i = 0; i < n; ++i) {
            const TrainLoraParam& p = lora.params[i];
            if (!load_component(i, "lora_A", p.lora_A, loop.A_state) ||
                !load_component(i, "lora_B", p.lora_B, loop.B_state) ||
                !load_component(i, "M_xs", p.M_xs, loop.mxs_state) ||
                !load_component(i, "magnitude", p.magnitude, loop.mag_state) ||
                !load_component(i, "magnitude_r", p.magnitude_r, loop.mag_r_state) ||
                !load_component(i, "magnitude_c", p.magnitude_c, loop.mag_c_state)) return false;
        }
        g.free();
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// Step checkpoints are immutable. The state file is renamed last and therefore acts as the marker
// that both temporary files completed successfully.
inline bool write_train_checkpoint_pair(const TrainLoraState& lora, const TrainLoopState& loop,
                                        TrainResumeProgress progress, const std::string& adapter_path,
                                        const std::string& state_path, std::string& err) {
    namespace fs = std::filesystem;
    if (fs::exists(adapter_path) || fs::exists(state_path)) {
        err = "refusing to overwrite an existing immutable training checkpoint at step " +
              std::to_string(loop.step) + "; resume the latest checkpoint or choose --out";
        return false;
    }
    progress.adapter_file = fs::path(adapter_path).filename().string();
    const std::string adapter_tmp = adapter_path + ".tmp";
    const std::string state_tmp = state_path + ".tmp";
    std::error_code ec;
    fs::remove(adapter_tmp, ec); ec.clear(); fs::remove(state_tmp, ec);
    if (!write_train_lora_gguf(lora, adapter_tmp, err) ||
        (progress.adapter_fingerprint = train_resume_file_fingerprint(adapter_tmp, err)).empty() ||
        !write_train_state_gguf(lora, loop, progress, state_tmp, err)) {
        fs::remove(adapter_tmp, ec); ec.clear(); fs::remove(state_tmp, ec);
        return false;
    }
    fs::rename(adapter_tmp, adapter_path, ec);
    if (ec) {
        err = "failed to publish adapter checkpoint: " + ec.message();
        fs::remove(adapter_tmp, ec); fs::remove(state_tmp, ec);
        return false;
    }
    ec.clear();
    fs::rename(state_tmp, state_path, ec);
    if (ec) {
        err = "adapter was written, but trainer state could not be published: " + ec.message();
        fs::remove(state_tmp, ec);
        return false;
    }
    return true;
}

inline bool write_train_final_adapter(const TrainLoraState& lora, const std::string& path,
                                      std::string& err) {
    namespace fs = std::filesystem;
    const std::string tmp = path + ".tmp";
    std::error_code ec;
    fs::remove(tmp, ec);
    if (!write_train_lora_gguf(lora, tmp, err)) return false;
    ec.clear(); fs::remove(path, ec); ec.clear(); fs::rename(tmp, path, ec);
    if (ec) { err = "failed to publish final adapter: " + ec.message(); fs::remove(tmp, ec); return false; }
    return true;
}

} // namespace sa3
