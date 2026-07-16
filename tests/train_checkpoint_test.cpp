#include "lora.h"
#include "train_checkpoint.h"
#include "train_resume.h"

#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main() {
    namespace fs = std::filesystem;
    int fails = 0;
    sa3::TrainLoraState st;
    st.adapter_type = "lora";
    st.rank = 2;
    st.alpha = 4.0f;
    sa3::TrainLoraParam p;
    p.target.weight_name = "dit.0.self.qkv.weight";
    p.target.stem = "dit.0.self.qkv";
    p.target.in = 3;
    p.target.out = 5;
    p.lora_A = {0, 1, 2, 3, 4, 5};
    p.lora_B = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    st.params.push_back(p);
    const fs::path out = fs::temp_directory_path() / "sa3_train_checkpoint_test.gguf";
    std::string err;
    fails += expect(sa3::write_train_lora_gguf(st, out.string(), err), err.c_str());
    sa3::LoraAdapter loaded = sa3::load_lora(out.string().c_str(), 0.75f);
    fails += expect(loaded.type == "lora", "loaded type");
    fails += expect(loaded.rank == 2, "loaded rank");
    fails += expect(loaded.alpha > 3.99f && loaded.alpha < 4.01f, "loaded alpha");
    fails += expect(loaded.strength > 0.74f && loaded.strength < 0.76f, "loaded strength");
    ggml_tensor* A = loaded.gguf.get("dit.0.self.qkv.lora_A");
    ggml_tensor* B = loaded.gguf.get("dit.0.self.qkv.lora_B");
    fails += expect(A->ne[0] == 3 && A->ne[1] == 2, "A shape");
    fails += expect(B->ne[0] == 2 && B->ne[1] == 5, "B shape");
    loaded.gguf.free();
    sa3::TrainLoraState resumed;
    err.clear();
    fails += expect(sa3::load_train_lora_gguf(out.string(), resumed, err), err.c_str());
    fails += expect(resumed.adapter_type == "lora", "resumed type");
    fails += expect(resumed.rank == 2 && resumed.params.size() == 1, "resumed shape metadata");
    fails += expect(resumed.params[0].target.stem == "dit.0.self.qkv", "resumed stem");
    fails += expect(resumed.params[0].lora_A == p.lora_A, "resumed A values");
    fails += expect(resumed.params[0].lora_B == p.lora_B, "resumed B values");

    // Exact continuation contract: uninterrupted AdamW must equal checkpoint + reload + continue.
    auto apply_step = [&](sa3::TrainLoraState& state, sa3::TrainLoopState& loop, int update) {
        sa3::TrainLoraGradAccum accum;
        const size_t n = state.params.size();
        accum.count = 1;
        accum.A.resize(n); accum.B.resize(n); accum.mxs.resize(n);
        accum.mag.resize(n); accum.mag_r.resize(n); accum.mag_c.resize(n);
        for (size_t i = 0; i < n; ++i) {
            accum.A[i].resize(state.params[i].lora_A.size());
            accum.B[i].resize(state.params[i].lora_B.size());
            for (size_t j = 0; j < accum.A[i].size(); ++j)
                accum.A[i][j] = 0.01f * (float)(update + 1) + 0.001f * (float)j;
            for (size_t j = 0; j < accum.B[i].size(); ++j)
                accum.B[i][j] = -0.02f * (float)(update + 1) + 0.0005f * (float)j;
        }
        sa3::TrainAdamWParams hp;
        hp.learning_rate = 1.0e-3f;
        hp.beta1 = 0.9f;
        hp.beta2 = 0.95f;
        hp.weight_decay = 0.01f;
        std::string step_err;
        return sa3::train_apply_accumulated_adamw(state, accum, loop, hp, step_err);
    };

    sa3::TrainLoraState uninterrupted = st;
    sa3::TrainLoraState split = st;
    sa3::TrainLoopState uninterrupted_loop, split_loop;
    for (int i = 0; i < 6; ++i) fails += expect(apply_step(uninterrupted, uninterrupted_loop, i), "uninterrupted AdamW step");
    for (int i = 0; i < 3; ++i) fails += expect(apply_step(split, split_loop, i), "pre-checkpoint AdamW step");

    std::mt19937_64 rng(12345);
    (void)rng(); (void)rng();
    sa3::TrainDiffusionSampler diffusion(6789, "trunc_logit_normal");
    std::vector<float> latent = {0.1f, -0.2f, 0.3f};
    sa3::TrainDiffusionSample first_sample;
    fails += expect(diffusion.sample(latent, first_sample, err), "pre-checkpoint diffusion sample");

    sa3::TrainResumeProgress progress;
    progress.epoch = 2;
    progress.next_sample = 1;
    progress.order = {1, 0};
    progress.compatibility = "test-compatibility";
    progress.shuffle_rng = sa3::train_serialize_random_state(rng);
    progress.crop_rng = progress.shuffle_rng;
    progress.cfg_rng = progress.shuffle_rng;
    progress.prompt_rng = progress.shuffle_rng;
    progress.inpaint_rng = progress.shuffle_rng;
    progress.diffusion_rng = diffusion.serialize_state();

    const fs::path pair_dir = fs::temp_directory_path() / "sa3_train_resume_checkpoint_test";
    const fs::path pair_adapter = pair_dir / "adapter-step-3.gguf";
    const fs::path pair_state = pair_dir / "trainer-state-step-3.gguf";
    fs::remove_all(pair_dir); fs::create_directories(pair_dir);
    err.clear();
    fails += expect(sa3::write_train_checkpoint_pair(split, split_loop, progress,
                                                     pair_adapter.string(), pair_state.string(), err),
                    err.c_str());
    fails += expect(!sa3::write_train_checkpoint_pair(split, split_loop, progress,
                                                      pair_adapter.string(), pair_state.string(), err),
                    "immutable checkpoint overwrite rejected");

    std::string resolved_adapter, resolved_state;
    err.clear();
    fails += expect(sa3::train_resolve_resume_pair(pair_adapter.string(), resolved_adapter, resolved_state, err),
                    "adapter resume path resolves");
    fails += expect(fs::path(resolved_state) == pair_state, "adapter derives state sidecar");
    fails += expect(sa3::train_resolve_resume_pair(pair_state.string(), resolved_adapter, resolved_state, err),
                    "state resume path resolves");
    fails += expect(fs::path(resolved_adapter) == pair_adapter, "state derives adapter sidecar");

    sa3::TrainLoraState reloaded_adapter;
    err.clear();
    fails += expect(sa3::load_train_lora_gguf(pair_adapter.string(), reloaded_adapter, err), err.c_str());
    sa3::TrainLoraState continued = st; // expected target order/shapes come from the current model.
    fails += expect(sa3::train_restore_adapter_values(continued, reloaded_adapter, err), err.c_str());
    sa3::TrainLoopState continued_loop;
    sa3::TrainResumeProgress reloaded_progress;
    fails += expect(sa3::load_train_state_gguf(pair_state.string(), continued, continued_loop,
                                              reloaded_progress, err), err.c_str());
    fails += expect(continued_loop.step == 3, "optimizer step restored");
    fails += expect(reloaded_progress.epoch == 2 && reloaded_progress.next_sample == 1,
                    "dataset cursor restored");
    fails += expect(reloaded_progress.order == progress.order, "dataset order restored");
    const std::string reloaded_adapter_fingerprint =
        sa3::train_resume_file_fingerprint(pair_adapter.string(), err);
    fails += expect(!reloaded_adapter_fingerprint.empty() &&
                    reloaded_adapter_fingerprint == reloaded_progress.adapter_fingerprint,
                    "trainer state is bound to the exact adapter file");
    const fs::path tampered_adapter = pair_dir / "tampered.gguf";
    fs::copy_file(pair_adapter, tampered_adapter);
    {
        std::fstream f(tampered_adapter, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(-1, std::ios::end);
        char byte = 0;
        f.read(&byte, 1);
        byte ^= 1;
        f.seekp(-1, std::ios::end);
        f.write(&byte, 1);
    }
    err.clear();
    fails += expect(!sa3::train_validate_resume_adapter_fingerprint(
                        tampered_adapter.string(), reloaded_progress.adapter_fingerprint, err),
                    "mismatched adapter/state pair rejected");
    fails += expect(sa3::train_validate_resume_compatibility(reloaded_progress.compatibility,
                                                            progress.compatibility, err),
                    "matching compatibility accepted");
    fails += expect(!sa3::train_validate_resume_compatibility(reloaded_progress.compatibility,
                                                             "different", err),
                    "incompatible continuation rejected");

    std::mt19937_64 restored_rng;
    err.clear();
    fails += expect(sa3::train_restore_random_state(reloaded_progress.shuffle_rng, restored_rng,
                                                   "shuffle", err), err.c_str());
    fails += expect(rng() == restored_rng(), "random engine continues exactly");
    sa3::TrainDiffusionSampler restored_diffusion(1, "trunc_logit_normal");
    fails += expect(restored_diffusion.restore_state(reloaded_progress.diffusion_rng, err), err.c_str());
    sa3::TrainDiffusionSample next_a, next_b;
    fails += expect(diffusion.sample(latent, next_a, err) && restored_diffusion.sample(latent, next_b, err),
                    "diffusion samplers continue");
    fails += expect(next_a.t == next_b.t && next_a.noise == next_b.noise,
                    "diffusion normal cache and RNG continue exactly");

    for (int i = 3; i < 6; ++i) fails += expect(apply_step(continued, continued_loop, i), "post-resume AdamW step");
    fails += expect(continued_loop.step == uninterrupted_loop.step, "final optimizer step exact");
    fails += expect(continued.params[0].lora_A == uninterrupted.params[0].lora_A &&
                    continued.params[0].lora_B == uninterrupted.params[0].lora_B,
                    "final adapter parameters exact");
    fails += expect(continued_loop.A_state[0].m == uninterrupted_loop.A_state[0].m &&
                    continued_loop.A_state[0].v == uninterrupted_loop.A_state[0].v &&
                    continued_loop.B_state[0].m == uninterrupted_loop.B_state[0].m &&
                    continued_loop.B_state[0].v == uninterrupted_loop.B_state[0].v,
                    "final AdamW moments exact");

    fs::remove(out);
    fs::remove_all(pair_dir);
    if (fails) return 1;
    std::printf("train_checkpoint_test: ok\n");
    return 0;
}
