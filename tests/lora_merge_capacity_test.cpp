// A realistic number of adapted weights must fit when blending 3+ adapters.
// Small composition fixtures hid a fixed per-weight graph budget behind its
// spare 64 nodes. Compare the graph merge with the independent host merge on
// every available model backend, for both unquantized base types.
#include "lora.h"
#include "rng.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static constexpr int kTargets = 24, kIn = 128, kOut = 32, kRank = 4;
struct Spec { std::string name; int in, out; std::vector<float> data; };

static void build(sa3::GgufModel& model, ggml_backend_t backend,
                  const std::vector<Spec>& specs, ggml_type type) {
    ggml_init_params ip = {(specs.size() + 4) * ggml_tensor_overhead(), nullptr, true};
    model.ctx = ggml_init(ip);
    model.backend = backend;
    model.owns_backend = false;
    for (const auto& s : specs) {
        auto* t = ggml_new_tensor_2d(model.ctx, type, s.in, s.out);
        ggml_set_name(t, s.name.c_str());
        model.tensors[s.name] = t;
    }
    model.buf = ggml_backend_alloc_ctx_tensors(model.ctx, backend);
    for (const auto& s : specs) sa3::write_from_f32(model.get(s.name), s.data);
}

static bool run(ggml_backend_t backend, ggml_type type, int count, bool mixed) {
    sa3::Rng rng(20261008);
    auto spec = [&](std::string name, int in, int out, float scale, float bias = 0.0f) {
        Spec s{std::move(name), in, out, std::vector<float>((size_t)in * out)};
        for (float& v : s.data) v = bias + scale * rng.normal();
        return s;
    };
    std::vector<std::string> targets;
    std::vector<Spec> weights;
    for (int t = 0; t < kTargets; ++t) {
        targets.push_back("dit." + std::to_string(t) + ".self.qkv.weight");
        weights.push_back(spec(targets.back(), kIn, kOut, 0.05f));
    }
    sa3::GgufModel actual, reference;
    build(actual, backend, weights, type);
    build(reference, sa3_test_cpu_backend(), weights, type);
    std::vector<sa3::LoraAdapter> adapters(count), ref_adapters(count);
    for (int i = 0; i < count; ++i) {
        const bool dora = !mixed || i % 2 == 0;
        std::vector<Spec> tensors;
        for (int t = 0; t < kTargets; ++t) {
            // Partial overlap and an identity adapter must not change the budget
            // or the numerical result of the remaining ordered chain.
            if (mixed && i % 3 == 1 && t % 5 == 0) continue;
            const auto stem = targets[t].substr(0, targets[t].size() - 7);
            tensors.push_back(spec(stem + ".lora_A", kIn, kRank, 0.025f));
            tensors.push_back(spec(stem + ".lora_B", kRank, kOut, 0.025f));
            if (dora) tensors.push_back(spec(stem + ".magnitude", kOut, 1, 0.05f, 0.6f));
        }
        build(adapters[i].gguf, backend, tensors, GGML_TYPE_F32);
        build(ref_adapters[i].gguf, sa3_test_cpu_backend(), tensors, GGML_TYPE_F32);
        for (auto* a : {&adapters[i], &ref_adapters[i]}) {
            a->type = dora ? "dora-rows" : "lora";
            a->rank = kRank;
            a->alpha = 2.0f * kRank;
            a->strength = mixed && i == count - 1 ? 0.0f : (i % 2 ? -0.4f : 0.7f);
        }
    }
    sa3::apply_loras_host(reference, ref_adapters, targets);
    sa3::apply_loras(actual, adapters);
    double max_error = 0.0, max_value = 0.0;
    for (const auto& name : targets) {
        std::vector<float> a, r;
        sa3::read_to_f32(actual.get(name), a);
        sa3::read_to_f32(reference.get(name), r);
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i])) return false;
            max_error = std::max(max_error, (double)std::fabs(a[i] - r[i]));
            max_value = std::max(max_value, (double)std::fabs(r[i]));
        }
    }
    const double relative = max_error / max_value;
    std::printf("%s: %s, %d %s adapters, error %.3e\n",
                ggml_backend_name(backend), ggml_type_name(type), count,
                mixed ? "mixed/partial/zero-strength" : "DoRA", relative);
    // GPU tensor-core matmuls can round their inputs even for F32 storage;
    // the CPU reference accumulates in F32. F16 also rounds the final weights.
    return relative < (type == GGML_TYPE_F16 ? 2e-3 : 5e-4);
}

int main(int argc, char**) {
    // --cpu makes the pre-fix assertion reproducible without GPU hardware.
    std::vector<Sa3TestBackend> backends = argc > 1
        ? std::vector<Sa3TestBackend>{{"CPU", sa3_test_cpu_backend()}}
        : sa3_test_backends();
    for (const auto& b : backends)
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
            for (int count : {1, 2, 3, 4, 8})
                for (bool mixed : {false, true})
                    if (!run(b.backend, type, count, mixed)) {
                        std::fprintf(stderr, "FAIL: LoRA graph merge differs from host reference\n");
                        return 1;
                    }
    std::puts("lora_merge_capacity_test: ok");
    return 0;
}
