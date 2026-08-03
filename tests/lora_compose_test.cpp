// lora_compose_test: build_functional_lora must compose a chain of adapters to exactly what
// sequentially merging them produces. The reference here is the real merge path (apply_loras,
// which rewrites W_eff in place, each adapter reading the previous one's result) rather than a
// reimplementation of it, so the test pins the two paths to each other rather than to my algebra.
//
// The composed form under test (see DitLoraParam::terms) is
//     y = c_0 ⊙ (W@x) + Σ_k s_k · c_k ⊙ (B_k @ (A_k @ x)),
// with c_0 = Π_{dora j} d_j and c_k = Π_{dora j >= k} d_j. Since a later dora rescales earlier
// residuals, order matters, and the last case below checks it actually does.
#include "lora.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int fails = 0;

static void expect(bool ok, const std::string& msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg.c_str()); fails++; }
}

// ---- in-memory GgufModel construction (no file, no gguf metadata: apply_loras and
// build_functional_lora both read rank/alpha/strength off LoraAdapter, not off the gguf) ----
struct TensorSpec { std::string name; int64_t ne0 = 0, ne1 = 1; std::vector<float> data; };

static void build_model(sa3::GgufModel& m, const std::vector<TensorSpec>& specs) {
    ggml_init_params ip = { (specs.size() + 4) * ggml_tensor_overhead(), nullptr, true };
    m.ctx = ggml_init(ip);
    m.backend = sa3_test_cpu_backend();
    m.owns_backend = false;                        // the CPU backend is shared and static
    for (const auto& s : specs) {
        ggml_tensor* t = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, s.ne0, s.ne1);
        ggml_set_name(t, s.name.c_str());
        m.tensors[s.name] = t;
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.backend);
    for (const auto& s : specs)
        ggml_backend_tensor_set(m.tensors[s.name], s.data.data(), 0, s.data.size() * sizeof(float));
}

// The DiT weights this test adapts. `in`/`out` differ per target, and `solo` is targeted by only
// one adapter so the per-target chain really is per-target.
struct Target { std::string stem; int64_t in, out; };
static const std::vector<Target> kTargets = {
    { "dit.0.self.qkv", 8, 6 },
    { "dit.0.ff.up",    6, 10 },
    { "dit.1.self.out", 10, 4 },   // the "solo" target
};

struct AdapterSpec {
    std::string type;
    int rank = 2;
    float alpha = 4.0f, strength = 1.0f;
    bool skip_solo = false;        // leave dit.1.self.out untargeted
};

int main() {
    std::mt19937 rng(20260730);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    const int64_t SEQ = 5;

    // ---- the shared base weights, generated once and reused by every case ----
    std::vector<TensorSpec> base_specs;
    for (const auto& t : kTargets) {
        TensorSpec s{ t.stem + ".weight", t.in, t.out, {} };
        s.data.resize((size_t)t.in * t.out);
        for (auto& v : s.data) v = nd(rng);
        base_specs.push_back(std::move(s));
    }
    // activations, one per distinct input width
    std::vector<std::vector<float>> xs;
    for (const auto& t : kTargets) {
        std::vector<float> v((size_t)t.in * SEQ);
        for (auto& e : v) e = nd(rng);
        xs.push_back(std::move(v));
    }

    // ---- adapter tensor data, generated once per (spec index, target) so the merge and the
    // functional path see bit-identical adapters ----
    auto make_adapter_specs = [&](const AdapterSpec& as, uint32_t seed) {
        std::vector<TensorSpec> out;
        std::mt19937 r(seed);
        std::normal_distribution<float> n(0.0f, 0.4f);
        const bool xs_fam = sa3::functional_lora_is_xs(as.type);
        for (const auto& t : kTargets) {
            if (as.skip_solo && t.stem == "dit.1.self.out") continue;
            const int64_t rk = as.rank;
            if (xs_fam) {
                // U [rank,out], V [rank,in], M_xs [rank,rank] -- delta = U @ M_xs @ Vᵗ
                TensorSpec U{ t.stem + ".U", rk, t.out, {} };  U.data.resize((size_t)rk * t.out);
                TensorSpec V{ t.stem + ".V", rk, t.in,  {} };  V.data.resize((size_t)rk * t.in);
                TensorSpec M{ t.stem + ".M_xs", rk, rk, {} };  M.data.resize((size_t)rk * rk);
                for (auto& v : U.data) v = n(r);
                for (auto& v : V.data) v = n(r);
                for (auto& v : M.data) v = n(r);
                out.push_back(std::move(U)); out.push_back(std::move(V)); out.push_back(std::move(M));
            } else {
                TensorSpec A{ t.stem + ".lora_A", t.in, rk, {} };  A.data.resize((size_t)t.in * rk);
                TensorSpec B{ t.stem + ".lora_B", rk, t.out, {} }; B.data.resize((size_t)rk * t.out);
                for (auto& v : A.data) v = n(r);
                for (auto& v : B.data) v = n(r);
                out.push_back(std::move(A)); out.push_back(std::move(B));
            }
            if (as.type.rfind("dora-rows", 0) == 0) {
                // magnitude near the base row norm, as the trainer inits it, but perturbed so a
                // magnitude == norm coincidence cannot mask a bug.
                TensorSpec mg{ t.stem + ".magnitude", t.out, 1, {} };
                mg.data.resize((size_t)t.out);
                const std::vector<float>& w = base_specs[&t - &kTargets[0]].data;
                for (int64_t o = 0; o < t.out; o++) {
                    double s = 0;
                    for (int64_t i = 0; i < t.in; i++) { const float v = w[(size_t)o*t.in + i]; s += (double)v*v; }
                    mg.data[(size_t)o] = (float)std::sqrt(s) * 1.15f + 0.05f;
                }
                out.push_back(std::move(mg));
            }
        }
        return out;
    };

    // Run one chain: merge reference vs composed functional, over every target.
    auto run_case = [&](const std::string& label, const std::vector<AdapterSpec>& specs) {
        std::vector<std::vector<TensorSpec>> ad_specs;
        for (size_t i = 0; i < specs.size(); i++)
            ad_specs.push_back(make_adapter_specs(specs[i], 7000u + (uint32_t)i * 31u));

        auto load_adapters = [&]() {
            std::vector<sa3::LoraAdapter> v(specs.size());
            for (size_t i = 0; i < specs.size(); i++) {
                build_model(v[i].gguf, ad_specs[i]);
                v[i].type     = specs[i].type;
                v[i].rank     = specs[i].rank;
                v[i].alpha    = specs[i].alpha;
                v[i].strength = specs[i].strength;
            }
            return v;
        };

        // reference: sequential in-place merge, the shipped f16/f32 path
        sa3::GgufModel merged;  build_model(merged, base_specs);
        std::vector<sa3::LoraAdapter> ad_ref = load_adapters();
        sa3::apply_loras(merged, ad_ref);

        // under test: composed functional adapters over a pristine base
        sa3::GgufModel func;    build_model(func, base_specs);
        std::vector<sa3::LoraAdapter> ad_fun = load_adapters();
        sa3::FunctionalLora fl = sa3::build_functional_lora(func, ad_fun, "");
        expect(fl.active, label + ": functional path engaged");
        if (!fl.active) return;

        ggml_init_params ip = { 32u << 20, nullptr, false };
        ggml_context* c = ggml_init(ip);
        ggml_cgraph* g = ggml_new_graph_custom(c, 4096, false);
        std::vector<ggml_tensor*> refs, funs;
        for (size_t ti = 0; ti < kTargets.size(); ti++) {
            const Target& t = kTargets[ti];
            const std::string wname = t.stem + ".weight";
            ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, t.in, SEQ);
            std::memcpy(x->data, xs[ti].data(), xs[ti].size() * sizeof(float));
            ggml_tensor* rw = ggml_new_tensor_2d(c, GGML_TYPE_F32, t.in, t.out);
            ggml_backend_tensor_get(merged.tensors[wname], rw->data,
                                    0, (size_t)t.in * t.out * sizeof(float));
            refs.push_back(ggml_mul_mat(c, rw, x));
            funs.push_back(sa3::dit_lin(c, func, wname, x, nullptr, &fl.map));
            ggml_build_forward_expand(g, refs.back());
            ggml_build_forward_expand(g, funs.back());
        }
        sa3_test_compute(g);

        for (size_t ti = 0; ti < kTargets.size(); ti++) {
            const Target& t = kTargets[ti];
            // Scale the error by the RMS of the reference output, not by each element. The two
            // paths sum in different orders -- the reference forms W+delta and contracts once over
            // `in`, the functional path adds a separately accumulated residual -- so wherever the
            // base and the residual nearly cancel, a per-element relative error explodes on a
            // near-zero output while the absolute error stays at f32 epsilon.
            double sumsq = 0.0, maxabs = 0.0, worst_r = 0.0;
            for (int64_t k = 0; k < t.out * SEQ; k++) {
                const double r = ((float*)refs[ti]->data)[k], f = ((float*)funs[ti]->data)[k];
                sumsq += r * r;
                if (std::fabs(r - f) > maxabs) { maxabs = std::fabs(r - f); worst_r = r; }
            }
            const double rms = std::sqrt(sumsq / (double)(t.out * SEQ));
            const double err = maxabs / rms;
            if (getenv("VERBOSE"))
                std::printf("  %-34s %-16s err=%.3e  (maxabs=%.2e at |ref|=%.2e, rms=%.2e)\n",
                            label.c_str(), t.stem.c_str(), err, maxabs, std::fabs(worst_r), rms);
            expect(err < 1e-5, label + " @ " + t.stem + " (err " + std::to_string(err) + ")");
        }
        ggml_free(c);
        fl.free();
    };

    const AdapterSpec dora   { "dora-rows", 2, 4.0f, 1.0f, false };
    const AdapterSpec dora2  { "dora-rows", 4, 8.0f, 1.0f, false };
    const AdapterSpec plain  { "lora",      2, 4.0f, 1.0f, false };
    const AdapterSpec doraS  { "dora-rows", 2, 4.0f, 0.6f, false };   // non-unit strength
    const AdapterSpec plainS { "lora",      3, 6.0f, 1.7f, false };
    const AdapterSpec doraSolo{ "dora-rows", 2, 4.0f, 1.0f, true };   // skips one target
    const AdapterSpec doraXs { "dora-rows-xs", 2, 4.0f, 1.0f, false };
    const AdapterSpec loraXs { "lora-xs",      2, 4.0f, 1.0f, false };

    // single adapter: the already-validated path must not regress
    run_case("1x dora-rows",           { dora });
    run_case("1x lora",                { plain });
    // composition
    run_case("2x dora-rows",           { dora, dora2 });
    run_case("3x dora-rows",           { dora, dora2, doraS });
    run_case("2x lora",                { plain, plainS });
    run_case("lora then dora-rows",    { plain, dora });
    run_case("dora-rows then lora",    { dora, plain });
    run_case("dora, lora, dora",       { dora, plainS, dora2 });
    run_case("2x dora, strengths",     { doraS, dora2 });
    // partial overlap: the chain is per target, not global
    run_case("dora + dora(skips one)", { dora, doraSolo });
    run_case("dora(skips one) + dora", { doraSolo, dora });
    // -xs families: delta = U @ M_xs @ Vᵗ regrouped into the same term structure
    run_case("1x dora-rows-xs",        { doraXs });
    run_case("1x lora-xs",             { loraXs });
    run_case("dora-rows-xs + lora",    { doraXs, plain });
    run_case("lora-xs + dora-rows",    { loraXs, dora });
    run_case("dora-rows + dora-rows-xs", { dora, doraXs });

    // The chain must not commute: swapping two dora adapters has to change the output. Both
    // orders match their own merge reference above, so this only has to prove they differ.
    {
        auto composed_out = [&](const std::vector<AdapterSpec>& specs) {
            std::vector<std::vector<TensorSpec>> ad_specs;
            for (size_t i = 0; i < specs.size(); i++)
                ad_specs.push_back(make_adapter_specs(specs[i], 4242u + (uint32_t)i * 17u));
            sa3::GgufModel base; build_model(base, base_specs);
            std::vector<sa3::LoraAdapter> ads(specs.size());
            for (size_t i = 0; i < specs.size(); i++) {
                build_model(ads[i].gguf, ad_specs[i]);
                ads[i].type = specs[i].type; ads[i].rank = specs[i].rank;
                ads[i].alpha = specs[i].alpha; ads[i].strength = specs[i].strength;
            }
            sa3::FunctionalLora fl = sa3::build_functional_lora(base, ads, "");
            ggml_init_params ip = { 32u << 20, nullptr, false };
            ggml_context* c = ggml_init(ip);
            ggml_cgraph* g = ggml_new_graph_custom(c, 2048, false);
            const Target& t = kTargets[0];
            ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, t.in, SEQ);
            std::memcpy(x->data, xs[0].data(), xs[0].size() * sizeof(float));
            ggml_tensor* y = sa3::dit_lin(c, base, t.stem + ".weight", x, nullptr, &fl.map);
            ggml_build_forward_expand(g, y);
            sa3_test_compute(g);
            std::vector<float> out((float*)y->data, (float*)y->data + t.out * SEQ);
            ggml_free(c); fl.free();
            return out;
        };
        // two dora adapters that differ (different rank/alpha), so order genuinely matters
        std::vector<float> ab = composed_out({ dora, dora2 });
        std::vector<float> ba = composed_out({ dora2, dora });
        double maxrel = 0.0;
        for (size_t k = 0; k < ab.size(); k++)
            maxrel = std::max(maxrel, (double)std::fabs(ab[k] - ba[k]) / (std::fabs(ab[k]) + 1e-4));
        expect(maxrel > 1e-3, "dora chain does not commute (order is respected)");
    }

    // dora-cols / bora stay on the merge path: their scale factors onto the input, which needs a
    // chain-dependent column norm of the base and a Wᵗ-side contraction (see lora.h).
    {
        std::vector<sa3::LoraAdapter> v(1);
        v[0].type = "bora";
        expect(!sa3::functional_lora_ok(v), "bora is rejected by the functional path");
        v[0].type = "dora-cols";
        expect(!sa3::functional_lora_ok(v), "dora-cols is rejected by the functional path");
        v[0].type = "dora-rows";
        expect(sa3::functional_lora_ok(v), "dora-rows is accepted");
        std::vector<sa3::LoraAdapter> two(2);
        two[0].type = "dora-rows"; two[1].type = "lora-xs";
        expect(sa3::functional_lora_ok(two), "a mixed row-family chain is accepted");
        two[1].type = "dora-cols";
        expect(!sa3::functional_lora_ok(two), "one column-family member rejects the whole chain");
    }

    if (fails) return 1;
    std::printf("lora_compose_test: ok\n");
    return 0;
}
