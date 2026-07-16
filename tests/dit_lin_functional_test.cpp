// dit_lin_functional_test: the functional LoRA/DoRA path in dit_lin must match the materialized
// mul_mat(train_lora_effective_weight(...), x) reference (see functional-lora-speed-plan). Also
// checks that dl==nullptr leaves dit_lin byte-identical to a plain mul_mat (inference unchanged),
// and that the trainable adapter tensors receive finite-difference-correct gradients.
#include "dit.h"
#include "test_backend.h"
#include "train_lora.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int expect(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); return 1; }
    return 0;
}

int main() {
    int fails = 0;
    const int64_t IN = 8, OUT = 6, RK = 2, SEQ = 4;
    const float alpha = 4.0f;
    const std::string name = "dit.0.self.qkv.weight";

    ggml_init_params ip = { 64 * 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);

    // Random base weight, adapter A/B, dora magnitude, and input activations.
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    ggml_tensor* Wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IN, OUT);   // [in,out]
    ggml_tensor* A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IN, RK);    // [in,rank]
    ggml_tensor* B  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, RK, OUT);   // [rank,out]
    ggml_tensor* mag = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OUT);      // [out]
    ggml_tensor* x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IN, SEQ);   // [in,seq]
    for (int64_t k = 0; k < IN * OUT; ++k) ((float*)Wt->data)[k] = nd(rng);
    for (int64_t k = 0; k < IN * RK;  ++k) ((float*)A->data)[k]  = nd(rng);
    for (int64_t k = 0; k < RK * OUT; ++k) ((float*)B->data)[k]  = nd(rng);
    for (int64_t k = 0; k < IN * SEQ; ++k) ((float*)x->data)[k]  = nd(rng);
    // dora magnitude = per-output column norm of the base (the init the trainer uses); perturbed a
    // little so a mag==norm coincidence can't hide a bug.
    for (int64_t o = 0; o < OUT; ++o) {
        double s = 0; for (int64_t i = 0; i < IN; ++i) { float v = ((float*)Wt->data)[o*IN+i]; s += (double)v*v; }
        ((float*)mag->data)[o] = (float)std::sqrt(s) * 1.1f + 0.05f;
    }
    ggml_set_name(Wt, name.c_str());
    sa3::GgufModel m;
    m.tensors[name] = Wt;

    // base_norm_sq host constant for the functional dora path.
    std::vector<float> bnsq = sa3::train_row_norm_sq(Wt, IN, OUT, 1e-12f);
    ggml_tensor* base_norm_sq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OUT);
    for (int64_t o = 0; o < OUT; ++o) ((float*)base_norm_sq->data)[o] = bnsq[o];

    sa3::TrainLoraGraphParam gp;
    gp.lora_A = A; gp.lora_B = B; gp.magnitude = mag;

    auto compare = [&](const std::string& fam, bool dora) -> double {
        sa3::DitLora dl;
        sa3::DitLoraParam dp;
        dp.A = A; dp.B = B; dp.scale = alpha / (float)RK; dp.in = IN;
        if (dora) { dp.dora = true; dp.magnitude = mag; dp.base_norm_sq = base_norm_sq; }
        dl[name] = dp;

        ggml_tensor* ref_w = sa3::train_lora_effective_weight(ctx, Wt, gp, fam, (int)RK, alpha);
        ggml_tensor* ref_y = ggml_mul_mat(ctx, ref_w, x);                 // [out,seq]
        ggml_tensor* fun_y = sa3::dit_lin(ctx, m, name, x, nullptr, &dl); // [out,seq]

        ggml_cgraph* g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, ref_y);
        ggml_build_forward_expand(g, fun_y);
        sa3_test_compute(g);

        double maxrel = 0.0;
        for (int64_t k = 0; k < OUT * SEQ; ++k) {
            const double r = ((float*)ref_y->data)[k], f = ((float*)fun_y->data)[k];
            maxrel = std::max(maxrel, std::fabs(r - f) / (std::fabs(r) + 1e-4));
        }
        return maxrel;
    };

    fails += expect(compare("lora", false) < 1e-4, "functional lora matches materialized W_eff");
    fails += expect(compare("dora-rows", true) < 1e-4, "functional dora-rows matches materialized W_eff");

    // f16 base weight: dit_lin no longer casts the base to f32 in-graph (the cast held a full f32
    // weight copy live across fwd+bwd; out_prod now handles f16 src0 directly). The functional
    // result must match the f32 materialized reference within f16 rounding of the base.
    {
        ggml_tensor* Wh = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, IN, OUT);
        ggml_fp32_to_fp16_row((const float*)Wt->data, (ggml_fp16_t*)Wh->data, IN * OUT);
        ggml_set_name(Wh, name.c_str());
        sa3::GgufModel m16;
        m16.tensors[name] = Wh;

        sa3::DitLora dl;
        sa3::DitLoraParam dp;
        dp.A = A; dp.B = B; dp.scale = alpha / (float)RK; dp.in = IN;
        dp.dora = true; dp.magnitude = mag; dp.base_norm_sq = base_norm_sq;
        dl[name] = dp;

        ggml_tensor* ref_w = sa3::train_lora_effective_weight(ctx, Wt, gp, "dora-rows", (int)RK, alpha);
        ggml_tensor* ref_y = ggml_mul_mat(ctx, ref_w, x);
        ggml_tensor* fun_y = sa3::dit_lin(ctx, m16, name, x, nullptr, &dl);
        ggml_cgraph* g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, ref_y);
        ggml_build_forward_expand(g, fun_y);
        sa3_test_compute(g);
        double maxrel = 0.0;
        for (int64_t k = 0; k < OUT * SEQ; ++k) {
            const double r = ((float*)ref_y->data)[k], f = ((float*)fun_y->data)[k];
            maxrel = std::max(maxrel, std::fabs(r - f) / (std::fabs(r) + 1e-4));
        }
        fails += expect(maxrel < 5e-3, "functional dora-rows with f16 base matches f32 reference");
    }

    // dl == nullptr must be byte-identical to a plain mul_mat (inference path unchanged).
    {
        ggml_tensor* plain = ggml_mul_mat(ctx, Wt, x);
        ggml_tensor* viadl = sa3::dit_lin(ctx, m, name, x, nullptr, nullptr);
        ggml_cgraph* g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, plain);
        ggml_build_forward_expand(g, viadl);
        sa3_test_compute(g);
        double maxabs = 0.0;
        for (int64_t k = 0; k < OUT * SEQ; ++k)
            maxabs = std::max(maxabs, (double)std::fabs(((float*)plain->data)[k] - ((float*)viadl->data)[k]));
        fails += expect(maxabs == 0.0, "dit_lin(dl=null) == plain mul_mat");
    }
    ggml_free(ctx);

    // --- gradient check: A, B, magnitude all get finite-difference-correct grads through dit_lin ---
    {
    // Run the gradient check twice: once with an f32 base and once with an f16 base. The f16 pass
    // exercises the out_prod f16-src0 path in the backward (grad w.r.t. x and, via the dora norm
    // term mul_mat(W,A), grad w.r.t. A both route through out_prod(src0=W)).
    std::vector<float> gx_f32;
    for (int pass = 0; pass < 2; ++pass) {
        const bool f16_base = pass == 1;
        ggml_init_params ip2 = { 64 * 1024 * 1024, nullptr, false };
        ggml_context* c2 = ggml_init(ip2);
        ggml_tensor* W2 = ggml_new_tensor_2d(c2, GGML_TYPE_F32, IN, OUT);
        ggml_tensor* A2 = ggml_new_tensor_2d(c2, GGML_TYPE_F32, IN, RK);
        ggml_tensor* B2 = ggml_new_tensor_2d(c2, GGML_TYPE_F32, RK, OUT);
        ggml_tensor* mag2 = ggml_new_tensor_1d(c2, GGML_TYPE_F32, OUT);
        ggml_tensor* x2 = ggml_new_tensor_2d(c2, GGML_TYPE_F32, IN, SEQ);
        ggml_tensor* tgt = ggml_new_tensor_2d(c2, GGML_TYPE_F32, OUT, SEQ);
        std::mt19937 rng2(99);
        for (int64_t k = 0; k < IN*OUT; ++k) ((float*)W2->data)[k] = nd(rng2);
        for (int64_t k = 0; k < IN*RK;  ++k) ((float*)A2->data)[k] = nd(rng2) * 0.3f;
        for (int64_t k = 0; k < RK*OUT; ++k) ((float*)B2->data)[k] = nd(rng2) * 0.3f;
        for (int64_t k = 0; k < IN*SEQ; ++k) ((float*)x2->data)[k] = nd(rng2);
        for (int64_t k = 0; k < OUT*SEQ;++k) ((float*)tgt->data)[k] = nd(rng2);
        for (int64_t o = 0; o < OUT; ++o) {
            double s = 0; for (int64_t i = 0; i < IN; ++i) { float v = ((float*)W2->data)[o*IN+i]; s += (double)v*v; }
            ((float*)mag2->data)[o] = (float)std::sqrt(s) + 0.1f;
        }
        ggml_tensor* Wbase = W2;
        if (f16_base) {
            ggml_tensor* W2h = ggml_new_tensor_2d(c2, GGML_TYPE_F16, IN, OUT);
            ggml_fp32_to_fp16_row((const float*)W2->data, (ggml_fp16_t*)W2h->data, IN * OUT);
            Wbase = W2h;
        }
        std::vector<float> bnsq2 = sa3::train_row_norm_sq(Wbase, IN, OUT, 1e-12f);
        ggml_tensor* bns2 = ggml_new_tensor_1d(c2, GGML_TYPE_F32, OUT);
        for (int64_t o = 0; o < OUT; ++o) ((float*)bns2->data)[o] = bnsq2[o];
        ggml_set_name(Wbase, name.c_str());
        sa3::GgufModel m2; m2.tensors[name] = Wbase;

        ggml_set_param(A2); ggml_set_param(B2); ggml_set_param(mag2);
        // x as param too: its grad goes through out_prod(src0=W) — the training-graph situation
        // (activations always need grads), and the op that gained an f16-src0 path.
        ggml_set_param(x2);
        sa3::DitLora dl; sa3::DitLoraParam dp;
        dp.A = A2; dp.B = B2; dp.scale = alpha/(float)RK; dp.in = IN;
        dp.dora = true; dp.magnitude = mag2; dp.base_norm_sq = bns2;
        dl[name] = dp;
        ggml_tensor* y = sa3::dit_lin(c2, m2, name, x2, nullptr, &dl);
        ggml_tensor* loss = ggml_sum(c2, ggml_sqr(c2, ggml_sub(c2, y, tgt)));
        ggml_set_loss(loss);
        ggml_cgraph* gb = ggml_new_graph_custom(c2, 2048, true);
        ggml_build_forward_expand(gb, loss);
        ggml_build_backward_expand(c2, gb, nullptr);
        ggml_graph_reset(gb);
        sa3_test_compute(gb);

        auto check_grad = [&](ggml_tensor* param, const char* label) {
            ggml_tensor* gT = ggml_graph_get_grad(gb, param);
            if (!gT) { fails += expect(false, label); return; }
            const int64_t n = ggml_nelements(param);
            double gmax = 0;
            for (int64_t k = 0; k < n; ++k) {
                const float ana = ((float*)gT->data)[k];
                const float save = ((float*)param->data)[k];
                ((float*)param->data)[k] = save + 1e-3f;
                ggml_graph_reset(gb); sa3_test_compute(gb);
                const double lp = ((float*)loss->data)[0];
                ((float*)param->data)[k] = save - 1e-3f;
                ggml_graph_reset(gb); sa3_test_compute(gb);
                const double lm = ((float*)loss->data)[0];
                ((float*)param->data)[k] = save;
                const double num = (lp - lm) / 2e-3;
                const double rel = std::fabs(num - ana) / (std::fabs(num) + 1e-2);
                if (rel > 2e-2 && k < 8) {
                    std::fprintf(stderr, "  %s[%lld]: analytic=%.6f numeric=%.6f\n",
                                 label, (long long)k, ana, num);
                }
                gmax = std::max(gmax, rel);
            }
            fails += expect(gmax < 2e-2, label);
        };
        check_grad(A2, f16_base ? "A gradient matches finite differences (f16 base)"
                                : "A gradient matches finite differences");
        check_grad(B2, f16_base ? "B gradient matches finite differences (f16 base)"
                                : "B gradient matches finite differences");
        check_grad(mag2, f16_base ? "magnitude gradient matches finite differences (f16 base)"
                                  : "magnitude gradient matches finite differences");

        // x-grad routes through out_prod(src0=W) — the training-graph situation (activations always
        // need grads) and the op that gained an f16-src0 path. Plain FD is unreliable for x (f32
        // loss noise at h=1e-3; the f16 forward quantizes x, distorting the step), so: pass 0
        // FD-checks with a large step (loss is quadratic in x, central FD is exact), and pass 1
        // compares the f16-base analytic x-grad against pass 0's f32-base analytic x-grad.
        {
            ggml_tensor* gX = ggml_graph_get_grad(gb, x2);
            fails += expect(gX != nullptr, "x gradient exists");
            ggml_graph_reset(gb); sa3_test_compute(gb);
            if (!f16_base) {
                // snapshot the analytic grad first: the FD recomputes below overwrite gX with
                // gradients evaluated at perturbed x
                std::vector<float> ana_all((float*)gX->data, (float*)gX->data + IN*SEQ);
                double gmax = 0;
                for (int64_t k = 0; k < IN*SEQ; ++k) {
                    const float ana = ana_all[(size_t)k];
                    const float save = ((float*)x2->data)[k];
                    const float h = 1e-2f;
                    ((float*)x2->data)[k] = save + h;
                    ggml_graph_reset(gb); sa3_test_compute(gb);
                    const double lp = ((float*)loss->data)[0];
                    ((float*)x2->data)[k] = save - h;
                    ggml_graph_reset(gb); sa3_test_compute(gb);
                    const double lm = ((float*)loss->data)[0];
                    ((float*)x2->data)[k] = save;
                    const double num = (lp - lm) / (2.0 * h);
                    gmax = std::max(gmax, std::fabs(num - ana) / (std::fabs(num) + 1e-1));
                }
                fails += expect(gmax < 2e-2, "x gradient matches finite differences");
                ggml_graph_reset(gb); sa3_test_compute(gb);
                gx_f32.assign((float*)gX->data, (float*)gX->data + IN*SEQ);
            } else {
                double gmax = 0;
                for (int64_t k = 0; k < IN*SEQ; ++k) {
                    const double r = gx_f32[(size_t)k], f = ((float*)gX->data)[k];
                    gmax = std::max(gmax, std::fabs(r - f) / (std::fabs(r) + 1e-2));
                }
                fails += expect(gmax < 2e-2, "x gradient with f16 base matches f32-base gradient (out_prod f16)");
            }
        }
        ggml_free(c2);
    }
    }

    if (fails) return 1;
    std::printf("dit_lin_functional_test: ok\n");
    return 0;
}
