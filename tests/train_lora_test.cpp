#include "train_lora.h"
#include "test_backend.h"

#include <cstdio>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main() {
    int fails = 0;
    ggml_init_params ip = { 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    sa3::GgufModel m;
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    ggml_set_name(a, "dit.0.self.qkv.weight");
    float* ad = (float*)a->data;
    for (int i = 0; i < 32; ++i) ad[i] = (float)(i + 1);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(b, "dit.0.ff.out.bias");
    ggml_tensor* c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 5);
    ggml_set_name(c, "other.weight");
    m.tensors[a->name] = a;
    m.tensors[b->name] = b;
    m.tensors[c->name] = c;
    std::vector<sa3::TrainLoraTarget> targets = sa3::enumerate_train_lora_targets(m);
    fails += expect(targets.size() == 1, "one DiT 2D weight target");
    fails += expect(targets[0].stem == "dit.0.self.qkv", "target stem");
    fails += expect(targets[0].in == 4 && targets[0].out == 8, "target shape");
    sa3::TrainLoraState st;
    std::string err;
    fails += expect(sa3::init_train_lora_state(m, targets, "dora-rows", 2, 4.0f, 7, st, err), "init dora rows");
    fails += expect(st.params.size() == 1, "state target count");
    fails += expect(st.params[0].lora_A.size() == 8, "lora A shape");
    fails += expect(st.params[0].lora_B.size() == 16, "lora B shape");
    fails += expect(st.params[0].magnitude.size() == 8, "dora magnitude shape");
    fails += expect(st.params[0].magnitude[0] > 5.47f && st.params[0].magnitude[0] < 5.48f, "dora row norm");
    fails += expect(sa3::init_train_lora_state(m, targets, "lora-xs", 3, 3.0f, 7, st, err), "init xs");
    fails += expect(st.params[0].U.size() == 24 && st.params[0].V.size() == 12 && st.params[0].M_xs.size() == 9, "xs shapes");
    fails += expect(sa3::init_train_lora_state(m, targets, "bora-xs", 3, 3.0f, 7, st, err), "init bora xs");
    fails += expect(st.params[0].U.size() == 24 && st.params[0].V.size() == 12 && st.params[0].M_xs.size() == 9,
                    "bora xs low-rank shapes");
    fails += expect(st.params[0].magnitude_r.size() == 8 && st.params[0].magnitude_c.size() == 4,
                    "bora xs magnitude shapes");
    fails += expect(!sa3::init_train_lora_state(m, targets, "lora", 9, 9.0f, 7, st, err),
                    "infeasible rank rejected");
    ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 8);
    ggml_tensor* mag = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor* magc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    sa3::TrainLoraGraphParam gp;
    gp.lora_A = A;
    gp.lora_B = B;
    gp.magnitude = mag;
    gp.magnitude_r = mag;
    gp.magnitude_c = magc;
    ggml_tensor* ew = sa3::train_lora_effective_weight(ctx, a, gp, "lora", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "lora effective graph shape");
    ew = sa3::train_lora_effective_weight(ctx, a, gp, "dora-rows", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "dora rows graph shape");
    gp.magnitude = magc;
    ew = sa3::train_lora_effective_weight(ctx, a, gp, "dora-cols", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "dora cols graph shape");
    ew = sa3::train_lora_effective_weight(ctx, a, gp, "bora", 2, 2.0f);
    fails += expect(ew && ew->ne[0] == 4 && ew->ne[1] == 8, "bora graph shape");
    ggml_free(ctx);

    // --- randomized SVD recovers a known low-rank subspace ---
    {
        const int M_ = 6, N_ = 5, R_ = 2;
        const float U0[6][2] = {{1,0},{0,1},{1,1},{2,0},{0,2},{1,-1}};
        const float V0[5][2] = {{1,0},{0,1},{1,1},{-1,1},{2,0}};
        const float S0[2] = {5.0f, 2.0f};
        std::vector<float> Amat((size_t)M_ * N_, 0.0f);
        for (int o = 0; o < M_; ++o) for (int i = 0; i < N_; ++i) {
            float s = 0; for (int r = 0; r < R_; ++r) s += U0[o][r] * S0[r] * V0[i][r];
            Amat[(size_t)o * N_ + i] = s;
        }
        std::vector<float> Us, Vs;
        sa3::randomized_svd_topr(Amat, M_, N_, R_, 123, Us, Vs);
        // For a rank-R_ matrix, span(U) == column space, so U U^T A == A exactly.
        double err2 = 0, nrm2 = 0;
        for (int o = 0; o < M_; ++o) for (int i = 0; i < N_; ++i) {
            double rec = 0;
            for (int o2 = 0; o2 < M_; ++o2) {
                double d = 0; for (int r = 0; r < R_; ++r) d += Us[(size_t)o * R_ + r] * Us[(size_t)o2 * R_ + r];
                rec += d * Amat[(size_t)o2 * N_ + i];
            }
            const double a = Amat[(size_t)o * N_ + i];
            err2 += (rec - a) * (rec - a); nrm2 += a * a;
        }
        fails += expect(err2 < 1e-8 * nrm2, "randomized svd recovers rank-2 subspace");
    }

    // --- XS forward parity: train graph effective weight == U @ M_xs @ V^T reference ---
    {
        ggml_init_params ip2 = { 16 * 1024 * 1024, nullptr, false };
        ggml_context* c2 = ggml_init(ip2);
        const int IN = 4, OUT = 6, RK = 2; const float alpha = 3.0f;
        ggml_tensor* W  = ggml_new_tensor_2d(c2, GGML_TYPE_F32, IN, OUT);
        ggml_tensor* Ut = ggml_new_tensor_2d(c2, GGML_TYPE_F32, RK, OUT);  // U_g[a,o] = Um[o][a]
        ggml_tensor* Vt = ggml_new_tensor_2d(c2, GGML_TYPE_F32, RK, IN);   // V_g[a,i] = Vm[i][a]
        ggml_tensor* Mt = ggml_new_tensor_2d(c2, GGML_TYPE_F32, RK, RK);   // data[a*RK+b] = Mm[a][b]
        ggml_tensor* mag = ggml_new_tensor_1d(c2, GGML_TYPE_F32, OUT);
        const float Um[6][2] = {{0.2f,-0.1f},{0.3f,0.4f},{-0.5f,0.2f},{0.1f,0.1f},{0.6f,-0.3f},{-0.2f,0.5f}};
        const float Vm[4][2] = {{0.4f,0.1f},{-0.2f,0.3f},{0.5f,-0.4f},{0.1f,0.2f}};
        const float Mm[2][2] = {{0.7f,-0.2f},{0.3f,0.9f}};
        const float magv[6] = {1.1f, 0.9f, 1.3f, 0.7f, 1.0f, 1.2f};
        float* Wd = (float*)W->data; for (int k = 0; k < IN * OUT; ++k) Wd[k] = 0.1f * (float)(k + 1);
        for (int o = 0; o < OUT; ++o) for (int a = 0; a < RK; ++a) ((float*)Ut->data)[a + o * RK] = Um[o][a];
        for (int i = 0; i < IN; ++i)  for (int a = 0; a < RK; ++a) ((float*)Vt->data)[a + i * RK] = Vm[i][a];
        for (int a = 0; a < RK; ++a)  for (int b = 0; b < RK; ++b) ((float*)Mt->data)[a * RK + b] = Mm[a][b];
        for (int o = 0; o < OUT; ++o) ((float*)mag->data)[o] = magv[o];
        sa3::TrainLoraGraphParam gp; gp.U = Ut; gp.V = Vt; gp.M_xs = Mt; gp.magnitude = mag;

        // lora-xs: additive, exact
        ggml_tensor* ew = sa3::train_lora_effective_weight(c2, W, gp, "lora-xs", RK, alpha);
        ggml_cgraph* g = ggml_new_graph(c2); ggml_build_forward_expand(g, ew);
        sa3_test_compute(g);
        const float* ed = (const float*)ew->data;
        double maxerr = 0;
        std::vector<float> vref((size_t)IN * OUT);
        for (int o = 0; o < OUT; ++o) for (int i = 0; i < IN; ++i) {
            double delta = 0;
            for (int a = 0; a < RK; ++a) for (int b = 0; b < RK; ++b) delta += Um[o][a] * Mm[a][b] * Vm[i][b];
            const double v = 0.1 * (double)(i + o * IN + 1) + (alpha / RK) * delta;
            vref[(size_t)i + (size_t)o * IN] = (float)v;
            maxerr = std::max(maxerr, std::fabs(v - ed[i + o * IN]));
        }
        fails += expect(maxerr < 1e-4, "lora-xs effective weight matches U@M_xs@V^T");

        // dora-rows-xs: same delta, then per-output row-normalize and scale by magnitude[out]
        ggml_tensor* ew2 = sa3::train_lora_effective_weight(c2, W, gp, "dora-rows-xs", RK, alpha);
        ggml_cgraph* g2 = ggml_new_graph(c2); ggml_build_forward_expand(g2, ew2);
        sa3_test_compute(g2);
        const float* ed2 = (const float*)ew2->data;
        double maxerr2 = 0;
        for (int o = 0; o < OUT; ++o) {
            double rown = 0; for (int i = 0; i < IN; ++i) { const double x = vref[(size_t)i + (size_t)o * IN]; rown += x * x; }
            rown = std::sqrt(rown);
            for (int i = 0; i < IN; ++i) {
                const double exp = magv[o] * vref[(size_t)i + (size_t)o * IN] / rown;
                maxerr2 = std::max(maxerr2, std::fabs(exp - ed2[i + o * IN]));
            }
        }
        fails += expect(maxerr2 < 1e-4, "dora-rows-xs applies row-norm and magnitude");

        // M_xs receives a correct gradient through U @ M_xs @ V^T (finite-difference check).
        ggml_tensor* Mp = ggml_new_tensor_2d(c2, GGML_TYPE_F32, RK, RK);
        for (int a = 0; a < RK; ++a) for (int b = 0; b < RK; ++b) ((float*)Mp->data)[a * RK + b] = Mm[a][b];
        ggml_set_param(Mp);
        ggml_tensor* tgt = ggml_new_tensor_2d(c2, GGML_TYPE_F32, IN, OUT);
        for (int k = 0; k < IN * OUT; ++k) ((float*)tgt->data)[k] = 0.05f * (float)(k + 1);
        sa3::TrainLoraGraphParam gpg; gpg.U = Ut; gpg.V = Vt; gpg.M_xs = Mp;
        ggml_tensor* eff = sa3::train_lora_effective_weight(c2, W, gpg, "lora-xs", RK, alpha);
        ggml_tensor* loss = ggml_sum(c2, ggml_sqr(c2, ggml_sub(c2, eff, tgt)));
        ggml_set_loss(loss);
        ggml_cgraph* gb = ggml_new_graph_custom(c2, 1024, true);
        ggml_build_forward_expand(gb, loss);
        ggml_build_backward_expand(c2, gb, nullptr);
        ggml_graph_reset(gb);
        sa3_test_compute(gb);
        ggml_tensor* gM = ggml_graph_get_grad(gb, Mp);
        fails += expect(gM != nullptr, "M_xs has a gradient in the training graph");
        std::vector<float> ana((size_t)RK * RK);
        if (gM) for (int k = 0; k < RK * RK; ++k) ana[k] = ((float*)gM->data)[k];
        auto loss_at = [&](int a, int b, float d) -> double {
            const float save = ((float*)Mp->data)[a * RK + b];
            ((float*)Mp->data)[a * RK + b] = save + d;
            ggml_graph_reset(gb); sa3_test_compute(gb);
            const double L = ((float*)loss->data)[0];
            ((float*)Mp->data)[a * RK + b] = save;
            return L;
        };
        double gmax = 0;
        if (gM) for (int a = 0; a < RK; ++a) for (int b = 0; b < RK; ++b) {
            const double num = (loss_at(a, b, 1e-2f) - loss_at(a, b, -1e-2f)) / (2e-2);
            gmax = std::max(gmax, std::fabs(num - ana[a * RK + b]) / (std::fabs(num) + 1e-4));
        }
        fails += expect(gmax < 1e-2, "M_xs gradient matches finite differences");
        ggml_free(c2);
    }

    if (fails) return 1;
    std::printf("train_lora_test: inventory ok\n");
    return 0;
}
