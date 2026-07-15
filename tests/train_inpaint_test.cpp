#include "train_inpaint.h"

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
    const int F = 64;

    // FULL only ([0,1,0]) => mask all zeros (generate everything).
    {
        std::mt19937_64 rng(1);
        for (int i = 0; i < 20; ++i) {
            sa3::InpaintMaskType ty;
            auto m = sa3::generate_inpaint_mask(F, F, {0.0, 1.0, 0.0}, rng, ty);
            bool all0 = true; for (float v : m) all0 = all0 && v == 0.0f;
            fails += expect(ty == sa3::InpaintMaskType::FullMask && all0, "FULL => all zeros");
        }
    }
    // CAUSAL only ([0,0,1]) => keep a prefix then generate the rest: mask is 1..1 0..0.
    {
        std::mt19937_64 rng(2);
        for (int i = 0; i < 200; ++i) {
            sa3::InpaintMaskType ty;
            auto m = sa3::generate_inpaint_mask(F, F, {0.0, 0.0, 1.0}, rng, ty);
            bool seen_zero = false, monotonic = true;
            for (float v : m) { if (v == 0.0f) seen_zero = true; else if (seen_zero) monotonic = false; }
            fails += expect(ty == sa3::InpaintMaskType::Causal && monotonic, "CAUSAL => prefix of ones");
        }
    }
    // SEGMENTS only ([1,0,0]) => at least some kept and structure varies; never errors.
    {
        std::mt19937_64 rng(3);
        sa3::InpaintMaskType ty;
        auto m = sa3::generate_inpaint_mask(F, F, {1.0, 0.0, 0.0}, rng, ty);
        fails += expect(ty == sa3::InpaintMaskType::RandomSegments, "SEGMENTS type");
        fails += expect((int)m.size() == F, "mask length F");
    }
    // Type distribution for [0.2,0.6,0.2]: all-zeros (== FULL, plus rare causal prefix=0) ~ 0.6.
    {
        std::mt19937_64 rng(1234);
        int all0 = 0; const int N = 20000;
        for (int i = 0; i < N; ++i) {
            sa3::InpaintMaskType ty;
            auto m = sa3::generate_inpaint_mask(F, F, {0.2, 0.6, 0.2}, rng, ty);
            bool a0 = true; for (float v : m) a0 = a0 && v == 0.0f;
            if (a0) ++all0;
        }
        const double frac = (double)all0 / N;
        fails += expect(frac > 0.57 && frac < 0.66, "FULL fraction ~0.6");
    }

    // build_train_inpaint: local layout + loss-weight normalization.
    {
        const int io = 4, frames = 6, local_dim = io + 1;
        std::vector<float> z((size_t)io * frames);
        for (size_t i = 0; i < z.size(); ++i) z[i] = (float)(i + 1);
        // mask: keep first 2 frames (ctx), generate last 4 (gen).
        std::vector<float> mask = {1, 1, 0, 0, 0, 0};
        auto ip = sa3::build_train_inpaint(z, mask, io, frames, local_dim, 1.0f);
        fails += expect(ip.n_ctx == 2 && ip.n_gen == 4, "n_ctx/n_gen counts");
        // channel 0 == mask; channels 1..io == z*mask.
        bool layout_ok = true;
        for (int t = 0; t < frames; ++t) {
            if (ip.local[(size_t)t * local_dim + 0] != mask[t]) layout_ok = false;
            for (int c = 0; c < io; ++c)
                if (ip.local[(size_t)t * local_dim + 1 + c] != z[(size_t)t * io + c] * mask[t]) layout_ok = false;
        }
        fails += expect(layout_ok, "local layout [mask | z*mask]");
        // Default (mask_padding_attention=true, the reference underfit branch): loss is the mean
        // over the GENERATE region only — gen weights sum to 1, context weights are ZERO.
        double sum_gen = 0, sum_ctx = 0;
        for (int t = 0; t < frames; ++t)
            for (int c = 0; c < io; ++c) {
                double w = ip.loss_weight[(size_t)t * io + c];
                if (mask[t] > 0.5f) sum_ctx += w; else sum_gen += w;
            }
        fails += expect(std::fabs(sum_gen - 1.0) < 1e-5, "gen weights sum to 1 (mask_padding_attention)");
        fails += expect(sum_ctx == 0.0, "ctx weights are zero (mask_padding_attention)");
        // mask_padding_attention=false: weighted-pooled mean (gen_sum + w*ctx_sum)/(gen_n + w*ctx_n).
        auto ip2 = sa3::build_train_inpaint(z, mask, io, frames, local_dim, 1.0f, false);
        double sum_gen2 = 0, sum_ctx2 = 0;
        for (int t = 0; t < frames; ++t)
            for (int c = 0; c < io; ++c) {
                double w = ip2.loss_weight[(size_t)t * io + c];
                if (mask[t] > 0.5f) sum_ctx2 += w; else sum_gen2 += w;
            }
        const double gen_n = 4.0 * io, ctx_n = 2.0 * io;
        fails += expect(std::fabs(sum_gen2 - gen_n / (gen_n + ctx_n)) < 1e-5, "pooled gen weight share");
        fails += expect(std::fabs(sum_ctx2 - ctx_n / (gen_n + ctx_n)) < 1e-5, "pooled ctx weight share");
    }
    // FULL mask (all generate): loss weight reduces to the uniform mean 1/(io*frames).
    {
        const int io = 4, frames = 6, local_dim = io + 1;
        std::vector<float> z((size_t)io * frames, 0.5f);
        std::vector<float> mask((size_t)frames, 0.0f);   // all generate
        auto ip = sa3::build_train_inpaint(z, mask, io, frames, local_dim, 1.0f);
        fails += expect(ip.n_gen == frames && ip.n_ctx == 0, "FULL: all gen");
        bool uniform = true;
        const float expect_w = 1.0f / (float)(io * frames);
        for (float w : ip.loss_weight) if (std::fabs(w - expect_w) > 1e-9f) uniform = false;
        fails += expect(uniform, "FULL loss weight == uniform 1/(io*frames)");
    }

    if (fails) return 1;
    std::printf("train_inpaint_test: ok\n");
    return 0;
}
