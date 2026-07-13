#include "train_diffusion.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main() {
    int fails = 0;
    std::vector<float> z = {1.0f, -2.0f, 0.5f, 0.0f};
    sa3::TrainDiffusionSampler sampler_a(123), sampler_b(123);
    sa3::TrainDiffusionSample a, b;
    std::string err;
    fails += expect(sampler_a.sample(z, a, err), "sample a");
    fails += expect(sampler_b.sample(z, b, err), "sample b");
    fails += expect(a.t == b.t, "deterministic t");
    fails += expect(a.noise == b.noise, "deterministic noise");
    for (size_t i = 0; i < z.size(); ++i) {
        const float denoised = a.x_t[i] - a.t * a.velocity_target[i];
        fails += expect(std::fabs(denoised - z[i]) < 1.0e-6f, "velocity recovers clean latent");
    }
    // draw_t + sample_at must reproduce sample() exactly (same RNG order), so a caller
    // can warp t between the draw and the noising (training dist-shift).
    sa3::TrainDiffusionSampler sampler_c(123);
    sa3::TrainDiffusionSample c;
    fails += expect(sampler_c.sample_at(z, sampler_c.draw_t(), c, err), "sample_at c");
    fails += expect(a.t == c.t, "sample_at t matches sample");
    fails += expect(a.noise == c.noise, "sample_at noise matches sample");
    fails += expect(a.x_t == c.x_t, "sample_at x_t matches sample");
    sa3::TrainDiffusionSample warped;
    fails += expect(sampler_c.sample_at(z, 0.25f, warped, err), "sample_at explicit t");
    fails += expect(warped.t == 0.25f, "explicit t stored");
    sa3::TrainDiffusionSample empty;
    err.clear();
    fails += expect(!sampler_a.sample({}, empty, err), "empty latent rejected");
    if (fails) return 1;
    std::printf("train_diffusion_test: ok\n");
    return 0;
}
