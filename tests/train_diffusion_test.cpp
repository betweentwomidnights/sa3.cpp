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
    sa3::TrainDiffusionSample empty;
    err.clear();
    fails += expect(!sampler_a.sample({}, empty, err), "empty latent rejected");
    if (fails) return 1;
    std::printf("train_diffusion_test: ok\n");
    return 0;
}
