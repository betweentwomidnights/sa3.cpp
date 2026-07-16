#include "train_optimizer.h"

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
    std::vector<float> p = {1.0f, -1.0f};
    std::vector<float> g = {0.5f, -0.25f};
    sa3::TrainAdamWTensorState st;
    sa3::TrainAdamWParams hp;
    hp.learning_rate = 0.1f;
    hp.weight_decay = 0.01f;
    std::string err;
    fails += expect(sa3::adamw_update_vector(p, g, st, hp, 1, err), "adamw update");
    fails += expect(p[0] < 1.0f, "positive grad decreases param");
    fails += expect(p[1] > -1.0f, "negative grad increases param");
    fails += expect(st.m.size() == 2 && st.v.size() == 2, "state initialized");
    std::vector<float> bad = {1.0f};
    fails += expect(!sa3::adamw_update_vector(p, bad, st, hp, 2, err), "size mismatch rejected");

    // InverseLR (Stage 11) closed form vs the Python reference at the ratatat-2 config
    // (base_lr 1e-4, inv_gamma 1e6, power 0.5, warmup 0.995, final_lr 0). Values from
    // stable_audio_3/training/utils.py InverseLR._get_closed_form_lr.
    struct { int n; float lr; } ref[] = {
        {0,    5.0000000000e-07f}, {1,   9.9749950125e-07f}, {10,  5.3645151961e-06f},
        {100,  3.9723855426e-05f}, {500, 9.1860637497e-05f}, {1000, 9.9288298382e-05f},
        {2499, 9.9874922989e-05f},
    };
    for (auto& r : ref) {
        const float got = sa3::inverse_lr(1.0e-4f, r.n, 1.0e6f, 0.5f, 0.995f, 0.0f);
        const float rel = std::fabs(got - r.lr) / r.lr;
        fails += expect(rel < 1.0e-5f, "inverse_lr matches reference");
    }
    // warmup=0 disables the warmup factor (lr == base_lr * lr_mult at step 0).
    fails += expect(std::fabs(sa3::inverse_lr(1.0e-4f, 0, 1.0e6f, 0.5f, 0.0f, 0.0f) - 1.0e-4f) < 1.0e-10f,
                    "inverse_lr no-warmup step0 == base_lr");
    if (fails) return 1;
    std::printf("train_optimizer_test: ok\n");
    return 0;
}
