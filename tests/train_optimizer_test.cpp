#include "train_optimizer.h"

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
    if (fails) return 1;
    std::printf("train_optimizer_test: ok\n");
    return 0;
}
