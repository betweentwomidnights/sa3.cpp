#include "train_conditioning.h"

#include <cstdio>

int main() {
    sa3::TrainConditioning c;
    if (c.cond_dim != 0 || c.ctx_len != 0 || c.token_count != 0) return 1;
    std::printf("train_conditioning_compile_test: ok\n");
    return 0;
}
