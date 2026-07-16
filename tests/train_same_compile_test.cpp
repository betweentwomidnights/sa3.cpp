#include "train_same.h"

#include <cstdio>

int main() {
    sa3::TrainLatents lat;
    if (lat.frames != 0 || lat.latent != 0 || !lat.z.empty()) return 1;
    std::printf("train_same_compile_test: ok\n");
    return 0;
}
