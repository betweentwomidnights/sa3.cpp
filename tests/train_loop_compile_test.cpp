#include "train_loop.h"

#include <cstdio>

int main() {
    sa3::TrainLoopState s;
    if (s.step != 0) return 1;
    std::printf("train_loop_compile_test: ok\n");
    return 0;
}
