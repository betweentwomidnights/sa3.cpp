#include "train_dit.h"

#include <cstdio>

int main() {
    sa3::TrainDitGraph g;
    if (g.loss != nullptr || g.target != nullptr) return 1;
    sa3::free_train_dit_graph(g);
    std::printf("train_dit_compile_test: ok\n");
    return 0;
}
