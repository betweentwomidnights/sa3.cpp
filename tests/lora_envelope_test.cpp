#include "sa3_pipeline.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using Env = std::vector<std::pair<float, float>>;

static int fails = 0;

static void expect_near(float got, float want, const std::string& what) {
    if (std::fabs(got - want) > 1e-5f) {
        std::fprintf(stderr, "FAIL: %s: got %g, want %g\n", what.c_str(), got, want);
        ++fails;
    }
}

static float at(const Env& points, float t) { return sa3::envelope_amount(sa3::sorted_envelope(points), t); }

int main() {
    std::printf("lora_envelope_test\n");

    // A vertical drop: up to 2 at 4 s, straight down to 0, flat after.
    const Env drop = {{0.0f, 0.0f}, {4.0f, 2.0f}, {4.0f, 0.0f}, {8.0f, 0.0f}};
    expect_near(at(drop, 2.0f), 1.0f, "drop: ramp up before the edge");
    expect_near(at(drop, 3.999f), 1.9995f, "drop: just before the edge reads the first amount");
    expect_near(at(drop, 4.0f), 0.0f, "drop: at the edge reads the second amount");
    expect_near(at(drop, 5.0f), 0.0f, "drop: after the edge stays down");
    expect_near(at(drop, 7.0f), 0.0f, "drop: no ramp from the peak to the next point");

    // A vertical rise: 0 until 4 s, then straight up to 2 and held.
    const Env rise = {{0.0f, 0.0f}, {4.0f, 0.0f}, {4.0f, 2.0f}, {8.0f, 2.0f}};
    expect_near(at(rise, 3.999f), 0.0f, "rise: before the edge stays down");
    expect_near(at(rise, 4.0f), 2.0f, "rise: at the edge reads the second amount");
    expect_near(at(rise, 6.0f), 2.0f, "rise: after the edge stays up");

    // Order among tied points comes from the input, not from the amounts, wherever they sit.
    const Env shuffled = {{8.0f, 0.0f}, {4.0f, 2.0f}, {0.0f, 0.0f}, {4.0f, 0.0f}};
    expect_near(at(shuffled, 3.0f), 1.5f, "shuffled drop: ramp up before the edge");
    expect_near(at(shuffled, 5.0f), 0.0f, "shuffled drop: down after the edge");
    const auto sorted = sa3::sorted_envelope(shuffled);
    expect_near(sorted[1].second, 2.0f, "sorted: the first tied point stays first");
    expect_near(sorted[2].second, 0.0f, "sorted: the second tied point stays second");

    // Ties at either end: the first amount holds before the start, the last after the end.
    const Env ends = {{1.0f, 0.5f}, {1.0f, 1.0f}, {3.0f, 0.0f}, {3.0f, 0.25f}};
    expect_near(at(ends, 0.0f), 0.5f, "leading tie: before the start holds the first point");
    expect_near(at(ends, 1.0f), 1.0f, "leading tie: the start reads the second point");
    expect_near(at(ends, 2.0f), 0.5f, "leading tie: linear to the next time");
    expect_near(at(ends, 3.0f), 0.25f, "trailing tie: the end reads the last point");
    expect_near(at(ends, 9.0f), 0.25f, "trailing tie: held after the end");

    // Plain curves are unchanged: linear between points, held flat past either end.
    const Env plain = {{2.0f, 0.2f}, {10.0f, 1.0f}};
    expect_near(at(plain, 0.0f), 0.2f, "plain: held before the start");
    expect_near(at(plain, 6.0f), 0.6f, "plain: linear between points");
    expect_near(at(plain, 12.0f), 1.0f, "plain: held after the end");
    expect_near(at({{5.0f, 0.7f}}, 1.0f), 0.7f, "single point: constant");

    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("ok\n");
    return 0;
}
