#include "sa3_pipeline.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;

static void expect(bool ok, const std::string& message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message.c_str()); ++fails; }
}

int main() {
    std::printf("sampling_schedule_test\n");
    const std::vector<std::string> shifts = {"LogSNR", "Flux", "Full", "None"};
    const std::vector<float> starts = {0.01f, 0.5f, 0.85f, 1.0f};

    for (const auto& shift : shifts) {
        float p1 = 0.f, p2 = 0.f, p3 = 0.f, p4 = 0.f;
        sa3::dist_shift_defaults(shift, p1, p2, p3, p4);
        for (float sigma_max : starts) {
            for (int steps : {1, 2, 8, 16}) {
                const auto schedule = sa3::make_sa3_schedule(steps, sigma_max, 128,
                                                             shift, p1, p2, p3, p4);
                const std::string tag = shift + " sigma=" + std::to_string(sigma_max)
                                      + " steps=" + std::to_string(steps);
                expect(schedule.size() == (size_t)steps + 1, tag + ": wrong size");
                expect(std::fabs(schedule.front() - sigma_max) < 1e-7f, tag + ": wrong start");
                expect(schedule.back() == 0.0f, tag + ": wrong end");
                for (size_t i = 1; i < schedule.size(); ++i) {
                    expect(std::isfinite(schedule[i]), tag + ": non-finite timestep");
                    expect(schedule[i] >= 0.0f && schedule[i] <= schedule[i - 1],
                           tag + ": schedule is not descending");
                }
            }
        }
    }

    // Regression: the old code warped 0.5 * 7/8 directly and produced ~0.83, above its 0.5 start.
    float p1 = 0.f, p2 = 0.f, p3 = 0.f, p4 = 0.f;
    sa3::dist_shift_defaults("LogSNR", p1, p2, p3, p4);
    const auto transform = sa3::make_sa3_schedule(8, 0.5f, 128, "LogSNR", p1, p2, p3, p4);
    expect(transform[1] < transform[0], "transform LogSNR first step must remain below sigma_max");

    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("OK\n");
    return 0;
}
