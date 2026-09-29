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

    const auto one_second = sa3::text2music_duration(44100, 12, 4096);
    expect(one_second.schedule_frames == 11, "1 s schedule length must use requested samples, not SAME-S even alignment");
    expect(std::fabs(one_second.seconds_total - 1.0f) < 1e-7f,
           "1 s conditioner must use exact requested duration");
    const auto frames_only = sa3::text2music_duration(0, 12, 4096);
    expect(frames_only.schedule_frames == 12, "--frames schedule length must use its frame count");
    expect(std::fabs(frames_only.seconds_total - 12.0f * 4096.0f / 44100.0f) < 1e-7f,
           "--frames conditioner must infer duration from frames");

    const auto transform_length = sa3::transform_duration(330750, 4096, 1);
    expect(transform_length.schedule_frames == 87, "transform schedule must include the 0.5 s margin");
    expect(std::fabs(transform_length.seconds_total - 8.0f) < 1e-7f,
           "transform conditioner must include the 0.5 s margin");
    expect(transform_length.canvas_samples == 151 * 4096,
           "transform canvas must include 6 s of hidden headroom");
    const auto small_transform_length = sa3::transform_duration(330750, 4096, 2);
    expect(small_transform_length.canvas_samples == 152 * 4096,
           "SAME-S transform canvas must align to an even latent frame count");

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
                    expect(schedule[i] >= 0.0f && schedule[i] <= 1.0f,
                           tag + ": timestep is out of range");
                }
            }
        }
    }

    // PyTorch builds linspace(sigma_max, 0) before LogSNR shift. Its first interior
    // point can rise above sigma_max, and ping-pong must accept that schedule.
    float p1 = 0.f, p2 = 0.f, p3 = 0.f, p4 = 0.f;
    sa3::dist_shift_defaults("LogSNR", p1, p2, p3, p4);
    const auto transform = sa3::make_sa3_schedule(8, 0.5f, 128, "LogSNR", p1, p2, p3, p4);
    expect(transform[1] > transform[0], "transform LogSNR must shift the scaled timestep");
    const float logsnr = p4 - (0.5f * 7.0f / 8.0f) * (p4 - p2);
    expect(std::fabs(transform[1] - 1.0f / (1.0f + std::exp(logsnr))) < 1e-6f,
           "transform LogSNR must match the upstream first interior timestep");
    const auto legacy = sa3::make_sa3_schedule(8, 0.5f, 128, "LogSNR", p1, p2, p3, p4, true);
    expect(legacy[1] < legacy[0], "legacy schedule must remain descending");
    const float old_logsnr = p4 - (7.0f / 8.0f) * (p4 - p2);
    const float old_first = 0.5f / (1.0f + std::exp(old_logsnr));
    expect(std::fabs(legacy[1] - old_first) < 1e-6f,
           "legacy schedule must preserve the former shift-then-scale calculation");
    float x = 0.25f, velocity = 0.1f, noise = -0.4f;
    sa3::sampling::rf_pingpong_step(&x, &velocity, &noise, 1, transform[0], transform[1]);
    const float expected = (1.0f - transform[1]) * (0.25f - transform[0] * velocity)
                         + transform[1] * noise;
    expect(std::fabs(x - expected) < 1e-6f, "ping-pong must accept an upward first timestep");

    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("OK\n");
    return 0;
}
