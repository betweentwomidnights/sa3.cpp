// What --rng torch-cuda relies on to reproduce stable_audio_3's noise: TorchCudaRandn against
// values printed by torch 2.7 on CUDA, and the text2music canvas sized as _adapt_sample_size does.
#include "sa3_pipeline.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

static int fails = 0;

static void expect(bool ok, const std::string& message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message.c_str()); ++fails; }
}

// GPU sin/log run in fast-math, so torch's values sit within a few 1e-6 of the exact ones
static void expect_values(const std::vector<float>& got, const std::vector<std::pair<size_t, float>>& want,
                          const std::string& what) {
    for (const auto& w : want)
        expect(std::fabs(got[w.first] - w.second) < 2e-5f,
               what + "[" + std::to_string(w.first) + "] = " + std::to_string(got[w.first]) +
               ", torch gives " + std::to_string(w.second));
}

int main() {
    std::printf("torch_parity_test\n");

    // torch.manual_seed(42); a = torch.randn([1, 256, 16], device="cuda"); b = torch.randn_like(a)
    // c = torch.randn([524288], device="cuda") on a GPU whose grid cap is 216 blocks (RTX 5070
    // Laptop), so c takes three grid-stride trips per thread.
    sa3::TorchCudaRandn r(42, 216);
    std::vector<float> a(4096), b(4096), c(524288);
    r.fill(a.data(), a.size());
    r.fill(b.data(), b.size());
    r.fill(c.data(), c.size());
    expect_values(a, {{0, 0.194019f}, {1, 2.161374f}, {2, -0.17205f}, {3, 0.84906f},
                      {255, -1.800972f}, {256, -1.935122f}, {4095, -1.253528f}}, "first draw");
    expect_values(b, {{0, 0.139138f}, {1, -0.108217f}, {4095, 1.116468f}}, "second draw (offset 4)");
    expect_values(c, {{0, -0.518749f}, {55295, -0.511168f}, {55296, -0.336962f}, {221183, 1.191622f},
                      {221184, -0.697366f}, {524287, -0.349366f}}, "grid-capped draw (offset 8)");
    expect(r.offset == 8 + 12, "a grid-capped call advances the offset by 4 per trip");

    // An uncapped generator agrees wherever the tensor fits the cap.
    sa3::TorchCudaRandn u(42);
    std::vector<float> a2(4096);
    u.fill(a2.data(), a2.size());
    expect(a2 == a, "max_blocks only matters past 256 * cap elements");

    // _adapt_sample_size: int((10 + 6) * 44100) = 705600 samples -> 173 frames of 4096 -> even 174
    expect(sa3::text2music_canvas_frames(10.0f, 6.0f, 4096) == 174, "10 s canvas is 174 frames");
    expect(sa3::text2music_canvas_frames(30.0f, 6.0f, 4096) == 388, "30 s canvas is 388 frames");
    expect(sa3::text2music_canvas_frames(0.5f, 6.0f, 4096) == 70, "0.5 s canvas is 70 frames");

    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("ok\n");
    return 0;
}
