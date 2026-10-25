// rng.h — deterministic Gaussian noise, reproducible within sa3.cpp.
// Rng (the default): std::mt19937 is standardized; Box-Muller is hand-rolled so output is
// identical across compilers/platforms for a given seed. Not bit-matched to torch.randn.
// TorchCudaRandn (opt-in): reproduces torch.randn on a CUDA device, for seed parity with
// stable_audio_3 running on a GPU.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace sa3 {

struct Rng {
    std::mt19937 g;
    bool has_spare = false;
    float spare = 0.0f;

    explicit Rng(uint64_t seed) : g((uint32_t)seed) {}

    float uniform() { return ((g() >> 8) + 0.5f) * (1.0f / 16777216.0f); } // (0,1)

    float normal() {
        if (has_spare) { has_spare = false; return spare; }
        float u1 = uniform(), u2 = uniform();
        float r = sqrtf(-2.0f * logf(u1)), th = 6.283185307179586f * u2;
        spare = r * sinf(th); has_spare = true;
        return r * cosf(th);
    }

    void fill_normal(float* x, size_t n) { for (size_t i = 0; i < n; i++) x[i] = normal(); }
};

// The sequence of torch.randn / randn_like draws that follows torch.manual_seed(seed) on a CUDA
// device, in torch's C order. PyTorch's normal kernel (ATen DistributionTemplates.h) runs a
// grid-stride loop of 256-thread blocks: thread idx is cuRAND Philox4x32-10 subsequence idx, each
// curand_normal4 call yields 4 Box-Muller normals for the elements idx + k*stride (k = 0..3,
// stride = 256 * grid), and every call to the kernel advances the generator's counter offset by
// 4 per loop trip. The grid is capped at SMs * (max threads per SM / 256), so once a tensor needs
// more blocks than that, which element lands where depends on the GPU: max_blocks must then be
// the cap of the GPU being matched (0 = uncapped, exact for any tensor of at most 256 * cap
// elements). torch gives it as get_device_properties(0).multi_processor_count *
// (max_threads_per_multi_processor // 256): an RTX 5070 Laptop is 216, an RTX 4090 768, an
// L4 348, an A100 864. Draws in float32; a half-precision model rounds them afterwards.
struct TorchCudaRandn {
    uint64_t seed;
    uint64_t offset = 0;          // the generator's philox offset, in 32-bit outputs
    uint32_t max_blocks;

    explicit TorchCudaRandn(uint64_t seed, uint32_t max_blocks = 0) : seed(seed), max_blocks(max_blocks) {}

    struct Ctr { uint32_t x, y, z, w; };

    static Ctr philox(Ctr c, uint32_t k0, uint32_t k1) {
        for (int r = 0; r < 10; r++) {
            const uint64_t p0 = (uint64_t)0xD2511F53u * c.x, p1 = (uint64_t)0xCD9E8D57u * c.z;
            c = { (uint32_t)(p1 >> 32) ^ c.y ^ k0, (uint32_t)p1, (uint32_t)(p0 >> 32) ^ c.w ^ k1, (uint32_t)p0 };
            k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
        }
        return c;
    }

    // cuRAND _curand_box_muller: (s*sin v, s*cos v)
    static void box_muller(uint32_t a, uint32_t b, float& n0, float& n1) {
        const float inv = 2.3283064365386963e-10f, inv_2pi = 1.4629180792671596e-09f;
        const float u = (float)a * inv + inv * 0.5f;
        const float v = (float)b * inv_2pi + inv_2pi * 0.5f;
        const float s = sqrtf(-2.0f * logf(u));
        n0 = s * sinf(v); n1 = s * cosf(v);
    }

    // One torch.randn call of numel elements; out is C order.
    void fill(float* out, size_t numel) {
        if (numel == 0) return;
        const uint64_t block = 256;
        uint64_t grid = (numel + block - 1) / block;
        if (max_blocks > 0 && grid > max_blocks) grid = max_blocks;
        const uint64_t stride = block * grid, trips = (numel - 1) / (4 * stride) + 1;
        const uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
        for (uint64_t idx = 0; idx < stride; idx++)
            for (uint64_t j = 0; j < trips; j++) {
                const uint64_t ctr = offset / 4 + j;
                const Ctr r = philox({ (uint32_t)ctr, (uint32_t)(ctr >> 32), (uint32_t)idx, (uint32_t)(idx >> 32) }, k0, k1);
                float n[4];
                box_muller(r.x, r.y, n[0], n[1]);
                box_muller(r.z, r.w, n[2], n[3]);
                for (uint64_t k = 0; k < 4; k++) {
                    const uint64_t li = idx + (j * 4 + k) * stride;
                    if (li < numel) out[li] = n[k];
                }
            }
        offset += trips * 4;
    }
};

} // namespace sa3
