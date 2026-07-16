// train_inpaint.h - inpainting-mask training objective for native SA3 LoRA training (Stage 12).
//
// Ports stable_audio_3/models/inpainting.py random_inpaint_mask (single item, batch=1) plus the
// training wrapper's local-cond assembly and inpaint-aware loss weighting
// (training/diffusion.py lines 405-489). Per sample:
//   1. pick a mask type by mask_type_probabilities (default [0.1,0.8,0.1]; ratatat-2 [0.2,0.6,0.2]);
//   2. build inpaint_mask over the latent frames (1 = keep/context, 0 = inpaint/generate);
//   3. local_add_cond = [inpaint_mask(1ch) | latent*mask(io ch)]  (== inference layout in
//      sa3_pipeline.h, so the same dit.local.* weights apply);
//   4. loss weight per frame folds in the reference's mean_gen + mask_loss_weight*mean_ctx:
//      generated frames get 1/(io*N_gen); context frames get mask_loss_weight/(io*N_ctx).
//
// RNG matches the reference *distribution* (a deterministic std::mt19937_64), not Python's exact
// sequence. Only 3-type ([SEGMENTS,FULL,CAUSAL]) and 4-type ([...,RANDOM_SPANS]) prob vectors are
// supported, matching the reference's num_probs branch.
#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace sa3 {

enum class InpaintMaskType { RandomSegments = 0, FullMask = 1, Causal = 2, RandomSpans = 3 };

// random.randint(a,b): inclusive both ends.
inline int inpaint_randint(std::mt19937_64& rng, int a, int b) {
    if (b < a) return a;
    return std::uniform_int_distribution<int>(a, b)(rng);
}
inline double inpaint_rand01(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}
// random.uniform(lo,hi).
inline double inpaint_uniform(std::mt19937_64& rng, double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
}
// random.choices(indices, weights, k=1)[0] over [0, weights.size()).
inline int inpaint_weighted_choice(const std::vector<double>& weights, std::mt19937_64& rng) {
    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total <= 0.0) return inpaint_randint(rng, 0, (int)weights.size() - 1);
    double u = inpaint_rand01(rng) * total, acc = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) { acc += weights[i]; if (u < acc) return (int)i; }
    return (int)weights.size() - 1;
}

// Port of _generate_random_spans_mask: 1..max_spans non-overlapping spans covering a target ratio.
inline void inpaint_random_spans(std::vector<float>& mask, int real_len, std::mt19937_64& rng,
                                 int max_spans, double ratio_lo, double ratio_hi,
                                 const std::vector<double>& span_count_weights) {
    if (real_len == 0) return;
    const double target_ratio = inpaint_uniform(rng, ratio_lo, ratio_hi);
    int target = std::max(1, (int)(target_ratio * real_len));

    std::vector<double> weights = span_count_weights.empty()
        ? std::vector<double>{4, 2, 1, 1} : span_count_weights;
    weights.resize(max_spans, weights.empty() ? 1.0 : weights.back());
    int num_spans = inpaint_weighted_choice(weights, rng) + 1;   // choices over [1..max_spans]
    num_spans = std::min(num_spans, real_len);
    if (num_spans <= 0) return;

    std::vector<double> raw(num_spans);
    double tot = 0.0;
    for (double& w : raw) { w = inpaint_rand01(rng); tot += w; }
    if (tot <= 0.0) tot = 1.0;
    std::vector<int> span_len(num_spans);
    for (int i = 0; i < num_spans; ++i) span_len[i] = std::max(1, (int)(raw[i] / tot * target));

    int diff = target - std::accumulate(span_len.begin(), span_len.end(), 0);
    if (diff > 0) {
        span_len[0] += diff;
    } else if (diff < 0) {
        for (int i = 0; i < num_spans && diff < 0; ++i) {
            int reduce = std::min(-diff, span_len[i] - 1);
            span_len[i] -= reduce; diff += reduce;
        }
    }
    std::sort(span_len.begin(), span_len.end(), std::greater<int>());

    std::vector<std::pair<int, int>> placed;
    auto overlaps = [&](int s, int e) {
        for (auto& p : placed) if (p.first < e && s < p.second) return true;
        return false;
    };
    for (int length : span_len) {
        length = std::min(length, real_len);
        if (length <= 0) continue;
        int max_start = real_len - length;
        if (max_start < 0) continue;
        bool done = false;
        for (int a = 0; a < 50 && !done; ++a) {
            int s = inpaint_randint(rng, 0, max_start);
            if (!overlaps(s, s + length)) { placed.push_back({s, s + length}); for (int t = s; t < s + length; ++t) mask[t] = 0.0f; done = true; }
        }
        if (!done) for (int s = 0; s <= max_start; ++s)
            if (!overlaps(s, s + length)) { placed.push_back({s, s + length}); for (int t = s; t < s + length; ++t) mask[t] = 0.0f; break; }
    }
}

// Generate an inpaint mask over `frames` (1 = keep, 0 = generate). real_len = # non-padding frames.
inline std::vector<float> generate_inpaint_mask(int frames, int real_len,
        const std::vector<double>& probs, std::mt19937_64& rng, InpaintMaskType& type_out,
        bool mask_padding = true, int max_mask_segments = 10, int max_spans = 4,
        double mask_ratio_lo = 0.2, double mask_ratio_hi = 1.0,
        const std::vector<double>& span_count_weights = {}) {
    std::vector<float> mask((size_t)frames, 1.0f);
    // Map prob-vector length to the active type set, matching the reference num_probs branch.
    std::vector<InpaintMaskType> types;
    if (probs.size() == 4) types = {InpaintMaskType::RandomSegments, InpaintMaskType::FullMask,
                                    InpaintMaskType::Causal, InpaintMaskType::RandomSpans};
    else                    types = {InpaintMaskType::RandomSegments, InpaintMaskType::FullMask,
                                    InpaintMaskType::Causal};
    const InpaintMaskType type = types[(size_t)inpaint_weighted_choice(probs, rng)];
    type_out = type;

    if (type == InpaintMaskType::FullMask) {
        std::fill(mask.begin(), mask.end(), 0.0f);
        return mask;
    }
    if (real_len == 0) return mask;   // all keep

    if (type == InpaintMaskType::RandomSegments) {
        int num_segments = inpaint_randint(rng, 1, max_mask_segments);
        int max_len = std::max(1, real_len / num_segments);
        for (int s = 0; s < num_segments; ++s) {
            int seg = inpaint_randint(rng, 1, max_len);
            if (real_len - seg < 0) continue;
            int start = inpaint_randint(rng, 0, real_len - seg);
            for (int t = start; t < start + seg; ++t) mask[t] = 0.0f;
        }
    } else if (type == InpaintMaskType::RandomSpans) {
        inpaint_random_spans(mask, real_len, rng, max_spans, mask_ratio_lo, mask_ratio_hi, span_count_weights);
    } else if (type == InpaintMaskType::Causal) {
        int prefix = inpaint_randint(rng, 0, real_len);   // keep [0,prefix), generate [prefix,real_len)
        for (int t = prefix; t < real_len; ++t) mask[t] = 0.0f;
    }
    // Attention-masked training: padding region is not provided context.
    if (mask_padding) for (int t = real_len; t < frames; ++t) mask[t] = 0.0f;
    return mask;
}

// Host-built inpainting inputs for one training step.
struct TrainInpaint {
    std::vector<float> local;        // [local_dim * frames], idx = t*local_dim + ch (ch0=mask, ch1..io=z*mask)
    std::vector<float> loss_weight;  // [io * frames],        idx = t*io + c   (per-frame, broadcast over c)
    InpaintMaskType type = InpaintMaskType::FullMask;
    int n_gen = 0, n_ctx = 0;
};

// Assemble the local-add conditioning and the inpaint-aware per-position loss weight from the clean
// latent z ([io, frames], idx = t*io + c) and a frame mask (1 keep / 0 generate).
//
// The loss weights reproduce underfit training/loss.py compute_masked_loss — the trainer the
// reference ratatat runs actually used — NOT the stock stable-audio-3 diffusion.py loss:
//   mask_padding_attention = true  (the sa3-medium model config; all ratatat runs):
//       loss = gen_sum / (gen_count + 1e-8)          — the context region contributes ZERO and
//                                                       mask_loss_weight is IGNORED
//   mask_padding_attention = false:
//       loss = (gen_sum + w*ctx_sum) / (gen_count + w*ctx_count + 1e-8)
// (counts include channels: gen_count = io * n_gen). The earlier port summed two per-region
// means (mean_gen + w*mean_ctx), which spent ~half the gradient signal of every segments/causal
// step on a trivial copy-the-given-context task and produced prompt-ignoring collapsed adapters.
inline TrainInpaint build_train_inpaint(const std::vector<float>& z, const std::vector<float>& mask,
                                        int io, int frames, int local_dim, float mask_loss_weight,
                                        bool mask_padding_attention = true) {
    TrainInpaint out;
    out.local.assign((size_t)local_dim * frames, 0.0f);
    for (int t = 0; t < frames; ++t) {
        const float m = mask[(size_t)t];
        out.local[(size_t)t * local_dim + 0] = m;
        for (int c = 0; c < io; ++c) out.local[(size_t)t * local_dim + 1 + c] = z[(size_t)t * io + c] * m;
        if (m > 0.5f) ++out.n_ctx; else ++out.n_gen;
    }
    float wg, wc;
    if (mask_padding_attention) {
        wg = 1.0f / ((float)io * (float)out.n_gen + 1e-8f);
        wc = 0.0f;
    } else {
        const float denom = (float)io * (float)out.n_gen +
                            mask_loss_weight * (float)io * (float)out.n_ctx + 1e-8f;
        wg = 1.0f / denom;
        wc = mask_loss_weight / denom;
    }
    out.loss_weight.assign((size_t)io * frames, 0.0f);
    for (int t = 0; t < frames; ++t) {
        const float w = mask[(size_t)t] > 0.5f ? wc : wg;
        for (int c = 0; c < io; ++c) out.loss_weight[(size_t)t * io + c] = w;
    }
    return out;
}

} // namespace sa3
