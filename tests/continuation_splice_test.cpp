// The splice arithmetic alone — no model, synthetic buffers. Proves the seam lands where the mask
// bounds say it does, and that the degenerate cases the reference handles are handled the same way.
#include "audio_post.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int expect(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); return 1; }
    return 0;
}

static bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }

// A DC buffer per channel makes every assertion below readable: the restored head is the source
// level, the untouched tail is the model level, and the crossfade is the two in known proportion.
static std::vector<float> dc(int n_samp, int n_ch, float value) {
    return std::vector<float>((size_t)n_samp * n_ch, value);
}

int main() {
    int fails = 0;
    const int sr = 44100;

    // ---- mask bounds: the pullback, and the floor under it ----
    {
        float ov = 0.0f, ms = 0.0f;
        sa3::continuation_mask_bounds(10.0f, 0.2f, ov, ms);
        fails += expect(near(ov, 0.2f) && near(ms, 9.8f), "10 s source, 0.2 s overlap -> mask at 9.8 s");

        sa3::continuation_mask_bounds(10.0f, 0.0f, ov, ms);
        fails += expect(near(ov, 0.0f) && near(ms, 10.0f), "zero overlap keeps the requested window");

        // A source shorter than the requested overlap still leaves 50 ms of real audio.
        sa3::continuation_mask_bounds(0.1f, 0.2f, ov, ms);
        fails += expect(near(ov, 0.05f) && near(ms, 0.05f), "short source clamps to 50 ms kept");

        sa3::continuation_mask_bounds(0.02f, 0.2f, ov, ms);
        fails += expect(near(ov, 0.0f) && near(ms, 0.02f), "source under 50 ms gets no pullback");

        sa3::continuation_mask_bounds(10.0f, -1.0f, ov, ms);
        fails += expect(near(ov, 0.0f) && near(ms, 10.0f), "negative overlap is clamped to 0");
    }

    // ---- the ordinary splice: head restored, tail untouched, equal-power seam between ----
    {
        const int n = sr, n_ch = 2;                     // 1 s of output
        std::vector<float> audio = dc(n, n_ch, 0.5f);
        const std::vector<float> src = dc(n, n_ch, 0.5f);   // same level: gain match is a no-op
        sa3::SpliceParams p; p.xfade = 0.03f;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.5f, sr, p, m);

        const int splice_end = sr / 2, xf = (int)std::llround(0.03 * sr);
        fails += expect(m.applied, "splice applied");
        fails += expect(near(m.splice_end_seconds, 0.5f), "splice_end reported in seconds");
        fails += expect(near(m.xfade_applied, (float)xf / sr), "xfade reported in seconds");
        fails += expect(near(m.gain, 1.0f), "matched levels -> unit gain");
        // Everything past the mask start is the model's, untouched.
        fails += expect(near(audio[splice_end + 10], 0.5f), "tail is the model's audio");
        fails += expect(near(audio[(size_t)n + splice_end + 10], 0.5f), "tail untouched on ch 1");
    }

    // ---- distinguishable levels: prove which samples came from where ----
    {
        const int n = 1000, n_ch = 1, rate = 1000;      // 1 s at 1 kHz, so seconds are samples/1000
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(n, n_ch, 1.0f);
        sa3::SpliceParams p; p.xfade = 0.100f; p.gain_match = false;   // 100 samples of crossfade
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.5f, rate, p, m);

        const int splice_end = 500, xf = 100, hard_end = splice_end - xf;
        // Equal-power: cos + sin over a quarter turn, both operands 1.0.
        for (int i = 0; i < xf; i++) {
            const double ph = 1.5707963267948966 * (double)i / (double)(xf - 1);
            const float want = (float)(std::cos(ph) + std::sin(ph));
            if (!near(audio[hard_end + i], want, 1e-4f)) {
                fails += expect(false, "crossfade follows cos/sin");
                break;
            }
        }
        fails += expect(near(audio[hard_end - 1], 1.0f), "hard region is pure source");
        fails += expect(near(audio[splice_end], 1.0f), "first model sample after the seam");
    }

    // ---- xfade longer than the window: clamped to half, never past the splice ----
    {
        const int n = 1000, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 0.5f);
        const std::vector<float> src = dc(n, n_ch, 0.5f);
        sa3::SpliceParams p; p.xfade = 10.0f; p.gain_match = false;   // absurd: 10 s into a 0.1 s head
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.1f, rate, p, m);
        fails += expect(m.applied && near(m.xfade_applied, 0.05f), "xfade clamps to half the window");
        fails += expect(near(audio[200], 0.5f), "nothing written past splice_end");
    }

    // ---- xfade == 1 sample: degenerate, so the model's sample stands ----
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(n, n_ch, 0.25f);
        sa3::SpliceParams p; p.xfade = 0.001f; p.gain_match = false;   // exactly one sample
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.05f, rate, p, m);
        fails += expect(near(m.xfade_applied, 0.001f), "one-sample xfade reported");
        fails += expect(near(audio[48], 0.25f), "hard region still source");
        fails += expect(near(audio[49], 1.0f), "the single xfade sample keeps the model's value");
    }

    // ---- splice_end == 0: mask start at the very front, nothing to restore ----
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(n, n_ch, 0.25f);
        sa3::SpliceParams p;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.0f, rate, p, m);
        fails += expect(!m.applied && near(audio[0], 1.0f), "mask start 0 -> untouched");
    }

    // ---- mono source into stereo output: broadcast, not silence on ch 1 ----
    {
        const int n = 100, n_ch = 2, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(n, 1, 0.25f);
        sa3::SpliceParams p; p.xfade = 0.0f; p.gain_match = false;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, 1, 0.05f, rate, p, m);
        fails += expect(near(audio[10], 0.25f), "mono source reaches ch 0");
        fails += expect(near(audio[(size_t)n + 10], 0.25f), "mono source repeats onto ch 1");
    }

    // ---- stereo source into mono output: downmixed ----
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        std::vector<float> src((size_t)n * 2, 0.0f);
        for (int s = 0; s < n; s++) { src[s] = 0.2f; src[(size_t)n + s] = 0.6f; }
        sa3::SpliceParams p; p.xfade = 0.0f; p.gain_match = false;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, 2, 0.05f, rate, p, m);
        fails += expect(near(audio[10], 0.4f), "stereo source downmixes to the mono mean");
    }

    // ---- a silent source: gain match must not divide by ~0 ----
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(n, n_ch, 0.0f);
        sa3::SpliceParams p; p.xfade = 0.0f;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.05f, rate, p, m);
        fails += expect(near(m.gain, 1.0f) && std::isfinite(audio[10]), "silent source -> unit gain");
        fails += expect(near(audio[10], 0.0f), "silence is still restored, just unscaled");
    }

    // ---- gain match: pulls a quiet head up to the model's level, and clamps ----
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 0.8f);
        const std::vector<float> src = dc(n, n_ch, 0.4f);
        sa3::SpliceParams p; p.xfade = 0.0f;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.05f, rate, p, m);
        fails += expect(near(m.gain, 2.0f), "gain = model_rms / source_rms");
        fails += expect(near(audio[10], 0.8f), "restored head sits at the model's level");
    }
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(n, n_ch, 0.001f);   // ratio 1000, way past the clamp
        sa3::SpliceParams p; p.xfade = 0.0f;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.05f, rate, p, m);
        fails += expect(near(m.gain, 4.0f), "gain clamps at 4.0");
    }
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 0.001f);
        const std::vector<float> src = dc(n, n_ch, 1.0f);
        sa3::SpliceParams p; p.xfade = 0.0f;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.05f, rate, p, m);
        fails += expect(near(m.gain, 0.25f), "gain clamps at 0.25");
    }

    // ---- a source shorter than the mask start bounds the splice ----
    {
        const int n = 1000, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(300, n_ch, 0.25f);
        sa3::SpliceParams p; p.xfade = 0.0f; p.gain_match = false;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), 300, 300, n_ch, 0.5f, rate, p, m);
        fails += expect(near(m.splice_end_seconds, 0.3f), "splice_end clamps to the source length");
        fails += expect(near(audio[299], 0.25f) && near(audio[300], 1.0f), "splice stops at the source end");
    }

    // ---- a padded source buffer: stride is not the valid length ----
    {
        const int n = 1000, n_ch = 2, rate = 1000;
        const int src_n = 400, stride = 512;               // the pipeline's generation buffer shape
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        std::vector<float> src((size_t)stride * n_ch, 0.0f);
        for (int c = 0; c < n_ch; c++)
            for (int s = 0; s < src_n; s++) src[(size_t)c * stride + s] = 0.25f;
        sa3::SpliceParams p; p.xfade = 0.0f; p.gain_match = false;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), src_n, stride, n_ch, 0.5f, rate, p, m);
        fails += expect(near(m.splice_end_seconds, 0.4f), "valid length bounds the splice, not the stride");
        fails += expect(near(audio[(size_t)n + 399], 0.25f), "ch 1 reads at the source stride");
        fails += expect(near(audio[(size_t)n + 400], 1.0f), "the zero padding is never pasted");
    }

    // ---- disabled: byte-for-byte untouched ----
    {
        const int n = 100, n_ch = 1, rate = 1000;
        std::vector<float> audio = dc(n, n_ch, 1.0f);
        const std::vector<float> src = dc(n, n_ch, 0.25f);
        sa3::SpliceParams p; p.enabled = false;
        sa3::SpliceMeta m;
        sa3::splice_continuation_source(audio, n, n_ch, src.data(), n, n, n_ch, 0.05f, rate, p, m);
        fails += expect(!m.applied && near(audio[0], 1.0f), "splice off -> untouched");
    }

    if (fails) return 1;
    std::printf("continuation_splice_test: ok\n");
    return 0;
}
