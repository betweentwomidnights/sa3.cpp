// lora_quant_merge_test: merging an adapter into a QUANTIZED base.
//
// The autoencoder has no functional (unmerged) path — same_decode folds W straight into
// mul_mat and nothing hands it adapter tensors — so a q4/q5/q8 AE can only carry a decoder or
// encoder adapter if write_from_f32 can re-pack W_eff into the base's own quantized type.
//
// Two things are worth pinning, and they are different claims:
//
//   1. It does not CORRUPT. The pre-existing else-branch wrote n*sizeof(float) into whatever
//      buffer it was handed; for Q4_K (~0.56 bytes/element) that is a ~7x overrun reported as a
//      successful merge. A type that cannot be packed must raise instead.
//
//   2. It is ACCURATE ENOUGH. W_eff gets quantized a second time here (the base was already
//      quantized once), so the merged result cannot equal the f32 reference. What it must do is
//      land within the error the base's own quantization already carries — i.e. re-quantizing
//      is not materially worse than the quantization that shipped. That is measured against a
//      control: quantize the UNADAPTED W and compare, which is the error floor of the format.
#include "lora.h"
#include "test_backend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int fails = 0;

static void expect(bool ok, const std::string& msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg.c_str()); fails++; }
}

// k-quants work on blocks of 256, so every row length here is a multiple of it — as every
// weight in the real AE is (256 / 1536 / 4608).
static const int64_t kIn = 512, kOut = 64, kRank = 8;

static ggml_backend_t backend() { return sa3_test_cpu_backend(); }

// One tensor of `type`, holding the quantized form of `w` (row-major [out][in]).
static void build_base(sa3::GgufModel& m, const std::string& name, ggml_type type,
                       const std::vector<float>& w) {
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    m.ctx = ggml_init(ip);
    m.backend = backend();
    m.owns_backend = false;
    ggml_tensor* t = ggml_new_tensor_2d(m.ctx, type, kIn, kOut);
    ggml_set_name(t, name.c_str());
    m.tensors[name] = t;
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.backend);
    sa3::write_from_f32(t, w);          // the quantize-on-write path, used here to seed the base
}

// An adapter gguf-substitute: lora_A [rank][in], lora_B [out][rank], both f32.
static void build_adapter(sa3::GgufModel& m, const std::string& stem,
                          const std::vector<float>& A, const std::vector<float>& B) {
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    m.ctx = ggml_init(ip);
    m.backend = backend();
    m.owns_backend = false;
    ggml_tensor* a = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, kIn, kRank);
    ggml_tensor* b = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, kRank, kOut);
    ggml_set_name(a, (stem + ".lora_A").c_str());
    ggml_set_name(b, (stem + ".lora_B").c_str());
    m.tensors[stem + ".lora_A"] = a;
    m.tensors[stem + ".lora_B"] = b;
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.backend);
    ggml_backend_tensor_set(a, A.data(), 0, A.size() * sizeof(float));
    ggml_backend_tensor_set(b, B.data(), 0, B.size() * sizeof(float));
}

static double rms(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) { const double d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s / (double)a.size());
}

static void run_type(ggml_type type) {
    const char* tname = ggml_type_name(type);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 0.05f);

    std::vector<float> W((size_t)kIn * kOut), A((size_t)kRank * kIn), B((size_t)kOut * kRank);
    for (auto& v : W) v = nd(rng);
    for (auto& v : A) v = nd(rng);
    for (auto& v : B) v = nd(rng);

    const float alpha = (float)kRank, strength = 1.0f;   // scaling = alpha/rank = 1
    const float sc = (alpha / (float)kRank) * strength;

    // Reference: the merge done in f32, no quantization anywhere.
    std::vector<float> W_ref((size_t)kIn * kOut);
    for (int64_t o = 0; o < kOut; o++)
        for (int64_t i = 0; i < kIn; i++) {
            double d = 0;
            for (int64_t r = 0; r < kRank; r++) d += (double)B[o * kRank + r] * A[r * kIn + i];
            W_ref[o * kIn + i] = W[o * kIn + i] + sc * (float)d;
        }

    // Control: what the FORMAT costs on its own. Quantize the adapted reference directly, with
    // no merge involved. Any merged result should be about this far from W_ref, not further.
    sa3::GgufModel ctrl;
    build_base(ctrl, "w", type, W_ref);
    std::vector<float> ctrl_out;
    sa3::read_to_f32(ctrl.tensors["w"], ctrl_out);
    const double floor_err = rms(ctrl_out, W_ref);
    ctrl.free();

    // Under test: quantized base + adapter, merged in place through apply_loras.
    sa3::GgufModel base;
    build_base(base, "w.weight", type, W);

    // GgufModel is move-only (it owns a ggml context + backend buffer), so the adapter is built
    // straight into the LoraAdapter that owns it rather than assigned in.
    std::vector<sa3::LoraAdapter> stack(1);
    build_adapter(stack[0].gguf, "w", A, B);
    stack[0].type = "lora";
    stack[0].rank = (int)kRank;
    stack[0].alpha = alpha;
    stack[0].strength = strength;

    bool threw = false;
    try {
        sa3::apply_loras(base, stack);
    } catch (const std::exception& e) {
        threw = true;
        std::fprintf(stderr, "  %s: apply_loras threw: %s\n", tname, e.what());
    }
    expect(!threw, std::string(tname) + ": merging into a quantized base must not throw");

    std::vector<float> got;
    sa3::read_to_f32(base.tensors["w.weight"], got);
    const double err = rms(got, W_ref);

    // Measured ratio is ~1.41x = sqrt(2) for every type here (q4_K, q6_K, q8_0, f16): two
    // independent roundings adding in quadrature, which is the theoretical floor for merging
    // into a quantized base rather than any interaction between them. The bound is kept at 3x
    // so it pins the ORDER and not the exact constant — a corruption or a wrong code path
    // lands orders of magnitude out, not at 1.5x.
    expect(err < floor_err * 3.0 + 1e-9,
           std::string(tname) + ": merged error " + std::to_string(err) +
           " should stay near the format floor " + std::to_string(floor_err));
    std::printf("  %-8s merged rms err %.3e   format floor %.3e   ratio %.2fx\n",
                tname, err, floor_err, floor_err > 0 ? err / floor_err : 0.0);

    base.free();
    stack[0].gguf.free();
}

// A type write_from_f32 has no route for must raise, not write short.
static void run_unsupported() {
    sa3::GgufModel m;
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    m.ctx = ggml_init(ip);
    m.backend = backend();
    m.owns_backend = false;
    ggml_tensor* t = ggml_new_tensor_2d(m.ctx, GGML_TYPE_I32, kIn, kOut);
    ggml_set_name(t, "w.weight");
    m.tensors["w.weight"] = t;
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.backend);

    std::vector<float> w((size_t)kIn * kOut, 0.5f);
    bool threw = false;
    try { sa3::write_from_f32(t, w); } catch (const std::exception&) { threw = true; }
    expect(threw, "an unwritable tensor type must throw rather than write past the tensor");
    m.free();
}

// An adapter whose factors are the wrong width for the base must be refused.
//
// This is the SAME-L-adapter-on-a-SAME-S-base case, which is not hypothetical: the two share a
// tensor namespace, so 25 of SAME-L's 49 decoder stems exist on SAME-S and 24 of those are
// half the width. Nothing overruns -- the apply loops size themselves from the base and read
// the larger adapter at the wrong stride -- so without a check it merges garbage and reports
// success.
static void run_shape_mismatch() {
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 0.05f);

    // Base is HALF the width the adapter was built for, as SAME-S is to SAME-L.
    const int64_t narrow_in = kIn / 2, narrow_out = kOut / 2;
    std::vector<float> W((size_t)narrow_in * narrow_out);
    for (auto& v : W) v = nd(rng);

    sa3::GgufModel base;
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    base.ctx = ggml_init(ip);
    base.backend = backend();
    base.owns_backend = false;
    ggml_tensor* t = ggml_new_tensor_2d(base.ctx, GGML_TYPE_F32, narrow_in, narrow_out);
    ggml_set_name(t, "w.weight");
    base.tensors["w.weight"] = t;
    base.buf = ggml_backend_alloc_ctx_tensors(base.ctx, base.backend);
    ggml_backend_tensor_set(t, W.data(), 0, W.size() * sizeof(float));

    std::vector<float> A((size_t)kRank * kIn), B((size_t)kOut * kRank);
    for (auto& v : A) v = nd(rng);
    for (auto& v : B) v = nd(rng);
    std::vector<sa3::LoraAdapter> stack(1);
    build_adapter(stack[0].gguf, "w", A, B);        // built for the WIDE base
    stack[0].type = "lora";
    stack[0].rank = (int)kRank;
    stack[0].alpha = (float)kRank;
    stack[0].strength = 1.0f;

    bool threw = false;
    try { sa3::apply_loras(base, stack); } catch (const std::exception&) { threw = true; }
    expect(threw, "an adapter sized for a different variant must be refused, not merged");

    // And the base must be untouched by the refusal.
    std::vector<float> after;
    sa3::read_to_f32(base.tensors["w.weight"], after);
    expect(rms(after, W) == 0.0, "a refused merge must leave the base weights alone");

    base.free();
    stack[0].gguf.free();
}

int main() {
    std::printf("lora_quant_merge_test\n");
    for (ggml_type t : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F16 })
        run_type(t);
    run_unsupported();
    run_shape_mismatch();

    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("OK\n");
    return 0;
}
