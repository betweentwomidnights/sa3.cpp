// sa3-quant-check: per-weight dequant validation for quantized SA3 models.
// Loads an F16 (or any reference) model and a quantized model, and for every
// weight tensor that differs in type, dequantizes the quantized one on CPU and
// compares cosine similarity to the reference. Outlier weights (low cos) reveal
// which tensor the quantizer corrupted.
//
// Usage: sa3-quant-check --ref <f16.gguf> --quant <q4km.gguf> [--threshold 0.99]
#include "gguf_model.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <inttypes.h>
#include <string>
#include <vector>

// Read a tensor's raw bytes from its backend buffer.
static std::vector<char> read_raw(ggml_tensor* t) {
    int64_t nbytes = ggml_nbytes(t);
    std::vector<char> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    return raw;
}

// Dequantize a tensor (any supported type) into a float buffer (n logical elems).
static void tensor_to_f32(ggml_tensor* t, float* dst, int64_t n) {
    std::vector<char> raw = read_raw(t);
    switch ((int)t->type) {
        case GGML_TYPE_F32:   memcpy(dst, raw.data(), n * sizeof(float)); break;
        case GGML_TYPE_F16: {
            const ggml_fp16_t* h = (const ggml_fp16_t*)raw.data();
            for (int64_t i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(h[i]);
            break;
        }
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K*)raw.data(), dst, n); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K*)raw.data(), dst, n); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K*)raw.data(), dst, n); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0*)raw.data(), dst, n); break;
        default:
            fprintf(stderr, "tensor_to_f32: unsupported type %d (%s)\n", (int)t->type, ggml_type_name(t->type));
            exit(1);
    }
}

static double cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    double d = sqrt(na * nb);
    return d > 0 ? dot / d : 0;
}

static bool is_quant(int type) {
    return ggml_is_quantized((ggml_type)type);
}

int main(int argc, char** argv) {
    const char* ref_path = nullptr;
    const char* quant_path = nullptr;
    double threshold = 0.99;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref") && i + 1 < argc)   ref_path = argv[++i];
        else if (!strcmp(argv[i], "--quant") && i + 1 < argc) quant_path = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) threshold = atof(argv[++i]);
        else { fprintf(stderr, "usage: sa3-quant-check --ref <f16.gguf> --quant <q4km.gguf> [--threshold 0.99]\n"); return 1; }
    }
    if (!ref_path || !quant_path) { fprintf(stderr, "error: --ref and --quant required\n"); return 1; }

    auto R = sa3::load_gguf(ref_path, ggml_backend_cpu_init());
    auto Q = sa3::load_gguf(quant_path, ggml_backend_cpu_init());

    fprintf(stdout, "ref tensors=%zu quant tensors=%zu\n", R.tensors.size(), Q.tensors.size());

    int n_compared = 0, n_bad = 0;
    for (auto& kv : Q.tensors) {
        const std::string& name = kv.first;
        ggml_tensor* qt = kv.second;
        auto rit = R.tensors.find(name);
        if (rit == R.tensors.end()) { fprintf(stdout, "  [absent-in-ref] %s\n", name.c_str()); continue; }
        ggml_tensor* rt = rit->second;

        if (!is_quant((int)qt->type)) continue;          // only check quantized weights
        if ((int)qt->type == (int)rt->type) continue;     // unchanged type, skip

        int64_t n = ggml_nelements(qt);
        if (n != (int64_t)ggml_nelements(rt)) {
            fprintf(stdout, "  [size-mismatch] %s qtype=%d rtype=%d\n", name.c_str(), (int)qt->type, (int)rt->type);
            continue;
        }

        std::vector<float> qv(n), rv(n);
        tensor_to_f32(qt, qv.data(), n);
        tensor_to_f32(rt, rv.data(), n);
        double c = cosine(qv.data(), rv.data(), n);
        n_compared++;
        if (c < threshold) {
            n_bad++;
            fprintf(stdout, "  BAD  cos=%.5f  %s  qtype=%d  ne=[%" PRId64 ",%" PRId64 "]  n=%lld\n",
                    c, name.c_str(), (int)qt->type, qt->ne[0], qt->ne[1], (long long)n);
        }
    }

    fprintf(stdout, "compared=%d  below-threshold=%d  threshold=%.3f\n", n_compared, n_bad, threshold);
    return n_bad ? 2 : 0;
}
