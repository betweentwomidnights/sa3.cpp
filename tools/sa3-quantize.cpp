#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>

static bool use_q6_k(const char * name) {
    // V projection (quality-critical attention component)
    if (strstr(name, ".v.weight"))   return true;   // T5Gemma: te.{N}.v.weight (not conv.weight)
    if (strstr(name, "kv.weight"))   return true;   // DiT: dit.{N}.cross.kv.weight
    if (strstr(name, "to_kv.weight")) return true;  // DiT cross-attn KV
    if (strstr(name, "to_qkv.weight")) return true; // AE fused QKV

    // Down projection (FFN second layer)
    if (strstr(name, "ff.out.weight")) return true;  // DiT/AE: dit.{N}.ff.out, ae.*.ff.out
    if (strstr(name, "down.weight"))  return true;   // T5Gemma: te.{N}.down.weight

    // Token embedding (top-level only, not cond_embed/time_embed/global_embed)
    if (strstr(name, "embed.weight")) return true;   // T5Gemma: te.embed.weight

    return false;
}

static enum ggml_type target_type(const char * name) {
    return use_q6_k(name) ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
}

static bool should_quantize(const char * name, int n_dims, const int64_t * ne) {
    if (n_dims != 2) return false;
    if (strstr(name, "tokens"))     return false;
    if (strstr(name, "running_std")) return false;
    // The row length must be a whole number of blocks of the TARGET type. Q4_K/Q6_K are
    // k-quants with QK_K=256 super-blocks, not 32: a `% 32` test lets ne[0] of 32/64/128
    // through and ggml_quantize_chunk then writes past the allocated rows. A rank-32 LoRA
    // (ne[0]==32) is the case that actually shows up in practice.
    if (ne[0] % ggml_blck_size(target_type(name)) != 0) return false;
    return true;
}

// Size of `path` in bytes, or 0 on error. NOT ftell(): `long` is 32-bit on MSVC, so ftell
// overflows on any gguf >= 2 GiB (the medium DiT is 2.7 GiB at F16) and the bogus length
// then feeds a bad allocation. std::filesystem is 64-bit clean on every target.
static size_t file_size(const char * path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    if (ec) {
        fprintf(stderr, "error: cannot stat %s (%s)\n", path, ec.message().c_str());
        return 0;
    }
    return (size_t)sz;
}

static void tensor_meta_init(struct ggml_tensor * t, enum ggml_type type, int n_dims, const int64_t * ne, const char * name) {
    memset(t, 0, sizeof(*t));
    t->type = type;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        t->ne[d] = d < n_dims ? ne[d] : 1;
    }
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(type));
    for (int d = 2; d < GGML_MAX_DIMS; d++) {
        t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
    }
    strncpy(t->name, name, sizeof(t->name) - 1);
    t->name[sizeof(t->name) - 1] = '\0';
}

int main(int argc, char ** argv) {
    const char * in_path  = nullptr;
    const char * out_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) {
            in_path = argv[++i];
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        }
    }

    if (!in_path || !out_path) {
        fprintf(stderr, "Usage: sa3-quantize --in <input.gguf> --out <output.gguf>\n");
        fprintf(stderr, "Q4_K_M quantization: V/down/embed->Q6_K, rest->Q4_K. Skips sa3-conditioner/sa3-tokenizer.\n");
        return 1;
    }

    // ── Open input ────
    struct gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };  // C++17: no designated initializers
    struct gguf_context * in_ctx = gguf_init_from_file(in_path, params);
    if (!in_ctx) { fprintf(stderr, "error: cannot read %s\n", in_path); return 1; }

    const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
    const size_t  data_off_in = gguf_get_data_offset(in_ctx);

    // Reject sidecar models that must stay F32
    {
        int arch_key = gguf_find_key(in_ctx, "general.architecture");
        if (arch_key >= 0) {
            const char * arch = gguf_get_val_str(in_ctx, arch_key);
            if (!strcmp(arch, "sa3-conditioner") || !strcmp(arch, "sa3-tokenizer")) {
                fprintf(stderr, "error: '%s' architecture must stay F32, refusing to quantize\n", arch);
                return 1;
            }
        }
    }

    // Map entire input file into memory
    const size_t fsize = file_size(in_path);
    if (fsize == 0) return 1;
    FILE * f = fopen(in_path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open %s\n", in_path); return 1; }
    std::vector<char> file_data;
    try {
        file_data.resize(fsize);
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "error: cannot allocate %zu MiB to read %s\n", fsize / 1048576, in_path);
        fclose(f); return 1;
    }
    // fread is size_t-based, but read in chunks so a >4 GiB gguf can't trip any 32-bit
    // limit inside the CRT and so a short read is reported at the offset where it happened.
    size_t got = 0;
    while (got < fsize) {
        const size_t want = fsize - got < (64u << 20) ? fsize - got : (64u << 20);
        const size_t n = fread(file_data.data() + got, 1, want, f);
        if (n == 0) break;
        got += n;
    }
    if (got != fsize) {
        fprintf(stderr, "error: short read on %s (%zu of %zu bytes)\n", in_path, got, fsize);
        fclose(f); return 1;
    }
    fclose(f);

    // Pre-init quantization tables
    ggml_quantize_init(GGML_TYPE_Q4_K);
    ggml_quantize_init(GGML_TYPE_Q6_K);

    // ── Build output context (metadata only, no data yet) ────
    struct gguf_context * out_ctx = gguf_init_empty();
    if (!out_ctx) { fprintf(stderr, "error: gguf_init_empty\n"); return 1; }
    gguf_set_kv(out_ctx, in_ctx);

    // First pass: add all tensor metadata so offsets are computed
    // Store which tensors get quantized and their target types
    struct TensorPlan {
        enum ggml_type stype;
        enum ggml_type dtype; // = stype if skip
        int64_t ne0;
        int64_t ne1;
    };
    std::vector<TensorPlan> plan(n_tensors);
    int n_q4k = 0, n_q6k = 0, n_skip = 0;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name  = gguf_get_tensor_name(in_ctx, i);
        int          ndims = gguf_get_tensor_ndims(in_ctx, i);
        const int64_t * ne  = gguf_get_tensor_ne(in_ctx, i);
        enum ggml_type stype = gguf_get_tensor_type(in_ctx, i);

        plan[i].stype = stype;
        plan[i].ne0   = ne[0];
        plan[i].ne1   = ndims >= 2 ? ne[1] : 1;

        if (!should_quantize(name, ndims, ne)) {
            plan[i].dtype = stype;
            n_skip++;

            struct ggml_tensor t;
            tensor_meta_init(&t, stype, ndims, ne, name);
            gguf_add_tensor(out_ctx, &t);
        } else {
            enum ggml_type qtype = target_type(name);
            plan[i].dtype = qtype;
            if (qtype == GGML_TYPE_Q4_K) n_q4k++; else n_q6k++;

            struct ggml_tensor t;
            tensor_meta_init(&t, qtype, ndims, ne, name);
            gguf_add_tensor(out_ctx, &t);
        }
    }

    // ── Allocate single persistent data buffer ────
    const size_t data_off_out = gguf_get_data_offset(out_ctx);
    size_t total_data_size = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        size_t tend = gguf_get_tensor_offset(out_ctx, i) + gguf_get_tensor_size(out_ctx, i);
        if (tend > total_data_size) total_data_size = tend;
    }
    std::vector<char> data_buf(total_data_size);

    // ── Second pass: quantize/copy each tensor into data_buf ────
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name   = gguf_get_tensor_name(in_ctx, i);
        fprintf(stderr, "  [%3lld/%lld] %-40s  ", (long long)i, (long long)n_tensors - 1, name);
        fflush(stderr);

        enum ggml_type stype = plan[i].stype;
        enum ggml_type dtype = plan[i].dtype;
        int64_t ne0 = plan[i].ne0;
        int64_t ne1 = plan[i].ne1;

        size_t in_off  = data_off_in + gguf_get_tensor_offset(in_ctx, i);
        size_t out_off = gguf_get_tensor_offset(out_ctx, i);
        size_t src_sz  = gguf_get_tensor_size(in_ctx, i);

        if (dtype == stype) {
            memcpy(data_buf.data() + out_off, file_data.data() + in_off, src_sz);
            fprintf(stderr, "skip (%s)\n", ggml_type_name(stype));
            continue;
        }

        int64_t n_elems = ne0 * ne1;
        std::vector<float> f32_buf(n_elems);

        const char * src_ptr = file_data.data() + in_off;
        if (stype == GGML_TYPE_F32) {
            memcpy(f32_buf.data(), src_ptr, src_sz);
        } else if (stype == GGML_TYPE_F16) {
            const ggml_fp16_t * f16 = (const ggml_fp16_t *)src_ptr;
            for (int64_t j = 0; j < n_elems; j++) {
                f32_buf[j] = ggml_fp16_to_fp32(f16[j]);
            }
        } else {
            fprintf(stderr, "ERROR: unsupported type %s\n", ggml_type_name(stype));
            return 1;
        }

        char * dst = data_buf.data() + out_off;
        ggml_quantize_chunk(dtype, f32_buf.data(), dst, 0, ne1, ne0, nullptr);

        fprintf(stderr, "%s  [%lld x %lld]\n",
                ggml_type_name(dtype), (long long)ne0, (long long)ne1);
    }

    // ── Set tensor data pointers and write ────
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(out_ctx, i);
        size_t out_off = gguf_get_tensor_offset(out_ctx, i);
        gguf_set_tensor_data(out_ctx, name, data_buf.data() + out_off);
    }

    if (!gguf_write_to_file(out_ctx, out_path, false)) {
        fprintf(stderr, "error: failed to write %s\n", out_path);
        return 1;
    }

    ggml_quantize_free();
    gguf_free(in_ctx);
    gguf_free(out_ctx);

    fprintf(stderr, "\nDone: wrote %s  (%d Q4_K, %d Q6_K, %d skipped, data %zu MiB)\n",
            out_path, n_q4k, n_q6k, n_skip, total_data_size / 1048576);
    return 0;
}
