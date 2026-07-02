// tokenizer.h — Gemma byte-fallback BPE tokenizer (loads vocab+merges from GGUF).
// Matches the HF fast tokenizer: normalize space->U+2581, BPE-merge the whole
// normalized string by merge rank, byte-fallback for out-of-vocab symbols.
// No protobuf / sentencepiece at runtime.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sa3 {

inline std::runtime_error tokenizer_error(const std::string& message) {
    return std::runtime_error("[tok] " + message);
}

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;        // piece -> id
    std::unordered_map<std::string, int> merge_rank;   // "A B" -> rank
    int bos_id = 2, eos_id = 1, pad_id = 0, unk_id = 3;
    bool add_bos = false;

    static Tokenizer load(const char* path) {
        Tokenizer t;
        ggml_context* mctx = nullptr;
        gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/&mctx };
        gguf_context* g_raw = gguf_init_from_file(path, gp);
        auto free_ctx = [](ggml_context* ctx) { if (ctx) ggml_free(ctx); };
        std::unique_ptr<ggml_context, decltype(free_ctx)> ctx_guard(mctx, free_ctx);
        if (!g_raw) throw tokenizer_error("failed to open " + std::string(path));
        std::unique_ptr<gguf_context, decltype(&gguf_free)> g(g_raw, gguf_free);

        int kt = gguf_find_key(g.get(), "tok.tokens");
        int km = gguf_find_key(g.get(), "tok.merges");
        if (kt < 0 || km < 0) throw tokenizer_error("missing token/merge arrays");
        const size_t nt = gguf_get_arr_n(g.get(), kt);
        t.vocab.reserve(nt * 2);
        for (size_t i = 0; i < nt; i++) t.vocab.emplace(gguf_get_arr_str(g.get(), kt, i), (int)i);
        const size_t nm = gguf_get_arr_n(g.get(), km);
        t.merge_rank.reserve(nm * 2);
        for (size_t i = 0; i < nm; i++) t.merge_rank.emplace(gguf_get_arr_str(g.get(), km, i), (int)i);

        auto u32 = [&](const char* k, int def){ int i = gguf_find_key(g.get(), k); return i < 0 ? def : (int)gguf_get_val_u32(g.get(), i); };
        t.bos_id = u32("tok.bos_id", 2); t.eos_id = u32("tok.eos_id", 1);
        t.pad_id = u32("tok.pad_id", 0); t.unk_id = u32("tok.unk_id", 3);
        { int i = gguf_find_key(g.get(), "tok.add_bos"); if (i >= 0) t.add_bos = gguf_get_val_bool(g.get(), i); }
        return t;
    }

    // Encode text -> token ids (no padding; the pipeline pads to max_length).
    std::vector<int32_t> encode(const std::string& text) const {
        // 1. normalize: ascii space -> U+2581 (UTF-8 E2 96 81)
        std::string norm;
        norm.reserve(text.size() * 2);
        for (char ch : text) {
            if (ch == ' ') norm += "\xE2\x96\x81";
            else norm += ch;
        }
        // 2. split into UTF-8 characters (initial symbols)
        std::vector<std::string> sym;
        for (size_t i = 0; i < norm.size();) {
            size_t len = 1;
            unsigned char c = (unsigned char)norm[i];
            if      ((c & 0x80) == 0x00) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            sym.push_back(norm.substr(i, len));
            i += len;
        }
        // 3. greedy merge by rank
        while (sym.size() > 1) {
            int best_rank = INT_MAX, best_i = -1;
            for (size_t i = 0; i + 1 < sym.size(); i++) {
                auto it = merge_rank.find(sym[i] + " " + sym[i+1]);
                if (it != merge_rank.end() && it->second < best_rank) { best_rank = it->second; best_i = (int)i; }
            }
            if (best_i < 0) break;
            sym[best_i] += sym[best_i + 1];
            sym.erase(sym.begin() + best_i + 1);
        }
        // 4. map symbols -> ids, byte-fallback for the rest
        std::vector<int32_t> ids;
        if (add_bos) ids.push_back(bos_id);
        char buf[8];
        for (const std::string& s : sym) {
            auto it = vocab.find(s);
            if (it != vocab.end()) { ids.push_back(it->second); continue; }
            for (unsigned char b : s) {                         // byte fallback: <0xHH>
                snprintf(buf, sizeof(buf), "<0x%02X>", b);
                auto bit = vocab.find(buf);
                ids.push_back(bit != vocab.end() ? bit->second : unk_id);
            }
        }
        return ids;
    }
};

} // namespace sa3
