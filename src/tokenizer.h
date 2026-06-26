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
#include <string>
#include <unordered_map>
#include <vector>

namespace sa3 {

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;        // piece -> id
    std::unordered_map<std::string, int> merge_rank;   // "A B" -> rank
    int bos_id = 2, eos_id = 1, pad_id = 0, unk_id = 3;
    bool add_bos = false;

    static Tokenizer load(const char* path) {
        Tokenizer t;
        ggml_context* mctx = nullptr;
        gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/&mctx };
        gguf_context* g = gguf_init_from_file(path, gp);
        if (!g) { fprintf(stderr, "[tok] failed to open %s\n", path); exit(1); }

        int kt = gguf_find_key(g, "tok.tokens");
        int km = gguf_find_key(g, "tok.merges");
        if (kt < 0 || km < 0) { fprintf(stderr, "[tok] missing token/merge arrays\n"); exit(1); }
        const size_t nt = gguf_get_arr_n(g, kt);
        t.vocab.reserve(nt * 2);
        for (size_t i = 0; i < nt; i++) t.vocab.emplace(gguf_get_arr_str(g, kt, i), (int)i);
        const size_t nm = gguf_get_arr_n(g, km);
        t.merge_rank.reserve(nm * 2);
        for (size_t i = 0; i < nm; i++) t.merge_rank.emplace(gguf_get_arr_str(g, km, i), (int)i);

        auto u32 = [&](const char* k, int def){ int i = gguf_find_key(g, k); return i < 0 ? def : (int)gguf_get_val_u32(g, i); };
        t.bos_id = u32("tok.bos_id", 2); t.eos_id = u32("tok.eos_id", 1);
        t.pad_id = u32("tok.pad_id", 0); t.unk_id = u32("tok.unk_id", 3);
        { int i = gguf_find_key(g, "tok.add_bos"); if (i >= 0) t.add_bos = gguf_get_val_bool(g, i); }

        gguf_free(g);
        if (mctx) ggml_free(mctx);
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
