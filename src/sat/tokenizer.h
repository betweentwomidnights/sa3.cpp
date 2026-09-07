// sat/tokenizer.h -- compact SentencePiece-Unigram tokenizer for classic T5.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sa3::sat {

struct UnigramTokenizer {
    struct Node {
        std::unordered_map<unsigned char, int> next;
        int token = -1;
    };
    std::vector<Node> trie = {Node{}};
    std::vector<float> scores;
    int pad_id = 0, eos_id = 1, unk_id = 2;
    bool add_eos = true;
    std::string metaspace = "\xE2\x96\x81";

    static UnigramTokenizer load(const char* path) {
        UnigramTokenizer t;
        ggml_context* mctx = nullptr;
        gguf_init_params gp = {true, &mctx};
        gguf_context* raw = gguf_init_from_file(path, gp);
        auto free_ctx = [](ggml_context* p) { if (p) ggml_free(p); };
        std::unique_ptr<ggml_context, decltype(free_ctx)> cg(mctx, free_ctx);
        if (!raw) throw std::runtime_error("[sat tokenizer] failed to open " + std::string(path));
        std::unique_ptr<gguf_context, decltype(&gguf_free)> g(raw, gguf_free);
        auto key = [&](const char* name) {
            int i = gguf_find_key(g.get(), name);
            if (i < 0) throw std::runtime_error("[sat tokenizer] missing key " + std::string(name));
            return i;
        };
        if (std::string(gguf_get_val_str(g.get(), key("tok.model"))) != "sentencepiece-unigram")
            throw std::runtime_error("[sat tokenizer] GGUF does not contain a Unigram tokenizer");
        const int kt = key("tok.tokens"), ks = key("tok.scores");
        const size_t n = gguf_get_arr_n(g.get(), kt);
        if (gguf_get_arr_n(g.get(), ks) != n || gguf_get_arr_type(g.get(), ks) != GGUF_TYPE_FLOAT32)
            throw std::runtime_error("[sat tokenizer] invalid token scores");
        const float* sp = static_cast<const float*>(gguf_get_arr_data(g.get(), ks));
        t.scores.assign(sp, sp + n);
        for (size_t id = 0; id < n; ++id) {
            const std::string piece = gguf_get_arr_str(g.get(), kt, id);
            int node = 0;
            for (unsigned char byte : piece) {
                const int new_index = (int)t.trie.size();
                auto [it, inserted] = t.trie[(size_t)node].next.emplace(byte, new_index);
                const int next_node = it->second;
                if (inserted) t.trie.push_back(Node{});
                node = next_node;
            }
            t.trie[(size_t)node].token = (int)id;
        }
        t.pad_id = (int)gguf_get_val_u32(g.get(), key("tok.pad_id"));
        t.eos_id = (int)gguf_get_val_u32(g.get(), key("tok.eos_id"));
        t.unk_id = (int)gguf_get_val_u32(g.get(), key("tok.unk_id"));
        t.add_eos = gguf_get_val_bool(g.get(), key("tok.add_eos"));
        t.metaspace = gguf_get_val_str(g.get(), key("tok.metaspace"));
        return t;
    }

    std::vector<int32_t> encode_piece(const std::string& input) const {
        const size_t n = input.size();
        const float neg_inf = -INFINITY;
        std::vector<float> best(n + 1, neg_inf);
        std::vector<int> previous(n + 1, -1), token(n + 1, -1);
        best[0] = 0.0f;
        for (size_t start = 0; start < n; ++start) {
            if (!std::isfinite(best[start])) continue;
            int node = 0;
            for (size_t end = start; end < n; ++end) {
                auto it = trie[(size_t)node].next.find((unsigned char)input[end]);
                if (it == trie[(size_t)node].next.end()) break;
                node = it->second;
                const int id = trie[(size_t)node].token;
                if (id >= 0 && id != unk_id) {
                    const float candidate = best[start] + scores[(size_t)id];
                    if (candidate > best[end + 1]) {
                        best[end + 1] = candidate;
                        previous[end + 1] = (int)start;
                        token[end + 1] = id;
                    }
                }
            }
            // SentencePiece emits one unknown token for an unsegmentable Unicode scalar.
            size_t next = start + 1;
            const unsigned char lead = (unsigned char)input[start];
            if ((lead & 0xe0) == 0xc0) next = std::min(n, start + 2);
            else if ((lead & 0xf0) == 0xe0) next = std::min(n, start + 3);
            else if ((lead & 0xf8) == 0xf0) next = std::min(n, start + 4);
            const float unknown = best[start] - 100.0f;
            if (unknown > best[next]) {
                best[next] = unknown; previous[next] = (int)start; token[next] = unk_id;
            }
        }
        if (previous[n] < 0 && n) throw std::runtime_error("[sat tokenizer] segmentation failed");
        std::vector<int32_t> out;
        for (int at = (int)n; at > 0; at = previous[(size_t)at])
            out.push_back(token[(size_t)at]);
        std::reverse(out.begin(), out.end());
        // SentencePiece coalesces adjacent unknown characters.
        out.erase(std::unique(out.begin(), out.end(), [&](int32_t a, int32_t b) {
            return a == unk_id && b == unk_id;
        }), out.end());
        return out;
    }

    std::vector<int32_t> encode(const std::string& text, int max_length) const {
        std::vector<int32_t> ids;
        // This reproduces T5's WhitespaceSplit + Metaspace pre-tokenizer for UTF-8
        // prompts. The tokenizer's precompiled Unicode normalization is handled by
        // upstream clients for now; ordinary ASCII/UTF-8 prompt text is unchanged.
        for (size_t i = 0; i < text.size();) {
            while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
            if (i == text.size()) break;
            size_t end = i;
            while (end < text.size() && !std::isspace((unsigned char)text[end])) ++end;
            std::vector<int32_t> word = encode_piece(metaspace + text.substr(i, end - i));
            ids.insert(ids.end(), word.begin(), word.end());
            i = end;
        }
        if (add_eos) ids.push_back(eos_id);
        if (max_length > 0 && (int)ids.size() > max_length) {
            ids.resize((size_t)max_length);
            if (add_eos) ids.back() = eos_id;
        }
        return ids;
    }
};

} // namespace sa3::sat
