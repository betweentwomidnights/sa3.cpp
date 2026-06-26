// sa3-tokenize: encode a string with the Gemma BPE tokenizer and print the ids.
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    const char* tok_path = nullptr; std::string text;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--tok")  && i+1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i+1 < argc) text = argv[++i];
    }
    if (!tok_path) { fprintf(stderr, "usage: sa3-tokenize --tok <gguf> --text \"...\"\n"); return 1; }
    sa3::Tokenizer tok = sa3::Tokenizer::load(tok_path);
    std::vector<int32_t> ids = tok.encode(text);
    printf("[");
    for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");
    return 0;
}
