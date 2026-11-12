#include "lora_filter.h"
#include <cstdio>
#include <functional>
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* message) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", message); ++failures; } };
    auto matches = [](const std::string& stem, const std::string& filter) {
        return sa3::lora_layer_excluded(stem, sa3::compile_lora_filter(filter));
    };
    check(!matches("dit.0.self.qkv", " \t\n"), "empty filter preserves every layer");
    check(matches("dit.0.self.qkv", "  SELF_ATTN  "), "original module name and case/whitespace");
    check(matches("dit.0.self.qkv", "transformer.layers."), "original parent namespace");
    check(!matches("dit.0.self.qkv", ".transformer.layers."), "upstream docstring example: no module name starts with a dot");
    check(matches("dit.2.cross.q", "layers.[0-3].cross_attn"), "inclusive ascending range");
    check(matches("dit.2.cross.q", "layers.[3-0].cross_attn"), "inclusive descending range");
    check(!matches("dit.4.cross.q", "layers.[0-3].cross_attn"), "range excludes only selected layers");
    check(matches("dit.global_embed.0", "to_global_embed., .self_attn."), "comma OR and direct alias");
    check(matches("dit.3.local.2", "layers.[0-3].to_local_embed.[0-2]"), "multiple bracket ranges");
    check(matches("dit.0.ff.proj", ".ff.ff.0.proj"), "feed-forward alias");
    check(matches("dit.0.ff.out", ".ff.ff.2"), "feed-forward output alias");
    check(matches("dit.proj_in", "transformer.project_in"), "project-in alias");
    check(matches("dit.time_embed.2", "to_timestep_embed"), "timestep alias");
    check(matches("ae.dec.0.ff.out", "ae.dec."), "native autoencoder name");
    check(matches("dit.0.self.qkv", "dit.0.self"), "native DiT name");
    check(!matches("dit.0.self.qkv", ",,,"), "empty comma tokens are ignored");
    check(matches("dit.0.self.qkv", ",  ,"), "Python whitespace-only comma token means empty substring");
    check(!matches("dit.0.self.qkv", "layers[0-3]"), "brackets don't invent a separating dot");
    check(!matches("dit.0.self.qkv", "model."), "names are relative to DiffusionTransformer, as upstream");
    check(matches("dit.pre_conv", "preprocess_conv"), "top-level DiT module alias");
    for (const std::string& invalid : {std::string(4097, 'x'), std::string("[0-9999999]"),
                                      std::string("[0-100][0-100]"), std::string("[18446744073709551616-0]")}) {
        bool refused = false;
        try { sa3::compile_lora_filter(invalid); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "bounded expansion rejects excessive inputs");
    }
    return failures ? 1 : 0;
}
