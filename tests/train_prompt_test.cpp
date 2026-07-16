#include "train_prompt.h"

#include <cstdio>
#include <fstream>
#include <random>
#include <string>

static int expect(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); return 1; }
    return 0;
}

int main() {
    int fails = 0;

    // ── Tag builder: only the `prompt` tag present (the ratatat-2 reality). Regardless of shuffle,
    //    a single part comes back verbatim. ──
    {
        sa3::PromptConfig pc;
        pc.loaded = true; pc.use_tags = true; pc.use_paths = false;
        pc.tag_keys = {"prompt", "title", "artist", "genre", "bpm"};
        sa3::PromptMetadata md;
        md.tags["prompt"] = "Synthwave chiptune, 125 bpm, G minor";
        std::mt19937_64 rng(1);
        for (int i = 0; i < 50; ++i)
            fails += expect(sa3::prompt_build_tags(md, pc, rng) == md.tags["prompt"], "single tag verbatim");
    }

    // ── Tag builder: multiple tags, hide_tag_names=false => "Label: value" parts. ──
    {
        sa3::PromptConfig pc;
        pc.loaded = true; pc.shuffle = false;   // deterministic order
        pc.tag_keys = {"prompt", "artist", "bpm"};
        sa3::PromptMetadata md;
        md.tags["prompt"] = "warm pads";
        md.tags["artist"] = "Ratatat";
        md.tags["bpm"] = "125";
        std::mt19937_64 rng(2);
        fails += expect(sa3::prompt_build_tags(md, pc, rng) == "warm pads, Artist: Ratatat, BPM: 125",
                        "labelled multi-tag join");
    }

    // ── Path builder: bare filename relpath, default opts => filename verbatim; hide_ext strips it. ──
    {
        sa3::PromptConfig pc; pc.loaded = true;
        sa3::PromptMetadata md; md.relpath = "01 - Montanita [XrXqKoCPvE0].npy";
        fails += expect(sa3::prompt_build_path(md, pc) == "01 - Montanita [XrXqKoCPvE0].npy", "bare path verbatim");
        pc.path_hide_ext = true;
        fails += expect(sa3::prompt_build_path(md, pc) == "01 - Montanita [XrXqKoCPvE0]", "path hide_ext");
    }
    {
        // Nested relpath with hide options.
        sa3::PromptConfig pc; pc.loaded = true;
        sa3::PromptMetadata md; md.relpath = "genre/artist/song.wav";
        fails += expect(sa3::prompt_build_path(md, pc) == "genre/artist/song.wav", "nested path full");
        pc.path_hide_dirs = true;
        fails += expect(sa3::prompt_build_path(md, pc) == "genre/song.wav", "hide_dirs keeps topmost dir");
        pc.path_hide_topmost = true;
        fails += expect(sa3::prompt_build_path(md, pc) == "song.wav", "hide_dirs+topmost => filename");
    }

    // ── Composition distribution: the ratatat-2 config (tags 40 / paths 30, only prompt tag). Over
    //    many draws, ~57.1% caption vs ~42.9% filename. ──
    {
        sa3::PromptConfig pc;
        pc.loaded = true; pc.use_tags = true; pc.use_paths = true; pc.use_fixed = false;
        pc.balance_tags = 40; pc.balance_paths = 30; pc.balance_fixed = 60;   // fixed unused
        pc.tag_keys = {"prompt", "title", "artist", "genre", "bpm"};
        pc.shuffle = true;
        sa3::PromptMetadata md;
        md.tags["prompt"] = "funk, 88 bpm, G minor";
        md.relpath = "06 - Loud Pipes [BcoPKWzLjrE].npy";
        std::mt19937_64 rng(12345);
        int caption = 0, filename = 0, other = 0;
        const int N = 20000;
        for (int i = 0; i < N; ++i) {
            std::string p = sa3::prompt_compose(pc, md, rng);
            if (p == md.tags["prompt"]) ++caption;
            else if (p == md.relpath) ++filename;
            else ++other;
        }
        fails += expect(other == 0, "only caption or filename produced");
        const double frac_caption = (double)caption / N;   // expect ~0.5714
        fails += expect(frac_caption > 0.55 && frac_caption < 0.59, "caption fraction ~40/70");
    }

    // ── load_prompt_config from a dataset.json-shaped file (descends into prompt_config). ──
    {
        const char* path = "train_prompt_test_ds.json";
        { std::ofstream f(path);
          f << R"({"dataset_type":"pre_encoded","prompt_config":{)"
            << R"("use_tags":true,"use_paths":true,"use_fixed":false,)"
            << R"("balance":{"tags":40,"paths":30,"fixed":60},)"
            << R"("tag_keys":["prompt","title","artist","genre","bpm"],)"
            << R"("hide_tag_names":false,"shuffle":true}})"; }
        sa3::PromptConfig pc; std::string err;
        fails += expect(sa3::load_prompt_config(path, pc, err), "load dataset.json prompt_config");
        fails += expect(pc.loaded && pc.use_tags && pc.use_paths && !pc.use_fixed, "flags parsed");
        fails += expect(pc.balance_tags == 40 && pc.balance_paths == 30, "balance parsed");
        fails += expect(pc.tag_keys.size() == 5 && pc.tag_keys[0] == "prompt", "tag_keys parsed");
        std::remove(path);
    }

    if (fails) return 1;
    std::printf("train_prompt_test: ok\n");
    return 0;
}
