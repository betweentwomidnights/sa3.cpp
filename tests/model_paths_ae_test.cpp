// model_paths_ae_test: the autoencoder resolves on its own axis from the DiT.
//
// SAME used to share `encoding` with the DiT, so --encoding q4_k_m took the autoencoder to
// Q4_K/Q6_K along with it, and neither sa3-server nor libsa3 had any field that could say
// otherwise. That is the one net you least want quantized -- it is the last thing the audio
// crosses, and continuation/transform push the signal through it twice per iteration.
//
// The behaviour pinned here:
//   - auto prefers F32, then F16, and never a quantized tier while either exists;
//   - an explicit request resolves EXACTLY, including a quantized tier (still supported, just
//     no longer a side effect of the DiT's);
//   - with nothing but a quantized SAME on disk, auto still resolves it rather than failing --
//     `download_models.py --encoding q4_k_m` produced exactly that set, so refusing would break
//     every existing install -- but it is no longer silent;
//   - the DiT's own tier is unaffected by any of it.
//
// resolve_one() matches on filenames only, so empty files are a faithful fixture.
#include "sa3_pipeline.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int fails = 0;

static void expect(bool ok, const std::string& msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg.c_str()); fails++; }
}

static void touch(const fs::path& p) { std::ofstream(p) << ""; }

// A models dir holding the DiT/conditioner/tokenizer/text-encoder every resolve needs, plus
// whichever SAME encodings the case under test wants.
static fs::path make_dir(const std::string& tag, const std::string& variant,
                         const std::string& dit_enc, const std::vector<std::string>& same_encs) {
    fs::path d = fs::temp_directory_path() / ("sa3_ae_test_" + tag);
    fs::remove_all(d);
    fs::create_directories(d);
    touch(d / "t5gemma-b-b-ul2-v1.0-vocab.gguf");
    touch(d / "t5gemma-b-b-ul2-encoder-0.3B-v1.0-F16.gguf");
    touch(d / ("stable-audio-3-" + variant + "-conditioner-v1.0-F32.gguf"));
    touch(d / ("stable-audio-3-" + variant + "-dit-1.5B-v1.0-" + dit_enc + ".gguf"));
    for (const auto& e : same_encs)
        touch(d / ("stable-audio-3-" + variant + "-same-l-v1.0-" + e + ".gguf"));
    return d;
}

static std::string same_enc_of(const std::string& path) {
    const std::string stem = fs::path(path).stem().string();
    const size_t dash = stem.rfind('-');
    return dash == std::string::npos ? "" : stem.substr(dash + 1);
}

static void case_(const std::string& tag, const std::string& dit_enc,
                  const std::vector<std::string>& same_encs, const std::string& ae_encoding,
                  const std::string& want_same, const std::string& want_dit) {
    fs::path d = make_dir(tag, "medium", dit_enc, same_encs);
    sa3::ModelPaths mp;
    std::string err;
    const bool ok = sa3::ModelPaths::resolve(d.string(), "medium", dit_enc, "F16", ae_encoding,
                                             mp, err);
    if (want_same.empty()) {                       // expected to fail
        expect(!ok, tag + ": expected resolve to fail, got " + mp.same);
        std::printf("  %-28s -> refused (%s)\n", tag.c_str(), err.c_str());
        fs::remove_all(d);
        return;
    }
    if (!ok) {
        expect(false, tag + ": resolve failed: " + err);
        fs::remove_all(d);
        return;
    }
    const std::string got_same = same_enc_of(mp.same), got_dit = same_enc_of(mp.dit);
    expect(got_same == want_same, tag + ": SAME resolved " + got_same + ", wanted " + want_same);
    expect(got_dit == want_dit, tag + ": DiT resolved " + got_dit + ", wanted " + want_dit);
    std::printf("  %-28s -> DiT %-7s SAME %s\n", tag.c_str(), got_dit.c_str(), got_same.c_str());
    fs::remove_all(d);
}

int main() {
    std::printf("model_paths_ae_test\n");

    // The regression this whole axis exists for: a quantized DiT must not drag SAME down with it.
    case_("q4 dit, all SAME present", "Q4_K_M", {"F32", "F16", "Q4_K_M"}, "", "F32", "Q4_K_M");

    // Auto order: F32 first, F16 when F32 is absent, quantized only as a last resort.
    case_("auto prefers F32", "F16", {"F32", "F16"}, "", "F32", "F16");
    case_("auto falls to F16", "F16", {"F16", "Q4_K_M"}, "", "F16", "F16");
    case_("auto last-resorts to q4", "Q4_K_M", {"Q4_K_M"}, "", "Q4_K_M", "Q4_K_M");

    // Explicit still wins, quantized included -- the point is that it must be ASKED for.
    case_("explicit q4 SAME", "Q4_K_M", {"F32", "F16", "Q4_K_M"}, "q4_k_m", "Q4_K_M", "Q4_K_M");
    case_("explicit f16 SAME", "F32", {"F32", "F16"}, "f16", "F16", "F32");

    // An explicit request for something absent is an error, not a silent substitution.
    case_("explicit missing tier", "F16", {"F16"}, "q8_0", "", "");
    case_("explicit bogus tier", "F16", {"F16"}, "nonsense", "", "");

    if (fails) { std::fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    std::printf("OK\n");
    return 0;
}
