#include "train_dataset.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static int expect(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return 1;
    }
    return 0;
}

int main() {
    namespace fs = std::filesystem;
    int fails = 0;
    const fs::path root = fs::temp_directory_path() / "sa3_train_dataset_test";
    fs::remove_all(root);
    fs::create_directories(root / "train" / "audio");
    fs::create_directories(root / "train" / "lyrics");
    {
        std::ofstream(root / "train" / "audio" / "a.mp3") << "mp3";
        std::ofstream(root / "train" / "audio" / "a.txt") << "caption";
        std::ofstream(root / "train" / "audio" / "b.mp3") << "mp3";
        std::ofstream(root / "train" / "audio" / "b.txt") << "caption";
    }
    {
        std::ofstream f(root / "train" / "filelist.txt");
        f << "audio/a.mp3\n\n audio/b.mp3 \n";
    }
    {
        std::ofstream f(root / "train" / "metadata.jsonl");
        f << "{\"id\":\"a\",\"split\":\"train\",\"audio_path\":\"audio/a.mp3\","
          << "\"caption_path\":\"audio/a.txt\",\"lyrics_path\":\"lyrics/a.txt\","
          << "\"audio_sha256\":\"abc\",\"duration_seconds\":1.5}\n";
        f << "{\"id\":\"b\",\"split\":\"train\",\"audio_path\":\"audio/b.mp3\","
          << "\"caption_path\":\"audio/b.txt\",\"duration_seconds\":2}\n";
    }

    sa3::TrainSplitManifest m;
    std::string err;
    fails += expect(sa3::load_train_split_manifest(root.string(), "train", m, err), "manifest load succeeds");
    fails += expect(m.filelist.size() == 2, "filelist count");
    fails += expect(m.filelist[1] == "audio/b.mp3", "filelist trim");
    fails += expect(m.records.size() == 2, "record count");
    fails += expect(m.records[0].id == "a", "record id");
    fails += expect(m.records[0].caption_path == "audio/a.txt", "caption path");
    fails += expect(m.records[0].duration_seconds > 1.49 && m.records[0].duration_seconds < 1.51, "duration");
    std::vector<sa3::TrainAudioCaptionPair> pairs;
    fails += expect(sa3::resolve_train_pairs(m, pairs, err), "pair resolution succeeds");
    fails += expect(pairs.size() == 2, "pair count follows filelist");
    fails += expect(pairs[0].id == "a", "pair id from metadata");
    fails += expect(pairs[0].audio_rel == "audio/a.mp3", "pair audio rel");
    fails += expect(pairs[0].caption_rel == "audio/a.txt", "pair caption from metadata");
    fails += expect(pairs[1].caption_rel == "audio/b.txt", "pair caption fallback");
    fails += expect(pairs[1].audio_path.find("audio/b.mp3") != std::string::npos, "pair audio joined path");
    fails += expect(sa3::validate_train_split_pairs(m, pairs, err), "split validation succeeds");

    fs::create_directories(root / "bad");
    {
        std::ofstream f(root / "bad" / "filelist.txt");
        f << "audio/missing.mp3\n";
    }
    {
        std::ofstream f(root / "bad" / "metadata.jsonl");
        f << "{\"audio_path\":\"audio/missing.mp3\"}\n";
    }
    err.clear();
    fails += expect(!sa3::load_train_split_manifest(root.string(), "bad", m, err), "missing id rejected");
    fails += expect(err.find("missing id") != std::string::npos, "missing id error");

    fs::create_directories(root / "dup" / "audio");
    {
        std::ofstream(root / "dup" / "audio" / "same.mp3") << "mp3";
        std::ofstream(root / "dup" / "audio" / "same.txt") << "caption";
        std::ofstream(root / "dup" / "audio" / "same2.mp3") << "mp3";
        std::ofstream(root / "dup" / "audio" / "same2.txt") << "caption";
        std::ofstream f(root / "dup" / "filelist.txt");
        f << "audio/same.mp3\n";
        f << "audio/same2.mp3\n";
        std::ofstream j(root / "dup" / "metadata.jsonl");
        j << "{\"id\":\"same\",\"split\":\"dup\",\"audio_path\":\"audio/same.mp3\"}\n";
        j << "{\"id\":\"same\",\"split\":\"dup\",\"audio_path\":\"audio/same2.mp3\"}\n";
    }
    err.clear();
    fails += expect(sa3::load_train_split_manifest(root.string(), "dup", m, err), "duplicate fixture loads");
    fails += expect(sa3::resolve_train_pairs(m, pairs, err), "duplicate fixture pairs resolve");
    fails += expect(!sa3::validate_train_split_pairs(m, pairs, err), "duplicate id rejected");
    fails += expect(err.find("duplicate id") != std::string::npos, "duplicate id error");

    fs::create_directories(root / "missing" / "audio");
    {
        std::ofstream(root / "missing" / "audio" / "x.mp3") << "mp3";
        std::ofstream f(root / "missing" / "filelist.txt");
        f << "audio/x.mp3\n";
        std::ofstream j(root / "missing" / "metadata.jsonl");
        j << "{\"id\":\"x\",\"split\":\"missing\",\"audio_path\":\"audio/x.mp3\"}\n";
    }
    err.clear();
    fails += expect(sa3::load_train_split_manifest(root.string(), "missing", m, err), "missing caption fixture loads");
    fails += expect(sa3::resolve_train_pairs(m, pairs, err), "missing caption fixture pairs resolve");
    fails += expect(!sa3::validate_train_split_pairs(m, pairs, err), "missing caption rejected");
    fails += expect(err.find("missing caption") != std::string::npos, "missing caption error");

    {
        std::vector<sa3::TrainAudioCaptionPair> train = pairs;
        train.clear();
        sa3::TrainAudioCaptionPair a;
        a.id = "a";
        a.audio_rel = "audio/a.mp3";
        a.audio_path = (root / "train" / "audio" / "a.mp3").string();
        a.audio_sha256 = "sha-a";
        train.push_back(a);

        std::vector<sa3::TrainAudioCaptionPair> heldout;
        sa3::TrainAudioCaptionPair b;
        b.id = "b";
        b.audio_rel = "audio/b.mp3";
        b.audio_path = (root / "train" / "audio" / "b.mp3").string();
        b.audio_sha256 = "sha-b";
        heldout.push_back(b);
        err.clear();
        fails += expect(sa3::validate_no_training_contamination(train, heldout, "test", err), "clean contamination check");

        b.audio_rel = "audio/a.mp3";
        heldout[0] = b;
        err.clear();
        fails += expect(!sa3::validate_no_training_contamination(train, heldout, "test", err), "basename contamination rejected");
        fails += expect(err.find("basename") != std::string::npos, "basename contamination error");

        b.audio_rel = "audio/c.mp3";
        b.audio_path = a.audio_path;
        heldout[0] = b;
        err.clear();
        fails += expect(!sa3::validate_no_training_contamination(train, heldout, "test", err), "path contamination rejected");
        fails += expect(err.find("path") != std::string::npos, "path contamination error");

        b.audio_path = (root / "train" / "audio" / "b.mp3").string();
        b.audio_sha256 = "sha-a";
        heldout[0] = b;
        err.clear();
        fails += expect(!sa3::validate_no_training_contamination(train, heldout, "evaluation", err), "hash contamination rejected");
        fails += expect(err.find("audio_sha256") != std::string::npos, "hash contamination error");
    }

    fs::remove_all(root);
    if (fails) return 1;
    std::printf("train_dataset_test: ok\n");
    return 0;
}
