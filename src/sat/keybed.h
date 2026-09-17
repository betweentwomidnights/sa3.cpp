// sat/keybed.h -- Foundation-1.2 Keybeds "text-to-playable instrument" policy.
//
// Ported from RoyalCities' stable-audio-tools keybed tab, prompt builder, and sampler
// exporter (RC-stable-audio-tools 43dcbb4b; MIT, originally Copyright (c) 2023 Stability
// AI). A keybed is rendered as chromatic chunks of up to six notes in one conditioned
// window, all with one shared seed, then sliced on a fixed 3.25 s grid into one sample
// per key. Nothing here touches ggml, so CLIs, tests, and plugins can share it.
//
// The random descriptor stream is stable within sa3.cpp but does not reproduce
// Python's random.Random.
#pragma once

#include "sat/foundation_prompt.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sa3::sat::keybed {

inline constexpr double kSecondsPerNote = 3.0;
inline constexpr double kSequenceGapSeconds = 0.25;
inline constexpr int kChunkSize = 6;
inline constexpr double kChunkFadeMs = 8.0;

// Measured with the Keybeds checkpoint (piano, sine, and bass prompts): the rendered
// pitch is exactly one octave below the prompt's note label, e.g. "A4" -> 220 Hz.
// RC's exporter maps labels straight to MIDI; sa3.cpp maps by sounding pitch instead.
inline constexpr int kSoundingOffsetSemitones = -12;

inline constexpr int kMinLabelMidi = 12;      // C0
inline constexpr int kMaxLabelMidi = 108;     // C8
inline constexpr int kPreviewRootMin = 36;    // C2
inline constexpr int kPreviewRootMax = 84;    // C6
inline constexpr int kSequenceMaxLabel = 104; // G#7

struct SamplerDefaults {
    int steps = 80;
    float cfg_scale = 6.0f;
    float sigma_min = 0.03f;
    float sigma_max = 500.0f;
};

inline int label_to_sounding_midi(int label_midi) {
    return label_midi + kSoundingOffsetSemitones;
}

namespace detail {

inline std::string trim(const std::string& value) {
    size_t begin = 0, end = value.size();
    while (begin < end && std::isspace((unsigned char)value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char)value[end - 1])) --end;
    return value.substr(begin, end - begin);
}

inline std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

inline std::string upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::toupper(c); });
    return value;
}

// Python's dedupe_keep_order: strip, drop empties, exact-match de-duplication.
inline std::vector<std::string> dedupe_keep_order(const std::vector<std::string>& values) {
    std::vector<std::string> out;
    for (const std::string& raw : values) {
        const std::string value = trim(raw);
        if (!value.empty() && std::find(out.begin(), out.end(), value) == out.end())
            out.push_back(value);
    }
    return out;
}

inline std::string join_prompt(const std::vector<std::string>& tokens) {
    std::string out;
    for (const std::string& token : tokens) {
        if (trim(token).empty()) continue;
        if (!out.empty()) out += ", ";
        out += token;
    }
    return out;
}

inline std::vector<std::string> split_commas(const std::string& text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        size_t comma = text.find(',', start);
        if (comma == std::string::npos) comma = text.size();
        const std::string token = trim(text.substr(start, comma - start));
        if (!token.empty()) out.push_back(token);
        start = comma + 1;
    }
    return out;
}

// Accepts [A-G](#|b)?-?digits case-insensitively, matching NOTE_TOKEN_RE.
inline bool is_note_token(const std::string& token) {
    if (token.size() < 2) return false;
    const char letter = (char)std::toupper((unsigned char)token[0]);
    if (letter < 'A' || letter > 'G') return false;
    size_t i = 1;
    if (token[i] == '#' || token[i] == 'b' || token[i] == 'B') ++i;
    if (i < token.size() && token[i] == '-') ++i;
    if (i >= token.size()) return false;
    for (; i < token.size(); ++i)
        if (!std::isdigit((unsigned char)token[i])) return false;
    return true;
}

inline bool is_position_token(const std::string& token) {
    static const std::string prefix = "keybed_pos_";
    if (token.size() != prefix.size() + 3 || lower(token.substr(0, prefix.size())) != prefix)
        return false;
    for (size_t i = prefix.size(); i < token.size(); ++i)
        if (!std::isdigit((unsigned char)token[i])) return false;
    return true;
}

} // namespace detail

// ---------------------------------------------------------------------------------------
// Notes

// C4 = 60. Accepts sharps or flats (C#4, Db4, Bb3); returns nullopt for anything else.
inline std::optional<int> note_name_to_midi(const std::string& note) {
    // Mirrors RC: upper-case, ^([A-G](#|B)?)(-?\d+)$, then a fixed name table ("BB" = Bb).
    const std::string s = detail::upper(detail::trim(note));
    if (s.size() < 2 || s[0] < 'A' || s[0] > 'G') return std::nullopt;
    const size_t name_len = (s[1] == '#' || s[1] == 'B') ? 2 : 1;
    static const std::pair<const char*, int> table[] = {
        {"C", 0}, {"C#", 1}, {"DB", 1}, {"D", 2}, {"D#", 3}, {"EB", 3}, {"E", 4}, {"F", 5},
        {"F#", 6}, {"GB", 6}, {"G", 7}, {"G#", 8}, {"AB", 8}, {"A", 9}, {"A#", 10},
        {"BB", 10}, {"B", 11},
    };
    const std::string name = s.substr(0, name_len);
    int pc = -1;
    for (const auto& entry : table)
        if (name == entry.first) pc = entry.second;
    if (pc < 0) return std::nullopt;
    size_t digits = name_len;
    if (digits < s.size() && s[digits] == '-') ++digits;
    if (digits >= s.size() || s.size() - digits > 3) return std::nullopt;
    for (size_t j = digits; j < s.size(); ++j)
        if (!std::isdigit((unsigned char)s[j])) return std::nullopt;
    const int octave = std::stoi(s.substr(name_len));
    return (octave + 1) * 12 + pc;
}

inline std::string midi_to_note_name(int midi) {
    static const char* names[] = {"C", "C#", "D", "D#", "E", "F",
                                  "F#", "G", "G#", "A", "A#", "B"};
    const int pc = ((midi % 12) + 12) % 12;
    const int octave = (int)std::floor(midi / 12.0) - 1;
    return std::string(names[pc]) + std::to_string(octave);
}

// ---------------------------------------------------------------------------------------
// Prompts

struct DescriptorTokens {
    std::vector<std::string> body;
    std::vector<std::string> fx;
};

inline bool is_fx_token(const std::string& token) {
    const std::string low = detail::lower(detail::trim(token));
    for (const char* word : {"reverb", "delay", "distortion", "phaser", "bitcrush"})
        if (low.find(word) != std::string::npos) return true;
    return false;
}

// Accepts a clean descriptor or a pasted full keybed prompt and strips the keybed
// grammar (Keybed, Sequence, Timbre Profile, Target Note <n>, Note Sequence <n...>,
// registers, Wet/Dry, legacy keybed_pos_NNN) so builders can reassemble it.
inline DescriptorTokens split_descriptor_tokens(const std::string& descriptor) {
    static const std::vector<std::string> control = {
        "keybed", "sequence", "timbre profile", "target note", "target position",
        "chromatic chunk", "note sequence", "wet", "dry", "sub register", "low register",
        "medium register", "high register", "top register",
    };
    DescriptorTokens out;
    bool skip_next = false;
    for (const std::string& token : detail::split_commas(descriptor)) {
        const std::string low = detail::lower(token);
        if (skip_next) { skip_next = false; continue; }
        if (low == "keybed" || low == "sequence" || low == "timbre profile" ||
            low == "chromatic chunk")
            continue;
        if (low == "target note" || low == "target position") { skip_next = true; continue; }
        if (low == "note sequence") break;
        if (detail::is_position_token(token) || detail::is_note_token(token)) continue;
        if (std::find(control.begin(), control.end(), low) != control.end()) continue;
        (is_fx_token(token) ? out.fx : out.body).push_back(token);
    }
    out.body = detail::dedupe_keep_order(out.body);
    out.fx = detail::dedupe_keep_order(out.fx);
    return out;
}

inline std::string clean_descriptor(const std::string& descriptor) {
    DescriptorTokens tokens = split_descriptor_tokens(descriptor);
    tokens.body.insert(tokens.body.end(), tokens.fx.begin(), tokens.fx.end());
    return detail::join_prompt(tokens.body);
}

// Exactly one Wet/Dry token. FX survive only when wet: explicit `fx` wins, otherwise FX
// words found in the descriptor are used. Wet with no FX is just "Wet".
inline std::string build_base_prompt(const std::string& descriptor, bool sequence, bool wet,
                                     const std::vector<std::string>* fx = nullptr) {
    const DescriptorTokens parts = split_descriptor_tokens(descriptor);
    std::vector<std::string> tokens{"Keybed"};
    if (sequence) tokens.push_back("Sequence");
    tokens.push_back("Timbre Profile");
    tokens.insert(tokens.end(), parts.body.begin(), parts.body.end());
    tokens.push_back(wet ? "Wet" : "Dry");
    if (wet) {
        const std::vector<std::string>& chosen = fx ? *fx : parts.fx;
        tokens.insert(tokens.end(), chosen.begin(), chosen.end());
    }
    return detail::join_prompt(detail::dedupe_keep_order(tokens));
}

inline void require_label_range(int label_midi) {
    if (label_midi < kMinLabelMidi || label_midi > kMaxLabelMidi)
        throw std::invalid_argument("keybed note outside C0-C8: " + std::to_string(label_midi));
}

inline std::string build_sequence_prompt(const std::string& descriptor,
                                         const std::vector<int>& label_midis, bool wet,
                                         const std::vector<std::string>* fx = nullptr) {
    std::vector<std::string> tokens{build_base_prompt(descriptor, true, wet, fx),
                                    "Chromatic Chunk", "Note Sequence"};
    for (int midi : label_midis) {
        require_label_range(midi);
        tokens.push_back(midi_to_note_name(midi)); // notes are never de-duplicated
    }
    return detail::join_prompt(tokens);
}

inline std::string build_single_note_prompt(const std::string& descriptor, int label_midi,
                                            bool wet,
                                            const std::vector<std::string>* fx = nullptr) {
    require_label_range(label_midi);
    return detail::join_prompt({build_base_prompt(descriptor, false, wet, fx), "Target Note",
                                midi_to_note_name(label_midi)});
}

// ---------------------------------------------------------------------------------------
// Chunk planning

enum class FullRange { C2ToB5, C2ToF6, C2ToB6 };

inline std::pair<int, int> full_range_labels(FullRange range) {
    switch (range) {
        case FullRange::C2ToB5: return {36, 83};
        case FullRange::C2ToF6: return {36, 89};
        case FullRange::C2ToB6: return {36, 95};
    }
    return {36, 83};
}

struct Chunk {
    std::vector<int> label_midis;
    double actual_seconds = 0.0;  // exact crop: n*3 + (n-1)*0.25
    int seconds_total = 0;        // whole-second conditioning value
};

inline double sequence_actual_seconds(int note_count) {
    if (note_count <= 1) return kSecondsPerNote;
    return note_count * kSecondsPerNote + (note_count - 1) * kSequenceGapSeconds;
}

inline int sequence_seconds_total(int note_count) {
    return (int)std::ceil(sequence_actual_seconds(note_count));
}

// Chunks of `chunk_size`; a trailing chunk with fewer than two notes is dropped.
inline std::vector<Chunk> chunk_labels(const std::vector<int>& label_midis,
                                       int chunk_size = kChunkSize) {
    if (chunk_size < 1) throw std::invalid_argument("keybed chunk size must be positive");
    std::vector<Chunk> chunks;
    for (size_t i = 0; i < label_midis.size(); i += (size_t)chunk_size) {
        const size_t end = std::min(label_midis.size(), i + (size_t)chunk_size);
        if (end - i < 2) continue;
        Chunk chunk;
        chunk.label_midis.assign(label_midis.begin() + (long)i, label_midis.begin() + (long)end);
        chunk.actual_seconds = sequence_actual_seconds((int)chunk.label_midis.size());
        chunk.seconds_total = sequence_seconds_total((int)chunk.label_midis.size());
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

inline std::vector<Chunk> plan_full_range(FullRange range) {
    const auto [lo, hi] = full_range_labels(range);
    std::vector<int> labels;
    for (int midi = lo; midi <= hi; ++midi) labels.push_back(midi);
    return chunk_labels(labels);
}

// Preview counts are 6, 12, or 24 chromatic notes from a root clamped to C2-C6 and so
// the run never passes G#7. Returns the clamped root label.
inline int clamp_preview_root(int root_label_midi, int note_count) {
    const int max_start = std::min(kPreviewRootMax, kSequenceMaxLabel - note_count + 1);
    return std::max(kPreviewRootMin, std::min(root_label_midi, max_start));
}

inline std::vector<Chunk> plan_preview(int root_label_midi, int note_count) {
    if (note_count != 6 && note_count != 12 && note_count != 24)
        throw std::invalid_argument("keybed preview note count must be 6, 12, or 24");
    const int root = clamp_preview_root(root_label_midi, note_count);
    std::vector<int> labels;
    for (int i = 0; i < note_count; ++i) labels.push_back(root + i);
    return chunk_labels(labels);
}

// ---------------------------------------------------------------------------------------
// Audio: planar float buffers laid out as data[channel * frames + frame].

struct NoteSlice {
    int label_midi = 0;
    int sounding_midi = 0;
    int64_t start_frame = 0;
    int64_t end_frame = 0;
};

inline int64_t round_half_even(double value) {
    return (int64_t)std::nearbyint(value); // Python round() semantics
}

inline std::vector<NoteSlice> note_slices(const std::vector<int>& label_midis, int sample_rate,
                                          int64_t total_frames) {
    std::vector<NoteSlice> slices;
    double cursor = 0.0;
    for (int midi : label_midis) {
        NoteSlice slice;
        slice.label_midi = midi;
        slice.sounding_midi = label_to_sounding_midi(midi);
        slice.start_frame = std::max<int64_t>(0, round_half_even(cursor * sample_rate));
        const int64_t end = round_half_even((cursor + kSecondsPerNote) * sample_rate);
        slice.end_frame = std::min(total_frames, std::max(slice.start_frame + 1, end));
        if (slice.start_frame < total_frames) slices.push_back(slice);
        cursor += kSecondsPerNote + kSequenceGapSeconds;
    }
    return slices;
}

inline std::vector<float> copy_planar_range(const float* planar, int channels,
                                            int64_t frames, int64_t start, int64_t end) {
    if (start < 0 || end > frames || end <= start)
        throw std::invalid_argument("invalid keybed slice range");
    const int64_t n = end - start;
    std::vector<float> out((size_t)(n * channels));
    for (int c = 0; c < channels; ++c)
        std::copy_n(planar + c * frames + start, n, out.data() + c * n);
    return out;
}

inline int64_t fade_length(int sample_rate, double ms, int64_t frames) {
    const int64_t n = round_half_even(ms / 1000.0 * sample_rate);
    return std::min(n, frames);
}

// torch.linspace-style ramps (inclusive endpoints), as RC's _apply_short_fade.
inline void apply_fade_in(std::vector<float>& planar, int channels, int sample_rate, double ms) {
    const int64_t frames = channels > 0 ? (int64_t)planar.size() / channels : 0;
    const int64_t n = fade_length(sample_rate, ms, frames);
    if (n <= 1 || frames <= 1) return;
    for (int c = 0; c < channels; ++c)
        for (int64_t i = 0; i < n; ++i)
            planar[(size_t)(c * frames + i)] *= (float)i / (float)(n - 1);
}

inline void apply_fade_out(std::vector<float>& planar, int channels, int sample_rate, double ms) {
    const int64_t frames = channels > 0 ? (int64_t)planar.size() / channels : 0;
    const int64_t n = fade_length(sample_rate, ms, frames);
    if (n <= 1 || frames <= 1) return;
    for (int c = 0; c < channels; ++c)
        for (int64_t i = 0; i < n; ++i)
            planar[(size_t)(c * frames + frames - n + i)] *= 1.0f - (float)i / (float)(n - 1);
}

// Post-process a decoded chunk the way RC does before slicing: crop, clamp, 8 ms fades.
inline std::vector<float> finish_chunk_audio(const float* planar, int channels,
                                             int64_t decoded_frames, int sample_rate,
                                             double actual_seconds) {
    const int64_t clip = std::max<int64_t>(
        1, std::min(decoded_frames, round_half_even(actual_seconds * sample_rate)));
    std::vector<float> out = copy_planar_range(planar, channels, decoded_frames, 0, clip);
    for (float& v : out) v = std::clamp(v, -1.0f, 1.0f);
    apply_fade_in(out, channels, sample_rate, kChunkFadeMs);
    apply_fade_out(out, channels, sample_rate, kChunkFadeMs);
    return out;
}

struct TrimSettings {
    double threshold_db = -60.0;
    double frame_ms = 10.0;
    double tail_pad_ms = 80.0;
    double fade_ms = 8.0;
    double terminal_fade_ms = 120.0;
    double min_keep_ms = 250.0;
    double min_trim_ms = 40.0;
};

// Trim only trailing silence inside a sliced note (RC trim_tail_conservative).
// Returns true when the buffer was shortened.
inline bool trim_tail_conservative(std::vector<float>& planar, int channels, int sample_rate,
                                   const TrimSettings& s = {}) {
    if (channels <= 0) return false;
    const int64_t n = (int64_t)planar.size() / channels;
    if (n <= 1) return false;
    const int64_t frame_len = std::max<int64_t>(1, round_half_even(sample_rate * s.frame_ms / 1000.0));
    const int64_t tail_pad = std::max<int64_t>(0, round_half_even(sample_rate * s.tail_pad_ms / 1000.0));
    const int64_t min_keep = std::max<int64_t>(1, round_half_even(sample_rate * s.min_keep_ms / 1000.0));
    const int64_t min_trim = std::max<int64_t>(1, round_half_even(sample_rate * s.min_trim_ms / 1000.0));
    const float threshold = (float)std::pow(10.0, s.threshold_db / 20.0);

    int64_t last_active = -1;
    const int64_t n_frames = (n + frame_len - 1) / frame_len;
    for (int64_t f = 0; f < n_frames; ++f) {
        float peak = 0.0f;
        const int64_t end = std::min(n, (f + 1) * frame_len);
        for (int c = 0; c < channels; ++c)
            for (int64_t i = f * frame_len; i < end; ++i)
                peak = std::max(peak, std::fabs(planar[(size_t)(c * n + i)]));
        if (peak > threshold) last_active = f;
    }
    const int64_t proposed = last_active < 0
        ? std::min(n, min_keep)
        : std::min(n, std::max(min_keep, (last_active + 1) * frame_len + tail_pad));
    if (proposed >= n - min_trim) return false;

    const int64_t keep = std::max<int64_t>(1, proposed);
    std::vector<float> trimmed = copy_planar_range(planar.data(), channels, n, 0, keep);
    apply_fade_out(trimmed, channels, sample_rate, s.fade_ms);
    planar = std::move(trimmed);
    return true;
}

// One playable sample: slice, conservative tail trim, then the 120 ms terminal fade.
inline std::vector<float> extract_note_sample(const std::vector<float>& chunk_planar,
                                              int channels, int sample_rate,
                                              const NoteSlice& slice,
                                              const TrimSettings& s = {}) {
    const int64_t frames = (int64_t)chunk_planar.size() / channels;
    std::vector<float> note = copy_planar_range(chunk_planar.data(), channels, frames,
                                                slice.start_frame, slice.end_frame);
    trim_tail_conservative(note, channels, sample_rate, s);
    if (s.terminal_fade_ms > 0.0) apply_fade_out(note, channels, sample_rate, s.terminal_fade_ms);
    return note;
}

// ---------------------------------------------------------------------------------------
// Export

struct SfzRegion {
    std::string sample_path; // relative to the .sfz, forward slashes
    int midi = 60;           // sounding pitch; lokey = hikey = pitch_keycenter
};

inline std::string sfz_text(const std::vector<SfzRegion>& regions, double attack = 0.005,
                            double release = 0.25) {
    char buffer[64];
    std::string out = "// Auto-generated Foundation-1.2 keybed export (sa3.cpp)\n<group>\n";
    std::snprintf(buffer, sizeof(buffer), "ampeg_attack=%.4f\n", attack);
    out += buffer;
    out += "ampeg_decay=0.0000\nampeg_sustain=100\n";
    std::snprintf(buffer, sizeof(buffer), "ampeg_release=%.4f\n\n", release);
    out += buffer;
    std::vector<SfzRegion> sorted = regions;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const SfzRegion& a, const SfzRegion& b) { return a.midi < b.midi; });
    for (const SfzRegion& r : sorted) {
        const std::string key = std::to_string(r.midi);
        out += "<region> sample=" + r.sample_path + " lokey=" + key + " hikey=" + key +
               " pitch_keycenter=" + key + "\n";
    }
    return out;
}

// Filesystem-friendly note name, e.g. "C#3" -> "Csharp3".
inline std::string note_filename(int midi) {
    std::string name = midi_to_note_name(midi);
    const size_t sharp = name.find('#');
    if (sharp != std::string::npos) name.replace(sharp, 1, "sharp");
    return name;
}

// ---------------------------------------------------------------------------------------
// Random descriptors (RC "simple" keybed profile)

namespace vocab {

using Weighted = foundation_prompt_detail::Weighted;

inline const std::vector<Weighted>& families() {
    static const std::vector<Weighted> v = {
        {"Synth", 30}, {"Keys", 24}, {"Bowed Strings", 13}, {"Bass", 12}, {"Mallet", 7},
        {"Wind", 6}, {"Brass", 3}, {"Guitar", 2}, {"Vocal", 2}, {"Pure Tone", 1},
    };
    return v;
}

inline const std::vector<Weighted>& subfamilies(const std::string& family) {
    static const std::vector<Weighted> synth = {
        {"Synth Lead", 70}, {"Supersaw", 21}, {"Pluck", 20}, {"Pad", 17}, {"Atmosphere", 15},
        {"Synth Bass", 11}, {"Ensemble", 6}, {"Bell", 4}, {"Texture", 4}, {"FM Synth", 3},
        {"Wavetable Synth", 2}, {"Flute", 1}, {"Digital Organ", 1}, {"Digital Strings", 1},
    };
    static const std::vector<Weighted> keys = {
        {"Grand Piano", 21}, {"Digital Piano", 17}, {"Digital Organ", 11}, {"Rhodes Piano", 10},
        {"Bell", 6}, {"Ensemble", 6}, {"Synth Lead", 4}, {"Pad", 3}, {"Pipe Organ", 2},
        {"Church Organ", 2}, {"Pluck", 2}, {"Harpsichord", 2}, {"Hammond Organ", 2},
        {"Digital Strings", 2}, {"Clavinet", 1}, {"Celesta", 1}, {"Felt Piano", 1},
        {"Tubular Bells", 1}, {"Harp", 1}, {"Wurlitzer Piano", 1}, {"Tack Piano", 1},
        {"Harmonium", 1},
    };
    static const std::vector<Weighted> bass = {
        {"Pluck", 20}, {"Reese Bass", 9}, {"808", 5}, {"Electric Bass", 5},
        {"Wavetable Bass", 3}, {"Synth Bass", 2}, {"Fingered Bass", 1}, {"FX", 1},
    };
    static const std::vector<Weighted> bowed = {
        {"Ensemble", 18}, {"Violin", 17}, {"Cello", 15}, {"Digital Strings", 13},
        {"Fiddle", 2}, {"Viola", 1}, {"Pad", 1},
    };
    static const std::vector<Weighted> mallet = {
        {"Marimba", 9}, {"Bell", 4}, {"Kalimba", 2}, {"Ensemble", 2}, {"Steel Drums", 2},
        {"Tubular Bells", 2}, {"Xylophone", 1}, {"Vibraphone", 1}, {"Glockenspiel", 1},
        {"Music Box", 1}, {"Toy Bell", 1},
    };
    static const std::vector<Weighted> wind = {
        {"Woodwinds", 11}, {"Flute", 6}, {"Saxophones", 5}, {"World Winds", 5},
        {"Pan Flute", 5}, {"Clarinet", 3}, {"Sax", 3}, {"Alto Sax", 2}, {"Oboe", 1},
        {"Bassoon", 1}, {"Tenor Sax", 1}, {"Irish Flute", 1}, {"Soprano Sax", 1},
        {"Ensemble", 1}, {"Ocarina", 1},
    };
    static const std::vector<Weighted> brass = {
        {"Trumpet", 7}, {"French Horn", 3}, {"Tuba", 2}, {"War Horn", 2},
        {"Tenor Trombone", 1}, {"Bass Trombone", 1}, {"Trombone", 1},
    };
    static const std::vector<Weighted> guitar = {
        {"Electric Guitar", 4}, {"Acoustic Guitar", 3}, {"Koto", 3}, {"Sitar", 2},
        {"Ukulele", 1}, {"Lute", 1}, {"Banjo", 1},
    };
    static const std::vector<Weighted> vocal = {
        {"Synthetic", 7}, {"Choir", 4}, {"Pad", 1}, {"Atmosphere", 1},
    };
    static const std::vector<Weighted> pure = {
        {"Sine", 1}, {"Saw", 1}, {"Triangle", 1}, {"Pulse", 1},
    };
    static const std::vector<Weighted> none;
    if (family == "Synth") return synth;
    if (family == "Keys") return keys;
    if (family == "Bass") return bass;
    if (family == "Bowed Strings") return bowed;
    if (family == "Mallet") return mallet;
    if (family == "Wind") return wind;
    if (family == "Brass") return brass;
    if (family == "Guitar") return guitar;
    if (family == "Vocal") return vocal;
    if (family == "Pure Tone") return pure;
    return none;
}

inline const std::vector<Weighted>& timbre() {
    static const std::vector<Weighted> v = {
        {"Warm", 13}, {"Bright", 12}, {"Dark", 8}, {"Airy", 9}, {"Rich", 11}, {"Clean", 12},
        {"Gritty", 9}, {"Crisp", 8}, {"Focused", 8}, {"Metallic", 8}, {"Smooth", 8},
        {"Cold", 5}, {"Buzzy", 5}, {"Round", 6}, {"Fat", 8}, {"Punchy", 8}, {"Thin", 5},
        {"Soft", 7}, {"Woody", 6}, {"Hollow", 6}, {"Nasal", 5}, {"Biting", 5},
        {"Overdriven", 5}, {"Subdued", 4}, {"Breathy", 4}, {"Glassy", 5}, {"Sparkly", 7},
        {"Shiny", 5}, {"Noisy", 4}, {"Muffled", 4}, {"Distant", 3}, {"Wide", 6}, {"Mono", 3},
        {"Near", 3}, {"Far", 3}, {"Spacey", 4}, {"Ambient", 4}, {"Intimate", 3}, {"Small", 2},
        {"Big", 4}, {"Deep", 6}, {"Rumble", 5}, {"Growl", 5}, {"Wavetable", 5}, {"Digital", 6},
        {"Analog", 6}, {"Retro", 6}, {"Vintage", 5}, {"Dubstep", 4}, {"Chiptune", 4},
        {"Formant Vocal", 2}, {"Synthetic Vox", 2}, {"Choir", 3}, {"Pluck", 6},
        {"Sustained", 6}, {"Short", 4}, {"Staccato", 4}, {"Snappy", 5}, {"Pizzicato", 3},
        {"Spiccato", 3}, {"Impact", 3}, {"Hit", 3}, {"Swell", 3}, {"Thick", 9}, {"Present", 6},
        {"Sharp", 4}, {"Harsh", 4}, {"Bell", 7}, {"Full", 8}, {"Silky", 6}, {"Square", 7},
        {"Pulse", 6}, {"Saw", 5}, {"Sine", 2}, {"Triangle", 5}, {"White Noise", 5},
        {"Pure Tone", 4}, {"FM", 4}, {"Supersaw", 5}, {"Reese", 4}, {"Filter", 4},
    };
    return v;
}

inline const std::vector<std::string>& family_boost(const std::string& family) {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> table = {
        {"Bass", {"Warm", "Thick", "Fat", "Deep", "Punchy", "Gritty", "Clean", "Full", "Dark", "Rumble"}},
        {"Keys", {"Warm", "Clean", "Bright", "Rich", "Smooth", "Sparkly", "Bell", "Soft", "Full"}},
        {"Synth", {"Warm", "Bright", "Thick", "Fat", "Digital", "Analog", "Pulse", "Square", "Saw", "Wavetable", "Supersaw", "Clean", "Gritty"}},
        {"Bowed Strings", {"Warm", "Rich", "Smooth", "Airy", "Sustained", "Dark", "Bright", "Full"}},
        {"Mallet", {"Bright", "Sparkly", "Metallic", "Bell", "Clean", "Pluck", "Woody", "Crisp"}},
        {"Wind", {"Airy", "Breathy", "Hollow", "Woody", "Thin", "Bright", "Smooth"}},
        {"Brass", {"Bright", "Nasal", "Biting", "Big", "Present", "Warm", "Harsh"}},
        {"Guitar", {"Clean", "Bright", "Woody", "Pluck", "Gritty", "Warm"}},
        {"Vocal", {"Airy", "Choir", "Synthetic Vox", "Warm", "Smooth", "Distant"}},
        {"Pure Tone", {"Clean", "Full", "Bright", "Warm", "Thin"}},
    };
    static const std::vector<std::string> none;
    for (const auto& entry : table)
        if (entry.first == family) return entry.second;
    return none;
}

inline const std::vector<std::vector<std::string>>& mutex_groups() {
    static const std::vector<std::vector<std::string>> v = {
        {"Sustained", "Short", "Staccato"},
        {"Pizzicato", "Spiccato", "Sustained"},
        {"Sine", "Saw", "Triangle", "Pulse", "Square", "White Noise"},
    };
    return v;
}

inline const std::vector<Weighted>& oscillators() {
    static const std::vector<Weighted> v = {
        {"Pure Tone", 9}, {"Sine", 4}, {"Saw", 5}, {"Triangle", 3}, {"Pulse", 6},
        {"Square", 6}, {"White Noise", 1},
    };
    return v;
}

// UI choices for the Wet FX picker.
inline const std::vector<std::string>& fx_choices() {
    static const std::vector<std::string> v = {
        "Low Reverb", "Medium Reverb", "High Reverb", "Plate Reverb", "Low Delay",
        "Medium Delay", "High Delay", "Ping Pong Delay", "Stereo Delay", "Cross Delay",
        "Mono Delay", "Low Distortion", "Medium Distortion", "High Distortion", "Phaser",
        "Low Phaser", "Medium Phaser", "High Phaser", "Bitcrush", "High Bitcrush",
    };
    return v;
}

} // namespace vocab

struct RandomDescriptor {
    std::string descriptor; // family, subfamily, tags (no Wet/Dry or FX)
    std::string family;
    std::string subfamily;
};

inline RandomDescriptor random_descriptor(uint64_t seed, bool wet = false) {
    using foundation_prompt_detail::weighted_choice;
    using foundation_prompt_detail::weighted_unique;
    std::mt19937_64 rng(seed);
    auto pick = [&](std::initializer_list<int> values) {
        const size_t i = std::uniform_int_distribution<size_t>(0, values.size() - 1)(rng);
        return *(values.begin() + i);
    };
    auto chance = [&](double p) { return std::uniform_real_distribution<double>(0, 1)(rng) < p; };

    RandomDescriptor out;
    out.family = weighted_choice(rng, vocab::families());
    const auto& subs = vocab::subfamilies(out.family);
    if (!subs.empty()) out.subfamily = weighted_choice(rng, subs);

    int k_base = 0, max_tags = 7;
    if (out.family == "Pure Tone") { k_base = pick({0, 1, 1, 2}); max_tags = 4; }
    else k_base = pick({4, 5, 6});

    std::vector<std::string> tags = weighted_unique(rng, vocab::timbre(), k_base);
    std::vector<std::string> boost = vocab::family_boost(out.family);
    if (!boost.empty()) {
        std::shuffle(boost.begin(), boost.end(), rng);
        const int k = std::min<int>(pick({1, 1, 2}), (int)boost.size());
        tags.insert(tags.end(), boost.begin(), boost.begin() + k);
    }
    auto nudge = [&](std::initializer_list<const char*> values) {
        const size_t i = std::uniform_int_distribution<size_t>(0, values.size() - 1)(rng);
        tags.push_back(*(values.begin() + i));
    };
    const std::string& sub = out.subfamily;
    if (sub == "Reese Bass" || sub == "808" || sub == "Wavetable Bass" || sub == "Synth Bass")
        nudge({"Deep", "Fat", "Rumble", "Growl", "Thick"});
    else if (sub == "Bell" || sub == "Tubular Bells" || sub == "Music Box" || sub == "Toy Bell")
        nudge({"Bright", "Sparkly", "Metallic", "Bell"});
    else if (out.family == "Bowed Strings" && (sub == "Violin" || sub == "Cello" || sub == "Viola" ||
                                               sub == "Digital Strings" || sub == "Ensemble"))
        nudge({"Sustained", "Smooth", "Warm", "Nasal", "Rich"});
    else if (sub == "Synth Lead" || sub == "Supersaw" || sub == "FM Synth" || sub == "Wavetable Synth")
        nudge({"Bright", "Digital", "Analog", "Wide", "Present"});
    if (out.family != "Pure Tone" && chance(0.07)) {
        const std::vector<std::string> osc =
            weighted_unique(rng, vocab::oscillators(), chance(0.65) ? 1 : 2);
        tags.insert(tags.end(), osc.begin(), osc.end());
    }

    tags = detail::dedupe_keep_order(tags);
    for (const auto& group : vocab::mutex_groups()) {
        std::vector<std::string> hits;
        for (const auto& t : tags)
            if (std::find(group.begin(), group.end(), t) != group.end()) hits.push_back(t);
        if (hits.size() <= 1) continue;
        const std::string keep = hits[std::uniform_int_distribution<size_t>(0, hits.size() - 1)(rng)];
        std::vector<std::string> next;
        for (const auto& t : tags)
            if (std::find(group.begin(), group.end(), t) == group.end() || t == keep)
                next.push_back(t);
        tags = std::move(next);
    }
    if (!wet) {
        tags.erase(std::remove_if(tags.begin(), tags.end(), [](const std::string& t) {
            return t == "Distant" || t == "Far" || t == "Spacey" || t == "Ambient";
        }), tags.end());
    }
    if ((int)tags.size() > max_tags) {
        std::shuffle(tags.begin(), tags.end(), rng);
        tags.resize((size_t)max_tags);
    }

    std::vector<std::string> tokens{out.family};
    if (!out.subfamily.empty()) tokens.push_back(out.subfamily);
    tokens.insert(tokens.end(), tags.begin(), tags.end());
    out.descriptor = detail::join_prompt(detail::dedupe_keep_order(tokens));
    return out;
}

} // namespace sa3::sat::keybed
