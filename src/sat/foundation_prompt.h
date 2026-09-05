// sat/foundation_prompt.h -- Foundation-1's structured prompt policy.
//
// The vocabulary and M1/T1 organization are adapted from RoyalCities'
// stable-audio-tools Foundation prompt engine (MIT; originally Copyright
// (c) 2023 Stability AI). The C++ RNG stream is intentionally stable within
// sa3.cpp, but is not expected to reproduce Python's random.Random byte-for-byte.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sa3::sat {

enum class FoundationPromptMode { Standard, Mix };

struct FoundationRandomPrompt {
    std::string description;
    std::string family;
    std::string subfamily;
    std::string variant;
};

namespace foundation_prompt_detail {

using Weighted = std::pair<const char*, int>;

inline std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value;
}

inline const std::vector<Weighted>& families(FoundationPromptMode mode) {
    static const std::vector<Weighted> standard = {
        {"Synth", 36}, {"Keys", 22}, {"Bass", 14}, {"Bowed Strings", 8},
        {"Mallet", 6}, {"Wind", 4}, {"Guitar", 4}, {"Brass", 3},
        {"Vocal", 2}, {"Plucked Strings", 1},
    };
    static const std::vector<Weighted> mix = {
        {"Synth", 55}, {"Keys", 14}, {"Bass", 18}, {"Bowed Strings", 3},
        {"Mallet", 2}, {"Wind", 2}, {"Guitar", 2}, {"Brass", 1},
        {"Vocal", 1}, {"Plucked Strings", 0},
    };
    return mode == FoundationPromptMode::Mix ? mix : standard;
}

inline const std::vector<Weighted>& subfamilies(const std::string& family) {
    static const std::vector<Weighted> synth = {
        {"Synth Lead", 40}, {"Pluck", 15}, {"Pad", 12}, {"Supersaw", 10},
        {"FM Synth", 8}, {"Wavetable Synth", 8}, {"Atmosphere", 4}, {"Texture", 3},
    };
    static const std::vector<Weighted> keys = {
        {"Grand Piano", 20}, {"Digital Piano", 25}, {"Rhodes Piano", 20},
        {"Felt Piano", 8}, {"Wurlitzer Piano", 8}, {"Clavinet", 6},
        {"Hammond Organ", 6}, {"Church Organ", 4}, {"Harpsichord", 3},
    };
    static const std::vector<Weighted> bass = {
        {"Wavetable Bass", 25}, {"Reese Bass", 20}, {"Sub Bass", 18},
        {"Electric Bass", 12}, {"Analog Bass", 8}, {"FM Bass", 7},
        {"Picked Bass", 5}, {"Digital Bass", 5},
    };
    static const std::vector<Weighted> strings = {
        {"Violin", 35}, {"Cello", 30}, {"Viola", 10}, {"Fiddle", 10},
        {"Digital Strings", 15},
    };
    static const std::vector<Weighted> mallet = {
        {"Bell", 30}, {"Marimba", 25}, {"Vibraphone", 15},
        {"Glockenspiel", 10}, {"Kalimba", 10}, {"Xylophone", 10},
    };
    static const std::vector<Weighted> wind = {
        {"Flute", 40}, {"Pan Flute", 20}, {"Piccolo", 8}, {"Clarinet", 8},
        {"Oboe", 6}, {"Bassoon", 4}, {"Ocarina", 6}, {"World Winds", 8},
    };
    static const std::vector<Weighted> guitar = {
        {"Electric Guitar", 50}, {"Acoustic Guitar", 30}, {"Nylon Guitar", 20},
    };
    static const std::vector<Weighted> brass = {
        {"Trumpet", 40}, {"Brass", 25}, {"French Horn", 10}, {"Tuba", 8},
        {"Tenor Trombone", 9}, {"Bass Trombone", 8},
    };
    static const std::vector<Weighted> vocal = {
        {"Texture", 45}, {"Choir", 25}, {"Ensemble", 15}, {"Synthetic Choir", 15},
    };
    static const std::vector<Weighted> plucked = {
        {"Harp", 45}, {"Concert Harp", 20}, {"Celtic Harp", 15},
        {"Koto", 10}, {"Sitar", 10},
    };
    static const std::vector<Weighted> empty;
    if (family == "Synth") return synth;
    if (family == "Keys") return keys;
    if (family == "Bass") return bass;
    if (family == "Bowed Strings") return strings;
    if (family == "Mallet") return mallet;
    if (family == "Wind") return wind;
    if (family == "Guitar") return guitar;
    if (family == "Brass") return brass;
    if (family == "Vocal") return vocal;
    if (family == "Plucked Strings") return plucked;
    return empty;
}

inline const std::vector<Weighted>& timbre() {
    static const std::vector<Weighted> values = {
        {"warm",12},{"bright",10},{"tight",11},{"thick",11},{"airy",9},{"rich",11},
        {"clean",9},{"gritty",9},{"crisp",9},{"focused",8},{"metallic",8},{"dark",8},
        {"shiny",7},{"present",8},{"silky",8},{"sparkly",7},{"smooth",6},{"cold",5},
        {"buzzy",5},{"round",5},{"fat",5},{"punchy",5},{"thin",4},{"soft",4},
        {"woody",4},{"hollow",4},{"nasal",3},{"biting",3},{"overdriven",3},
        {"subdued",2},{"breathy",2},{"glassy",2},{"pizzicato",2},{"staccato",2},
        {"snappy",2},{"full",2},{"harsh",1},{"knock",1},{"muddy",1},{"steel",1},
        {"veiled",1},{"rubbery",1},{"rumble",1},{"noisy",1},{"boomy",1},
        {"crispy",1},{"dreamy",1},{"heavy",1},{"tiny",1},{"spiccato",2},
    };
    return values;
}

inline const std::vector<Weighted>& spatial() {
    static const std::vector<Weighted> values = {
        {"wide",14},{"mono",6},{"near",14},{"far",10},{"spacey",10},{"ambient",10},
        {"distant",6},{"intimate",10},{"small",8},{"big",8},{"deep",4},
    };
    return values;
}

inline const std::vector<Weighted>& bands() {
    static const std::vector<Weighted> values = {
        {"sub",5},{"sub bass",11},{"bass",12},{"low mids",11},{"mids",10},
        {"upper mids",10},{"highs",10},{"air",9},
    };
    return values;
}

inline const std::vector<Weighted>& wave_tech() {
    static const std::vector<Weighted> values = {
        {"saw",12},{"square",12},{"sine",12},{"triangle",6},{"pulse",7},
        {"analog",10},{"digital",11},{"fm",8},{"supersaw",8},{"reese",7},
        {"pitch bend",3},{"white noise",2},{"filter",2},
    };
    return values;
}

inline const std::vector<Weighted>& styles() {
    static const std::vector<Weighted> values = {
        {"dubstep",10},{"chiptune",10},{"acid",6},{"303",8},{"retro",16},
        {"vintage",14},{"laser",8},{"siren",6},{"fx",8},{"formant vocal",10},
        {"growl",12},
    };
    return values;
}

inline const std::vector<Weighted>& effects() {
    static const std::vector<Weighted> values = {
        {"Low Reverb",37},{"Medium Reverb",45},{"High Reverb",17},{"Plate Reverb",1},
        {"Low Delay",28},{"Medium Delay",25},{"Ping Pong Delay",27},{"Stereo Delay",10},
        {"Cross Delay",3},{"Delay",4},{"High Delay",2},{"Mono Delay",1},
        {"Low Distortion",18},{"Medium Distortion",17},{"High Distortion",10},
        {"Phaser",10},{"Low Phaser",6},{"Medium Phaser",5},{"High Phaser",5},
        {"Bitcrush",5},{"High Bitcrush",1},
    };
    return values;
}

inline const std::vector<std::string>& boosts(const std::string& family) {
    static const std::vector<std::string> synth = {"digital","analog","fm","supersaw","wide","laser","saw","square"};
    static const std::vector<std::string> keys = {"warm","clean","soft","rich","smooth"};
    static const std::vector<std::string> bass = {"fat","punchy","tight","gritty","dark","sub bass","bass"};
    static const std::vector<std::string> strings = {"pizzicato","staccato","spiccato","rich","warm"};
    static const std::vector<std::string> mallet = {"woody","sparkly","shiny","crisp","bright"};
    static const std::vector<std::string> wind = {"hollow","airy","breathy","thin","woody"};
    static const std::vector<std::string> guitar = {"crisp","woody","bright","clean","gritty"};
    static const std::vector<std::string> brass = {"nasal","present","biting","bright","big"};
    static const std::vector<std::string> vocal = {"formant vocal","breathy","intimate","airy"};
    static const std::vector<std::string> empty;
    if (family == "Synth") return synth;
    if (family == "Keys") return keys;
    if (family == "Bass") return bass;
    if (family == "Bowed Strings") return strings;
    if (family == "Mallet") return mallet;
    if (family == "Wind") return wind;
    if (family == "Guitar") return guitar;
    if (family == "Brass") return brass;
    if (family == "Vocal") return vocal;
    return empty;
}

template <typename Rng>
inline std::string weighted_choice(Rng& rng, const std::vector<Weighted>& values) {
    int total = 0;
    for (const auto& value : values) total += std::max(0, value.second);
    if (total <= 0) throw std::invalid_argument("empty weighted Foundation prompt category");
    int pick = std::uniform_int_distribution<int>(1, total)(rng);
    for (const auto& value : values) {
        pick -= std::max(0, value.second);
        if (pick <= 0) return value.first;
    }
    return values.back().first;
}

template <typename Rng>
inline std::vector<std::string> weighted_unique(Rng& rng,
                                                const std::vector<Weighted>& values,
                                                int count) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (int tries = 0; (int)out.size() < count && tries < 512; ++tries) {
        const std::string value = weighted_choice(rng, values);
        if (seen.insert(value).second) out.push_back(value);
    }
    return out;
}

inline void append_unique(std::vector<std::string>& out, const std::string& value) {
    if (!value.empty() && std::find(out.begin(), out.end(), value) == out.end())
        out.push_back(value);
}

template <typename Rng>
inline std::string random_item(Rng& rng, std::initializer_list<const char*> values) {
    const size_t i = std::uniform_int_distribution<size_t>(0, values.size() - 1)(rng);
    return *(values.begin() + i);
}

inline std::string canonical_family(const std::string& hint) {
    if (hint.empty()) return {};
    const std::string wanted = lower(hint);
    for (const auto& item : families(FoundationPromptMode::Standard))
        if (lower(item.first) == wanted) return item.first;
    throw std::invalid_argument(
        "unknown Foundation family; use Synth, Keys, Bass, Bowed Strings, Mallet, Wind, "
        "Guitar, Brass, Vocal, or Plucked Strings");
}

template <typename Rng>
inline std::vector<std::string> sample_tags(Rng& rng, const std::string& family,
                                            FoundationPromptMode mode) {
    const bool mix = mode == FoundationPromptMode::Mix;
    const bool synthy = family == "Synth" || family == "Bass";
    std::vector<std::string> out;
    if (std::bernoulli_distribution(mix ? 0.66 : 0.50)(rng))
        for (const auto& v : weighted_unique(rng, bands(), 1)) append_unique(out, v);
    const int timbre_count = mix
        ? std::uniform_int_distribution<int>(synthy ? 4 : 3, synthy ? 7 : 6)(rng)
        : std::uniform_int_distribution<int>(3, 5)(rng);
    for (const auto& v : weighted_unique(rng, timbre(), timbre_count)) append_unique(out, v);
    const auto& family_boosts = boosts(family);
    if (!family_boosts.empty()) {
        const int count = std::uniform_int_distribution<int>(0, 2)(rng);
        for (int i = 0; i < count; ++i)
            append_unique(out, family_boosts[std::uniform_int_distribution<size_t>(0, family_boosts.size() - 1)(rng)]);
    }
    const int spatial_count = mix
        ? std::uniform_int_distribution<int>(1, 3)(rng)
        : std::uniform_int_distribution<int>(0, 2)(rng);
    for (const auto& v : weighted_unique(rng, spatial(), spatial_count)) append_unique(out, v);
    const int wave_count = mix && synthy ? std::uniform_int_distribution<int>(2, 4)(rng)
                                         : std::uniform_int_distribution<int>(0, 2)(rng);
    if (synthy || std::bernoulli_distribution(mix ? 0.35 : 0.25)(rng))
        for (const auto& v : weighted_unique(rng, wave_tech(), synthy ? wave_count : std::min(1, wave_count)))
            append_unique(out, v);
    if (std::bernoulli_distribution(mix ? (synthy ? 0.80 : 0.65) : 0.60)(rng)) {
        std::string style;
        do { style = weighted_choice(rng, styles()); }
        while (family != "Synth" && family != "Bass" && (style == "acid" || style == "303"));
        append_unique(out, style);
    }
    // RoyalCities treats these articulations as mutually exclusive.
    std::vector<size_t> articulation;
    for (size_t i = 0; i < out.size(); ++i)
        if (out[i] == "pizzicato" || out[i] == "staccato" || out[i] == "spiccato")
            articulation.push_back(i);
    for (size_t i = articulation.size(); i > 1; --i)
        out.erase(out.begin() + (long)articulation[i - 1]);
    return out;
}

template <typename Rng>
inline std::vector<std::string> melody(Rng& rng, const std::string& family,
                                       FoundationPromptMode mode) {
    std::vector<std::string> out;
    if (std::bernoulli_distribution(mode == FoundationPromptMode::Mix ? 0.65 : 0.55)(rng))
        append_unique(out, random_item(rng, {"slow speed", "medium speed", "fast speed"}));
    const int rhythms = std::uniform_int_distribution<int>(0, 2)(rng);
    for (int i = 0; i < rhythms; ++i)
        append_unique(out, random_item(rng, {"off beat", "alternating", "triplets", "strummed", "arp"}));
    if (family == "Bass")
        append_unique(out, random_item(rng, {"chord progression", "dance chord progression", "arp", "melody", "bassline"}));
    else
        append_unique(out, random_item(rng, {"chord progression", "dance chord progression", "arp", "melody"}));
    const int contours = std::uniform_int_distribution<int>(0, 2)(rng);
    for (int i = 0; i < contours; ++i)
        append_unique(out, random_item(rng, {"rising", "falling", "bounce", "rolling", "sustained", "choppy", "top"}));
    const int densities = std::uniform_int_distribution<int>(0, 2)(rng);
    for (int i = 0; i < densities; ++i)
        append_unique(out, random_item(rng, {"simple", "repeating", "catchy", "complex", "epic"}));
    return out;
}

inline std::string join(const std::vector<std::string>& values) {
    std::string result;
    for (const auto& value : values) {
        if (!result.empty()) result += ", ";
        result += value;
    }
    return result;
}

} // namespace foundation_prompt_detail

inline FoundationPromptMode parse_foundation_prompt_mode(const std::string& value) {
    const std::string mode = foundation_prompt_detail::lower(value);
    if (mode == "standard" || mode == "m1") return FoundationPromptMode::Standard;
    if (mode == "mix" || mode == "experimental" || mode == "t1")
        return FoundationPromptMode::Mix;
    throw std::invalid_argument("Foundation randomize mode must be standard/M1 or mix/T1");
}

inline const char* foundation_prompt_mode_name(FoundationPromptMode mode) {
    return mode == FoundationPromptMode::Mix ? "mix" : "standard";
}

inline FoundationRandomPrompt randomize_foundation_prompt(
    uint64_t seed, FoundationPromptMode mode = FoundationPromptMode::Standard,
    const std::string& family_hint = {}) {
    using namespace foundation_prompt_detail;
    std::mt19937_64 rng(seed);
    FoundationRandomPrompt result;
    result.variant = mode == FoundationPromptMode::Mix ? "T1" : "M1";
    result.family = canonical_family(family_hint);
    if (result.family.empty()) result.family = weighted_choice(rng, families(mode));
    result.subfamily = weighted_choice(rng, subfamilies(result.family));

    std::vector<std::string> family_block = {result.family, result.subfamily};
    std::vector<std::string> tags = sample_tags(rng, result.family, mode);
    if (mode == FoundationPromptMode::Mix &&
        std::bernoulli_distribution(result.family == "Synth" || result.family == "Bass" ? 0.18 : 0.28)(rng)) {
        std::string second;
        do { second = weighted_choice(rng, families(FoundationPromptMode::Mix)); }
        while (second == result.family);
        append_unique(family_block, second);
        append_unique(family_block, weighted_choice(rng, subfamilies(second)));
        for (const auto& tag : sample_tags(rng, second, FoundationPromptMode::Mix)) append_unique(tags, tag);
        if (tags.size() > 18) {
            std::shuffle(tags.begin(), tags.end(), rng);
            tags.resize(18);
        }
    } else if (tags.size() > 14) {
        std::shuffle(tags.begin(), tags.end(), rng);
        tags.resize(14);
    }

    std::vector<std::string> fx;
    if (std::bernoulli_distribution(mode == FoundationPromptMode::Mix ? 0.80 : 0.70)(rng)) {
        append_unique(fx, weighted_choice(rng, effects()));
        if (std::bernoulli_distribution(0.25)(rng)) append_unique(fx, weighted_choice(rng, effects()));
    }
    std::vector<std::string> melody_block = melody(rng, result.family, mode);
    std::shuffle(family_block.begin(), family_block.end(), rng);
    std::shuffle(tags.begin(), tags.end(), rng);
    std::shuffle(fx.begin(), fx.end(), rng);
    std::shuffle(melody_block.begin(), melody_block.end(), rng);

    std::vector<std::vector<std::string>> blocks = {family_block, tags, fx, melody_block};
    if (!std::bernoulli_distribution(0.75)(rng))
        std::shuffle(blocks.begin(), blocks.end(), rng);
    else
        std::shuffle(blocks.begin() + 1, blocks.end(), rng);
    std::vector<std::string> tokens;
    for (const auto& block : blocks)
        for (const auto& value : block) append_unique(tokens, value);
    result.description = join(tokens);
    return result;
}

} // namespace sa3::sat
