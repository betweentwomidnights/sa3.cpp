// Legacy PyTorch "dora" exports must resolve before conversion and inference,
// including cached GGUFs from the previous converter and both normalization axes.
#include "lora.h"
#include "lora_convert.h"
#include "test_backend.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Spec {
    std::string module, stem, kind;
    std::vector<int64_t> shape;
    std::vector<float> values;
};
static std::vector<Spec> specs(bool columns, bool matrix, bool half) {
    std::vector<Spec> result;
    for (int index = 0; index < 2; ++index) {
        const int in = 32, out = index ? 32 : 48;
        const std::string module = "model.to_cond_embed." + std::to_string(index * 2);
        const std::string stem = "dit.cond_embed." + std::to_string(index * 2);
        for (const std::string kind : {"lora_A", "lora_B", "magnitude"}) {
            const int nmag = columns ? in : out;
            auto shape = kind == "lora_A" ? std::vector<int64_t>{4, in}
                       : kind == "lora_B" ? std::vector<int64_t>{out, 4}
                       : !matrix ? std::vector<int64_t>{nmag}
                       : columns ? std::vector<int64_t>{1, nmag}
                                 : std::vector<int64_t>{nmag, 1};
            size_t n = 1; for (auto d : shape) n *= (size_t)d;
            Spec s{module, stem, kind, shape, std::vector<float>(n)};
            for (size_t i = 0; i < n; ++i) {
                float v = kind == "magnitude" ? 0.6f + 0.001f * i : 0.02f * std::sin((float)i);
                if (half) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
                s.values[i] = v;
            }
            result.push_back(std::move(s));
        }
    }
    return result;
}
static const std::string config = "{\"rank\":4,\"alpha\":4,\"adapter_type\":\"dora\"}";

static void write_safe(const std::filesystem::path& path, const std::vector<Spec>& data, bool half) {
    std::ostringstream header;
    header << R"({"__metadata__":{"lora_config":"{\"rank\":4,\"alpha\":4,\"adapter_type\":\"dora\"}"})";
    size_t offset = 0;
    for (const auto& s : data) {
        header << ",\"" << s.module << ".parametrizations.weight.0." << s.kind
               << "\":{\"dtype\":\"" << (half ? "F16" : "F32") << "\",\"shape\":[";
        for (size_t i = 0; i < s.shape.size(); ++i) header << (i ? "," : "") << s.shape[i];
        const size_t end = offset + s.values.size() * (half ? 2 : 4);
        header << "],\"data_offsets\":[" << offset << "," << end << "]}";
        offset = end;
    }
    header << "}";
    const std::string h = header.str(); const uint64_t size = h.size();
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)&size, 8); f.write(h.data(), h.size());
    for (const auto& s : data) for (float v : s.values) {
        if (half) { const auto h16 = ggml_fp32_to_fp16(v); f.write((const char*)&h16, 2); }
        else f.write((const char*)&v, 4);
    }
    check(f.good(), "fixture safetensors write failed");
}
static void write_old_gguf(const std::filesystem::path& path, const std::vector<Spec>& data) {
    ggml_init_params ip = {1u << 20, nullptr, false};
    auto* ctx = ggml_init(ip); auto* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "sa3-lora");
    gguf_set_val_str(g, "lora.adapter_type", "dora");
    gguf_set_val_u32(g, "lora.rank", 4); gguf_set_val_f32(g, "lora.alpha", 4.0f);
    for (const auto& s : data) {
        int64_t ne[4] = {1,1,1,1};
        for (size_t i = 0; i < s.shape.size(); ++i) ne[i] = s.shape[s.shape.size()-1-i];
        auto* t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int)s.shape.size(), ne);
        ggml_set_name(t, (s.stem + "." + s.kind).c_str());
        std::memcpy(t->data, s.values.data(), s.values.size() * sizeof(float));
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.string().c_str(), false);
    gguf_free(g); ggml_free(ctx); check(ok, "fixture GGUF write failed");
}
static void base(sa3::GgufModel& m) {
    ggml_init_params ip = {8 * ggml_tensor_overhead(), nullptr, true};
    m.ctx = ggml_init(ip); m.backend = sa3_test_cpu_backend(); m.owns_backend = false;
    for (int i = 0; i < 2; ++i) {
        const std::string n = "dit.cond_embed." + std::to_string(i * 2) + ".weight";
        m.tensors[n] = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, 32, i ? 32 : 48);
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.backend);
    for (const auto& kv : m.tensors) {
        std::vector<float> values((size_t)ggml_nelements(kv.second));
        for (size_t i = 0; i < values.size(); ++i) values[i] = 0.05f * std::cos((float)i);
        sa3::write_from_f32(kv.second, values);
    }
}
int main() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("sa3-legacy-dora-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(std::filesystem::create_directory(dir), "fixture directory already exists");
    try {
        check(sa3::resolve_legacy_dora({{32,32,32}}) == "dora-rows", "square fallback differs from Python");
        for (const auto shapes : {std::vector<sa3::LegacyDoraShape>{}, {{32,48,17}},
                                  {{32,48,48}, {32,48,32}}, {{32,48,48,0}}}) {
            bool threw = false;
            try { sa3::resolve_legacy_dora(shapes); } catch (const std::runtime_error&) { threw = true; }
            check(threw, "missing/invalid/conflicting axis must be rejected");
        }
        for (bool columns : {false, true}) for (bool matrix : {false, true})
        for (bool half : {false, true}) for (bool sidecar : {false, true}) {
            const auto data = specs(columns, matrix, half);
            write_safe(dir / "adapter.safetensors", data, half);
            std::ofstream(dir / "adapter.json") << config;
            std::string error;
            const bool converted = sa3::convert_lora_safetensors((dir / "adapter.safetensors").string(),
                  sidecar ? (dir / "adapter.json").string() : "", (dir / "converted.gguf").string(), error);
            check(converted, error.c_str());
            write_old_gguf(dir / "cached.gguf", data);
            auto canonical = sa3::load_lora((dir / "converted.gguf").string().c_str(), 0.7f, sa3_test_cpu_backend());
            auto cached = sa3::load_lora((dir / "cached.gguf").string().c_str(), 0.7f, sa3_test_cpu_backend());
            const std::string expected = columns ? "dora-cols" : "dora-rows";
            check(canonical.type == expected && cached.type == expected, "wrong resolved DoRA axis");
            const int key = gguf_find_key(canonical.gguf.gguf, "lora.adapter_type");
            check(expected == gguf_get_val_str(canonical.gguf.gguf, key), "converter did not canonicalize metadata");
            std::vector<sa3::LoraAdapter> actual, reference;
            actual.push_back(std::move(cached)); reference.push_back(std::move(canonical));
            check(sa3::functional_lora_ok(actual) == !columns, "wrong quantized/functional eligibility");
            sa3::GgufModel a, r; base(a); base(r);
            sa3::apply_loras(a, actual); sa3::apply_loras(r, reference);
            for (const auto& kv : a.tensors) {
                std::vector<float> av, rv; sa3::read_to_f32(kv.second, av); sa3::read_to_f32(r.get(kv.first), rv);
                for (size_t i = 0; i < av.size(); ++i)
                    check(std::isfinite(av[i]) && std::fabs(av[i]-rv[i]) < 1e-6f, "cached/native merge mismatch");
            }
        }
        // Square-only flattened Gary adapters retain the Python rows default.
        auto square_only = specs(false, false, false);
        square_only.erase(square_only.begin(), square_only.begin() + 3);
        write_old_gguf(dir / "square-only.gguf", square_only);
        auto square = sa3::load_lora((dir / "square-only.gguf").string().c_str(), 1.0f, sa3_test_cpu_backend());
        check(square.type == "dora-rows", "square-only cached Gary adapter lost rows default");
        std::filesystem::remove_all(dir);
        std::puts("lora_legacy_test: ok (16 conversion/cache/merge cases)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s (fixtures retained at %s)\n", e.what(), dir.string().c_str());
        return 1;
    }
}
