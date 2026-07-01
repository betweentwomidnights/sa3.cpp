// sa3-server: a small HTTP server over the shared sa3 Pipeline — a PROOF OF CONCEPT that mirrors
// gary4local's async job model, so a gary4juce-style client can drive it as a drop-in (SA3 on :8006).
//   POST /generate {prompt, frames|seconds, steps, seed, negative_prompt, cfg_*, dist_shift,
//                   duration_padding_sec, loras[], init_path, keep_models}
//                 -> {session_id, seed}   (generation runs in the background)
//   GET  /poll_status/<session_id>
//                 -> {success, generation_in_progress, progress, step, total_steps, status,
//                     queue_status, audio_data (base64 wav, on "completed"), meta:{seed}}
//   POST /unload   -> free the model (full VRAM release; orchestrator owns the unload policy)
//   GET  /health   -> {status, model, encoding, loaded}
// The Pipeline carries the reusable primitives (incl. GenParams::on_progress); a synchronous or SSE
// transport is left to real apps — this server only demonstrates the poll_status pattern.
#include "sa3_pipeline.h"
#include "wav.h"

#include "httplib.h"
#include "yyjson.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>

namespace {

std::mutex g_mtx;                         // serialize: one generation (one GPU graph) at a time
std::unique_ptr<sa3::Pipeline> g_pipe;    // loaded lazily on first generate; freed on /unload
std::atomic<bool> g_loaded{false};        // lock-free view for /health (won't block during a gen)
std::string g_variant   = "medium";
std::string g_encoding  = "f16";
std::string g_models_dir;
std::string g_adapters_dir;

// --- async job registry (mirrors gary4local /poll_status) ---
struct Job {
    std::string status = "queued";        // queued | generating | encoding | completed | failed
    int      progress = 0;                // 0..100
    int      step = 0, total_steps = 0;
    std::string audio_b64;                // base64 wav, filled on completion
    std::string error;
    uint64_t seed = 0;
    double   created = 0.0;
};
std::mutex jobs_mtx;
std::unordered_map<std::string, Job> jobs;

std::string json_err(const std::string& msg) { return "{\"error\":\"" + msg + "\"}"; }

// minimal JSON string escaping (quotes/backslashes/control) for error text we splice into bodies
std::string json_escape(const std::string& s) {
    std::string o; o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

std::string b64_encode(const std::string& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 3 <= in.size(); i += 3) {
        unsigned n = (unsigned char)in[i] << 16 | (unsigned char)in[i+1] << 8 | (unsigned char)in[i+2];
        out += T[(n>>18)&63]; out += T[(n>>12)&63]; out += T[(n>>6)&63]; out += T[n&63];
    }
    if (i < in.size()) {
        unsigned n = (unsigned char)in[i] << 16;
        if (i + 1 < in.size()) n |= (unsigned char)in[i+1] << 8;
        out += T[(n>>18)&63]; out += T[(n>>12)&63];
        out += (i + 1 < in.size()) ? T[(n>>6)&63] : '=';
        out += '=';
    }
    return out;
}

std::string new_session_id() {
    static const char* H = "0123456789abcdef";
    std::random_device rd;
    uint64_t r = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
    std::string s(12, '0');
    for (int i = 0; i < 12; i++) { s[i] = H[r & 0xF]; r >>= 4; }
    return s;
}

// drop finished jobs older than 10 min so the registry doesn't grow unbounded (call under jobs_mtx).
void jobs_prune() {
    const double now = sa3::wall_time_s();
    for (auto it = jobs.begin(); it != jobs.end(); ) {
        const bool done = it->second.status == "completed" || it->second.status == "failed";
        if (done && now - it->second.created > 600.0) it = jobs.erase(it);
        else ++it;
    }
}

// (re)load the pipeline under the caller's g_mtx. Returns false + message on failure.
bool ensure_loaded(std::string& err) {
    if (g_pipe && g_pipe->loaded()) { g_loaded = true; return true; }
    sa3::ModelPaths mp;
    if (!sa3::ModelPaths::resolve(g_models_dir, g_variant, g_encoding, mp, err)) return false;
    try {
        g_pipe = std::make_unique<sa3::Pipeline>();
        g_pipe->load(mp);
    } catch (const std::exception& e) { g_pipe.reset(); g_loaded = false; err = e.what(); return false; }
    g_loaded = true;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8086;
    if (const char* e = getenv("SA3_MODELS_DIR"))   g_models_dir   = e;
    if (const char* e = getenv("SA3_ADAPTERS_DIR")) g_adapters_dir = e;
    if (g_models_dir.empty()) g_models_dir = "models";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* d){ return i + 1 < argc ? argv[++i] : d; };
        if      (a == "--host")         host = next("127.0.0.1");
        else if (a == "--port")         port = atoi(next("8086"));
        else if (a == "--model")        g_variant = next("medium");
        else if (a == "--encoding")     g_encoding = next("f16");
        else if (a == "--models-dir")   g_models_dir = next("models");
        else if (a == "--adapters-dir") g_adapters_dir = next("");
    }
    const std::string adir = g_adapters_dir.empty() ? g_models_dir : g_adapters_dir;

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        const bool loaded = g_loaded.load();   // atomic: never blocks behind an in-flight generation
        std::string body = "{\"status\":\"ok\",\"model\":\"" + g_variant + "\",\"encoding\":\"" +
                           g_encoding + "\",\"loaded\":" + (loaded ? "true" : "false") + "}";
        res.set_content(body, "application/json");
    });

    svr.Post("/unload", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_pipe.reset();   // Pipeline dtor frees nets + backend (full VRAM release)
        g_loaded = false;
        res.set_content("{\"status\":\"unloaded\"}", "application/json");
    });

    // POST /generate: parse + validate on the request thread, then run the generation on a background
    // thread and return {session_id} immediately. The client polls /poll_status/<session_id>.
    svr.Post("/generate", [&adir](const httplib::Request& req, httplib::Response& res) {
        yyjson_doc* doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (!doc) { res.status = 400; res.set_content(json_err("invalid json"), "application/json"); return; }
        yyjson_val* root = yyjson_doc_get_root(doc);
        auto S = [&](const char* k, const char* d) { yyjson_val* v = yyjson_obj_get(root, k); return std::string(v && yyjson_is_str(v) ? yyjson_get_str(v) : d); };
        auto I = [&](const char* k, int d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_int(v) ? (int)yyjson_get_int(v) : d; };
        auto D = [&](const char* k, double d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_num(v) ? yyjson_get_num(v) : d; };
        auto B = [&](const char* k, bool d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_bool(v) ? yyjson_get_bool(v) : d; };

        sa3::GenParams params;
        params.prompt           = S("prompt", "");
        params.frames           = I("frames", 128);
        if (yyjson_obj_get(root, "seconds"))                 // ~10.77 latent frames/sec (44100/4096)
            params.frames = (int)(D("seconds", 12.0) * 44100.0 / 4096.0 + 0.5);
        params.steps            = I("steps", 8);
        const uint64_t seed_resolved = sa3::pick_seed(I("seed", 0));   // seed -1 => random
        params.seed             = seed_resolved;
        params.keep_models      = B("keep_models", false);   // FRUGAL default
        params.init_noise_level = (float)D("init_noise_level", 0.85);
        params.inpaint_start    = (float)D("inpaint_start", -1.0);   // inpaint/continuation region (sec)
        params.inpaint_end      = (float)D("inpaint_end", -1.0);
        params.duration_padding_sec = (float)D("duration_padding_sec", 6.0);   // text2music schedule headroom (0 = let it end)
        // classifier-free guidance (all inert unless cfg_scale != 1.0)
        params.negative_prompt   = S("negative_prompt", "");
        params.cfg_scale         = (float)D("cfg_scale", 1.0);
        params.cfg_rescale       = (float)D("cfg_rescale", 0.0);
        params.apg_scale         = (float)D("apg_scale", 1.0);
        params.cfg_norm_threshold= (float)D("cfg_norm_threshold", 0.0);
        params.cfg_interval_min  = (float)D("cfg_interval_min", 0.0);
        params.cfg_interval_max  = (float)D("cfg_interval_max", 1.0);
        // schedule warp: "dist_shift" type + its defaults, optionally overridden by a 4-number "dist_shift_params".
        params.dist_shift       = S("dist_shift", "LogSNR");
        sa3::dist_shift_defaults(params.dist_shift, params.ds_p1, params.ds_p2, params.ds_p3, params.ds_p4);
        if (yyjson_val* dsp = yyjson_obj_get(root, "dist_shift_params"); dsp && yyjson_is_arr(dsp)) {
            float* slots[4] = { &params.ds_p1, &params.ds_p2, &params.ds_p3, &params.ds_p4 };
            yyjson_val* v; yyjson_arr_iter di; yyjson_arr_iter_init(dsp, &di);
            for (int k = 0; k < 4 && (v = yyjson_arr_iter_next(&di)); k++)
                if (yyjson_is_num(v)) *slots[k] = (float)yyjson_get_num(v);
        }
        std::string init_path   = S("init_path", "");                // local WAV for audio2audio / inpaint

        std::string perr;
        yyjson_val* loras = yyjson_obj_get(root, "loras");
        if (loras && yyjson_is_arr(loras)) {
            yyjson_val* it; yyjson_arr_iter ai; yyjson_arr_iter_init(loras, &ai);
            while ((it = yyjson_arr_iter_next(&ai))) {
                yyjson_val* nv = yyjson_obj_get(it, "name");
                if (!nv) nv = yyjson_obj_get(it, "path");
                std::string name = nv && yyjson_is_str(nv) ? yyjson_get_str(nv) : "";
                yyjson_val* sv = yyjson_obj_get(it, "strength");
                float strength = sv && yyjson_is_num(sv) ? (float)yyjson_get_num(sv) : 1.0f;
                if (name.empty()) continue;
                std::string p = std::filesystem::exists(name) ? name
                                : sa3::resolve_one(adir, "lora-" + name + "-", ".gguf");
                if (p.empty()) { perr = "unknown lora '" + name + "'"; break; }
                params.loras.push_back({p, strength});
            }
        }
        yyjson_doc_free(doc);

        if (!perr.empty())            { res.status = 400; res.set_content(json_err(perr), "application/json"); return; }
        if (params.prompt.empty())    { res.status = 400; res.set_content(json_err("prompt required"), "application/json"); return; }

        if (!init_path.empty()) {     // audio2audio / inpaint source (local path — the server is localhost)
            if (!std::filesystem::exists(init_path)) {
                res.status = 400; res.set_content(json_err("init_path not found: " + init_path), "application/json"); return;
            }
            int ns = 0, nc = 0, sr = 0;
            params.init_audio = sa3::read_wav_planar(init_path, ns, nc, sr);
            params.init_n_samp = ns; params.init_n_ch = nc;
        }

        // register the job, then hand off to a worker thread and reply immediately
        const std::string sid = new_session_id();
        {
            std::lock_guard<std::mutex> lk(jobs_mtx);
            jobs_prune();
            Job j; j.total_steps = params.steps; j.seed = seed_resolved; j.created = sa3::wall_time_s();
            jobs[sid] = std::move(j);
        }
        std::thread([sid, seed_resolved, params = std::move(params)]() mutable {
            params.on_progress = [sid](const sa3::Progress& p) {   // sampling->generating, decoding->encoding
                std::lock_guard<std::mutex> lk(jobs_mtx);
                auto it = jobs.find(sid); if (it == jobs.end()) return;
                it->second.progress = (int)(p.fraction * 100.0f);
                it->second.step     = p.step;
                if      (!strcmp(p.stage, "sampling")) it->second.status = "generating";
                else if (!strcmp(p.stage, "decoding")) it->second.status = "encoding";
            };
            std::lock_guard<std::mutex> lk(g_mtx);   // serialize: one generation at a time
            { std::lock_guard<std::mutex> jl(jobs_mtx); if (auto it = jobs.find(sid); it != jobs.end()) it->second.status = "generating"; }
            std::string err;
            if (!ensure_loaded(err)) {
                std::lock_guard<std::mutex> jl(jobs_mtx);
                if (auto it = jobs.find(sid); it != jobs.end()) { it->second.status = "failed"; it->second.error = err; }
                return;
            }
            try {
                sa3::GenResult r = g_pipe->generate(params);
                std::string b64 = b64_encode(sa3::wav_planar_bytes(r.samples.data(), r.n_samp, r.n_ch, r.sample_rate));
                std::lock_guard<std::mutex> jl(jobs_mtx);
                if (auto it = jobs.find(sid); it != jobs.end()) {
                    it->second.audio_b64 = std::move(b64);
                    it->second.status = "completed"; it->second.progress = 100;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> jl(jobs_mtx);
                if (auto it = jobs.find(sid); it != jobs.end()) { it->second.status = "failed"; it->second.error = e.what(); }
            }
        }).detach();

        res.set_content("{\"session_id\":\"" + sid + "\",\"seed\":" + std::to_string(seed_resolved) + "}", "application/json");
    });

    // GET /poll_status/<session_id>: progress + (on completion) the base64 wav. Matches gary4local.
    svr.Get(R"(/poll_status/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        const std::string sid = req.matches[1];
        std::lock_guard<std::mutex> lk(jobs_mtx);
        auto it = jobs.find(sid);
        if (it == jobs.end()) {
            res.status = 404;
            res.set_content("{\"success\":false,\"error\":\"unknown session: " + json_escape(sid) + "\"}", "application/json");
            return;
        }
        const Job& j = it->second;
        const bool in_prog = j.status == "queued" || j.status == "generating" || j.status == "encoding";
        std::string qs = j.status == "queued"
            ? "{\"status\":\"queued\",\"position\":1,\"total_queued\":1,\"message\":\"queued locally\",\"estimated_seconds\":5}"
            : in_prog ? "{\"status\":\"ready\"}" : "{}";
        std::string body = "{";
        body += "\"success\":" + std::string(j.status == "failed" ? "false" : "true") + ",";
        body += "\"generation_in_progress\":" + std::string(in_prog ? "true" : "false") + ",";
        body += "\"transform_in_progress\":false,";
        body += "\"progress\":" + std::to_string(j.progress) + ",";
        body += "\"step\":" + std::to_string(j.step) + ",";
        body += "\"total_steps\":" + std::to_string(j.total_steps) + ",";
        body += "\"status\":\"" + j.status + "\",";
        body += "\"queue_status\":" + qs;
        if (j.status == "completed")
            body += ",\"audio_data\":\"" + j.audio_b64 + "\",\"meta\":{\"seed\":" + std::to_string(j.seed) + "}";
        if (j.status == "failed")
            body += ",\"error\":\"" + json_escape(j.error) + "\"";
        body += "}";
        res.set_content(body, "application/json");
    });

    fprintf(stderr, "[sa3-server] http://%s:%d  model=%s/%s  models=%s  adapters=%s  (async /poll_status; frugal default)\n",
            host.c_str(), port, g_variant.c_str(), g_encoding.c_str(), g_models_dir.c_str(), adir.c_str());
    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "[sa3-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
