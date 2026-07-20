// sa3-train-web: a small standalone HTTP companion that drives `sa3-train`
// behind a browser UI. It does NOT modify sa3-server or the training code; it
// simply spawns `sa3-train` as a subprocess (writing a --config json), streams
// its metrics.jsonl + stdout, and serves the training artifacts.
//
//   POST /api/train/start   {json config} -> {run_id, output_dir}
//   GET  /api/train/runs    list all runs (history + active)
//   GET  /api/train/runs/<id>  full detail (config, metrics, artifacts, log)
//   GET  /api/train/status  status of the active (or ?run_id=) run
//   GET  /api/train/log?offset=N[&run_id=]   raw child log tail
//   GET  /api/train/metrics[?run_id=]        recent metrics.jsonl objects
//   GET  /api/train/artifacts[?run_id=]      file list in output_dir
//   GET  /api/train/download?file=<n>[&run_id=]  stream an artifact
//   POST /api/train/stop[?run_id=]           SIGTERM the active run
//   GET  /api/health
//   GET  /                    embedded train.html
//   GET  /train.js           embedded train.js
#include "train_web_run.h"
#include "embedded_train_web.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "httplib.h"
#include "yyjson.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using namespace sa3_train_web;

const int kDefaultPort = 8016;
const char* kArtifactPatterns[] = {
    "adapter-step-", "adapter-final", "trainer-state", "preview.wav", "metrics.jsonl", "command.txt"};

// ---- global state (serialized by g_mtx) -----------------------------------
std::mutex g_mtx;
RunRegistry g_registry;
std::string g_index_path;     // registry persistence path
std::string g_train_bin;      // resolved sa3-train binary
std::string g_models_dir;     // optional SA3_MODELS_DIR passthrough
int64_t g_log_tail_max = 1 << 20;  // cap in-memory log at ~1MB

// Read everything currently in metrics.jsonl and return it as a JSON array
// string. Returns "[]" on any error.
std::string read_metrics_array(const std::string& dir, int limit = 0) {
    std::filesystem::path p = std::filesystem::path(dir) / "metrics.jsonl";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) return "[]";
    std::ifstream f(p);
    if (!f) return "[]";
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        lines.push_back(line);
    }
    // Optionally keep only the last `limit` lines.
    size_t start = 0;
    if (limit > 0 && lines.size() > (size_t)limit) start = lines.size() - (size_t)limit;
    std::string out = "[";
    bool first = true;
    for (size_t i = start; i < lines.size(); ++i) {
        if (!first) out += ",";
        first = false;
        out += lines[i];
    }
    out += "]";
    return out;
}

// Build a JSON object describing one run's summary fields.
std::string run_summary_json(const TrainRun& r) {
    std::stringstream ss;
    ss << "{";
    ss << "\"id\":\"" << json_escape(r.id) << "\""
       << ",\"output_dir\":\"" << json_escape(r.output_dir) << "\""
       << ",\"dataset\":\"" << json_escape(r.dataset) << "\""
       << ",\"model\":\"" << json_escape(r.model) << "\""
       << ",\"adapter_type\":\"" << json_escape(r.adapter_type) << "\""
       << ",\"max_steps\":" << r.max_steps
       << ",\"started_at\":" << r.started_at
       << ",\"finished_at\":" << r.finished_at
       << ",\"status\":\"" << run_status_str(r.status) << "\""
       << ",\"step\":" << r.latest.step
       << ",\"loss\":" << r.latest.loss
       << ",\"lr\":" << r.latest.lr
       << ",\"grad_norm\":" << r.latest.grad_norm;
    ss << "}";
    return ss.str();
}

// List artifacts for a run as a JSON array of {name,size,downloadable}.
std::string artifacts_json(const TrainRun& r) {
    std::vector<std::string> pats(kArtifactPatterns,
                                  kArtifactPatterns + sizeof(kArtifactPatterns) / sizeof(kArtifactPatterns[0]));
    auto files = list_artifacts(r.output_dir, pats);
    std::string out = "[";
    bool first = true;
    for (auto& f : files) {
        std::error_code ec;
        auto sz = std::filesystem::file_size(f, ec);
        if (!first) out += ",";
        first = false;
        out += "{\"name\":\"" + json_escape(f.filename().string()) + "\"";
        out += ",\"size\":" + std::to_string(ec ? 0 : (long long)sz);
        out += ",\"is_wav\":" + std::string(f.extension() == ".wav" ? "true" : "false");
        out += "}";
    }
    out += "]";
    return out;
}

// Resolve the sa3-train binary: SA3_TRAIN_BIN -> sibling -> PATH.
std::string resolve_train_bin(const std::string& argv0) {
    if (const char* e = std::getenv("SA3_TRAIN_BIN")) {
        if (std::strlen(e) > 0) return e;
    }
    // sibling of this executable
    std::error_code ec;
    std::filesystem::path self = std::filesystem::canonical(argv0, ec);
    if (!ec) {
        std::filesystem::path sib = self.parent_path() / "sa3-train";
        if (std::filesystem::is_regular_file(sib, ec)) return sib.string();
    }
    // PATH
    const char* path = std::getenv("PATH");
    if (path) {
        std::stringstream ss(path);
        std::string seg;
        while (std::getline(ss, seg, ':')) {
            std::filesystem::path cand = std::filesystem::path(seg) / "sa3-train";
            if (std::filesystem::is_regular_file(cand, ec)) return cand.string();
        }
    }
    return "";
}

// Spawn sa3-train with the given config file. Returns pid (>0) on success,
// or -1 on failure (and sets perr).
pid_t spawn_train(const std::string& bin, const std::string& cfg_path, int& out_fd, std::string& perr) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { perr = "pipe() failed"; return -1; }
    pid_t pid = fork();
    if (pid < 0) { perr = "fork() failed"; close(pipefd[0]); close(pipefd[1]); return -1; }
    if (pid == 0) {
        // child: dup pipe to stdout+stderr, exec
        dup2(pipefd[1], 1);
        dup2(pipefd[1], 2);
        close(pipefd[0]);
        close(pipefd[1]);
        // Pass through SA3 env so the trainer finds models.
        execlp(bin.c_str(), bin.c_str(), "--config", cfg_path.c_str(), (char*)nullptr);
        // if exec fails:
        std::cerr << "sa3-train-web: execlp failed: " << strerror(errno) << "\n";
        _exit(127);
    }
    // parent
    close(pipefd[1]);
    // make read end non-blocking so our reader thread can poll
    int fl = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, fl | O_NONBLOCK);
    out_fd = pipefd[0];
    return pid;
}

// Reader thread: drain the child pipe into the run's log buffer and update
// latest metrics from metrics.jsonl periodically.
void reader_loop(TrainRun* run) {
    const int fd = run->stdout_fd;
    char buf[4096];
    while (true) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n > 0) {
            std::lock_guard<std::mutex> lk(g_mtx);
            run->log.append(buf, (size_t)n);
            if (run->log.size() > (size_t)g_log_tail_max) {
                run->log.erase(0, run->log.size() - (size_t)g_log_tail_max);
            }
        } else if (n == 0) {
            break;  // EOF: child closed its end
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // no data yet; refresh latest metrics then sleep
            } else if (errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
        // Refresh latest metrics from the file (cheap; last line wins).
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            std::string arr = read_metrics_array(run->output_dir, 1);
            // arr is "[{...}]" or "[]"; extract the object
            if (arr.size() > 2) {
                std::string line = arr.substr(1, arr.size() - 2);
                MetricSample s;
                if (parse_metric_line(line, s)) run->latest = s;
            }
        }
        if (n > 0) continue;  // loop immediately to drain more
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    close(fd);
    // Child finished: reap and update status.
    int status = 0;
    waitpid(run->pid, &status, 0);
    std::lock_guard<std::mutex> lk(g_mtx);
    if (run->status == RunStatus::Running) {
        if (WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM) {
            run->status = RunStatus::Stopped;
        } else if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            run->status = RunStatus::Completed;
        } else {
            run->status = RunStatus::Failed;
        }
    }
    run->finished_at = now_unix();
    run->stdout_fd = -1;
    run->pid = -1;
    g_registry.save(g_index_path);
}

// Start a run from a parsed JSON config doc. Returns error string (empty=ok).
std::string start_run(yyjson_val* root, TrainRun& out_run) {
    auto get_s = [&](const char* k, const char* d) -> std::string {
        yyjson_val* v = yyjson_obj_get(root, k);
        return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string(d);
    };
    auto get_i = [&](const char* k, int d) -> int {
        yyjson_val* v = yyjson_obj_get(root, k);
        return (v && yyjson_is_num(v)) ? (int)yyjson_get_num(v) : d;
    };
    auto get_f = [&](const char* k, double d) -> double {
        yyjson_val* v = yyjson_obj_get(root, k);
        return (v && yyjson_is_num(v)) ? yyjson_get_num(v) : d;
    };
    auto get_b = [&](const char* k, bool d) -> bool {
        yyjson_val* v = yyjson_obj_get(root, k);
        return (v && yyjson_is_bool(v)) ? yyjson_get_bool(v) : d;
    };

    std::string dataset = get_s("dataset", "");
    if (dataset.empty()) return "dataset is required";

    // Build the --config json file from the requested keys, validating a few
    // against the same rules as the CLI (so the trainer rejects early).
    std::string model = get_s("model", "medium");
    std::string adapter_type = get_s("adapter_type", "dora-rows");
    int rank = get_i("rank", 16);
    double alpha = get_f("alpha", 16.0);
    double lr = get_f("learning_rate", 1e-4);
    int max_steps = get_i("max_steps", 10000);
    int frames = get_i("frames", 512);

    if (model != "medium" && model != "small-music" && model != "small-sfx")
        return "unsupported model variant: " + model;
    bool ok_adapter = (adapter_type == "lora" || adapter_type == "dora-rows" ||
                       adapter_type == "dora-cols" || adapter_type == "bora" ||
                       adapter_type == "lora-xs" || adapter_type == "dora-rows-xs" ||
                       adapter_type == "dora-cols-xs" || adapter_type == "bora-xs");
    if (!ok_adapter) return "unsupported adapter_type: " + adapter_type;
    if (rank <= 0) return "rank must be positive";
    if (alpha <= 0) return "alpha must be positive";
    if (lr <= 0) return "learning_rate must be positive";
    if (frames <= 0) return "frames must be positive";
    if (max_steps <= 0) return "max_steps must be positive";

    // Resolve output_dir.
    std::string out_dir = get_s("out", "");
    if (out_dir.empty()) {
        std::filesystem::path ds = std::filesystem::path(dataset).lexically_normal();
        std::string name = ds.filename().string();
        if (name.empty() || name == "." || name == "..") name = "sa3-lora";
        std::filesystem::path base = std::filesystem::path("train-runs") / name;
        std::filesystem::path cand = base;
        std::error_code ec;
        for (int sfx = 2; std::filesystem::exists(cand, ec); ++sfx)
            cand = std::filesystem::path(base.string() + "-" + std::to_string(sfx));
        out_dir = cand.string();
    }
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) return "cannot create output_dir: " + out_dir;
    out_dir = std::filesystem::absolute(out_dir).lexically_normal().string();

    // Write the --config json. Pass through every key the trainer understands.
    std::string cfg_path = (std::filesystem::path(out_dir) / "run.json").string();
    {
        std::ofstream f(cfg_path, std::ios::trunc);
        if (!f) return "cannot write config: " + cfg_path;
        // Forward a curated set of keys (exact names the trainer accepts).
        auto w = [&](const char* key, const std::string& val) { f << ",\"" << key << "\":\"" << json_escape(val) << "\""; };
        auto wi = [&](const char* key, int val) { f << ",\"" << key << "\":" << val; };
        auto wf = [&](const char* key, double val) { f << ",\"" << key << "\":" << val; };
        auto wb = [&](const char* key, bool val) { f << ",\"" << key << "\":" << (val ? "true" : "false"); };
        f << "{";
        w("dataset", dataset);
        w("model", model);
        w("encoding", get_s("encoding", "f16"));
        w("adapter_type", adapter_type);
        wi("rank", rank);
        wf("alpha", alpha);
        wf("learning_rate", lr);
        wf("weight_decay", get_f("weight_decay", 0.01));
        wf("adam_beta1", get_f("adam_beta1", 0.9));
        wf("adam_beta2", get_f("adam_beta2", 0.95));
        wf("adam_eps", get_f("adam_eps", 1e-8));
        wi("batch_size", get_i("batch_size", 1));
        wi("frames", frames);
        wi("max_steps", max_steps);
        wi("checkpoint_every", get_i("checkpoint_every", 500));
        wi("seed", get_i("seed", 42));
        wb("inpainting", get_b("inpainting", true));
        wf("cfg_dropout_prob", get_f("cfg_dropout_prob", 0.1));
        w("lr_scheduler", get_s("lr_scheduler", "inverse_lr"));
        wf("grad_clip", get_f("grad_clip", 1.0));
        w("timestep_sampler", get_s("timestep_sampler", "trunc_logit_normal"));
        w("dist_shift", get_s("dist_shift", "Full"));
        std::string res = get_s("resume", "");
        if (!res.empty()) w("resume", res);
        std::string svd = get_s("svd_bases", "");
        if (!svd.empty()) w("svd_bases", svd);
        f << "}";
    }

    // Spawn.
    int out_fd = -1;
    std::string perr;
    pid_t pid = spawn_train(g_train_bin, cfg_path, out_fd, perr);
    if (pid < 0) return "failed to start sa3-train: " + perr;

    TrainRun r;
    r.id = std::to_string(now_unix()) + "-" + std::to_string((long long)pid);
    r.output_dir = out_dir;
    r.config_path = cfg_path;
    r.dataset = dataset;
    r.model = model;
    r.adapter_type = adapter_type;
    r.max_steps = max_steps;
    r.started_at = now_unix();
    r.status = RunStatus::Running;
    r.pid = pid;
    r.stdout_fd = out_fd;

    g_registry.upsert(r);
    g_registry.save(g_index_path);
    out_run = r;

    std::thread(reader_loop, &g_registry.runs.back()).detach();
    return "";
}

// Return the run targeted by ?run_id=, else the active run, else null.
TrainRun* resolve_run(const httplib::Request& req) {
    if (req.has_param("run_id")) {
        TrainRun* r = g_registry.find(req.get_param_value("run_id"));
        return r;
    }
    return g_registry.active_run();
}

void send_json(httplib::Response& res, const std::string& body) {
    res.set_header("Content-Type", "application/json");
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(body, "application/json");
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = kDefaultPort;
    std::string arg_models_dir;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* d) { return i + 1 < argc ? argv[++i] : d; };
        if (a == "--host") host = next("127.0.0.1");
        else if (a == "--port") port = std::atoi(next("8016"));
        else if (a == "--models-dir") arg_models_dir = next("models");
        else if (a == "--index") g_index_path = next("train-runs/.sa3-train-web-index.json");
        else if (a == "-h" || a == "--help") {
            std::cout << "sa3-train-web [--host 127.0.0.1] [--port 8016] "
                         "[--models-dir DIR] [--index PATH]\n";
            return 0;
        }
    }

    if (const char* e = std::getenv("SA3_MODELS_DIR")) g_models_dir = e;
    if (!arg_models_dir.empty()) g_models_dir = arg_models_dir;
    if (g_models_dir.empty() && std::filesystem::is_directory("models"))
        g_models_dir = "models";

    g_train_bin = resolve_train_bin(argc > 0 ? argv[0] : "sa3-train-web");
    if (g_train_bin.empty()) {
        std::cerr << "[sa3-train-web] ERROR: could not find sa3-train binary.\n"
                  << "  Set SA3_TRAIN_BIN or place sa3-train next to this executable.\n";
        return 1;
    }
    std::cerr << "[sa3-train-web] using trainer: " << g_train_bin << "\n";

    // Default index path beside the working dir.
    if (g_index_path.empty()) {
        std::error_code ec;
        std::filesystem::path base = g_models_dir.empty()
                                         ? std::filesystem::path("train-runs")
                                         : std::filesystem::path(g_models_dir) / "train-runs";
        std::filesystem::create_directories(base, ec);
        g_index_path = (base / ".sa3-train-web-index.json").string();
    }

    // Load history (marks any prior 'running' runs as failed).
    g_registry.load(g_index_path);
    std::cerr << "[sa3-train-web] loaded " << g_registry.runs.size() << " historical run(s)\n";

    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(embedded_train_web::index_html, "text/html");
    });
    svr.Get("/train.js", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(embedded_train_web::train_js, "application/javascript");
    });

    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        bool running = g_registry.active_run() != nullptr;
        std::stringstream ss;
        ss << "{\"status\":\"ok\""
           << ",\"train_bin\":\"" << json_escape(g_train_bin) << "\""
           << ",\"running\":" << (running ? "true" : "false")
           << ",\"models_dir\":\"" << json_escape(g_models_dir) << "\"}";
        send_json(res, ss.str());
    });

    // List all runs (history + active), newest first.
    svr.Get("/api/train/runs", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        std::string out = "[";
        // newest first by started_at
        std::vector<const TrainRun*> ordered;
        for (auto& r : g_registry.runs) ordered.push_back(&r);
        std::sort(ordered.begin(), ordered.end(),
                  [](const TrainRun* a, const TrainRun* b) { return a->started_at > b->started_at; });
        for (size_t i = 0; i < ordered.size(); ++i) {
            if (i) out += ",";
            out += run_summary_json(*ordered[i]);
        }
        out += "]";
        send_json(res, out);
    });

    // Full detail for one run.
    svr.Get(R"(/api/train/runs/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        std::string id = req.matches[1];
        std::lock_guard<std::mutex> lk(g_mtx);
        TrainRun* r = g_registry.find(id);
        if (!r) { send_json(res, "{\"error\":\"run not found\"}"); res.status = 404; return; }
        std::string body = run_summary_json(*r);
        // Note: run_summary_json emits a leading '{' already; inject extras before '}'.
        body.pop_back();  // remove closing }
        body += ",\"config_path\":\"" + json_escape(r->config_path) + "\"";
        body += ",\"metrics\":" + read_metrics_array(r->output_dir, 0);
        body += ",\"artifacts\":" + artifacts_json(*r);
        body += "}";
        send_json(res, body);
    });

    // Start a new run.
    svr.Post("/api/train/start", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (g_registry.active_run() != nullptr) {
            send_json(res, "{\"error\":\"a training run is already active; stop it first\"}");
            res.status = 409;
            return;
        }
        yyjson_doc* doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (!doc || !yyjson_doc_get_root(doc) || !yyjson_is_obj(yyjson_doc_get_root(doc))) {
            send_json(res, "{\"error\":\"invalid JSON body\"}");
            res.status = 400;
            if (doc) yyjson_doc_free(doc);
            return;
        }
        TrainRun out;
        std::string err = start_run(yyjson_doc_get_root(doc), out);
        yyjson_doc_free(doc);
        if (!err.empty()) {
            send_json(res, "{\"error\":\"" + json_escape(err) + "\"}");
            res.status = 400;
            return;
        }
        std::stringstream ss;
        ss << "{\"run_id\":\"" << json_escape(out.id) << "\""
           << ",\"output_dir\":\"" << json_escape(out.output_dir) << "\"}";
        send_json(res, ss.str());
    });

    // Status of active (or ?run_id=) run.
    svr.Get("/api/train/status", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        TrainRun* r = resolve_run(req);
        if (!r) { send_json(res, "{\"running\":false}"); return; }
        std::stringstream ss;
        ss << "{\"running\":" << (r->status == RunStatus::Running ? "true" : "false")
           << ",\"run_id\":\"" << json_escape(r->id) << "\""
           << ",\"status\":\"" << run_status_str(r->status) << "\""
           << ",\"step\":" << r->latest.step
           << ",\"max_steps\":" << r->max_steps
           << ",\"progress\":" << (r->max_steps > 0 ? (double)r->latest.step / r->max_steps : 0.0)
           << ",\"loss\":" << r->latest.loss
           << ",\"lr\":" << r->latest.lr
           << ",\"grad_norm\":" << r->latest.grad_norm
           << ",\"output_dir\":\"" << json_escape(r->output_dir) << "\"}";
        send_json(res, ss.str());
    });

    // Raw log tail.
    svr.Get("/api/train/log", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        TrainRun* r = resolve_run(req);
        if (!r) { send_json(res, "{\"log\":\"\"}"); return; }
        size_t offset = 0;
        if (req.has_param("offset"))
            offset = (size_t)std::strtoull(req.get_param_value("offset").c_str(), nullptr, 10);
        if (offset > r->log.size()) offset = r->log.size();
        std::string tail = r->log.substr(offset);
        std::stringstream ss;
        ss << "{\"offset\":" << r->log.size() << ",\"log\":\"" << json_escape(tail) << "\"}";
        send_json(res, ss.str());
    });

    // Recent metrics array.
    svr.Get("/api/train/metrics", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        TrainRun* r = resolve_run(req);
        if (!r) { send_json(res, "[]"); return; }
        int limit = 0;
        if (req.has_param("limit")) limit = std::atoi(req.get_param_value("limit").c_str());
        send_json(res, read_metrics_array(r->output_dir, limit));
    });

    // Artifacts list.
    svr.Get("/api/train/artifacts", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        TrainRun* r = resolve_run(req);
        if (!r) { send_json(res, "[]"); return; }
        send_json(res, artifacts_json(*r));
    });

    // Download an artifact (or preview.wav) from any run.
    svr.Get("/api/train/download", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        TrainRun* r = resolve_run(req);
        if (!r || !req.has_param("file")) { res.status = 400; send_json(res, "{\"error\":\"run_id and file required\"}"); return; }
        std::string name = req.get_param_value("file");
        // prevent path traversal
        if (name.find("..") != std::string::npos || name.find('/') != std::string::npos) {
            res.status = 400; send_json(res, "{\"error\":\"invalid file\"}"); return;
        }
        std::filesystem::path p = std::filesystem::path(r->output_dir) / name;
        std::error_code ec;
        if (!std::filesystem::is_regular_file(p, ec)) { res.status = 404; send_json(res, "{\"error\":\"not found\"}"); return; }
        std::ifstream in(p, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::string mime = (p.extension() == ".wav") ? "audio/wav" : "application/octet-stream";
        res.set_header("Content-Type", mime.c_str());
        res.set_header("Content-Disposition",
                       ("attachment; filename=\"" + name + "\"").c_str());
        res.set_content(data, mime.c_str());
    });

    // Stop the active (or ?run_id=) run.
    svr.Post("/api/train/stop", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        TrainRun* r = resolve_run(req);
        if (!r || !r->owned_by_us()) { send_json(res, "{\"error\":\"no active run to stop\"}"); return; }
        ::kill(r->pid, SIGTERM);
        r->status = RunStatus::Stopped;  // reader thread confirms + finalizes
        g_registry.save(g_index_path);
        send_json(res, "{\"stopped\":true,\"run_id\":\"" + json_escape(r->id) + "\"}");
    });

    std::cerr << "[sa3-train-web] listening on " << host << ":" << port << "\n";
    if (!svr.listen(host.c_str(), port)) {
        std::cerr << "[sa3-train-web] failed to bind " << host << ":" << port << "\n";
        return 1;
    }
    return 0;
}
