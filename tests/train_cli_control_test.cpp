#include "train_cli_control.h"
#include <cstdio>
#include <fstream>
#include <limits>

static void expect(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

int main() {
    namespace fs = std::filesystem;
    sa3::TrainCliControl control;
    std::vector<char*> args;
    std::string err;
    char a0[] = "trainer", a1[] = "--progress-file", a2[] = "progress.json";
    char a3[] = "--steps", a4[] = "3", a5[] = "--cancel-file", a6[] = "cancel.requested";
    char* argv[] = {a0,a1,a2,a3,a4,a5,a6};
    expect(control.parse(7, argv, args, err), "parse host controls");
    expect(args.size() == 3 && std::string(args[1]) == "--steps", "preserve training arguments");
    char* missing[] = {a0,a1};
    sa3::TrainCliControl invalid; args.clear();
    expect(!invalid.parse(2, missing, args, err), "missing control path refused");
    char* duplicate[] = {a0,a1,a2,a1,a2};
    invalid = {}; args.clear();
    expect(!invalid.parse(5, duplicate, args, err), "duplicate control path refused");
    char* same[] = {a0,a1,a2,a5,a2};
    invalid = {}; args.clear();
    expect(!invalid.parse(5, same, args, err), "progress cannot trigger its own cancellation");

#ifdef _WIN32
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = getpid();
#endif
    const fs::path root = fs::temp_directory_path() / ("sa3-train-cli-control-" + std::to_string(pid));
    expect(!fs::exists(root), "test must have isolated storage");
    fs::create_directories(root);
    control.progress_file = (root / "progress.json").u8string();
    control.cancel_file = (root / "cancel.requested").u8string();
    expect(!control.should_cancel(), "cancel is initially absent");
    control.message = "quotes \" and unicode \xc3\xa9";
    control.loss = std::numeric_limits<double>::infinity();
    control.publish();
    control.step = 2; control.status = "running"; control.publish();
    std::ifstream file(root / "progress.json");
    const std::string data((std::istreambuf_iterator<char>(file)), {});
    yyjson_doc* doc = yyjson_read(data.data(), data.size(), 0);
    expect(doc != nullptr, "atomic replacement remains valid JSON");
    yyjson_val* value = yyjson_doc_get_root(doc);
    expect(yyjson_get_int(yyjson_obj_get(value,"step")) == 2, "latest progress wins");
    expect(std::string(yyjson_get_str(yyjson_obj_get(value,"message"))) == control.message, "JSON escaping preserves message");
    expect(yyjson_is_null(yyjson_obj_get(value,"loss")), "nonfinite metrics are explicitly unavailable");
    yyjson_doc_free(doc); file.close();
    std::ofstream(root / "cancel.requested") << "stop";
    expect(control.should_cancel(), "cancel file observed without removing it");
    for (const auto& item : fs::directory_iterator(root))
        expect(item.path().filename().string().find(".tmp-") == std::string::npos, "staging files retired");
    fs::remove_all(root);
    std::puts("PASS native trainer file controls");
}
