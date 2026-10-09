#include "train_cli_control.h"
#include <cstdio>
#include <fstream>
#include <limits>
#include <thread>
#include <chrono>

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
#ifdef _WIN32
    // Windows readers (including scanners) can briefly deny rename/delete.
    // Such a reader must not abort an otherwise healthy training run.
    HANDLE reader = CreateFileW((root / "progress.json").c_str(), GENERIC_READ,
                               FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    expect(reader != INVALID_HANDLE_VALUE, "open a reader without delete sharing");
    std::thread release_reader([reader]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        CloseHandle(reader);
    });
    control.step = 3;
    std::string publish_error;
    try { control.publish(); } catch (const std::exception& e) { publish_error = e.what(); }
    release_reader.join();
    if (!publish_error.empty()) std::fprintf(stderr, "%s\n", publish_error.c_str());
    expect(publish_error.empty(), "transient Windows reader lock must not fail training progress");
    expect(SetFileAttributesW((root / "progress.json").c_str(), FILE_ATTRIBUTE_READONLY) != 0,
           "make destination permanently non-replaceable");
    control.step = 4;
    publish_error.clear();
    try { control.publish(); } catch (const std::exception& e) { publish_error = e.what(); }
    expect(SetFileAttributesW((root / "progress.json").c_str(), FILE_ATTRIBUTE_NORMAL) != 0,
           "restore destination permissions");
    expect(!publish_error.empty(), "permanent permission failure remains an error after bounded retries");
    std::ifstream preserved(root / "progress.json");
    const std::string preserved_data((std::istreambuf_iterator<char>(preserved)), {});
    yyjson_doc* preserved_doc = yyjson_read(preserved_data.data(), preserved_data.size(), 0);
    expect(preserved_doc && yyjson_get_int(yyjson_obj_get(yyjson_doc_get_root(preserved_doc), "step")) == 3,
           "failed publication preserves the previous complete progress");
    yyjson_doc_free(preserved_doc); preserved.close();
#endif
    std::ofstream(root / "cancel.requested") << "stop";
    expect(control.should_cancel(), "cancel file observed without removing it");
    for (const auto& item : fs::directory_iterator(root))
        expect(item.path().filename().string().find(".tmp-") == std::string::npos, "staging files retired");
    fs::remove_all(root);
    std::puts("PASS native trainer file controls");
}
