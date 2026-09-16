# embedding sa3 in a host app (libsa3)

**some stuff we learned while building an iPlug2 vst just for sa3 medium. small won't have to worry about some of this.**

`libsa3` is a tiny C ABI over the same generation pipeline the CLI and server use, so you can call
sa3 **in-process** from a host — a JUCE / IPlug2 plugin, a game, any C or C++ program — with no CLI
subprocess and no HTTP. The header is [`src/libsa3_v1.h`](../src/libsa3_v1.h) (plus
[`src/libsa3_training_v1.h`](../src/libsa3_training_v1.h) if you train); the pipeline it wraps is
[`src/sa3_pipeline.h`](../src/sa3_pipeline.h). The contract and its growth rules are in
[`docs/C_ABI_V1.md`](C_ABI_V1.md).

> ✅ **validated end-to-end in a real plugin:** [sa3.cpp-iplug2-demo](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo)
> embeds `libsa3` in an IPlug2 VST3 / standalone (text2music, transform/audio2audio, continue/inpaint, LoRAs,
> in-process `.safetensors`→gguf conversion). the guidance below is what actually shook out — including the
> footguns. [`tools/sa3-libtest.c`](../tools/sa3-libtest.c) is the minimal pure-C usage example; the demo is the
> full one.

## the api

One symbol to resolve, one table to call. The full contract is in
[`src/libsa3_v1.h`](../src/libsa3_v1.h) and [`docs/C_ABI_V1.md`](C_ABI_V1.md). The shape:

```c
const sa3_api_v1* api = sa3_get_api(SA3_ABI_VERSION_1);   // NULL = this build has no V1
if (!api) return;

sa3_context_config_v1 cfg = {0};
cfg.size = sizeof cfg;
api->context_config_init(&cfg);                  // defaults in, THEN your overrides
cfg.models_dir = absolute_models_dir;
cfg.variant = "small-music";
cfg.dit_encoding = "f16";
cfg.text_encoder_encoding = "f16";               // NULL = auto (prefers F16); see DISTRIBUTION.md
cfg.cpu_threads = 8;                             // CPU backend only; 0 = SA3_THREADS/default

sa3_error_v1 err = {0}; err.size = sizeof err; api->error_init(&err);
sa3_context* ctx = NULL;
if (api->context_create(&cfg, &ctx, &err) != SA3_STATUS_OK_V1) { /* err.message */ }

sa3_request_v1 req = {0}; req.size = sizeof req; api->request_init(&req);
req.operation = SA3_OPERATION_GENERATE_V1;       // or TRANSFORM / CONTINUE
req.prompt = "warm analog house groove";
req.duration_seconds = 12.0;                     // exactly what comes back

sa3_result_v1 out = {0}; out.size = sizeof out; api->result_init(&out);
api->generate(ctx, &req, &out, &err);            // planar float; out.seed is the resolved seed
// ... use out.samples (planar: samples[ch * n_samples + s]) ...
api->result_free(&out);                          // library-owned; never free() it yourself
api->context_unload(ctx);                        // optional: drop models, keep ctx; next call reloads
api->context_destroy(ctx);
```

- **zero it, size it, initialize it.** Every public struct starts with `uint32_t size`. Set it,
  call the matching `*_init`, then set your fields. The initializer is what establishes defaults —
  **a zero you leave behind is a zero you asked for**, not a request for the default. Seed `0` is
  the seed zero; transform noise `0` is no noise; continuation overlap `0` is no overlap.
- **errors** don't throw. Every call returns a `sa3_status_v1`, and `sa3_error_v1` carries the same
  status plus a copyable message. The error object is optional — the return value is enough.
- **not reentrant** — serialize `generate` calls on a given context.
- **ownership** — `generate` allocates `out.samples`; release it with `result_free`, including
  across a Windows DLL/CRT boundary. It is safe on a zeroed or already-freed result.
- **adapters** — one strided array, applied in order and removed after the call. Each entry takes a
  registry name (resolved under `adapters_dir`) or a full path; the gguf's own metadata decides
  whether it lands on the DiT, decoder or encoder. `convert_lora` turns a `.safetensors` adapter
  into a gguf in-process, no Python.
- **the library owns duration planning.** Seconds in, seconds out: it converts to model frames,
  satisfies SAME-S alignment, supplies generation and continuation headroom, resamples input to
  44.1 kHz, splices the continuation source, and trims to the promised sample count. Don't
  reimplement any of that in the host — that was the single biggest source of drift in the
  frontends before V1.

Transform and Continue take input audio, planar or interleaved, at any sample rate:

```c
sa3_request_v1 req = {0}; req.size = sizeof req; api->request_init(&req);
req.operation = SA3_OPERATION_CONTINUE_V1;   // or TRANSFORM
req.prompt = "turn this into a bright synth loop";
req.duration_seconds = 6.0;                  // CONTINUE: seconds to ADD. TRANSFORM ignores this.
req.residency = SA3_RESIDENCY_FRUGAL_V1;     // early-free — see below (the 8GB reality)
req.transform_noise_level = 0.5f;            // TRANSFORM only; ~0.5 is a good mid-strength
req.decode_chunk_size = 128; req.decode_overlap = 32;   // REQUIRED for long decode
req.encode_chunk_size = 128; req.encode_overlap = 32;   // REQUIRED to encode long input

req.input_audio.size = sizeof req.input_audio;
req.input_audio.samples = planar;            // samples[ch * n_samples + s]
req.input_audio.n_samples = n_samples;       // per channel
req.input_audio.n_channels = n_channels;
req.input_audio.sample_rate = sample_rate;   // resampled to 44.1k inside libsa3
req.input_audio.layout = SA3_AUDIO_PLANAR_V1;   // or SA3_AUDIO_INTERLEAVED_V1

api->generate(ctx, &req, &out, &err);
```

`sa3_result_v1` also reports what the run measured and applied — decoded peak, the loudness gains,
and for a continuation the splice bounds and gain. That used to be unavailable at the ABI entirely.

## training

Training runs a whole LoRA/DoRA job in-process, through the same code the CLI uses
([`src/train_job.h`](../src/train_job.h)). An adapter it writes is **byte-identical** to one
`sa3-train` writes with the same config — verified against the CLI on every change.

It lives in its own table, resolved separately from inference, so it can gain a major version
without forcing inference-only hosts to migrate. That table initializes the shared error type
itself, so a training-only host never resolves the inference table at all.

```c
const sa3_training_api_v1* train = sa3_get_training_api(SA3_TRAINING_ABI_VERSION_1);

sa3_training_config_v1 cfg = {0};
cfg.size = sizeof cfg;
train->config_init(&cfg);
cfg.dataset_dir = "/path/to/dataset";      // laid out as docs/TRAINING.md describes
cfg.output_dir  = "/path/to/run";
cfg.variant     = "small-music";
cfg.dit_encoding = "q4_k_m";                // quantized bases train on every backend
cfg.text_encoder_encoding = "f16";          // NULL = auto; F16 halves the encoder for free
cfg.steps       = 2000;
cfg.frames      = 128;                      // ~11.9 s crops; 512 is the reference regime
cfg.seed        = 42;
cfg.checkpoint_every = 0;                   // 0 = no intermediate checkpoints

sa3_training_callbacks_v1 hooks = {0};
hooks.size = sizeof hooks;
train->callbacks_init(&hooks);
hooks.on_log        = on_log;               // one formatted line at a time
hooks.on_step       = on_step;              // per-update loss / grad norm / lr
hooks.should_cancel = should_cancel;        // non-zero to stop at the next sample
hooks.user          = self;

sa3_training_result_v1 res = {0}; res.size = sizeof res; train->result_init(&res);
sa3_error_v1 err = {0}; err.size = sizeof err; train->error_init(&err);

const sa3_status_v1 status = train->run(&cfg, &hooks, &res, &err);
// BOTH OK and CANCELLED return a valid result: a cancelled training phase has usually still
// written a final adapter. Inspect res.cancelled and res.final_adapter rather than discarding it.
if (status != SA3_STATUS_OK_V1 && status != SA3_STATUS_CANCELLED_V1) { /* err.message */ }
// res.final_adapter is the .gguf you now pass to generate as an adapter
```

- **it blocks** for the length of the run (hours). Call it on a worker thread; the callbacks are
  invoked on that thread.
- **cancel is cooperative** — `should_cancel` is checked at each sample boundary, and the run still
  writes a checkpoint pair and a final adapter before returning, so `cfg.resume_path` picks it back
  up exactly where it stopped. `res.cancelled` tells you which way it ended.
- **`config_path`** takes a JSON file of any train-config key and is applied *before* the struct
  fields, so the struct stays small without capping what a host can set.
- **the result feeds inference directly** — put `res.final_adapter` in a `sa3_adapter_v1` entry and
  pass that entry through `sa3_request_v1.adapters`.

### training without a filesystem or ffmpeg

Audio normally comes off disk: WAV already at 44.1 kHz is read natively, and anything else is
decoded by shelling out to `ffmpeg`. A sandboxed host (iOS, a locked-down plugin) can do neither, so
`hooks.load_audio` lets it hand over planar float PCM per item instead:

```c
static sa3_training_audio_status_v1 SA3_CALL load_audio(
        void* user, const char* audio_path, uint32_t sample_rate, uint32_t channels,
        sa3_training_audio_buffer_v1* out, sa3_error_v1* err) {
    struct host* h = (struct host*)user;
    if (!decode_with_the_platform(h, audio_path, sample_rate, channels))
        return SA3_TRAINING_AUDIO_NOT_HANDLED_V1;          // libsa3 reads the file itself
    out->audio.size = sizeof out->audio;
    out->audio.samples = h->planar;                        // or interleaved; say which below
    out->audio.n_samples = h->n_samples;                   // per channel
    out->audio.n_channels = channels;
    out->audio.sample_rate = sample_rate;
    out->audio.layout = SA3_AUDIO_PLANAR_V1;
    out->owner = h->buffer_handle;                          // opaque; libsa3 never inspects it
    return SA3_TRAINING_AUDIO_READY_V1;
}

static void SA3_CALL release_audio(void* user, void* owner) { free_buffer(user, owner); }
```

Three answers, not two. `NOT_HANDLED` falls back to the file, so supplying audio for only part of a
dataset works. `ERROR` stops the run and may fill `err` with a typed failure. `READY` transfers
ownership temporarily: libsa3 copies the samples and then calls `release_audio` **exactly once** —
including when validation or the copy fails — so the lifetime is explicit rather than implied by a
callback's stack frame. Install `load_audio` and `release_audio` together; one without the other is
rejected.

The dataset directory itself is still read normally (manifests, captions), and checkpoints and
`metrics.jsonl` are still written under `output_dir` — a host that wants training needs somewhere
writable, just not a decoder.

[`tools/sa3-libtrain.c`](../tools/sa3-libtrain.c) is the runnable pure-C example.

## what you build and ship

`libsa3` is a normal build target, produced alongside the tools by the build scripts:

```bash
./build.cmd cuda      # or: cmake --build build-cuda --target sa3_shared --config Release
```

| Artifact | Role | Location (Windows / MSVC) |
|---|---|---|
| `src/libsa3_v1.h` | the API you `#include` | in the repo |
| `sa3.dll` | runtime lib you **ship** (+ load at runtime — see below) | `build-<backend>/bin/Release/sa3.dll` |
| `ggml*.dll` | ggml backends `sa3.dll` needs | same `bin/Release/` (`ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda`, …) |
| `sa3.lib` | import lib (only if you link at load time) | `build-<backend>/Release/sa3.lib` |
| the gguf model set | weights loaded at runtime | any folder; pass its **absolute** path |

On Unix/macOS the library is `libsa3.so` / `libsa3.dylib` (ship it + the `libggml*` set).

## load the DLL yourself — don't import it at module load

The footgun: if the plugin **statically imports** `sa3.dll` (links `sa3.lib`), strict VST3 hosts fail the plugin
during scan when they can't resolve `sa3.dll` + all the `ggml*.dll` yet — the plugin never even shows up. The
demo instead **loads `sa3.dll` on demand at render time** and resolves the functions with `GetProcAddress`:

```cpp
// from beside your own module (works for both the VST3 bundle and the standalone .exe)
HMODULE self = nullptr;
GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                   (LPCWSTR)&some_function_in_your_module, &self);
wchar_t path[1024]; GetModuleFileNameW(self, path, 1024);
std::wstring dir(path); dir.resize(dir.find_last_of(L"\\/"));
HMODULE dll = LoadLibraryExW((dir + L"\\sa3.dll").c_str(), nullptr,
                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
auto get_api = (decltype(&sa3_get_api))GetProcAddress(dll, "sa3_get_api");   // the only one you need
const sa3_api_v1* api = get_api ? get_api(SA3_ABI_VERSION_1) : NULL;         // everything else is in the table
```

`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` makes `sa3.dll`'s own dependencies (`ggml*.dll`, CUDA runtime) resolve from
**its** folder — so a post-build step must copy `sa3.dll` + every `ggml*.dll` (and, for the CUDA build,
`cudart64_*.dll` / `cublas*` ) beside your binary: `MyPlugin.vst3\Contents\x86_64-win\` for the VST3, next to the
`.exe` for the standalone.

**models:** the DAW's working directory is unknown, so don't rely on the `"models"` default — pass an absolute
`sa3_context_config_v1.models_dir` (bundle the ggufs, resolve the path at runtime,
or read an env var like `SA3_MODELS_DIR`).

## threading — the part that matters

`context_create` and `generate` **block for seconds**. Two hard rules:

- **never** call either from `ProcessBlock` (the audio thread) — you'll stall the DAW and get dropouts.
- **never** call `context_create` in the plugin constructor — the host runs that during *scan*; a multi-second load
  there makes the DAW look hung (and see the DLL note above — don't even touch `sa3.dll` at construction).

everything model-related runs on a worker thread you own; the audio thread only ever reads a finished buffer.
Lazily `context_create` on the worker (not the ctor), generate, then hand a planar buffer to the audio thread via a
mutex-guarded swap + an atomic flag. The demo's `RenderWorkerMain` in
[`SA3IPlug2Demo.cpp`](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo/blob/main/SA3IPlug2Demo/SA3IPlug2Demo.cpp)
is the worked example.

## model residency — the 8 GB reality (use frugal)

counterintuitive but important: on a memory-tight GPU (the demo targets an **8 GB laptop RTX 5070**),
**`SA3_RESIDENCY_FRUGAL_V1` (early-free) is the right default in a DAW, and effectively required for long
text2music.**

- `SA3_RESIDENCY_FRUGAL_V1` frees T5 before sampling, the DiT before decode, and the autoencoder after decode, reloading
  freed nets (~0.5–1.5 s) on the next `generate`. This keeps peak VRAM low.
- `SA3_RESIDENCY_RESIDENT_V1` is snappy for short repeats, but for **long** generations it thrashes/OOMs on 8 GB —
  the decoder's SwiGLU FF activation grows with output length and overflows what's left once the other nets are
  resident (e.g. a 120 s render: ~2 s decode early-free vs tens of seconds thrashing resident). See
  [BENCHMARKS.md](BENCHMARKS.md).

so: set `req.residency = SA3_RESIDENCY_FRUGAL_V1` by default; only consider resident mode for short
clips on a big-VRAM card, and expose `api->context_unload(ctx)` (e.g. when the plugin is hidden/idle)
to hand VRAM back.

## chunked encode/decode — required for long audio

the sliding-window autoencoder must be **outer-chunked** for long clips or you'll hit crashes/hangs building one
giant graph. set them on `sa3_request_v1`:

- **`decode_chunk_size = 128, decode_overlap = 32`** — for *every* mode's decode (text2music included). This is
  what makes long output decode safely.
- **`encode_chunk_size = 128, encode_overlap = 32`** — for transform/continue, to encode long *init* audio.

(these are SAME-L; they're no-ops on SAME-S, which chunks internally. `0` = monolithic — fine only for short clips.)

## cancellation — so closing mid-render can't crash the host

long renders must be abortable, and plugin teardown must not `join()` a worker that's still deep in `generate`.
libsa3 supports a **cooperative cancel callback** on `sa3_request_v1`:

```c
req.should_cancel = [](void* user) -> int { return ((MyPlugin*)user)->mCancelRequested.load(); };
req.callback_user = this;
```

`generate` polls it between steps and returns `SA3_STATUS_CANCELLED_V1` with no audio when it fires. On
teardown: set the cancel flag, then `join()` the worker, then `api->context_destroy(ctx)`.
[`tools/sa3-libcancel.c`](../tools/sa3-libcancel.c) is a
small pure-C smoke test — it drives `generate` with the same frugal/chunked request shape the
demo uses and verifies a cooperative cancel exits cleanly without output audio:

```bash
sa3-libcancel ./models
```

## caveats you'll actually feel

- **VRAM per instance.** Each plugin instance that calls `context_create` loads its **own** model (several GB for
  medium). Ten instances = ten models. This host-contention reality is exactly *why* the networked path (one
  shared `sa3-server`, see [SERVER.md](SERVER.md)) exists — libsa3 is for single-instance / embedded / no-network
  cases.
- **Which backend to ship.** The CUDA `sa3.dll` needs a CUDA GPU + driver on the user's machine. For a portable
  plugin ship a **CPU or Vulkan** build instead (slower, runs anywhere) — same `sa3.dll` name, different
  `ggml-*.dll` set. 
