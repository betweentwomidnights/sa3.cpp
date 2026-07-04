# embedding sa3 in a host app (libsa3)

**some stuff we learned while building an iPlug2 vst just for sa3 medium. small won't have to worry about some of this.**

`libsa3` is a tiny C ABI over the same generation pipeline the CLI and server use, so you can call
sa3 **in-process** from a host — a JUCE / IPlug2 plugin, a game, any C or C++ program — with no CLI
subprocess and no HTTP. The header is [`src/libsa3.h`](../src/libsa3.h); the pipeline it wraps is
[`src/sa3_pipeline.h`](../src/sa3_pipeline.h).

> ✅ **validated end-to-end in a real plugin:** [sa3.cpp-iplug2-demo](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo)
> embeds `libsa3` in an IPlug2 VST3 / standalone (text2music, transform/audio2audio, continue/inpaint, LoRAs,
> in-process `.safetensors`→gguf conversion). the guidance below is what actually shook out — including the
> footguns. [`tools/sa3-libtest.c`](../tools/sa3-libtest.c) is the minimal pure-C usage example; the demo is the
> full one.

## the api

full contract is in [`src/libsa3.h`](../src/libsa3.h). The shape:

```c
sa3_config_ex cfg = {0};
cfg.config.models_dir = absolute_models_dir;
cfg.config.variant = "small-music";
cfg.config.encoding = "f16";
cfg.cpu_threads = 8;                                      // CPU backend only; 0 = SA3_THREADS/default
sa3_context* ctx = sa3_init_ex(&cfg, err, sizeof err);     // load the models (blocks ~seconds)
sa3_audio audio = {0};
sa3_generate(ctx, &req, &audio, err, sizeof err);          // prompt -> planar float samples (blocks)
// or: sa3_generate_ex(ctx, &req_ex, &audio, err, sizeof err); // a2a / inpaint + chunk/cancel controls
// ... use audio.samples (planar: samples[ch*n_samp + s]); audio.seed is the resolved seed ...
sa3_free_audio(&audio);
sa3_unload(ctx);   // optional: drop the models (free VRAM), keep ctx; next generate reloads
sa3_free(ctx);     // destroy
```

- **zero-initialize** every struct (`{0}` / `memset 0`); a `0`/`NULL` field means "default".
- **errors** don't throw — failures return `NULL`/non-zero and fill your `err` buffer.
- **not reentrant** — serialize `sa3_generate` calls on a given context (one at a time).
- **ownership** — `sa3_generate` allocates `audio.samples`; release it with `sa3_free_audio` (same
  library/CRT — don't `free()` it yourself across the DLL boundary).
- **CPU threads** — `sa3_init_ex` lets a host set `cpu_threads` directly; `sa3_init` still honors
  `SA3_THREADS` through the shared backend defaults.
- **loras** — pass adapter names (resolved in the adapters dir) or full paths via the parallel `lora_names`/
  `lora_strengths` arrays. `sa3_convert_lora(safetensors, json, out_gguf, ...)` converts a `.safetensors`
  adapter to gguf in-process (no Python) — the demo calls it on import.

use `sa3_generate_ex` for audio2audio / inpaint **and** for the chunk/cancel controls below — even for plain
text2music you'll want it (chunked decode + cooperative cancel live on `sa3_request_ex`):

```c
sa3_request_ex req = {0};
req.request.prompt = "turn this into a bright synth loop";
req.request.steps = 8;
req.request.seed = -1;                         // <0 => random; resolved seed comes back in audio.seed
req.request.keep_models = 0;                    // frugal/early-free — see below (the 8GB reality)
req.decode_chunk_size = 128; req.decode_overlap = 32;   // REQUIRED for long text2music decode

req.init_audio.mode = SA3_INIT_AUDIO_A2A;      // or SA3_INIT_AUDIO_INPAINT (omit for text2music)
req.init_audio.samples = planar;               // samples[ch * n_samp + s]
req.init_audio.n_samp = n_samp;
req.init_audio.n_ch = n_ch;
req.init_audio.sample_rate = sample_rate;      // resampled to 44.1k inside libsa3 if needed
req.init_audio.init_noise_level = 0.5f;        // <=0 uses 0.85; ~0.5 is a good mid-strength transform
req.encode_chunk_size = 128; req.encode_overlap = 32;   // REQUIRED to encode long init audio
// inpaint/continue only:  req.init_audio.inpaint_start = 8.0f; req.init_audio.inpaint_end = 30.0f;

sa3_generate_ex(ctx, &req, &audio, err, sizeof err);
```

## what you build and ship

`libsa3` is a normal build target, produced alongside the tools by the build scripts:

```bash
./build.cmd cuda      # or: cmake --build build-cuda --target sa3_shared --config Release
```

| Artifact | Role | Location (Windows / MSVC) |
|---|---|---|
| `src/libsa3.h` | the API you `#include` | in the repo |
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
auto sa3_init_fn = (decltype(&sa3_init))GetProcAddress(dll, "sa3_init");   // …and the rest
```

`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` makes `sa3.dll`'s own dependencies (`ggml*.dll`, CUDA runtime) resolve from
**its** folder — so a post-build step must copy `sa3.dll` + every `ggml*.dll` (and, for the CUDA build,
`cudart64_*.dll` / `cublas*` ) beside your binary: `MyPlugin.vst3\Contents\x86_64-win\` for the VST3, next to the
`.exe` for the standalone.

**models:** the DAW's working directory is unknown, so don't rely on the `"models"` default — pass an absolute
`sa3_config.models_dir` / `sa3_config_ex.config.models_dir` (bundle the ggufs, resolve the path at runtime,
or read an env var like `SA3_MODELS_DIR`).

## threading — the part that matters

`sa3_init` and `sa3_generate` **block for seconds**. Two hard rules:

- **never** call either from `ProcessBlock` (the audio thread) — you'll stall the DAW and get dropouts.
- **never** call `sa3_init` in the plugin constructor — the host runs that during *scan*; a multi-second load
  there makes the DAW look hung (and see the DLL note above — don't even touch `sa3.dll` at construction).

everything model-related runs on a worker thread you own; the audio thread only ever reads a finished buffer.
Lazily `sa3_init` on the worker (not the ctor), generate, then hand a planar buffer to the audio thread via a
mutex-guarded swap + an atomic flag. The demo's `RenderWorkerMain` in
[`SA3IPlug2Demo.cpp`](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo/blob/main/SA3IPlug2Demo/SA3IPlug2Demo.cpp)
is the worked example.

## `keep_models` — the 8 GB reality (use `0`)

counterintuitive but important: on a memory-tight GPU (the demo targets an **8 GB laptop RTX 5070**),
**`keep_models = 0` (frugal / early-free) is the right default in a DAW, and effectively required for long
text2music.**

- `keep_models = 0` frees T5 before sampling, the DiT before decode, and the autoencoder after decode, reloading
  freed nets (~0.5–1.5 s) on the next `generate`. This keeps peak VRAM low.
- `keep_models = 1` (resident) is snappy for short repeats, but for **long** generations it thrashes/OOMs on 8 GB —
  the decoder's SwiGLU FF activation grows with output length and overflows what's left once the other nets are
  resident (e.g. a 120 s render: ~2 s decode early-free vs tens of seconds thrashing resident). See
  [BENCHMARKS.md](BENCHMARKS.md).

so: frugal by default; only consider `keep_models = 1` for short clips on a big-VRAM card, and expose
`sa3_unload(ctx)` (e.g. when the plugin is hidden/idle) to hand VRAM back.

## chunked encode/decode — required for long audio

the sliding-window autoencoder must be **outer-chunked** for long clips or you'll hit crashes/hangs building one
giant graph. set them on `sa3_request_ex`:

- **`decode_chunk_size = 128, decode_overlap = 32`** — for *every* mode's decode (text2music included). This is
  what makes long output decode safely.
- **`encode_chunk_size = 128, encode_overlap = 32`** — for transform/continue, to encode long *init* audio.

(these are SAME-L; they're no-ops on SAME-S, which chunks internally. `0` = monolithic — fine only for short clips.)

## cancellation — so closing mid-render can't crash the host

long renders must be abortable, and plugin teardown must not `join()` a worker that's still deep in `generate`.
libsa3 supports a **cooperative cancel callback** on `sa3_request_ex`:

```c
req.should_cancel = [](void* user) -> int { return ((MyPlugin*)user)->mCancelRequested.load(); };
req.cancel_user = this;
```

`generate` polls it between steps and returns early (non-zero, no audio) when it fires. On teardown: set the
cancel flag, then `join()` the worker, then `sa3_free`. [`tools/sa3-libcancel.c`](../tools/sa3-libcancel.c) is a
small pure-C smoke test — it drives `sa3_generate_ex` with the same frugal/chunked text2music request shape the
demo uses and verifies a cooperative cancel exits cleanly without output audio:

```bash
sa3-libcancel ./models
```

## caveats you'll actually feel

- **VRAM per instance.** Each plugin instance that calls `sa3_init` loads its **own** model (several GB for
  medium). Ten instances = ten models. This host-contention reality is exactly *why* the networked path (one
  shared `sa3-server`, see [SERVER.md](SERVER.md)) exists — libsa3 is for single-instance / embedded / no-network
  cases.
- **Which backend to ship.** The CUDA `sa3.dll` needs a CUDA GPU + driver on the user's machine. For a portable
  plugin ship a **CPU or Vulkan** build instead (slower, runs anywhere) — same `sa3.dll` name, different
  `ggml-*.dll` set. 
