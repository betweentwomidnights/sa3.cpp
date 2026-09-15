# libsa3 C ABI V1

`libsa3_v1.h` is the release-candidate embedding boundary for sa3.cpp. It replaces the accumulated
`sa3_init*` and `sa3_generate*` entry points with one version negotiation symbol and a stable
function table:

```c
const sa3_api_v1* api = sa3_get_api(SA3_ABI_VERSION_1);
```

A dynamically loaded host only resolves `sa3_get_api`. A statically linked host, including iOS,
uses the same call directly. `NULL` means that the requested ABI major is unavailable.

## Contract

- V1 is a C11-compatible ABI. No C++ type or exception crosses it.
- A `sa3_context` is not reentrant. A host serializes calls that use the same context.
- Strings and request audio are borrowed for the duration of the synchronous call.
- Generated audio is owned by libsa3 and must be released with `result_free`, including across a
  Windows DLL/CRT boundary.
- Initializer functions establish every default. A caller zero-initializes a struct, sets its
  `size` to `sizeof(struct)`, then calls the matching initializer. Meaningful zero values are not
  default sentinels in V1: seed 0, transform noise 0, and continuation overlap 0 remain zero.
- Public structures start with `uint32_t size`. Top-level structs, callback payloads, and strided
  array entries may only be appended, and validation uses named frozen V1 prefix sizes rather than
  a future `sizeof(struct)`. Audio, loudness, and continuation values embedded directly in a request
  are frozen for ABI V1 so growing them cannot shift later request fields. A table is likewise size-
  and major-version-tagged; functions may only be appended or consume reserved slots.
- Callbacks execute synchronously on the thread that called `generate`. Callback strings and
  progress objects are borrowed only until the callback returns.
- `sa3_error_v1` carries a typed status and a stable, copyable message. Functions also return the
  same status so an error object is optional.

## Operations

The operation describes intent, not the model plumbing:

| Operation | Input | `duration_seconds` | Exact returned duration |
| --- | --- | --- | --- |
| Generate | none | requested output length | requested length |
| Transform | required | ignored | resampled input length |
| Continue | required | seconds to add | resampled input + added length |

The library converts seconds to model frames, satisfies SAME-S frame alignment, supplies generation
or continuation tail headroom, resamples input to 44.1 kHz, performs continuation source splicing,
and trims the planar result to the promised sample count. Frontends should not repeat that logic.

Input may be planar or interleaved. Output V1 audio is always planar at the pipeline's 44.1 kHz
rate. Transform noise is in `[0, 1]`. The initialized request uses an 0.85 transform noise level,
six seconds of generation and continuation headroom, resident model loading, the LogSNR schedule,
and the tuned loudness and continuation-splice defaults.

## Adapter and model behavior

`sa3_adapter_v1` accepts either an existing path or a registry name resolved under `adapters_dir`.
Adapters are applied in request order and removed after the call. The GGUF metadata decides whether
an adapter targets the DiT, decoder, or encoder. A frontend should still keep model-family-specific
registries so it never offers a SAME-L adapter to SAME-S or a DiT adapter to the wrong base model.

Safetensors conversion is part of the V1 table because all current hosts import adapters. A null or
empty JSON sidecar means that conversion reads configuration from safetensors metadata.

## Migration gate

The V1 draft is not merged to `main` until the same sa3.cpp commit passes these consumers:

- `sa3.cpp-iplug2-demo`: Windows VST3 and REAPER extension; Generate, Transform, Continue, adapter
  import/selection, cancellation, status errors, exact duration, and DLL unload/reload.
- `sa3-ableton-extension`: the embedded C ABI backend on Windows; Generate, Transform, Continue,
  adapter import/selection, cancellation, exact duration, and native-addon unload/reload.
- `sa3.cpp-ios`: a static device build on the MacBook; context lifecycle, Generate, adapter import,
  cancellation, exact duration, audio ownership, and Training V1 on simulator and device.

## Training capability

Training is intentionally not frozen into the inference table. Hosts resolve the independent
`sa3_get_training_api(SA3_TRAINING_ABI_VERSION_1)` capability from
`libsa3_training_v1.h`. Training can therefore gain a new major without forcing inference-only
hosts to migrate. Its table initializes the shared V1 error type itself, so a training-only host
does not need to resolve the inference table.

The training table provides explicit initialized defaults, a size-tagged configuration and result,
size-tagged optimizer-step reports, cooperative cancellation, log and progress callbacks, and an
optional audio callback for sandboxed hosts that cannot let libsa3 decode dataset files itself.
Callback audio may be planar or interleaved. A tri-state result distinguishes fallback-to-libsa3,
success, and a host decode error. Successful buffers carry an opaque ownership token and libsa3
calls the paired release callback exactly once after copying, even if validation or allocation
fails. This makes buffer lifetime explicit for Swift and other managed-language hosts.

`run` is synchronous and training jobs must be serialized within a process. Both
`SA3_STATUS_OK_V1` and `SA3_STATUS_CANCELLED_V1` return a valid result. A cancelled training phase
may have produced a final adapter/checkpoint, while cancellation during pre-encode may not have.
Callers inspect `cancelled` and `final_adapter` rather than discarding the result.

The optional JSON config is applied first. Initialized scalar fields and non-null string fields in
the V1 config then override their JSON counterparts, while the JSON remains an escape hatch for
advanced trainer options not yet represented in V1.

## Proving a change kept the contract

Two tools, and they cover different halves.

`sa3-lib-v1-contract` is a CTest. It needs no models and checks the shape of the ABI: struct sizes
and frozen prefixes, what the initializers establish, strided adapter walking through a padded
stride, tail preservation for a future caller, and the error paths a host branches on — including
that an unresolvable model set is `MODEL_ERROR_V1`, and that an empty `variant` is passed through
rather than quietly becoming `medium`. Those last ones are pinned precisely because they are easy
to "improve" during a refactor and the change is silent.

`sa3-lib-v1-baseline` is the other half and needs a model set, so it is a tool rather than a test.
It runs twelve real scenarios and prints exact sample counts, every reported metadata field, and an
FNV-1a hash of the decoded audio. Capture it before a change, capture it after, diff:

```sh
export SA3_MODELS_DIR=/path/to/models
export SA3_BASELINE_DEVICE=cpu SA3_BASELINE_SECONDS=1.0 SA3_BASELINE_STEPS=2
sa3-lib-v1-baseline > before.txt      # then make the change and rebuild
sa3-lib-v1-baseline > after.txt
diff before.txt after.txt
```

Anything that re-points how a request reaches the pipeline is a change no compiler can check: a
wrong field is a different take, not a build error. This is what catches that.

**Run it on CPU.** GPU backends are not necessarily run-to-run reproducible. On an AMD Radeon Pro
5300M via Metal the same binary and the same seed hash differently every run, because a
float-ordering difference in the first sampling step compounds into a different take — so a GPU
comparison reports drift that is not there. Confirm it on your own hardware by running the tool
twice unchanged before trusting any diff from it.

## No legacy surface

`libsa3.h` and the `sa3_init*` / `sa3_generate*` / `sa3_train` entry points it declared are gone.
All three controlled frontends migrated to V1, so the shims had no consumers left, and V1 no longer
routes through the legacy request structs internally either: `generate` builds the pipeline's own
parameters directly.

That removes the last place where one field meant two things. The shared mapping used to take an
`explicit_values` flag that switched ten fields between "zero means default" and "zero means zero",
because the legacy ABI wanted the first and V1 promises the second. With one ABI there is one
answer, and the initializers are the only thing that decides a default.

It also removes the only per-call state on a context. Generation metadata used to be parked on the
context between the call and a separate getter because the legacy result struct had nowhere to
carry it; `sa3_result_v1` does, so a context now holds models and nothing else.
