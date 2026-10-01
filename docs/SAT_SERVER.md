# SAT model HTTP service

`sat-server` is the native HTTP entry point for Stable Audio Tools models. Start
one process per model: Foundation-1 and SAOS finetunes keep their own model cache,
port, and request contract, while all processes use the same `sa3.cpp` runtime
directory and CPU/CUDA/Vulkan backend libraries. Model GGUFs are downloaded
separately.

```powershell
.\sat-server.exe --model foundation-1 --models-dir C:\models\foundation-1 --encoding Q4_K_M --port 8015
.\sat-server.exe --model arc --models-dir C:\models\saos --encoding Q4_K_M --port 8018
# For a published SAOS finetune, use --model kickbass or --model jerry-grunge
```

`--device` selects a ggml device; without it, the pipeline uses `SA3_DEVICE`
or the best available GPU. `--t5-encoding` and `--ae-encoding` override the
DiT tier. `SA3_PORT` and `SA3_MODELS_DIR` can configure supervised launches;
explicit CLI options take precedence. `--version` prints the compiled version,
and `--props` lists available devices without loading a model or binding a port.
The default bind address is `127.0.0.1`. Each service resolves its
three GGUF files from `--models-dir` and reports a missing set through
`GET /health` (`503`, `status: model_missing`). It loads on the first generation
and retains weights for subsequent requests. `POST /unload` releases them when
the service is idle.

## Foundation-1

`POST /generate` and `POST /generate/loop` accept the same JSON body. `prompt`
is a descriptor; the server appends the exact bar, model BPM, and key tags.
The existing gary4local descriptor fields (`family`, `subfamily`, descriptor
knobs, tags, and enabled FX) can also supply the descriptor. A
`custom_prompt_override` with matching timing and key tags is accepted without
duplicating them. `randomize: true` instead chooses a descriptor and omitted
timing/key controls from the native RoyalCities prompt engine; it accepts
`randomize_mode` (`standard` or `mix`) and `family`.

```json
{"prompt":"Synth, Pad, Warm","bars":4,"bpm":128,"key_root":"C","key_mode":"minor","seed":42,"inference_profile":"gary"}
```

The bars are 4 or 8; model BPM is one of 100, 110, 120, 128, 130, 140, 150.
The server uses `resolve_foundation_timing` for the exact sample crop, whole
second duration conditioning, and padded Oobleck frame count. The default
`gary` sampler profile matches the current gary4local service; `royalcities`
is also available. `steps`, `guidance_scale`/`cfg_scale`, `sampler`,
`sigma_min`, `sigma_max`, `sigma_rho`/`rho`, and `sde_eta` can override the profile.

`host_bpm` accepts 20–999 BPM. When it falls between the model's trained tempos,
the server selects the nearest model BPM and applies a pitch-preserving offline
stretch to the final audio. An explicit `bpm` locks the model tempo. The response
reports `foundation_bpm`, `host_bpm`, and `stretch_ratio`, and the completed WAV
has the exact host bar length. The stretch uses the MIT-licensed Signalsmith
Stretch/Linear headers already built into `sat-server`; it adds no DLL to the
runtime package. Large tempo ratios can audibly degrade the result, so musical
quality should be checked in downstream host workflows.

## SAOS and finetunes

Start a separate instance with `--model arc`, `kickbass`, or `jerry-grunge`.
`POST /generate` accepts `prompt`, optional `seconds` (default 11, maximum 11),
`seed`, and sampler overrides. `POST /generate/loop` accepts `prompt`, `bpm`,
optional `bars` (1, 2, 4, or 8), and `loop_type` (`auto`, `drums`, or
`instruments`). It generates on the model's 11-second canvas and crops the WAV
to the exact number of samples for `bars × 4 × 60 / bpm`. If bars is omitted,
the longest supported count fitting 11 seconds is chosen. BPM can also come
from a `120 bpm` tag in the prompt. A selected BPM is added to the prompt if
the tag is absent. An explicit overlong loop is rejected.

```json
{"prompt":"tight funk drum break","bpm":120,"bars":4,"loop_type":"drums","seed":42}
```

SAO 1.0 can run with `--model sao1` through `/generate`; the SAOS loop route is
not offered for that model.

## Asynchronous response

Both generation routes return `success`, `session_id`, resolved `seed`,
`prompt`, `bars`, `bpm`, and `gen_duration` immediately. Poll
`GET /poll_status/<session_id>` for `status`, `progress`, `step`, and
`total_steps`; completion includes `audio_data` as base64 PCM WAV and metadata.
`?consume=1` returns and removes a completed result. One job runs per process;
a concurrent request returns `409`. Completed jobs expire after ten minutes.
This is the same polling shape as `sa3-server` and the current gary4local
Foundation service.
