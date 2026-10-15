# SAT model HTTP service

`sat-server` is the native HTTP entry point for Stable Audio Tools models. Start
one process per model family: Foundation-1 and SAOS (with its finetunes) keep
their own model cache, port, and request contract, while all processes use the
same `sa3.cpp` runtime directory and CPU/CUDA/Vulkan backend libraries. Model
GGUFs are downloaded separately.

```powershell
.\sat-server.exe --model foundation-1 --models-dir C:\models\foundation-1 --encoding Q4_K_M --port 8015
.\sat-server.exe --model saos --models-dir C:\models\saos --encoding Q4_K_M --port 8005
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
the service is idle. Foundation-1 and SAO 1.0 generate
[asynchronously](#asynchronous-response); SAOS answers each request directly.

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

`POST /randomize` returns a whole preset from the same native prompt engine,
split into the controls a plugin shows, without loading the model. It keeps
gary4local's Python request and response contract: the request takes `seed`
(`-1` or omitted draws one below 2^31), `mode` (`standard` or `mix`),
`variant` (`auto`, `M1`, or `T1`, where `T1` means mix) and `family_hint`.
The response carries `family` and `subfamily`; `descriptor_knob_a` to `_c`
and `descriptors_extra` (timbre); `spatial_tags`, `band_tags`,
`wave_tech_tags` and `style_tags`; `reverb_amount`, `delay_type`,
`distortion_amount`, `phaser_amount` and `bitcrush_amount`, each with an
`_enabled` flag; `speed`, `structure`, `rhythm`, `contour` and `density`
(with `behavior_tags` holding all of them); `all_tags`; and the joined
descriptor `prompt`. The same seed always gives the same preset.

`host_bpm` accepts 20–999 BPM. When it falls between the model's trained tempos,
the server selects the nearest model BPM and applies a pitch-preserving offline
stretch to the final audio. An explicit `bpm` locks the model tempo. The response
reports `foundation_bpm`, `host_bpm`, and `stretch_ratio`, and the completed WAV
has the exact host bar length. The stretch uses the MIT-licensed Signalsmith
Stretch/Linear headers already built into `sat-server`; it adds no DLL to the
runtime package. Large tempo ratios can audibly degrade the result, so musical
quality should be checked in downstream host workflows.

`POST /audio2audio` transforms input audio toward a Foundation prompt, keeping
gary4local's Python contract. It takes the `/generate` body plus `audio_data` (a
base64 WAV) and `host_bpm`, both required; `bars` defaults to 8 and `prompt` is
accepted as the descriptor. `init_noise_level` (0.01–1, default 0.25) says how far
to move from the input: it is the noise the input latent starts from and the
schedule's `sigma_max`. The input is cropped or zero-padded to the bars at the host
tempo, stretched to the model BPM, encoded, and the result is stretched back to the
host tempo like any `/generate` output. The response and polling add `mode:
"audio2audio"` and `init_noise_level`.

The schedule has to descend from the init noise level. When the profile's
`sigma_min` is at or above it (the `gary` profile's 0.5 sits above most of
gary4juce's range), the request uses RoyalCities' 0.01 floor instead; Python's
service ran the rising schedule, which returns NaN. An explicit `sigma_min` at or
above the level is refused. Audio-to-audio needs
`stable-audio-open-oobleck-encoder-v1.0-F16.gguf` beside the model files, from the
Foundation-1 GGUF repository; without it the route answers `503` and generation is
unaffected.

## SAOS and finetunes

One SAOS instance (`--model saos`) serves the ARC base model and every published
finetune whose DiT is in `--models-dir` at the server's encoding, beside the
shared T5 and Oobleck files: `finetunes/kickbass/` and `finetunes/jerry-grunge/`,
as in the GGUF repository. `--model arc`, `kickbass`, or `jerry-grunge` starts
the same server with that variant active. The variants share the T5 and
autoencoder, so one DiT is resident at a time and switching swaps only the DiT.

The routes keep gary4local's Python stable-audio contract, which gary4juce's
jerry tab drives. A request names its model the way it named a checkpoint
before: `model_type: "standard"` for ARC, or `"finetune"` with `finetune_repo`
and `finetune_checkpoint` naming the checkpoint the GGUF was converted from:

| variant | `finetune_repo` | `finetune_checkpoint` |
|---|---|---|
| `kickbass` | `S3Sound/kickbass` | `kickbass_v1_saos_e257_s18870.ckpt` |
| `jerry-grunge` | `thepatch/jerry_grunge` | `jerry_encoded_bs64_epoch=374-step=3000.ckpt` |

Generation is synchronous. With `return_format: "base64"` the response is
`{success, audio_base64, metadata}`; the default, `"file"`, returns the 16-bit
WAV. The audio is scaled to a full-scale peak, as Python's was. `steps` (1–250)
and `cfg_scale` (0–20) default per model, to 8 and 1 for ARC and 50 and 4 for the
rectified-flow finetunes. `sampler_type` is `pingpong` (ARC's default), `euler`
(the finetunes' default) or `dpmpp`. A drawn seed stays below 2^32.

`POST /generate` takes `prompt`, `negative_prompt` and `seconds_total` (whole
seconds, default and maximum 11). Generation uses the whole canvas, conditions
on `seconds_total`, and crops the output to it. The metadata carries `prompt`,
`negative_prompt`, `steps`, `cfg_scale`, `seed`, `sample_rate`,
`duration_seconds`, `generation_time`, `realtime_factor`, `detected_bpm` and
`device`, plus `model`, `model_key` and `sampler`.

`POST /generate/loop` needs a BPM, from a `120bpm` tag in the prompt (or a
`bpm` field, which adds the tag). `bars` (1, 2, 4, or 8) defaults to the longest
count that leaves a second of the 11.89-second canvas spare; an explicit count
longer than the canvas is refused. `loop_type` is `auto`, `drums` (adds "drum
loop" when the prompt names no drums, and a negative prompt against melody), or
`instruments` (removes "drums" from the prompt, and adds a negative prompt
against percussion). The WAV is cropped to exactly `bars × 4 × 60 / bpm`. The
metadata adds `original_prompt`, `loop_type`, `bars`, `loop_duration_seconds`,
`calculated_duration_seconds`, `seconds_per_bar` and `model_type`.

```json
{"prompt":"tight funk drum break 120bpm","model_type":"standard","steps":8,"cfg_scale":1,"sampler_type":"pingpong","loop_type":"drums","return_format":"base64","seed":-1}
```

`GET /models/status` lists the installed variants in `model_details`, keyed as
Python keyed its model cache (`standard_saos`, or
`finetune_<repo>_<checkpoint>` with `/` and `.` as `_`), each with its `type`
and `source` (`stabilityai/stable-audio-open-small`, or
`<repo>/<checkpoint>`). `cache_status.usage_order` lists them least recently
used first, so the last is the active one. `GET /models/prompts` returns a
finetune's dice prompt bank from `finetunes/<variant>/prompts.json`, chosen by
`?key=`, by `?repo=` and `?checkpoint=`, or else the active variant (`?prefer=finetune`
prefers the most recent finetune). A variant without a bank, or with an
unreadable one, answers Python's empty bank.

SAO 1.0 can run with `--model sao1` through `/generate`; the SAOS loop route is
not offered for that model.

## Asynchronous response

For Foundation-1 and SAO 1.0, both generation routes return `success`,
`session_id`, resolved `seed`, `prompt`, `bars`, `bpm`, and `gen_duration`
immediately. Poll
`GET /poll_status/<session_id>` for `status`, `progress`, `step`, and
`total_steps`; completion includes `audio_data` as base64 PCM WAV and metadata.
`?consume=1` returns and removes a completed result. One job runs per process;
a concurrent request returns `409`. Completed jobs expire after ten minutes.
This is the same polling shape as `sa3-server` and the current gary4local
Foundation service.
