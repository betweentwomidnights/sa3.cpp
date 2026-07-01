# sa3-server — HTTP over the pipeline

A small local HTTP server that wraps the same generation pipeline the CLI uses. Built by any of the
`build` scripts (it's a normal target). It's a **proof of concept that mirrors gary4local's async job
model** — `POST /generate` returns a `session_id` immediately and the client polls `/poll_status/<id>`
for progress and, on completion, the base64 audio. That makes it a drop-in for a gary4juce-style client
(SA3 on `:8006`). The reusable primitives live in the pipeline (`src/sa3_pipeline.h`, incl.
`GenParams::on_progress`); a synchronous or SSE transport is left to real apps.

## Run

```bash
./build/bin/Release/sa3-server.exe --model medium --encoding f16 --port 8086
# args: --host (default 127.0.0.1) --port (8086) --model <variant> --encoding f16|f32
#       --models-dir DIR (or SA3_MODELS_DIR) --adapters-dir DIR (or SA3_ADAPTERS_DIR)
#       --prompts-dir DIR (or SA3_PROMPTS_DIR)
#       --source-loras-dir DIR (or SA3_SOURCE_LORAS_DIR)
```

On Windows, after `.\build.cmd cuda`, this keeps the server in the terminal you can close or Ctrl+C:

```powershell
.\scripts\run-server.ps1
```

It binds to `127.0.0.1` by default (local only). The model loads lazily on the first `/generate`.

## Endpoints

| method | path | body / result |
|---|---|---|
| `GET`  | `/loras`    | `{success, loras:[{index,name,path}], adapters_dir, model_loaded}` |
| `GET`  | `/prompts`  | prompt dice pools, optionally blended with `?lora=name` or `?lora=a,b` |
| `GET`  | `/health`   | `{status, model, encoding, loaded}` (lock-free — never blocks behind a gen) |
| `POST` | `/generate` | JSON request (below) → **`{session_id, seed}`** immediately; generation runs in the background |
| `GET`  | `/poll_status/<session_id>` | `{success, generation_in_progress, progress, step, total_steps, status, queue_status, ...}`; on `status:"completed"` also `audio_data` (base64 wav) + `meta:{seed}` |
| `POST` | `/unload`   | frees the model (full VRAM release) → `{status:"unloaded"}` |

`status` runs `queued → generating → encoding → completed` (or `failed`); `progress` is `0..100`
(sampling `0→90`, decode `→100`). Poll until `status == "completed"`, then base64-decode `audio_data`.
Finished jobs are pruned after 2 min. Clients can poll `/poll_status/<id>?consume=1` to return the
completed audio once and immediately remove that job from server memory.

**`/generate` request:**
```json
{
  "prompt": "breakcore 140bpm",
  "seconds": 12,                 // or "frames": 128  (frames win if both given)
  "steps": 8,
  "seed": 0,
  "loras": [{"name": "kev", "strength": 1.0}, {"name": "keygen", "strength": 0.8}],
  "keep_models": false,          // default: frugal (free after each gen, reload next) — vst/daw-safe

  // schedule headroom (text2music). default 6.0 leaves room so the model doesn't "end" the piece
  // (full-energy tail -> good for continuation/looping); 0 lets it end (fade/silence tail):
  "duration_padding_sec": 6.0,

  // classifier-free guidance (all inert at cfg_scale=1.0, which is a single conditioned pass):
  "cfg_scale": 1.0,              // >1 or <1 runs an extra unconditioned pass per step and guides
  "negative_prompt": "low quality",
  "cfg_rescale": 0.0,           // rescale guided output toward the conditioned std (scale_phi)
  "cfg_interval_min": 0.0,      // apply CFG only when the step t is within [min, max]
  "cfg_interval_max": 1.0,
  "apg_scale": 1.0,             // 1.0 = full APG (orthogonal), 0.0 = vanilla CFG, else blend
  "cfg_norm_threshold": 0.0,    // >0 clamps the guidance-delta L2 norm

  // sampling-schedule distribution shift (optional) — mirrors the official SA3 gradio selector.
  // default "LogSNR" reproduces the prior schedule exactly. dist_shift_params overrides the 4
  // per-type params (meaning depends on type); omit it to use that type's defaults:
  "dist_shift": "LogSNR",        // "LogSNR" | "Flux" | "Full" | "None"
  "dist_shift_params": [2000, -6.2, 0.0, 2.0],  // LogSNR:(anchor_length,anchor_logsnr,rate,logsnr_end)
                                                //  Flux:(min_len,max_len,alpha_min,alpha_max)
                                                //  Full:(base_shift,max_shift,min_len,max_len)

  // audio2audio / inpaint (optional) — init_path is a LOCAL wav (the server is localhost):
  "init_path": "in.wav",         // audio2audio source; output length follows it
  "init_noise_level": 0.5,       // a2a strength (sigma_max); 1.0 == text2music
  "inpaint_start": 4.0,          // continuation/inpaint: regenerate [start,end] seconds, keep the rest
  "inpaint_end": 30.0            // also the total output duration (a short clip can extend)
}
```
LoRA `name` resolves to `<adapters-dir>/lora-<name>-*.gguf`; a full `"path"` also works. Set
`keep_models: true` to keep the model resident between requests (lower latency, more VRAM); the server
reloads a clean DiT only when a request's adapter set changes, so strengths can vary per request either way.

## LoRA and prompt discovery

`GET /loras` scans the adapters directory and returns GGUF adapter names that can be passed back in a
generation request. It also reports source `.ckpt` / `.safetensors` exports from `--source-loras-dir`
under `source_loras`; those need conversion before the C++ runtime can load them.

```json
{
  "success": true,
  "loras": [{"index": 0, "name": "kev", "path": "models/lora-kev-f32.gguf"}]
}
```

The resolver accepts:

- full adapter paths
- exact files in the adapters directory
- `lora-<name>-*.gguf`
- `<name>-*.gguf`

Source adapters can be converted to runtime GGUF adapters with:

```powershell
.venv\Scripts\python.exe tools\convert_lora.py --in loras\kev --out models\lora-kev-f32.gguf
.venv\Scripts\python.exe tools\convert_lora.py --in loras\keygen --out models\lora-keygen-f32.gguf
```

`GET /prompts` reads dice prompt pools from `--prompts-dir` / `SA3_PROMPTS_DIR`
and returns the same shape used by gary4local:

```json
{
  "success": true,
  "missing_loras": [],
  "prompts": {
    "version": 1,
    "dice": {
      "generic": ["warm analog groove"]
    }
  }
}
```

The default file is `prompts/defaults.json`. Per-adapter prompt files live beside it
as `<lora>.json`, for example `prompts/kev.json`. A client can request an adapter's
training-distribution prompts with:

```text
/prompts?lora=kev
/prompts?lora=kev,keygen
```

When a LoRA-specific prompt file contains a bucket such as `generic` or `drums`,
that bucket replaces the default bucket for the response. This lets adapter prompt
pools stay narrow and distribution-faithful without losing unrelated buckets.

## Calling it (async flow)

**PowerShell** — `curl` is an alias for `Invoke-WebRequest`, so use `Invoke-RestMethod`:
```powershell
$body = '{"prompt":"breakcore 140bpm","seconds":12,"loras":[{"name":"kev","strength":1.0}]}'
$sid = (Invoke-RestMethod http://localhost:8086/generate -Method Post -ContentType application/json -Body $body).session_id
do { Start-Sleep 1; $p = Invoke-RestMethod "http://localhost:8086/poll_status/$sid"; $p.progress } while ($p.status -ne "completed")
[IO.File]::WriteAllBytes("song.wav", [Convert]::FromBase64String($p.audio_data))
```

**Git Bash / cmd / macOS / Linux** — real `curl` (+ `jq`):
```bash
sid=$(curl -s -X POST http://localhost:8086/generate -H "Content-Type: application/json" \
  -d '{"prompt":"breakcore 140bpm","seconds":12}' | jq -r .session_id)
until [ "$(curl -s localhost:8086/poll_status/$sid | jq -r .status)" = completed ]; do sleep 1; done
curl -s localhost:8086/poll_status/$sid | jq -r .audio_data | base64 -d > song.wav
```
(To use `curl.exe` *from PowerShell*, pass the body from a file — `-d "@body.json"` — inline JSON quoting
is mangled by PowerShell's native-argument handling.)

## Residency / lifecycle

Default **frugal** (`keep_models:false`): the model is freed after each generation and reloaded on the
next request — keeps host-process memory low (good for an embedded/VST context) and makes per-request LoRA
strength correct for free. For a long-running service that wants lowest latency, send `keep_models:true`
and call `POST /unload` from your orchestrator when you need the VRAM back (model-switch, idle, pressure) —
the same pattern as the PyTorch sa3 service.

## Notes

`/generate` covers the full pipeline — text2music, LoRA, audio2audio, and inpaint/continuation. The init
audio is passed as a **local file path** (`init_path`), which is the simple, correct thing for a localhost
backend; a base64/multipart upload path could be added later if a *remote* or fully in-memory client ever
needs it. (For an in-process C-ABI alternative to the server, a `libsa3` shim is possible but deferred —
HTTP already covers the JUCE/IPlug2/gary4local consumers.)
