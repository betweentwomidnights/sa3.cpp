# sa3-server — HTTP over the pipeline

A small local HTTP server that wraps the same generation pipeline the CLI uses. Built by any of the
`build` scripts (it's a normal target). Meant to be spawned by a host app (a JUCE/IPlug2 plugin, a
gary4local-style backend) which then POSTs to it — the host never touches the CLI.

## Run

```bash
./build/bin/Release/sa3-server.exe --model medium --encoding f16 --port 8086
# args: --host (default 127.0.0.1) --port (8086) --model <variant> --encoding f16|f32
#       --models-dir DIR (or SA3_MODELS_DIR) --adapters-dir DIR (or SA3_ADAPTERS_DIR)
```

It binds to `127.0.0.1` by default (local only). The model loads lazily on the first `/generate`.

## Endpoints

| method | path | body / result |
|---|---|---|
| `GET`  | `/health`   | `{status, model, encoding, loaded}` |
| `POST` | `/generate` | JSON request (below) → **`audio/wav`** bytes (or `{error}` + 4xx/5xx) |
| `POST` | `/unload`   | frees the model (full VRAM release) → `{status:"unloaded"}` |

**`/generate` request:**
```json
{
  "prompt": "breakcore 140bpm",
  "seconds": 12,                 // or "frames": 128  (frames win if both given)
  "steps": 8,
  "seed": 0,
  "loras": [{"name": "kev", "strength": 1.0}, {"name": "keygen", "strength": 0.8}],
  "keep_models": false           // default: frugal (free after each gen, reload next) — vst/daw-safe
}
```
LoRA `name` resolves to `<adapters-dir>/lora-<name>-*.gguf`; a full `"path"` also works. Set
`keep_models: true` to keep the model resident between requests (lower latency, more VRAM); the server
reloads a clean DiT only when a request's adapter set changes, so strengths can vary per request either way.

## Calling it (platform-correct)

**PowerShell** — `curl` is an alias for `Invoke-WebRequest`, so use `Invoke-RestMethod`:
```powershell
$body = '{"prompt":"breakcore 140bpm","seconds":12,"loras":[{"name":"kev","strength":1.0}]}'
Invoke-RestMethod http://localhost:8086/generate -Method Post -ContentType application/json -Body $body -OutFile song.wav
```

**Git Bash / cmd / macOS / Linux** — real `curl`:
```bash
curl -s -X POST http://localhost:8086/generate -H "Content-Type: application/json" \
  -d '{"prompt":"breakcore 140bpm","seconds":12}' -o song.wav
```
(To use `curl.exe` *from PowerShell*, pass the body from a file — `-d "@body.json"` — inline JSON quoting
is mangled by PowerShell's native-argument handling.)

## Residency / lifecycle

Default **frugal** (`keep_models:false`): the model is freed after each generation and reloaded on the
next request — keeps host-process memory low (good for an embedded/VST context) and makes per-request LoRA
strength correct for free. For a long-running service that wants lowest latency, send `keep_models:true`
and call `POST /unload` from your orchestrator when you need the VRAM back (model-switch, idle, pressure) —
the same pattern as the PyTorch sa3 service.

## Not yet over HTTP

`/generate` currently does text2music + LoRA. audio2audio / inpaint work in the pipeline but need the init
WAV uploaded (multipart or base64) — a follow-up. For a C-ABI in-process alternative, see `libsa3` (planned).
