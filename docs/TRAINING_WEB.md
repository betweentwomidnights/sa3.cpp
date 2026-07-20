# sa3-train-web — Training Web UI Architecture

`sa3-train-web` is a standalone HTTP companion that drives the [`sa3-train`](TRAINING.md)
CLI behind a browser UI. It lets you configure, launch, monitor, and download LoRA/DoRA
training runs from a web page instead of the command line.

It is deliberately **decoupled**: it does **not** modify `sa3-server`, the training
library (`src/train_*.h`), or the trainer itself (`tools/sa3-train.cpp`). It only spawns
`sa3-train` as a subprocess, reads the artifacts that the trainer already writes, and
serves them over HTTP. This mirrors the "companion, not a fork" relationship that keeps
the inference path (`sa3-server`) untouched.

> Proof-of-concept status, same as `sa3-server`: it binds to `127.0.0.1` by default and
> has no authentication. Run it locally.

---

## System overview

```
  browser ──HTTP──►  sa3-train-web  (C++ / cpp-httplib, port 8016)
                          │  fork + execlp (no shell)
                          ▼
                     sa3-train  --config <out>/run.json
                          │  writes (trainer owns these files)
                          ▼
        <output_dir>/  metrics.jsonl        (one JSON object per optimizer step)
                       adapter-step-*.gguf  (periodic checkpoints)
                       adapter-final.gguf   (final adapter)
                       trainer-state-*.gguf (optimizer moments, for resume)
                       preview.wav          (post-training sample)
                       command.txt          (reproduction command)
                       run.json             (the config sa3-train-web wrote)
```

The child process is the single source of truth for training progress. `sa3-train-web`
never computes gradients or touches ggml — it forwards a config, tails the child's
stdout/stderr, reads `metrics.jsonl`, and serves the run directory.

---

## Components

| File | Role |
|------|------|
| `tools/sa3-train-web.cpp` | HTTP server, argument parsing, trainer discovery, subprocess spawn, endpoint handlers |
| `src/train_web_run.h` | `TrainRun` state, `RunRegistry` (multi-run history + JSON persistence), metrics parsing helpers |
| `web/train.html` | UI shell — history sidebar, config form, live detail panel |
| `web/train.js` | Vanilla-JS frontend logic (fetch, polling, sparkline, artifacts) — no build step |
| `tools/gen_embedded_train_web.py` | Generates `src/embedded_train_web.h` from the two web assets |
| `src/embedded_train_web.h` | **AUTO-GENERATED** embedded copies of `train.html` + `train.js` |

The server, `httplib`, and `yyjson` are the same vendored libraries used by `sa3-server`
(`vendor/cpp-httplib`, `vendor/yyjson`), so there is no new third-party dependency and no
Node/npm toolchain.

---

## Process model

### Trainer discovery
At startup the binary resolves the `sa3-train` executable in priority order:

1. `SA3_TRAIN_BIN` environment variable (explicit override),
2. a sibling `sa3-train` next to the running `sa3-train-web` executable (the common case
   after `./build.sh`),
3. the first `sa3-train` found on `PATH`.

If none is found, the process exits with a clear error before binding the port.

### Spawning a run
`start_run()` (in `sa3-train-web.cpp`):

1. Validates the requested config against the same rules the CLI enforces (model variant,
   adapter type, positive rank/alpha/lr/frames/steps). This gives the UI immediate,
   friendly errors instead of a child that dies on launch.
2. Resolves `output_dir`. If the client omits `out`, it derives a non-colliding
   `train-runs/<dataset-name>[-N]` directory, matching `train_finalize_defaults()` in
   `src/train_config.h`.
3. Writes `<output_dir>/run.json` — a curated JSON config whose keys map 1:1 to the
   trainer's `train_set_config_value()` accepted keys.
4. `fork()`s; the child `dup2()`s a pipe onto stdout **and** stderr, then
   `execlp(sa3-train, "--config", run.json)`. No shell is involved, so dataset paths with
   spaces or shell metacharacters are safe.
5. Registers a `TrainRun` and detaches a **reader thread** for it.

### Reader thread (`reader_loop`)
One detached `std::thread` per run:

- Drains the child pipe (non-blocking read) into an in-memory log buffer, capped at
  ~1 MiB (`g_log_tail_max`) so a long run cannot exhaust host memory.
- Periodically re-reads the **last line** of `metrics.jsonl` and parses it into the run's
  `latest` sample (step, lr, loss, grad_norm) for cheap `/status` responses.
- On child EOF, `waitpid()`s and maps the exit into a terminal status:
  `Completed` (exit 0), `Stopped` (killed by `SIGTERM`), or `Failed` (anything else),
  then persists the registry.

### Concurrency policy
Only **one run may be `Running` at a time**. This is intentional: the target hardware
(4 GB GTX 960M) cannot host two training graphs, and even on larger GPUs concurrent
training runs would contend for VRAM. `POST /api/train/start` returns HTTP `409` while a
run is active. Finished/stopped/failed runs remain in the registry as browsable history.

---

## Multi-run history & persistence

`RunRegistry` (in `src/train_web_run.h`) holds every run this instance knows about and
serialises a lightweight index to disk:

- **Index path:** `<models_dir>/train-runs/.sa3-train-web-index.json` by default, or an
  explicit `--index PATH`.
- **Persisted per run:** `id`, `output_dir`, `config_path`, `dataset`, `model`,
  `adapter_type`, `max_steps`, `started_at`, `finished_at`, `status`. The heavy data
  (metrics series, artifacts, logs) is **not** duplicated in the index — it is read back
  from each run's `output_dir` on demand.
- **Boot recovery:** on startup the index is loaded and any run left in `running`/`queued`
  by a previous crash is downgraded to `failed` (its child is no longer owned by this
  process). Its artifacts and metrics remain fully browsable.

Run identity is `"<unix_seconds>-<pid>"`, unique per launch.

> **Implementation note:** the reader thread holds a pointer to its `TrainRun` inside the
> registry's `std::vector`. Because only one run is ever active — and a new run cannot
> start until the previous one has left the `Running` state — the active element is stable
> in practice. If the concurrency model is ever relaxed to allow parallel runs, switch the
> registry to a container with pointer stability (e.g. `std::deque` or
> `std::vector<std::unique_ptr<TrainRun>>`).

---

## HTTP API

All JSON responses include `Access-Control-Allow-Origin: *`. Endpoints that operate on a
run accept an optional `?run_id=<id>`; without it they target the current **active** run.

| Method | Path | Body / Query | Result |
|--------|------|--------------|--------|
| `GET`  | `/` | — | embedded `train.html` |
| `GET`  | `/train.js` | — | embedded `train.js` |
| `GET`  | `/api/health` | — | `{status, train_bin, running, models_dir}` |
| `POST` | `/api/train/start` | JSON config | `{run_id, output_dir}`; `409` if a run is active; `400` on invalid config |
| `GET`  | `/api/train/runs` | — | array of run summaries, newest first |
| `GET`  | `/api/train/runs/<id>` | — | run summary + full `metrics` array + `artifacts` |
| `GET`  | `/api/train/status` | `?run_id=` | `{running, status, step, max_steps, progress, loss, lr, grad_norm, output_dir}` |
| `GET`  | `/api/train/log` | `?run_id=&offset=N` | `{offset, log}` — raw child output since byte `N` |
| `GET`  | `/api/train/metrics` | `?run_id=&limit=N` | array of `metrics.jsonl` objects (last `N`) |
| `GET`  | `/api/train/artifacts` | `?run_id=` | array of `{name, size, is_wav}` |
| `GET`  | `/api/train/download` | `?run_id=&file=` | streams the file (`audio/wav` for `.wav`, else octet-stream) |
| `POST` | `/api/train/stop` | `?run_id=` | `SIGTERM`s the active child → `{stopped, run_id}` |

### Start request body
Keys map directly to `sa3-train` config keys. `dataset` is required; everything else has
the same defaults as the CLI's validated recipe.

```json
{
  "dataset": "/path/to/dataset",
  "model": "medium",                    // medium | small-music | small-sfx
  "encoding": "f16",
  "adapter_type": "dora-rows",          // lora|dora-rows|dora-cols|bora (+ -xs variants)
  "rank": 16,
  "alpha": 16,
  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8,
  "batch_size": 1,
  "frames": 512,
  "max_steps": 10000,
  "checkpoint_every": 500,
  "seed": 42,
  "inpainting": true,
  "cfg_dropout_prob": 0.1,
  "lr_scheduler": "inverse_lr",
  "grad_clip": 1.0,
  "timestep_sampler": "trunc_logit_normal",
  "dist_shift": "Full",
  "resume": "",                          // optional: path to trainer-state-*.gguf
  "svd_bases": "",                       // optional: precomputed SVD bases for -xs adapters
  "out": ""                              // optional: explicit output dir (auto if empty)
}
```

### metrics.jsonl schema (produced by `sa3-train`)
One line per optimizer step; `sa3-train-web` reads these verbatim:

```json
{"epoch":0,"update":42,"split":"train","id":"clip123","t":0.53,"cfg_drop":0,
 "mask":"inpaint","n_gen":128,"n_ctx":384,"lr":9.87e-05,"loss":1.2345,"grad_norm":0.87}
```

`/api/train/status` surfaces `update`→`step`, plus `lr`, `loss`, `grad_norm`.

---

## Frontend

`web/train.html` + `web/train.js` are hand-written vanilla JavaScript with **no build
step**, exactly like `web/app.js` for the inference server (loaded via a plain
`<script src="train.js">`). Conventions: ES2023+, `const`/`let` only, `async`/`await` with
`try`/`catch`, optional chaining/nullish coalescing.

Layout:

- **Sidebar** — the run history list (newest first). Each row shows dataset, model,
  adapter type, a status badge, step progress, and a progress bar. Clicking selects a run.
- **Config form** — opened by "+ New training"; disabled while a run is active. Submits to
  `POST /api/train/start`, then auto-selects the new run.
- **Detail panel** — status/step/loss/lr/grad-norm stat cards, a progress bar, a
  dependency-free `<canvas>` sparkline (loss on a linear axis, learning rate on a log
  axis), a live raw-log pane (incremental via `offset`), and an artifacts list. `.gguf`
  files are download links; `preview.wav` also renders an inline `<audio controls>` player.
- **Polling** — while a run is selected, the UI polls `/api/train/runs` +
  `/api/train/status` + `/api/train/metrics` + `/api/train/log` every ~1.5 s. Historical
  (non-running) runs display their final metrics without live polling churn.

Editing the UI: change `web/train.html` / `web/train.js`, then regenerate the embedded
header and rebuild (see below). Do **not** edit `src/embedded_train_web.h` by hand.

---

## Build & run

`sa3-train-web` is a normal CMake target (links `sa3`, `Threads`, and compiles the
vendored `httplib.cpp` + `yyjson.c`). It is built by any backend build:

```bash
./build.sh cpu      # or: cuda | vulkan | hip | metal
```

Regenerate the embedded web assets after any `web/train.*` change:

```bash
python3 tools/gen_embedded_train_web.py
./build.sh cpu      # rebuild to embed the new assets
```

Run it (defaults shown):

```bash
./build/bin/sa3-train-web --host 127.0.0.1 --port 8016 --models-dir models
# optional: --index PATH   (registry/history location)
# optional: SA3_TRAIN_BIN=/path/to/sa3-train  (override trainer discovery)
```

Then open `http://127.0.0.1:8016`. The default port is **8016**, distinct from
`sa3-server`'s 8006, so both can run side by side.

### CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--host` | `127.0.0.1` | bind address (local only by default) |
| `--port` | `8016` | listen port |
| `--models-dir` | `models` (or `$SA3_MODELS_DIR`) | model directory; also anchors the default index path |
| `--index` | `<models_dir>/train-runs/.sa3-train-web-index.json` | run-history registry file |

---

## Design decisions

- **Companion, not integration.** Keeping training out of `sa3-server` means the inference
  service (and its VRAM/latency profile) is never affected by a long training job, and the
  two evolve independently.
- **Subprocess over linking.** Spawning `sa3-train` (rather than linking the training
  library) isolates crashes and OOMs in the child, gives a natural cancel primitive
  (`SIGTERM`), and reuses the trainer's existing, tested config/checkpoint/resume paths
  verbatim.
- **Files as the contract.** The trainer already writes `metrics.jsonl` and GGUF
  artifacts; the web app reads them. There is no private IPC channel to keep in sync, and
  the same files remain usable directly from the CLI or `sa3-generate`.
- **No new toolchain.** Vanilla JS + vendored C++ HTTP means no Node, npm, or bundler.

---

## Limitations & future work (v1)

- Single active run only (no run queue). A queue could hold pending configs and start the
  next automatically on completion.
- No dataset validation report in the UI (train/test/eval overlap is still checked by
  `sa3-train` itself and surfaced through the log).
- No authentication or TLS — localhost only, matching `sa3-server`.
- The log buffer is capped in memory; the full child output is not persisted to disk by
  the web app (though `sa3-train`'s own outputs and `metrics.jsonl` are complete on disk).
- Resume is supported via the `resume` config key, but there is no dedicated
  checkpoint-picker UI yet.

---

## Verification status (2026-07-20)

- Builds and links on CPU; `--help` accurate.
- Live subprocess lifecycle validated with a stand-in trainer: `start` → `running` with
  live `step`/`loss`/`lr`/`grad_norm` from `metrics.jsonl`, incremental log streaming via
  `offset`, and `stop` → `SIGTERM` → `stopped`.
- Multi-run history validated by booting with a seeded index: `/api/train/runs`,
  `/api/train/runs/<id>` (metrics + artifacts), and `/api/train/download` (including WAV
  streaming) all return correct data across a restart.
- A real end-to-end GPU training run additionally requires a dataset and the matching
  `*-base` DiT GGUF (see [TRAINING.md](TRAINING.md)).
