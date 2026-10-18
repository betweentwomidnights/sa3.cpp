# CLI usage and configuration

For the shortest build → download → generate path, start with the
[README quickstart](../README.md#quickstart). This guide keeps the extra model,
adapter, configuration, and compatibility details in one place.

## Build and backends

Builds need CMake and a C++17 compiler. On Windows, use Visual Studio 2022 with
the C++ tools installed. CUDA needs the CUDA Toolkit; Vulkan needs the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home); Metal is macOS-only.

`build.cmd BACKEND` and `./build.sh BACKEND` write each backend to its own build
directory, so builds can coexist. The helpers also generate `env.cmd` / `env.ps1`
on Windows or `env.sh` on macOS/Linux to put that build's tools on your shell's
PATH and point `SA3_MODELS_DIR` at this checkout's models folder.

Use `env.cmd` in Command Prompt, `. .\env.ps1` in PowerShell, or
`source ./env.sh` in bash. Available build choices include `cpu`, `cuda`,
`vulkan`, `metal`, `hip`, and `all`. Vulkan also supports AMD GPUs; HIP/ROCm
remains unvalidated. Backend-specific
requirements and measurements are in [Vulkan](VULKAN.md), [Metal](METAL.md),
[HIP](HIP.md), and [generation benchmarks](BENCHMARKS.md).

The build helpers enable the Stable Audio Tools component with
`SA3_BUILD_SAT=ON`. With direct CMake configuration, that component is opt-in.
Its model downloads and CLI examples live in the
[Stable Audio Open Small](STABLE_AUDIO_OPEN_SMALL.md) and
[Stable Audio Open 1.0 / Foundation-1](STABLE_AUDIO_OPEN_1.md) guides;
[SAT_SERVER.md](SAT_SERVER.md) covers its separate HTTP service.

## Models and precision

`--model medium`, `--model small-music`, and `--model small-sfx` resolve a GGUF
model set in `models/` or `--models-dir DIR`. The explicit
`--tok`, `--t5`, `--cond`, `--dit`, and `--same` paths still work and override
individual components. See [model distribution](DISTRIBUTION.md) for naming and
the per-model components.

Use `--duration SEC` for an exact text-to-audio output length, or `--frames N`
for the lower-level latent length. Do not combine them. Transform and
continuation lengths come from the init audio and inpaint endpoint instead.

Precision can be chosen separately for the DiT, text encoder, and autoencoder.
For example, this downloads a quantized DiT and training base while keeping the
default F32 autoencoder:

```bash
./models.sh --encoding q4_k_m --training-base
```

To quantize the autoencoder too, add `--ae-encoding q4_k_m`. On Windows, use
`models.cmd` with the same flags. Pass the corresponding `--encoding`,
`--t5-encoding`, and `--ae-encoding` choices to the generation tool; their
defaults and model-resolution rules are documented in [DISTRIBUTION.md](DISTRIBUTION.md).

## Adapters

`--lora NAME` resolves `lora-NAME-*.gguf` in the adapters directory, which defaults
to the models directory. A file path also works. Repeat `--lora` to compose
adapters; `--lora-strength` sets the strength of the preceding adapter.

```bash
sa3-generate --model medium --lora kev --lora-strength 0.7 --lora keygen --prompt "neo-classical lofi hiphop, 90 BPM, C# minor" --out song.wav
```

These example adapters must already be installed. See the [adapter folder guide](../loras/README.md)
and [training guide](TRAINING.md) for importing and creating adapters. The
[automation lanes API](SERVER.md#lora-automation-lanes-lora_lanes-lora_envelope)
documents time-varying strengths, lane prompts, and the CLI envelope flags.

## Transform and continue

Transform an existing WAV, keeping its length:

```bash
sa3-generate --model medium --init source.wav --init-noise-level 0.65 --prompt "warm piano with soft tape saturation" --out transformed.wav
```

For a 10-second source, add a 30-second continuation by regenerating up to the
40-second endpoint:

```bash
sa3-generate --model medium --init source.wav --inpaint-start 10 --inpaint-end 40 --prompt "continue the groove with a brighter melody" --out continued.wav
```

The pipeline applies its continuation mask overlap and source splice. See
[CONTINUATION.md](CONTINUATION.md) for the splice controls and how the kept
source is restored.

Transform uses the upstream init-noise schedule by default. Pass
`--legacy-schedule` to compare with the former C++ shift-then-scale schedule.
The server accepts `"legacy_schedule": true`, and libsa3 V1 has the optional
`legacy_schedule` request field. This changes the schedule; transform canvas
padding and output length keep the same behavior.

## Configuration

The tools read a `.env` in the working directory; see [`.env.example`](../.env.example).
Precedence is **flag > environment variable > `.env` > default**. Downstream
applications can set environment variables in the process they launch.

| Setting | Environment variable | Corresponding flag | Default |
|---|---|---|---|
| Model GGUFs | `SA3_MODELS_DIR` | `--models-dir` | `models/` |
| Adapters | `SA3_ADAPTERS_DIR` | `--adapters-dir` | Models directory |
| Source adapter exports | `SA3_SOURCE_LORAS_DIR` | `--source-loras-dir` | `loras/` |
| Prompt dice pools | `SA3_PROMPTS_DIR` | `--prompts-dir` | `prompts/` |
| Compute controls | `SA3_DEVICE`, `SA3_GPU`, `SA3_THREADS`, `SA3_FLASH_ATTN` | `--threads` for CPU threads | Automatic device selection |

Flags vary by tool; source-export and prompt-pool flags are server controls.
Output processing is documented in [LOUDNESS.md](LOUDNESS.md), and backend
tuning in [BENCHMARKS.md](BENCHMARKS.md).

## Updating an existing checkout

The one-time ggml URL migration and exact-pin update commands are documented in
[GGML_FORK.md](GGML_FORK.md#updating-existing-clones-and-downstream-forks). That
guide also explains the `not our ref` error caused by a cached submodule URL.
