# sa3.cpp native runtime release

The Windows runtime follows the same split package layout as yuey.cpp. A supervisor
such as gary4local installs the core archive and one backend archive into one
directory, checks each archive against its pinned SHA-256, and launches
`sa3-server.exe`. The release contains binaries and licenses; users download
GGUF model weights separately.

| Archive | Contents |
|---|---|
| `sa3-<tag>-windows-x64-core.zip` | `sa3-server`, `sa3-train-web`, `sa3-train`, `sa3-generate`, `sat-generate`, `sa3-lora-convert`, `sa3-smoke`, `sa3.dll`, ggml core and CPU variants, project and third-party licenses, and this guide |
| `sa3-<tag>-windows-x64-cuda.zip` | `ggml-cuda.dll` for NVIDIA GPUs |
| `sa3-<tag>-windows-x64-vulkan.zip` | `ggml-vulkan.dll` for AMD, Intel, or Vulkan-capable NVIDIA GPUs |
| `cudart-<CUDA version>-windows-x64.zip` | CUDA runtime DLLs and NVIDIA EULA; needed with the CUDA backend |

All archives unpack flat into the same runtime directory, except the CUDA
runtime may live in a shared directory on `PATH`. `SHA256SUMS` accompanies the
archives. The core includes the project `LICENSE`, individual ggml, cpp-httplib,
and yyjson license files, and `THIRD_PARTY_NOTICES.md`. The CUDA runtime archive
includes NVIDIA's EULA. The core includes the optional Stable Audio Tools component in
`sa3.dll` and `sat-generate.exe`, so Stable Audio Open and Foundation-1 families
do not require a separate build.

To run both browser views from the unpacked directory:

```powershell
$env:SA3_MODELS_DIR = 'C:\path\to\models'
.\sa3-server.exe --port 8006
# In another terminal:
.\sa3-train-web.exe --port 8016
```

Open `http://127.0.0.1:8006/` for inference or `http://127.0.0.1:8016/`
for LoRA training. The pages link to one another. Training requires the matching
base DiT GGUF and a dataset. The training page lists devices detected by ggml
and lets the user select a downloaded base model tier. The server defaults to
loopback; callers that expose it to a network must supply their own access
controls.

## Build and publish

On Windows with Visual Studio 2022, CUDA Toolkit 12.8, and Vulkan SDK:

```powershell
.\ci\package-windows.ps1 -Version v0.1.0
```

For a quick local check of the core package without GPU toolchains, use
`-CpuOnly`. For a local GPU packaging check without compiling every CUDA
architecture, use `-CudaArch native -BuildDir build-dist-native -OutDir dist-native`.
Both outputs are for smoke testing and are not published releases.
The script configures `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`,
`GGML_NATIVE=OFF`, and `SA3_BUILD_SAT=ON`, runs CTest, stages the archives,
and writes SHA-256 sums. The GitHub release workflow runs the same script on
a published tag or by manual dispatch for an existing tag. The workflow uploads
the four archives and `SHA256SUMS` to that tag.

Before tagging, run a local package and model backed smoke check on CUDA and
Vulkan, inspect the staged contents and checksum file, and choose the tag.
After CI attaches the artifacts, pin the published URLs and hashes in the
gary4local native service manifest. That manifest update is a separate step;
gary4local's current sa3, Stable Audio, and Foundation services still use
their Python backends.
