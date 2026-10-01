# sa3.cpp native runtime release

The Windows release follows the yuey.cpp package contract. Gary4local installs
the core archive and one backend archive, verifies their SHA-256 hashes, and
uses its own shared CUDA runtime. Direct users take one standalone archive with
both backends and the CUDA runtime included. Model weights are downloaded
separately.

| Archive | Contents |
|---|---|
| `sa3-<tag>-windows-x64-core.zip` | Servers, training and generation tools, `sa3.dll`, ggml core and CPU variants, licenses, `BUILD-INFO.json`, and documentation; for gary4local |
| `sa3-<tag>-windows-x64-cuda.zip` | `ggml-cuda.dll` for NVIDIA GPUs |
| `sa3-<tag>-windows-x64-vulkan.zip` | `ggml-vulkan.dll` for AMD, Intel, or Vulkan-capable NVIDIA GPUs |
| `sa3-<tag>-windows-x64-standalone.zip` | Core, both GPU backends, CUDA runtime and EULA, `models.cmd`, default prompt pools, and startup guide; for direct use |

The archives are flat at their root. Gary4local unpacks core plus its selected
backend into one directory and makes its separately installed CUDA runtime
available on `PATH` when needed. `SHA256SUMS` covers the four release archives.
The core contains `sa3-server.exe`, `sat-server.exe`, `sat-generate.exe`,
`sa3-train-web.exe`, the other tools, and individual third-party licenses.
Both servers answer `--version` and `--props` before loading models; the package
script rejects a tag that disagrees with their compiled version. For supervised
launches, `SA3_PORT`, `SA3_MODELS_DIR`, and `SA3_DEVICE` set the port, model
directory, and ggml backend in each server process.

The same core and backend files can serve SA3, Foundation-1, and SAOS on
separate ports. Each SAT model gets its own `sat-server` process, keeping its
weight cache and model-specific API separate without installing a second CUDA
runtime. See [SAT_SERVER.md](SAT_SERVER.md) for endpoints and host-tempo
stretch behavior.

To run both browser views from the unpacked directory with one command:

```powershell
$env:SA3_MODELS_DIR = 'C:\path\to\models'
.\studio.ps1 -Port 8006 -TrainPort 8016
```

Open `http://127.0.0.1:8006/` for inference or `http://127.0.0.1:8016/`
for LoRA training. The pages link to one another. You can also launch
`sa3-server.exe` or `sa3-train-web.exe` alone when only one service is needed.
Training requires the matching
base DiT GGUF and a dataset. The training page lists devices detected by ggml
and lets the user select a downloaded base model tier. The server defaults to
loopback; callers that expose it to a network must supply their own access
controls.

## Build and publish

On Windows with Visual Studio 2022, CUDA Toolkit 12.8, and Vulkan SDK:

```powershell
.\ci\package-windows.ps1 -Version v0.1.0
# Add a separate cudart zip for testing gary4local's shared-runtime install:
.\ci\package-windows.ps1 -Version v0.1.0 -CudaRuntime
```

For a quick local check of the core package without GPU toolchains, use
`-CpuOnly`. For a local GPU packaging check without compiling every CUDA
architecture, use `-CudaArch native -BuildDir build-dist-native -OutDir dist-native`.
Both outputs are for smoke testing and are not published releases.
The script configures `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`,
`GGML_NATIVE=OFF`, and `SA3_BUILD_SAT=ON`, runs CTest, checks both servers'
versions, stages the archives, and writes SHA-256 sums. Its `BUILD-INFO.json`
records the service, version, source and ggml commits, toolchains, and whether
local tracked files differed from the commit.

## Release checks

1. **Dry run:** dispatch `release.yml` with no tag, on the intended branch:
   `gh workflow run release.yml -R betweentwomidnights/sa3.cpp --ref main`.
   CI builds the full portable package and retains the four zips plus
   `SHA256SUMS` as a workflow artifact for 14 days. It creates no release.
2. **Install test:** download that artifact, check `SHA256SUMS`, and test the
   standalone zip directly. Build locally with `-CudaRuntime` when testing
   gary4local's `GARY4LOCAL_NATIVE_PACKAGE_DIR` override: that folder then
   contains the split service packages and a shared-runtime test zip. Exercise
   both CUDA and Vulkan and inspect `--props` from the unpacked core.
3. **Publish:** tag and publish the verified commit as `v0.1.0`. The release
   event runs the same package script on that tag, attests each zip, and
   attaches the four zips and `SHA256SUMS`. A dispatch with an existing tag
   can replace missing assets after a runner failure, before a consumer pins it.
4. **Pin and test:** verify the published checksums and attestation, then pin
   the service assets in gary4local's native service manifest. Its shared CUDA
   runtime comes from gary-localhost-installer. Test installation from those
   published URLs before switching the Python services over.

The current gary4local SA3, Stable Audio, and Foundation services still run
their Python backends; this package prepares the native install path.
