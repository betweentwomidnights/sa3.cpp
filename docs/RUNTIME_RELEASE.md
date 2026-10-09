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
.\ci\package-windows.ps1 -Version v0.1.2
# Add a separate cudart zip for testing gary4local's shared-runtime install:
.\ci\package-windows.ps1 -Version v0.1.2 -CudaRuntime
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
3. **Publish:** tag and publish the verified commit as `v0.1.2`. The release
   event runs the same package script on that tag, attests each zip, and
   attaches the four zips and `SHA256SUMS`. A dispatch with an existing tag
   can replace missing assets after a runner failure, before a consumer pins it.
4. **Pin and test:** verify the published checksums and attestation, then pin
   the service assets in gary4local's native service manifest. Its shared CUDA
   runtime comes from gary-localhost-installer. Test installation from those
   published URLs before switching the Python services over.

The current gary4local SA3, Stable Audio, and Foundation services still run
their Python backends; this package prepares the native install path.

## macOS universal package

`sa3-<tag>-macos-universal.zip` contains the servers, generation/training tools,
`libsa3.dylib`, ggml dylibs, licenses, model downloader and prompt pools. It is
flat at the root. Keep the libraries beside the tools. Apple Silicon uses Metal
or CPU; Intel uses CPU (AVX2 baseline, macOS 13.3 or newer). Model weights are
separate. `SHA256SUMS-macos` covers this archive without conflicting with the
Windows checksum asset.

Build on Apple Silicon with Xcode command line tools, CMake and Rosetta:

```sh
./ci/package-macos.sh --version v0.1.2 --jobs 3
```

The script builds each architecture separately, runs CTest (Intel under Rosetta
on macOS 15+), merges matching binaries with lipo, rewrites bundled library
imports to `@loader_path`, and checks server startup and the ABI contract from
outside the package directory. The Metal dylib is arm64 only. Hosted CI checks
compilation and packaging; real Metal inference must also be tested on a Mac.

Release builds require Developer ID signing and notarization. Set these
repository Actions secrets, matching the stems.cpp workflow:

- `MACOS_CERT_P12`: base64 of the exported Developer ID Application certificate
  **and its private key** (.p12).
- `MACOS_CERT_PASSWORD`: export password.
- `APPLE_TEAM_ID`: certificate's developer team ID.
- `APPLE_NOTARY_KEY_P8`: base64 of the App Store Connect API private key (.p8).
- `APPLE_NOTARY_KEY_ID` and `APPLE_NOTARY_ISSUER_ID`: notarization key metadata.

The original certificate/key can be reused across sibling repositories. Store
copies in each repository; GitHub cannot return existing secret values. No new
Apple key or certificate is needed solely because the repository is different.
The workflow uses a temporary keychain, signs every bundled executable and
library with hardened runtime and a timestamp, and requires an Accepted notary
submission before uploading. Bare binaries in a zip cannot be stapled.

Dispatch `release.yml` with `platforms=macos` for a macOS-only dry run, or
`platforms=all` for both platforms. Dry runs work without credentials and keep
ad-hoc signatures; tagged release runs fail if signing credentials are missing.
Local signing uses `SA3_SIGN_IDENTITY`, `SA3_NOTARY_KEY` (path to .p8),
`SA3_NOTARY_KEY_ID`, and `SA3_NOTARY_ISSUER`; pass `--require-signing` for release
packages. Publish v0.1.2 only after the dry runs and Mac inference checks pass.


### CPU audio analysis

The Windows core and macOS tool list include `sa3-audio-analyze`, a standalone
CPU metadata helper. It has no model/backend dependencies; `--control-info`
reports schema 1. See `AUDIO_ANALYSIS.md`. It reads WAVs directly. Hosts decoding
compressed inputs still need a decoder such as FFmpeg; FFmpeg is not bundled in
these core packages. The Windows packaging script checks the staged helper's
capabilities before creating the archive.

## Offline host controls

`sa3-server --control-info` returns schema 1, service/version, and the same
capabilities advertised by `/health`: `fixed_prefix`, `request_splice`,
`conditioning_duration`, and `model_lifecycle`. It exits before reading `.env`,
initializing backends, loading models, or binding ports. Hosts should require
these capabilities rather than relying on a version string alone. SAT has its
own API and does not advertise the SA3 control contract.

Both packaging scripts check the staged servers' versions/devices, SA3 server
controls, cooperative trainer cancellation/atomic progress, and CPU audio
analysis from an empty working directory. These checks also run with
`-SkipTests` / `--skip-tests`. Python 3 is required on the build machine for
package verification; the installed runtime remains native. macOS checks each
runnable universal slice after library rewriting/signing. Use
`python ci/check-runtime-package.py <unpacked-core> <tag>` to repeat the check
on a downloaded package (add `--arch arm64` or `--arch x86_64` on macOS).

These controls are included in v0.1.2; the published v0.1.1 assets do not
satisfy the complete migration contract. Hosts must check capabilities before
enabling migration.

## Windows build provenance

Windows `BUILD-INFO.json` distinguishes `portable`, `cpu-smoke` and `gpu-smoke`
builds and records the CUDA architecture policy. Local `-CpuOnly` or
`-CudaArch native` packages are smoke artifacts, even when their version matches
a release tag. Omitting `-CudaArch` clears a cached architecture override before
configuration, restoring ggml's portable defaults rather than silently reusing
a previous native-only build. The release workflow uses that portable path.

Windows external projects use a short directory under the build root. This
keeps Vulkan's shader helper compiler probes and MSBuild tracking paths out of
the deeply nested default layout without modifying the ggml submodule.
