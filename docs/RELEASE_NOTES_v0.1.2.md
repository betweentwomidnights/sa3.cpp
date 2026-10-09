# sa3.cpp v0.1.2

- Update ggml to `9d0d910b`, adding CPU and Metal im2col fast paths. The
  inference and training APIs retain their existing defaults.
- Add optional fixed-prefix continuation, per-request splice controls and
  independent conditioning duration to the unified SA3 generation endpoint.
  Completed results include measured splice and latent-length metadata.
- Add explicit model load/reload/unload and readiness controls. Lifecycle
  changes reject requests while generation is running or queued.
- Add model-free host capability probes, atomic trainer progress and cooperative
  cancellation. Windows progress publication tolerates transient reader locks.
- Fix the LoRA merge graph allocation that could abort with three full-scope
  DoRAs. Resolve legacy PyTorch `dora` metadata during conversion and when
  loading existing GGUFs; Gary's historical row normalization is preserved.
- Restrict legacy checkpoint export to tensor-only PyTorch loading, and add
  a native CPU WAV BPM/key analysis tool for dataset sidecars.
- Verify staged server/trainer/analyzer contracts in Windows and macOS
  packages. Windows packaging records portable versus local smoke builds and
  avoids excessively nested Vulkan helper build paths.

The existing standalone inference/training Studio and C ABI remain available.
Automation-envelope/lane changes are separate work and are not in this release.
Gary4local-specific request wrappers, storage migration and cleanup belong to
the host application.

Model weights are downloaded separately. Windows checksums are in `SHA256SUMS`;
macOS checksums are in `SHA256SUMS-macos`. Windows retains split core/CUDA/Vulkan
and standalone packages; macOS retains the signed, notarized universal archive.
Keep each archive's runtime libraries beside its tools. See
[RUNTIME_RELEASE.md](RUNTIME_RELEASE.md) for installation and build details.
