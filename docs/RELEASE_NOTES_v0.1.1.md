# sa3.cpp v0.1.1

- Update ggml to `60f49e09` with validated backend correctness fixes and Vulkan
  BF16 OUT_PROD support for frozen-base training.
- Add opt-in `SA3_PRIVATE_GGML` for plugin builds. Names include the backend and
  ggml commit so older plugins cannot supply an incompatible ggml DLL in a DAW.
  Windows runtime packages retain the existing plain DLL names.
- Add a universal macOS runtime archive: Metal/CPU on Apple Silicon and CPU on
  Intel, with Developer ID signing and Apple notarization for published assets.
- Keep Windows core, CUDA, Vulkan and standalone package names/contracts.

Model weights are downloaded separately. Windows checksums are in `SHA256SUMS`;
macOS checksums are in `SHA256SUMS-macos`. Keep each archive's runtime libraries
beside its tools. See RUNTIME_RELEASE.md for installation and build details.
