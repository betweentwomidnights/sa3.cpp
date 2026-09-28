# sa3.cpp v0.1.0 — release notes draft

The first sa3.cpp runtime release packages the local Stable Audio 3 runtime and
its browser tools for downstream applications. The project release version is
`v0.1.0`; the embedded C interface remains ABI V1.

## Included

- Stable Audio 3 inference for medium, small-music, and small-sfx, including
  text-to-music, transform, continuation, and inpainting.
- Native LoRA-family inference and training, with resumable training
  checkpoints.
- `libsa3` with the stable C ABI V1 for embedding inference and training in
  downstream applications.
- The local inference studio (`sa3-server`) and training studio
  (`sa3-train-web`), with pillopaus-project's authorship and UI credit retained.
- The optional Stable Audio Tools runtime for Stable Audio Open Small, Stable
  Audio Open 1.0, and Foundation-1.
- Split Windows x64 runtime archives for the portable CPU core, CUDA backend,
  Vulkan backend, and CUDA runtime, with SHA-256 checksums. Model weights are
  downloaded separately.

## Scope and limitations

- These release archives target Windows x64. Other platforms can build from
  source; see the backend-specific guides.
- HIP/ROCm has not been validated on real hardware. See [the HIP notes](HIP.md).
- The browser servers bind to loopback by default and do not provide
  authentication. Keep them local unless you provide network access controls.
- Model weights have their own licenses and download terms and are not included
  in the runtime archives.

This draft should be finalized against the exact tagged commit and the artifacts
produced by the release workflow before publishing.
