# sa3.cpp v0.1.3

- Update ggml from `9d0d910b` to `4ad3b30b`, matching the shared sibling
  repository pin. Vulkan direct 2D/3D convolutions with F32 kernels now retain
  F32 staging and accumulation on GPUs with cooperative matrices. F16 kernels
  keep their existing path.
- SA3 inference/training and SAT audio graphs do not call the changed direct
  convolution operations. Existing APIs, model files, adapters and defaults are
  unchanged. Vulkan validation on the RTX 5070 Laptop passed 48/48 tests at both
  pins; 22 output files were byte-identical and three training metric files had
  identical loss and gradient-norm fields, including resumed and library training.

This patch includes all v0.1.2 native host controls and LoRA compatibility fixes.
Automation-envelope/lane work is excluded. Model weights are downloaded separately.
Windows retains split core/CUDA/Vulkan and standalone ZIPs; macOS retains the signed,
notarized universal package. See [RUNTIME_RELEASE.md](RUNTIME_RELEASE.md).
