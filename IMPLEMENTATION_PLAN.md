# Native ggML LoRA Training Implementation Plan

This backlog is intentionally atomic. Each item has one concrete deliverable and an expected validation path. Status is updated as work lands.

## Work Items

- [x] 01. Training configuration schema: add a C++ config struct and CLI/config-file parser for model paths, dataset path, split names, adapter type, rank, alpha, learning rate, batch size, duration/frames, precision, seed, checkpoint cadence, output directory, optimizer settings, evaluation captions, and generation smoke settings.
- [x] 02. Dataset manifest loader: read `filelist.txt` and `metadata.jsonl` for train/test/evaluation splits into typed records.
- [x] 03. Caption/audio pair discovery: resolve same-basename `.mp3` and `.txt` pairs under each split, honoring `filelist.txt`.
- [x] 04. Split validation: report missing audio, missing captions, duplicate ids, and malformed records with actionable errors.
- [x] 05. Contamination guard: reject training if any train item overlaps test/evaluation by basename, canonical path, or content hash.
- [x] 06. Non-GPU dataset tests: add tests for manifest loading, caption discovery, missing-file failures, and contamination rejection.
- [x] 07. Model path resolver reuse: resolve tokenizer, T5, conditioner, DiT, and SAME GGUFs using the existing model naming conventions from training config.
- [x] 08. MP3 decode integration: add native audio decode for dataset MP3s to planar float at the model sample rate.
- [x] 09. Audio trimming/padding: crop or pad decoded training audio to configured duration/sample size while preserving channel layout.
- [x] 10. SAME encoder wrapper: reuse `same_encode` to encode train audio into latent targets with chunk handling and host download.
- [x] 11. Caption conditioning wrapper: reuse tokenizer, T5/Gemma, conditioner padding, and seconds embedding to produce DiT conditioning tensors for each caption.
- [x] 12. Diffusion timestep/noise sampler: generate reproducible timestep, sigma, noise, noisy latent, and target velocity tensors for flow-matching training.
- [x] 13. LoRA target inventory: enumerate DiT weight tensors that can be adapted and map them to the existing GGUF adapter tensor names.
- [x] 14. Trainable adapter initialization: allocate LoRA/DoRA parameters for configured targets with correct shapes, rank, alpha, and initialization.
- [x] 15. Adapter parametrization forward path: build effective adapted weights for lora, dora-rows, dora-cols, bora, and feasible `-xs` variants in the training graph using the same math as inference.
- [x] 16. DiT training forward graph: run DiT velocity prediction using noisy latents, timestep features, cross conditioning, global conditioning, positions, and adapter-applied weights.
- [x] 17. Loss graph: compute mean squared error between DiT velocity prediction and true flow target.
- [x] 18. Backward graph: mark adapter tensors trainable and compute ggml gradients for adapter parameters only.
- [x] 19. AdamW optimizer: update adapter parameters with configurable learning rate, betas, epsilon, weight decay, and step state.
- [x] 20. Batch loop: iterate train split records, build batches, accumulate/run updates, and log real training loss.
- [x] 21. Checkpoint GGUF writer: write native trained adapter checkpoints in the exact `sa3-lora` GGUF layout consumed by `src/lora.h`.
- [x] 22. Checkpoint round-trip test: create a tiny synthetic adapter, write GGUF, load with `load_lora`, and verify metadata/tensor values.
- [x] 23. Resume/load support: load a training checkpoint as initial adapter parameters for continued training.
- [x] 24. Evaluation loss workflow: run test/evaluation split forward loss without optimizer updates and without touching train state.
- [x] 25. Generation workflow: invoke or share `sa3-generate --lora` path to render held-out captions from test/evaluation with the trained checkpoint.
- [x] 26. `sa3-train` entrypoint: add a CLI tool following `tools/sa3-*.cpp` conventions and wire it into CMake/build outputs.
- [x] 27. Training logs and output layout: write checkpoints, config snapshot, metrics JSONL, and final command transcript under the configured output directory.
- [x] 28. Documentation: document setup, build, model download, dataset layout, training command, evaluation command, outputs, and troubleshooting.
- [x] 29. Git ignore audit: ensure generated checkpoints, logs, model files, and build artifacts are ignored while source/docs/tests remain trackable.
- [x] 30. CPU/non-GPU CI build: compile default build and run non-GPU tests.
- [x] 31. CUDA build: run `build.sh cuda` and fix compile/link/runtime issues.
- [x] 32. Real train run: train on `../datasets/mnesia-audio-v1/train` using actual MP3/caption pairs and produce a GGUF adapter checkpoint.
- [x] 33. Adapter load acceptance: verify the produced checkpoint loads through the unmodified `sa3-generate --lora` path.
- [x] 34. Held-out generation acceptance: generate audio from a test/evaluation caption with `sa3-generate --lora` and record the exact command/output.
- [x] 35. Completion audit: check every evaluation prompt criterion, record proof commands/outcomes, commit final implementation, and verify clean `git status --short`.
