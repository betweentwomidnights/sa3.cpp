# Native LoRA Training Acceptance Record

Date: 2026-07-04

## Build

Command:

```sh
./build.sh cuda
```

Outcome:

```text
[sa3] done -> build-cuda/bin/
```

## Tests

Command:

```sh
build-cuda/bin/sa3-train-config-test && build-cuda/bin/sa3-train-dataset-test && build-cuda/bin/sa3-train-model-paths-test && build-cuda/bin/sa3-train-audio-test ../datasets/mnesia-audio-v1/test/audio/single-fluke.mp3 && build-cuda/bin/sa3-train-same-compile-test && build-cuda/bin/sa3-train-conditioning-compile-test && build-cuda/bin/sa3-train-diffusion-test && build-cuda/bin/sa3-train-lora-test && build-cuda/bin/sa3-train-dit-compile-test && build-cuda/bin/sa3-train-optimizer-test && build-cuda/bin/sa3-train-loop-compile-test && build-cuda/bin/sa3-train-checkpoint-test && build-cuda/bin/sa3-train-generation-test
```

Outcome:

```text
train_config_test: ok
train_dataset_test: ok
train_model_paths_test: ok
train_audio_test: ok (6048000 samples, 2 ch)
train_same_compile_test: ok
train_conditioning_compile_test: ok
train_diffusion_test: ok
train_lora_test: inventory ok
train_dit_compile_test: ok
train_optimizer_test: ok
train_loop_compile_test: ok
train_checkpoint_test: ok
train_generation_test: ok
```

## Real Training Run

Command:

```sh
SA3_DEVICE=cpu build-cuda/bin/sa3-train --model medium --encoding f32 --models-dir models --dataset ../datasets/mnesia-audio-v1 --adapter-type lora --rank 1 --alpha 1 --learning-rate 0.0001 --frames 2 --batch-size 1 --checkpoint-every 1 --seed 42 --out train-runs/mnesia-native-medium-f32-r1-f2-final
```

Outcome:

```text
train 1/38 update=1 id=album-causality-01-conflicted loss=0.158849
...
train 38/38 update=38 id=single-where-were-you loss=0.165949
checkpoint: train-runs/mnesia-native-medium-f32-r1-f2-final/adapter-step-38.gguf
final checkpoint: train-runs/mnesia-native-medium-f32-r1-f2-final/adapter-final.gguf
```

Final checkpoint:

```text
train-runs/mnesia-native-medium-f32-r1-f2-final/adapter-final.gguf
```

## Held-Out Generation

Held-out caption source:

```text
../datasets/mnesia-audio-v1/test/audio/single-fluke.txt
```

Command:

```sh
SA3_DEVICE=cpu build-cuda/bin/sa3-generate --model medium --encoding f32 --models-dir models --prompt "$(tr '\n' ' ' < ../datasets/mnesia-audio-v1/test/audio/single-fluke.txt)" --lora train-runs/mnesia-native-medium-f32-r1-f2-final/adapter-final.gguf --lora-strength 1 --frames 2 --steps 1 --seed 123 --out train-runs/mnesia-native-medium-f32-r1-f2-final/heldout-single-fluke.wav
```

Outcome:

```text
lora: applied 1 adapter(s):
  [0] train-runs/mnesia-native-medium-f32-r1-f2-final/adapter-final.gguf  type=lora strength=1.00
step 1/1  t=1.0000
wrote train-runs/mnesia-native-medium-f32-r1-f2-final/heldout-single-fluke.wav  (audio 0.19s, elapsed 17.33s, seed 123)
```
