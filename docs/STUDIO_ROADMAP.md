# Studio direction

The current sa3.cpp studio has two linked localhost views: inference on
`sa3-server` (`:8006`) and LoRA training on `sa3-train-web` (`:8016`). The
inference view supports prompt generation, audio input for transform and
continuation, LoRA selection, progress, and a history of generated audio. The
training view starts and monitors a native trainer subprocess, lists detected
devices, and lets the user choose a downloaded base model tier. Both views
originated with [pillopaus-project](https://github.com/pillopaus-project/sa3.cpp);
the original commits retain that authorship.

The iOS experiment suggests a useful next step for inference: turn short
generations into a playable kit. This is a later studio milestone, after the
native runtime release is sound. The smallest useful version would:

1. Let the user select and audition a time range from the current result.
2. Save that range to one of several pads, with name, gain, and trim handles.
3. Play pads by pointer, touch, and keyboard with low latency.
4. Record pad events into a current take, then play back and export that take.
5. Save and reopen a kit without rewriting the original generated audio.

The web implementation can use Web Audio for audition and pad playback, while
the server supplies durable audio and kit storage. Export and recording need
a defined timeline and file format before code is added; neither exists in the
current web UI. The [sa3.cpp iOS project](https://github.com/betweentwomidnights/sa3.cpp-ios)
is a UX reference for this milestone, not a claim that these web features are
already available.

[Underfit](https://github.com/dada-bots/underfit) is a reference for later
training workflow improvements, especially dataset preparation, checkpoint
previews, and resuming a run. Its dashboard drives a different Python/MLX
runtime, so those controls need to be connected to the native sa3.cpp trainer
and its GGUF outputs before they belong in this studio.

Further integration with gary4local follows the same pattern as yuey: publish
a tagged native runtime, pin its release URLs and SHA-256 values in the service
manifest, then adapt the existing Python service routes and model management.
The current package is a prerequisite; it does not replace the three Python
services on its own.
