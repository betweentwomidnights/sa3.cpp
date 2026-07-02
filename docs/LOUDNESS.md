# Loudness controls

Stable Audio 3 LoRAs often run hotter than the base model. In DAW use this shows up quickly as
clipping when generated float audio is written back to 16-bit PCM WAV. The practical fix we have
liked in Gary is simple: normalize the decoded waveform peak, then catch overs with a gentle limiter.

The defaults in this server mirror the SA3 service in
[`gary-localhost-installer`](https://github.com/betweentwomidnights/gary-localhost-installer):

- `peak_normalize_db = 2.0`
- `limiter_ceiling_db = -0.3`
- `limiter_knee = 0.8`

The positive peak-normalize target is intentional. It brings quieter outputs up into a useful range,
then the limiter shapes any hot transients before the WAV writer can clip. In practice this has been
good enough for the LoRAs we have tested, including repeated DAW transforms.

## Latent controls

Gary's Python SA3 service also exposes latent post-processing knobs:

- `latent_rescale` / `SA3_LATENT_RESCALE`, default `1.0`
- `latent_shift` / `SA3_LATENT_SHIFT`, default `0.0`
- `latent_target_std` / `SA3_LATENT_TARGET_STD`, default off

These are no-op by default. The naming is `latent_rescale`, not `latent_scale`; that matches the
ACE-Step-style implementation we originally referenced. The operation is just:

```text
latents = latents * latent_rescale + latent_shift
```

This happens before VAE/SAME decode. It is useful to keep around for experiments, but it is not the
recommended clipping fix. In listening tests, lowering latent scale can make outputs feel thinner, and
it changes the material before the decoder rather than solving the final-output headroom problem.

Recommendation for V1: keep peak normalization and the limiter as the real loudness path. Treat latent
rescale/shift as advanced compatibility/debug controls, leave them at their no-op defaults, and do not
put them in the primary UI unless a model or LoRA clearly benefits from them.
