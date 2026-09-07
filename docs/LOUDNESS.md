# loudness controls

stable audio models often run hotter than the available PCM headroom. stable audio 3 loras
are a common example, and Stable Audio Open 1.0 can also decode above 0 dBFS. in DAW use this shows up quickly as
clipping when generated float audio is written back to 16-bit PCM WAV. the practical fix we have
liked in [gary4local](https://github.com/betweentwomidnights/gary-localhost-installer) is simple: normalize the decoded waveform peak, then catch overs with a gentle limiter.

The defaults shared by the SA3 and optional SAT pipelines mirror the SA3 service in
[`gary-localhost-installer`](https://github.com/betweentwomidnights/gary-localhost-installer):

- `peak_normalize_db = 2.0`
- `limiter_ceiling_db = -0.3`
- `limiter_knee = 0.8`

the positive peak-normalize target is intentional. It brings quieter outputs up into a useful range,
then the limiter shapes any hot transients before the WAV writer can clip. in practice this has been
good enough for the loras we have tested, including repeated DAW transforms.

## latent controls

gary's python sa3 service also exposes latent post-processing knobs:

- `latent_rescale` / `SA3_LATENT_RESCALE`, default `1.0`
- `latent_shift` / `SA3_LATENT_SHIFT`, default `0.0`
- `latent_target_std` / `SA3_LATENT_TARGET_STD`, default off

these are no-op by default. they were brought over from ace-step's implementation (which it turns out they actually don't even use either...). the operation is just:

```text
latents = latents * latent_rescale + latent_shift
```

this happens before VAE/SAME decode. it could be useful to keep around for experiments, but it is not the
recommended clipping fix. In listening tests, lowering latent scale can make outputs feel thinner, and
it changes the material before the decoder rather than solving the final-output headroom problem.

recommendation for v1: keep peak normalization and the limiter as the real loudness path. treat latent
rescale/shift as advanced compatibility/debug controls, leave them at their no-op defaults, and do not
put them in the primary UI unless a model or lora clearly benefits from them.

i may just remove those, honestly.
