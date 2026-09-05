# continuation splice

continuation feeds a clip back in, regenerates an inpaint window at the end of it, and decodes the
result. everything before that window is *supposed* to survive untouched. it does not. the source
makes a full round trip through the autoencoder on the way in and back out, so the region the model
was told to keep still comes back carrying the codec's artefacts -- and since each continuation's
output is the next one's input, they compound.

the python sa3 service in
[gary4local](https://github.com/betweentwomidnights/gary-localhost-installer) stopped fighting that
and started pasting the original audio back over the head of the output. it is a large quality win
and it takes a lot of pressure off decoder/encoder loras, which is why it lives here at the library
layer rather than in one client.

it is **on by default**, the same way peak normalization is: a zero-initialized request already gets
gary4local's tuned loudness rather than raw output, and this is the same kind of default.

## the two halves

they are separable and the second is useless without the first.

**1. mask overlap.** the inpaint window starts *before* the source ends, so the model regenerates
the handoff instead of butting up against a hard boundary:

```text
max_overlap = max(0, source_duration - 0.05)      # always keep 50 ms of real source
overlap     = min(max_overlap, max(0, requested)) # requested default 0.2 s
mask_start  = max(0, source_duration - overlap)
```

**2. splice.** after decode, before any loudness shaping, restore the original source over
`[0, mask_start)` and equal-power crossfade its last 30 ms into the model's audio. the deliberately
regenerated overlap is never pasted over:

```text
splice_end = min(round(mask_start * sr), source_len, audio_len)
xfade      = clamp(round(xfade_seconds * sr), 0, splice_end / 2)
hard_end   = splice_end - xfade

# optional RMS match, both measured over [0, splice_end)
if gain_match and source_rms > 1e-6 and model_rms > 1e-6:
    gain   = clamp(model_rms / source_rms, 0.25, 4.0)
    source = source * gain

out[0 : hard_end]          = source[0 : hard_end]
out[hard_end : splice_end] = source * cos(phase) + audio * sin(phase)
                             where phase = linspace(0, pi/2, xfade)
# out[splice_end : ] is the model's audio, untouched
```

the gain is applied to the *source*, pulling the restored head up to the model's level -- not the
other way round. loudness shaping then treats the whole thing as one signal, so the seam gets the
same normalize and limiter as the audio on either side of it. that ordering is load-bearing.

`xfade == 1` is degenerate -- a single point at phase 0 would be pure source, not a crossfade -- so
the model's sample stands there. this matches the reference.

## how mask_overlap meets inpaint_start

the python service never sees an explicit mask start; it derives one from the source duration.
libsa3's callers pass `inpaint_start` directly, so the two have to be reconciled.

**with the splice on, `inpaint_start` means "where the source ends and the continuation should
musically begin"**, and the library owns the mechanics from there: it pulls the sampler's window
back by `mask_overlap` (clamped to leave at least 50 ms of real source) and splices up to that same
pulled-back point. one number, computed once, used by both halves -- which is the property that
makes the splice correct, since `splice_end` must equal the mask start the sampler actually used.

this is what lets an existing caller get the improvement without changing its call: a client passing
`inpaint_start = source duration` is already asking for exactly this. `mask_overlap = 0` disables
the pullback for anyone who wants literally the window they asked for.

this is the one place turning the splice on changes what an existing field means.

### mid-clip inpainting

the semantics above are continuation semantics, and the splice reads every inpaint request through
them. for a window in the *middle* of a clip -- `--inpaint-start 2 --inpaint-end 4` on a ten second
source -- that means two things: the window start moves earlier by `mask_overlap`, and the audio
before it is restored from the source while the audio after `inpaint_end` is not. the restored head
is strictly an improvement (that region was meant to be kept), but the window is not quite the one
you asked for.

if you are inpainting rather than continuing, pass `mask_overlap = 0` for exactly the window you
asked for, or `splice = 0` (`--no-splice`) for the old behaviour outright.

## knobs

| field | env | default |
| --- | --- | --- |
| `splice` | `SA3_CONTINUE_SPLICE_SOURCE` | on |
| `mask_overlap` | `SA3_CONTINUE_MASK_OVERLAP` | `0.2` s |
| `xfade` | `SA3_CONTINUE_SPLICE_XFADE` | `0.03` s |
| `gain_match` | `SA3_CONTINUE_SPLICE_GAIN_MATCH` | on |

the env names are the ones gary4local already uses, so a knob learned there transfers verbatim.

from C, `sa3_splice` on `sa3_request_ex` mirrors `sa3_loudness`: leave `set = 0` for the defaults
plus any env overrides, or set `set = 1` to drive it per-request -- including the old un-spliced
behaviour with `splice = 0`, exactly as a caller asks for raw audio today.

from the CLI:

```sh
sa3-generate --model small-music --init take.wav --inpaint-start 12.0 --inpaint-end 18.0 \
             --prompt "..." --out continued.wav
sa3-generate ... --no-splice            # A/B against the old behaviour
sa3-generate ... --mask-overlap 0.4 --splice-xfade 0.05 --no-splice-gain-match
```

## what it reports

`sa3_last_meta()` fills `splice_applied`, `splice_end_seconds`, `splice_xfade_applied`,
`splice_gain`, `mask_start_seconds` and `mask_overlap_applied` alongside the loudness measurements.
the pipeline also prints both halves:

```text
continue: mask pulled back 0.200s to 11.800s (source 12.000s), splicing on decode
continue splice: kept 0-11.800s, xfade 30ms, gain 1.043
```
