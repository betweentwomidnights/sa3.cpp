# Native audio metadata suggestions

`sa3-audio-analyze --in audio.wav` emits one JSON object with `schema_version: 1`,
`bpm`, `keyscale`, confidence values, sources, and a bare caption suggestion such
as `117 bpm, C minor`. `--control-info` reports the CPU/WAV contract. Errors emit
`ok: false` and a nonzero exit code. No models, backend libraries or Python are
needed. The executable is included in the Windows core and macOS packages.

The CPU algorithms follow Gary's existing `services/sa3/bpm_analysis.py` and
`key_analysis.py`: mean channels to mono, centered Kaiser polyphase resampling
to 11025 Hz, RMS-onset autocorrelation, Hann-window chroma accumulation and
Krumhansl-Kessler profile scoring. They are suggestions for human correction;
matching the old estimator does not prove the track's true tempo or key.
Short clips or silence can return missing estimates. Non-finite audio is rejected.

The CLI reads PCM 8/16/24/32-bit and float 32/64-bit WAVs, including extended WAV
format headers and Unicode paths. It never writes to the source. It preserves
float amplitudes for analysis; the existing training/inference WAV reader retains
its default clamping behavior. Hosts can decode other formats to a temporary WAV
using their audio decoder. Gary uses FFmpeg with direct arguments, retains the
original sample rate/channels, and removes only its own temporary decoded file.

The model-free `sa3-audio-analysis-cli-test` exercises the real tool with a known
C major chord, Unicode/literal paths, 8-bit silence, malformed inputs, and NaN
rejection. `sa3-train-audio-test` verifies the shared WAV reader's existing
training behavior. Gary's `smoke-tests/compare_sa3_native_analysis.py` compares
real WAV datasets against the original helper and checks source hashes.
