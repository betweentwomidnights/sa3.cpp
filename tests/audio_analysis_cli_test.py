"""Model-free CLI analysis contract, using only Python's standard library."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

binary = Path(sys.argv[1]).resolve()
def run(*args, success=True):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)
    return json.loads(result.stdout)

assert run("--control-info") == {"schema_version":1,"cpu_only":True,"wav_input":True}
with tempfile.TemporaryDirectory(prefix="sa3-analysis-") as temporary:
    root = Path(temporary)
    path = root / "sidecar ♭ and %TEMP% & quote's.wav"
    sample_rate = 11025
    # A steady C major triad. The explicit sidecar suggestion has no confidence gate.
    data = bytearray()
    for i in range(sample_rate*8):
        value = sum(math.sin(2*math.pi*f*i/sample_rate) for f in [261.625565,329.627557,391.995436])/4
        data += struct.pack("<h", round(value*32767))
    with wave.open(str(path), "wb") as file:
        file.setparams((1,2,sample_rate,0,"NONE","not compressed"))
        file.writeframes(data)
    original = path.read_bytes()
    result = run("--in", path)
    assert result["ok"] and result["schema_version"] == 1
    assert result["keyscale"] == "C major", result
    assert result["suggestion"].endswith("C major"), result
    assert result["key_confidence"] >= 0
    assert path.read_bytes() == original
    with wave.open(str(path), "wb") as file:
        file.setparams((2,1,sample_rate,0,"NONE","not compressed"))
        file.writeframes(bytes([128])*(sample_rate*2*2))
    result = run("--in", path)
    assert result["bpm"] is None and result["keyscale"] == "" and result["suggestion"] == "", result
    path.write_bytes(b"not a WAV")
    assert not run("--in", path, success=False)["ok"]
    assert not run("--in", root / "missing.wav", success=False)["ok"]
    assert not run("--in", path, "--extra", success=False)["ok"]
    # IEEE float audio must reject NaN before emitting a JSON confidence value.
    payload = struct.pack("<f", float("nan"))*9000
    path.write_bytes(b"RIFF"+struct.pack("<I",36+len(payload))+b"WAVEfmt "+struct.pack("<IHHIIHH",16,3,1,sample_rate,sample_rate*4,4,32)+b"data"+struct.pack("<I",len(payload))+payload)
    assert "non-finite" in run("--in", path, success=False)["error"]
print("PASS native CPU analysis, Unicode/literal paths, silence/8-bit WAV, invalid inputs, and source preservation")
