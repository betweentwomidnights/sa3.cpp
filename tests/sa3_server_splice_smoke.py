"""Optional real-model check of completed splice metadata and consume polling.

Usage: python tests/sa3_server_splice_smoke.py SERVER_EXE MODELS_DIR
Runs a tiny one-step CPU continuation; this is a transport check, not audio QA.
"""

import base64
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import wave

from sa3_server_request_test import request


def main():
    binary = str(Path(sys.argv[1]).resolve())
    models = str(Path(sys.argv[2]).resolve())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="sa3-splice-smoke-") as directory:
        source = Path(directory) / "source.wav"
        # Exercise native resampling too; the client's recording rate need not
        # match the native model's 44.1 kHz output.
        with wave.open(str(source), "wb") as wav:
            wav.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
            wav.writeframes(b"".join(
                struct.pack("<hh", *([int(3000 * math.sin(2 * math.pi * 220 * n / 48000))] * 2))
                for n in range(12000)
            ))
        env = os.environ.copy()
        env["SA3_DEVICE"] = "cpu"
        with open(Path(directory) / "server.log", "wb") as log:
            process = subprocess.Popen(
                [binary, "--models-dir", models, "--model", "small-music",
                 "--encoding", "q4_k_m", "--ae-encoding", "f16", "--t5-encoding", "f16",
                 "--port", str(port), "--threads", "8"],
                cwd=directory, env=env, stdout=log, stderr=log,
            )
            try:
                for _ in range(100):
                    try:
                        status, _ = request(port, "/health")
                        if status == 200:
                            break
                    except (OSError, TimeoutError):
                        time.sleep(0.1)
                else:
                    raise AssertionError("server did not start")
                status, ready = request(port, "/ready")
                assert status == 503 and ready["ready"] is False
                status, loaded = request(port, "/load", {})
                assert status == 200 and loaded["status"] == "loaded" and loaded["load_seconds"] > 0
                status, loaded = request(port, "/load", {})
                assert status == 200 and loaded["status"] == "already_loaded" and loaded["load_seconds"] == 0
                status, ready = request(port, "/ready")
                assert status == 200 and ready["ready"] is True
                status, reloaded = request(port, "/reload", {})
                assert status == 200 and reloaded["success"] is True

                # Two admitted jobs cover both running and queued generations.
                # Lifecycle routes must answer promptly, not wait for g_mtx and
                # then unload/switch between those jobs.
                pending = []
                for _ in range(2):
                    status, submitted = request(port, "/generate", {
                        "prompt": "a soft tone", "duration": 1, "steps": 32, "seed": 42,
                        "duration_padding_sec": 0, "keep_models": False,
                    })
                    assert status == 200 and submitted["success"] is True
                    pending.append(submitted["session_id"])
                _, busy = request(port, "/health")
                assert busy["active_generations"] == 2, busy
                started = time.monotonic()
                status, already_loaded = request(port, "/load", {})
                assert status == 200 and already_loaded["status"] == "already_loaded"
                assert time.monotonic() - started < 1
                for path, payload in [("/unload", {}), ("/reload", {}),
                                      ("/models/select", {"variant": "small-music", "encoding": "q4_k_m"})]:
                    started = time.monotonic()
                    status, response = request(port, path, payload)
                    assert status == 409 and "in progress" in response["error"], (path, status, response)
                    assert time.monotonic() - started < 1, "busy lifecycle request blocked behind inference"
                for sid in pending:
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        _, poll = request(port, "/poll_status/" + sid + "?consume=1")
                        assert poll["status"] != "failed", poll
                        if poll["status"] == "completed":
                            break
                        time.sleep(0.1)
                    else:
                        raise AssertionError("queued lifecycle test job did not finish")
                for _ in range(100):
                    _, health = request(port, "/health")
                    if health["active_generations"] == 0:
                        break
                    time.sleep(0.01)
                assert health["loaded"] is True and health["active_generations"] == 0
                # Ready means initialized/validated, not persistent weight
                # residency: both completed jobs used the frugal policy.
                status, ready = request(port, "/ready")
                assert status == 200 and ready["ready"] is True
                status, unloaded = request(port, "/unload", {})
                assert status == 200 and unloaded["success"] is True
                status, ready = request(port, "/ready")
                assert status == 503 and ready["ready"] is False
                print("PASS: native load/reload/readiness, frugal initialization and prompt idle-only lifecycle controls")
                for fixed_prefix in (False, True):
                    status, submitted = request(port, "/generate", {
                        "prompt": "a soft tone", "duration": 1, "steps": 1, "seed": 42,
                        "init_path": str(source), "inpaint_start": 0.25, "inpaint_end": 1,
                        "target_samples": 22050, "duration_padding_sec": 0,
                        "splice_source": True, "mask_overlap": 0.1, "fixed_prefix": fixed_prefix,
                        "splice_xfade": 0.03, "splice_gain_match": False,
                        "peak_normalize_db": None, "limiter_ceiling_db": None,
                    })
                    assert status == 200 and submitted["success"], (status, submitted)
                    path = "/poll_status/" + submitted["session_id"]
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        status, polled = request(port, path)
                        assert status == 200 and polled["status"] != "failed", polled
                        if polled["status"] == "completed":
                            break
                        time.sleep(0.2)
                    else:
                        raise AssertionError("generation timeout")
                    assert polled["meta"]["prefix_latent_tokens"] == (2 if fixed_prefix else 0), polled["meta"]
                    assert polled["meta"]["latent_sample_size"] >= 11, polled["meta"]
                    splice = polled["meta"]["splice"]
                    assert splice["splice_applied"] is True, splice
                    assert abs(splice["mask_start_seconds"] - 0.15) < 1e-5, splice
                    assert abs(splice["mask_overlap"] - 0.1) < 1e-5, splice
                    assert abs(splice["splice_xfade_applied"] - 0.03) < 1e-4, splice
                    assert splice["splice_gain"] == 1, splice
                    output = base64.b64decode(polled["audio_data"])
                    assert output[0:4] == b"RIFF"
                    offset = 12
                    alignment = rate = data_size = None
                    while offset + 8 <= len(output):
                        chunk, size = struct.unpack_from("<4sI", output, offset)
                        if chunk == b"fmt ":
                            _, _, rate, _, alignment, _ = struct.unpack_from("<HHIIHH", output, offset + 8)
                        elif chunk == b"data":
                            data_size = size
                        offset += 8 + size + (size % 2)
                    assert rate == 44100 and alignment and data_size == 22050 * alignment
                    status, consumed = request(port, path + "?consume=1")
                    assert status == 200 and consumed["meta"] == polled["meta"]
                    assert consumed["audio_data"] == polled["audio_data"]
                    status, _ = request(port, path)
                    assert status == 404
                    print(f"PASS: fixed_prefix={fixed_prefix}, native resampling, exact output length, measured metadata, normal/consume polling")
            except Exception:
                log.flush()
                print((Path(directory) / "server.log").read_text(errors="replace")[-8000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
