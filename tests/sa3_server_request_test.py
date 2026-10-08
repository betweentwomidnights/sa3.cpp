"""Validate unified /generate splice controls without model downloads."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def request(port, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main():
    executable = str(Path(sys.argv[1]).resolve())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="sa3-http-contract-") as directory:
        env = os.environ.copy()
        # Model lookup must not inherit a developer's real model paths or .env.
        env["SA3_DEVICE"] = "cpu"
        env["SA3_CONTINUE_MASK_OVERLAP"] = "0.2"
        env["SA3_CONTINUE_SPLICE_XFADE"] = "0.03"
        process = subprocess.Popen(
            [executable, "--models-dir", str(Path(directory) / "missing-models"),
             "--port", str(port)], cwd=directory, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise AssertionError("server exited before startup")
                try:
                    status, health = request(port, "/health")
                    assert status == 200
                    break
                except (OSError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise AssertionError("server did not start")

            assert health["capabilities"]["model_lifecycle"] is True
            assert health["loaded"] is False and health["loading"] is False and health["error"] is None
            status, ready = request(port, "/ready")
            assert status == 503 and ready["ready"] is False and ready["error"] is None
            for path in ("/load", "/reload"):
                status, failure = request(port, path, {})
                assert status == 503 and failure["success"] is False and failure["error"], (path, status, failure)
                status, health = request(port, "/health")
                assert status == 200 and health["error"] == failure["error"] and health["loading"] is False
                assert health["active_generations"] == 0 and health["lifecycle_busy"] is False
                status, ready = request(port, "/ready")
                assert status == 503 and ready["error"] == failure["error"]
                status, unloaded = request(port, "/unload", {})
                assert status == 200 and unloaded["success"] is True and unloaded["loaded"] is False
                _, health = request(port, "/health")
                assert health["error"] is None

            for path in ("/generate", "/generate/loop"):
                for payload, message in [
                    ([], "JSON object required"),
                    ({"mask_overlap": -0.1}, "mask_overlap"),
                    ({"mask_overlap": "nan"}, "finite number"),
                    ({"splice_xfade": -0.1}, "xfade"),
                    ({"splice_xfade": "inf"}, "finite number"),
                    ({"splice_source": "false"}, "must be a boolean"),
                    ({"splice_gain_match": 1}, "must be a boolean"),
                    ({"fixed_prefix": "true"}, "must be a boolean"),
                    ({"fixed_prefix": True}, "requires init_path"),
                    ({"conditioning_seconds_total": -1}, "duration limit"),
                    ({"inpaint_padding_sec": 61}, "[0, 60]"),
                    ({"inpaint_padding_sec": "nan"}, "finite number"),
                    ({"seed": 1.5}, "integer"),
                ]:
                    status, body = request(port, path, payload)
                    assert status == 400 and message in body["error"], (path, status, body)

            for overrides in ({"seed": 4294967295}, {"splice_source": False, "splice_gain_match": False,
                                    "mask_overlap": 0, "splice_xfade": 0.05}):
                status, body = request(port, "/generate", {"prompt": "test", "duration": 1,
                                                            "steps": 1, **overrides})
                assert status == 200 and body["success"], (status, body)
                if "seed" in overrides:
                    assert body["seed"] == overrides["seed"], body
                for _ in range(100):
                    status, poll = request(port, "/poll_status/" + body["session_id"])
                    if poll["status"] == "failed":
                        assert "error" in poll
                        break
                    time.sleep(0.05)
                else:
                    raise AssertionError("missing-model job did not fail promptly")
            _, health = request(port, "/health")
            assert health["loaded"] is False and health["error"]
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
