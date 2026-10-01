"""Exercise SAT HTTP request validation without downloading model weights."""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def check(executable, model, cases):
    port = free_port()
    process = subprocess.Popen(
        [executable, "--model", model, "--models-dir", "missing-sat-models",
         "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(50):
            if process.poll() is not None:
                raise AssertionError(f"{model} server exited early: {process.stderr.read()!r}")
            try:
                status, body = request(port, "/health")
                break
            except (OSError, TimeoutError):
                time.sleep(0.1)
        else:
            raise AssertionError(f"{model} server did not start")
        assert status == 503 and body["status"] == "model_missing", (status, body)
        for path, payload, expected, fragment in cases:
            status, body = request(port, path, payload)
            assert status == expected and fragment in body.get("error", ""), (status, body)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


def main():
    binary = sys.argv[1]
    check(binary, "foundation-1", [
        ("/generate", {"prompt": "Synth", "bars": 5, "bpm": 120}, 400, "4 or 8 bars"),
        ("/generate", {"prompt": "Synth", "bars": 4, "host_bpm": 1000}, 400, "between 20 and 999"),
        ("/generate", {"prompt": "Synth", "bars": 4, "host_bpm": 123}, 503, "missing"),
        ("/generate", {"prompt": "Synth", "bars": 4, "bpm": 120}, 503, "missing"),
    ])
    check(binary, "arc", [
        ("/generate", {"prompt": "drums", "seconds": 12}, 400, "out of range"),
        ("/generate/loop", {"prompt": "drums", "bpm": 100, "bars": 8}, 400, "exceeds"),
        ("/generate", {"prompt": "drums", "seconds": 1}, 503, "missing"),
    ])


if __name__ == "__main__":
    main()
