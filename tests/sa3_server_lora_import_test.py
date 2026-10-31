"""Exercise streamed LoRA + dice imports without downloading or loading models."""

from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def adapter(architecture="sa3-lora", target="decoder"):
    # A complete, tiny GGUF with a LoRA factor pair. Decoder discovery needs no base model.
    fields = [("general.architecture", 8, string(architecture)),
              ("lora.target", 8, string(target)), ("lora.base_model", 8, string("same-l")),
              ("lora.rank", 4, struct.pack("<I", 1)),
              ("lora.alpha", 6, struct.pack("<f", 1))]
    data = b"GGUF" + struct.pack("<IQQ", 3, 2, len(fields))
    for key, kind, value in fields:
        data += string(key) + struct.pack("<I", kind) + value
    for suffix, offset in (("A", 0), ("B", 32)):
        data += string("decoder.test.lora_" + suffix) + struct.pack("<IQIQ", 1, 1, 0, offset)
    data += b"\0" * (-len(data) % 32)
    return data + struct.pack("<f", 1) + b"\0" * 28 + struct.pack("<f", 1) + b"\0" * 28


def multipart(parts):
    boundary = "sa3-import-test-boundary"
    body = b""
    for field, filename, data in parts:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                 f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
        body += data + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def main():
    executable = str(Path(sys.argv[1]).resolve())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def request(path, parts=None):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            body, headers = (None, {}) if parts is None else multipart(parts)
            connection.request("GET" if parts is None else "POST", path, body, headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def upload(name, parts):
        return request("/loras/import?name=" + quote(name, safe=""), parts)

    with tempfile.TemporaryDirectory(prefix="sa3-lora-import-") as directory:
        root = Path(directory)
        weights, pools = root / "adapters", root / "prompts"
        env = os.environ.copy()
        env["SA3_DEVICE"] = "cpu"
        process = subprocess.Popen(
            [executable, "--models-dir", str(root / "missing-models"), "--adapters-dir", str(weights),
             "--prompts-dir", str(pools), "--port", str(port)], cwd=root, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                assert process.poll() is None, "server exited"
                try:
                    if request("/health")[0] == 200:
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError("server did not start")

            gguf = ("file", "long-original-name.gguf", adapter())
            captions = ("prompts", "unrelated-name.json", json.dumps({
                "version": 1, "dice": {"instrumental": ["Dream pop", "Indie folk"]}}).encode())
            status, body = upload("billie", [gguf, captions])
            assert status == 200 and body["success"] and body["prompt_count"] == 2, (status, body)
            assert (weights / "lora-billie.gguf").read_bytes() == gguf[2]
            assert (pools / "billie.json").read_bytes() == captions[2]
            assert request("/loras")[1]["loras"][0]["name"] == "billie"
            prompts = request("/prompts?lora=billie")[1]
            assert prompts["missing_loras"] == []
            assert prompts["prompts"]["dice"]["instrumental"] == ["Dream pop", "Indie folk"]
            before = {p: p.read_bytes() for p in weights.iterdir()}
            assert upload("billie", [gguf, captions])[0] == 409
            assert before == {p: p.read_bytes() for p in weights.iterdir()}
            assert (pools / "billie.json").read_bytes() == captions[2]

            # Optional captions and names which filename inference would otherwise strip.
            assert upload("v2", [gguf])[0] == 200
            assert {x["name"] for x in request("/loras")[1]["loras"]} == {"billie", "v2"}
            assert not (pools / "v2.json").exists()
            assert upload("reverse", [captions, gguf])[0] == 200
            for name in ("../escape", "C:/escape", "defaults", "nul", "con", "com1", "Billie", "", "x" * 65):
                assert upload(name, [gguf])[0] == 400, name

            for parts, expected in [
                ([], 400), ([captions], 400), ([gguf, gguf], 400),
                ([("file", "adapter.ckpt", adapter())], 400),
                ([("file", "adapter.gguf", b"invalid")], 400),
                ([("file", "adapter.gguf", adapter("base-model"))], 400),
                ([("file", "adapter.gguf", adapter(target="invalid"))], 400),
                ([("file", "adapter.gguf", adapter()[:-40])], 400),
                ([gguf, ("prompts", "x.json", b"not json")], 400),
                ([gguf, ("prompts", "x.json", b'{"dice":{}}')], 400),
                ([gguf, ("prompts", "x.json", b'{"dice":{"instrumental":[3]}}')], 400),
                ([gguf, ("prompts", "x.json", b" " * (1024 * 1024 + 1))], 413),
            ]:
                status, body = upload("failed", parts)
                assert status == expected and body["error"], (status, body)
                assert not any("failed" in p.name or p.name.startswith(".import-") for p in weights.iterdir())
                assert not (pools / "failed.json").exists()

            # Captions already present must not be replaced, even if no adapter exists yet.
            (pools / "orphan.json").write_text("{}")
            assert upload("orphan", [gguf, captions])[0] == 409
            assert (pools / "orphan.json").read_text() == "{}"
            assert not (weights / "lora-orphan.gguf").exists()
            # A destination write failure must not leave runnable weights or a registry.
            pools.rename(root / "saved-prompts")
            pools.write_text("not a directory")
            assert upload("unwritable", [gguf, captions])[0] == 500
            assert not (weights / "lora-unwritable.gguf").exists()
            assert not (weights / "lora-unwritable.gguf.json").exists()
            pools.unlink()
            (root / "saved-prompts").rename(pools)

            # An interrupted stream leaves no staged adapter behind.
            body, headers = multipart([gguf, captions])
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                head = ("POST /loras/import?name=interrupted HTTP/1.1\r\nHost: localhost\r\n"
                        f"Content-Type: {headers['Content-Type']}\r\nContent-Length: {len(body)}\r\n\r\n").encode()
                sock.sendall(head + body[:200])
            for _ in range(100):
                if not list(weights.glob(".import-*")):
                    break
                time.sleep(0.02)
            assert not (weights / "lora-interrupted.gguf").exists()
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: upload("racing", [gguf, captions])[0], range(2)))
            assert sorted(results) == [200, 409], results
            assert not list(weights.glob(".import-*"))
            assert request("/health")[0] == 200
            print("LoRA import, caption association, validation, rollback and collision checks passed")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
