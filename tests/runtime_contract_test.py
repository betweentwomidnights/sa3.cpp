"""Check the executable contract used by gary4local before model installation."""

import json
import subprocess
import sys


binary, version, service = sys.argv[1:4]


def output(flag):
    return subprocess.check_output([binary, flag], text=True, timeout=30).strip()


assert output("--version") == version
if service == "sa3":
    info = json.loads(output("--control-info"))
    assert info["schema_version"] == 1 and info["service"] == service
    assert info["version"] == version
    assert all(info["capabilities"].get(key) is True for key in (
        "fixed_prefix", "request_splice", "conditioning_duration", "model_lifecycle"))
props = json.loads(output("--props"))
assert props["success"] is True
assert props["service"] == service
assert props["version"] == version
assert props["devices"]
assert any(device["backend"].upper().startswith("CPU") for device in props["devices"])
for device in props["devices"]:
    assert device["backend"] and device["description"]
    assert device["type"] in {"cpu", "gpu", "integrated_gpu"}
    assert isinstance(device["memory_total_bytes"], int)
