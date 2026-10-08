"""Exercise the real CLI control contract without models or a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
info = subprocess.run([binary, "--control-info"], capture_output=True, text=True, check=True)
capabilities = json.loads(info.stdout)
assert capabilities == {"schema_version": 1, "cooperative_cancel_file": True, "atomic_progress_file": True}
with tempfile.TemporaryDirectory(prefix="sa3-cli-control-") as temporary:
    root = Path(temporary)
    dataset = root / "dataset"
    dataset.mkdir()
    status = root / "progress.json"
    cancel = root / "cancel.requested"
    common = [binary, "--dataset", dataset, "--models-dir", root / "missing-models",
              "--out", root / "run", "--steps", "1", "--device", "cpu",
              "--progress-file", status, "--cancel-file", cancel]
    failed = subprocess.run(common, capture_output=True, text=True)
    assert failed.returncode != 0
    state = json.loads(status.read_text())
    assert state["status"] == "failed" and state["error"], (state, failed.stderr)
    assert state["final_adapter"] == "" and state["step"] == 0
    cancel.write_text("stop")
    cancelled = subprocess.run(common, capture_output=True, text=True, check=True)
    state = json.loads(status.read_text())
    assert state["status"] == "cancelled" and state["step"] == 0, state
    assert state["final_adapter"] == "" and cancel.is_file()
    assert not list(root.glob("progress.json.tmp-*"))
    assert "cancelled before loading" in cancelled.stdout
print("PASS real trainer capabilities, terminal failure, and pre-load cooperative cancellation")
