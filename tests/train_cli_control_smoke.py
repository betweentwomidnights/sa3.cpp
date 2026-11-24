"""Exercise the real CLI control contract without models or a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
info = subprocess.run([binary, "--control-info"], capture_output=True, text=True, check=True)
capabilities = json.loads(info.stdout)
assert capabilities == {"schema_version": 1, "cooperative_cancel_file": True, "atomic_progress_file": True,
                        "init_adapter": True}
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
# Arguments and paths are UTF-8 end to end. On Windows the trainer used to read argv in the system
# code page, so a cp1252 name ("Jóhannsson") reached the progress JSON as invalid UTF-8 and the run
# died with "cannot serialize training progress"; anything outside cp1252 became "?".
with tempfile.TemporaryDirectory(prefix="sa3-cli-Jóhannsson-データ-") as temporary:
    root = Path(temporary)
    dataset = root / "dataset ü"
    dataset.mkdir()
    models = root / "modèles データ"
    status = root / "progress é.json"
    failed = subprocess.run([binary, "--dataset", dataset, "--models-dir", models,
                             "--out", root / "run", "--steps", "1", "--device", "cpu",
                             "--progress-file", status],
                            capture_output=True)
    assert failed.returncode != 0
    assert status.is_file(), failed.stderr
    state = json.loads(status.read_bytes().decode("utf-8"))
    assert state["status"] == "failed", (state, failed.stderr)
    assert str(models) in state["error"], (state["error"], str(models))
print("PASS real trainer capabilities, terminal failure, pre-load cooperative cancellation, and UTF-8 paths")
