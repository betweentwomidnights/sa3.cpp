"""Check a staged runtime from an empty working directory, without models.

Python is a packaging dependency only. The installed native runtime needs none.
On macOS pass --arch arm64 or x86_64 to check each universal slice.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    if not __debug__:
        raise RuntimeError("Package checks require Python assertions; remove -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("version")
    parser.add_argument("--arch", choices=("arm64", "x86_64"))
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    version = args.version[1:] if args.version.startswith("v") else args.version
    suffix = ".exe" if os.name == "nt" else ""
    prefix = ["arch", "-" + args.arch] if args.arch else []
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("SA3_", "SAT_", "GGML_")):
            del env[key]
    # Offline control probes must bypass even invalid launch configuration.
    env["SA3_PORT"] = "-1"
    env["SA3_MODELS_DIR"] = str(directory / "absent-contract-models")
    with tempfile.TemporaryDirectory(prefix="sa3-package-contract-") as scratch:
        def output(tool, flag):
            executable = directory / (tool + suffix)
            result = subprocess.run(
                [*prefix, str(executable), flag,
                 *(["--port", "0"] if tool == "sa3-server" and flag == "--control-info" else [])],
                cwd=scratch, env=env,
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            if result.returncode:
                raise RuntimeError(f"{tool} {flag} failed: {result.stderr}")
            return result.stdout.strip()

        for service, tool in (("sa3", "sa3-server"), ("sat", "sat-server")):
            assert output(tool, "--version") == version, tool
            props = json.loads(output(tool, "--props"))
            assert props["success"] is True and props["service"] == service, props
            assert props["version"] == version, props
            assert any(device["backend"].upper().startswith("CPU")
                       for device in props["devices"]), props

        info = json.loads(output("sa3-server", "--control-info"))
        assert info["schema_version"] == 1 and info["service"] == "sa3", info
        assert info["version"] == version, info
        assert all(info["capabilities"].get(key) is True for key in (
            "fixed_prefix", "request_splice", "conditioning_duration", "model_lifecycle")), info
        training = json.loads(output("sa3-train", "--control-info"))
        assert training["schema_version"] == 1, training
        assert training.get("atomic_progress_file") is True, training
        assert training.get("cooperative_cancel_file") is True, training
        analysis = json.loads(output("sa3-audio-analyze", "--control-info"))
        assert analysis["schema_version"] == 1, analysis
        assert analysis.get("cpu_only") is True and analysis.get("wav_input") is True, analysis
        assert not list(Path(scratch).iterdir()), "Offline probes wrote runtime files"

    build = json.loads((directory / "BUILD-INFO.json").read_bytes())
    assert build["service"] == "sa3" and build["version"] == "v" + version, build
    if build.get("platform") == "windows-x64" and "build_flavor" in build:
        flavor = build["build_flavor"]
        assert flavor in {"portable", "cpu-smoke", "gpu-smoke"}, build
        if flavor == "cpu-smoke":
            assert build["backends"] == [] and build["cuda_architecture_policy"] is None, build
        else:
            assert set(build["backends"]) == {"cuda", "vulkan"}, build
            assert build["cuda_architecture_policy"], build
            if flavor == "portable":
                assert build["cuda_architecture_policy"] == "ggml-default", build
    print(f"PASS packaged SA3/SAT versions/devices and SA3 server/trainer/analyzer controls ({args.arch or 'native'})")


if __name__ == "__main__":
    main()
