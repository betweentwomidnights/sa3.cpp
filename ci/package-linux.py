#!/usr/bin/env python3
"""Build flat Linux core/backend ZIPs. Python is a build dependency only.

Example Spark smoke build:
  python3 ci/package-linux.py --version v0.2.3 --backends cuda --cuda-arch native --require-gpu
Without --cuda-arch the CUDA build uses ggml's portable architecture policy.
CUDA user libraries and the Vulkan loader/driver remain system dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("sa3-server", "sat-server", "sat-generate", "sa3-train-web", "sa3-train",
         "sa3-generate", "sa3-lora-convert", "sa3-audio-analyze", "sa3-smoke")


def run(*command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def output(*command):
    return subprocess.check_output(list(map(str, command)), text=True).strip()


def owned_directory(value, kind):
    path = (ROOT / value).resolve()
    if ROOT not in path.parents or path.parts[len(ROOT.parts)] in {
        ".git", "ggml", "src", "tools", "vendor", "models", "docs", "tests", "ci", "prompts"
    }:
        raise ValueError(f"{kind} must be a dedicated directory inside {ROOT}")
    marker = path / ".sa3-linux-package"
    if path.exists() and not marker.is_file() and any(path.iterdir()):
        raise ValueError(f"refusing nonempty, unowned {kind}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    marker.write_text("Owned by ci/package-linux.py\n", encoding="utf-8")
    return path


def cache_values(build):
    result = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^:#]+):[^=]+=(.*)", line)
        if match:
            result[match[1]] = match[2]
    return result


def elf_info(directory):
    """Record actual loader requirements and reject build-directory RPATHs."""
    needed = set()
    versions = {"GLIBC": set(), "GLIBCXX": set()}
    for path in directory.iterdir():
        if not is_elf(path):
            continue
        dynamic = output("readelf", "-d", path)
        for tag, value in re.findall(r"\((NEEDED|RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic):
            if tag == "NEEDED":
                needed.add(value)
            elif value != "$ORIGIN":
                raise ValueError(f"{path.name} has non-relocatable {tag}: {value}")
        symbols = output("readelf", "--version-info", path)
        for family in versions:
            versions[family].update(re.findall(rf"\b{family}_(\d+(?:\.\d+)+)\b", symbols))
    return {
        "external_sonames": sorted(name for name in needed if not (directory / name).exists()),
        "symbol_versions": {family: max(items, key=lambda v: tuple(map(int, v.split('.'))))
                            for family, items in versions.items() if items},
    }


def write_zip(directory, destination):
    # Dereference the library aliases: gary4local extracts ordinary files, not symlinks.
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())


def unpack(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
            archive.extract(member, destination)
            if not member.is_dir():
                target.chmod((member.external_attr >> 16) & 0o777)


def check_loader_paths(directory):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    for path in directory.iterdir():
        if not is_elf(path):
            continue
        loaded = subprocess.check_output(["ldd", str(path)], env=env, text=True)
        if "not found" in loaded:
            raise ValueError(f"unresolved package dependencies for {path}:\n{loaded}")
        for name, resolved in re.findall(r"(lib(?:ggml[^ ]*|sa3)\.so[^ ]*) => (\S+)", loaded):
            if Path(resolved).resolve().parent != directory.resolve():
                raise ValueError(f"{path.name} loads {name} outside the extracted package: {resolved}")


def is_elf(path):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--backends", nargs="*", choices=("cuda", "vulkan"), default=["cuda", "vulkan"])
    parser.add_argument("--cuda-arch", default="")
    parser.add_argument("--build-dir", default="build-dist-linux")
    parser.add_argument("--out-dir", default="dist-linux")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--require-gpu", action="store_true", help="require a real device in each selected backend's --props")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "x86_64"}:
        parser.error("build natively on ARM64 or x64 Linux")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.version) or args.jobs < 1:
        parser.error("use --version vX.Y.Z and a positive --jobs")
    if args.cuda_arch and "cuda" not in args.backends:
        parser.error("--cuda-arch requires the CUDA backend")
    if args.require_gpu and not args.backends:
        parser.error("--require-gpu needs at least one backend")
    for command in ("cmake", "ctest", "git", "readelf", "ldd"):
        if not shutil.which(command):
            parser.error(f"missing build dependency: {command}")
    os.chdir(ROOT)
    declared = re.search(r"project\(sa3_cpp VERSION (\d+\.\d+\.\d+)",
                         (ROOT / "CMakeLists.txt").read_text())
    if not declared or args.version != "v" + declared[1]:
        parser.error("--version must match CMakeLists.txt project(VERSION)")
    dirty = bool(output("git", "status", "--porcelain"))
    if args.require_clean and dirty:
        parser.error("--require-clean: commit or remove source changes first")
    pin = output("git", "rev-parse", "HEAD:ggml")
    ggml = output("git", "-C", "ggml", "rev-parse", "HEAD")
    if ggml != pin or output("git", "-C", "ggml", "status", "--porcelain"):
        parser.error("ggml must be clean at the pinned submodule revision")
    build_candidate, out_candidate = (ROOT / args.build_dir).resolve(), (ROOT / args.out_dir).resolve()
    if (build_candidate == out_candidate or build_candidate in out_candidate.parents
            or out_candidate in build_candidate.parents):
        parser.error("build and output directories must be separate")
    build = owned_directory(args.build_dir, "build directory")
    destination = owned_directory(args.out_dir, "output directory")
    old_cache = cache_values(build) if (build / "CMakeCache.txt").is_file() else {}
    compilers = {
        "C": os.environ.get("CC") or old_cache.get("CMAKE_C_COMPILER") or shutil.which("cc"),
        "CXX": os.environ.get("CXX") or old_cache.get("CMAKE_CXX_COMPILER") or shutil.which("c++"),
    }
    if not all(compilers.values()):
        parser.error("C and C++ compilers are required; select them with CC/CXX")
    # ggml's full ARM variant set includes SME. Fail before compiling CUDA when
    # the host compiler cannot build it (Ubuntu 24.04's default GCC 13 cannot).
    if platform.machine() == "aarch64":
        for language, compiler in compilers.items():
            probe = subprocess.run([compiler, "-march=armv9.2-a+sme", "-x",
                                    "c" if language == "C" else "c++", "-fsyntax-only", "-"],
                                   input="int main(void) { return 0; }", text=True,
                                   capture_output=True)
            if probe.returncode:
                parser.error(f"{compiler} cannot build ggml's ARM SME variants. "
                             "Select capable compilers with CC/CXX (tested with GCC 14).\n" + probe.stderr)
    platform_key = "linux-arm64" if platform.machine() == "aarch64" else "linux-x64"
    configure = ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_SHARED_LIBS=ON", "-DGGML_NATIVE=OFF", "-DGGML_BACKEND_DL=ON",
                 "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_METAL=OFF", "-DSA3_METAL=OFF",
                 "-DSA3_HIP=OFF", "-DGGML_HIP=OFF", "-DSA3_PRIVATE_GGML=OFF",
                 "-DSA3_STATIC=OFF", "-DSA3_BUILD_TOOLS=ON", "-DSA3_BUILD_SAT=ON",
                 "-DBUILD_TESTING=ON", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                 "-DCMAKE_INSTALL_RPATH=$ORIGIN", "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF",
                 f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build / 'bin'}"]
    for backend in ("cuda", "vulkan"):
        enabled = "ON" if backend in args.backends else "OFF"
        configure += [f"-DSA3_{backend.upper()}={enabled}", f"-DGGML_{backend.upper()}={enabled}"]
    for language, compiler in compilers.items():
        configure.append(f"-DCMAKE_{language}_COMPILER={compiler}")
    configure += ([f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}"] if args.cuda_arch
                  else ["-U", "CMAKE_CUDA_ARCHITECTURES"])
    run(*configure)
    if any(old_cache.get(f"CMAKE_{language}_COMPILER") not in {None, compiler}
           for language, compiler in compilers.items()):
        # Changing compilers makes CMake clear cached policy values and rerun
        # with defaults. Reapply the package policy before building anything.
        run(*configure)
    run("cmake", "--build", build, "--parallel", args.jobs)
    if not args.skip_tests:
        run("ctest", "--test-dir", build, "--output-on-failure", "--timeout", "120")
    package = build / "package"
    if package.exists():
        shutil.rmtree(package)
    core = package / "core"
    core.mkdir(parents=True)
    binary_dir = build / "bin"
    for tool in TOOLS:
        shutil.copy2(binary_dir / tool, core / tool)
    libraries = list(binary_dir.glob("libggml*.so*")) + list(binary_dir.glob("libsa3.so*"))
    for library in libraries:
        if not any(library.name.startswith(f"libggml-{backend}.") for backend in ("cuda", "vulkan")):
            shutil.copy2(library, core / library.name)
    if not (core / "libsa3.so").is_file() or not list(core.glob("libggml-cpu*.so*")):
        raise ValueError("missing core/CPU shared libraries")
    for source, name in (
        ("LICENSE", "LICENSE"), ("ggml/LICENSE", "LICENSE-ggml.txt"),
        ("vendor/cpp-httplib/LICENSE", "LICENSE-cpp-httplib.txt"),
        ("vendor/yyjson/LICENSE", "LICENSE-yyjson.txt"),
        ("vendor/signalsmith-stretch/LICENSE.txt", "LICENSE-signalsmith-stretch.txt"),
        ("vendor/signalsmith-linear/LICENSE.txt", "LICENSE-signalsmith-linear.txt"),
        ("docs/RUNTIME_RELEASE.md", "RUNTIME_README.md"),
        ("docs/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
        ("docs/SAT_SERVER.md", "SAT_SERVER.md"), ("docs/AUDIO_ANALYSIS.md", "AUDIO_ANALYSIS.md"),
        ("src/libsa3_v1.h", "libsa3_v1.h"), ("src/libsa3_training_v1.h", "libsa3_training_v1.h"),
        ("models.sh", "models.sh"),
    ):
        shutil.copy2(ROOT / source, core / name)
    for source in output("git", "ls-files", "prompts").splitlines():
        target = core / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)
    stages = {"core": core}
    for backend in dict.fromkeys(args.backends):
        stage = package / backend
        stage.mkdir()
        library = binary_dir / f"libggml-{backend}.so"
        shutil.copy2(library, stage / library.name)
        stages[backend] = stage
    cache = cache_values(build)
    info = {
        "service": "sa3", "version": args.version, "platform": platform_key,
        "commit": output("git", "rev-parse", "HEAD"), "ggml_commit": ggml, "dirty": dirty,
        "build_flavor": "gpu-smoke" if args.cuda_arch else "portable",
        "backends": list(dict.fromkeys(args.backends)),
        "cuda_architecture_policy": (args.cuda_arch or "ggml-default") if "cuda" in args.backends else None,
        "cuda_architectures": cache.get("CMAKE_CUDA_ARCHITECTURES"),
        "cuda_codegen": sorted(set(re.findall(r'--generate-code=[^\s"]+', "\n".join(
            path.read_text() for path in [*build.glob("**/ggml-cuda.dir/flags.make"), build / "build.ninja"]
            if path.is_file())))) if "cuda" in args.backends else [],
        "cuda_compiler": output(cache["CMAKE_CUDA_COMPILER"], "--version") if "cuda" in args.backends else None,
        "compiler": output(cache["CMAKE_CXX_COMPILER"], "--version").splitlines()[0],
        "cmake": output("cmake", "--version").splitlines()[0],
        "system": platform.freedesktop_os_release(),
        "dependencies": {name: elf_info(stage) for name, stage in stages.items()},
        "runtime_policy": "system GPU driver/loader and CUDA user libraries; no redistributables bundled",
        "tests_run": not args.skip_tests, "gpu_probe_required": args.require_gpu,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (core / "BUILD-INFO.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    archives = {}
    for name, stage in stages.items():
        archive = package / f"sa3-{args.version}-{platform_key}-{name}.zip"
        write_zip(stage, archive)
        archives[name] = archive
    # Test the archives after extracting them elsewhere, not the original build/stage.
    for backend in [None, *dict.fromkeys(args.backends)]:
        with tempfile.TemporaryDirectory(prefix=f"check-{backend or 'core'}-", dir=build) as scratch:
            extracted = Path(scratch) / "runtime"
            extracted.mkdir()
            unpack(archives["core"], extracted)
            if backend:
                unpack(archives[backend], extracted)
            check_loader_paths(extracted)
            command = ["python3", ROOT / "ci/check-runtime-package.py", extracted, args.version,
                       "--expect-platform", platform_key]
            if backend and args.require_gpu:
                command += ["--require-backend", backend]
            run(*command)
    # Only completed, verified archives reach dist; the checksum file is written last.
    sums_text = "".join(f"{sha256(path)}  {path.name}\n" for path in archives.values())
    for archive in archives.values():
        shutil.copy2(archive, destination / (archive.name + ".partial"))
        (destination / (archive.name + ".partial")).replace(destination / archive.name)
    sums = destination / f"SHA256SUMS-{platform_key}"
    sums.with_suffix(".partial").write_text(sums_text, encoding="utf-8")
    sums.with_suffix(".partial").replace(sums)
    print(f"Verified packages: {destination}")
    run("du", "-sh", build, destination)


if __name__ == "__main__":
    main()
