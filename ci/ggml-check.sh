#!/bin/sh
# ABOUTME: Builds sa3 against whatever ggml revision is checked out and runs
# ABOUTME: its test suite, for pull requests that move the pointer.
#
# The shared ggml fork backs several projects (audiocraft.cpp, sa3.cpp,
# acestep.cpp, yue2.cpp), so a change there can break a consumer that the
# change's author never builds. Every consumer carries this script under the
# same path, which lets one job validate any of them without knowing how any
# of them are built: check the repo out, point its ggml at the revision under
# test, run ci/ggml-check.sh.
#
# CPU only and no model weights, so it runs on an ordinary hosted runner. That
# means a backend-specific change compiles here but is not exercised: the CUDA
# and Metal paths still need a machine that has one. What this does catch is
# the common case, a ggml change that alters an interface or a behaviour some
# consumer depends on.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

build=${BUILD_DIR:-build-ggml-check}
jobs=${JOBS:-}
if [ -z "$jobs" ]; then
    jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
fi

# Which backend to build. cpu is the default and the only one that runs the
# tests; cuda compiles the CUDA backend and stops there, because a hosted
# runner has no GPU to run it on. Compiling is most of the value: ggml-cuda.cu
# is not built at all by a cpu job, so a change to it reaches every consumer
# completely unchecked.
backend=${GGML_CHECK_BACKEND:-cpu}
case "$backend" in
    cpu|cuda) ;;
    *) echo "ci/ggml-check.sh: GGML_CHECK_BACKEND must be cpu or cuda" >&2; exit 1 ;;
esac

configure_extra=""
if [ "$backend" = cuda ]; then
    # A machine without an NVIDIA driver has no libcuda.so.1 to link against.
    # The toolkit ships a stub for exactly this; point the linker at it, and
    # only when the real one is absent, so this is a no-op on a real GPU host.
    configure_extra="-DSA3_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75-virtual"
    # libcuda.so.1 exactly: matching "libcuda" also matches libcudart.so, which
    # every toolkit image has, so a loose pattern decides a driver is present on
    # a machine that has none and leaves the link with nothing to resolve
    # cuMemCreate against.
    if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1' && [ -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then
        stubs="$PWD/${BUILD_DIR:-build-ggml-check}-cuda-stubs"
        mkdir -p "$stubs"
        [ -e "$stubs/libcuda.so.1" ] || ln -s /usr/local/cuda/lib64/stubs/libcuda.so "$stubs/libcuda.so.1"
        configure_extra="$configure_extra -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$stubs"
        LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
        export LIBRARY_PATH
    fi
fi

if [ ! -f ggml/CMakeLists.txt ]; then
    echo "ci/ggml-check.sh: ggml/ is empty; clone with --recurse-submodules" >&2
    exit 1
fi

# Say which revision is under test. On a bump PR this is the whole point of the
# run, and it is the first thing worth seeing in the log.
printf 'sa3.cpp  %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf 'ggml     %s\n' "$(git -C ggml rev-parse HEAD 2>/dev/null || echo unknown)"
printf 'backend  %s\n' "$backend"

# SA3_BUILD_SAT is left off, matching the default. It pulls in the classic
# stable-audio-tools component and its own tests, which is a larger build than
# this question needs: what is being asked is whether ggml still works here.
# GGML_METAL has to be turned off rather than merely left alone. ggml defaults
# it ON for Apple, and SA3_METAL=OFF only declines to force it ON, so a macOS
# runner would build and run Metal while this script claims to be CPU only.
# That aborted three of audiocraft.cpp's tests on a virtualised runner.
cmake -S . -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSA3_BUILD_TOOLS=ON \
    -DBUILD_TESTING=ON \
    -DGGML_METAL=OFF \
    $configure_extra
cmake --build "$build" --config Release -j "$jobs"

# A cuda build stops at compiling. Running its tests would need a GPU, and
# anything that can run on the cpu was already run by the cpu job.
if [ "$backend" != cpu ]; then
    echo "ci/ggml-check.sh: $backend build succeeded; tests need a GPU and are skipped"
    exit 0
fi

# --output-on-failure so a red test explains itself in the job log rather than
# only naming which one died.
ctest --test-dir "$build" --build-config Release --output-on-failure
