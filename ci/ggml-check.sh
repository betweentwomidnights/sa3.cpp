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

if [ ! -f ggml/CMakeLists.txt ]; then
    echo "ci/ggml-check.sh: ggml/ is empty; clone with --recurse-submodules" >&2
    exit 1
fi

# Say which revision is under test. On a bump PR this is the whole point of the
# run, and it is the first thing worth seeing in the log.
printf 'sa3.cpp  %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf 'ggml     %s\n' "$(git -C ggml rev-parse HEAD 2>/dev/null || echo unknown)"

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
    -DGGML_METAL=OFF
cmake --build "$build" --config Release -j "$jobs"

# --output-on-failure so a red test explains itself in the job log rather than
# only naming which one died.
# sa3-dit-lin-functional-test stays excluded, but for a different reason than
# before. Its inputs are portable now; what it reports is a real divergence.
#
# Comparing the f16-base analytic gradient against the f32-base one, B diverges
# by 0.2% and the magnitude by 0.08%, both about what quantising the base
# weight to f16 should cost. A diverges by 7.2%, roughly thirty-five times its
# siblings. The old oracle could not have shown this: it finite-differenced an
# f16 forward, so it was measuring the quantiser's staircase rather than the
# derivative, and its tolerances only ever held for the particular data MSVC's
# std::normal_distribution happened to lay down.
#
# Whether 7.2% is expected for this path is a question about sa3's DoRA
# numerics, not about CI, and it is not one to settle by widening a tolerance
# until the check goes quiet. Remove this exclusion once that is answered.
ctest --test-dir "$build" --build-config Release --output-on-failure \
    -E sa3-dit-lin-functional-test
