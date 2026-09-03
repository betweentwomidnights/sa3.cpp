#!/usr/bin/env bash
# Explicitly filtered tests only: never run the full backend suite on the board.
set -euo pipefail
mode=${1:-test}
device=${2:-GPUOpenCL}
case "$mode" in test|perf) ;; *) exit 2;; esac
case "$device" in CPU|GPUOpenCL) ;; *) exit 2;; esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${SA3_UNOQ_BUILD:-$repo/build-opencl-profile}
results=${SA3_UNOQ_RESULTS:-/home/arduino/sa3-experiments}
mkdir -p -- "$results"
run=$(mktemp -d "$results/precision-$mode-$device-XXXXXX")
printf 'Artifacts: %s\n' "$run"
export LD_LIBRARY_PATH=/home/arduino/mesa26/root/usr/lib/aarch64-linux-gnu
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export LIBCLC_PATH=/home/arduino/mesa26/root/usr/lib/clc
export RUSTICL_FEATURES=intel
export GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT=1
export GGML_OPENCL_RUSTICL_TILED_K=1
export GGML_OPENCL_RUSTICL_TILED_FLOAT=1
export GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS=${GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS:-256}
unset GGML_OPENCL_RUSTICL_FINISH_INTERVAL GGML_OPENCL_DISPATCH_TRACE
filter='m=768,n=256,k=768'
if [[ $mode == test ]]; then filter='m=768,n=256,k=768|m=35,n=33,k=(256|37)'; fi
cd -- "$run"
{
    date -u
    printf 'mode=%s device=%s filter=%s\n' "$mode" "$device" "$filter"
    env | sort | grep -E 'GGML_OPENCL|RUSTICL|LIBCLC|OCL_ICD'
    git -C "$repo" rev-parse HEAD
    git -C "$repo/ggml" rev-parse HEAD
    git -C "$repo/ggml" diff --stat
    sha256sum "$build/bin/test-backend-ops" "$repo/ggml/src/ggml-opencl/ggml-opencl.cpp" \
        "$repo/ggml/src/ggml-opencl/kernels/mul_mat_f16_f32.cl"
} > source-state.txt
ulimit -c 0
set +e
/usr/bin/time -v timeout --signal=TERM --kill-after=10s 300s \
    "$build/bin/test-backend-ops" "$mode" -b "$device" -o MUL_MAT -p "$filter" > run.log 2>&1
status=$?
set -e
tail -40 run.log
printf 'Exit: %d; artifacts: %s\n' "$status" "$run"
exit "$status"
