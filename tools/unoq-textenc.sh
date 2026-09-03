#!/usr/bin/env bash
# Local UNO Q experiment only. Each invocation gets a fresh artifact directory.
set -euo pipefail
mode=${1:-safe}
layers=${2:-1}
repeats=${3:-3}
seq=${4:-256}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
models=${SA3_MODELS_DIR:-/home/arduino/sa3-models}
weights=${SA3_UNOQ_T5_WEIGHTS:-$models/t5gemma-b-b-ul2-encoder-0.3B-v1.0-Q4_K_M_OPENCL.gguf}
build=${SA3_UNOQ_BUILD:-$repo/build-opencl-profile}
results=${SA3_UNOQ_RESULTS:-/home/arduino/sa3-experiments}
case "$mode" in safe|cpu|async) ;; *) echo 'mode must be safe, cpu, or async' >&2; exit 2;; esac
mkdir -p -- "$results"
run=$(mktemp -d "$results/textenc-$mode-L$layers-S$seq-XXXXXX")
printf 'Artifacts: %s\n' "$run"
cd -- "$run"
{
    date -u
    printf 'mode=%s layers=%s repeats=%s seq=%s build=%s\n' "$mode" "$layers" "$repeats" "$seq" "$build"
    printf 'GGML_OPENCL_RUSTICL_TILED_K=%s\n' "${GGML_OPENCL_RUSTICL_TILED_K-unset}"
    printf 'GGML_OPENCL_RUSTICL_TILED_FLOAT=%s weights=%s\n' "${GGML_OPENCL_RUSTICL_TILED_FLOAT-unset}" "$weights"
    printf 'GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS=%s\n' "${GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS:-256}"
    sha256sum "$weights"
    git -C "$repo" log -1 --format='%H %s'
    git -C "$repo" status --short
    git -C "$repo/ggml" log -1 --format='%H %s'
    git -C "$repo/ggml" status --short
    sha256sum "$build/bin/sa3-textenc" "$repo/tools/sa3-textenc.cpp" \
        "$repo/ggml/src/ggml-opencl/ggml-opencl.cpp"
    grep -E 'GGML_OPENCL|SA3_OPENCL|CMAKE_BUILD_TYPE:' "$build/CMakeCache.txt"
} > source-state.txt
export LD_LIBRARY_PATH=/home/arduino/mesa26/root/usr/lib/aarch64-linux-gnu
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export LIBCLC_PATH=/home/arduino/mesa26/root/usr/lib/clc
export RUSTICL_FEATURES=intel
export GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT=1
export GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS=${GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS:-256}
unset GGML_OPENCL_RUSTICL_FINISH_INTERVAL GGML_OPENCL_DISPATCH_TRACE SA3_DEVICE SA3_GPU
if [[ $mode == cpu ]]; then
    export SA3_DEVICE=cpu
else
    export GGML_OPENCL_DISPATCH_TRACE="$run/dispatch.tsv"
    if [[ $mode == async ]]; then
        export GGML_OPENCL_RUSTICL_FINISH_INTERVAL=0
        printf 'WARNING: asynchronous diagnostic run may fault the GPU.\n' >&2
    fi
fi
# No core dumps on the small eMMC; timeout bounds a hung diagnostic without rebooting the board.
ulimit -c 0
set +e
/usr/bin/time -v timeout --signal=TERM --kill-after=10s 300s \
    "$build/bin/sa3-textenc" \
    --gguf "$weights" \
    --tok "$models/t5gemma-b-b-ul2-v1.0-vocab.gguf" \
    --prompt 'short electronic tone' --seq "$seq" --layers "$layers" \
    --repeat "$repeats" --threads 4 --out "$run" > run.log 2>&1
status=$?
set -e
tail -30 run.log
printf 'Exit: %d; artifacts: %s\n' "$status" "$run"
exit "$status"
