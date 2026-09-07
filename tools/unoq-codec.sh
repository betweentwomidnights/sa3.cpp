#!/usr/bin/env bash
# Fixed-latent decoder diagnostic: same input for every backend/precision.
set -euo pipefail
device=${1:-gpu}
encoding=${2:-F16}
latent=${3:?provide an existing raw latent file}
frames=${4:-2}
repeats=${SA3_UNOQ_REPEATS:-3}
case "$device" in cpu|gpu) ;; *) exit 2;; esac
case "$encoding" in F16|F32|Q4_K_M|Q6_K|Q8_0) ;; *) exit 2;; esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${SA3_UNOQ_BUILD:-$repo/build-opencl-profile}
models=${SA3_MODELS_DIR:-/home/arduino/sa3-models}
weights="$models/stable-audio-3-small-music-same-s-v1.0-$encoding.gguf"
latent=$(realpath -- "$latent")
test -f "$latent" && test -f "$weights"
results=${SA3_UNOQ_RESULTS:-/home/arduino/sa3-experiments}
mkdir -p -- "$results"
run=$(mktemp -d "$results/codec-$device-$encoding-XXXXXX")
printf 'Artifacts: %s\n' "$run"
export LD_LIBRARY_PATH="${SA3_UNOQ_OPENCL_LIBRARY_DIR:-$build/ggml/src/ggml-opencl}:/home/arduino/mesa26/root/usr/lib/aarch64-linux-gnu"
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export LIBCLC_PATH=/home/arduino/mesa26/root/usr/lib/clc
export RUSTICL_FEATURES=intel
export GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT=1
export GGML_OPENCL_RUSTICL_TILED_K=1
export GGML_OPENCL_RUSTICL_TILED_FLOAT=1
export GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS=${GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS:-256}
export SA3_THREADS=4
unset GGML_OPENCL_RUSTICL_FINISH_INTERVAL GGML_OPENCL_DISPATCH_TRACE SA3_DEVICE SA3_GPU
if [[ $device == cpu ]]; then export SA3_DEVICE=cpu; fi
if [[ ${SA3_UNOQ_TRACE:-0} == 1 && $device == gpu ]]; then
    export GGML_OPENCL_DISPATCH_TRACE="$run/dispatch.tsv"
fi
cd -- "$run"
{
    date -u
    printf 'device=%s encoding=%s frames=%s repeats=%s\n' "$device" "$encoding" "$frames" "$repeats"
    env | sort | grep -E '^(GGML_OPENCL|RUSTICL|LIBCLC|OCL_ICD|SA3_THREADS)'
    git -C "$repo" rev-parse HEAD
    git -C "$repo/ggml" rev-parse HEAD
    sha256sum "$build/bin/sa3-codec" "$latent" "$weights"
} > source-state.txt
sha256sum "${SA3_UNOQ_OPENCL_LIBRARY_DIR:-$build/ggml/src/ggml-opencl}/libggml-opencl.so" \
    "$repo/ggml/src/ggml-opencl/kernels/mul_mm_q4_k_f32_l4_lm.cl" \
    "$repo/ggml/src/ggml-opencl/kernels/mul_mm_q6_k_f32_l4_lm.cl" >> source-state.txt
ulimit -c 0
set +e
/usr/bin/time -v timeout --signal=TERM --kill-after=10s 300s \
    "$build/bin/sa3-codec" --gguf "$weights" --in "$latent" --frames "$frames" \
    --repeat "$repeats" --out "$run" > run.log 2>&1
status=$?
set -e
tail -35 run.log
printf 'Exit: %d; artifacts: %s\n' "$status" "$run"
exit "$status"
