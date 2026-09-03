#!/usr/bin/env bash
# One-step smoke diagnostic, NOT a musical quality or serving benchmark.
set -euo pipefail
device=${1:-gpu}
dit=${2:-q4_k_m}
ae=${3:-q4_k_m}
t5=${4:-q4_k_m_opencl}
duration=${SA3_UNOQ_DURATION:-0.1}
chunk=${SA3_UNOQ_DECODE_CHUNK:-0}
overlap=${SA3_UNOQ_DECODE_OVERLAP:-16}
case "$device" in cpu|gpu) ;; *) exit 2;; esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${SA3_UNOQ_BUILD:-$repo/build-opencl-profile}
models=${SA3_MODELS_DIR:-/home/arduino/sa3-models}
results=${SA3_UNOQ_RESULTS:-/home/arduino/sa3-experiments}
mkdir -p -- "$results"
run=$(mktemp -d "$results/generate-$device-$dit-$ae-$t5-XXXXXX")
printf 'Artifacts: %s\n' "$run"
export LD_LIBRARY_PATH=/home/arduino/mesa26/root/usr/lib/aarch64-linux-gnu
export OCL_ICD_VENDORS=/etc/OpenCL/vendors
export LIBCLC_PATH=/home/arduino/mesa26/root/usr/lib/clc
export RUSTICL_FEATURES=intel
export GGML_OPENCL_RUSTICL_SUBGROUP32_COMPAT=1
export GGML_OPENCL_RUSTICL_TILED_K=1
export GGML_OPENCL_RUSTICL_TILED_FLOAT=1
export GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS=${GGML_OPENCL_RUSTICL_GEMM_MAX_COLUMNS:-256}
unset GGML_OPENCL_RUSTICL_FINISH_INTERVAL GGML_OPENCL_DISPATCH_TRACE SA3_DEVICE SA3_GPU
if [[ $device == cpu ]]; then export SA3_DEVICE=cpu; fi
export SA3_PROFILE=1 SA3_DUMP_GEN="$run" SA3_DUMP_COND="$run"
cd -- "$run"
{
    date -u
    printf 'device=%s dit=%s ae=%s t5=%s\n' "$device" "$dit" "$ae" "$t5"
    printf 'duration=%s decode_chunk=%s decode_overlap=%s\n' "$duration" "$chunk" "$overlap"
    env | sort | grep -E '^(GGML_OPENCL|SA3_|RUSTICL|LIBCLC|OCL_ICD)'
    git -C "$repo" rev-parse HEAD
    git -C "$repo/ggml" rev-parse HEAD
    git -C "$repo/ggml" diff --stat
    sha256sum "$build/bin/sa3-generate" "$repo/src/sa3_pipeline.h"
} > source-state.txt
git -C "$repo" diff --binary > source.patch
git -C "$repo/ggml" diff --binary > ggml-source.patch
ulimit -c 0
set +e
/usr/bin/time -v timeout --signal=TERM --kill-after=10s 300s \
    "$build/bin/sa3-generate" --model small-music --models-dir "$models" \
    --encoding "$dit" --ae-encoding "$ae" --t5-encoding "$t5" \
    --prompt 'short electronic tone' --duration "$duration" --duration-padding 0 \
    --decode-chunk-size "$chunk" --decode-overlap "$overlap" \
    --steps 1 --threads 4 --seed 1 --out "$run/audio.wav" > run.log 2>&1
status=$?
set -e
tail -45 run.log
printf 'Exit: %d; artifacts: %s\n' "$status" "$run"
exit "$status"
