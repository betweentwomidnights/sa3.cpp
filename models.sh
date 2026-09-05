#!/usr/bin/env bash
# Download the sa3.cpp GGUF model set from HuggingFace (public repos) with curl — no Python.
#
# Usage: ./models.sh [--variant medium|small-music|small-sfx] [--encoding f16|f32|q4_k_m|q5_k_m|q8_0]
#                    [--t5-encoding f16|f32|q8_0] [--ae-encoding f16|f32|q4_k_m|q5_k_m|q8_0]
#                    [--training-base] [--namespace <hf-user>] [--out DIR] [--dry-run]
#        ./models.sh --sat [--sat-model saos] [--saos-variant arc|kickbass|jerry-grunge]
#   default: medium f16 DiT, f32 autoencoder, into ./models
#
# Grabs one variant's DiT at the chosen encoding, its autoencoder (SAME) at --ae-encoding, the
# (always-F32) conditioner, plus the shared T5Gemma encoder + tokenizer. --training-base also grabs
# the matching base DiT used by sa3-train.
# Cross-platform via git-bash on Windows (or use models.cmd).
set -eu

VARIANT="medium"
ENCODING="f16"
ENCODING_SET=0
# The text encoder is resolved apart from the DiT: F16 is equivalent to F32 at half the size.
T5_ENCODING="f16"
T5_ENCODING_SET=0
# So is the autoencoder, and it defaults to F32. SAME used to ride on --encoding, so asking for a
# quantized DiT quietly fetched a quantized SAME too -- the last net the audio crosses, and the one
# continuation/transform cross twice per iteration. It is also the cheap one to keep: SAME-S is
# 413 MB at F32 against 72 MB at Q4_K_M, beside a DiT that dominates the footprint either way.
# Quantized autoencoders are still published -- ask for one with --ae-encoding.
AE_ENCODING="f32"
AE_ENCODING_SET=0
SAT=0
SAT_MODEL="saos"
SAOS_VARIANT="arc"
NAMESPACE="thepatch"
OUT="models"
TRAINING_BASE=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --variant)   VARIANT="$2"; shift ;;
    --encoding)  ENCODING="$2"; ENCODING_SET=1; shift ;;
    --t5-encoding) T5_ENCODING="$2"; T5_ENCODING_SET=1; shift ;;
    --ae-encoding) AE_ENCODING="$2"; AE_ENCODING_SET=1; shift ;;
    --sat)       SAT=1 ;;
    --sat-model) SAT_MODEL="$2"; shift ;;
    --saos-variant) SAOS_VARIANT="$2"; shift ;;
    --namespace) NAMESPACE="$2"; shift ;;
    --out)       OUT="$2"; shift ;;
    --training-base) TRAINING_BASE=1 ;;
    --dry-run)   DRY_RUN=1 ;;
    -h|--help)   sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

# PowerShell may resolve `bash` to WSL. Normalize a Windows drive path so
# `--out C:/models` does not become a literal relative `C:` directory there.
case "$OUT" in
  [A-Za-z]:/*)
    if command -v wslpath >/dev/null 2>&1; then
      OUT=$(wslpath -u "$OUT")
    fi
    ;;
esac

sat_dl() {   # sat_dl <repo> <filename>
  local repo="$1" file="$2" dst="$OUT/$2"
  mkdir -p "$(dirname "$dst")"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[plan] https://huggingface.co/$repo/resolve/main/$file -> $dst"
    return
  fi
  if [ -f "$dst" ]; then echo "[check/resume] $file"; else echo "[download] $repo/$file"; fi
  curl -fL --retry 3 --continue-at - -o "$dst" "https://huggingface.co/$repo/resolve/main/$file"
}

if [ "$SAT" -eq 1 ]; then
  [ "$TRAINING_BASE" -eq 0 ] || { echo "--training-base applies to SA3, not --sat" >&2; exit 2; }
  [ "$SAT_MODEL" = "saos" ] || { echo "unknown --sat-model '$SAT_MODEL' (currently: saos)" >&2; exit 2; }
  [ "$ENCODING_SET" -eq 1 ] || ENCODING="q5_k_m"
  [ "$T5_ENCODING_SET" -eq 1 ] || T5_ENCODING="$ENCODING"
  [ "$AE_ENCODING_SET" -eq 1 ] || AE_ENCODING="$ENCODING"
  case "$ENCODING" in f16|F16) ENC=F16;; q8_0|Q8_0) ENC=Q8_0;; q5_k_m|Q5_K_M) ENC=Q5_K_M;; q4_k_m|Q4_K_M) ENC=Q4_K_M;; *) echo "unsupported SAOS encoding '$ENCODING'" >&2; exit 2;; esac
  case "$T5_ENCODING" in f16|F16) T5_ENC=F16;; q8_0|Q8_0) T5_ENC=Q8_0;; q5_k_m|Q5_K_M) T5_ENC=Q5_K_M;; q4_k_m|Q4_K_M) T5_ENC=Q4_K_M;; *) echo "unsupported SAOS T5 encoding '$T5_ENCODING'" >&2; exit 2;; esac
  case "$AE_ENCODING" in f16|F16) AE_ENC=F16;; q8_0|Q8_0) AE_ENC=Q8_0;; q5_k_m|Q5_K_M) AE_ENC=Q5_K_M;; q4_k_m|Q4_K_M) AE_ENC=Q4_K_M;; *) echo "unsupported SAOS Oobleck encoding '$AE_ENCODING'" >&2; exit 2;; esac
  case "$SAOS_VARIANT" in
    arc) SAOS_DIT="stable-audio-open-small-dit-0.3B-v1.0-$ENC.gguf" ;;
    kickbass) SAOS_DIT="finetunes/kickbass/kickbass-v1-e257-dit-0.3B-v1.0-$ENC.gguf" ;;
    jerry-grunge) SAOS_DIT="finetunes/jerry-grunge/jerry-grunge-bs64-step3000-dit-0.3B-v1.0-$ENC.gguf" ;;
    *) echo "unknown SAOS variant '$SAOS_VARIANT'" >&2; exit 2 ;;
  esac
  SAOS_REPO="$NAMESPACE/stable-audio-open-small-GGUF"
  sat_dl "$SAOS_REPO" "$SAOS_DIT"
  sat_dl "$SAOS_REPO" "t5-base-encoder-0.1B-v1.0-$T5_ENC.gguf"
  sat_dl "$SAOS_REPO" "stable-audio-open-small-oobleck-v1.0-$AE_ENC.gguf"
  echo "[done] SAOS $SAOS_VARIANT ($ENC) -> $OUT"
  exit 0
fi

case "$VARIANT" in
  medium)                DIT_SIZE="1.5B"; SAME="same-l" ;;
  small-music|small-sfx) DIT_SIZE="0.5B"; SAME="same-s" ;;
  *) echo "unknown variant: $VARIANT (medium|small-music|small-sfx)" >&2; exit 1 ;;
esac

ENC=$(printf '%s' "$ENCODING" | tr '[:lower:]' '[:upper:]')   # F16 / F32
# Training bases are published F16, F32 and Q4_K_M -- training on a quantized base works on every
# backend. Q5_K_M/Q8_0 bases are not published, so those requests pull an F16 base DiT instead. The
# [fetch] lines below show which one was resolved.
BASE_ENC="$ENC"
case "$T5_ENCODING" in
  f16|F16)   T5_ENC="F16" ;;
  f32|F32)   T5_ENC="F32" ;;
  q8_0|Q8_0) T5_ENC="Q8_0" ;;
  *) echo "unknown --t5-encoding '$T5_ENCODING' (expected f16|f32|q8_0)" >&2; exit 2 ;;
esac
case "$ENC" in
  F16|F32|Q4_K_M) ;;
  Q5_K_M|Q8_0) BASE_ENC="F16" ;;
  *) echo "unknown encoding: $ENCODING (f16|f32|q4_k_m|q5_k_m|q8_0)" >&2; exit 1 ;;
esac
case "$AE_ENCODING" in
  f16|F16)     AE_ENC="F16" ;;
  f32|F32)     AE_ENC="F32" ;;
  q4_k_m|Q4_K_M) AE_ENC="Q4_K_M" ;;
  q5_k_m|Q5_K_M) AE_ENC="Q5_K_M" ;;
  q8_0|Q8_0)   AE_ENC="Q8_0" ;;
  *) echo "unknown --ae-encoding '$AE_ENCODING' (expected f16|f32|q4_k_m|q5_k_m|q8_0)" >&2; exit 2 ;;
esac
VAR_REPO="$NAMESPACE/stable-audio-3-$VARIANT-GGUF"
BASE_REPO="$NAMESPACE/stable-audio-3-$VARIANT-base-GGUF"
SHARED="$NAMESPACE/t5gemma-b-b-ul2-GGUF"
BASE="stable-audio-3-$VARIANT"

mkdir -p "$OUT"

dl() {   # dl <repo> <filename>
  local repo="$1" file="$2"
  local dst="$OUT/$file"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[plan] https://huggingface.co/$repo/resolve/main/$file -> $dst"
    return
  fi
  if [ -f "$dst" ]; then
    echo "[check/resume] $file"
  else
    echo "[download] $repo/$file"
  fi
  curl -fL --retry 3 --continue-at - -o "$dst" "https://huggingface.co/$repo/resolve/main/$file"
}

dl "$VAR_REPO" "$BASE-dit-$DIT_SIZE-v1.0-$ENC.gguf"
if [ "$TRAINING_BASE" -eq 1 ]; then
  dl "$BASE_REPO" "$BASE-base-dit-$DIT_SIZE-v1.0-$BASE_ENC.gguf"
fi
dl "$VAR_REPO" "$BASE-$SAME-v1.0-$AE_ENC.gguf"
dl "$VAR_REPO" "$BASE-conditioner-v1.0-F32.gguf"
dl "$SHARED"   "t5gemma-b-b-ul2-encoder-0.3B-v1.0-$T5_ENC.gguf"
dl "$SHARED"   "t5gemma-b-b-ul2-v1.0-vocab.gguf"

if [ "$TRAINING_BASE" -eq 1 ]; then
  echo "[done] $VARIANT (DiT $ENC, SAME $AE_ENC) + training base -> $OUT"
else
  echo "[done] $VARIANT (DiT $ENC, SAME $AE_ENC) -> $OUT"
fi
