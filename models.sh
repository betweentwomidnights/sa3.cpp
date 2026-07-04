#!/usr/bin/env bash
# Download the sa3.cpp GGUF model set from HuggingFace (public repos) with curl — no Python.
#
# Usage: ./models.sh [--variant medium|small-music|small-sfx] [--encoding f16|f32]
#                    [--namespace <hf-user>] [--out DIR]
#   default: medium f16 into ./models
#
# Grabs one variant's DiT + SAME at the chosen encoding, its (always-F32) conditioner, plus the shared
# T5Gemma encoder + tokenizer. Cross-platform via git-bash on Windows (or use models.cmd).
set -eu

VARIANT="medium"
ENCODING="f16"
NAMESPACE="thepatch"
OUT="models"

while [ $# -gt 0 ]; do
  case "$1" in
    --variant)   VARIANT="$2"; shift ;;
    --encoding)  ENCODING="$2"; shift ;;
    --namespace) NAMESPACE="$2"; shift ;;
    --out)       OUT="$2"; shift ;;
    -h|--help)   sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

case "$VARIANT" in
  medium)                DIT_SIZE="1.5B"; SAME="same-l" ;;
  small-music|small-sfx) DIT_SIZE="0.5B"; SAME="same-s" ;;
  *) echo "unknown variant: $VARIANT (medium|small-music|small-sfx)" >&2; exit 1 ;;
esac

ENC=$(printf '%s' "$ENCODING" | tr '[:lower:]' '[:upper:]')   # F16 / F32
VAR_REPO="$NAMESPACE/stable-audio-3-$VARIANT-GGUF"
SHARED="$NAMESPACE/t5gemma-b-b-ul2-GGUF"
BASE="stable-audio-3-$VARIANT"

mkdir -p "$OUT"

dl() {   # dl <repo> <filename>
  local repo="$1" file="$2" dst="$OUT/$file"
  if [ -f "$dst" ]; then
    echo "[check/resume] $file"
  else
    echo "[download] $repo/$file"
  fi
  curl -fL --retry 3 --continue-at - -o "$dst" "https://huggingface.co/$repo/resolve/main/$file"
}

dl "$VAR_REPO" "$BASE-dit-$DIT_SIZE-v1.0-$ENC.gguf"
dl "$VAR_REPO" "$BASE-$SAME-v1.0-$ENC.gguf"
dl "$VAR_REPO" "$BASE-conditioner-v1.0-F32.gguf"
dl "$SHARED"   "t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf"
dl "$SHARED"   "t5gemma-b-b-ul2-v1.0-vocab.gguf"

echo "[done] $VARIANT ($ENCODING) -> $OUT/"
