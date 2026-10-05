#!/usr/bin/env bash
# Build the flat universal macOS runtime: arm64 Metal/CPU and x86_64 CPU.
# Uses shared ggml dylibs, rewrites imports to @loader_path, then signs every
# executable/dylib with Developer ID and notarizes the zip. Bare binaries in a
# zip cannot be stapled; Gatekeeper checks the notarization ticket online.
# SA3_SIGN_IDENTITY, SA3_NOTARY_KEY (P8 path), SA3_NOTARY_KEY_ID and
# SA3_NOTARY_ISSUER provide credentials. --require-signing rejects missing ones.
# Usage: ci/package-macos.sh --version v0.1.1 [--jobs 3] [--require-signing]
# Builds and tests both slices on Apple Silicon with Rosetta (macOS 15+).
set -euo pipefail

VERSION=""
BUILD_DIR="build-dist-macos"
OUT_DIR="dist"
SKIP_TESTS=0
REQUIRE_SIGNING=0
JOBS=4
DEPLOYMENT_TARGET="13.3"

fail() { echo "package-macos: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift ;;
    --build-dir) BUILD_DIR="$2"; shift ;;
    --out-dir) OUT_DIR="$2"; shift ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --require-signing) REQUIRE_SIGNING=1 ;;
    --jobs) JOBS="$2"; shift ;;
    -h|--help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "--version must look like v0.1.0"
[ "$(uname -s)" = Darwin ] || fail "this script builds the macOS package; run it on a Mac"
[ "$(uname -m)" = arm64 ] || fail "build on Apple Silicon: only there can the Metal slice be checked"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be positive"

# The script clears its staging and output folders, so keep both inside this checkout.
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$ROOT/$1" ;; esac; }
BUILD_PATH="$(abspath "$BUILD_DIR")"
OUT_PATH="$(abspath "$OUT_DIR")"
for p in "$BUILD_PATH" "$OUT_PATH"; do
  case "$p/" in "$ROOT"/?*) ;; *) fail "$p must be inside $ROOT" ;; esac
  case "$p" in *..*) fail "$p must not contain .." ;; esac
done


SIGN=0
if [ -n "${SA3_SIGN_IDENTITY:-}" ]; then SIGN=1; fi
NOTARIZE=0
if [ -n "${SA3_NOTARY_KEY:-}" ] && [ -n "${SA3_NOTARY_KEY_ID:-}" ] && [ -n "${SA3_NOTARY_ISSUER:-}" ]; then
  NOTARIZE=1
  [ -f "$SA3_NOTARY_KEY" ] || fail "SA3_NOTARY_KEY $SA3_NOTARY_KEY does not exist"
fi
if [ "$REQUIRE_SIGNING" = 1 ] && { [ "$SIGN" = 0 ] || [ "$NOTARIZE" = 0 ]; }; then
  fail "--require-signing: SA3_SIGN_IDENTITY and the SA3_NOTARY_* variables must all be set"
fi
if [ "$NOTARIZE" = 1 ] && [ "$SIGN" = 0 ]; then fail "notarizing needs SA3_SIGN_IDENTITY too"; fi

command -v cmake >/dev/null || fail "cmake is not on PATH"
CTEST="$(dirname "$(command -v cmake)")/ctest"
ROSETTA=0
if arch -x86_64 /usr/bin/true 2>/dev/null; then ROSETTA=1; fi

echo "sa3.cpp  $(git rev-parse --short HEAD)"
echo "ggml       $(git -C ggml rev-parse HEAD)"
echo "version    $VERSION"
echo "macOS      $(sw_vers -productVersion), deployment target $DEPLOYMENT_TARGET"
# sed reads all of xcodebuild's output: with head, xcodebuild crashes on the closed pipe.
XCODE="$(xcodebuild -version 2>/dev/null | sed -n 1p)"
echo "xcode      $XCODE"
echo "signing    $([ "$SIGN" = 1 ] && echo "Developer ID" || echo "ad-hoc (local check only)")"
echo "notarize   $([ "$NOTARIZE" = 1 ] && echo yes || echo no)"
echo "rosetta    $([ "$ROSETTA" = 1 ] && echo "yes: the x86_64 slice is tested" || echo "no: the x86_64 slice is built but not run")"

# --- build and stage ---------------------------------------------------------
build_slice() {
  cmake -S . -B "$BUILD_PATH/$1" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="$1" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON \
    -DGGML_METAL="$2" -DSA3_METAL="$2" -DGGML_METAL_EMBED_LIBRARY=ON \
    -DSA3_CUDA=OFF -DSA3_VULKAN=OFF -DSA3_BUILD_TOOLS=ON \
    -DSA3_BUILD_SAT=ON -DSA3_PRIVATE_GGML=OFF -DBUILD_TESTING=ON
  cmake --build "$BUILD_PATH/$1" --config Release --parallel "$JOBS"
}
build_slice arm64 ON
build_slice x86_64 OFF
ARM="$BUILD_PATH/arm64/bin"
X86="$BUILD_PATH/x86_64/bin"
if [ "$SKIP_TESTS" = 0 ]; then
  "$CTEST" --test-dir "$BUILD_PATH/arm64" -C Release --output-on-failure
  if [ "$ROSETTA" = 1 ]; then
    "$CTEST" --test-dir "$BUILD_PATH/x86_64" -C Release --output-on-failure
  fi
fi
STAGE="$BUILD_PATH/package/macos"
rm -rf "$BUILD_PATH/package"
mkdir -p "$STAGE"
TOOLS=(sa3-server sat-server sat-generate sa3-train-web sa3-generate sa3-textmusic
       sa3-codec sa3-textenc sa3-dit sa3-tokenize sa3-train sa3-lora-convert
       sa3-quantize sa3-quant-check sa3-quant-eval)
BINARIES=()
for tool in "${TOOLS[@]}"; do
  lipo -create "$ARM/$tool" "$X86/$tool" -output "$STAGE/$tool"
  BINARIES+=("$tool")
done
# Merge each dylib's soname once. Metal is arm64-only; Intel has the CPU backend.
while IFS= read -r arm; do
  name="$(basename "$arm")"
  if [ "$name" != libsa3.dylib ] && ! [[ "$name" =~ ^libggml[-a-z]*\.0\.dylib$ ]]; then continue; fi
  x86="$(find -L "$BUILD_PATH/x86_64" -type f -name "$name" -print -quit)"
  if [ -n "$x86" ]; then
    lipo -create "$arm" "$x86" -output "$STAGE/$name"
  elif [ "$name" = libggml-metal.0.dylib ]; then
    cp "$arm" "$STAGE/$name"
  else
    fail "no Intel counterpart for $name"
  fi
  BINARIES+=("$name")
done < <(find -L "$BUILD_PATH/arm64" -type f -name '*.dylib')
[ -f "$STAGE/libsa3.dylib" ] || fail "no libsa3.dylib staged"
# Rewrite all non-system imports to the flat package before codesigning.
for name in "${BINARIES[@]}"; do
  file="$STAGE/$name"
  case "$name" in *.dylib) install_name_tool -id "@rpath/$name" "$file" ;; esac
  while IFS= read -r dep; do
    case "$dep" in /usr/lib/*|/System/Library/*) continue ;; esac
    [ "$dep" != "@rpath/$name" ] || continue
    base="$(basename "$dep")"
    [ -f "$STAGE/$base" ] || fail "$name depends on unstaged $dep"
    install_name_tool -change "$dep" "@loader_path/$base" "$file"
  done < <(otool -L "$file" | awk '/^[[:space:]]/ {print $1}' | sort -u)
  codesign --force --sign - "$file"
done
for tool in sa3-server sat-server; do
  for a in arm64 x86_64; do
    [ "$a" = x86_64 ] && [ "$ROSETTA" = 0 ] && continue
    reported="$(arch -"$a" "$STAGE/$tool" --version 2>/dev/null)"
    [ "v$reported" = "$VERSION" ] || fail "$tool reports $reported instead of $VERSION"
  done
done
cp LICENSE models.sh "$STAGE/"
cp src/libsa3_v1.h src/libsa3_training_v1.h "$STAGE/"
cp docs/RUNTIME_RELEASE.md "$STAGE/RUNTIME_README.md"
cp docs/SAT_SERVER.md docs/THIRD_PARTY_NOTICES.md "$STAGE/"
cp ggml/LICENSE "$STAGE/LICENSE-ggml.txt"
cp vendor/cpp-httplib/LICENSE "$STAGE/LICENSE-cpp-httplib.txt"
cp vendor/yyjson/LICENSE "$STAGE/LICENSE-yyjson.txt"
cp vendor/signalsmith-stretch/LICENSE.txt "$STAGE/LICENSE-signalsmith-stretch.txt"
cp vendor/signalsmith-linear/LICENSE.txt "$STAGE/LICENSE-signalsmith-linear.txt"
cp -R prompts "$STAGE/"
DIRTY=false
[ -z "$(git status --porcelain --untracked-files=no)" ] || DIRTY=true
cat > "$STAGE/BUILD-INFO.json" <<EOF
{
  "service": "sa3", "version": "$VERSION",
  "commit": "$(git rev-parse HEAD)", "dirty": $DIRTY,
  "ggml_commit": "$(git -C ggml rev-parse HEAD)",
  "platform": "macos-universal",
  "architectures": {"arm64": ["metal", "cpu"], "x86_64": ["cpu"]},
  "macos_deployment_target": "$DEPLOYMENT_TARGET", "xcode": "$XCODE",
  "signed": $([ "$SIGN" = 1 ] && echo true || echo false),
  "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
# --- sign -------------------------------------------------------------------
#
# Hardened runtime with no entitlements: nothing here JITs or loads foreign code, and
# Metal compiles its embedded shaders without one. The secure timestamp is required
# for notarization.

if [ "$SIGN" = 1 ]; then
  for f in "${BINARIES[@]}"; do
    codesign --force --options runtime --timestamp --sign "$SA3_SIGN_IDENTITY" "$STAGE/$f"
    codesign --verify --strict --verbose=2 "$STAGE/$f"
  done
  codesign -dvv "$STAGE/libsa3.dylib" 2>&1 | grep -E '^(Authority|TeamIdentifier|Timestamp)=' | head -3
fi

# Check startup and the linked ABI contract from outside the package directory.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/sa3-package-check.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
for a in arm64 x86_64; do
  [ "$a" = x86_64 ] && [ "$ROSETTA" = 0 ] && continue
  cp "$BUILD_PATH/$a/bin/sa3-lib-v1-contract" "$SCRATCH/contract-$a"
  # Point the scratch test at the packaged dylib, rather than its build directory.
  while IFS= read -r dep; do
    case "$dep" in *libsa3.dylib) install_name_tool -change "$dep" "$STAGE/libsa3.dylib" "$SCRATCH/contract-$a" ;; esac
  done < <(otool -L "$SCRATCH/contract-$a" | awk '/^[[:space:]]/ {print $1}')
  codesign --force --sign - "$SCRATCH/contract-$a"
  (cd "$SCRATCH"; arch -"$a" ./contract-"$a"; arch -"$a" "$STAGE/sa3-server" --props > "props-$a.json")
done
# --- zip, notarize, checksum --------------------------------------------------

mkdir -p "$OUT_PATH"
for f in "$OUT_PATH"/*; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in
    sa3-v*-macos-universal.zip|SHA256SUMS-macos) rm -f "$f" ;;
    *) fail "$OUT_PATH holds files that are not macOS package outputs: $(basename "$f")" ;;
  esac
done

ZIP_NAME="sa3-$VERSION-macos-universal.zip"
ZIP="$OUT_PATH/$ZIP_NAME"
# Flat at the root, like the Windows zips. ditto keeps the signatures and modes intact.
ditto -c -k --norsrc "$STAGE" "$ZIP"

if [ "$NOTARIZE" = 1 ]; then
  echo "notarizing $ZIP_NAME"
  result="$(xcrun notarytool submit "$ZIP" --key "$SA3_NOTARY_KEY" --key-id "$SA3_NOTARY_KEY_ID" \
            --issuer "$SA3_NOTARY_ISSUER" --wait --timeout 30m --output-format json)" || true
  status="$(printf '%s' "$result" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)"
  id="$(printf '%s' "$result" | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)"
  echo "notarization $id: ${status:-no status}"
  if [ "$status" != "Accepted" ]; then
    echo "$result" >&2
    [ -n "$id" ] && xcrun notarytool log "$id" --key "$SA3_NOTARY_KEY" --key-id "$SA3_NOTARY_KEY_ID" \
                      --issuer "$SA3_NOTARY_ISSUER" >&2 || true
    fail "notarization was not accepted"
  fi
fi

( cd "$OUT_PATH" && shasum -a 256 "$ZIP_NAME" > SHA256SUMS-macos )
printf '%-36s %8s MB  %s\n' "$ZIP_NAME" "$(du -m "$ZIP" | cut -f1)" "$(cut -d' ' -f1 "$OUT_PATH/SHA256SUMS-macos")"
echo "package -> $OUT_PATH"
