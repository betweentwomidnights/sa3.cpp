#!/usr/bin/env bash
# Launch sa3-server with the first sa3-server binary it finds. Extra args pass through,
# e.g. ./server.sh --model small-music --port 9000.  See docs/SERVER.md.
set -eu
BIN=""
for d in build-cuda build-metal build-vulkan build-hip build-all build; do
    if [ -x "$d/bin/sa3-server" ]; then BIN="$d/bin/sa3-server"; break; fi
    if [ -x "$d/bin/Release/sa3-server.exe" ]; then BIN="$d/bin/Release/sa3-server.exe"; break; fi
done
if [ -z "$BIN" ]; then echo "sa3-server not built — run ./build.sh <backend> first" >&2; exit 1; fi
exec "$BIN" --model medium --encoding f16 --port 8086 "$@"
