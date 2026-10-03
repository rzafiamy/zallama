#!/usr/bin/env bash
# build-xtts.sh
# Build xtts (Coqui XTTS-v2 in Rust/GGUF, rzafiamy/xtts-rs) and install it as
# ./bin/xtts for the `xtts-server` backend.
#
# Usage: ./build-xtts.sh [branch-or-tag] [--cpu|--metal]
#   (default)        CUDA build (needs nvcc; ~13x real time, first audio in
#                    ~40 ms on an RTX 4090)
#   --cpu            CPU-only build (~0.4x real time: too slow to stream)
#   --metal          Apple GPU build
# Environment:
#   XTTS_SRC           existing checkout to build instead of cloning
#   XTTS_REPO          git URL      (default: https://github.com/rzafiamy/xtts-rs)
#   CUDA_COMPUTE_CAP   e.g. 89      (default: detected by xtts-rs's build.sh)
#
# No root needed: building needs cargo (https://rustup.rs) and, for the CUDA
# build, the CUDA Toolkit (nvcc). Clones into a temp dir unless XTTS_SRC is
# set. Models: `bin/xtts convert <coqui XTTS-v2 dir> -o xtts-v2-q4k.gguf
# --gpt-dtype q4k --no-cloning` (see CONFIG.md, backend xtts-server).

set -euo pipefail

REF="main"
MODE="--cuda"
for arg in "$@"; do
    case "$arg" in
        --cpu) MODE="" ;;
        --metal) MODE="--metal" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) REF="$arg" ;;
    esac
done
REPO="${XTTS_REPO:-https://github.com/rzafiamy/xtts-rs}"
BIN_DIR="$(cd "$(dirname "$0")" && pwd)/bin"
export PATH="$HOME/.cargo/bin:$PATH"

command -v cargo >/dev/null || { echo "cargo not found: install Rust from https://rustup.rs" >&2; exit 1; }

if [ -n "${XTTS_SRC:-}" ]; then
    SRC="$XTTS_SRC"
else
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    git clone --depth 1 --branch "$REF" "$REPO" "$TMP/xtts-rs"
    SRC="$TMP/xtts-rs"
fi

# xtts-rs's build.sh detects the GPU's compute capability and prints
# "artifact: <path>" as its last line.
ARTIFACT=$(cd "$SRC" && ./build.sh $MODE | tee /dev/stderr | sed -n 's/^artifact: //p' | tail -1)
[ -n "$ARTIFACT" ] || { echo "xtts build produced no artifact" >&2; exit 1; }
mkdir -p "$BIN_DIR"
install -m 755 "$SRC/$ARTIFACT" "$BIN_DIR/xtts"
"$BIN_DIR/xtts" --version
echo "Installed $BIN_DIR/xtts"
