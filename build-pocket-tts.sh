#!/usr/bin/env bash
# build-pocket-tts.sh
# Build pocket-tts (Kyutai Pocket TTS in Rust/GGUF, rzafiamy/pocket-tts-rs) and
# install it as ./bin/pocket-tts for the `pocket-tts-server` backend.
#
# Usage: ./build-pocket-tts.sh [branch-or-tag] [--cpu|--metal]
#   (default)        CUDA build (needs nvcc; ~28x real time on an RTX 4090)
#   --cpu            CPU-only build (~7x real time on a desktop CPU)
#   --metal          Apple GPU build
# Environment:
#   POCKET_TTS_SRC     existing checkout to build instead of cloning
#   POCKET_TTS_REPO    git URL      (default: https://github.com/rzafiamy/pocket-tts-rs)
#   CUDA_COMPUTE_CAP   e.g. 89      (default: detected by pocket-tts-rs's build.sh)
#
# No root needed: building needs cargo (https://rustup.rs) and, for the CUDA
# build, the CUDA Toolkit (nvcc). Clones into a temp dir unless POCKET_TTS_SRC
# is set. Models: `bin/pocket-tts convert --variant french --voices all`
# writes french-q8_0.gguf (see CONFIG.md, backend pocket-tts-server).

set -euo pipefail

REF="main"
MODE="--cuda"
for arg in "$@"; do
    case "$arg" in
        --cpu) MODE="" ;;
        --metal) MODE="--metal" ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) REF="$arg" ;;
    esac
done
REPO="${POCKET_TTS_REPO:-https://github.com/rzafiamy/pocket-tts-rs}"
BIN_DIR="$(cd "$(dirname "$0")" && pwd)/bin"
export PATH="$HOME/.cargo/bin:$PATH"

command -v cargo >/dev/null || { echo "cargo not found: install Rust from https://rustup.rs" >&2; exit 1; }

if [ -n "${POCKET_TTS_SRC:-}" ]; then
    SRC="$POCKET_TTS_SRC"
else
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    git clone --depth 1 --branch "$REF" "$REPO" "$TMP/pocket-tts-rs"
    SRC="$TMP/pocket-tts-rs"
fi

# pocket-tts-rs's build.sh detects the GPU's compute capability and prints
# "artifact: <path>" as its last line.
ARTIFACT=$(cd "$SRC" && ./build.sh $MODE | tee /dev/stderr | sed -n 's/^artifact: //p' | tail -1)
[ -n "$ARTIFACT" ] || { echo "pocket-tts build produced no artifact" >&2; exit 1; }
mkdir -p "$BIN_DIR"
install -m 755 "$SRC/$ARTIFACT" "$BIN_DIR/pocket-tts"
"$BIN_DIR/pocket-tts" --version
echo "Installed $BIN_DIR/pocket-tts"
