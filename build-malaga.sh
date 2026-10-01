#!/usr/bin/env bash
# build-malaga.sh
# Build malaga (NLLB-200 translation server, rzafiamy/malaga) and install it
# as ./bin/malaga for the `malaga-server` backend.
#
# Usage: ./build-malaga.sh [branch-or-tag] [--cpu]
#   --cpu            CPU-only build (no nvcc needed; ~40 ms/token instead of ~0.8).
# Environment:
#   MALAGA_SRC         existing checkout to build instead of cloning
#   MALAGA_REPO        git URL      (default: https://github.com/rzafiamy/malaga)
#   CUDA_COMPUTE_CAP   e.g. 89      (default: detected by malaga's build.sh)
#
# No root needed: building needs cargo (https://rustup.rs) and, for the GPU
# build, the CUDA Toolkit (nvcc). Clones into a temp dir unless MALAGA_SRC is set.

set -euo pipefail

REF="main"
MODE="--cuda"
for arg in "$@"; do
    case "$arg" in
        --cpu) MODE="" ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) REF="$arg" ;;
    esac
done
REPO="${MALAGA_REPO:-https://github.com/rzafiamy/malaga}"
BIN_DIR="$(cd "$(dirname "$0")" && pwd)/bin"
export PATH="$HOME/.cargo/bin:$PATH"

command -v cargo >/dev/null || { echo "cargo not found: install Rust from https://rustup.rs" >&2; exit 1; }

if [ -n "${MALAGA_SRC:-}" ]; then
    SRC="$MALAGA_SRC"
else
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    git clone --depth 1 --branch "$REF" "$REPO" "$TMP/malaga"
    SRC="$TMP/malaga"
fi

# malaga's build.sh picks the GPU's compute capability and prints the artefact path.
(cd "$SRC" && ./build.sh $MODE)
mkdir -p "$BIN_DIR"
install -m 755 "$SRC/target/release/malaga" "$BIN_DIR/malaga"
"$BIN_DIR/malaga" --version
echo "Installed $BIN_DIR/malaga"
