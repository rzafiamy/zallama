#!/usr/bin/env bash
# build-teleocr.sh
# Build teleocr (TeleOCR document parsing in Rust/GGUF, rzafiamy/teleocr-rs)
# and install it as ./bin/teleocr, with libpdfium.so for PDF input, for the
# `teleocr-server` backend.
#
# Usage: ./build-teleocr.sh [branch-or-tag] [--cpu]
#   (default)        CUDA build (needs nvcc)
#   --cpu            CPU-only build
# Environment:
#   TELEOCR_SRC        existing checkout to build instead of cloning
#   TELEOCR_REPO       git URL      (default: https://github.com/rzafiamy/teleocr-rs)
#   CUDA_COMPUTE_CAP   e.g. 89      (default: detected with nvidia-smi)
#
# Model: `bin/teleocr convert <hf-dir> -o models/teleocr-q8_0.gguf` from a
# download of XingChen-AGI/TeleOCR (see CONFIG.md, backend teleocr-server).

set -euo pipefail

REF="main"
CUDA=1
for arg in "$@"; do
    case "$arg" in
        --cpu) CUDA=0 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) REF="$arg" ;;
    esac
done
REPO="${TELEOCR_REPO:-https://github.com/rzafiamy/teleocr-rs}"
BIN_DIR="$(cd "$(dirname "$0")" && pwd)/bin"
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "cargo not found: install Rust from https://rustup.rs" >&2; exit 1; }

if [ -n "${TELEOCR_SRC:-}" ]; then
    SRC="$TELEOCR_SRC"
else
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    git clone --depth 1 --branch "$REF" "$REPO" "$TMP/teleocr-rs"
    SRC="$TMP/teleocr-rs"
fi

cd "$SRC"
if [ "$CUDA" = 1 ]; then
    command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
    command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA Toolkit or use --cpu" >&2; exit 1; }
    if [ -z "${CUDA_COMPUTE_CAP:-}" ]; then
        CUDA_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
        export CUDA_COMPUTE_CAP
    fi
    cargo build --release -p teleocr-cli --features cuda --target-dir target-cuda
    OUT=target-cuda/release
else
    cargo build --release -p teleocr-cli
    OUT=target/release
fi
scripts/fetch-pdfium.sh "$OUT"
mkdir -p "$BIN_DIR"
install -m 755 "$OUT/teleocr" "$BIN_DIR/teleocr"
install -m 644 "$OUT/libpdfium.so" "$BIN_DIR/libpdfium.so"
"$BIN_DIR/teleocr" --version
echo "Installed $BIN_DIR/teleocr"
