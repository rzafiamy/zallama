#!/usr/bin/env bash
# build-tn.sh
# Build tn-server (text normalization for TTS, rzafiamy/tn-rs) and install it
# as ./bin/tn-server for the `tn-server` backend (modality: normalization).
#
# Usage: ./build-tn.sh [branch-or-tag]
# Environment:
#   TN_SRC     existing checkout to build instead of cloning
#   TN_REPO    git URL (default: https://github.com/rzafiamy/tn-rs)
#
# No root needed: building needs cargo (https://rustup.rs) only; the binary is
# CPU-only and has no runtime dependency. Clones into a temp dir unless TN_SRC
# is set. Lexicon: models/registry.example.yaml (entry `tn`).

set -euo pipefail

REF="main"
for arg in "$@"; do
    case "$arg" in
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) REF="$arg" ;;
    esac
done
REPO="${TN_REPO:-https://github.com/rzafiamy/tn-rs}"
BIN_DIR="$(cd "$(dirname "$0")" && pwd)/bin"
export PATH="$HOME/.cargo/bin:$PATH"

command -v cargo >/dev/null || { echo "cargo not found: install Rust from https://rustup.rs" >&2; exit 1; }

if [ -n "${TN_SRC:-}" ]; then
    SRC="$TN_SRC"
else
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    git clone --depth 1 --branch "$REF" "$REPO" "$TMP/tn-rs"
    SRC="$TMP/tn-rs"
fi

# tn-rs's build.sh prints "artifact: <path>" as its last line.
ARTIFACT=$(cd "$SRC" && ./build.sh | tee /dev/stderr | sed -n 's/^artifact: //p' | tail -1)
[ -n "$ARTIFACT" ] || { echo "tn build produced no artifact" >&2; exit 1; }
mkdir -p "$BIN_DIR"
install -m 755 "$SRC/$ARTIFACT" "$BIN_DIR/tn-server"
"$BIN_DIR/tn-server" --version
echo "Installed $BIN_DIR/tn-server"
