#!/usr/bin/env bash
# build-ggml-llama.cpp.sh
# Build llama.cpp for a given branch or tag and install binaries into ./bin/
# Nothing is left behind in the repo: cloning and building happen in a temp dir.

set -euo pipefail

BRANCH_OR_TAG="${1:-main}"

echo "🐪 Building llama.cpp for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y
apt install -y git cmake build-essential libcurl4-openssl-dev

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"

# Build in a throwaway temp dir so the repo stays clean
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT
REPO_DIR="${BUILD_ROOT}/llama.cpp"

# --- Clone ---
echo "📦 Cloning llama.cpp into temp dir..."
git clone https://github.com/ggml-org/llama.cpp "$REPO_DIR"

cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"

# --- Build ---
echo "⚙️ Configuring build..."
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF

echo "🚀 Compiling..."
cmake --build build --config Release -j 16

# --- Install into ./bin ---
echo "📦 Copying binaries into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"

cp build/bin/llama-cli "$BIN_DIR/" || echo "⚠️ llama-cli not found!"
cp build/bin/llama-server "$BIN_DIR/" || echo "⚠️ llama-server not found!"

# --- Summary ---
echo "✅ Build complete! Binaries installed to:"
echo "   → ${BIN_DIR}"
ls -l "${BIN_DIR}/llama-cli" "${BIN_DIR}/llama-server" 2>/dev/null || true
