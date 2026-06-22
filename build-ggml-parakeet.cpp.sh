#!/usr/bin/env bash
# build-ggml-parakeet.cpp.sh
# Build parakeet.cpp for a given branch or tag and install binaries into ./bin/
# Nothing is left behind in the repo: cloning and building happen in a temp dir.

set -euo pipefail

BRANCH_OR_TAG="${1:-master}"

echo "🦜 Building parakeet.cpp for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y
apt install -y git cmake build-essential libcurl4-openssl-dev

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"

# Build in a throwaway temp dir so the repo stays clean
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT
REPO_DIR="${BUILD_ROOT}/parakeet.cpp"

# --- Clone (recursive: pulls in third_party/ggml) ---
echo "📦 Cloning parakeet.cpp into temp dir..."
git clone --recursive https://github.com/mudler/parakeet.cpp "$REPO_DIR"

cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"
# Make sure ggml (and any other) submodules match the checked-out revision
git submodule update --init --recursive

# --- Build ---
echo "⚙️ Configuring build..."
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAKEET_GGML_CUDA=ON \
    -DPARAKEET_BUILD_CLI=ON

echo "🚀 Compiling..."
cmake --build build --config Release -j 16

# --- Install into ./bin ---
echo "📦 Copying binaries into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"

cp build/examples/cli/parakeet-cli "$BIN_DIR/" || echo "⚠️ parakeet-cli not found!"
cp build/examples/server/parakeet-server "$BIN_DIR/" || echo "⚠️ parakeet-server not found!"

# --- Summary ---
echo "✅ Build complete! Binaries installed to:"
echo "   → ${BIN_DIR}"
ls -l "${BIN_DIR}/parakeet-"* 2>/dev/null || true
