#!/usr/bin/env bash
# build-llama.sh
# Build llama.cpp for a given branch or tag and package binaries neatly

set -euo pipefail

BRANCH_OR_TAG="${1:-main}"

echo "🐪 Building llama.cpp for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y
apt install -y git cmake build-essential libcurl4-openssl-dev

WORKDIR=$(pwd)
REPO_DIR="${WORKDIR}/llama.cpp"
DIST_DIR="${WORKDIR}/dist-${BRANCH_OR_TAG}"

# --- Clone or update repo ---
if [ ! -d "$REPO_DIR" ]; then
    echo "📦 Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp "$REPO_DIR"
else
    echo "🔄 Updating existing llama.cpp repo..."
    git -C "$REPO_DIR" fetch --all --tags
fi

cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"

# --- Build ---
echo "⚙️ Configuring build..."
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF

echo "🚀 Compiling..."
cmake --build build --config Release -j 16

# --- Package ---
echo "📦 Creating package in ${DIST_DIR}..."
mkdir -p "$DIST_DIR"

cp build/bin/llama-cli "$DIST_DIR/" || echo "⚠️ llama-cli not found!"
cp build/bin/llama-server "$DIST_DIR/" || echo "⚠️ llama-server not found!"

# Write version info
echo "${BRANCH_OR_TAG}" > "${DIST_DIR}/version.txt"

# --- Summary ---
echo "✅ Build & packaging complete!"
echo "   Binaries and version info located at:"
echo "   → ${DIST_DIR}"
ls -l "${DIST_DIR}"
