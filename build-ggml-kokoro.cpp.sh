#!/usr/bin/env bash
# build-ggml-kokoro.cpp.sh
# Build kokoro.cpp (TTS) for a given branch or tag and install binaries into ./bin/.
# Nothing is left behind in the repo: cloning and building happen in a temp dir.
#
# Unlike llama/parakeet, kokoro.cpp statically links ONNX Runtime, zlib, simdutf
# and cpp-httplib, so the resulting kokoro-cli / kokoro-server are standalone —
# no libggml*.so to copy and no patchelf/RPATH fix-ups are needed.

set -euo pipefail

BRANCH_OR_TAG="${1:-v0.1.0}"

echo "🗣️  Building kokoro.cpp for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y
# kokoro.cpp needs a C++20 toolchain + CMake; FetchContent pulls the rest.
apt install -y git cmake build-essential

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"

# kokoro.cpp v0.1.0 pulls xtl, which requires CMake 3.29 or newer. Ubuntu
# 24.04's apt package is 3.28, so allow callers to select a newer install.
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
if [[ -z "${CMAKE_BIN}" || ! -x "${CMAKE_BIN}" ]]; then
    echo "❌ CMake was not found. Install CMake 3.29 or newer." >&2
    exit 1
fi
CMAKE_VERSION="$("${CMAKE_BIN}" --version | awk 'NR == 1 { print $3 }')"
if [[ "$(printf '%s\n' "3.29" "${CMAKE_VERSION}" | sort -V | head -n 1)" != "3.29" ]]; then
    echo "❌ kokoro.cpp requires CMake 3.29 or newer; found ${CMAKE_VERSION}." >&2
    echo "   Install a newer CMake, then rerun with CMAKE_BIN=/path/to/cmake." >&2
    exit 1
fi
echo "🔧 Using CMake ${CMAKE_VERSION}: ${CMAKE_BIN}"

# Build in a throwaway temp dir so the repo stays clean.
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT
REPO_DIR="${BUILD_ROOT}/kokoro.cpp"

# --- Clone ---
echo "📦 Cloning kokoro.cpp into temp dir..."
git clone https://github.com/rzafiamy/kokoro.cpp "$REPO_DIR"

cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"

# --- Build ---
echo "⚙️  Configuring build..."
"${CMAKE_BIN}" -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLI=ON \
    -DBUILD_SERVER=ON

echo "🚀 Compiling (this fetches + statically links ONNX Runtime, may take a while)..."
"${CMAKE_BIN}" --build build --config Release -j "$(nproc)"

# --- Install into ./bin ---
echo "📦 Copying binaries into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"
# kokoro.cpp writes its executables into <repo>/bin via RUNTIME_OUTPUT_DIRECTORY.
cp "${REPO_DIR}/bin/kokoro-server" "$BIN_DIR/" || echo "⚠️ kokoro-server not found!"
cp "${REPO_DIR}/bin/kokoro-cli"    "$BIN_DIR/" || echo "⚠️ kokoro-cli not found!"

# --- Summary ---
echo "✅ Build complete! Binaries installed to:"
echo "   → ${BIN_DIR}"
ls -l "${BIN_DIR}/kokoro-"* 2>/dev/null || true
echo
echo "ℹ️  Pull a TTS model with:  zallama pull kokoro:82m"
