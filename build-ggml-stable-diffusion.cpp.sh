#!/usr/bin/env bash
# build-ggml-stable-diffusion.cpp.sh
# Build stable-diffusion.cpp (Image Generation) for a given branch or tag and install binaries into ./bin/
# Nothing is left behind in the repo: cloning and building happen in a temp dir.

set -euo pipefail

BRANCH_OR_TAG="${1:-master}"

echo "🎨 Building stable-diffusion.cpp for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y 2>/dev/null || true
apt install -y git cmake build-essential libcurl4-openssl-dev patchelf 2>/dev/null || true

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"

# --- Locate CUDA toolkit ---
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$(command -v nvcc || true)"
if [[ -z "${NVCC}" ]]; then
    for c in "${CUDA_ROOT}/bin/nvcc" /usr/local/cuda-*/bin/nvcc; do
        [[ -x "$c" ]] && NVCC="$c" && break
    done
fi

# sd-server lives under examples/server, so the examples target must be on.
# SD_BUILD_SERVER is honoured by newer revisions; on older ones it is simply an
# unused CMake variable (a warning, not an error).
CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DSD_BUILD_SERVER=ON -DSD_BUILD_EXAMPLES=ON)

if [[ -n "${NVCC}" ]]; then
    CUDA_ROOT="$(dirname "$(dirname "$NVCC")")"
    export PATH="${CUDA_ROOT}/bin:${PATH}"
    export CUDACXX="${NVCC}"
    echo "🔧 Using nvcc: ${NVCC} (CUDA root: ${CUDA_ROOT})"
    CMAKE_FLAGS+=(-DSD_CUDA=ON -DCMAKE_CUDA_COMPILER="${NVCC}")
else
    echo "ℹ️ CUDA (nvcc) not found; building CPU version."
fi

# Build in a throwaway temp dir so the repo stays clean
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT
REPO_DIR="${BUILD_ROOT}/stable-diffusion.cpp"

# --- Clone (recursive: pulls in submodules if any) ---
echo "📦 Cloning stable-diffusion.cpp into temp dir..."
git clone --recursive https://github.com/leejet/stable-diffusion.cpp "$REPO_DIR"

cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"
git submodule update --init --recursive 2>/dev/null || true

# --- Build ---
echo "⚙️ Configuring build..."
cmake -B build "${CMAKE_FLAGS[@]}" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'

echo "🚀 Compiling..."
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
cmake --build build --config Release -j "${NPROC}"

# --- Install into ./bin ---
echo "📦 Copying binaries into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"

# Copy sd-server and sd CLI binaries (handling CMake binary output paths)
found_server=0
for candidate in build/bin/sd-server build/sd-server build/examples/server/sd-server build/bin/sd_server build/sd_server; do
    if [[ -f "$candidate" ]]; then
        cp "$candidate" "$BIN_DIR/sd-server"
        found_server=1
        break
    fi
done
if [[ $found_server -eq 0 ]]; then
    echo "⚠️ sd-server binary not found after build!"
fi

for candidate in build/bin/sd build/sd build/examples/cli/sd build/bin/sd-cli build/sd-cli; do
    if [[ -f "$candidate" ]]; then
        cp "$candidate" "$BIN_DIR/sd"
        break
    fi
done

# Shared libraries if any are built
echo "📦 Copying shared libraries (if present)..."
found_libs=$(find build -name 'lib*.so*' -o -name 'lib*.dylib*' 2>/dev/null || true)
if [[ -n "$found_libs" ]]; then
    find build \( -name 'lib*.so*' -o -name 'lib*.dylib*' \) -exec cp -a -t "$BIN_DIR/" {} + 2>/dev/null || true
fi

# Set RPATH to $ORIGIN if patchelf is available
if command -v patchelf >/dev/null 2>&1; then
    echo "🔧 Setting RPATH=\$ORIGIN on binaries..."
    for f in "$BIN_DIR"/sd-server "$BIN_DIR"/sd "$BIN_DIR"/lib*.so*; do
        [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null \
            && echo "   patched $(basename "$f")" || true
    done
fi

# --- Summary ---
echo "✅ Build complete! Binaries installed to:"
echo "   → ${BIN_DIR}"
ls -l "${BIN_DIR}/sd"* 2>/dev/null || true
echo
echo "ℹ️  Pull an image generation model with: zallama pull sd:1.5"
