#!/usr/bin/env bash
# build-ggml-audio.cpp.sh
# Build audio.cpp's server (ASR — Voxtral Mini 4B Realtime) for a given
# branch/tag and install it into ./bin/. Nothing is left behind in the repo:
# cloning and building happen in a temp dir.
#
# Only the voxtral_realtime model family is compiled in (via audio.cpp's
# composite-build support) rather than all 40+ supported families, keeping
# the build fast and the binary lean.

set -euo pipefail

BRANCH_OR_TAG="${1:-release-0.1}"

echo "🎙️  Building audio.cpp (voxtral_realtime) for branch/tag: ${BRANCH_OR_TAG}"

# --- Setup & dependencies ---
apt update -y
apt install -y git cmake build-essential patchelf

WORKDIR=$(pwd)
BIN_DIR="${WORKDIR}/bin"

# --- Locate CUDA toolkit ---
# sudo sanitizes PATH (secure_path), so nvcc on /usr/local/cuda/bin is usually
# not visible even when CUDA is installed. Find it explicitly and tell CMake.
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$(command -v nvcc || true)"
if [[ -z "${NVCC}" ]]; then
    for c in "${CUDA_ROOT}/bin/nvcc" /usr/local/cuda-*/bin/nvcc; do
        [[ -x "$c" ]] && NVCC="$c" && break
    done
fi
if [[ -z "${NVCC}" ]]; then
    echo "❌ nvcc not found. Install the CUDA Toolkit compiler package or set CUDA_HOME." >&2
    exit 1
fi
CUDA_ROOT="$(dirname "$(dirname "$NVCC")")"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export CUDACXX="${NVCC}"
echo "🔧 Using nvcc: ${NVCC} (CUDA root: ${CUDA_ROOT})"

# Build in a throwaway temp dir so the repo stays clean.
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "${BUILD_ROOT}"' EXIT
REPO_DIR="${BUILD_ROOT}/audio.cpp"

# --- Clone ---
echo "📦 Cloning audio.cpp into temp dir..."
git clone --recursive https://github.com/mirek190/audio.cpp "$REPO_DIR"

cd "$REPO_DIR"
git checkout "${BRANCH_OR_TAG}"
git submodule update --init --recursive

# --- Build (composite: only the voxtral_realtime model family) ---
echo "⚙️  Configuring + compiling audiocpp_server (this may take a while)..."
scripts/build_linux.sh \
    --backend cuda \
    --model-set custom \
    --models voxtral_realtime \
    --target audiocpp_server

# --- Locate the built binary ---
# build_linux.sh writes to an aligned build dir, e.g. build/linux-cuda-release.
BUILT_BIN="$(find build -maxdepth 4 -type f -name 'audiocpp_server' -print -quit)"
if [[ -z "${BUILT_BIN}" ]]; then
    echo "❌ audiocpp_server binary not found after build — check the build log above." >&2
    exit 1
fi
BUILT_DIR="$(dirname "${BUILT_BIN}")"

# --- Install into ./bin ---
echo "📦 Copying binary into ${BIN_DIR}..."
mkdir -p "$BIN_DIR"
cp "${BUILT_BIN}" "${BIN_DIR}/audiocpp-server"

# audio.cpp may link ggml (and friends) as shared libs depending on backend
# config. Copy anything present next to the binary and rewrite RPATH=$ORIGIN,
# same as build-ggml-parakeet.cpp.sh — harmless no-op if everything is static.
echo "📦 Copying any shared libraries..."
found_libs=$(find "${BUILT_DIR}" -maxdepth 1 -name '*.so*' -printf '%f\n' 2>/dev/null | sort -u)
if [ -n "$found_libs" ]; then
    find "${BUILT_DIR}" -maxdepth 1 -name '*.so*' -exec cp -a -t "$BIN_DIR/" {} +
    if command -v patchelf >/dev/null 2>&1; then
        echo "🔧 Setting RPATH=\$ORIGIN on binary and libraries..."
        for f in "$BIN_DIR/audiocpp-server" "$BIN_DIR"/*.so.*.*; do
            [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null \
                && echo "   patched $(basename "$f")"
        done
    else
        echo "⚠️ patchelf not found — install it (apt install patchelf) if audiocpp-server"
        echo "   fails at runtime with 'lib*.so: cannot open shared object file'."
    fi
else
    echo "   (no shared libraries found — binary appears statically linked)"
fi

# --- Summary ---
echo "✅ Build complete! Binary installed to:"
echo "   → ${BIN_DIR}/audiocpp-server"
ls -l "${BIN_DIR}/audiocpp-server" 2>/dev/null || true
echo
echo "ℹ️  Register a Voxtral Realtime model with backend: audiocpp-server,"
echo "   modality: asr — see CONFIG.md for the registry entry shape."
